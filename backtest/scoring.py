"""No-lookahead scoring for backtest.

For each backtest date `t`:
  - History is sliced to data <= t (eliminates lookahead bias)
  - Fundamentals are static (frozen at fetch time — minor acceptable lookahead)
  - News/sentiment/options are excluded (not historical, would be unfair)

This produces a slightly lower composite score than live (no sentiment/options
contributions), but it's apples-to-apples across all backtest dates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from analysis import technical, fundamental, momentum, forecast
from analysis.indicators import atr, annualized_volatility
from config import SCORE_WEIGHTS

from .data_loader import HistoricalData


@dataclass
class BacktestScore:
    symbol: str
    market: str
    sector: str
    price: float
    score: float
    technical: dict
    fundamental: dict
    momentum: dict
    forecast: dict
    atr_value: float
    annual_vol: float


def _slice(hd: HistoricalData, asof: pd.Timestamp) -> pd.DataFrame:
    """Return history with index <= asof. Strips timezone for comparison."""
    idx = hd.history.index
    if idx.tz is not None:
        sliced = hd.history[idx.tz_localize(None) <= asof]
    else:
        sliced = hd.history[idx <= asof]
    return sliced


def score_at(hd: HistoricalData, asof: pd.Timestamp,
             include_forecast: bool = False) -> Optional[BacktestScore]:
    """Compute composite score using only data <= asof.

    Set include_forecast=False to skip the forecast component (faster, and the
    linear/prophet forecasters add little value at backtest scale).
    """
    hist = _slice(hd, asof)
    if len(hist) < 60:
        return None

    info = hd.info_static or {}

    tech = technical.compute(hist)
    fund = fundamental.compute(info)
    mom = momentum.compute(hist)
    fcst = forecast.compute(hist) if include_forecast else {"score": 50.0, "signals": []}

    # Reweight: drop sentiment + options components, redistribute to technical/momentum
    # Original weights sum to 1.0 across {technical, fundamental, momentum, sentiment, forecast, options}
    w = dict(SCORE_WEIGHTS)
    dropped = w.pop("sentiment", 0) + w.pop("options", 0)
    if not include_forecast:
        dropped += w.pop("forecast", 0)
    # spread dropped weight to technical (60%) and momentum (40%)
    w["technical"] = w.get("technical", 0) + dropped * 0.6
    w["momentum"] = w.get("momentum", 0) + dropped * 0.4

    parts = {
        "technical": tech.get("score", 50),
        "fundamental": fund.get("score", 50),
        "momentum": mom.get("score", 50),
    }
    if include_forecast:
        parts["forecast"] = fcst.get("score", 50)

    composite = sum(parts[k] * w.get(k, 0) for k in parts)

    price = float(hist["Close"].iloc[-1])
    return BacktestScore(
        symbol=hd.symbol,
        market=hd.market,
        sector=hd.sector,
        price=price,
        score=round(composite, 2),
        technical=tech,
        fundamental=fund,
        momentum=mom,
        forecast=fcst,
        atr_value=float(atr(hist) or 0.0),
        annual_vol=float(annualized_volatility(hist) or 0.30),
    )


def price_at(hd: HistoricalData, asof: pd.Timestamp) -> Optional[float]:
    """Return the close price on/before asof. Used to update open positions."""
    hist = _slice(hd, asof)
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])
