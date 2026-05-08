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

from analysis import technical, fundamental, momentum, forecast, quality, earnings_drift
from analysis.indicators import atr, annualized_volatility
from config import SCORE_WEIGHTS

from .data_loader import HistoricalData


@dataclass
class BacktestScore:
    symbol: str
    market: str
    sector: str
    price: float
    score: float                # per-ticker weighted composite
    technical: dict
    fundamental: dict
    momentum: dict
    forecast: dict
    quality: dict
    earnings_drift: dict
    atr_value: float
    annual_vol: float
    adjusted_score: float = 0.0  # after cross-sectional pass; defaults to score
    cross_sectional: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.adjusted_score == 0.0:
            self.adjusted_score = self.score
        if self.cross_sectional is None:
            self.cross_sectional = {}


def _slice(hd: HistoricalData, asof: pd.Timestamp) -> pd.DataFrame:
    """Return history with index <= asof. Strips timezone for comparison."""
    idx = hd.history.index
    if idx.tz is not None:
        sliced = hd.history[idx.tz_localize(None) <= asof]
    else:
        sliced = hd.history[idx <= asof]
    return sliced


def score_at(hd: HistoricalData, asof: pd.Timestamp,
             include_forecast: bool = False,
             live_weights: bool = True) -> Optional[BacktestScore]:
    """Compute composite score using only data <= asof.

    Set include_forecast=False to skip the forecast component (faster, and the
    linear/prophet forecasters add little value at backtest scale).

    When ``live_weights=True`` (default), use the exact same weight scheme as
    the live scorer in ``config.SCORE_WEIGHTS`` and treat unavailable
    components (sentiment / options / optionally forecast) as neutral 50.
    This matches the live composite behavior — components with no signal
    contribute 50 × weight rather than being redistributed to drivers.

    When ``live_weights=False`` (legacy), drop unavailable components and
    redistribute their weight to technical (60%) and momentum (40%).
    """
    hist = _slice(hd, asof)
    if len(hist) < 60:
        return None

    info = hd.info_static or {}

    tech = technical.compute(hist)
    fund = fundamental.compute(info)
    mom = momentum.compute(hist)
    fcst = forecast.compute(hist) if include_forecast else {"score": 50.0, "signals": []}
    qual = quality.compute(info)
    edrift = earnings_drift.compute(hist, info)

    w = dict(SCORE_WEIGHTS)
    # 'valuation' is sector-relative (cross-sectional) and unavailable in
    # per-ticker backtest scoring — drop it and renormalize via fundamental.
    valuation_w = w.pop("valuation", 0.0)
    w["fundamental"] = w.get("fundamental", 0.0) + valuation_w

    if live_weights:
        # Live-equivalent: keep all weights, neutral 50 for missing components.
        parts = {
            "technical":      tech.get("score", 50),
            "fundamental":    fund.get("score", 50),
            "momentum":       mom.get("score", 50),
            "sentiment":      50.0,                                  # no historical news
            "options":        50.0,                                  # no historical options
            "forecast":       fcst.get("score", 50) if include_forecast else 50.0,
            "quality":        qual.get("score", 50),
            "earnings_drift": edrift.get("score", 50),
        }
    else:
        # Legacy behavior: drop missing components, redistribute their weight
        # to technical (60%) and momentum (40%).
        dropped = w.pop("sentiment", 0) + w.pop("options", 0)
        if not include_forecast:
            dropped += w.pop("forecast", 0)
        w["technical"] = w.get("technical", 0) + dropped * 0.6
        w["momentum"] = w.get("momentum", 0) + dropped * 0.4

        parts = {
            "technical":      tech.get("score", 50),
            "fundamental":    fund.get("score", 50),
            "momentum":       mom.get("score", 50),
            "quality":        qual.get("score", 50),
            "earnings_drift": edrift.get("score", 50),
        }
        if include_forecast:
            parts["forecast"] = fcst.get("score", 50)

    # Renormalize so composite is on [0,100] regardless of whether weights
    # sum exactly to 1.0. Defensive: prevents silent scaling bugs when
    # SCORE_WEIGHTS is edited and the sum drifts off 1.0.
    total_w = sum(w.get(k, 0) for k in parts)
    if total_w > 0:
        composite = sum(parts[k] * w.get(k, 0) for k in parts) / total_w
    else:
        composite = 50.0

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
        quality=qual,
        earnings_drift=edrift,
        atr_value=float(atr(hist) or 0.0),
        annual_vol=float(annualized_volatility(hist) or 0.30),
    )


def price_at(hd: HistoricalData, asof: pd.Timestamp) -> Optional[float]:
    """Return the close price on/before asof. Used to update open positions."""
    hist = _slice(hd, asof)
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])
