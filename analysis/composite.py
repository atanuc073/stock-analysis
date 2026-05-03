"""Composite scorer — aggregates all analyzer outputs into a single 0-100 score and verdict."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any

from config import SCORE_WEIGHTS
from analysis import technical, fundamental, momentum, sentiment, forecast, options_flow
from data_sources.yahoo import TickerData


@dataclass
class StockReport:
    symbol: str
    name: str = ""
    market: str = "US"
    sector: str = ""
    price: float = 0.0
    composite_score: float = 0.0
    verdict: str = "HOLD"
    technical: dict = field(default_factory=dict)
    fundamental: dict = field(default_factory=dict)
    momentum: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    forecast: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    all_signals: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verdict(score: float) -> str:
    if score >= 75:
        return "STRONG BUY"
    if score >= 62:
        return "BUY"
    if score >= 45:
        return "HOLD"
    if score >= 32:
        return "REDUCE"
    return "AVOID"


def analyze(td: TickerData) -> StockReport:
    rep = StockReport(symbol=td.symbol, market="IN" if td.is_indian else "US")
    if not td.ok:
        rep.error = td.error or "no data"
        rep.composite_score = 0
        rep.verdict = "N/A"
        return rep

    rep.technical = technical.compute(td.history)
    rep.fundamental = fundamental.compute(td.info)
    rep.momentum = momentum.compute(td.history)
    rep.sentiment = sentiment.compute(td.news)
    rep.forecast = forecast.compute(td.history)
    rep.options = options_flow.compute(td.options_summary)

    rep.name = rep.fundamental.get("name") or td.symbol
    rep.sector = rep.fundamental.get("sector") or ""
    rep.price = rep.technical.get("price", 0.0)

    # Weighted composite. Redistribute options weight if not available.
    weights = dict(SCORE_WEIGHTS)
    if not rep.options.get("available"):
        opt_w = weights.pop("options")
        # spread to technical + momentum
        weights["technical"] += opt_w * 0.6
        weights["momentum"] += opt_w * 0.4

    parts = {
        "technical": rep.technical.get("score", 50),
        "fundamental": rep.fundamental.get("score", 50),
        "momentum": rep.momentum.get("score", 50),
        "sentiment": rep.sentiment.get("score", 50),
        "forecast": rep.forecast.get("score", 50),
        "options": rep.options.get("score", 50),
    }
    composite = sum(parts[k] * weights.get(k, 0) for k in parts)
    rep.composite_score = round(composite, 2)
    rep.verdict = _verdict(composite)

    rep.all_signals = (
        rep.technical.get("signals", [])
        + rep.fundamental.get("signals", [])
        + rep.momentum.get("signals", [])
        + rep.sentiment.get("signals", [])
        + rep.forecast.get("signals", [])
        + rep.options.get("signals", [])
    )
    return rep
