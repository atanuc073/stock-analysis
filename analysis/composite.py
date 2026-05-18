"""Composite scorer — aggregates all analyzer outputs into a single 0-100 score and verdict."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Iterable
from tqdm import tqdm

from config import SCORE_WEIGHTS
from analysis import (
    technical, fundamental, momentum, sentiment, forecast, options_flow,
    quality, earnings_drift, cross_sectional, uptrend,
)
from data_sources.yahoo import TickerData


@dataclass
class StockReport:
    symbol: str
    name: str = ""
    market: str = "US"
    sector: str = ""
    price: float = 0.0
    composite_score: float = 0.0
    adjusted_score: float = 0.0          # after cross-sectional pass
    verdict: str = "HOLD"
    technical: dict = field(default_factory=dict)
    fundamental: dict = field(default_factory=dict)
    momentum: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    forecast: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    earnings_drift: dict = field(default_factory=dict)
    uptrend: dict = field(default_factory=dict)
    cross_sectional: dict = field(default_factory=dict)
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


def analyze(td: TickerData, regime: Optional[dict] = None) -> StockReport:
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
    if SCORE_WEIGHTS.get("forecast", 0.0) > 0.0:
        rep.forecast = forecast.compute(td.history)
    else:
        rep.forecast = {"score": 50.0, "signals": []}
    rep.options = options_flow.compute(td.options_summary)
    rep.quality = quality.compute(td.info)
    rep.earnings_drift = earnings_drift.compute(td.history, td.info)
    
    # Pass regime for dynamic stops
    regime_name = regime.get("regime", "Neutral") if regime else "Neutral"
    rep.uptrend = uptrend.compute(td.history, regime=regime_name)

    rep.name = rep.fundamental.get("name") or td.symbol
    rep.sector = rep.fundamental.get("sector") or ""
    rep.price = rep.technical.get("price", 0.0)

    # Weighted composite. Redistribute options weight if not available.
    # `valuation` is universe-aware and applied later in cross_sectional.apply
    # — drop it here so the per-ticker score sums to (1 - valuation_w).
    weights = dict(SCORE_WEIGHTS)
    valuation_w = weights.pop("valuation", 0.0)
    if not rep.options.get("available"):
        opt_w = weights.pop("options")
        # spread to technical + momentum
        weights["technical"] += opt_w * 0.6
        weights["momentum"] += opt_w * 0.4

    parts = {
        "technical":      rep.technical.get("score", 50),
        "fundamental":    rep.fundamental.get("score", 50),
        "momentum":       rep.momentum.get("score", 50),
        "sentiment":      rep.sentiment.get("score", 50),
        "forecast":       rep.forecast.get("score", 50),
        "options":        rep.options.get("score", 50),
        "quality":        rep.quality.get("score", 50),
        "earnings_drift": rep.earnings_drift.get("score", 50),
    }
    # Re-normalize so the per-ticker score is on [0,100] even though the
    # `valuation` weight is applied later. Effectively redistributes its
    # weight uniformly across surviving components for the per-ticker pass.
    surviving = sum(weights.get(k, 0) for k in parts)
    if surviving > 0:
        composite = sum(parts[k] * weights.get(k, 0) for k in parts) / surviving
    else:
        composite = 50.0
    rep.composite_score = round(composite, 2)
    rep.adjusted_score = rep.composite_score
    rep.verdict = _verdict(composite)

    rep.all_signals = (
        rep.technical.get("signals", [])
        + rep.fundamental.get("signals", [])
        + rep.momentum.get("signals", [])
        + rep.sentiment.get("signals", [])
        + rep.forecast.get("signals", [])
        + rep.options.get("signals", [])
        + rep.quality.get("signals", [])
        + rep.earnings_drift.get("signals", [])
    )
    return rep


def analyze_batch(tickers: Iterable[TickerData]) -> list[StockReport]:
    """Score a batch of tickers and apply cross-sectional adjustments.

    Use this entry point when you have the full universe — it enables
    sector-relative valuation and cross-sectional momentum/quality ranking
    that single-ticker `analyze()` cannot compute. After the post-processor
    runs, each report carries:
        - `composite_score`   : per-ticker weighted score
        - `adjusted_score`    : universe-aware score (use this for ranking)
        - `cross_sectional`   : detail dict (sector_val_score, mom_z, ...)
    Verdicts are recomputed from the adjusted score.
    """
    # 0. Fetch regimes for dynamic stops
    try:
        from data_sources.yahoo import fetch_many
        indices = fetch_many(["^GSPC", "^NSEI"], period="1y", show_progress=False)
        regimes = {
            "US": uptrend.check_market_regime(indices["^GSPC"].history) if indices.get("^GSPC") and indices["^GSPC"].ok else None,
            "IN": uptrend.check_market_regime(indices["^NSEI"].history) if indices.get("^NSEI") and indices["^NSEI"].ok else None
        }
    except Exception:
        regimes = {"US": None, "IN": None}

    # 1. Individual analysis
    reports = []
    for td in tqdm(tickers, desc="Analyzing", unit="stock"):
        r_context = regimes["IN"] if td.is_indian else regimes["US"]
        reports.append(analyze(td, regime=r_context))
    
    # 2. Cross-sectional valuation & rank bonus
    cross_sectional.apply(reports)
    
    # 3. Uptrend Relative Strength
    
    # Split reports by market and apply RS + Regime separately
    us_reps = [r for r in reports if r.market == "US"]
    in_reps = [r for r in reports if r.market == "IN"]
    
    if us_reps:
        uptrend.apply_rs(us_reps, market_regime=regimes["US"])
    if in_reps:
        uptrend.apply_rs(in_reps, market_regime=regimes["IN"])

    # 4. Finalize verdicts
    for r in reports:
        if r.composite_score > 0:   # don't overwrite N/A errors
            r.verdict = _verdict(r.adjusted_score)

    # 5. Entry-quality guards (mirror backtest engine's hard blocks).
    # Downgrade BUY/STRONG BUY → HOLD when the chase filters would have
    # blocked entry in the backtest. Keeps the live screener and the
    # validated engine logic in sync.
    #   - max extension above 200DMA: 25%
    #   - require ≥1.5% pullback from 52WH unless breakout_today fires
    MAX_EXTENSION_PCT = 25.0
    MAX_PCT_FROM_52WH = -1.5
    for r in reports:
        if r.verdict not in ("BUY", "STRONG BUY"):
            continue
        ext = float(r.technical.get("pct_above_sma200") or 0.0)
        pct_from_high = float(r.technical.get("pct_from_52w_high") or 0.0)
        breakout_today = bool((r.uptrend or {}).get("breakout_today", False))
        chase_extension = ext > MAX_EXTENSION_PCT
        chase_high = (pct_from_high > MAX_PCT_FROM_52WH) and not breakout_today
        if chase_extension or chase_high:
            reasons = []
            if chase_extension:
                reasons.append(f"+{ext:.0f}% over 200DMA")
            if chase_high:
                reasons.append(f"{pct_from_high:+.1f}% from 52WH, no breakout")
            r.verdict = "HOLD"
            r.all_signals.append(
                "⛔ Entry guard: " + ", ".join(reasons) + " — downgraded BUY→HOLD"
            )
    return reports
