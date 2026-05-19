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

from analysis import technical, fundamental, momentum, forecast, quality, earnings_drift, uptrend
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
    uptrend_score: float = 0.0
    uptrend_data: dict = None  # type: ignore[assignment]
    suggested_stop: Optional[float] = None
    stop_method: str = "n/a"
    adjusted_score: float = 0.0  # after cross-sectional pass; defaults to score
    cross_sectional: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.adjusted_score == 0.0:
            self.adjusted_score = self.score
        if self.cross_sectional is None:
            self.cross_sectional = {}
        if self.uptrend_data is None:
            self.uptrend_data = {}


def _precompute_hd_series(hd: HistoricalData) -> None:
    """Pre-compute all technical indicators Series on the full history once."""
    close = hd.history["Close"]
    vol = hd.history["Volume"]
    high = hd.history["High"]
    low = hd.history["Low"]
    
    import ta
    import numpy as np
    
    # 1. RSI
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    
    # 2. MACD
    macd = ta.trend.MACD(close)
    macd_line = macd.macd()
    macd_sig = macd.macd_signal()
    
    # 3. SMAs
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    if len(close) >= 200:
        sma200 = close.rolling(200).mean()
    else:
        sma200 = close.rolling(len(close) // 2).mean()
        
    # 4. Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_pct = bb.bollinger_pband()
    
    # 5. ATR (14-day Average True Range series)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean()
    
    # 6. Annualized Volatility (rolling 90-day annualized std of daily returns)
    log_rets = np.log(close / close.shift(1))
    vol_series = log_rets.rolling(90).std() * np.sqrt(252)
    
    # 7. Accumulation/Distribution metrics
    up_mask = close > close.shift(1)
    dn_mask = close < close.shift(1)
    
    # Rolling sum of up and down volume over 50 days
    up_vol_50 = vol.where(up_mask, 0.0).rolling(50).sum()
    dn_vol_50 = vol.where(dn_mask, 0.0).rolling(50).sum()
    ud_ratio = up_vol_50 / dn_vol_50.replace(0.0, np.nan)
    ud_ratio = ud_ratio.fillna(1.0)
    
    # Accumulation/Distribution net days (25-day)
    vol_up = vol > vol.shift(1)
    acc_days = (up_mask & vol_up).astype(float).rolling(25).sum()
    dist_days = (dn_mask & vol_up).astype(float).rolling(25).sum()
    net_acc = acc_days - dist_days
    
    # Chaikin Money Flow
    cmf = ta.volume.ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=vol, window=20
    ).chaikin_money_flow()
    
    hd._precomputed_series = {
        "rsi": rsi,
        "macd_line": macd_line,
        "macd_sig": macd_sig,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "bb_pct": bb_pct,
        "atr": atr_series,
        "vol": vol_series,
        "ud_ratio": ud_ratio,
        "net_acc": net_acc,
        "cmf": cmf,
    }
    
    # Add columns directly to history for uptrend.py fast lookups
    n = len(close)
    hd.history["sma50"] = sma50
    hd.history["sma150"] = close.rolling(min(150, n)).mean()
    hd.history["sma200"] = sma200
    
    sma_bb = close.rolling(20).mean()
    sd_bb = close.rolling(20).std(ddof=0)
    hd.history["bbw"] = (4 * sd_bb) / sma_bb.replace(0, np.nan)
    
    vol_up = vol > vol.shift(1)
    
    # ATR-filtered Dollar-Volume U/D Ratios (from uptrend.py)
    ret = close.pct_change()
    dol_vol = close * vol
    threshold = 0.25 * (atr_series / close)
    sig = ret.abs() > threshold
    
    up_dol = dol_vol.where(sig & (ret > 0), 0.0)
    dn_dol = dol_vol.where(sig & (ret < 0), 0.0)
    
    up_50 = up_dol.rolling(50).sum()
    dn_50 = dn_dol.rolling(50).sum()
    hd.history["ud_50"] = up_50 / dn_50.replace(0, np.nan)
    hd.history["ud_50"] = hd.history["ud_50"].fillna(5.0)
    
    up_15 = up_dol.rolling(15).sum()
    dn_15 = dn_dol.rolling(15).sum()
    hd.history["ud_15"] = up_15 / dn_15.replace(0, np.nan)
    hd.history["ud_15"] = hd.history["ud_15"].fillna(5.0)
    
    # ADR%
    rng = (high - low) / close.replace(0, np.nan)
    hd.history["adr_pct"] = rng.rolling(20).mean() * 100



def _slice(hd: HistoricalData, asof: pd.Timestamp) -> pd.DataFrame:
    """Return history with index <= asof. Strips timezone for comparison."""
    idx = hd.history.index
    if idx.tz is not None:
        sliced = hd.history[idx.tz_localize(None) <= asof].copy()
    else:
        sliced = hd.history[idx <= asof].copy()
        
    # Pre-compute indicators Series on full history once
    if not hasattr(hd, "_precomputed_series"):
        _precompute_hd_series(hd)
        
    sliced._parent_hd = hd
    sliced._asof = asof
    return sliced



def score_at(hd: HistoricalData, asof: pd.Timestamp,
             include_forecast: bool = False,
             live_weights: bool = True,
             weights_override: Optional[dict] = None,
             regime_label: str = "Neutral") -> Optional[BacktestScore]:
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

    w = dict(weights_override) if weights_override else dict(SCORE_WEIGHTS)
    # 'valuation' is sector-relative (cross-sectional) and unavailable in
    # per-ticker backtest scoring — drop it and renormalize via fundamental.
    valuation_w = w.pop("valuation", 0.0)
    w["fundamental"] = w.get("fundamental", 0.0) + valuation_w

    if live_weights:
        # Live-equivalent weights, but DROP components that have no historical
        # equivalent (sentiment from news, options flow). Anchoring them at
        # neutral 50 systematically depresses backtest composites by ~6 points
        # vs. live and prevents stocks from clearing the buy threshold.
        # Renormalization happens below via total_w.
        w.pop("sentiment", None)
        w.pop("options", None)
        if not include_forecast:
            w.pop("forecast", None)
        parts = {
            "technical":      tech.get("score", 50),
            "fundamental":    fund.get("score", 50),
            "momentum":       mom.get("score", 50),
            "quality":        qual.get("score", 50),
            "earnings_drift": edrift.get("score", 50),
        }
        if include_forecast:
            parts["forecast"] = fcst.get("score", 50)
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
    # ── Risk Management (Smart Stops) ──────────────────────────────────
    # Re-use the same logic as the live scanner
    from analysis import stops
    # Map engine regime label (BULL/CAUTIOUS/...) to uptrend.compute's
    # expected vocabulary (Bullish/Neutral/Caution/Bearish) so the stop
    # method picks the right tightness for the regime.
    _regime_map = {
        "BULL":         "Bullish",
        "NEUTRAL_BULL": "Bullish",
        "NEUTRAL":      "Neutral",
        "CAUTIOUS":     "Caution",
        "BEAR":         "Bearish",
    }
    up_regime = _regime_map.get(regime_label, regime_label or "Neutral")
    up_data = uptrend.compute(hist, regime=up_regime)
    up_score = up_data.get("score", 50.0)
    
    # Extract the suggested stop from the uptrend data
    s_stop = up_data.get("stop_suggested")
    s_method = up_data.get("stop_method", "n/a")

    total_w = sum(w.get(k, 0) for k in parts)

    if total_w > 0:
        composite = sum(parts[k] * w.get(k, 0) for k in parts) / total_w
    else:
        composite = 50.0

    price = float(hist["Close"].iloc[-1])
    
    # Fast lookup from cached Series
    cache = hd._precomputed_series
    asof_tz = asof
    if cache["atr"].index.tz is not None and asof.tzinfo is None:
        try:
            asof_tz = asof.tz_localize(cache["atr"].index.tz)
        except Exception:
            asof_tz = asof.tz_localize("UTC").tz_convert(cache["atr"].index.tz)
    elif cache["atr"].index.tz is None and asof.tzinfo is not None:
        asof_tz = asof.tz_localize(None)
        
    atr_s = cache["atr"].loc[:asof_tz]
    vol_s = cache["vol"].loc[:asof_tz]
    atr_val = float(atr_s.iloc[-1]) if (not pd.isna(atr_s.iloc[-1])) else float(atr(hist) or 0.0)
    vol_val = float(vol_s.iloc[-1]) if (not pd.isna(vol_s.iloc[-1])) else float(annualized_volatility(hist) or 0.30)
    
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
        atr_value=atr_val,
        annual_vol=vol_val,
        uptrend_score=up_score,
        uptrend_data=up_data,
        suggested_stop=s_stop,
        stop_method=s_method,
    )


def price_at(hd: HistoricalData, asof: pd.Timestamp) -> Optional[float]:
    """Return the close price on/before asof. Used to update open positions."""
    idx = hd.history.index
    
    asof_tz = asof
    if idx.tz is not None and asof.tzinfo is None:
        try:
            asof_tz = asof.tz_localize(idx.tz)
        except Exception:
            asof_tz = asof.tz_localize("UTC").tz_convert(idx.tz)
    elif idx.tz is None and asof.tzinfo is not None:
        asof_tz = asof.tz_localize(None)
        
    loc = idx.searchsorted(asof_tz, side="right")
    if loc == 0:
        return None
    return float(hd.history["Close"].iloc[loc - 1])


# ── Cross-sectional RS pass for the backtest ────────────────────────────────
# Mirrors analysis.uptrend.apply_rs() but operates on BacktestScore objects
# (which have .uptrend_data dict + .uptrend_score float instead of .uptrend
# dict on a StockReport). Without this, the backtest scores stocks using only
# the *absolute* uptrend score and never benefits from 12-1 momentum RS rank,
# the single most validated alpha factor in trend-following. Wiring this in
# closes the largest leak vs the live screener.
def apply_rs_to_bt(scores: "list[BacktestScore]",
                   market_regime_mult: float = 1.0) -> None:
    """Compute IBD-style RS percentile + sector strength bonus across the
    candidate universe and update each BacktestScore in place.

    Updates:
      - score.uptrend_data['rs_pct']
      - score.uptrend_data['trend_template_full'] / 'trend_template_pass'
      - score.uptrend_data['is_leader']
      - score.uptrend_score (clipped to [0,100]) — absorbs RS bump + sector bump
      - score.adjusted_score — same bump applied so non-uptrend_mode also benefits
    """
    import numpy as np

    if not scores:
        return

    # 1. Collect raw 12-1 momentum (fallback to 6-1 * 1.4 if 12m missing)
    raws: list = []
    for s in scores:
        ud = s.uptrend_data or {}
        v = ud.get("rs_raw_12_1")
        if v is None:
            v6 = ud.get("rs_raw_6_1")
            v = v6 * 1.4 if v6 is not None else None
        raws.append(v)

    valid = np.array([x for x in raws if x is not None], dtype=float)
    if valid.size < 5:
        # Universe too small to rank — set neutral and bail.
        for s in scores:
            s.uptrend_data["rs_pct"] = 50.0
            s.uptrend_data.setdefault("trend_template_full",
                                       s.uptrend_data.get("trend_template", 0))
            s.uptrend_data.setdefault("trend_template_pass",
                                       bool(s.uptrend_data.get("trend_template", 0) >= 7))
            s.uptrend_data.setdefault("is_leader", False)
        return

    sorted_valid = np.sort(valid)

    # 2. RS percentile per ticker
    for s, raw in zip(scores, raws):
        if raw is None:
            rs_pct = 50.0
        else:
            rank = float(np.searchsorted(sorted_valid, raw, side="left"))
            rs_pct = float(np.clip(rank / max(len(sorted_valid) - 1, 1) * 100, 0, 100))
        s.uptrend_data["rs_pct"] = round(rs_pct, 1)

    # 3. Sector strength (avg uptrend score per sector, min 2 members)
    sector_scores: dict[str, list[float]] = {}
    for s in scores:
        sec = s.sector or "Unknown"
        sector_scores.setdefault(sec, []).append(float(s.uptrend_score or 50.0))
    sector_avgs = {sec: float(np.mean(vals))
                   for sec, vals in sector_scores.items() if len(vals) >= 2}

    # 4. Apply RS bump + sector bump → uptrend_score AND adjusted_score
    for s in scores:
        ud = s.uptrend_data
        rs_pct = float(ud.get("rs_pct", 50.0))

        # 8th trend-template check
        tt = int(ud.get("trend_template", 0))
        tt_full = tt + (1 if rs_pct >= 70 else 0)
        ud["trend_template_full"] = tt_full
        ud["trend_template_pass"] = bool(tt_full >= 7)

        # RS decile bump
        if rs_pct >= 90:
            bump = 10.0
        elif rs_pct >= 80:
            bump = 7.0
        elif rs_pct >= 70:
            bump = 4.0
        elif rs_pct <= 30:
            bump = -5.0
        else:
            bump = 0.0

        # Sector strength bump
        sec_avg = sector_avgs.get(s.sector or "Unknown", 50.0)
        if sec_avg >= 65:
            bump += 5.0
        elif sec_avg <= 40:
            bump -= 5.0

        # Apply to both score channels
        new_up = float(np.clip((s.uptrend_score or 50.0) + bump, 0, 100))
        s.uptrend_score = round(new_up, 2)

        # Also nudge adjusted_score (used in the default, non-uptrend-mode
        # entry path) so the RS signal actually influences the ranking.
        new_adj = float(np.clip((s.adjusted_score or s.score) + bump, 0, 100))
        s.adjusted_score = round(new_adj, 2)

        # Leader flag (used by future hard filter / reporter)
        ud["is_leader"] = bool(
            ud.get("stage2")
            and rs_pct >= 70
            and ud.get("pct_from_52w_high", -100) >= -25
            and (ud.get("ud_50") or 0) >= 1.0
        )
