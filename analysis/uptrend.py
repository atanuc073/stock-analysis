"""Uptrend / Breakout scanner.

Identifies stocks in confirmed Stage-2 uptrends with high-quality breakout
setups. Synthesizes well-validated signals from the trend-following / CAN-SLIM
/ Minervini playbook:

  - **Stage 2 trend** (Weinstein) — price > SMA150/200, MAs stacked & rising
  - **Trend Template** (Minervini) — 7 of 8 criteria (RS rank added later)
  - **52-week-high proximity** — strongest individual momentum factor
  - **VCP / volatility contraction** — Bollinger-Band-Width percentile + range shrink
  - **U/D ratio** (50d, dollar-volume, ATR-filtered) — accumulation proxy
  - **ADR%** regime — filters sleepy & wild names
  - **Pivot proximity + breakout trigger** — actionable entry signal
  - **12-1 momentum** (raw value, RS percentile computed cross-sectionally)

Per-ticker `compute()` returns absolute metrics + a 0-100 absolute score.
The cross-sectional RS percentile is layered on top in `apply_rs()` after the
full universe has been analyzed.

This module is intentionally dependency-light: only `numpy` + `pandas`.
"""
from typing import Iterable, Optional
import numpy as np
import pandas as pd
from analysis import stops


# ── Tunables ───────────────────────────────────────────────────────────────
_BB_WINDOW = 20
_BB_LOOKBACK = 252        # percentile window for BBW squeeze
_UD_WINDOW = 50           # primary U/D ratio window
_UD_SHORT = 15            # confirmation window
_VCP_SHORT, _VCP_MID, _VCP_LONG = 20, 60, 120
_PIVOT_WINDOW = 20        # base/pivot lookback
_BREAKOUT_VOL_MULT = 1.5
_VOL_DRYUP_RATIO = 0.75   # 5d/50d avg vol < this = quiet base
_ATR_WINDOW = 14
_ATR_FILTER_K = 0.25      # only count days where |ret| > k * ATR/close


# ── Helpers ────────────────────────────────────────────────────────────────
def _safe_iloc(s: pd.Series, i: int = -1) -> Optional[float]:
    if s is None or len(s) == 0:
        return None
    v = s.iloc[i]
    if pd.isna(v):
        return None
    return float(v)


def _atr(df: pd.DataFrame, window: int = _ATR_WINDOW) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _bbw_percentile(df: pd.DataFrame) -> Optional[float]:
    """Return current Bollinger Band Width's percentile rank (0–100) over
    the trailing `_BB_LOOKBACK` days. Low percentile = squeeze."""
    if "bbw" in df:
        bbw = df["bbw"].dropna()
    else:
        close = df["Close"]
        if len(close) < _BB_WINDOW + 30:
            return None
        sma = close.rolling(_BB_WINDOW).mean()
        sd = close.rolling(_BB_WINDOW).std(ddof=0)
        bbw = (4 * sd) / sma.replace(0, np.nan)
        bbw = bbw.dropna()
    if bbw.empty:
        return None
    window = bbw.tail(_BB_LOOKBACK)
    cur = float(window.iloc[-1])
    rank = float((window < cur).sum()) / max(len(window) - 1, 1)
    return float(np.clip(rank * 100, 0, 100))


def _vcp_score(df: pd.DataFrame) -> Optional[float]:
    """Volatility-contraction score in [0, 1].

    Compares the high-low range over short / mid / long windows. A clean VCP
    has progressively tighter ranges (short < mid < long). Returns ~1.0 for
    a tightly coiled spring, ~0.0 for an expanding (loose) base.
    """
    if len(df) < _VCP_LONG + 5:
        return None
    def _norm_range(window: int) -> float:
        seg = df.tail(window)
        m = float(seg["Close"].mean())
        if m <= 0:
            return np.nan
        return float((seg["High"].max() - seg["Low"].min()) / m)

    r20 = _norm_range(_VCP_SHORT)
    r60 = _norm_range(_VCP_MID)
    r120 = _norm_range(_VCP_LONG)
    if any(not np.isfinite(x) or x <= 0 for x in (r20, r60, r120)):
        return None
    # Each contraction step normalized to [0,1]
    c1 = max(0.0, 1.0 - r20 / r60)
    c2 = max(0.0, 1.0 - r60 / r120)
    return float(np.clip(0.6 * c1 + 0.4 * c2, 0.0, 1.0))


def _ud_ratio(df: pd.DataFrame, window: int) -> Optional[float]:
    """Dollar-volume U/D ratio over `window` days, ATR-filtered to drop
    insignificant moves. Returns None if data insufficient."""
    col = f"ud_{window}"
    if col in df:
        val = df[col].iloc[-1]
        return float(val) if not pd.isna(val) else None
        
    if len(df) < window + _ATR_WINDOW + 2:
        return None
    seg = df.tail(window + 1).copy()
    seg["ret"] = seg["Close"].pct_change()
    seg["dol_vol"] = seg["Close"] * seg["Volume"]
    atr = _atr(df, _ATR_WINDOW).tail(window + 1)
    seg["atr"] = atr.values
    seg = seg.dropna(subset=["ret", "atr"])
    if seg.empty:
        return None
    threshold = _ATR_FILTER_K * (seg["atr"] / seg["Close"])
    sig = seg["ret"].abs() > threshold
    up = float(seg.loc[sig & (seg["ret"] > 0), "dol_vol"].sum())
    dn = float(seg.loc[sig & (seg["ret"] < 0), "dol_vol"].sum())
    if dn <= 0:
        return 5.0 if up > 0 else None  # cap when no down-volume
    return float(up / dn)


def _adr_pct(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    if "adr_pct" in df:
        val = df["adr_pct"].iloc[-1]
        return float(val) if not pd.isna(val) else None
        
    if len(df) < window + 1:
        return None
    seg = df.tail(window)
    rng = (seg["High"] - seg["Low"]) / seg["Close"].replace(0, np.nan)
    return float(rng.mean() * 100)


def _sma_slope_up(sma: pd.Series, lookback: int = 21) -> bool:
    """SMA is trending up if its current value exceeds its value ~lookback
    sessions ago (default ~1 month). Robust to single-day noise."""
    if len(sma.dropna()) < lookback + 1:
        return False
    cur = _safe_iloc(sma, -1)
    prev = _safe_iloc(sma, -1 - lookback)
    if cur is None or prev is None:
        return False
    return cur > prev


def check_market_regime(index_df: pd.DataFrame) -> dict:
    """Determine the broad market regime (Bullish/Bearish/Caution) from index data.
    
    Returns:
        dict: { "regime": str, "score_mult": float, "description": str }
    """
    if index_df is None or index_df.empty or len(index_df) < 200:
        return {"regime": "Neutral", "score_mult": 1.0, "description": "Unknown"}

    close = index_df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    
    p = float(close.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1])
    s200_rising = _sma_slope_up(sma200, 21)
    
    if p > s200:
        if s200_rising and p > s50:
            return {"regime": "Bullish", "score_mult": 1.0, "description": "Risk-On (confirmed uptrend)"}
        return {"regime": "Caution", "score_mult": 0.9, "description": "Risk-Neutral (pullback or slowing)"}
    elif p < s200 * 0.98:  # 2% buffer to avoid whipsaw at the line
        return {"regime": "Bearish", "score_mult": 0.6, "description": "Risk-Off (primary downtrend)"}
    else:
        return {"regime": "Neutral", "score_mult": 0.8, "description": "Sideways / Line-rider"}


# ── Main entry point ───────────────────────────────────────────────────────
def compute(df: pd.DataFrame, regime: str = "Neutral") -> dict:
    """Compute per-ticker uptrend / breakout metrics.

    The returned `score` is an *absolute* 0-100 score (no universe context).
    Use `apply_rs()` after the full universe is analyzed to add the RS
    percentile rank and finalize the score.
    """
    if df is None or df.empty or len(df) < 60:
        return {"score": 50.0, "signals": [], "stage2": False, "trend_template": 0}

    close = df["Close"]
    vol = df["Volume"]
    price = _safe_iloc(close, -1) or 0.0
    if price <= 0:
        return {"score": 50.0, "signals": [], "stage2": False, "trend_template": 0}

    n = len(close)
    sma50 = df["sma50"] if "sma50" in df else close.rolling(50).mean()
    sma150 = df["sma150"] if "sma150" in df else close.rolling(min(150, n)).mean()
    sma200 = df["sma200"] if "sma200" in df else close.rolling(min(200, n)).mean()

    s50 = _safe_iloc(sma50, -1)
    s150 = _safe_iloc(sma150, -1)
    s200 = _safe_iloc(sma200, -1)

    # ── Stage 2 (Weinstein) ────────────────────────────────────────────
    stage2 = bool(
        s150 is not None and s200 is not None
        and price > s150 and price > s200
        and s150 > s200
        and _sma_slope_up(sma150, 21)
        and _sma_slope_up(sma200, 21)
    )

    # ── Trend Template (Minervini, 7 of 8 — RS rank added later) ──────
    high_252 = float(close.tail(min(252, n)).max())
    low_252 = float(close.tail(min(252, n)).min())
    pct_from_52w_high = (price / high_252 - 1) * 100 if high_252 > 0 else 0.0
    pct_above_52w_low = (price / low_252 - 1) * 100 if low_252 > 0 else 0.0

    tt_checks = {
        "price>sma150_and_sma200": s150 is not None and s200 is not None and price > s150 and price > s200,
        "sma150>sma200":           s150 is not None and s200 is not None and s150 > s200,
        "sma200_trending_up":      _sma_slope_up(sma200, 21),
        "sma50>sma150>sma200":     all(x is not None for x in (s50, s150, s200)) and s50 > s150 > s200,
        "price>sma50":             s50 is not None and price > s50,
        "30pct_above_52w_low":     pct_above_52w_low >= 30,
        "within_25pct_of_high":    pct_from_52w_high >= -25,
    }
    tt_count = int(sum(tt_checks.values()))

    # ── Volatility contraction & squeeze ───────────────────────────────
    bbw_pct = _bbw_percentile(df)        # low = squeeze
    vcp = _vcp_score(df)                    # high = tight

    # ── U/D ratio (accumulation proxy) ─────────────────────────────────
    ud_50 = _ud_ratio(df, _UD_WINDOW)
    ud_15 = _ud_ratio(df, _UD_SHORT)
    ud_confirmed = bool(ud_50 is not None and ud_15 is not None
                        and ud_50 >= 1.25 and ud_15 >= 1.25)

    # ── ADR% (volatility regime) ───────────────────────────────────────
    adr = _adr_pct(df, 20)
    adr_in_range = bool(adr is not None and 1.5 <= adr <= 10.0)

    # ── Pivot proximity / breakout trigger ─────────────────────────────
    pivot_window = close.tail(_PIVOT_WINDOW + 1).iloc[:-1]   # exclude today
    pivot = float(pivot_window.max()) if not pivot_window.empty else price
    pivot_dist_pct = (price / pivot - 1) * 100 if pivot > 0 else 0.0

    vol_50_avg = float(vol.tail(50).mean()) if vol.tail(50).mean() else 1.0
    vol_5_avg = float(vol.tail(5).mean()) if len(vol) >= 5 else vol_50_avg
    vol_dryup = bool(vol_50_avg > 0 and (vol_5_avg / vol_50_avg) < _VOL_DRYUP_RATIO)
    today_vol_ratio = float(vol.iloc[-1] / vol_50_avg) if vol_50_avg > 0 else 1.0

    breakout_today = bool(
        price >= pivot * 1.001
        and today_vol_ratio >= _BREAKOUT_VOL_MULT
        and close.iloc[-1] >= (df["High"].iloc[-1] + df["Low"].iloc[-1]) / 2  # close in upper-half of day's range
    )

    # ── Raw momentum (for cross-sectional RS percentile later) ─────────
    def _ret(n_days: int) -> Optional[float]:
        if len(close) < n_days + 1:
            return None
        return float(close.iloc[-1] / close.iloc[-n_days - 1] - 1) * 100

    r_1m = _ret(21)
    r_3m = _ret(63)
    r_6m = _ret(126)
    r_12m = _ret(252)
    # 12-1 momentum (skip last month, the academic standard)
    rs_raw_12_1 = (r_12m - r_1m) if (r_12m is not None and r_1m is not None) else None
    rs_raw_6_1 = (r_6m - r_1m) if (r_6m is not None and r_1m is not None) else None

    # ── Absolute uptrend score (universe-agnostic) ─────────────────────
    score = 50.0
    signals: list[str] = []

    if stage2:
        score += 14; signals.append("Stage 2 uptrend")
    else:
        score -= 10

    score += tt_count * 1.8  # max +12.6 from 7 checks

    # 52w-high proximity zones
    if -10 <= pct_from_52w_high <= 0:
        pct_above_sma200 = (price / s200 - 1) * 100 if s200 and s200 > 0 else 0.0
        if pct_above_sma200 > 35.0:
            score -= 6; signals.append(f"⚠️ At 52WH & Extended ({pct_from_52w_high:+.1f}%, +{pct_above_sma200:.1f}% vs 200DMA)")
        else:
            score += 8; signals.append(f"Near/At 52WH ({pct_from_52w_high:+.1f}%)")
    elif -20 < pct_from_52w_high < -10:
        score += 4
    elif pct_from_52w_high < -30:
        score -= 4

    # Bollinger-Band-Width squeeze (low percentile = compressed)
    if bbw_pct is not None:
        if bbw_pct < 10:
            score += 10; signals.append(f"BBW squeeze ({bbw_pct:.0f}%ile)")
        elif bbw_pct < 25:
            score += 5; signals.append(f"BBW compressed ({bbw_pct:.0f}%ile)")
        elif bbw_pct > 85:
            score -= 3   # expanded volatility = late stage

    # VCP contraction (deeper = tighter spring)
    if vcp is not None:
        score += vcp * 8  # max +8
        if vcp >= 0.5:
            signals.append(f"VCP {vcp:.2f} (tight base)")

    # U/D ratio (accumulation)
    if ud_50 is not None:
        if ud_50 >= 1.5:
            score += 8; signals.append(f"U/D {ud_50:.2f} (accumulation)")
        elif ud_50 >= 1.2:
            score += 4
        elif ud_50 < 0.8:
            score -= 6; signals.append(f"U/D {ud_50:.2f} (distribution)")

    # ADR regime
    if adr is not None and not adr_in_range:
        if adr < 1.5:
            score -= 3; signals.append(f"ADR {adr:.1f}% (too sleepy)")
        elif adr > 10:
            # Waive penalty and reward high-volatility if supported by heavy accumulation
            if ud_50 is not None and ud_50 >= 1.3:
                score += 2; signals.append(f"ADR {adr:.1f}% (high-volatility + accumulation)")
            else:
                score -= 3; signals.append(f"ADR {adr:.1f}% (too noisy)")

    # Breakout trigger (highest-quality entry)
    if breakout_today:
        score += 10; signals.append(f"🚀 Breakout (vol {today_vol_ratio:.1f}x)")
    elif vol_dryup and -15 <= pivot_dist_pct <= -1:
        score += 4; signals.append("Volume dry-up at pivot")

    score = float(np.clip(score, 0, 100))

    # ── Risk Management (Stop Losses) ──────────────────────────────────
    stop_results = stops.compute_stops(
        df=df,
        entry=price,
        pivot=pivot,
        sma50=_safe_iloc(sma50, -1),
        pivot_dist_pct=pivot_dist_pct,
        breakout_today=breakout_today,
        stage2=stage2,
        regime=regime
    )

    out = {
        "score": score,
        "signals": signals,
        "stage2": stage2,
        "trend_template": tt_count,
        "trend_template_pass": bool(tt_count >= 7),
        "pct_from_52w_high": round(pct_from_52w_high, 2),
        "pct_above_52w_low": round(pct_above_52w_low, 2),
        "bbw_pctile": round(bbw_pct, 1) if bbw_pct is not None else None,
        "vcp": round(vcp, 3) if vcp is not None else None,
        "ud_50": round(ud_50, 2) if ud_50 is not None else None,
        "ud_15": round(ud_15, 2) if ud_15 is not None else None,
        "ud_confirmed": ud_confirmed,
        "adr_pct": round(adr, 2) if adr is not None else None,
        "pivot": round(pivot, 2),
        "pivot_dist_pct": round(pivot_dist_pct, 2),
        "vol_dryup": vol_dryup,
        "today_vol_ratio": round(today_vol_ratio, 2),
        "breakout_today": breakout_today,
        "rs_raw_12_1": rs_raw_12_1,         # used in apply_rs() below
        "rs_raw_6_1": rs_raw_6_1,
        "ret_1m": r_1m, "ret_3m": r_3m, "ret_6m": r_6m, "ret_1y": r_12m,
    }
    out.update(stop_results)
    return out


# ── Cross-sectional pass: RS percentile + final score adjustment ──────────
def apply_rs(reports: Iterable, market_regime: Optional[dict] = None) -> None:
    """Compute IBD-style RS percentile across the universe and finalize the
    uptrend score. Operates in place on the `.uptrend` dict of each report.

    Adds:
      1. RS Percentile (Relative Strength)
      2. Sector Strength Bonus (invest in leading groups)
      3. Market Regime Multiplier (defensive scaling)
    """
    reps = [r for r in reports if getattr(r, "uptrend", None)]
    if len(reps) < 5:
        for r in reps:
            r.uptrend.setdefault("rs_pct", 50.0)
            r.uptrend.setdefault("trend_template_full", r.uptrend.get("trend_template", 0))
        return

    # 1. Collect raw RS values
    raws: list[Optional[float]] = []
    for r in reps:
        v = r.uptrend.get("rs_raw_12_1")
        if v is None:
            v = r.uptrend.get("rs_raw_6_1")
            if v is not None:
                v = v * 1.4
        raws.append(v)

    valid = np.array([x for x in raws if x is not None], dtype=float)
    if valid.size < 5:
        for r in reps:
            r.uptrend["rs_pct"] = 50.0
            r.uptrend["trend_template_full"] = r.uptrend.get("trend_template", 0)
        return

    # 2. RS Percentiles
    sorted_valid = np.sort(valid)
    for r, raw in zip(reps, raws):
        if raw is None:
            rs_pct = 50.0
        else:
            rank = float(np.searchsorted(sorted_valid, raw, side="left"))
            rs_pct = float(np.clip(rank / max(len(sorted_valid) - 1, 1) * 100, 0, 100))
        r.uptrend["rs_pct"] = round(rs_pct, 1)

    # 3. Sector Strength Analysis
    # Group by sector and compute average uptrend score
    sector_scores: dict[str, list[float]] = {}
    for r in reps:
        sec = r.sector or "Unknown"
        score = r.uptrend.get("score", 50.0)
        sector_scores.setdefault(sec, []).append(score)
    
    sector_avgs = {s: float(np.mean(vals)) for s, vals in sector_scores.items() if len(vals) >= 2}
    
    # 4. Final scoring pass
    regime_mult = market_regime.get("score_mult", 1.0) if market_regime else 1.0

    for r in reps:
        # 8th Trend-Template check: RS rank ≥ 70
        rs_pct = r.uptrend["rs_pct"]
        tt = int(r.uptrend.get("trend_template", 0))
        tt_full = tt + (1 if rs_pct >= 70 else 0)
        r.uptrend["trend_template_full"] = tt_full
        r.uptrend["trend_template_pass"] = bool(tt_full >= 7)

        # Layer RS rank into the score
        base = float(r.uptrend.get("score", 50.0))
        bump = 0.0
        if rs_pct >= 90: bump = 10.0
        elif rs_pct >= 80: bump = 7.0
        elif rs_pct >= 70: bump = 4.0
        elif rs_pct <= 30: bump = -5.0

        # Sector Strength Bonus (invest in hot groups)
        sec = r.sector or "Unknown"
        sec_avg = sector_avgs.get(sec, 50.0)
        if sec_avg >= 65:
            bump += 5.0
            r.uptrend["signals"].append(f"Sector Leader ({sec})")
        elif sec_avg <= 40:
            bump -= 5.0
            r.uptrend["signals"].append(f"Sector Laggard ({sec})")

        # Apply Market Regime Scaling (Defensive mechanism moved to portfolio sizing)
        final_score = base + bump
        
        r.uptrend["score"] = round(float(np.clip(final_score, 0, 100)), 1)
        if market_regime and market_regime.get("regime") == "Bearish":
            r.uptrend["signals"].append(f"⚠️ Weak market conditions ({market_regime['regime']})")

        # Composite "Leader" flag
        r.uptrend["is_leader"] = bool(
            r.uptrend.get("stage2")
            and rs_pct >= 70
            and r.uptrend.get("pct_from_52w_high", -100) >= -25
            and (r.uptrend.get("ud_50") or 0) >= 1.0
        )
