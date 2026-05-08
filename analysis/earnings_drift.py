"""Post-Earnings Announcement Drift (PEAD) factor.

Empirical fact (Bernard & Thomas 1989; revisited many times since): stocks
that beat earnings drift in the direction of the surprise for 30–60 trading
days. Stocks that miss drift down. Trading on the drift — not the report —
is one of the most persistent anomalies in equities.

We can't get a true beat-vs-consensus from yfinance, so we approximate using:

  1. Earnings-quarterly-growth (yoy EPS growth, signed)            — surprise proxy
  2. Forward-vs-trailing EPS                                         — guidance proxy
  3. Largest single-day gap in the last 95 trading days + above SMA  — drift detection
  4. Days since that gap (drift is strongest in the first 60 days)

Output: standard {score, signals, ...} contract, neutral 50 baseline.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def _safe_float(v) -> float | None:
    try:
        f = float(v) if v is not None else None
        if f is None or not math.isfinite(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _detect_recent_gap(df: pd.DataFrame, window: int = 95,
                       min_gap_pct: float = 3.0) -> tuple[float | None, int | None]:
    """Find the largest abs single-day gap (open vs prior close) in the window.

    Returns (gap_pct, days_ago) or (None, None) if no significant gap.
    A gap > min_gap_pct in absolute terms is treated as the most likely
    earnings/news reaction.
    """
    if df.empty or len(df) < 5:
        return None, None
    tail = df.tail(window).copy()
    if "Open" not in tail.columns or "Close" not in tail.columns:
        return None, None
    prev_close = tail["Close"].shift(1)
    gap_pct = (tail["Open"] / prev_close - 1.0) * 100.0
    gap_pct = gap_pct.dropna()
    if gap_pct.empty:
        return None, None
    abs_gaps = gap_pct.abs()
    max_idx = abs_gaps.idxmax()
    max_gap = float(gap_pct.loc[max_idx])
    if abs(max_gap) < min_gap_pct:
        return None, None
    days_ago = len(tail) - tail.index.get_loc(max_idx) - 1
    return max_gap, int(days_ago)


def compute(history: pd.DataFrame, info: dict | None) -> dict:
    """Score recent earnings momentum + drift.

    Args:
        history: OHLCV daily DataFrame (must include Open, Close).
        info: yfinance .info dict (may be None or empty).
    """
    if history is None or history.empty:
        return {"score": 50.0, "signals": [], "available": False}

    info = info or {}
    score = 50.0
    signals: list[str] = []

    eps_q_growth = _safe_float(info.get("earningsQuarterlyGrowth"))   # yoy
    rev_q_growth = _safe_float(info.get("revenueQuarterlyGrowth"))    # yoy
    eps_ttm      = _safe_float(info.get("trailingEps"))
    eps_fwd      = _safe_float(info.get("forwardEps"))

    # ── 1. Quarterly EPS surprise proxy ──────────────────────────────
    if eps_q_growth is not None:
        # 0% → 0, +30% → +10, +60% → +12 (capped)
        pts = float(np.clip(eps_q_growth * 25, -12, 12))
        score += pts
        if eps_q_growth > 0.20:
            signals.append(f"EPS surge yoy +{eps_q_growth*100:.0f}%")
        elif eps_q_growth < -0.20:
            signals.append(f"EPS drop yoy {eps_q_growth*100:.0f}%")

    if rev_q_growth is not None:
        pts = float(np.clip(rev_q_growth * 20, -6, 6))
        score += pts
        if rev_q_growth > 0.15:
            signals.append(f"Rev surge yoy +{rev_q_growth*100:.0f}%")

    # ── 2. Forward guidance proxy ────────────────────────────────────
    if eps_ttm is not None and eps_fwd is not None and eps_ttm != 0:
        guidance = (eps_fwd - eps_ttm) / abs(eps_ttm)
        # +20% guidance bump → +6, -20% → -6
        score += float(np.clip(guidance * 30, -6, 6))
        if guidance > 0.15:
            signals.append(f"Forward EPS +{guidance*100:.0f}%")
        elif guidance < -0.15:
            signals.append(f"Forward EPS {guidance*100:.0f}%")

    # ── 3. Drift detection from price action ─────────────────────────
    gap_pct, days_ago = _detect_recent_gap(history, window=95, min_gap_pct=3.0)
    if gap_pct is not None and days_ago is not None:
        # Drift strength fades over time: full strength 0–30d, half 30–60d,
        # near zero >60d. Direction matches the gap sign.
        if days_ago <= 30:
            decay = 1.0
        elif days_ago <= 60:
            decay = 0.5
        else:
            decay = 0.15
        # Cap raw gap influence at ±10% so a 30% blow-off doesn't dominate
        capped = float(np.clip(gap_pct, -10, 10))
        pts = capped * decay
        score += pts
        if gap_pct > 5 and days_ago <= 60:
            signals.append(f"Earnings gap +{gap_pct:.1f}% ({days_ago}d ago)")
        elif gap_pct < -5 and days_ago <= 60:
            signals.append(f"⚠️ Earnings gap {gap_pct:.1f}% ({days_ago}d ago)")

    # ── 4. Mismatch flag: surprise positive but price didn't move ────
    # (= weak follow-through, drift unlikely to materialize)
    if (eps_q_growth is not None and eps_q_growth > 0.20
            and (gap_pct is None or abs(gap_pct) < 2)):
        score -= 3
        signals.append("EPS beat but no price reaction (weak)")

    score = float(np.clip(score, 0, 100))
    return {
        "score": score,
        "signals": signals,
        "available": True,
        "eps_q_growth": eps_q_growth,
        "rev_q_growth": rev_q_growth,
        "earnings_gap_pct": gap_pct,
        "earnings_gap_days_ago": days_ago,
    }
