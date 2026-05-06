"""Fundamental analysis — uses yfinance .info dict.

Scoring uses smooth (sigmoid / piecewise-linear) functions instead of hard
threshold cliffs so that small changes in inputs produce small changes in
score (e.g. P/E 14.99 vs 15.01 should not swing the score by 6 points).
"""
from __future__ import annotations
import math
import numpy as np


def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid, clamped for numeric safety."""
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _smooth_score(value: float, lo: float, hi: float, max_pts: float,
                  invert: bool = False) -> float:
    """Map ``value`` to a smooth score in [-max_pts, +max_pts].

    ``lo`` is the threshold where the score crosses 0 (neutral) on the bad
    side; ``hi`` is the threshold where it crosses 0 on the good side. Slope
    is set so the transition is gentle (~80% of the swing happens between
    lo and hi). Set ``invert=True`` when *lower* values are better (e.g. P/E).
    """
    if value is None or not np.isfinite(value):
        return 0.0
    midpoint = (lo + hi) / 2.0
    half_range = max(abs(hi - lo) / 2.0, 1e-9)
    # k chosen so sigmoid(±2)≈0.88 — most of the swing happens within [lo,hi]
    k = 4.0 / (2 * half_range)
    z = k * (value - midpoint)
    if invert:
        z = -z
    # Sigmoid in [0,1] → re-center to [-max_pts, +max_pts]
    return (2 * _sigmoid(z) - 1) * max_pts


def compute(info: dict) -> dict:
    if not info:
        return {"score": 50.0, "signals": [], "error": "no fundamentals"}

    pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
    pb = _safe_float(info.get("priceToBook"))
    roe = _safe_float(info.get("returnOnEquity"))
    de = _safe_float(info.get("debtToEquity"))
    eps_growth = _safe_float(info.get("earningsGrowth"))
    rev_growth = _safe_float(info.get("revenueGrowth"))
    margin = _safe_float(info.get("profitMargins"))
    div_yield = _safe_float(info.get("dividendYield"))
    market_cap = _safe_float(info.get("marketCap"))

    score = 50.0
    signals = []

    # ── Smooth P/E score ──────────────────────────────────────────────
    # Cheap (P/E ~10) → +9, fair (P/E ~25) → 0, expensive (P/E ~50) → -8
    if pe is not None and pe > 0:
        pe_pts = _smooth_score(pe, lo=15, hi=40, max_pts=9, invert=True)
        score += pe_pts
        if pe < 15:
            signals.append(f"Low P/E ({pe:.1f})")
        elif pe > 40:
            signals.append(f"Rich/High P/E ({pe:.1f})")

    # ── Smooth P/B score (lower better) ──────────────────────────────
    if pb is not None and pb > 0:
        pb_pts = _smooth_score(pb, lo=2, hi=6, max_pts=4, invert=True)
        score += pb_pts
        if pb < 2:
            signals.append(f"Low P/B ({pb:.2f})")

    # ── Smooth ROE score (higher better) ─────────────────────────────
    if roe is not None:
        # 5% ROE → -4, 18% ROE → +8 (smooth)
        roe_pts = _smooth_score(roe * 100, lo=5, hi=18, max_pts=8)
        score += roe_pts
        if roe > 0.18:
            signals.append(f"Strong ROE ({roe*100:.1f}%)")

    # ── Smooth debt/equity (lower better) ────────────────────────────
    if de is not None:
        # D/E 50 → +4, D/E 200 → -6 (skewed toward penalty for high debt)
        de_pts = _smooth_score(de, lo=50, hi=200, max_pts=5, invert=True)
        score += de_pts - 1  # slight overall penalty for any debt
        if de < 50:
            signals.append("Low debt")
        elif de > 200:
            signals.append(f"High debt (D/E {de:.0f})")

    # ── Growth (smooth) ──────────────────────────────────────────────
    if eps_growth is not None:
        # 0% → 0, 15% → +6
        eps_pts = _smooth_score(eps_growth * 100, lo=0, hi=20, max_pts=6)
        score += eps_pts
        if eps_growth > 0.15:
            signals.append(f"EPS growth {eps_growth*100:.0f}%")

    if rev_growth is not None:
        rev_pts = _smooth_score(rev_growth * 100, lo=0, hi=15, max_pts=4)
        score += rev_pts
        if rev_growth > 0.10:
            signals.append(f"Revenue growth {rev_growth*100:.0f}%")

    if margin is not None:
        # 5% → 0, 20% → +4
        margin_pts = _smooth_score(margin * 100, lo=5, hi=20, max_pts=4)
        score += margin_pts

    if div_yield and div_yield > 0.03:
        signals.append(f"Dividend {div_yield*100:.1f}%")

    score = float(np.clip(score, 0, 100))
    return {
        "score": score,
        "signals": signals,
        "pe": pe, "pb": pb, "roe": roe, "debt_to_equity": de,
        "eps_growth": eps_growth, "revenue_growth": rev_growth,
        "profit_margin": margin, "dividend_yield": div_yield,
        "market_cap": market_cap,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "name": info.get("shortName") or info.get("longName"),
    }
