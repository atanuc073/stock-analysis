"""Quality factor — Novy-Marx style profitability + earnings stability + balance-sheet health.

Quality is the single best filter against value traps and momentum-blowups.
Combines:
  - Gross-profit-to-assets (Novy-Marx 2013 — strongest single quality metric)
  - Operating margin / profit margin
  - ROA & ROIC proxy
  - Free-cash-flow margin (cash, not just accruals)
  - Earnings stability proxy (low EPS volatility / no recent loss)
  - Accruals red-flag (high accruals ratio = earnings manipulation risk)

Outputs the standard {score, signals, ...} contract.
"""
from __future__ import annotations
import math
import numpy as np


def _safe_float(val) -> float | None:
    try:
        f = float(val) if val is not None else None
        if f is None or not math.isfinite(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _smooth(value: float | None, lo: float, hi: float, max_pts: float,
            invert: bool = False) -> float:
    """Sigmoid mapping into [-max_pts, +max_pts]."""
    if value is None or not np.isfinite(value):
        return 0.0
    midpoint = (lo + hi) / 2.0
    half_range = max(abs(hi - lo) / 2.0, 1e-9)
    k = 4.0 / (2 * half_range)
    z = k * (value - midpoint)
    if invert:
        z = -z
    if z > 50:
        s = 1.0
    elif z < -50:
        s = 0.0
    else:
        s = 1.0 / (1.0 + math.exp(-z))
    return (2 * s - 1) * max_pts


def compute(info: dict) -> dict:
    """Return {score, signals, ...} for the quality factor."""
    if not info:
        return {"score": 50.0, "signals": [], "error": "no info"}

    score = 50.0
    signals: list[str] = []

    # ── Inputs ────────────────────────────────────────────────────────
    gross_margin = _safe_float(info.get("grossMargins"))      # 0..1
    op_margin    = _safe_float(info.get("operatingMargins"))  # 0..1
    profit_margin = _safe_float(info.get("profitMargins"))    # 0..1
    roa = _safe_float(info.get("returnOnAssets"))             # 0..1
    roe = _safe_float(info.get("returnOnEquity"))             # 0..1
    fcf = _safe_float(info.get("freeCashflow"))               # absolute $
    op_cf = _safe_float(info.get("operatingCashflow"))        # absolute $
    revenue = _safe_float(info.get("totalRevenue"))           # absolute $
    total_assets = _safe_float(info.get("totalAssets"))       # absolute $
    total_debt = _safe_float(info.get("totalDebt"))
    cash = _safe_float(info.get("totalCash"))
    market_cap = _safe_float(info.get("marketCap"))
    de = _safe_float(info.get("debtToEquity"))
    current_ratio = _safe_float(info.get("currentRatio"))
    eps_ttm = _safe_float(info.get("trailingEps"))
    eps_fwd = _safe_float(info.get("forwardEps"))

    # ── 1. Gross-profit-to-assets (Novy-Marx) ─────────────────────────
    gpa = None
    if gross_margin is not None and revenue is not None and total_assets and total_assets > 0:
        gpa = (gross_margin * revenue) / total_assets
        # 0.10 → 0, 0.40 → +10
        pts = _smooth(gpa, lo=0.10, hi=0.40, max_pts=10)
        score += pts
        if gpa > 0.30:
            signals.append(f"High GPA ({gpa:.2f})")

    # ── 2. Operating margin (≈ pricing power) ────────────────────────
    if op_margin is not None:
        # 5% → 0, 25% → +6
        score += _smooth(op_margin * 100, lo=5, hi=25, max_pts=6)
        if op_margin > 0.20:
            signals.append(f"Op margin {op_margin*100:.1f}%")

    # ── 3. ROA (capital efficiency) ───────────────────────────────────
    if roa is not None:
        # 3% → 0, 12% → +6
        score += _smooth(roa * 100, lo=3, hi=12, max_pts=6)
        if roa > 0.10:
            signals.append(f"Strong ROA ({roa*100:.1f}%)")

    # ── 4. ROIC proxy: ROE × (1 − debt share) ────────────────────────
    # If D/E is high, ROE is leveraged — penalize.
    if roe is not None and de is not None:
        debt_share = min(max(de / 100.0, 0.0), 1.0) / 2.0  # 100% D/E → 50% adj
        roic_proxy = roe * (1 - debt_share)
        score += _smooth(roic_proxy * 100, lo=4, hi=15, max_pts=6)
        if roic_proxy > 0.15:
            signals.append(f"High ROIC proxy ({roic_proxy*100:.1f}%)")

    # ── 5. FCF margin / FCF yield (cash quality) ─────────────────────
    if fcf is not None and revenue and revenue > 0:
        fcf_margin = fcf / revenue
        # 5% → 0, 20% → +5
        score += _smooth(fcf_margin * 100, lo=5, hi=20, max_pts=5)
        if fcf_margin > 0.15:
            signals.append(f"FCF margin {fcf_margin*100:.1f}%")

    if fcf is not None and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap
        # 2% → 0, 8% → +5 (cheap on cash terms)
        score += _smooth(fcf_yield * 100, lo=2, hi=8, max_pts=5)
        if fcf_yield > 0.06:
            signals.append(f"FCF yield {fcf_yield*100:.1f}%")

    # ── 6. Accruals red-flag ─────────────────────────────────────────
    # If profits are high but operating cashflow is much lower → accruals risk.
    if op_cf is not None and profit_margin is not None and revenue and revenue > 0:
        cash_margin = op_cf / revenue
        accrual_gap = profit_margin - cash_margin
        if accrual_gap > 0.10:           # net income >> cash
            score -= 6
            signals.append(f"⚠️ High accruals (gap {accrual_gap*100:.1f}%)")
        elif accrual_gap < -0.05:        # cash > earnings — conservative
            score += 2

    # ── 7. Balance-sheet health ──────────────────────────────────────
    if current_ratio is not None:
        if current_ratio < 1.0:
            score -= 4
            signals.append(f"⚠️ Current ratio {current_ratio:.2f}")
        elif current_ratio > 1.5:
            score += 2

    # Net cash position (cash > debt) — Buffett-style fortress
    if cash is not None and total_debt is not None and market_cap and market_cap > 0:
        net_cash = cash - total_debt
        net_cash_yield = net_cash / market_cap
        if net_cash_yield > 0.10:        # 10%+ of mcap is net cash
            score += 4
            signals.append(f"Net cash {net_cash_yield*100:.0f}% of mcap")
        elif net_cash_yield < -0.50:     # debt >> cash + heavily levered
            score -= 4

    # ── 8. Earnings stability proxy ──────────────────────────────────
    # No trailing EPS = recent loss / restructuring → quality penalty.
    if eps_ttm is not None and eps_ttm < 0:
        score -= 6
        signals.append("⚠️ TTM loss")
    elif eps_ttm is not None and eps_fwd is not None:
        # Forward EPS materially below trailing → declining quality
        if eps_ttm > 0 and eps_fwd < eps_ttm * 0.7:
            score -= 4
            signals.append("Declining forward EPS")

    score = float(np.clip(score, 0, 100))
    return {
        "score": score,
        "signals": signals,
        "gpa": gpa,
        "op_margin": op_margin,
        "roa": roa,
        "fcf_margin": (fcf / revenue) if fcf is not None and revenue else None,
        "fcf_yield": (fcf / market_cap) if fcf is not None and market_cap else None,
    }
