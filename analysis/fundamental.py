"""Fundamental analysis — uses yfinance .info dict."""
from __future__ import annotations
import numpy as np


def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None

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

    if pe is not None:
        if 0 < pe < 15:
            score += 8; signals.append(f"Low P/E ({pe:.1f})")
        elif pe > 60:
            score -= 6; signals.append(f"High P/E ({pe:.1f})")
    if pb is not None:
        if 0 < pb < 2:
            score += 4; signals.append(f"Low P/B ({pb:.2f})")
        elif pb > 8:
            score -= 4
    if roe is not None:
        if roe > 0.18:
            score += 8; signals.append(f"Strong ROE ({roe*100:.1f}%)")
        elif roe < 0.05:
            score -= 4
    if de is not None:
        if de < 50:
            score += 4; signals.append("Low debt")
        elif de > 200:
            score -= 6; signals.append(f"High debt (D/E {de:.0f})")
    if eps_growth is not None and eps_growth > 0.15:
        score += 6; signals.append(f"EPS growth {eps_growth*100:.0f}%")
    if rev_growth is not None and rev_growth > 0.10:
        score += 4; signals.append(f"Revenue growth {rev_growth*100:.0f}%")
    if margin is not None and margin > 0.15:
        score += 4
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
