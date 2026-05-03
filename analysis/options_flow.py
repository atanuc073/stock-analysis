"""Options flow analysis (US tickers via yfinance options chain)."""
from __future__ import annotations


def compute(options_summary: dict | None) -> dict:
    if not options_summary:
        return {"score": 50.0, "signals": [], "available": False}
    pcr = options_summary.get("pcr")
    cv = options_summary.get("call_volume", 0)
    pv = options_summary.get("put_volume", 0)
    score = 50.0
    signals = []
    if pcr is not None:
        if pcr < 0.6:
            score += 10; signals.append(f"Bullish options flow (PCR {pcr})")
        elif pcr > 1.3:
            score -= 10; signals.append(f"Bearish options flow (PCR {pcr})")
        # Unusual total volume hint
        if cv + pv > 50000:
            signals.append(f"Heavy options activity ({cv+pv:,} contracts)")
    return {
        "score": float(max(0, min(100, score))),
        "signals": signals,
        "available": True,
        "put_call_ratio": pcr,
        "call_volume": cv,
        "put_volume": pv,
        "expiry": options_summary.get("expiry"),
    }
