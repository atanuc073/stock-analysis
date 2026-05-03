"""Momentum: multi-period returns with academic 12-1 factor.

Key signal: 12-1 momentum (12-month return MINUS most recent month) is the
most robust momentum factor in finance literature (Jegadeesh & Titman 1993,
follow-on work). It rewards stocks with long-term strength + short-term pause
— exactly the entry pattern we want (= base before next leg up). Pure 1M
momentum catches tops because by definition a +25% 1M move IS a top.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {"score": 50.0, "signals": []}
    close = df["Close"]
    ret = lambda n: (close.iloc[-1] / close.iloc[-n] - 1) * 100 if len(close) >= n else None
    r5 = ret(5); r21 = ret(21); r63 = ret(63); r126 = ret(126); r252 = ret(252)

    # 12-1 momentum: 12M return minus 1M return → long-term strength + recent pause
    mom_12_1 = (r252 - r21) if (r252 is not None and r21 is not None) else None
    # 6-1 momentum (shorter-cycle equivalent for stocks without 252d history)
    mom_6_1 = (r126 - r21) if (r126 is not None and r21 is not None) else None

    score = 50.0
    signals = []

    # Primary signal: 12-1 (highest weight)
    if mom_12_1 is not None:
        # Reward: long-term up but not blow-off in last month
        score += np.clip(mom_12_1, -30, 40) * 0.5
        if mom_12_1 > 20:
            signals.append(f"12-1 mom +{mom_12_1:.1f}% (strong base)")
        elif mom_12_1 < -15:
            signals.append(f"12-1 mom {mom_12_1:.1f}% (downtrend)")
    elif mom_6_1 is not None:
        score += np.clip(mom_6_1, -25, 30) * 0.4
        if mom_6_1 > 15:
            signals.append(f"6-1 mom +{mom_6_1:.1f}%")

    # Penalize blow-off short-term spikes (= top-buying)
    if r21 is not None and r21 > 25:
        score -= 8
        signals.append(f"⚠️ Spike +{r21:.1f}% in 1M (extended)")
    elif r21 is not None and r21 > 15:
        score -= 3

    # Mild reward for steady 3M strength (not parabolic)
    if r63 is not None:
        if 5 <= r63 <= 25:
            score += 5
            signals.append(f"+{r63:.1f}% 3M (steady)")
        elif r63 < -15:
            score -= 6
            signals.append(f"{r63:.1f}% 3M (weak)")

    # Mild penalty if 1W is sharply negative (catching falling knife)
    if r5 is not None and r5 < -8:
        score -= 4
        signals.append(f"{r5:.1f}% last week (knife-catch risk)")

    score = float(np.clip(score, 0, 100))
    return {
        "score": score, "signals": signals,
        "ret_1w": r5, "ret_1m": r21, "ret_3m": r63,
        "ret_6m": r126, "ret_1y": r252,
        "mom_12_1": mom_12_1, "mom_6_1": mom_6_1,
    }
