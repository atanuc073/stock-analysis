"""Momentum: multi-period returns."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {"score": 50.0, "signals": []}
    close = df["Close"]
    ret = lambda n: (close.iloc[-1] / close.iloc[-n] - 1) * 100 if len(close) >= n else None
    r5 = ret(5); r21 = ret(21); r63 = ret(63); r252 = ret(252)

    score = 50.0
    signals = []
    for r, w, label in [(r5, 0.5, "1W"), (r21, 1.0, "1M"), (r63, 1.5, "3M"), (r252, 1.0, "1Y")]:
        if r is None:
            continue
        score += np.clip(r, -20, 20) * w * 0.5
        if r > 10:
            signals.append(f"+{r:.1f}% {label}")
        elif r < -10:
            signals.append(f"{r:.1f}% {label}")

    score = float(np.clip(score, 0, 100))
    return {
        "score": score, "signals": signals,
        "ret_1w": r5, "ret_1m": r21, "ret_3m": r63, "ret_1y": r252,
    }
