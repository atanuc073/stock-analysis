"""Indicator helpers used by sizing/lifecycle. Pure functions."""
from __future__ import annotations
import numpy as np
import pandas as pd


def atr(history: pd.DataFrame, window: int = 14) -> float:
    """14-day Average True Range. Returns 0 if insufficient data."""
    if history is None or len(history) < window + 1:
        return 0.0
    high = history["High"]
    low = history["Low"]
    close = history["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1])


def annualized_volatility(history: pd.DataFrame, window: int = 90) -> float:
    """Annualized volatility from daily log returns."""
    if history is None or len(history) < 30:
        return 0.30  # fallback
    rets = np.log(history["Close"] / history["Close"].shift(1)).dropna().tail(window)
    if len(rets) < 10:
        return 0.30
    return float(rets.std() * np.sqrt(252))
