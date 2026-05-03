"""Historical regime detector for backtest.

Mirrors `risk/regime.py` logic but computes from a pre-loaded benchmark history
(no live yfinance calls, no lookahead — slices to <=asof).

Score is 0-10 across these components, weighted to match the live detector:
  +2 price > 200DMA
  +2 50DMA > 200DMA
  +2 200DMA rising (vs 21d ago)
  +1 1M return > 0
  +1 3M return > 0
  +1 VIX < calm (18)
  +1 VIX < stress (28)

Labels and allocation multipliers identical to the live detector.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Same labels/allocations as risk/regime.py — keeps live and backtest aligned
LABELS = [
    ((8, 10), "BULL"),
    ((6, 7), "NEUTRAL_BULL"),
    ((4, 5), "NEUTRAL"),
    ((2, 3), "CAUTIOUS"),
    ((0, 1), "BEAR"),
]
ALLOCATION = {
    "BULL": 1.0, "NEUTRAL_BULL": 0.85, "NEUTRAL": 0.70,
    "CAUTIOUS": 0.45, "BEAR": 0.15,
}
SMA_SHORT = 50
SMA_LONG = 200
VIX_CALM = 20.0
VIX_STRESS = 28.0


@dataclass
class HistoricalRegime:
    score: int
    label: str
    allocation_multiplier: float
    notes: list[str]


def _label_for(score: int) -> str:
    for (lo, hi), lbl in LABELS:
        if lo <= score <= hi:
            return lbl
    return "NEUTRAL"


def _strip_tz(s: pd.Series) -> pd.Series:
    if s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s


def _slice(s: pd.Series, asof: pd.Timestamp) -> pd.Series:
    s = _strip_tz(s)
    return s[s.index <= asof]


def load_benchmark(symbol: str, start: str, end: str) -> pd.Series:
    """Load close prices for an index/VIX. Used at backtest startup."""
    try:
        df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return pd.Series(dtype=float)
        s = df["Close"]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s
    except Exception as e:
        log.warning("Failed to load benchmark %s: %s", symbol, e)
        return pd.Series(dtype=float)


def detect(
    index_history: pd.Series,
    vix_history: Optional[pd.Series],
    asof: pd.Timestamp,
) -> HistoricalRegime:
    """Compute regime as of `asof` using only data <= asof."""
    notes: list[str] = []
    score = 0

    idx = _slice(index_history, asof) if not index_history.empty else pd.Series(dtype=float)
    if len(idx) >= SMA_LONG:
        sma_s = idx.rolling(SMA_SHORT).mean().iloc[-1]
        sma_l = idx.rolling(SMA_LONG).mean().iloc[-1]
        price = idx.iloc[-1]
        sma_l_prev = (idx.rolling(SMA_LONG).mean().iloc[-21]
                      if len(idx) >= SMA_LONG + 21 else sma_l)

        if price > sma_l:
            score += 2
        if sma_s > sma_l:
            score += 2
        if sma_l > sma_l_prev:
            score += 2

        ret_1m = idx.iloc[-1] / idx.iloc[-21] - 1 if len(idx) >= 21 else 0
        ret_3m = idx.iloc[-1] / idx.iloc[-63] - 1 if len(idx) >= 63 else 0
        if ret_1m > 0:
            score += 1
        if ret_3m > 0:
            score += 1
        notes.append(f"Index 1M {ret_1m*100:+.1f}%, 3M {ret_3m*100:+.1f}%")
    else:
        notes.append("Insufficient index history")

    if vix_history is not None and not vix_history.empty:
        vsl = _slice(vix_history, asof)
        if not vsl.empty:
            vix = float(vsl.iloc[-1])
            if vix < VIX_CALM:
                score += 1
            if vix < VIX_STRESS:
                score += 1
            notes.append(f"VIX {vix:.1f}")

    score = min(10, score)
    label = _label_for(score)
    return HistoricalRegime(
        score=score,
        label=label,
        allocation_multiplier=ALLOCATION[label],
        notes=notes,
    )
