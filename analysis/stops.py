"""Stop-loss calculator.

Produces multiple stop-loss prices for a stock using established trend-trading
methodologies, then selects the best one based on entry context.

Strategies implemented
----------------------
1. **Pivot stop** — `pivot × (1 - buffer)`. Best for fresh breakouts.
   Standard buffer: 7-8% (O'Neil / IBD / Minervini).
2. **ATR stop** — `entry - k × ATR(14)`. Volatility-adjusted, mechanical.
   Standard k: 2.0 - 3.0; default 2.5.
3. **SMA-50 stop** — `SMA50 × (1 - buffer)`. Best for position trades in
   Stage 2 uptrends. Standard buffer: 2-3% below close-only basis.
4. **Swing-low stop** — N-day swing low × (1 - buffer). Tight base entries.
5. **Chandelier exit** — `HighestHigh(N) - k × ATR(N)`. Used for trailing.

Selection logic (`pick_best_stop`)
---------------------------------
- Fresh breakout (within 3% of pivot) → **Pivot stop** capped at 2.5×ATR.
- Stage 2, well-above 50-day MA (>8%) → **ATR stop**.
- Stage 2, near 50-day MA (within 5%) → **SMA-50 stop** capped at 3×ATR.
- Otherwise → **ATR stop** (default safe fallback).

In all cases the final stop is `MAX(structural_stop, entry - cap×ATR)` to
prevent giving back too much when the structural level is unusually far.

Position sizing
---------------
Given an account size and per-trade risk %, returns suggested shares:
    shares = floor((account × risk_pct) / (entry - stop))

This is the part most retail traders skip. Wider stops automatically mean
smaller positions, which is the only way a 30-50% win rate trend system
stays profitable.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
import pandas as pd


# ── Defaults (override per call if you want to experiment) ────────────────
DEFAULT_ATR_WINDOW = 14
DEFAULT_ATR_MULT = 2.5            # initial-stop multiplier
DEFAULT_PIVOT_BUFFER = 0.075      # 7.5% below pivot
DEFAULT_SMA_BUFFER = 0.03         # 3% below SMA50
DEFAULT_SWING_BUFFER = 0.02       # 2% below 20-day swing low
DEFAULT_SWING_WINDOW = 20
DEFAULT_CHANDELIER_WINDOW = 22
DEFAULT_CHANDELIER_K = 3.0
ATR_CAP_MULT = 3.0                # never let stop be wider than this × ATR

# Position sizing defaults — change here or pass via kwargs
DEFAULT_ACCOUNT_SIZE = 100_000.0
DEFAULT_RISK_PER_TRADE = 0.0075   # 0.75% of account per trade
MAX_POSITION_PCT = 0.20           # cap any single position at 20% of account


# ── ATR (Wilder-style with simple moving average smoothing) ───────────────
def atr(df: pd.DataFrame, window: int = DEFAULT_ATR_WINDOW) -> pd.Series:
    """Average True Range. Returns a pandas Series aligned to df.index."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _last(series: pd.Series) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    v = series.iloc[-1]
    if pd.isna(v):
        return None
    return float(v)


# ── Individual stop strategies ────────────────────────────────────────────
def pivot_stop(pivot: float, buffer: float = DEFAULT_PIVOT_BUFFER) -> Optional[float]:
    if pivot is None or pivot <= 0:
        return None
    return round(pivot * (1.0 - buffer), 2)


def atr_stop(entry: float, atr_value: float,
             mult: float = DEFAULT_ATR_MULT) -> Optional[float]:
    if entry is None or atr_value is None or entry <= 0 or atr_value <= 0:
        return None
    return round(entry - mult * atr_value, 2)


def sma_stop(sma_value: float, buffer: float = DEFAULT_SMA_BUFFER) -> Optional[float]:
    if sma_value is None or sma_value <= 0:
        return None
    return round(sma_value * (1.0 - buffer), 2)


def swing_low_stop(df: pd.DataFrame,
                   window: int = DEFAULT_SWING_WINDOW,
                   buffer: float = DEFAULT_SWING_BUFFER) -> Optional[float]:
    if df is None or len(df) < window + 1:
        return None
    low = float(df["Low"].tail(window).min())
    if low <= 0:
        return None
    return round(low * (1.0 - buffer), 2)


def chandelier_stop(df: pd.DataFrame,
                    atr_value: float,
                    window: int = DEFAULT_CHANDELIER_WINDOW,
                    k: float = DEFAULT_CHANDELIER_K) -> Optional[float]:
    """Trailing stop: highest high over `window` days minus `k × ATR`."""
    if df is None or atr_value is None or len(df) < window:
        return None
    hh = float(df["High"].tail(window).max())
    return round(hh - k * atr_value, 2)


# ── Position sizing ───────────────────────────────────────────────────────
def position_size(entry: float, stop: float,
                  account_size: float = DEFAULT_ACCOUNT_SIZE,
                  risk_pct: float = DEFAULT_RISK_PER_TRADE,
                  max_pos_pct: float = MAX_POSITION_PCT) -> dict:
    """Compute suggested share count and position $ value.

    Returns dict with: shares, position_value, position_pct, risk_dollars,
    risk_pct_of_entry.
    """
    out = {
        "shares": 0,
        "position_value": 0.0,
        "position_pct": 0.0,
        "risk_dollars": 0.0,
        "risk_pct_of_entry": 0.0,
    }
    if entry is None or stop is None or entry <= 0 or stop <= 0 or stop >= entry:
        return out

    risk_per_share = entry - stop
    risk_budget = account_size * risk_pct
    shares_by_risk = math.floor(risk_budget / risk_per_share)

    # Cap by maximum position size
    max_position_value = account_size * max_pos_pct
    shares_by_cap = math.floor(max_position_value / entry)

    shares = max(0, min(shares_by_risk, shares_by_cap))
    position_value = shares * entry
    out.update({
        "shares": int(shares),
        "position_value": round(position_value, 2),
        "position_pct": round(position_value / account_size * 100, 2) if account_size > 0 else 0.0,
        "risk_dollars": round(shares * risk_per_share, 2),
        "risk_pct_of_entry": round(risk_per_share / entry * 100, 2),
    })
    return out


# ── Master entry point: compute all stops for one ticker ──────────────────
def compute_stops(df: pd.DataFrame,
                  entry: Optional[float] = None,
                  pivot: Optional[float] = None,
                  sma50: Optional[float] = None,
                  pivot_dist_pct: Optional[float] = None,
                  breakout_today: bool = False,
                  stage2: bool = False,
                  regime: str = "Neutral",
                  account_size: float = DEFAULT_ACCOUNT_SIZE,
                  risk_pct: float = DEFAULT_RISK_PER_TRADE) -> dict:
    """Compute all candidate stops + a suggested final stop + position size."""
    if df is None or df.empty:
        return _empty_stops()

    entry = entry if entry and entry > 0 else _last(df["Close"])
    if entry is None or entry <= 0:
        return _empty_stops()

    # Volatility
    atr_series = atr(df, DEFAULT_ATR_WINDOW)
    atr_val = _last(atr_series)
    atr_p = (atr_val / entry * 100) if atr_val and entry else 0
    # True ADR: average of (daily high-low range / low)
    adr_val = ((df["High"] - df["Low"]) / df["Low"] * 100).rolling(20).mean().iloc[-1]
    
    # DYNAMIC BUFFERS
    # 1. Pivot buffer: Tighten in Bear markets
    p_buffer = 0.05 if regime.upper() in ("BEAR", "BEARISH") else DEFAULT_PIVOT_BUFFER
    
    # 2. SMA buffer: Wide enough to survive ADR, but at least 3%
    s_buffer = max(0.03, adr_val * 0.5 / 100)

    # Compute every candidate
    s_pivot = pivot_stop(pivot, buffer=p_buffer) if pivot else None
    s_atr = atr_stop(entry, atr_val, DEFAULT_ATR_MULT)
    s_sma = sma_stop(sma50, buffer=s_buffer) if sma50 else None
    s_swing = swing_low_stop(df)
    s_chand = chandelier_stop(df, atr_val)

    # Apply ATR cap (never let stop be wider than 3×ATR)
    atr_floor = (entry - ATR_CAP_MULT * atr_val) if atr_val else None
    if atr_floor is not None:
        if s_pivot is not None: s_pivot = max(s_pivot, atr_floor)
        if s_sma   is not None: s_sma   = max(s_sma,   atr_floor)
        if s_swing is not None: s_swing = max(s_swing, atr_floor)

    # ── Pick the best stop based on entry context ─────────────────────
    method, suggested = _pick_best_stop(
        entry=entry,
        breakout_today=breakout_today,
        pivot_dist_pct=pivot_dist_pct,
        stage2=stage2,
        sma50=sma50,
        s_pivot=s_pivot,
        s_atr=s_atr,
        s_sma=s_sma,
        s_swing=s_swing,
    )

    # Position sizing for the suggested stop
    sizing = position_size(entry, suggested, account_size, risk_pct)

    return {
        "atr_14":            round(atr_val, 2) if atr_val else None,
        "atr_pct":           round(atr_p, 2),
        "stop_pivot":        s_pivot,
        "stop_atr":          s_atr,
        "stop_sma50":        s_sma,
        "stop_swing":        s_swing,
        "stop_chandelier":   s_chand,
        "stop_suggested":    suggested,
        "stop_method":       method,
        "risk_pct":          sizing["risk_pct_of_entry"],
        "suggested_shares":  sizing["shares"],
        "position_value":    sizing["position_value"],
        "position_pct":      sizing["position_pct"],
        "risk_dollars":      sizing["risk_dollars"],
    }


def _pick_best_stop(entry: float,
                    breakout_today: bool,
                    pivot_dist_pct: Optional[float],
                    stage2: bool,
                    sma50: Optional[float],
                    s_pivot: Optional[float],
                    s_atr: Optional[float],
                    s_sma: Optional[float],
                    s_swing: Optional[float]) -> tuple[str, Optional[float]]:
    """Return (method_label, stop_price) using a deterministic rule set."""
    # Distance from SMA50 as % of entry
    dist_sma = None
    if sma50 and sma50 > 0:
        dist_sma = (entry / sma50 - 1.0) * 100

    # Rule 1: Fresh breakout → pivot stop
    if breakout_today and s_pivot is not None:
        return "Pivot (breakout)", s_pivot

    # Rule 2: Within 3% of pivot but no breakout yet → pivot stop
    if pivot_dist_pct is not None and -3.0 <= pivot_dist_pct <= 1.0 and s_pivot is not None:
        return "Pivot (at pivot)", s_pivot

    # Rule 3: Stage 2 + price near 50-day MA (within 5%) → SMA-50 stop
    if stage2 and dist_sma is not None and 0 <= dist_sma <= 5.0 and s_sma is not None:
        return "SMA-50", s_sma

    # Rule 4: Stage 2 + price well above 50-day (>8%) → ATR stop
    if stage2 and dist_sma is not None and dist_sma > 8.0 and s_atr is not None:
        return "ATR (2.5×)", s_atr

    # Rule 5: Tight base / sleepy stock → swing low stop if tighter than ATR
    if s_swing is not None and s_atr is not None and s_swing > s_atr:
        return "Swing Low", s_swing

    # Fallback: ATR stop, then any non-null
    for label, val in (("ATR (2.5×)", s_atr), ("SMA-50", s_sma),
                       ("Swing Low", s_swing), ("Pivot", s_pivot)):
        if val is not None:
            return label, val
    return "n/a", None


def _empty_stops() -> dict:
    return {
        "atr_14": None, "atr_pct": None,
        "stop_pivot": None, "stop_atr": None, "stop_sma50": None,
        "stop_swing": None, "stop_chandelier": None,
        "stop_suggested": None, "stop_method": "n/a",
        "risk_pct": 0.0, "suggested_shares": 0,
        "position_value": 0.0, "position_pct": 0.0, "risk_dollars": 0.0,
    }
