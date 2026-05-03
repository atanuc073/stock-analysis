"""Technical indicators & signal scoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
import ta

from config import RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULT


def compute(df: pd.DataFrame) -> dict:
    """Return dict of technical indicators + a 0-100 score."""
    if df.empty or len(df) < 60:
        return {"score": 50.0, "signals": [], "error": "insufficient data"}

    close = df["Close"]
    vol = df["Volume"]

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    macd_line = macd.macd()
    macd_sig = macd.macd_signal()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else close.rolling(len(close) // 2).mean()
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_pct = bb.bollinger_pband()

    last = -1
    price = float(close.iloc[last])
    rsi_v = float(rsi.iloc[last])
    macd_cross_up = bool(macd_line.iloc[last] > macd_sig.iloc[last] and macd_line.iloc[last - 1] <= macd_sig.iloc[last - 1])
    macd_cross_dn = bool(macd_line.iloc[last] < macd_sig.iloc[last] and macd_line.iloc[last - 1] >= macd_sig.iloc[last - 1])
    above_50 = price > float(sma50.iloc[last])
    above_200 = price > float(sma200.iloc[last])
    sma200_val = float(sma200.iloc[last]) if not np.isnan(sma200.iloc[last]) else price
    pct_above_sma200 = (price / sma200_val - 1) * 100 if sma200_val > 0 else 0.0
    golden_cross = bool(sma50.iloc[last] > sma200.iloc[last] and sma50.iloc[last - 1] <= sma200.iloc[last - 1])
    death_cross = bool(sma50.iloc[last] < sma200.iloc[last] and sma50.iloc[last - 1] >= sma200.iloc[last - 1])
    avg_vol_20 = float(vol.tail(20).mean()) if vol.tail(20).mean() else 1.0
    vol_spike = float(vol.iloc[last]) >= VOLUME_SPIKE_MULT * avg_vol_20
    high_52w = float(close.tail(252).max()) if len(close) >= 50 else float(close.max())
    low_52w = float(close.tail(252).min()) if len(close) >= 50 else float(close.min())
    pct_from_52w_high = (price / high_52w - 1) * 100  # negative number; 0 = at high
    pct_from_52w_low = (price / low_52w - 1) * 100    # positive number; 0 = at low

    # Extension / pullback context
    extended_at_high = pct_from_52w_high > -3      # within 3% of 52w high (danger zone)
    in_base = -25 <= pct_from_52w_high <= -8       # 8-25% off high (consolidation / base)
    deep_pullback = -40 <= pct_from_52w_high < -25  # deep pullback (early-stage opportunity)
    far_from_low = pct_from_52w_low > 30           # not in collapse mode

    # 30-day range — are we breaking out at the top, or pulling back inside the range?
    high_30 = float(close.tail(30).max()) if len(close) >= 30 else high_52w
    low_30 = float(close.tail(30).min()) if len(close) >= 30 else low_52w
    pct_from_30d_high = (price / high_30 - 1) * 100
    pulled_back_in_range = pct_from_30d_high <= -3 and far_from_low  # 3%+ off 30d high

    # Scoring
    score = 50.0
    signals = []
    if rsi_v < RSI_OVERSOLD:
        score += 8; signals.append(f"RSI oversold ({rsi_v:.1f})")
    elif rsi_v > RSI_OVERBOUGHT:
        score -= 10; signals.append(f"RSI overbought ({rsi_v:.1f})")
    elif rsi_v >= 60:
        # Late-stage strong: only neutral, not bonus
        score -= 2
    if macd_cross_up:
        score += 10; signals.append("MACD bullish cross")
    if macd_cross_dn:
        score -= 10; signals.append("MACD bearish cross")
    if above_50: score += 4
    if above_200: score += 6
    if golden_cross:
        score += 12; signals.append("Golden cross (50/200)")
    if death_cross:
        score -= 12; signals.append("Death cross (50/200)")
    if vol_spike:
        score += 5; signals.append(f"Volume spike {vol.iloc[last]/avg_vol_20:.1f}x")

    # ── Extension / Base scoring (FLIPPED from original) ─────────────────
    # The old code rewarded "near 52w high" — that's how we bought tops.
    # Now: penalize chasing tops, reward stocks consolidating in a base.
    if extended_at_high:
        score -= 12; signals.append(f"⚠️ Extended {pct_from_52w_high:+.1f}% from 52w high")
    elif in_base and far_from_low:
        score += 10; signals.append(f"In base ({pct_from_52w_high:.1f}% off high)")
    elif deep_pullback and above_200:
        score += 6; signals.append(f"Deep pullback in uptrend ({pct_from_52w_high:.1f}%)")

    if pulled_back_in_range and above_50:
        score += 4; signals.append("Pulled back in 30d range (better entry)")

    if 0 <= bb_pct.iloc[last] <= 0.05:
        score += 4; signals.append("At lower Bollinger band")
    if bb_pct.iloc[last] >= 0.95:
        score -= 6; signals.append("At upper Bollinger band (extended)")

    score = float(np.clip(score, 0, 100))

    return {
        "score": score,
        "signals": signals,
        "price": price,
        "rsi": rsi_v,
        "above_sma50": above_50,
        "above_sma200": above_200,
        "pct_above_sma200": pct_above_sma200,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_52w_high": pct_from_52w_high,
        "pct_from_52w_low": pct_from_52w_low,
        "in_base": bool(in_base),
        "extended_at_high": bool(extended_at_high),
        "volume_ratio": float(vol.iloc[last] / avg_vol_20) if avg_vol_20 else 1.0,
    }
