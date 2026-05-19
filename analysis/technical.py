"""Technical indicators & signal scoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
import ta

from config import RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_MULT
from analysis.indicators import atr


def _compute_accumulation_score(df: pd.DataFrame, ud_ratio=None, net_acc=None, cmf_v=None) -> tuple[float, list[str], dict]:
    """
    Calculate an 'Institutional Buy Probability' score (0-100).
    Based on:
    1. U/D Volume Ratio (50-day) - Marks Minervini's metric for institutional appetite.
    2. Chaikin Money Flow (CMF) - Institutional presence over 20 days.
    3. Accumulation/Distribution Day Count (25-day) - Net volume-supported moves.
    """
    if len(df) < 50:
        return 50.0, [], {}

    if ud_ratio is None or net_acc is None or cmf_v is None:
        close = df["Close"]
        vol = df["Volume"]
        
        # 1. U/D Volume Ratio (50-day)
        # Ratio of volume on 'up' days vs 'down' days. > 1.0 is accumulation.
        up_mask = close > close.shift(1)
        dn_mask = close < close.shift(1)
        up_vol = vol.where(up_mask).tail(50).sum()
        dn_vol = vol.where(dn_mask).tail(50).sum()
        ud_ratio = up_vol / dn_vol if dn_vol > 0 else 1.0
        
        # 2. Accumulation Days vs Distribution Days (25-day)
        # Accumulation: Price UP + Volume > Prior Day
        # Distribution: Price DOWN + Volume > Prior Day
        vol_up = vol > vol.shift(1)
        acc_days = (up_mask & vol_up).tail(25).sum()
        dist_days = (dn_mask & vol_up).tail(25).sum()
        net_acc = int(acc_days - dist_days)
        
        # 3. Chaikin Money Flow
        cmf_indicator = ta.volume.ChaikinMoneyFlowIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=20
        )
        cmf_v = float(cmf_indicator.chaikin_money_flow().iloc[-1])

    score = 50.0
    signals = []
    
    # U/D Ratio Scoring
    if ud_ratio > 1.5:
        score += 15; signals.append(f"Strong U/D ratio ({ud_ratio:.1f}) - heavy institutional buying")
    elif ud_ratio > 1.2:
        score += 8; signals.append(f"Healthy U/D ratio ({ud_ratio:.1f})")
    elif ud_ratio < 0.7:
        score -= 10; signals.append(f"Weak U/D ratio ({ud_ratio:.1f}) - possible distribution")

    # Net Accumulation Days Scoring
    if net_acc >= 4:
        score += 10; signals.append(f"Institutional accumulation (+{net_acc} net days)")
    elif net_acc <= -4:
        score -= 12; signals.append(f"⚠️ Institutional distribution ({net_acc} net days)")

    # CMF Scoring
    if cmf_v > 0.15:
        score += 8; signals.append(f"Positive Money Flow ({cmf_v:.2f})")
    elif cmf_v < -0.1:
        score -= 8; signals.append(f"Negative Money Flow ({cmf_v:.2f})")

    score = float(np.clip(score, 0, 100))
    metrics = {"ud_ratio": ud_ratio, "cmf": cmf_v, "net_acc": net_acc}
    return score, signals, metrics


def compute(df: pd.DataFrame) -> dict:
    """Return dict of technical indicators + a 0-100 score."""
    if df.empty or len(df) < 60:
        return {"score": 50.0, "signals": [], "error": "insufficient data"}

    close = df["Close"]
    vol = df["Volume"]

    # Fast cached lookup for backtesting to avoid massive ta library overhead
    if hasattr(df, "_parent_hd") and hasattr(df, "_asof"):
        hd = df._parent_hd
        asof = df._asof
        cache = hd._precomputed_series
        
        # Align timezone of asof to match cache series index
        asof_tz = asof
        if cache["rsi"].index.tz is not None and asof.tzinfo is None:
            try:
                asof_tz = asof.tz_localize(cache["rsi"].index.tz)
            except Exception:
                asof_tz = asof.tz_localize("UTC").tz_convert(cache["rsi"].index.tz)
        elif cache["rsi"].index.tz is None and asof.tzinfo is not None:
            asof_tz = asof.tz_localize(None)
            
        rsi = cache["rsi"].loc[:asof_tz]
        macd_line = cache["macd_line"].loc[:asof_tz]
        macd_sig = cache["macd_sig"].loc[:asof_tz]
        sma20 = cache["sma20"].loc[:asof_tz]
        sma50 = cache["sma50"].loc[:asof_tz]
        sma200 = cache["sma200"].loc[:asof_tz]
        bb_pct = cache["bb_pct"].loc[:asof_tz]
        
        ud_ratio_s = cache["ud_ratio"].loc[:asof_tz]
        net_acc_s = cache["net_acc"].loc[:asof_tz]
        cmf_s = cache["cmf"].loc[:asof_tz]
        atr_s = cache["atr"].loc[:asof_tz]
        
        ud_ratio = float(ud_ratio_s.iloc[-1]) if not pd.isna(ud_ratio_s.iloc[-1]) else 1.0
        net_acc = int(net_acc_s.iloc[-1]) if not pd.isna(net_acc_s.iloc[-1]) else 0
        cmf_v = float(cmf_s.iloc[-1]) if not pd.isna(cmf_s.iloc[-1]) else 0.0
        
        inst_score, inst_signals, inst_metrics = _compute_accumulation_score(df, ud_ratio=ud_ratio, net_acc=net_acc, cmf_v=cmf_v)
        atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
    else:
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd = ta.trend.MACD(close)
        macd_line = macd.macd()
        macd_sig = macd.macd_signal()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean() if len(close) >= 200 else close.rolling(len(close) // 2).mean()
        bb = ta.volatility.BollingerBands(close, window=20)
        bb_pct = bb.bollinger_pband()
        
        inst_score, inst_signals, inst_metrics = _compute_accumulation_score(df)
        atr_val = float(atr(df)) if len(df) >= 14 else 0.0

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

    # 30-day range
    high_30 = float(close.tail(30).max()) if len(close) >= 30 else high_52w
    low_30 = float(close.tail(30).min()) if len(close) >= 30 else low_52w
    pct_from_30d_high = (price / high_30 - 1) * 100
    pulled_back_in_range = pct_from_30d_high <= -3 and far_from_low  # 3%+ off 30d high

    # Institutional Buy Probability Pass
    inst_score, inst_signals, inst_metrics = _compute_accumulation_score(df)

    # Scoring
    score = 50.0
    signals = []
    
    # 1. Technical Indicators
    if rsi_v < RSI_OVERSOLD:
        score += 8; signals.append(f"RSI oversold ({rsi_v:.1f})")
    elif rsi_v > RSI_OVERBOUGHT:
        score -= 10; signals.append(f"RSI overbought ({rsi_v:.1f})")
    elif rsi_v >= 60:
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
    
    # 2. Institutional Buy Probability (Weighted into technical)
    # Contribution is roughly 25% of the technical sub-score
    score += (inst_score - 50) * 0.4
    signals.extend(inst_signals)

    # 3. Short-term extension (SMA20)
    sma20_val = float(sma20.iloc[last]) if not np.isnan(sma20.iloc[last]) else price
    ext_20 = (price / sma20_val - 1) * 100 if sma20_val > 0 else 0.0
    if ext_20 > 20:
        score -= 20; signals.append(f"⚠️ Extreme extension ({ext_20:.1f}% > SMA20)")
    elif ext_20 > 15:
        score -= 12; signals.append(f"⚠️ Parabolic ({ext_20:.1f}% > SMA20)")
    elif ext_20 > 10:
        score -= 5; signals.append(f"Extended ({ext_20:.1f}% > SMA20)")

    # 4. Extension / Base scoring
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
        "pct_from_sma20": ext_20,
        "in_base": bool(in_base),
        "extended_at_high": bool(extended_at_high),
        "volume_ratio": float(vol.iloc[last] / avg_vol_20) if avg_vol_20 else 1.0,
        "atr": atr_val,
        "dist_200": pct_above_sma200,
        "adr_20": float(((df["High"] - df["Low"]) / df["Low"] * 100).tail(20).mean()),
        "inst_metrics": inst_metrics,
        "inst_score": inst_score,
    }

