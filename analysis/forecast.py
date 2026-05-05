"""Forecast dispatcher — routes to linear / prophet / timesfm based on FORECASTER config.

Strategy pattern: each forecaster returns the same dict shape so the composite
scorer is agnostic. Falls back gracefully if the chosen model is unavailable.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import FORECASTER

log = logging.getLogger(__name__)

USE_PROPHET = False  # legacy toggle — prefer FORECASTER env var


def compute(df: pd.DataFrame, horizon_days: int = 21) -> dict:
    """Dispatch to configured forecaster, with safe fallback to linear trend."""
    if FORECASTER == "timesfm":
        try:
            from analysis import forecast_timesfm
            result = forecast_timesfm.compute(df, horizon_days=horizon_days)
            # If TimesFM unavailable/failed, fall through to prophet/linear
            if "error" not in result or result.get("score", 50.0) != 50.0:
                return result
            log.debug("TimesFM unavailable, falling back to prophet")
        except Exception as e:
            log.warning("TimesFM dispatch error: %s — falling back to prophet", e)
        try:
            return _prophet_compute(df, horizon_days)
        except Exception as e:
            log.warning("Prophet failed: %s — falling back to linear", e)
    elif FORECASTER == "prophet":
        try:
            return _prophet_compute(df, horizon_days)
        except Exception as e:
            log.warning("Prophet failed: %s — falling back to linear", e)
    return _linear_compute(df, horizon_days)


def _linear_compute(df: pd.DataFrame, horizon_days: int = 21) -> dict:
    """Lightweight linear-trend forecast on log prices (default, fast)."""
    if df.empty or len(df) < 60:
        return {"score": 50.0, "signals": [], "expected_return_pct": None, "model": "linear"}

    close = df["Close"].tail(120).values
    X = np.arange(len(close)).reshape(-1, 1)
    y = np.log(close)
    model = LinearRegression().fit(X, y)
    future_x = np.array([[len(close) + horizon_days - 1]])
    pred = float(np.exp(model.predict(future_x)[0]))
    cur = float(close[-1])
    expected_pct = (pred / cur - 1) * 100
    resid_std = float(np.std(y - model.predict(X))) * 100

    score = float(np.clip(50 + expected_pct * 2, 0, 100))
    signals = []
    if expected_pct > 5:
        signals.append(f"Trend +{expected_pct:.1f}% / {horizon_days}d")
    elif expected_pct < -5:
        signals.append(f"Trend {expected_pct:.1f}% / {horizon_days}d")

    return {
        "score": score, "signals": signals,
        "expected_return_pct": expected_pct,
        "forecast_price": pred,
        "trend_stability": resid_std,
        "horizon_days": horizon_days,
        "model": "linear",
    }


def _prophet_compute(df: pd.DataFrame, horizon_days: int = 30) -> dict:
    """Prophet forecast — heavier, more accurate on seasonal series."""
    from prophet import Prophet  # type: ignore
    if df.empty or len(df) < 60:
        return {"score": 50.0, "signals": [], "expected_return_pct": None, "model": "prophet"}

    d = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    d["ds"] = pd.to_datetime(d["ds"]).dt.tz_localize(None)
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(d)
    future = m.make_future_dataframe(periods=horizon_days)
    fcst = m.predict(future).tail(horizon_days)
    last_price = float(d["y"].iloc[-1])
    target = float(fcst["yhat"].iloc[-1])
    lo = float(fcst["yhat_lower"].iloc[-1])
    hi = float(fcst["yhat_upper"].iloc[-1])
    expected_pct = (target / last_price - 1) * 100
    score = float(np.clip(50 + expected_pct * 2, 0, 100))

    signals = []
    if expected_pct > 5:
        signals.append(f"Prophet +{expected_pct:.1f}% / {horizon_days}d")
    elif expected_pct < -5:
        signals.append(f"Prophet {expected_pct:.1f}% / {horizon_days}d")

    return {
        "score": score, "signals": signals,
        "expected_return_pct": expected_pct,
        "forecast_price": target, "lower": lo, "upper": hi,
        "horizon_days": horizon_days,
        "model": "prophet",
    }


def prophet_forecast(df: pd.DataFrame, horizon_days: int = 30) -> dict:
    """Backward-compat wrapper for the old prophet_forecast() callers."""
    try:
        return _prophet_compute(df, horizon_days)
    except ImportError:
        return {"error": "prophet not installed"}
    except Exception as e:
        return {"error": str(e)}
