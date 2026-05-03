"""Lightweight ML forecast — uses linear regression on recent trend (Prophet optional).

Prophet is heavy and slow; we default to a simple linear trend forecast for breadth.
For deeper analysis on top picks, set USE_PROPHET=True.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

USE_PROPHET = False  # toggle for deep mode on top picks


def compute(df: pd.DataFrame, horizon_days: int = 21) -> dict:
    if df.empty or len(df) < 60:
        return {"score": 50.0, "signals": [], "expected_return_pct": None}

    close = df["Close"].tail(120).values
    X = np.arange(len(close)).reshape(-1, 1)
    y = np.log(close)
    model = LinearRegression().fit(X, y)
    future_x = np.array([[len(close) + horizon_days - 1]])
    pred = float(np.exp(model.predict(future_x)[0]))
    cur = float(close[-1])
    expected_pct = (pred / cur - 1) * 100
    # Stability: residual std
    resid_std = float(np.std(y - model.predict(X))) * 100

    score = float(np.clip(50 + expected_pct * 2, 0, 100))
    signals = []
    if expected_pct > 5:
        signals.append(f"Trend forecast +{expected_pct:.1f}% / {horizon_days}d")
    elif expected_pct < -5:
        signals.append(f"Trend forecast {expected_pct:.1f}% / {horizon_days}d")

    return {
        "score": score, "signals": signals,
        "expected_return_pct": expected_pct,
        "forecast_price": pred,
        "trend_stability": resid_std,
        "horizon_days": horizon_days,
    }


def prophet_forecast(df: pd.DataFrame, horizon_days: int = 30) -> dict:
    """Heavier Prophet forecast — only call for top picks."""
    try:
        from prophet import Prophet  # noqa
    except ImportError:
        return {"error": "prophet not installed"}
    try:
        from prophet import Prophet
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
        return {
            "target_price": target, "lower": lo, "upper": hi,
            "expected_return_pct": (target / last_price - 1) * 100,
            "horizon_days": horizon_days,
        }
    except Exception as e:
        return {"error": str(e)}
