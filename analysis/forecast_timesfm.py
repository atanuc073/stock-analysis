"""TimesFM forecaster — Google's foundation model for time series.

Optional dependency. Falls back to Prophet/linear if not installed or model
download fails. Model is downloaded once from HuggingFace and cached locally.

Install (on personal machine with good network):
    pip install -r requirements-timesfm.txt

Usage (set in .env):
    FORECASTER=timesfm
"""
from __future__ import annotations
import logging
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Singleton model — TimesFM is ~500MB, loading is expensive
_MODEL = None
_MODEL_LOCK = Lock()
_LOAD_FAILED = False


def _get_model():
    """Lazy-load TimesFM model. Thread-safe singleton."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if _LOAD_FAILED:
            return None
        try:
            import timesfm  # type: ignore
            log.info("Loading TimesFM 2.0 model (first time may download ~800MB) ...")
            backend = _detect_backend()
            log.info("TimesFM backend: %s", backend)
            _MODEL = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=32,
                    horizon_len=128,
                    num_layers=50,
                    use_positional_embedding=False,
                    context_len=2048,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
                ),
            )
            log.info("TimesFM loaded successfully.")
            return _MODEL
        except ImportError:
            log.warning("timesfm package not installed. "
                        "Run: pip install -r requirements-timesfm.txt")
            _LOAD_FAILED = True
            return None
        except Exception as e:
            log.warning("TimesFM load failed: %s", e)
            _LOAD_FAILED = True
            return None


def _detect_backend() -> str:
    """Return 'gpu' if CUDA available, else 'cpu'."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass
    return "cpu"


def is_available() -> bool:
    """Cheap check — does not load the model."""
    try:
        import timesfm  # noqa: F401
        return True
    except ImportError:
        return False


def compute(df: pd.DataFrame, horizon_days: int = 21) -> dict:
    """Generate a forecast and score using TimesFM.

    Returns the same dict shape as analysis.forecast.compute() so it's
    drop-in compatible with the composite scorer.

    On failure (model unavailable, error), returns a neutral score so the
    caller can fall back gracefully.
    """
    if df.empty or len(df) < 60:
        return {"score": 50.0, "signals": [], "expected_return_pct": None,
                "model": "timesfm", "error": "insufficient history"}

    model = _get_model()
    if model is None:
        return {"score": 50.0, "signals": [], "expected_return_pct": None,
                "model": "timesfm", "error": "model unavailable"}

    try:
        # Use up to 512 days of history for context
        close = df["Close"].tail(512).values.astype(np.float32)
        cur = float(close[-1])

        # TimesFM expects list-of-arrays input
        forecast_input = [close]
        frequency_input = [0]  # 0 = high-freq (daily)

        point_fc, quantile_fc = model.forecast(
            forecast_input, freq=frequency_input
        )
        # point_fc shape: (1, horizon_len)
        horizon = min(horizon_days, point_fc.shape[1])
        target = float(point_fc[0, horizon - 1])
        path = point_fc[0, :horizon]

        # Quantile forecast (1, horizon_len, num_quantiles); use 10th & 90th
        # quantiles config is [0.1, 0.2, ..., 0.9] by default — index 0 = 10%, 8 = 90%
        try:
            lo = float(quantile_fc[0, horizon - 1, 0])
            hi = float(quantile_fc[0, horizon - 1, 8])
        except (IndexError, TypeError):
            lo, hi = target * 0.95, target * 1.05

        expected_pct = (target / cur - 1) * 100
        # Score: linear in expected return, capped at ±25 → 0..100 range
        score = float(np.clip(50 + expected_pct * 2, 0, 100))

        signals = []
        if expected_pct > 5:
            signals.append(f"TimesFM +{expected_pct:.1f}% / {horizon}d")
        elif expected_pct < -5:
            signals.append(f"TimesFM {expected_pct:.1f}% / {horizon}d")

        # Forecast monotonicity flag — strong directional signal
        if len(path) >= 5:
            up_days = int((np.diff(path) > 0).sum())
            if up_days >= len(path) * 0.75:
                signals.append("Strong uptrend forecast")
            elif up_days <= len(path) * 0.25:
                signals.append("Strong downtrend forecast")

        return {
            "score": score,
            "signals": signals,
            "expected_return_pct": expected_pct,
            "forecast_price": target,
            "lower": lo,
            "upper": hi,
            "horizon_days": horizon,
            "model": "timesfm",
        }
    except Exception as e:
        log.warning("TimesFM compute failed: %s", e)
        return {"score": 50.0, "signals": [], "expected_return_pct": None,
                "model": "timesfm", "error": str(e)}


def warmup() -> bool:
    """Optional: pre-load the model. Call once at startup to avoid first-call latency."""
    return _get_model() is not None
