"""12-feature sub-decomposition scorer.

This module is the canonical single source of truth for the sub-decomposition
factor model that replaced the 5-composite scheme in May 2026. It is used by:

  * ``analysis/composite.py``  : live ``analyze_batch()`` pipeline (daily report)
  * ``backtest/engine.py``     : historical backtester (cross-sectional sort)
  * ``backtest/factor_regression.py`` : walk-forward optimizer that *derived*
    the weights baked into ``SUB_SCORE_WEIGHTS`` below.

Why sub-decomposition?
----------------------
The 5 composite scores produced by analysis/technical.py, momentum.py, etc.
each bundle several orthogonal signals into one 0-100 number with hand-coded
weights. Walk-forward optimization showed that breaking them into 12 narrower
features lets the simplex-constrained optimizer find a far better OOS weight
vector:

  IC (test, mean across 34 monthly folds, nifty500 2023-01..2025-10):
      5-composite:   +0.0091   (50% positive folds)
      12-sub      :  +0.0364   (74% positive folds)   ← 4x improvement
  IC information ratio (mean / std):
      5-composite:    0.074
      12-sub      :   0.284   ← 4x improvement, with std essentially unchanged
  Top-quintile basket return (OOS, monthly):
      5-composite:   +35.5%
      12-sub      :  +42.8%

The 12 sub-features and their *direction-of-better* (higher feature = better
expected forward return after sign normalization):

  Technical (3):
    tech_trend       : pct_above_sma200          (+) longer-term uptrend strength
    tech_extension   : pct_from_sma20            (-) less extended = better (mean-reversion guard)
    tech_volume      : inst_score                (+) institutional accumulation proxy

  Momentum (3):
    mom_12_1         : Jegadeesh-Titman 12-1 momentum (+) the classic factor
    mom_3m           : 3-month return                  (+) intermediate momentum
    mom_rs           : composite relative strength     (+) cross-sectional momentum

  Quality (3):
    qual_gpa         : Novy-Marx gross profitability   (+)
    qual_fcf         : FCF yield                       (+)
    qual_roa         : return on assets                (+)

  Fundamental (2):
    fund_value       : P/E                             (-) lower P/E = better
    fund_growth      : EPS growth                      (+)

  Earnings (1):
    earnings_drift   : PEAD composite                  (+) keep as one signal

Cross-sectional rank transformation
-----------------------------------
Because the raw features live on heterogeneous scales (P/E in 0..∞, momentum
in -1..+∞, GPA in 0..1), we rank-transform per universe-cohort to 0-100
percentiles before applying the weights. Missing values get the median rank
(50) so they are neutral, not extreme. This makes the simplex-constrained
(``w >= 0, sum(w) = 1``) optimizer's weights meaningful and bounded.

After scoring, the final ``sub_decomp_score`` is also on 0-100 (a weighted
average of 0-100 ranks), but its distribution is tighter than the old
composite — most stocks cluster between 40-60. The relative *ordering* is
what matters for ranking and basket construction; if you previously used a
hard absolute threshold like ``score >= 70`` you should re-calibrate it
against the new distribution (or use a percentile-based filter instead).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Path to the WFO out-of-sample weights schedule. Each row is one month with
# weights trained on the preceding 36-month window, so using row R's weights
# at any date >= row R's test_start is honest out-of-sample evaluation.
WFO_SCHEDULE_PATH = Path(__file__).resolve().parents[1] / "reports" / "regression" / "sub" / "walk_forward_regression.csv"

# ── Feature definitions (canonical order) ─────────────────────────────────────
SUB_FACTORS: list[str] = [
    "tech_trend", "tech_extension", "tech_volume",
    "mom_12_1", "mom_3m", "mom_rs",
    "qual_gpa", "qual_fcf", "qual_roa",
    "fund_value", "fund_growth",
    "earnings_drift",
]

# Extractor: (lambda obj -> raw_value_or_None, direction_sign)
# `obj` must expose .technical, .momentum, .quality, .fundamental, .earnings_drift
# as dicts. Both ``analysis.composite.StockReport`` and
# ``backtest.scoring.BacktestScore`` satisfy this.
SUB_EXTRACTORS: dict[str, tuple[Callable[[Any], Any], int]] = {
    # technical
    "tech_trend":     (lambda s: (s.technical or {}).get("pct_above_sma200"), +1),
    "tech_extension": (lambda s: (s.technical or {}).get("pct_from_sma20"),   -1),  # less extension = better
    "tech_volume":    (lambda s: (s.technical or {}).get("inst_score"),       +1),
    # momentum
    "mom_12_1":       (lambda s: (s.momentum  or {}).get("mom_12_1"),         +1),
    "mom_3m":         (lambda s: (s.momentum  or {}).get("ret_3m"),           +1),
    "mom_rs":         (lambda s: (s.momentum  or {}).get("rs_value"),         +1),
    # quality
    "qual_gpa":       (lambda s: (s.quality   or {}).get("gpa"),              +1),
    "qual_fcf":       (lambda s: (s.quality   or {}).get("fcf_yield"),        +1),
    "qual_roa":       (lambda s: (s.quality   or {}).get("roa"),              +1),
    # fundamental
    "fund_value":     (lambda s: (s.fundamental or {}).get("pe"),             -1),  # lower P/E = better
    "fund_growth":    (lambda s: (s.fundamental or {}).get("eps_growth"),     +1),
    # earnings drift
    "earnings_drift": (lambda s: (s.earnings_drift or {}).get("score"),       +1),
}

# ── Default weights — derived from 34-fold walk-forward optimization ──────────
# Source: reports/regression/sub/walk_forward_regression.csv
# Window: Nifty 500, 2023-01..2025-10, monthly rebalance, 36-month lookback,
# Bayesian-GP optimizer with 60 calls per fold.
# Weights below are the mean across all 34 OOS folds (already sum to 1.0).
SUB_SCORE_WEIGHTS: dict[str, float] = {
    "tech_extension":  0.2257,
    "mom_rs":          0.1746,
    "mom_12_1":        0.1360,
    "qual_fcf":        0.0819,
    "earnings_drift":  0.0808,
    "fund_growth":     0.0695,
    "qual_gpa":        0.0601,
    "qual_roa":        0.0567,
    "mom_3m":          0.0379,
    "fund_value":      0.0310,
    "tech_trend":      0.0306,
    "tech_volume":     0.0153,
}
assert abs(sum(SUB_SCORE_WEIGHTS.values()) - 1.0) < 1e-3, (
    f"SUB_SCORE_WEIGHTS must sum to ~1.0, got {sum(SUB_SCORE_WEIGHTS.values())}"
)
assert set(SUB_SCORE_WEIGHTS) == set(SUB_FACTORS), (
    f"SUB_SCORE_WEIGHTS keys mismatch SUB_FACTORS: "
    f"missing={set(SUB_FACTORS)-set(SUB_SCORE_WEIGHTS)}, "
    f"extra={set(SUB_SCORE_WEIGHTS)-set(SUB_FACTORS)}"
)


# ── Public API ────────────────────────────────────────────────────────────────
def extract_raw(obj: Any) -> dict[str, float]:
    """Pull the 12 sign-normalized raw values from a Score-like object.

    Returns a dict {feature_name: signed_value_or_nan}. Sign normalization
    means higher-is-better for every feature after this call.
    """
    out: dict[str, float] = {}
    for name, (fn, direction) in SUB_EXTRACTORS.items():
        try:
            v = fn(obj)
            v = float(v) if v is not None else float("nan")
        except Exception:
            v = float("nan")
        out[name] = direction * v if np.isfinite(v) else float("nan")
    return out


def rank_transform(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank-transform → 0-100, NaN → 50.

    Operates on a wide DataFrame where rows are symbols and columns are
    feature names. The transform is applied independently per column.
    """
    out = pd.DataFrame(index=df_raw.index, columns=df_raw.columns, dtype=float)
    for col in df_raw.columns:
        ranked = df_raw[col].rank(pct=True, na_option="keep") * 100.0
        out[col] = ranked.fillna(50.0)
    return out


def apply_sub_decomp(
    reports: Iterable[Any],
    weights: dict[str, float] | None = None,
    *,
    write_attr: str = "adjusted_score",
    detail_attr: str = "sub_decomp",
) -> None:
    """Compute the sub-decomposition composite for a batch of reports.

    For every report, the function:
      1. extracts 12 raw sub-features (sign-normalized so higher = better),
      2. cross-sectionally rank-transforms them to 0-100 percentiles within
         the batch (NaN → 50, the neutral median),
      3. computes ``score = sum(weight[f] * rank[f] for f in SUB_FACTORS)``,
      4. writes the result onto each report at ``write_attr`` (default
         ``adjusted_score``, which is what the daily report renderer and the
         backtest engine already read for ranking), and
      5. stashes the per-feature ranks at ``detail_attr`` for transparency.

    The original ``adjusted_score`` (if any) is preserved at
    ``r._adjusted_score_pre_sub`` so the prior cross-sectional pass remains
    inspectable.

    Idempotent and safe to call on heterogeneous batches (US + IN mixed) —
    ranking is done across the full batch, which mirrors how the WFO trained
    the weights.
    """
    reports = [r for r in reports]  # materialize iterator
    if not reports:
        return
    w = dict(weights) if weights is not None else dict(SUB_SCORE_WEIGHTS)

    # 1) Extract raw signed values into a wide DataFrame (rows=symbols).
    symbols: list[str] = []
    raw_rows: list[dict[str, float]] = []
    for r in reports:
        sym = getattr(r, "symbol", None) or str(id(r))
        symbols.append(sym)
        raw_rows.append(extract_raw(r))
    df_raw = pd.DataFrame(raw_rows, index=symbols, columns=SUB_FACTORS)

    # 2) Per-feature cross-sectional rank → 0-100 (NaN → 50).
    df_rank = rank_transform(df_raw)

    # 3) Weighted sum across the 12 columns. Use only weights for features
    #    that survive (defensive in case caller passes a subset).
    active = [f for f in SUB_FACTORS if w.get(f, 0.0) > 0.0]
    w_arr = np.array([w[f] for f in active], dtype=float)
    w_arr = w_arr / w_arr.sum()  # renormalize defensively
    scores = (df_rank[active].to_numpy() * w_arr).sum(axis=1)

    # 4 + 5) Write back to each report.
    for r, sym, sc in zip(reports, symbols, scores):
        prior = getattr(r, write_attr, None)
        try:
            setattr(r, "_adjusted_score_pre_sub", float(prior) if prior is not None else None)
        except Exception:
            pass
        try:
            setattr(r, write_attr, round(float(sc), 2))
        except Exception:
            log.warning("sub_decomp: failed to write %s on %s", write_attr, sym)
        try:
            setattr(r, detail_attr, {
                "score": round(float(sc), 2),
                "features": {f: round(float(df_rank.at[sym, f]), 2) for f in SUB_FACTORS},
                "weights_used": {f: round(float(w.get(f, 0.0)), 4) for f in SUB_FACTORS},
            })
        except Exception:
            pass

    if log.isEnabledFor(logging.DEBUG):
        coverage = (~df_raw.isna()).mean() * 100.0
        log.debug("sub_decomp: applied to %d symbols; per-feature coverage:\n%s",
                  len(reports), coverage.round(1).to_string())


# ── WFO weight schedule (dynamic, date-aware weights) ──────────────────────────────
# These helpers replace the single static SUB_SCORE_WEIGHTS vector with a
# month-indexed schedule so that:
#   * The historical backtester uses each rebalance month's own out-of-sample
#     weights (no look-ahead bias: row R was trained on data BEFORE R).
#   * The live screener uses the most recent fold's weights, automatically
#     picking up new fits whenever the WFO is re-run.

_SCHEDULE_CACHE: pd.DataFrame | None = None
_SCHEDULE_MTIME: float = -1.0


def load_wfo_weight_schedule(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load the WFO out-of-sample weight schedule from CSV.

    Returns a DataFrame indexed by month-start ``pd.Timestamp`` with one
    column per ``SUB_FACTORS`` entry, or ``None`` if the file is missing.
    Cached and auto-invalidated on file mtime change so re-runs of the WFO
    are picked up without restarting the process.
    """
    global _SCHEDULE_CACHE, _SCHEDULE_MTIME
    p = Path(path) if path is not None else WFO_SCHEDULE_PATH
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    if _SCHEDULE_CACHE is not None and mtime == _SCHEDULE_MTIME:
        return _SCHEDULE_CACHE
    df = pd.read_csv(p)
    if "test_start" not in df.columns:
        log.warning("WFO schedule %s has no 'test_start' column; ignoring.", p)
        return None
    # Convert 'YYYY-MM' (or 'YYYY-MM-DD') to month-start Timestamp.
    df["_month"] = pd.to_datetime(df["test_start"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["_month"]).sort_values("_month").set_index("_month")
    # Keep only weight columns we actually have extractors for.
    keep = [f"w_{f}" for f in SUB_FACTORS if f"w_{f}" in df.columns]
    if len(keep) != len(SUB_FACTORS):
        missing = [f for f in SUB_FACTORS if f"w_{f}" not in df.columns]
        log.warning("WFO schedule missing weight columns for: %s. Falling back to baked defaults for those.", missing)
    wdf = df[keep].copy()
    wdf.columns = [c[2:] for c in wdf.columns]  # strip 'w_' prefix
    _SCHEDULE_CACHE = wdf
    _SCHEDULE_MTIME = mtime
    log.info("Loaded WFO weight schedule: %d folds from %s to %s (%s)",
             len(wdf), wdf.index.min().strftime("%Y-%m"), wdf.index.max().strftime("%Y-%m"), p)
    return wdf


# ── Live-mode blended-weights configuration ─────────────────────────────────
# When ``asof`` is None (live trading), a single fold's weights are too noisy
# (the latest fold can be ~0.40 L1 away from the long-run mean — see the
# May-2026 stability audit). We use a shrinkage blend instead:
#
#   w_live = α·mean(last N folds) + β·mean(all folds) + γ·uniform(1/K)
#
# Rationale per term:
#   * α·recent  : tracks genuine regime drift (most recent OOS evidence)
#   * β·long-run: anchors to long-run signal, dampens fold-to-fold noise
#   * γ·uniform : James-Stein-style insurance against single-fold spikes
#
# Empirically (34-fold Bayesian WFO, 2023-01..2025-10):
#   * latest_fold L1 vs mean_all = 0.40 (high variance, prone to overfit)
#   * blend       L1 vs mean_all = 0.18 (sensible regime tilt, stable core)
LIVE_BLEND_RECENT_N: int = 6      # how many recent folds to average
LIVE_BLEND_ALPHA: float = 0.50    # weight on recent-fold mean
LIVE_BLEND_BETA: float = 0.30     # weight on all-fold mean
LIVE_BLEND_GAMMA: float = 0.20    # weight on uniform 1/K shrinkage prior


def _blend_live_weights(
    schedule: pd.DataFrame,
    fallback: dict[str, float],
    *,
    recent_n: int = LIVE_BLEND_RECENT_N,
    alpha: float = LIVE_BLEND_ALPHA,
    beta: float = LIVE_BLEND_BETA,
    gamma: float = LIVE_BLEND_GAMMA,
) -> dict[str, float]:
    """Compute the shrinkage-blended live weights from the WFO schedule.

    Pure helper: takes a non-empty schedule + fallback dict, returns a
    normalized weight dict keyed by ``SUB_FACTORS``.
    """
    cols = [f for f in SUB_FACTORS if f in schedule.columns]
    if not cols:
        return dict(fallback)
    # Row-normalize each fold so missing/extra mass doesn't bias the average.
    sched = schedule[cols].copy()
    row_sum = sched.sum(axis=1).replace(0.0, pd.NA)
    sched = sched.div(row_sum, axis=0).dropna(how="all")
    if sched.empty:
        return dict(fallback)

    recent_mean = sched.tail(max(recent_n, 1)).mean()
    all_mean = sched.mean()
    k = len(SUB_FACTORS)
    uniform = pd.Series(1.0 / k, index=cols)

    blended = alpha * recent_mean + beta * all_mean + gamma * uniform
    out = dict(fallback)  # carry baked defaults for features absent from CSV
    for f in cols:
        v = blended.get(f)
        if v is not None and pd.notna(v):
            out[f] = float(v)
    total = sum(out.values())
    if total > 0:
        out = {k_: v_ / total for k_, v_ in out.items()}
    return out


def get_weights_for_date(
    asof: pd.Timestamp | str | None = None,
    *,
    schedule: pd.DataFrame | None = None,
    fallback: dict[str, float] | None = None,
    live_blend: bool = True,
) -> dict[str, float]:
    """Look up the WFO weights to use at ``asof`` date.

    Default behaviour (``live_blend=True``) returns the shrinkage blend so
    that **the backtest path is identical to the live path** — at any
    historical date D the engine sees the same blending formula it will see
    in production. Look-ahead safety is preserved by slicing the schedule
    to ``[:asof]`` BEFORE blending so only folds whose ``test_start <=
    asof`` participate.

    Resolution rule:
      * ``asof=None`` (LIVE):
          - ``live_blend=True``  → blend over the full schedule.
          - ``live_blend=False`` → single most recent fold (legacy / A/B).
      * ``asof=<date>`` (BACKTEST, look-ahead safe):
          - ``live_blend=True``  → blend over folds with month <= asof.
          - ``live_blend=False`` → single most recent eligible fold (legacy).
          - If no fold is eligible (asof predates all folds) return ``fallback``.
      * No schedule available at all → ``fallback``.

    ``fallback`` defaults to the baked-in ``SUB_SCORE_WEIGHTS``.
    """
    if fallback is None:
        fallback = dict(SUB_SCORE_WEIGHTS)
    if schedule is None:
        schedule = load_wfo_weight_schedule()
    if schedule is None or schedule.empty:
        return fallback

    # Slice schedule to look-ahead-safe window based on asof.
    if asof is None:
        eligible = schedule
    else:
        ts = pd.Timestamp(asof).to_period("M").to_timestamp()
        eligible = schedule.loc[:ts]
        if eligible.empty:
            return fallback  # asof predates all folds

    if live_blend:
        return _blend_live_weights(eligible, fallback)

    # Legacy single-fold path
    row = eligible.iloc[-1]
    out = dict(fallback)  # start with fallback so missing features keep defaults
    for f in SUB_FACTORS:
        if f in row.index and pd.notna(row[f]):
            out[f] = float(row[f])
    # Renormalize defensively in case the row's sum drifts off 1.0.
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out
