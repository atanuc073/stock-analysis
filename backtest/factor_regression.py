"""Factor-regression-based weight optimizer.

Skips the expensive backtest loop. For each (stock, rebalance-date) we look up
the per-factor sub-scores (technical/fundamental/momentum/quality/earnings_drift)
and the forward N-day return. Then we directly maximize a smooth ranking
objective:

    J(w) = IC(F @ w, fwd_ret)
           + lambda_mono * Monotonicity(F @ w, fwd_ret)
           + lambda_cagr * SoftmaxBasket_AnnRet(F @ w, fwd_ret; beta)

Two optimizer backends are available (selected via --method):
  * slsqp     : multi-start SLSQP on the simplex (w >= 0, sum(w) = 1) from
                random Dirichlet starts. Fast, deterministic for a given seed.
  * bayesian  : Gaussian-Process Bayesian Optimization on a softmax-
                reparameterized simplex, using Expected Improvement. More
                sample-efficient for noisy / rank-based objectives.
  * compare   : run both and print a head-to-head report.

Two evaluation modes:
  * Static train/val/test split (default). CLI defaults:
        Train: 2020-01-01 .. 2024-12-31
        Val:   2025-01-01 .. 2025-06-30
        Test:  2025-07-01 .. 2025-12-31
    Fit on train, pick on val, touch test once for an honest OOS number.
  * --walk-forward : rolling WFO with --wf-lookback months of train and
    --wf-step months of test, with optional weight smoothing across folds.

Outputs (under --output-dir, default reports/regression/):
  * candidates.csv             (top-K candidates sorted by val_consistent)
  * candidates_by_train.csv    (same set re-sorted by train J)
  * bayesian_convergence.csv   (per-eval J trace; Bayesian only)
  * walk_forward_regression.csv (per-fold weights & test metrics; WFO only)

These weight vectors are intended to be FED INTO `backtest.optimize` afterwards
as candidates for full walk-forward validation. The regression is a *filter*,
not a final answer.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from tqdm import tqdm

from .data_loader import load_universe
from .scoring import score_at
from .optimize import _resolve_universe  # reuse universe resolver

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

FACTORS = ["technical", "fundamental", "momentum", "quality", "earnings_drift"]
N_QUANTILES = 5

# ── Sub-factor decomposition (opt-in via --sub-decomp) ──────────────────────────
# Instead of 5 pre-aggregated composite scores, expose 12 narrower features so the
# optimizer can weight orthogonal sub-signals independently. Each extractor pulls
# a raw scalar from the per-factor dicts returned by analysis/*.py; `direction`
# normalizes sense so that AFTER per-date rank transform, higher = better. This
# preserves the simplex constraint (w >= 0, sum(w) = 1) while letting the
# optimizer assign zero weight to sub-signals that don't carry IC.
#
# Entries: name -> (lambda s -> raw_value_or_None, direction: +1 higher-better, -1 lower-better)
SUB_FACTORS = [
    "tech_trend", "tech_extension", "tech_volume",
    "mom_12_1", "mom_3m", "mom_rs",
    "qual_gpa", "qual_fcf", "qual_roa",
    "fund_value", "fund_growth",
    "earnings_drift",
]
_SUB_EXTRACTORS: dict = {
    # technical
    "tech_trend":     (lambda s: (s.technical or {}).get("pct_above_sma200"), +1),
    "tech_extension": (lambda s: (s.technical or {}).get("pct_from_sma20"),   -1),  # less extension = better
    "tech_volume":    (lambda s: (s.technical or {}).get("inst_score"),       +1),
    # momentum
    "mom_12_1":       (lambda s: (s.momentum  or {}).get("mom_12_1"),         +1),
    "mom_3m":         (lambda s: (s.momentum  or {}).get("ret_3m"),           +1),
    "mom_rs":         (lambda s: (s.momentum  or {}).get("rs_value"),         +1),
    # quality
    "qual_gpa":       (lambda s: (s.quality   or {}).get("gpa"),              +1),  # Novy-Marx
    "qual_fcf":       (lambda s: (s.quality   or {}).get("fcf_yield"),        +1),
    "qual_roa":       (lambda s: (s.quality   or {}).get("roa"),              +1),
    # fundamental
    "fund_value":     (lambda s: (s.fundamental or {}).get("pe"),             -1),  # lower P/E = better
    "fund_growth":    (lambda s: (s.fundamental or {}).get("eps_growth"),     +1),
    # earnings drift (keep as one)
    "earnings_drift": (lambda s: (s.earnings_drift or {}).get("score"),       +1),
}


# ── Factor matrix construction ────────────────────────────────────────────────
def _rebalance_dates(start: pd.Timestamp, end: pd.Timestamp, freq: str = "ME") -> list[pd.Timestamp]:
    """Month-end (or week-end) rebalance dates between start and end inclusive."""
    return list(pd.date_range(start, end, freq=freq))


def _forward_return(df_close: pd.Series, asof: pd.Timestamp, horizon_days: int) -> Optional[float]:
    """Forward return over `horizon_days` trading days from the first index >= asof."""
    asof_tz = asof
    if df_close.index.tz is not None and asof.tzinfo is None:
        try:
            asof_tz = asof.tz_localize(df_close.index.tz)
        except Exception:
            asof_tz = asof.tz_localize("UTC").tz_convert(df_close.index.tz)
    elif df_close.index.tz is None and asof.tzinfo is not None:
        asof_tz = asof.tz_localize(None)

    idx = df_close.index.searchsorted(asof_tz)
    if idx >= len(df_close):
        return None
    end_idx = idx + horizon_days
    if end_idx >= len(df_close):
        return None
    p0 = float(df_close.iloc[idx])
    p1 = float(df_close.iloc[end_idx])
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0



def build_factor_matrix(
    data: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon_days: int = 21,
    freq: str = "ME",
    sub_decomp: bool = False,
) -> pd.DataFrame:
    """Build a (stock x date) factor matrix with forward returns.

    One row per (symbol, rebalance-date). Columns: factor cols + fwd_ret + meta.
    Uses score_at() with no live weights, so each factor sub-score is independent
    of SCORE_WEIGHTS.

    Two modes:
      * sub_decomp=False (default): 5 composite scores (technical, fundamental,
        momentum, quality, earnings_drift), each already 0-100 from analysis/*.py.
      * sub_decomp=True: 12 raw sub-features extracted via _SUB_EXTRACTORS, then
        cross-sectionally rank-transformed per date to a clean 0-100 scale so the
        optimizer's softmax basket and IC routines see comparable scales.
    """
    dates = _rebalance_dates(start, end, freq=freq)
    rows: list[dict] = []

    active = SUB_FACTORS if sub_decomp else FACTORS
    log.info("Building factor matrix [%s]: %d dates x %d symbols (~%d evals, horizon=%dd, freq=%s)",
             "sub-decomp" if sub_decomp else "composite",
             len(dates), len(data), len(dates) * len(data), horizon_days, freq)

    skip_score = 0
    skip_fwd = 0
    pbar = tqdm(dates, desc="Dates", unit="date")
    for asof in pbar:
        n_before = len(rows)
        for sym, hd in data.items():
            try:
                s = score_at(hd, asof, include_forecast=False, live_weights=False)
            except Exception:
                skip_score += 1
                continue
            if s is None:
                skip_score += 1
                continue
            fwd = _forward_return(hd.history["Close"], asof, horizon_days)
            if fwd is None:
                skip_fwd += 1
                continue
            row = {
                "symbol": sym,
                "date": asof,
                "fwd_ret": fwd,
                "market": s.market,
                # Filter columns (NOT used in composite — used to gate rows pre-fit)
                "rs_pct": float((s.uptrend_data or {}).get("rs_pct", 0.0) or 0.0),
                "pct_above_sma200": float(s.technical.get("pct_above_sma200", 0.0) or 0.0),
            }
            if sub_decomp:
                # Raw extraction; NaNs are filled by per-date rank transform later.
                for name, (fn, direction) in _SUB_EXTRACTORS.items():
                    try:
                        v = fn(s)
                        v = float(v) if v is not None else np.nan
                    except Exception:
                        v = np.nan
                    if np.isfinite(v):
                        row[name] = direction * v  # sign-normalize so higher=better
                    else:
                        row[name] = np.nan
            else:
                for f in FACTORS:
                    d = getattr(s, f) or {}
                    row[f] = float(d.get("score") or 50.0)
            rows.append(row)
        pbar.set_postfix(rows=len(rows), kept=len(rows) - n_before)

    df = pd.DataFrame(rows)

    if sub_decomp and len(df):
        # Per-date cross-sectional rank transform → 0-100 percentile. NaN rows
        # get assigned to the median rank (50) so missing data is neutral, not
        # extreme. This makes scales comparable across heterogeneous features
        # (P/E, momentum %, GPA, etc.) and is what professional factor models do.
        log.info("  Rank-transforming %d sub-features per date → 0-100 percentile (NaN → 50) ...", len(SUB_FACTORS))
        for f in SUB_FACTORS:
            ranked = df.groupby("date")[f].rank(pct=True, na_option="keep") * 100.0
            df[f] = ranked.fillna(50.0)

    log.info("Factor matrix built: %d rows | %d unique symbols | %d unique dates | skipped: score=%d fwd_ret=%d",
             len(df), df["symbol"].nunique(), df["date"].nunique(), skip_score, skip_fwd)
    if len(df):
        log.info("  Factor score stats (mean ± std):")
        for f in active:
            if f in df.columns:
                log.info("    %-16s mean=%5.1f  std=%5.1f  min=%5.1f  max=%5.1f",
                         f, df[f].mean(), df[f].std(), df[f].min(), df[f].max())
        log.info("  Forward return: mean=%.4f  std=%.4f  median=%.4f",
                 df["fwd_ret"].mean(), df["fwd_ret"].std(), df["fwd_ret"].median())
    return df


# ── Objective: smooth, fast, no backtest ──────────────────────────────────────
def _spearman_ic(composite: np.ndarray, fwd: np.ndarray) -> float:
    """Single-pass Spearman correlation; NaN-safe."""
    if len(composite) < 10:
        return 0.0
    rho, _ = spearmanr(composite, fwd, nan_policy="omit")
    return float(rho) if np.isfinite(rho) else 0.0


def _grouped_ic(df: pd.DataFrame, composite: np.ndarray) -> float:
    """Per-date IC, averaged across dates (more honest than pooled IC)."""
    s = df.assign(composite=composite)
    by_date = s.groupby("date").apply(
        lambda g: _spearman_ic(g["composite"].values, g["fwd_ret"].values),
        include_groups=False,
    )
    return float(by_date.mean())


def _quintile_monotonicity(df: pd.DataFrame, composite: np.ndarray) -> tuple[float, float]:
    """Spearman of (quintile_idx, mean_fwd_ret) plus Q5-Q1 spread.

    Quintiles are computed *within each date* to remove cross-sectional drift.
    """
    s = df.assign(composite=composite)
    s["q"] = s.groupby("date")["composite"].transform(
        lambda x: pd.qcut(x, N_QUANTILES, labels=False, duplicates="drop")
    )
    s = s.dropna(subset=["q"])
    means = s.groupby("q")["fwd_ret"].mean()
    if len(means) < 2:
        return 0.0, 0.0
    q_idx = means.index.values.astype(float)
    rho, _ = spearmanr(q_idx, means.values)
    mono = float(rho) if np.isfinite(rho) else 0.0
    spread = float(means.iloc[-1] - means.iloc[0]) * 100.0  # in %
    return mono, spread


def _topk_basket_return(
    df: pd.DataFrame,
    composite: np.ndarray,
    top_pct: float = 0.2,
    horizon_days: int = 21,
) -> float:
    """Annualized expected return of an equal-weight top-`top_pct` basket.

    Reporting metric only (rank-based / non-differentiable). For each rebalance
    date, pick the top `top_pct` of stocks by composite score, equal-weight them,
    take the mean of their forward returns. Average across dates and annualize
    arithmetically as `mean * 252 / horizon_days`.

    This is the closest CAGR proxy that matches the real long-only top-quintile
    portfolio you would actually trade. It overstates true geometric CAGR by
    approximately sigma^2 / 2 (volatility drag), but the relative ordering
    across candidate weight vectors is preserved.

    NOTE: Not used inside the SLSQP objective because `nlargest` is piecewise
    constant in w (zero gradient almost everywhere). The optimizer uses the
    differentiable softmax basket below.
    """
    s = df.assign(composite=composite)

    def _basket(g: pd.DataFrame) -> float:
        n_top = max(1, int(round(len(g) * top_pct)))
        return float(g.nlargest(n_top, "composite")["fwd_ret"].mean())

    per_period = s.groupby("date").apply(_basket, include_groups=False).dropna()
    if len(per_period) == 0:
        return 0.0
    periods_per_year = 252.0 / max(int(horizon_days), 1)
    return float(per_period.mean() * periods_per_year)


def _precompute_groups(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pre-split (factor_matrix, fwd_ret_vector) per date for fast objective eval.

    Avoids pandas.groupby.apply inside SLSQP's inner loop. Returns a list of
    (F_d, r_d) tuples, one per unique date, where F_d is (n_stocks_d, n_factors)
    and r_d is (n_stocks_d,).
    """
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _, g in df.groupby("date", sort=False):
        F_d = g[FACTORS].to_numpy(dtype=np.float64, copy=False)
        r_d = g["fwd_ret"].to_numpy(dtype=np.float64, copy=False)
        # Drop rows with any NaN in factors or fwd_ret
        mask = np.isfinite(F_d).all(axis=1) & np.isfinite(r_d)
        if mask.sum() < 2:
            continue
        groups.append((F_d[mask], r_d[mask]))
    return groups


def _softmax_basket_return_grouped(
    groups: list[tuple[np.ndarray, np.ndarray]],
    w: np.ndarray,
    beta: float,
    horizon_days: int,
) -> float:
    """Smooth, differentiable analogue of the top-K basket.

    For each date d:
        composite_i = F_d[i] . w
        p_i = softmax(beta * composite_i)               # portfolio weights, sum to 1
        r_port_d = sum_i p_i * r_d[i]                   # weighted forward return
    Then return arithmetic annualization of mean(r_port_d).

    As beta -> infinity, p concentrates on argmax(composite) and r_port -> top-1
    return. As beta -> 0, p -> uniform and r_port -> universe mean. A finite
    beta (e.g. 5-20) corresponds roughly to top-decile through top-quintile mass.

    Crucially, r_port is C-infinity smooth in w, so SLSQP's finite-difference
    gradients carry real information about how to move w to raise basket return.
    """
    if not groups:
        return 0.0
    total = 0.0
    n = 0
    for F_d, r_d in groups:
        z = F_d @ w
        # Numerical stability: subtract max before exp
        z = beta * z
        z -= z.max()
        p = np.exp(z)
        s = p.sum()
        if s <= 0 or not np.isfinite(s):
            continue
        p /= s
        total += float(p @ r_d)
        n += 1
    if n == 0:
        return 0.0
    periods_per_year = 252.0 / max(int(horizon_days), 1)
    return (total / n) * periods_per_year


def _evaluate(
    df: pd.DataFrame,
    w: np.ndarray,
    top_pct: float = 0.2,
    horizon_days: int = 21,
    beta: float = 10.0,
) -> dict:
    """All metrics for a candidate w on a given dataset slice.

    Reports BOTH:
      - top_ret:     hard top-`top_pct` basket annualized return (what you'd trade)
      - soft_ret:    softmax-basket annualized return at the same beta the
                     optimizer used (sanity check that optimizer's surrogate
                     matches the reporting metric)
    """
    F = df[FACTORS].values
    composite = F @ w
    ic = _grouped_ic(df, composite)
    mono, spread = _quintile_monotonicity(df, composite)
    top_ret = _topk_basket_return(df, composite, top_pct=top_pct, horizon_days=horizon_days)
    groups = _precompute_groups(df)
    soft_ret = _softmax_basket_return_grouped(groups, w, beta=beta, horizon_days=horizon_days)
    return {"ic": ic, "mono": mono, "q5q1": spread, "top_ret": top_ret, "soft_ret": soft_ret}


# ── Optimization ──────────────────────────────────────────────────────────────
def _make_objective(
    df: pd.DataFrame,
    lambda_mono: float,
    lambda_cagr: float = 0.0,
    beta: float = 10.0,
    horizon_days: int = 21,
):
    """Closure: returns scalar to MINIMIZE (negative of J).

    J(w) = IC + lambda_mono * Mono + lambda_cagr * SoftmaxBasket_AnnRet

    The SoftmaxBasket term is a *differentiable surrogate* for the hard top-K
    basket return. Concretely, portfolio weights on date d are
        p_i(w) = softmax(beta * F_d[i] . w)
    so the basket return is C-infinity in w and SLSQP's finite-difference
    gradients carry meaningful information about how to raise CAGR.

    As beta -> inf, the softmax basket converges to the hard top-1 basket.
    Practical values: beta=5 (broad ~top-half tilt), beta=10 (~top-decile mass),
    beta=20 (~top-5%). Default beta=10 ~ top-decile.

    Set lambda_cagr=0 to skip the (somewhat expensive) basket term and recover
    the legacy IC + lambda_mono*Mono objective. Scales of the three terms:
      IC          ~ 0.02 .. 0.05
      Mono        ~ 0.5  .. 1.0
      SoftBasket  ~ 0.10 .. 0.30 (annualized)
    Default lambdas (mono=0.5, cagr=1.0) put all three on a comparable scale.
    """
    F = df[FACTORS].values
    # Precompute groups once for the softmax-basket fast path
    groups = _precompute_groups(df) if lambda_cagr > 0 else []

    def _neg_J(w: np.ndarray) -> float:
        # Renormalize defensively (SLSQP keeps sum=1 but tiny drift happens)
        s = float(w.sum())
        if s <= 0:
            return 1e6
        wn = w / s
        composite = F @ wn
        ic = _grouped_ic(df, composite)
        mono, _ = _quintile_monotonicity(df, composite)
        if lambda_cagr > 0:
            soft_ret = _softmax_basket_return_grouped(groups, wn, beta=beta, horizon_days=horizon_days)
        else:
            soft_ret = 0.0
        return -(ic + lambda_mono * mono + lambda_cagr * soft_ret)

    return _neg_J


def search_weights(
    df_train: pd.DataFrame,
    n_starts: int = 200,
    lambda_mono: float = 0.5,
    lambda_cagr: float = 0.0,
    beta: float = 10.0,
    horizon_days: int = 21,
    seed: int = 42,
) -> list[tuple[np.ndarray, float]]:
    """SLSQP from many random Dirichlet starts on TRAIN ONLY."""
    rng = np.random.default_rng(seed)
    neg_J = _make_objective(df_train, lambda_mono, lambda_cagr=lambda_cagr,
                            beta=beta, horizon_days=horizon_days)
    n = len(FACTORS)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]

    results: list[tuple[np.ndarray, float]] = []
    seen: set[tuple] = set()
    n_fail = 0
    n_dup = 0
    best_J = -np.inf
    log_every = max(1, n_starts // 20)
    log.info("Starting SLSQP search: %d random Dirichlet starts, lambda_mono=%.2f, lambda_cagr=%.2f (beta=%.1f, softmax basket)",
             n_starts, lambda_mono, lambda_cagr, beta)
    pbar = tqdm(range(n_starts), desc="SLSQP starts", unit="start")
    for i in pbar:
        x0 = rng.dirichlet(np.ones(n))
        try:
            res = minimize(
                neg_J, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-6, "disp": False},
            )
        except Exception:
            n_fail += 1
            continue
        if not res.success:
            n_fail += 1
            continue
        w = np.clip(res.x, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else x0
        key = tuple(np.round(w, 3))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        J = -float(res.fun)
        results.append((w, J))
        if J > best_J:
            best_J = J
            log.info("  [start %3d/%d] new best J=%.4f  w=[%s]",
                     i + 1, n_starts, J,
                     ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w)))
        pbar.set_postfix(unique=len(results), best=f"{best_J:.3f}")
        if (i + 1) % log_every == 0:
            log.info("  progress: %d/%d starts | unique=%d | failed=%d | dup=%d | best_J=%.4f",
                     i + 1, n_starts, len(results), n_fail, n_dup, best_J)

    results.sort(key=lambda t: t[1], reverse=True)
    log.info("SLSQP complete: %d unique candidates from %d starts (failed=%d, duplicates=%d)",
             len(results), n_starts, n_fail, n_dup)
    if results:
        top3 = results[:3]
        log.info("  Top 3 by train J:")
        for rank, (w, J) in enumerate(top3, 1):
            log.info("    #%d  J=%.4f  w=[%s]",
                     rank, J,
                     ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w)))
    return results


# ── Bayesian (GP) weight optimization ─────────────────────────────────────────
def _softmax(z: np.ndarray) -> np.ndarray:
    """Map unconstrained ℝ^K → simplex via softmax."""
    e = np.exp(z - z.max())
    return e / e.sum()


def _eval_objective_for_bayes(
    df: pd.DataFrame,
    w: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]],
    objective: str,
    lambda_mono: float,
    lambda_cagr: float,
    beta: float,
    horizon_days: int,
    top_pct: float = 0.2,
) -> float:
    """Evaluate a single objective value for Bayesian optimization.

    Returns the value to MAXIMIZE.
    """
    F = df[FACTORS].values
    composite = F @ w
    ic = _grouped_ic(df, composite)

    if objective == "ic":
        return ic

    mono, spread = _quintile_monotonicity(df, composite)
    if objective == "mono":
        return mono

    if objective == "top_ret":
        return _topk_basket_return(df, composite, top_pct=top_pct, horizon_days=horizon_days)

    # Default: blended (same as SLSQP objective)
    soft_ret = 0.0
    if lambda_cagr > 0 and groups:
        soft_ret = _softmax_basket_return_grouped(groups, w, beta=beta, horizon_days=horizon_days)
    return ic + lambda_mono * mono + lambda_cagr * soft_ret


def _expected_improvement(
    X_candidates: np.ndarray,
    gp,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function."""
    from scipy.stats import norm
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def bayesian_search_weights(
    df_train: pd.DataFrame,
    n_calls: int = 80,
    n_initial: int = 15,
    objective: str = "blended",
    lambda_mono: float = 0.5,
    lambda_cagr: float = 0.0,
    beta: float = 10.0,
    horizon_days: int = 21,
    seed: int = 42,
    output_dir: str = "reports/regression",
    top_pct: float = 0.2,
    convergence_suffix: str = "",
) -> list[tuple[np.ndarray, float]]:
    """Bayesian (GP) optimization on the simplex via softmax reparameterization.

    Instead of many SLSQP restarts, fits a Gaussian Process surrogate to
    observed (z -> w -> J(w)) pairs and uses Expected Improvement to pick
    the next point to evaluate. Much more sample-efficient for noisy,
    rank-based objectives (IC, monotonicity).

    Parameters
    ----------
    n_calls : total evaluations (initial random + GP-guided). ~60-100 is typical.
    n_initial : random Dirichlet points before GP takes over.
    objective : 'blended' (IC + lam*Mono + lam*SoftRet), 'ic', 'mono', or 'top_ret'.
    output_dir : directory for convergence CSV.
    top_pct : top fraction used when objective='top_ret'.
    convergence_suffix : appended to 'bayesian_convergence' filename so WFO
        folds do not overwrite each other (e.g. '_fold03').
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

    rng = np.random.default_rng(seed)
    n = len(FACTORS)
    groups = _precompute_groups(df_train) if lambda_cagr > 0 else []

    log.info("=" * 80)
    log.info("BAYESIAN GP OPTIMIZATION")
    log.info("=" * 80)
    log.info("  objective=%s  n_calls=%d  n_initial=%d  lambda_mono=%.2f  lambda_cagr=%.2f  beta=%.1f",
             objective, n_calls, n_initial, lambda_mono, lambda_cagr, beta)

    # ── Phase 1: random initial evaluations (Dirichlet on simplex) ────────
    Z_observed: list[np.ndarray] = []  # unconstrained space
    Y_observed: list[float] = []
    W_observed: list[np.ndarray] = []  # simplex weights
    convergence: list[dict] = []  # convergence tracking
    best_J = -np.inf
    best_w = None
    best_at_random_end = -np.inf  # best J after random phase (baseline)

    log.info("Phase 1: %d random initial evaluations ...", n_initial)
    for i in range(n_initial):
        w = rng.dirichlet(np.ones(n))
        # Invert softmax: z = log(w) (up to additive constant)
        z = np.log(np.clip(w, 1e-8, None))
        z -= z.mean()  # center for numerical stability

        J = _eval_objective_for_bayes(
            df_train, w, groups, objective,
            lambda_mono, lambda_cagr, beta, horizon_days,
            top_pct=top_pct,
        )
        Z_observed.append(z)
        Y_observed.append(J)
        W_observed.append(w)

        if J > best_J:
            best_J = J
            best_w = w
            log.info("  [init %2d/%d] new best J=%.4f  w=[%s]",
                     i + 1, n_initial, J,
                     ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w)))

        convergence.append({"step": i + 1, "phase": "random", "J": round(J, 5),
                            "best_so_far": round(best_J, 5)})

    best_at_random_end = best_J

    # ── Phase 2: GP-guided search ─────────────────────────────────────────
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)

    n_gp_iters = n_calls - n_initial
    log.info("Phase 2: %d GP-guided evaluations ...", n_gp_iters)
    pbar = tqdm(range(n_gp_iters), desc="Bayesian GP", unit="eval")

    for i in pbar:
        X = np.array(Z_observed)
        Y = np.array(Y_observed)

        # Fit GP
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, alpha=1e-6,
            normalize_y=True, random_state=seed + i,
        )
        gp.fit(X, Y)

        # Generate candidate points and pick the one with highest EI
        n_candidates = 2000
        z_candidates = []
        for _ in range(n_candidates):
            w_cand = rng.dirichlet(np.ones(n))
            z_cand = np.log(np.clip(w_cand, 1e-8, None))
            z_cand -= z_cand.mean()
            z_candidates.append(z_cand)
        Z_cand = np.array(z_candidates)

        ei = _expected_improvement(Z_cand, gp, y_best=best_J, xi=0.01)
        best_idx = np.argmax(ei)
        z_next = Z_cand[best_idx]
        w_next = _softmax(z_next)

        # Evaluate
        J = _eval_objective_for_bayes(
            df_train, w_next, groups, objective,
            lambda_mono, lambda_cagr, beta, horizon_days,
            top_pct=top_pct,
        )
        Z_observed.append(z_next)
        Y_observed.append(J)
        W_observed.append(w_next)

        if J > best_J:
            best_J = J
            best_w = w_next
            log.info("  [GP %3d/%d] new best J=%.4f  w=[%s]  (EI=%.4f)",
                     i + 1, n_gp_iters, J,
                     ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w_next)),
                     ei[best_idx])

        convergence.append({"step": n_initial + i + 1, "phase": "gp",
                            "J": round(J, 5), "best_so_far": round(best_J, 5)})
        pbar.set_postfix(best=f"{best_J:.4f}", ei_max=f"{ei.max():.4f}")

    # ── Convergence diagnostics ───────────────────────────────────────────
    gp_improvement = best_J - best_at_random_end
    log.info("CONVERGENCE: random_best=%.4f  final_best=%.4f  GP_improvement=%+.4f (%.1f%%)",
             best_at_random_end, best_J, gp_improvement,
             100 * gp_improvement / abs(best_at_random_end) if best_at_random_end != 0 else 0)
    if gp_improvement <= 0:
        log.warning("  GP phase did NOT improve over random — objective may be too flat or noisy.")

    # Save convergence CSV for analysis
    try:
        os.makedirs(output_dir, exist_ok=True)
        conv_df = pd.DataFrame(convergence)
        conv_path = os.path.join(
            output_dir, f"bayesian_convergence{convergence_suffix}.csv"
        )
        conv_df.to_csv(conv_path, index=False)
        log.info("  Convergence trace saved to %s", conv_path)
    except Exception as e:
        log.warning("  Could not save convergence CSV: %s", e)

    # ── Deduplicate and sort ──────────────────────────────────────────────
    results: list[tuple[np.ndarray, float]] = []
    seen: set[tuple] = set()
    for w, J in zip(W_observed, Y_observed):
        key = tuple(np.round(w, 3))
        if key in seen:
            continue
        seen.add(key)
        results.append((w, J))

    results.sort(key=lambda t: t[1], reverse=True)
    log.info("Bayesian GP complete: %d unique candidates from %d evaluations", len(results), n_calls)
    if results:
        log.info("  Top 3 by train J:")
        for rank, (w, J) in enumerate(results[:3], 1):
            log.info("    #%d  J=%.4f  w=[%s]",
                     rank, J,
                     ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w)))
    return results


def run_walk_forward(
    df: pd.DataFrame,
    args: argparse.Namespace,
    common_bayes_kw: dict,
    common_slsqp_kw: dict,
) -> None:
    """Run rolling Walk-Forward Optimization (WFO) over time."""
    # Ensure date is a DatetimeIndex
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Find all unique rebalance dates and sort them
    dates = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
    
    if len(dates) == 0:
        log.error("No dates available for WFO.")
        return
        
    # We will step month by month. To do this robustly, we use YearMonth periods
    df["ym"] = df["date"].dt.to_period("M")
    months = pd.Series(df["ym"].unique()).sort_values().reset_index(drop=True)
    
    if len(months) <= args.wf_lookback:
        log.error("Not enough months for WFO lookback. Have %d, need >%d", len(months), args.wf_lookback)
        return
        
    log.info("=" * 80)
    log.info("WALK-FORWARD OPTIMIZATION (WFO)")
    log.info("=" * 80)
    log.info("  Lookback: %d months  Step: %d month(s)  Smoothing: %.2f", args.wf_lookback, args.wf_step, args.wf_smoothing)
    
    results = []
    prev_w = None
    fold_idx = 0

    for i in range(args.wf_lookback, len(months), args.wf_step):
        fold_idx += 1
        train_start = months.iloc[i - args.wf_lookback]
        train_end = months.iloc[i - 1]
        
        test_start = months.iloc[i]
        test_end = months.iloc[min(i + args.wf_step - 1, len(months) - 1)]
        
        train_mask = (df["ym"] >= train_start) & (df["ym"] <= train_end)
        test_mask = (df["ym"] >= test_start) & (df["ym"] <= test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        log.info("-" * 80)
        log.info("WFO Fold %d: Train [%s .. %s] (rows=%d) -> Test [%s .. %s] (rows=%d)",
                 fold_idx, train_start, train_end, len(train_df), test_start, test_end, len(test_df))

        if len(train_df) < 100:
            log.warning("Not enough train data, skipping fold.")
            continue

        if args.method == "bayesian":
            cands = bayesian_search_weights(
                train_df, **common_bayes_kw,
                convergence_suffix=f"_fold{fold_idx:03d}",
            )
        else:
            cands = search_weights(train_df, **common_slsqp_kw)
            
        if not cands:
            log.warning("No candidates found, skipping fold.")
            continue
            
        w_best, j_best = cands[0]
        
        # Apply smoothing
        if prev_w is not None and args.wf_smoothing > 0:
            w_smoothed = args.wf_smoothing * prev_w + (1.0 - args.wf_smoothing) * w_best
            w_smoothed = w_smoothed / w_smoothed.sum()  # Re-normalize
        else:
            w_smoothed = w_best
            
        prev_w = w_smoothed
        
        # Evaluate on test
        if len(test_df) > 0:
            m_te = _evaluate(test_df, w_smoothed, top_pct=args.top_pct, horizon_days=args.horizon_days, beta=args.beta)
        else:
            m_te = {"ic": 0.0, "mono": 0.0, "q5q1": 0.0, "top_ret": 0.0}
            
        row = {
            "test_start": str(test_start),
            "test_end": str(test_end),
            **{f"w_{f}": round(float(w_smoothed[k]), 4) for k, f in enumerate(FACTORS)},
            "IC_test": round(m_te["ic"], 4),
            "Mono_test": round(m_te["mono"], 3),
            "TopRet_test_pct": round(100 * m_te["top_ret"], 2),
        }
        results.append(row)
        
        w_str = ", ".join(f"{f[:4]}:{w_smoothed[k]:.2f}" for k, f in enumerate(FACTORS))
        log.info("  -> w=[%s]  Test IC=%+.4f  Mono=%+.3f  TopRet=%+.1f%%",
                 w_str, m_te["ic"], m_te["mono"], 100 * m_te["top_ret"])
                 
    out_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "walk_forward_regression.csv")
    out_df.to_csv(csv_path, index=False)
    log.info("Saved WFO results to %s", csv_path)
    
    print("\n" + "=" * 100)
    print("WALK-FORWARD OPTIMIZATION SUMMARY")
    print("=" * 100)
    print(out_df.to_string(index=False))
    print("-" * 100)
    print(f"Mean Test IC:     {out_df['IC_test'].mean():+.4f}")
    print(f"Mean Test Mono:   {out_df['Mono_test'].mean():+.3f}")
    print(f"Mean Test TopRet: {out_df['TopRet_test_pct'].mean():+.2f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Factor-regression weight optimizer with train/val/test splits.")
    ap.add_argument("--universe", default="nifty500",
                    help="Universe: nifty500, nse_all, russell1000, russell2000, sp500, sp400, "
                         "broad, india, us, watchlist, or a comma-separated symbol list.")
    ap.add_argument("--sample-size", type=int, default=0,
                    help="Random subset of N symbols from the universe (0 = all).")
    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--train-end",   default="2024-12-31")
    ap.add_argument("--val-start",   default="2025-01-01")
    ap.add_argument("--val-end",     default="2025-06-30")
    ap.add_argument("--test-start",  default="2025-07-01")
    ap.add_argument("--test-end",    default="2025-12-31")
    ap.add_argument("--horizon-days", type=int, default=21,
                    help="Forward-return horizon in trading days (default 21 ~ 1 month).")
    ap.add_argument("--freq", default="ME",
                    help="Rebalance frequency for rows. ME=month-end, W-FRI=Friday-weekly.")
    ap.add_argument("--method", choices=["slsqp", "bayesian", "compare"], default="slsqp",
                    help="Optimization method. 'slsqp' (default): multi-start SLSQP. "
                         "'bayesian': Gaussian Process BO with Expected Improvement — "
                         "more sample-efficient for noisy rank-based objectives.")
    ap.add_argument("--n-starts", type=int, default=200,
                    help="Random Dirichlet restarts for SLSQP.")
    ap.add_argument("--lambda-mono", type=float, default=0.5,
                    help="Weight of monotonicity in objective.")
    ap.add_argument("--lambda-cagr", type=float, default=0.0,
                    help="Weight of softmax-basket annualized return in objective "
                         "(set 0 to disable; default 0.0). Differentiable surrogate "
                         "for top-K basket CAGR; SLSQP can actually climb its gradient.")
    ap.add_argument("--beta", type=float, default=10.0,
                    help="Softmax sharpness for basket return surrogate. Higher = more "
                         "concentrated basket. beta=5 ~ top-half tilt, beta=10 ~ top-decile, "
                         "beta=20 ~ top-5%%. Default 10.")
    ap.add_argument("--top-pct", type=float, default=0.2,
                    help="Top fraction of stocks per date for the *reporting* hard-basket "
                         "AnnRet metric (default 0.2 = top quintile). This is the basket "
                         "you would actually trade; it does NOT enter the SLSQP objective.")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--factors", default=None,
                    help="Comma-separated list of active factors to optimize. "
                         "Default: all 5 (technical,fundamental,momentum,quality,earnings_drift). "
                         "Example: --factors technical,momentum,fundamental")
    # ── Bayesian-specific args ────────────────────────────────────────────────
    ap.add_argument("--bayes-n-calls", type=int, default=80,
                    help="Total evaluations for Bayesian BO (initial + GP-guided). Default 80.")
    ap.add_argument("--bayes-n-initial", type=int, default=15,
                    help="Random Dirichlet evaluations before GP takes over. Default 15.")
    ap.add_argument("--bayes-objective", choices=["blended", "ic", "mono", "top_ret"],
                    default="blended",
                    help="Objective for Bayesian BO. 'blended' = IC + λ·Mono + λ·SoftRet "
                         "(same as SLSQP). 'ic' = maximize IC only. 'mono' = maximize "
                         "monotonicity only. 'top_ret' = maximize top-quintile basket return.")
                         
    # ── Walk-Forward args ─────────────────────────────────────────────────────
    ap.add_argument("--walk-forward", action="store_true",
                    help="Run rolling Walk-Forward Optimization (WFO) over time instead of static splits.")
    ap.add_argument("--wf-lookback", type=int, default=36,
                    help="Lookback window in months for WFO train period (default 36).")
    ap.add_argument("--wf-step", type=int, default=1,
                    help="Step size in months for WFO test period (default 1).")
    ap.add_argument("--wf-smoothing", type=float, default=0.0,
                    help="Weight smoothing factor. 0.0=no smoothing, 0.7=70%% old weight + 30%% new weight.")

    ap.add_argument("--cache-matrix", default=None,
                    help="Parquet path for the (stock x date) factor matrix. "
                         "Default: cache/factor_matrix_<universe>.parquet (universe-scoped to avoid "
                         "clobbering across markets).")
    ap.add_argument("--rebuild-matrix", action="store_true",
                    help="Force rebuild even if cache exists.")
    ap.add_argument("--output-dir", default="reports/regression")
    ap.add_argument("--max-workers", type=int, default=6)

    # ── Entry filters (mirror backtest/engine.py defaults) ────────────────────────────
    # The deployed scoring weights are only used on rows that pass the live filter
    # stack, so we must measure factor IC on the SAME filtered pool.
    ap.add_argument("--apply-filters", action="store_true",
                    help="Pre-filter rows to those that would pass entry gates (RS, 200DMA, extension).")
    ap.add_argument("--min-rs-pct", type=float, default=70.0,
                    help="Drop rows with rs_pct < this (default 70, matches engine).")
    ap.add_argument("--require-above-sma200", action="store_true", default=True,
                    help="Drop rows where price is below 200DMA (default on).")
    ap.add_argument("--max-extension-pct", type=float, default=40.0,
                    help="Drop rows extended > this %% above 200DMA (default 40, matches engine).")

    # ── Sub-decomposition (opt-in, additive: leaves 5-factor path untouched) ────
    ap.add_argument("--sub-decomp", action="store_true",
                    help="Use 12 raw sub-features (3 technical + 3 momentum + 3 quality + 2 "
                         "fundamental + 1 earnings_drift) instead of the 5 composite scores. "
                         "Features are cross-sectionally rank-transformed per date to 0-100 so "
                         "heterogeneous raw scales (P/E, %% returns, GPA, ...) don't break the "
                         "softmax basket. Output goes to <output-dir>/sub/ and uses a separate "
                         "cache file so the 5-factor cache is untouched. Revert by dropping the flag.")
    args = ap.parse_args()

    # ── Override active factors if --factors is specified ──────────────────────
    global FACTORS
    ALL_FACTORS = ["technical", "fundamental", "momentum", "quality", "earnings_drift"]
    if args.sub_decomp:
        if args.factors:
            log.error("--sub-decomp and --factors are mutually exclusive.")
            return
        FACTORS = list(SUB_FACTORS)
        log.info("=" * 80)
        log.info("SUB-DECOMPOSITION MODE — using %d sub-features instead of 5 composites", len(FACTORS))
        log.info("=" * 80)
        log.info("  Features: %s", FACTORS)
        # Redirect output + cache so existing 5-factor artifacts stay intact.
        args.output_dir = os.path.join(args.output_dir, "sub")
    elif args.factors:
        selected = [f.strip() for f in args.factors.split(",")]
        invalid = [f for f in selected if f not in ALL_FACTORS]
        if invalid:
            log.error("Unknown factors: %s. Valid: %s", invalid, ALL_FACTORS)
            return
        FACTORS = selected
        log.info("Active factors overridden: %s", FACTORS)
    else:
        FACTORS = list(ALL_FACTORS)

    os.makedirs(args.output_dir, exist_ok=True)

    # Universe-scoped cache so US (russell1000/sp500) and IN (nifty500) runs
    # don't overwrite each other's factor matrices. Sub-decomp uses its own
    # cache file so we can switch modes without rebuilding the composite matrix.
    if not args.cache_matrix:
        safe_uni = args.universe.lower().replace(",", "_").replace(" ", "")[:40]
        suffix = "_sub" if args.sub_decomp else ""
        args.cache_matrix = f"cache/factor_matrix_{safe_uni}{suffix}.parquet"
    log.info("Factor matrix cache: %s  (universe=%s)", args.cache_matrix, args.universe)

    # ── Build or load factor matrix ───────────────────────────────────────────
    if (not args.rebuild_matrix) and os.path.exists(args.cache_matrix):
        log.info("Loading cached factor matrix from %s", args.cache_matrix)
        df_all = pd.read_parquet(args.cache_matrix)
        df_all["date"] = pd.to_datetime(df_all["date"])
    else:
        symbols = _resolve_universe(args.universe)
        if args.sample_size and args.sample_size < len(symbols):
            rng = np.random.default_rng(42)
            symbols = list(rng.choice(symbols, size=args.sample_size, replace=False))
            log.info("Sampled %d symbols from %s universe", len(symbols), args.universe)
        # Need enough history for momentum (1y) + horizon
        data_start = (pd.Timestamp(args.train_start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        log.info("Loading OHLCV from %s to %s for %d symbols", data_start, args.test_end, len(symbols))
        data = load_universe(symbols, data_start, args.test_end, max_workers=args.max_workers)
        df_all = build_factor_matrix(
            data,
            start=pd.Timestamp(args.train_start),
            end=pd.Timestamp(args.test_end),
            horizon_days=args.horizon_days,
            freq=args.freq,
            sub_decomp=args.sub_decomp,
        )
        os.makedirs(os.path.dirname(args.cache_matrix) or ".", exist_ok=True)
        df_all.to_parquet(args.cache_matrix, index=False)
        log.info("Cached factor matrix to %s", args.cache_matrix)

    # ── Split ─────────────────────────────────────────────────────────────────
    df_all = df_all.sort_values("date").reset_index(drop=True)

    # ── Apply entry filters BEFORE splitting ──────────────────────────────────
    if args.apply_filters:
        n0 = len(df_all)
        log.info("Applying entry filters: rs_pct>=%.0f, above_200dma=%s, max_ext=%.0f%%",
                 args.min_rs_pct, args.require_above_sma200, args.max_extension_pct)
        if "rs_pct" not in df_all.columns or "pct_above_sma200" not in df_all.columns:
            log.warning("Cached matrix lacks filter columns; rebuild with --rebuild-matrix to enable filters.")
        else:
            mask = pd.Series(True, index=df_all.index)
            if args.min_rs_pct > 0:
                mask &= df_all["rs_pct"] >= args.min_rs_pct
                log.info("  rs_pct >= %.0f      keeps %d / %d (%.1f%%)",
                         args.min_rs_pct, mask.sum(), n0, 100 * mask.sum() / n0)
            if args.require_above_sma200:
                mask &= df_all["pct_above_sma200"] > 0
                log.info("  above 200DMA       keeps %d / %d (%.1f%%)",
                         mask.sum(), n0, 100 * mask.sum() / n0)
            if args.max_extension_pct > 0:
                mask &= df_all["pct_above_sma200"] <= args.max_extension_pct
                log.info("  ext <= %.0f%%        keeps %d / %d (%.1f%%)",
                         args.max_extension_pct, mask.sum(), n0, 100 * mask.sum() / n0)
            df_all = df_all[mask].copy()
            log.info("After filters: %d rows kept (%.1f%% of %d)",
                     len(df_all), 100 * len(df_all) / n0, n0)
    else:
        log.info("No entry filters applied (use --apply-filters to gate on RS/200DMA/extension).")

    _common_bayes_kw = dict(
        n_calls=args.bayes_n_calls, n_initial=args.bayes_n_initial,
        objective=args.bayes_objective, lambda_mono=args.lambda_mono,
        lambda_cagr=args.lambda_cagr, beta=args.beta,
        horizon_days=args.horizon_days, output_dir=args.output_dir,
        top_pct=args.top_pct,
    )
    _common_slsqp_kw = dict(
        n_starts=args.n_starts, lambda_mono=args.lambda_mono,
        lambda_cagr=args.lambda_cagr, beta=args.beta,
        horizon_days=args.horizon_days,
    )

    if args.walk_forward:
        if args.test_end:
            df_all = df_all[df_all["date"] <= args.test_end].copy()
        run_walk_forward(df_all, args, _common_bayes_kw, _common_slsqp_kw)
        return

    train = df_all[(df_all["date"] >= args.train_start) & (df_all["date"] <= args.train_end)].copy()
    val   = df_all[(df_all["date"] >= args.val_start)   & (df_all["date"] <= args.val_end)].copy()
    test  = df_all[(df_all["date"] >= args.test_start)  & (df_all["date"] <= args.test_end)].copy()
    log.info("=" * 80)
    log.info("SPLIT SUMMARY")
    log.info("=" * 80)
    for name, df_s, start_s, end_s in [
        ("TRAIN", train, args.train_start, args.train_end),
        ("VAL  ", val,   args.val_start,   args.val_end),
        ("TEST ", test,  args.test_start,  args.test_end),
    ]:
        if len(df_s):
            log.info("  %s [%s .. %s]  rows=%d  symbols=%d  dates=%d  fwd_ret_mean=%.4f",
                     name, start_s, end_s, len(df_s),
                     df_s["symbol"].nunique(), df_s["date"].nunique(),
                     df_s["fwd_ret"].mean())
        else:
            log.warning("  %s [%s .. %s]  EMPTY", name, start_s, end_s)
    log.info("=" * 80)
    if min(len(train), len(val), len(test)) < 100:
        log.warning("One split has <100 rows. Results will be noisy.")

    # ── Fit on train ──────────────────────────────────────────────────────────
    if args.method == "compare":
        # ── HEAD-TO-HEAD: run both methods, compare on val/test ───────────
        log.info("COMPARE MODE: running both SLSQP and Bayesian ...")
        cand_slsqp = search_weights(train, **_common_slsqp_kw)
        cand_bayes = bayesian_search_weights(train, **_common_bayes_kw)

        print("\n" + "=" * 130)
        print("HEAD-TO-HEAD COMPARISON: SLSQP vs BAYESIAN")
        print("=" * 130)
        header = f"  {'Method':<10} {'Weights':>50}  {'IC_tr':>7} {'IC_va':>7} {'IC_te':>7}  {'Mono_tr':>7} {'Mono_va':>7} {'Mono_te':>7}  {'TopRet_va':>10} {'TopRet_te':>10}  {'gap':>6}"
        print(header)
        print("-" * 130)

        for label, cands in [("SLSQP", cand_slsqp), ("Bayesian", cand_bayes)]:
            if not cands:
                print(f"  {label:<10}  NO CANDIDATES")
                continue
            w_best, j_best = cands[0]
            m_tr = _evaluate(train, w_best, top_pct=args.top_pct, horizon_days=args.horizon_days, beta=args.beta)
            m_va = _evaluate(val,   w_best, top_pct=args.top_pct, horizon_days=args.horizon_days, beta=args.beta)
            m_te = _evaluate(test,  w_best, top_pct=args.top_pct, horizon_days=args.horizon_days, beta=args.beta)
            gap = m_tr["ic"] - m_va["ic"]
            w_str = ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w_best))
            print(f"  {label:<10} {w_str:>50}  {m_tr['ic']:>+7.4f} {m_va['ic']:>+7.4f} {m_te['ic']:>+7.4f}"
                  f"  {m_tr['mono']:>+7.3f} {m_va['mono']:>+7.3f} {m_te['mono']:>+7.3f}"
                  f"  {100*m_va['top_ret']:>+10.2f} {100*m_te['top_ret']:>+10.2f}  {gap:>+6.3f}")
        print("=" * 130)
        print("\nUse the method whose VALIDATION metrics are stronger (IC_va, Mono_va, TopRet_va).")
        print("Test metrics are for honest OOS confirmation only — do NOT pick based on test.")
        print()

        # Use Bayesian as the primary candidate set for the rest of the report
        candidates = cand_bayes if cand_bayes else cand_slsqp
    elif args.method == "bayesian":
        candidates = bayesian_search_weights(train, **_common_bayes_kw)
    else:
        candidates = search_weights(train, **_common_slsqp_kw)

    if not candidates:
        log.error("No candidates produced. Check data / objective.")
        return

    # ── Score top-K on all splits ─────────────────────────────────────────────
    K = min(args.top_k, len(candidates))
    log.info("Evaluating top %d candidates on train / val / test ...", K)
    rows = []
    for i, (w, j_train) in enumerate(tqdm(candidates[:K], desc="Evaluating", unit="cand"), 1):
        m_tr = _evaluate(train, w, top_pct=args.top_pct, horizon_days=args.horizon_days, beta=args.beta)
        m_va = _evaluate(val,   w, top_pct=args.top_pct, horizon_days=args.horizon_days, beta=args.beta)
        m_te = _evaluate(test,  w, top_pct=args.top_pct, horizon_days=args.horizon_days, beta=args.beta)
        # Generalization gap (train IC vs val IC) — overfit detector
        gap = m_tr["ic"] - m_va["ic"]
        log.info("  Cand %2d/%d  w=[%s]  IC tr/va/te=%+.3f/%+.3f/%+.3f  Mono=%+.2f/%+.2f/%+.2f  TopRet=%+.1f%%/%+.1f%%/%+.1f%%  gap=%+.3f",
                 i, K,
                 ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w)),
                 m_tr["ic"], m_va["ic"], m_te["ic"],
                 m_tr["mono"], m_va["mono"], m_te["mono"],
                 100*m_tr["top_ret"], 100*m_va["top_ret"], 100*m_te["top_ret"], gap)
        rows.append({
            "rank": i,
            **{f"w_{f}": round(float(w[k]), 4) for k, f in enumerate(FACTORS)},
            "J_train": round(j_train, 4),
            "IC_train": round(m_tr["ic"], 4),
            "IC_val":   round(m_va["ic"], 4),
            "IC_test":  round(m_te["ic"], 4),
            "Mono_train": round(m_tr["mono"], 3),
            "Mono_val":   round(m_va["mono"], 3),
            "Mono_test":  round(m_te["mono"], 3),
            "Q5Q1_train_pct": round(m_tr["q5q1"], 2),
            "Q5Q1_val_pct":   round(m_va["q5q1"], 2),
            "Q5Q1_test_pct":  round(m_te["q5q1"], 2),
            "TopRet_train_pct": round(100 * m_tr["top_ret"], 2),
            "TopRet_val_pct":   round(100 * m_va["top_ret"], 2),
            "TopRet_test_pct":  round(100 * m_te["top_ret"], 2),
            "gap_ic": round(gap, 4),
        })
    out = pd.DataFrame(rows)

    # Val-consistency ranking now blends IC, monotonicity, AND top-basket return
    # on the val split, penalized by train→val IC gap. Tuned so each component
    # contributes comparably (TopRet is on [0,1], IC ~ [0, 0.05], Mono ~ [-1, 1]).
    out["val_consistent"] = (
        out["IC_val"]
        + 0.05 * out["Mono_val"]
        + 0.20 * (out["TopRet_val_pct"] / 100.0)
        - 0.5 * out["gap_ic"].abs()
    )
    out = out.sort_values("val_consistent", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_by_val_consistent", out.index + 1)

    csv_path = os.path.join(args.output_dir, "candidates.csv")
    out.to_csv(csv_path, index=False)
    log.info("Saved %d candidates to %s", len(out), csv_path)

    # Also save the train-ranked view for comparison
    out_by_train = out.sort_values("J_train", ascending=False).reset_index(drop=True)
    out_by_train.to_csv(os.path.join(args.output_dir, "candidates_by_train.csv"), index=False)

    # ── Console summary ──────────────────────────────────────────────────────
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print("\n" + "=" * 130)
    print(f"FACTOR REGRESSION — top {K} candidates  (sorted by val_consistent: IC_val + 0.05·Mono_val + 0.20·TopRet_val − 0.5·|train→val IC gap|)")
    print(f"  splits: train {args.train_start}..{args.train_end} | val {args.val_start}..{args.val_end} | test {args.test_start}..{args.test_end}")
    method_info = f"method={args.method}"
    if args.method in ("bayesian", "compare"):
        method_info += f"  bayes_obj={args.bayes_objective}  n_calls={args.bayes_n_calls}"
    else:
        method_info += f"  n_starts={args.n_starts}"
    print(f"  horizon={args.horizon_days}d  freq={args.freq}  lambda_mono={args.lambda_mono}  lambda_cagr={args.lambda_cagr}  beta={args.beta}  top_pct={args.top_pct}  {method_info}")
    print("=" * 130)
    cols = ["rank_by_val_consistent"] + [f"w_{f}" for f in FACTORS] + [
        "IC_train", "IC_val", "IC_test",
        "Mono_train", "Mono_val", "Mono_test",
        "TopRet_train_pct", "TopRet_val_pct", "TopRet_test_pct",
        "Q5Q1_train_pct", "Q5Q1_val_pct", "Q5Q1_test_pct",
        "gap_ic",
    ]
    print(out[cols].to_string(index=False))

    # ── Best candidate detail ────────────────────────────────────────────────
    best = out.iloc[0]
    print("\n" + "-" * 110)
    print("BEST CANDIDATE (by val_consistent)")
    print("-" * 110)
    print("  Weights:")
    for f in FACTORS:
        print(f"    {f:16s} = {best['w_' + f]:.4f}")
    print("  Performance:")
    print(f"    {'split':<8}  {'IC':>8}  {'Mono':>8}  {'Q5-Q1 %':>10}  {'TopRet %':>10}")
    for split_name in ["train", "val", "test"]:
        print(f"    {split_name:<8}  {best['IC_' + split_name]:>+8.4f}  {best['Mono_' + split_name]:>+8.3f}  {best['Q5Q1_' + split_name + '_pct']:>+10.3f}  {best['TopRet_' + split_name + '_pct']:>+10.2f}")
    print(f"  Train→Val IC gap: {best['gap_ic']:+.4f}")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    print("\n" + "-" * 110)
    print("DIAGNOSIS")
    print("-" * 110)
    healthy = True
    if abs(best["gap_ic"]) > 0.03:
        print(f"  [WARN] Large train→val IC gap ({best['gap_ic']:+.3f}). Possible overfit.")
        healthy = False
    if best["IC_val"] < 0.01:
        print(f"  [WARN] Validation IC very low ({best['IC_val']:+.3f}). Possible underfit or weak factors.")
        healthy = False
    if best["Mono_test"] < 0.3 and best["Mono_val"] >= 0.5:
        print(f"  [WARN] Test monotonicity ({best['Mono_test']:+.2f}) collapsed vs val ({best['Mono_val']:+.2f}). Regime-specific weights.")
        healthy = False
    if best["IC_test"] < 0 and best["IC_val"] > 0:
        print(f"  [WARN] Test IC flipped sign ({best['IC_test']:+.3f}). Possible regime break in 2024-2025.")
        healthy = False
    if healthy and best["IC_test"] > 0 and best["IC_val"] > 0 and best["IC_train"] > 0 and abs(best["gap_ic"]) <= 0.02:
        print("  [OK] Healthy: positive IC on all 3 splits, small train→val degradation.")

    # ── Suggested SCORE_WEIGHTS block (paste into config.py) ─────────────────
    print("\n" + "-" * 110)
    print("SUGGESTED SCORE_WEIGHTS (best candidate, paste into config.py):")
    print("-" * 110)
    print("SCORE_WEIGHTS = {")
    for f in FACTORS:
        print(f'    "{f}":{" " * (16 - len(f))}{best["w_" + f]:.3f},')
    print("}")

    print("\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {os.path.join(args.output_dir, 'candidates_by_train.csv')}")
    print(f"\nNext step: feed the top weights into backtest.optimize for full walk-forward validation.")


if __name__ == "__main__":
    main()
