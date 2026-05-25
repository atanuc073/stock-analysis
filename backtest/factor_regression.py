"""Factor-regression-based weight optimizer.

Skips the expensive backtest loop. For each (stock, rebalance-date) we look up
the per-factor sub-scores (technical/fundamental/momentum/quality/earnings_drift)
and the forward N-day return. Then we directly maximize a smooth ranking
objective:

    J(w) = IC(F @ w, fwd_ret)  +  lambda * Monotonicity(F @ w, fwd_ret)

via SLSQP on the simplex (w >= 0, sum(w) = 1), restarted from many random
Dirichlet starts.

Three-split (chronological, no shuffle):
    Train: 2014-01-01 .. 2020-12-31
    Val:   2021-01-01 .. 2023-12-31
    Test:  2024-01-01 .. 2025-12-31

The point of the splits:
  * fit on train (find candidates)
  * score on val (pick best, detect train->val degradation = overfit)
  * touch test once at the end for an honest OOS number

Output: reports/regression/<run>/candidates.csv  (top-K with all metrics).

These weight vectors are intended to be FED INTO `backtest.optimize` afterwards
as candidates for full walk-forward validation. The regression is a *filter*,
not a final answer.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
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
) -> pd.DataFrame:
    """Build a (stock x date) factor matrix with forward returns.

    One row per (symbol, rebalance-date). Columns: F1..F5 + fwd_ret + meta.
    Uses score_at() with no live weights, so each factor sub-score is independent
    of SCORE_WEIGHTS.
    """
    dates = _rebalance_dates(start, end, freq=freq)
    rows: list[dict] = []

    log.info("Building factor matrix: %d dates x %d symbols (~%d evals, horizon=%dd, freq=%s)",
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
            for f in FACTORS:
                d = getattr(s, f) or {}
                row[f] = float(d.get("score") or 50.0)
            rows.append(row)
        pbar.set_postfix(rows=len(rows), kept=len(rows) - n_before)

    df = pd.DataFrame(rows)
    log.info("Factor matrix built: %d rows | %d unique symbols | %d unique dates | skipped: score=%d fwd_ret=%d",
             len(df), df["symbol"].nunique(), df["date"].nunique(), skip_score, skip_fwd)
    if len(df):
        log.info("  Factor score stats (mean ± std):")
        for f in FACTORS:
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


def _evaluate(df: pd.DataFrame, w: np.ndarray) -> dict:
    """All metrics for a candidate w on a given dataset slice."""
    F = df[FACTORS].values
    composite = F @ w
    ic = _grouped_ic(df, composite)
    mono, spread = _quintile_monotonicity(df, composite)
    return {"ic": ic, "mono": mono, "q5q1": spread}


# ── Optimization ──────────────────────────────────────────────────────────────
def _make_objective(df: pd.DataFrame, lambda_mono: float):
    """Closure: returns scalar to MINIMIZE (negative of J)."""
    F = df[FACTORS].values

    def _neg_J(w: np.ndarray) -> float:
        # Renormalize defensively (SLSQP keeps sum=1 but tiny drift happens)
        s = float(w.sum())
        if s <= 0:
            return 1e6
        wn = w / s
        composite = F @ wn
        ic = _grouped_ic(df, composite)
        mono, _ = _quintile_monotonicity(df, composite)
        return -(ic + lambda_mono * mono)

    return _neg_J


def search_weights(
    df_train: pd.DataFrame,
    n_starts: int = 200,
    lambda_mono: float = 0.5,
    seed: int = 42,
) -> list[tuple[np.ndarray, float]]:
    """SLSQP from many random Dirichlet starts on TRAIN ONLY."""
    rng = np.random.default_rng(seed)
    neg_J = _make_objective(df_train, lambda_mono)
    n = len(FACTORS)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]

    results: list[tuple[np.ndarray, float]] = []
    seen: set[tuple] = set()
    n_fail = 0
    n_dup = 0
    best_J = -np.inf
    log_every = max(1, n_starts // 20)
    log.info("Starting SLSQP search: %d random Dirichlet starts, lambda_mono=%.2f", n_starts, lambda_mono)
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
    ap.add_argument("--n-starts", type=int, default=200,
                    help="Random Dirichlet restarts for SLSQP.")
    ap.add_argument("--lambda-mono", type=float, default=0.5,
                    help="Weight of monotonicity in objective.")
    ap.add_argument("--top-k", type=int, default=20)
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
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Universe-scoped cache so US (russell1000/sp500) and IN (nifty500) runs
    # don't overwrite each other's factor matrices.
    if not args.cache_matrix:
        safe_uni = args.universe.lower().replace(",", "_").replace(" ", "")[:40]
        args.cache_matrix = f"cache/factor_matrix_{safe_uni}.parquet"
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
    candidates = search_weights(train, n_starts=args.n_starts, lambda_mono=args.lambda_mono)
    if not candidates:
        log.error("No candidates produced. Check data / objective.")
        return

    # ── Score top-K on all splits ─────────────────────────────────────────────
    K = min(args.top_k, len(candidates))
    log.info("Evaluating top %d candidates on train / val / test ...", K)
    rows = []
    for i, (w, j_train) in enumerate(tqdm(candidates[:K], desc="Evaluating", unit="cand"), 1):
        m_tr = _evaluate(train, w)
        m_va = _evaluate(val, w)
        m_te = _evaluate(test, w)
        # Generalization gap (train IC vs val IC) — overfit detector
        gap = m_tr["ic"] - m_va["ic"]
        log.info("  Cand %2d/%d  w=[%s]  IC tr/va/te = %+.3f / %+.3f / %+.3f  Mono tr/va/te = %+.2f / %+.2f / %+.2f  gap=%+.3f",
                 i, K,
                 ", ".join(f"{f[:4]}:{wv:.2f}" for f, wv in zip(FACTORS, w)),
                 m_tr["ic"], m_va["ic"], m_te["ic"],
                 m_tr["mono"], m_va["mono"], m_te["mono"], gap)
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
            "gap_ic": round(gap, 4),
        })
    out = pd.DataFrame(rows)

    # Also rank by val-consistency (the honest selection rule)
    out["val_consistent"] = out["IC_val"] - 0.5 * out["gap_ic"].abs()
    out = out.sort_values("val_consistent", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_by_val_consistent", out.index + 1)

    csv_path = os.path.join(args.output_dir, "candidates.csv")
    out.to_csv(csv_path, index=False)
    log.info("Saved %d candidates to %s", len(out), csv_path)

    # Also save the train-ranked view for comparison
    out_by_train = out.sort_values("J_train", ascending=False).reset_index(drop=True)
    out_by_train.to_csv(os.path.join(args.output_dir, "candidates_by_train.csv"), index=False)

    # ── Console summary ──────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 110)
    print(f"FACTOR REGRESSION — top {K} candidates  (sorted by val_consistent = IC_val − 0.5·|train→val gap|)")
    print(f"  splits: train {args.train_start}..{args.train_end} | val {args.val_start}..{args.val_end} | test {args.test_start}..{args.test_end}")
    print(f"  horizon={args.horizon_days}d  freq={args.freq}  lambda_mono={args.lambda_mono}  n_starts={args.n_starts}")
    print("=" * 110)
    cols = ["rank_by_val_consistent"] + [f"w_{f}" for f in FACTORS] + [
        "IC_train", "IC_val", "IC_test",
        "Mono_train", "Mono_val", "Mono_test",
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
    print(f"    {'split':<8}  {'IC':>8}  {'Mono':>8}  {'Q5-Q1 %':>10}")
    for split_name in ["train", "val", "test"]:
        print(f"    {split_name:<8}  {best['IC_' + split_name]:>+8.4f}  {best['Mono_' + split_name]:>+8.3f}  {best['Q5Q1_' + split_name + '_pct']:>+10.3f}")
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



