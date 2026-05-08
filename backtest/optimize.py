"""Walk-forward weight optimizer for the composite score.

Searches for SCORE_WEIGHTS that maximize a chosen objective (Sharpe,
Sortino, Calmar, or CAGR) on out-of-sample data. Two modes:

    1. Single split (default): train on first window, test on second.
    2. Walk-forward (--walk-forward): rolling 3y train / 1y test windows.

Two search strategies:
    - 'random'   — Dirichlet-sampled weight vectors (default; fast, broad).
    - 'sleeves'  — single-factor sleeves (each factor at 1.0, others 0).
                   Used to measure the marginal contribution of each factor;
                   shows you which factors are pulling weight and which are
                   noise.

Examples:
    # Random search, 100 candidates, single split
    python -m backtest.optimize --start 2019-01-01 --end 2025-01-01 \\
        --candidates 100

    # Diagnose which factors carry the strategy (zero search; one run per factor)
    python -m backtest.optimize --start 2019-01-01 --end 2025-01-01 \\
        --strategy sleeves

    # Walk-forward — 3y train, 1y test, rolling
    python -m backtest.optimize --start 2018-01-01 --end 2025-01-01 \\
        --candidates 80 --walk-forward
"""
from __future__ import annotations
import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import REPORTS_DIR, WATCHLIST, WATCHLIST_INDIA, WATCHLIST_US

from .data_loader import load_universe, trading_dates
from .engine import BacktestConfig, BacktestEngine
from .results import compute as compute_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("backtest.optimize")

# Components that are scored in backtest live_weights mode (sentiment/options
# are dropped because no historical data; forecast off by default).
ACTIVE_FACTORS = ["technical", "fundamental", "momentum", "quality", "earnings_drift"]


def _resolve_universe(name: str) -> list[str]:
    n = name.lower()
    if n == "watchlist": return WATCHLIST
    if n == "india":     return WATCHLIST_INDIA
    if n == "us":        return WATCHLIST_US
    return [s.strip() for s in name.split(",") if s.strip()]


def _dirichlet_weights(n_candidates: int, factors: list[str],
                       seed: int = 42) -> list[dict]:
    """Sample weight vectors uniformly from the simplex (sum=1, each >=0).

    Uses Dirichlet(1, 1, ..., 1) which is uniform on the simplex.
    """
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(len(factors)), size=n_candidates)
    out = []
    for row in samples:
        w = {f: float(round(row[i], 4)) for i, f in enumerate(factors)}
        # Add zero-weighted slots for non-active factors so config-shape
        # is preserved end-to-end.
        for k in ("sentiment", "options", "forecast", "valuation"):
            w.setdefault(k, 0.0)
        # Renormalize after rounding drift
        total = sum(w[f] for f in factors)
        if total > 0:
            for f in factors:
                w[f] = w[f] / total
        out.append(w)
    return out


def _sleeve_weights(factors: list[str]) -> list[dict]:
    """One weight vector per factor: that factor at 1.0, all others at 0."""
    out = []
    for active in factors:
        w = {f: (1.0 if f == active else 0.0) for f in factors}
        for k in ("sentiment", "options", "forecast", "valuation"):
            w.setdefault(k, 0.0)
        out.append(w)
    return out


def _run_backtest(data, dates, weights: dict, capital: float,
                  threshold: float, max_pos: int) -> dict:
    """Run one backtest with given weights; return key stats dict."""
    cfg = BacktestConfig(
        initial_capital=capital,
        rebalance_freq_days=5,
        min_score=threshold,
        max_positions=max_pos,
        include_forecast=False,
        live_weights=True,
        use_regime=True,
        weights=weights,
    )
    engine = BacktestEngine(cfg, data)
    result = engine.run(dates)
    if not result.equity_curve:
        return {"sharpe": -99.0, "cagr": -99.0, "calmar": -99.0,
                "sortino": -99.0, "trades": 0, "max_dd": 0.0, "final": capital}
    try:
        stats = compute_stats(result)
    except Exception as e:
        log.warning("stats failed: %s", e)
        return {"sharpe": -99.0, "cagr": -99.0, "calmar": -99.0,
                "sortino": -99.0, "trades": 0, "max_dd": 0.0, "final": capital}
    return {
        "sharpe":   float(stats.sharpe_ratio),
        "sortino":  float(stats.sortino_ratio),
        "calmar":   float(stats.calmar_ratio),
        "cagr":     float(stats.cagr_pct),
        "max_dd":   float(stats.max_drawdown_pct),
        "trades":   int(stats.total_trades),
        "win_rate": float(stats.win_rate_pct),
        "final":    float(stats.final_equity),
    }


def _objective_value(stats: dict, objective: str) -> float:
    """Map a stats dict to a single scalar for ranking."""
    if stats["trades"] < 5:
        return -99.0  # exclude pathological zero-trade runs
    if objective == "sharpe":  return stats["sharpe"]
    if objective == "sortino": return stats["sortino"]
    if objective == "calmar":  return stats["calmar"]
    if objective == "cagr":    return stats["cagr"]
    raise ValueError(f"unknown objective: {objective}")


def _slice_dates(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    return dates[(dates >= s) & (dates <= e)]


def _walk_forward_windows(start: str, end: str,
                           train_years: int = 3,
                           test_years: int = 1) -> list[tuple[str, str, str, str]]:
    """Generate (train_start, train_end, test_start, test_end) tuples."""
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    windows = []
    cur = s
    while cur + pd.DateOffset(years=train_years + test_years) <= e:
        ts = cur
        te = ts + pd.DateOffset(years=train_years)
        vs = te + pd.Timedelta(days=1)
        ve = vs + pd.DateOffset(years=test_years)
        windows.append((ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d"),
                        vs.strftime("%Y-%m-%d"), ve.strftime("%Y-%m-%d")))
        cur = cur + pd.DateOffset(years=test_years)
    return windows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward weight optimizer.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--universe", default="watchlist")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--threshold", type=float, default=70.0)
    p.add_argument("--max-positions", type=int, default=12)
    p.add_argument("--candidates", type=int, default=80,
                   help="random search candidates (ignored for sleeves)")
    p.add_argument("--strategy", choices=["random", "sleeves"], default="random")
    p.add_argument("--objective", choices=["sharpe", "sortino", "calmar", "cagr"],
                   default="sharpe")
    p.add_argument("--walk-forward", action="store_true",
                   help="rolling 3y train / 1y test windows")
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR / "optimize"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate candidate weights ───────────────────────────────────
    if args.strategy == "sleeves":
        candidates = _sleeve_weights(ACTIVE_FACTORS)
        log.info("Strategy=sleeves: testing %d single-factor configurations",
                 len(candidates))
    else:
        candidates = _dirichlet_weights(args.candidates, ACTIVE_FACTORS, seed=args.seed)
        log.info("Strategy=random: %d Dirichlet-sampled candidates", len(candidates))

    # ── Load data once for the full span ─────────────────────────────
    symbols = _resolve_universe(args.universe)
    data = load_universe(symbols, args.start, args.end, max_workers=args.max_workers)
    if not data:
        log.error("No data loaded; aborting")
        return 1
    log.info("Loaded %d/%d symbols", len(data), len(symbols))

    all_dates = trading_dates(data, args.start, args.end)
    if len(all_dates) == 0:
        log.error("No trading dates")
        return 1

    # ── Single split or walk-forward ─────────────────────────────────
    if args.walk_forward:
        windows = _walk_forward_windows(args.start, args.end,
                                         args.train_years, args.test_years)
        if not windows:
            log.error("Span too short for %dy/%dy walk-forward",
                      args.train_years, args.test_years)
            return 1
        log.info("Walk-forward: %d windows", len(windows))
    else:
        # Use 70%/30% split
        split_idx = int(len(all_dates) * 0.7)
        ts = all_dates[0].strftime("%Y-%m-%d")
        te = all_dates[split_idx].strftime("%Y-%m-%d")
        vs = all_dates[split_idx + 1].strftime("%Y-%m-%d") if split_idx + 1 < len(all_dates) else te
        ve = all_dates[-1].strftime("%Y-%m-%d")
        windows = [(ts, te, vs, ve)]
        log.info("Single split: train=[%s..%s] test=[%s..%s]", ts, te, vs, ve)

    # ── For each window, score every candidate on train; record OOS ──
    fold_records: list[dict] = []
    for fold_idx, (ts, te, vs, ve) in enumerate(windows):
        train_dates = _slice_dates(all_dates, ts, te)
        test_dates = _slice_dates(all_dates, vs, ve)
        log.info("Fold %d: train %s..%s (%d days), test %s..%s (%d days)",
                 fold_idx + 1, ts, te, len(train_dates), vs, ve, len(test_dates))

        # 1) score every candidate on the TRAIN window
        train_results = []
        for ci, w in enumerate(candidates):
            stats = _run_backtest(data, train_dates, w,
                                  args.capital, args.threshold, args.max_positions)
            score = _objective_value(stats, args.objective)
            train_results.append((ci, score, stats))
            if (ci + 1) % max(1, len(candidates) // 10) == 0:
                log.info("  train: %d/%d candidates scored",
                         ci + 1, len(candidates))

        # 2) pick best on TRAIN, evaluate on TEST
        train_results.sort(key=lambda x: x[1], reverse=True)
        best_ci, best_train_score, best_train_stats = train_results[0]
        best_w = candidates[best_ci]

        test_stats = _run_backtest(data, test_dates, best_w,
                                   args.capital, args.threshold, args.max_positions)
        test_score = _objective_value(test_stats, args.objective)

        fold_records.append({
            "fold": fold_idx + 1, "train": f"{ts}..{te}", "test": f"{vs}..{ve}",
            "best_candidate": best_ci,
            "train_score": round(best_train_score, 3),
            "test_score": round(test_score, 3),
            "test_cagr": round(test_stats["cagr"], 2),
            "test_sharpe": round(test_stats["sharpe"], 3),
            "test_max_dd": round(test_stats["max_dd"], 2),
            "test_trades": test_stats["trades"],
            **{f"w_{k}": round(v, 3) for k, v in best_w.items()
               if k in ACTIVE_FACTORS},
        })

        # Also keep top-10 by-train table for diagnostics
        top10 = []
        for ci, s, st in train_results[:10]:
            row = {"candidate": ci, "train_score": round(s, 3),
                   "trades": st["trades"], "cagr": round(st["cagr"], 2),
                   "max_dd": round(st["max_dd"], 2)}
            row.update({f"w_{k}": round(candidates[ci][k], 3)
                        for k in ACTIVE_FACTORS})
            top10.append(row)
        pd.DataFrame(top10).to_csv(
            output_dir / f"fold{fold_idx + 1}_top10_train.csv", index=False)

    # ── Save & summarize ─────────────────────────────────────────────
    df = pd.DataFrame(fold_records)
    out_csv = output_dir / "walk_forward_results.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"WALK-FORWARD RESULTS (objective: {args.objective})")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Average best-weights across folds (the most defensible "deploy" choice)
    if len(df) > 0:
        avg_weights = {f: float(df[f"w_{f}"].mean()) for f in ACTIVE_FACTORS}
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {k: round(v / total, 3) for k, v in avg_weights.items()}
        print("Mean OOS test score:   ", round(df["test_score"].mean(), 3))
        print("Median OOS test score: ", round(df["test_score"].median(), 3))
        print()
        print("Suggested SCORE_WEIGHTS (mean of per-fold winners, renormalized):")
        for k, v in avg_weights.items():
            print(f"    \"{k}\":\t{v:.3f},")
        print()
    print(f"Detailed results: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
