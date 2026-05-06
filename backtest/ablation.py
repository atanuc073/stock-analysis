"""Ablation backtest — run the full backtest multiple times, each time
zeroing out ONE scoring component, and compare the resulting CAGR / Sharpe /
win rate. Tells you which factors actually pay vs. which are noise.

Usage:
  python -m backtest.ablation --start 2020-01-01 --end 2025-01-01 \
      --universe broad --capital 1000000

Components ablated: technical, fundamental, momentum, sentiment, options, forecast.
The "baseline" run uses the live SCORE_WEIGHTS unchanged; each ablation run
zeros the named component's weight and renormalizes the others to sum to 1.0.
"""
from __future__ import annotations
import argparse
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from config import REPORTS_DIR, WATCHLIST_INDIA, WATCHLIST_US, WATCHLIST

from .data_loader import load_universe, trading_dates
from .engine import BacktestConfig, BacktestEngine
from .results import compute as compute_stats
from . import regime as regime_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("backtest.ablation")


def _resolve_universe(name: str) -> list[str]:
    name = name.lower()
    if name == "watchlist":
        return WATCHLIST
    if name == "india":
        return WATCHLIST_INDIA
    if name == "us":
        return WATCHLIST_US
    if name == "broad":
        try:
            from data_sources.universe import broad_universe
            return broad_universe()
        except Exception as e:
            log.warning("broad universe unavailable (%s); falling back to watchlist", e)
            return WATCHLIST
    return [s.strip() for s in name.split(",") if s.strip()]


def _renorm(weights: dict, drop_key: str) -> dict:
    """Return new weights dict with ``drop_key`` zeroed and others renormalized."""
    w = dict(weights)
    if drop_key not in w:
        return w
    w[drop_key] = 0.0
    total = sum(v for k, v in w.items() if k != "valuation") or 1.0
    # Renormalize the non-valuation weights to sum to (1 - valuation_weight)
    val_w = w.get("valuation", 0.0)
    target = 1.0 - val_w
    scale = target / total if total > 0 else 1.0
    for k in list(w.keys()):
        if k == "valuation":
            continue
        w[k] = w[k] * scale
    return w


def _run_one(label: str, weights: dict, args, data, dates,
             regime_data) -> dict:
    """Run a single backtest with a patched SCORE_WEIGHTS dict."""
    log.info("─" * 60)
    log.info("Ablation run: %s", label)
    log.info("Weights: %s", {k: round(v, 3) for k, v in weights.items()})

    # Monkey-patch the global SCORE_WEIGHTS for this run
    original = deepcopy(config.SCORE_WEIGHTS)
    config.SCORE_WEIGHTS.clear()
    config.SCORE_WEIGHTS.update(weights)
    try:
        cfg = BacktestConfig(
            initial_capital=args.capital,
            rebalance_freq_days=args.rebalance_days,
            min_score=args.threshold,
            max_positions=args.max_positions,
            include_forecast=False,  # forecast is one of the ablated components
            live_weights=True,
            use_regime=not args.no_regime,
            regime_skip_below=args.regime_skip_below,
        )
        engine = BacktestEngine(data=data, config=cfg, regime_data=regime_data or None)
        result = engine.run(dates)
        stats = compute_stats(result)
    finally:
        # Restore
        config.SCORE_WEIGHTS.clear()
        config.SCORE_WEIGHTS.update(original)

    return {
        "Run": label,
        "CAGR_%": round(stats.cagr_pct, 2),
        "TotalRet_%": round(stats.total_return_pct, 2),
        "MaxDD_%": round(stats.max_drawdown_pct, 2),
        "Sharpe": round(stats.sharpe_ratio, 2),
        "Sortino": round(stats.sortino_ratio, 2),
        "Calmar": round(stats.calmar_ratio, 2),
        "WinRate_%": round(stats.win_rate_pct, 2),
        "Trades": stats.closed_trades,
        "Expectancy_%": round(stats.expectancy_pct, 2),
        "ProfitFactor": round(stats.profit_factor, 2),
        "FinalEquity": round(stats.final_equity, 0),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ablation study of SCORE_WEIGHTS components.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--universe", default="watchlist")
    p.add_argument("--threshold", type=float, default=70.0)
    p.add_argument("--rebalance-days", type=int, default=5)
    p.add_argument("--max-positions", type=int, default=12)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-regime", action="store_true")
    p.add_argument("--regime-skip-below", default="BEAR",
                   choices=["BEAR", "CAUTIOUS", "NEUTRAL", "NEUTRAL_BULL", "BULL"])
    p.add_argument("--components", default="technical,fundamental,momentum,sentiment,options,forecast",
                   help="Comma-separated components to ablate (default: all six)")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data ONCE — reused across all ablation runs
    symbols = _resolve_universe(args.universe)
    log.info("Universe: %d symbols (%s)", len(symbols), args.universe)

    data = load_universe(symbols, args.start, args.end, max_workers=args.max_workers)
    if not data:
        log.error("No data loaded; aborting")
        return 1
    log.info("Loaded %d/%d symbols", len(data), len(symbols))

    dates = trading_dates(data, args.start, args.end)
    if len(dates) == 0:
        log.error("No trading dates; aborting")
        return 1

    # Pre-load regime data once
    has_in = any(hd.is_indian for hd in data.values())
    has_us = any(not hd.is_indian for hd in data.values())
    regime_data: dict[str, dict[str, pd.Series]] = {}
    if not args.no_regime:
        if has_in:
            regime_data["IN"] = {
                "index": regime_mod.load_benchmark("^NSEI", args.start, args.end),
                "vix": regime_mod.load_benchmark("^INDIAVIX", args.start, args.end),
            }
        if has_us:
            regime_data["US"] = {
                "index": regime_mod.load_benchmark("^GSPC", args.start, args.end),
                "vix": regime_mod.load_benchmark("^VIX", args.start, args.end),
            }

    runs: list[dict] = []

    # Baseline
    baseline_w = deepcopy(config.SCORE_WEIGHTS)
    runs.append(_run_one("BASELINE (all components)", baseline_w, args,
                         data, dates, regime_data))

    # One ablation per component
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    for comp in components:
        if comp not in baseline_w:
            log.warning("Component '%s' not in SCORE_WEIGHTS; skipping", comp)
            continue
        weights = _renorm(baseline_w, comp)
        runs.append(_run_one(f"NO_{comp.upper()}", weights, args,
                             data, dates, regime_data))

    # Build report
    df = pd.DataFrame(runs)

    baseline_cagr = df.iloc[0]["CAGR_%"]
    df["Δ_CAGR"] = (df["CAGR_%"] - baseline_cagr).round(2)
    df["Δ_Sharpe"] = (df["Sharpe"] - df.iloc[0]["Sharpe"]).round(2)

    print("\n" + "=" * 90)
    print("ABLATION RESULTS — impact of dropping each component")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print("\nReading the table:")
    print("  Δ_CAGR > 0  → dropping the component HELPED   (component was hurting)")
    print("  Δ_CAGR < 0  → dropping the component HURT     (component was paying)")
    print("  Δ_CAGR ≈ 0  → component is NOISE — consider dropping its weight")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"ablation_{args.start}_{args.end}_{stamp}.csv"
    out_md = output_dir / f"ablation_{args.start}_{args.end}_{stamp}.md"
    df.to_csv(out_csv, index=False)

    md = ["# Ablation Study\n",
          f"**Period:** {args.start} → {args.end}  ",
          f"**Universe:** {args.universe} ({len(data)} symbols)  ",
          f"**Threshold:** ≥ {args.threshold}  ",
          f"**Baseline weights:** `{ {k: round(v,3) for k,v in baseline_w.items()} }`\n",
          "## Results\n",
          df.to_markdown(index=False),
          "\n## How to read",
          "- **Δ_CAGR > 0** — dropping the component *helped*; it was hurting performance.",
          "- **Δ_CAGR < 0** — dropping the component *hurt*; it was contributing alpha.",
          "- **Δ_CAGR ≈ 0** — component is noise; consider zeroing its weight in `config.py`."]
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
