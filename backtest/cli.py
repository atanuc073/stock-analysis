"""Backtest CLI entry point.

Examples:
  python -m backtest.cli --start 2019-01-01 --end 2024-12-31
  python -m backtest.cli --start 2020-01-01 --end 2024-12-31 \
      --capital 1000000 --universe india --threshold 70
  python -m backtest.cli --start 2018-01-01 --end 2024-12-31 \
      --universe watchlist --output-dir reports/backtest
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import REPORTS_DIR, WATCHLIST, WATCHLIST_INDIA, WATCHLIST_US

from .data_loader import load_universe, trading_dates
from .engine import BacktestConfig, BacktestEngine
from .results import compute as compute_stats
from .reporter import write_excel, write_markdown, write_chart
from . import regime as regime_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("backtest.cli")


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
    # Treat as comma-separated list of tickers
    return [s.strip() for s in name.split(",") if s.strip()]


def _load_benchmark(symbol: str, start: str, end: str) -> pd.Series:
    """Load benchmark close prices via yfinance."""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df["Close"]
    except Exception as e:
        log.warning("Failed to load benchmark %s: %s", symbol, e)
        return pd.Series(dtype=float)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Backtest the position-investing strategy.")
    p.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--capital", type=float, default=1_000_000.0,
                   help="Initial capital (default: ₹10,00,000)")
    p.add_argument("--universe", default="watchlist",
                   help="watchlist | india | us | broad | RELIANCE.NS,TCS.NS,...")
    p.add_argument("--threshold", type=float, default=60.0,
                   help="Min composite score to buy (default 60)")
    p.add_argument("--rebalance-days", type=int, default=5,
                   help="Rebalance every N trading days (default 5 = weekly)")
    p.add_argument("--max-positions", type=int, default=12)
    p.add_argument("--include-forecast", action="store_true",
                   help="Include forecast component (slower)")
    p.add_argument("--legacy-weights", action="store_true",
                   help="Use legacy weight redistribution (drop sentiment/options/forecast "
                        "and reallocate to technical+momentum). Default is live-equivalent "
                        "weights with neutral 50 for missing components.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default reports/backtest/)")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Concurrent fetches (default 8)")
    p.add_argument("--benchmark-india", default="^NSEI")
    p.add_argument("--benchmark-us", default="^GSPC")
    p.add_argument("--benchmark-india-broad", default="^CRSLDX",
                   help="Broad IN benchmark (default Nifty 500: ^CRSLDX)")
    p.add_argument("--benchmark-us-small", default="^RUT",
                   help="US smallcap benchmark (default Russell 2000: ^RUT)")
    p.add_argument("--no-regime", action="store_true",
                   help="Disable regime-aware sizing (default: enabled)")
    p.add_argument("--regime-skip-below", default="BEAR",
                   choices=["BEAR", "CAUTIOUS", "NEUTRAL", "NEUTRAL_BULL", "BULL"],
                   help="Skip new entries when regime label ≤ this (default BEAR)")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load universe
    symbols = _resolve_universe(args.universe)
    log.info("Universe: %d symbols (%s)", len(symbols), args.universe)

    data = load_universe(symbols, args.start, args.end, max_workers=args.max_workers)
    if not data:
        log.error("No data loaded; aborting")
        return 1
    log.info("Loaded %d/%d symbols with sufficient data", len(data), len(symbols))

    # 2) Trading dates
    dates = trading_dates(data, args.start, args.end)
    if len(dates) == 0:
        log.error("No overlapping trading dates; aborting")
        return 1

    # 3) Run engine
    cfg = BacktestConfig(
        initial_capital=args.capital,
        rebalance_freq_days=args.rebalance_days,
        min_score=args.threshold,
        max_positions=args.max_positions,
        include_forecast=args.include_forecast,
        live_weights=not args.legacy_weights,
        use_regime=not args.no_regime,
        regime_skip_below=args.regime_skip_below,
    )

    # Pre-load benchmarks (also used as regime input)
    has_in = any(hd.is_indian for hd in data.values())
    has_us = any(not hd.is_indian for hd in data.values())
    bench_in = _load_benchmark(args.benchmark_india, args.start, args.end) if has_in else pd.Series(dtype=float)
    bench_us = _load_benchmark(args.benchmark_us, args.start, args.end) if has_us else pd.Series(dtype=float)
    bench_in_broad = _load_benchmark(args.benchmark_india_broad, args.start, args.end) if has_in else pd.Series(dtype=float)
    bench_us_small = _load_benchmark(args.benchmark_us_small, args.start, args.end) if has_us else pd.Series(dtype=float)

    regime_data: dict[str, dict[str, pd.Series]] = {}
    if cfg.use_regime:
        if has_in:
            regime_data["IN"] = {
                "index": bench_in,
                "vix": regime_mod.load_benchmark("^INDIAVIX", args.start, args.end),
            }
        if has_us:
            regime_data["US"] = {
                "index": bench_us,
                "vix": regime_mod.load_benchmark("^VIX", args.start, args.end),
            }
        log.info("Regime-aware sizing ENABLED (skip below: %s)", args.regime_skip_below)
    else:
        log.info("Regime-aware sizing DISABLED")

    engine = BacktestEngine(data=data, config=cfg, regime_data=regime_data or None)
    result = engine.run(dates)

    # 4) Stats
    stats = compute_stats(result)

    # 5) Benchmarks for chart/report
    benchmarks: dict[str, pd.Series] = {}
    if has_in and not bench_in.empty:
        benchmarks["Nifty 50"] = bench_in
    if has_in and not bench_in_broad.empty:
        benchmarks["Nifty 500"] = bench_in_broad
    if has_us and not bench_us.empty:
        benchmarks["S&P 500"] = bench_us
    if has_us and not bench_us_small.empty:
        benchmarks["Russell 2000"] = bench_us_small

    # 6) Reports
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"backtest_{args.start}_{args.end}_{stamp}"
    excel_path = output_dir / f"{base}.xlsx"
    md_path = output_dir / f"{base}.md"
    png_path = output_dir / f"{base}.png"

    write_excel(result, stats, benchmarks, excel_path)
    write_markdown(result, stats, md_path)
    write_chart(result, benchmarks, png_path)

    # 7) Console summary
    print("\n" + "=" * 70)
    print(f"BACKTEST COMPLETE — {result.start} → {result.end}")
    print("=" * 70)
    print(f"Initial Capital   : ₹{stats.initial_capital:,.0f}")
    print(f"Final Equity      : ₹{stats.final_equity:,.0f}")
    print(f"Total Return      : {stats.total_return_pct:+.2f}%")
    print(f"CAGR              : {stats.cagr_pct:+.2f}%")
    print(f"Max Drawdown      : {stats.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio      : {stats.sharpe_ratio:.2f}")
    print(f"Win Rate          : {stats.win_rate_pct:.1f}%  ({stats.closed_trades} trades)")
    print(f"Expectancy/Trade  : {stats.expectancy_pct:+.2f}%")
    print(f"Profit Factor     : {stats.profit_factor:.2f}")
    print(f"Avg Hold (days)   : {stats.avg_holding_days:.0f}")
    print("\nReports:")
    print(f"  Excel    : {excel_path}")
    print(f"  Markdown : {md_path}")
    print(f"  Chart    : {png_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
