"""Historical (point-in-time) runner.

Pretend it's a past date. Generate suggestions using ONLY data that would
have been available then, then look forward to actual prices at +1 week,
+1 month and +3 months to validate the picks.

Forecast component uses Prophet (no TimesFM).

Usage:
    python historical_runner.py --date 2024-01-15 --mode russell1000 --top 10
    python historical_runner.py --date 2023-06-01 --mode nifty500 --top 15

Defaults:
    --date  : today (acts like main.py with no forward returns)
    --mode  : russell1000
    --top   : 10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Force prophet for the forecast factor BEFORE importing analysis modules
os.environ["FORECASTER"] = "prophet"

from data_sources.universe import (  # noqa: E402
    broad_universe,
    russell1000_tickers,
    nifty500_tickers,
    nse_all_tickers,
)
from data_sources.yahoo import fetch_many, TickerData  # noqa: E402
from analysis.composite import analyze_batch  # noqa: E402
from config import WATCHLIST  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("historical")

REPORT_DIR = Path("reports/historical")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
def _slice_history(td: TickerData, asof: pd.Timestamp) -> TickerData:
    """Return a copy of `td` with history truncated to <= asof.

    Forward bars stay only on the original object so we can look them up
    later for actual return computation.
    """
    snap = copy(td)
    if td.history is None or td.history.empty:
        snap.history = pd.DataFrame()
        return snap
    hist = td.history
    # Normalize tz so we can compare cleanly
    idx = hist.index
    if getattr(idx, "tz", None) is not None:
        cutoff = asof.tz_localize(idx.tz) if asof.tz is None else asof
    else:
        cutoff = asof.tz_localize(None) if asof.tz is not None else asof
    snap.history = hist.loc[:cutoff].copy()
    # Strip news that may carry post-asof items (yahoo news has no date filter
    # we can apply safely; safest is to drop news entirely for backtests)
    snap.news = []
    # Options are forward-looking by nature — drop
    snap.options_summary = None
    return snap


def _forward_price(td: TickerData, asof: pd.Timestamp, days: int) -> tuple[pd.Timestamp | None, float | None]:
    """First trading day on/after asof + `days` calendar days. Returns (date, close)."""
    if td.history is None or td.history.empty:
        return None, None
    target = asof + pd.Timedelta(days=days)
    idx = td.history.index
    if getattr(idx, "tz", None) is not None and target.tz is None:
        target = target.tz_localize(idx.tz)
    fwd = td.history.loc[td.history.index >= target]
    if fwd.empty:
        return None, None
    row = fwd.iloc[0]
    return fwd.index[0], float(row["Close"])


def _entry_price(td: TickerData, asof: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    """Last close on/before asof."""
    if td.history is None or td.history.empty:
        return None, None
    idx = td.history.index
    if getattr(idx, "tz", None) is not None and asof.tz is None:
        target = asof.tz_localize(idx.tz)
    else:
        target = asof
    past = td.history.loc[td.history.index <= target]
    if past.empty:
        return None, None
    return past.index[-1], float(past.iloc[-1]["Close"])


# ────────────────────────────────────────────────────────────────────
def run(asof_str: str | None, mode: str, top_n: int) -> Path:
    asof = pd.Timestamp(asof_str) if asof_str else pd.Timestamp.now().normalize()
    today = pd.Timestamp.now().normalize()

    if asof > today:
        raise SystemExit(f"--date {asof.date()} is in the future")

    # Universe
    if mode == "broad":
        symbols = broad_universe()
    elif mode == "russell1000":
        symbols = russell1000_tickers()
    elif mode == "nifty500":
        symbols = nifty500_tickers()
    elif mode == "niftytotal":
        symbols = nse_all_tickers()
    else:
        symbols = WATCHLIST
    log.info("Universe: %d tickers (mode=%s)", len(symbols), mode)

    # Need ~2y of price history BEFORE asof for indicators + Prophet seasonality,
    # plus enough AFTER asof for the longest forward window (3 months ≈ 95 days).
    days_before = max((today - asof).days + 365 * 2, 365 * 2)
    period = f"{days_before + 200}d"
    log.info("Fetching market data (period=%s) ...", period)
    data = fetch_many(symbols, period=period)
    log.info("Fetched %d/%d tickers", len(data), len(symbols))

    # Drop tickers without enough pre-asof history (need at least ~120 trading
    # days for technicals/momentum, and a current price > 0).
    snapshots: dict[str, TickerData] = {}
    originals: dict[str, TickerData] = {}
    for sym, td in data.items():
        snap = _slice_history(td, asof)
        if snap.history is None or len(snap.history) < 120:
            continue
        snapshots[sym] = snap
        originals[sym] = td
    log.info("Tickers with >=120 bars before %s: %d", asof.date(), len(snapshots))

    if not snapshots:
        raise SystemExit("No tickers with enough history before the requested date")

    log.info("Analyzing as of %s with Prophet forecast ...", asof.date())
    reports = analyze_batch(list(snapshots.values()))
    reports = [r for r in reports if r.composite_score > 0]
    reports.sort(key=lambda r: r.adjusted_score, reverse=True)
    picks = reports[:top_n]
    log.info("Top %d picks selected", len(picks))

    # Forward returns
    rows = []
    for r in tqdm(picks, desc="Forward returns"):
        td = originals.get(r.symbol)
        if td is None:
            continue
        ed, ep = _entry_price(td, asof)
        if ep is None or ep <= 0:
            continue
        d_1w, p_1w = _forward_price(td, asof, 7)
        d_1m, p_1m = _forward_price(td, asof, 30)
        d_3m, p_3m = _forward_price(td, asof, 90)
        rows.append({
            "Symbol": r.symbol,
            "Name": r.name[:35],
            "Mkt": r.market,
            "Sector": (r.sector or "")[:18],
            "Score": round(r.adjusted_score, 1),
            "Verdict": r.verdict,
            "EntryDate": ed.date().isoformat() if ed is not None else "",
            "EntryPrice": round(ep, 2),
            "P_1W": round(p_1w, 2) if p_1w is not None else None,
            "Ret_1W%": round((p_1w / ep - 1) * 100, 2) if p_1w else None,
            "P_1M": round(p_1m, 2) if p_1m is not None else None,
            "Ret_1M%": round((p_1m / ep - 1) * 100, 2) if p_1m else None,
            "P_3M": round(p_3m, 2) if p_3m is not None else None,
            "Ret_3M%": round((p_3m / ep - 1) * 100, 2) if p_3m else None,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No picks survived forward-return computation")

    # Summary stats per horizon
    def _hit_stats(col: str) -> dict:
        s = df[col].dropna()
        if s.empty:
            return {"n": 0, "avg": None, "median": None, "winrate": None}
        return {
            "n": int(len(s)),
            "avg": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "winrate": round(float((s > 0).mean() * 100), 1),
        }

    stats = {h: _hit_stats(f"Ret_{h}%") for h in ("1W", "1M", "3M")}

    # Write report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPORT_DIR / f"historical_{asof.date()}_{mode}_{ts}"
    md_path = base.with_suffix(".md")
    csv_path = base.with_suffix(".csv")

    df.to_csv(csv_path, index=False)

    L: list[str] = []
    L.append(f"# Historical Suggestion Report — As Of {asof.date()}\n")
    L.append(f"**Mode:** {mode}    **Universe:** {len(snapshots)} tickers analyzed    "
             f"**Top picks:** {len(df)}    **Forecast model:** Prophet\n")
    L.append("_Picks generated using only data available as of the as-of date "
             "(no look-ahead). Forward returns measured against actual prices.\n")

    L.append("\n## 📊 Forward-Return Summary\n")
    L.append("| Horizon | N | Avg Return % | Median Return % | Win Rate % |")
    L.append("|---|---:|---:|---:|---:|")
    for h, s in stats.items():
        L.append(f"| +{h} | {s['n']} | "
                 f"{s['avg'] if s['avg'] is not None else '—'} | "
                 f"{s['median'] if s['median'] is not None else '—'} | "
                 f"{s['winrate'] if s['winrate'] is not None else '—'} |")

    L.append("\n## 🎯 Picks & Forward Performance\n")
    L.append("| Symbol | Mkt | Sector | Score | Verdict | Entry | "
             "+1W $ | +1W % | +1M $ | +1M % | +3M $ | +3M % |")
    L.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        def fmt(v):
            return "—" if v is None or pd.isna(v) else v
        L.append(
            f"| `{r['Symbol']}` | {r['Mkt']} | {r['Sector']} | {r['Score']} | "
            f"{r['Verdict']} | {fmt(r['EntryPrice'])} | "
            f"{fmt(r['P_1W'])} | {fmt(r['Ret_1W%'])} | "
            f"{fmt(r['P_1M'])} | {fmt(r['Ret_1M%'])} | "
            f"{fmt(r['P_3M'])} | {fmt(r['Ret_3M%'])} |"
        )

    L.append(f"\n_CSV: `{csv_path.name}`_")
    md_path.write_text("\n".join(L), encoding="utf-8")
    log.info("Wrote %s", md_path)
    log.info("Wrote %s", csv_path)

    # Console summary
    print("\n" + "=" * 78)
    print(f"HISTORICAL RUN — As Of {asof.date()} — Mode {mode} — Top {len(df)}")
    print("=" * 78)
    for h, s in stats.items():
        if s["n"]:
            print(f"  +{h:<3}  n={s['n']:>3}   avg {s['avg']:+6.2f}%   "
                  f"median {s['median']:+6.2f}%   winrate {s['winrate']:>5.1f}%")
        else:
            print(f"  +{h:<3}  n=0  (insufficient forward data)")
    print(f"\n  Markdown : {md_path}")
    print(f"  CSV      : {csv_path}")
    print("=" * 78)
    return md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None,
                        help="As-of date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--mode",
                        choices=["watchlist", "broad", "russell1000",
                                 "nifty500", "niftytotal"],
                        default="russell1000")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    try:
        run(args.date, args.mode, args.top)
    except KeyboardInterrupt:
        sys.exit(1)
