"""Portfolio CLI — manage positions from the command line.

Usage:
  python portfolio_cli.py add SYMBOL --price PRICE [--qty Q | --rupees R] [--score 80]
  python portfolio_cli.py status
  python portfolio_cli.py exit SYMBOL --tier 1 --price PRICE [--qty Q]
  python portfolio_cli.py close SYMBOL --price PRICE --reason "..."
  python portfolio_cli.py history [--symbol X]
  python portfolio_cli.py evaluate              # dry-run exit checks
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime

import yfinance as yf

from analysis.indicators import atr, annualized_volatility
from factories import build_portfolio_service, build_sizer
from portfolio.models import ExitSignal, ExitType

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("portfolio")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _fetch_context(symbol: str) -> tuple[float, float, float, str, str]:
    """Returns (price, atr14, ann_vol, sector, market)."""
    t = yf.Ticker(symbol)
    hist = t.history(period="6mo", auto_adjust=True)
    if hist.empty:
        raise SystemExit(f"No history for {symbol}")
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    price = float(hist["Close"].iloc[-1])
    a = atr(hist)
    vol = annualized_volatility(hist)
    sector = info.get("sector", "")
    market = "IN" if symbol.upper().endswith((".NS", ".BO")) else "US"
    return price, a, vol, sector, market


def _live_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    out: dict[str, float] = {}
    try:
        df = yf.download(symbols, period="5d", auto_adjust=True, progress=False, group_by="ticker")
        for s in symbols:
            try:
                if hasattr(df.columns, "levels"):
                    out[s] = float(df[s]["Close"].dropna().iloc[-1])
                else:
                    out[s] = float(df["Close"].dropna().iloc[-1])
            except Exception:
                continue
    except Exception as e:
        log.warning("price fetch failed: %s", e)
    return out


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_add(args) -> None:
    svc = build_portfolio_service()
    sym = args.symbol.upper()
    price, atr14, vol, sector, market = _fetch_context(sym)
    if args.price:
        price = float(args.price)

    if args.qty:
        qty = float(args.qty)
    elif args.rupees:
        qty = float(args.rupees) / price
    else:
        raise SystemExit("provide --qty or --rupees")

    pos = svc.add_position(
        symbol=sym, qty=qty, entry_price=price, atr=atr14,
        sector=sector or args.sector, market=market, score=args.score,
    )
    print(f"✅ Added {sym}")
    print(f"   Qty: {pos.qty_open}  Entry: {price:.2f}  ATR: {atr14:.2f}")
    print(f"   Stop: {pos.stop_price:.2f} ({(pos.stop_price/price-1)*100:+.1f}%)")
    print(f"   T1 (+20%): {price*1.20:.2f}  →  sell 33%, stop→BE")
    print(f"   T2 (+35%): {price*1.35:.2f}  →  sell 33%, stop→+15%")
    print(f"   Sector: {sector or 'unknown'}  Market: {market}")


def cmd_status(args) -> None:
    svc = build_portfolio_service()
    state = svc.state()
    open_pos = state.open_positions()
    if not open_pos:
        print("No open positions.")
        return
    prices = _live_prices([p.symbol for p in open_pos])
    snap = svc.equity_snapshot(prices)

    print(f"\n💼 PORTFOLIO STATUS — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"   Cash: {state.cash:,.2f}  | Invested: {snap.invested:,.2f} "
          f"| MV: {snap.market_value:,.2f}  | Total: {snap.total:,.2f}")
    print(f"   Unrealized: {snap.unrealized_pnl:+,.2f}  "
          f"Realized: {snap.realized_pnl:+,.2f}  Drawdown: {snap.drawdown_pct:+.2f}%\n")

    print(f"   {'Symbol':<14} {'Qty':>9} {'Entry':>10} {'Price':>10} "
          f"{'P&L%':>7} {'Stop':>10} {'Tiers':<8} Sector")
    print("   " + "-" * 95)
    for p in open_pos:
        cp = prices.get(p.symbol, p.entry_price)
        pnl_pct = (cp / p.entry_price - 1) * 100
        triggered = "".join("●" if t.triggered else "○" for t in p.tiers)
        print(f"   {p.symbol:<14} {p.qty_open:>9.4f} {p.entry_price:>10.2f} {cp:>10.2f} "
              f"{pnl_pct:>+6.1f}% {p.stop_price:>10.2f} {triggered:<8} {p.sector[:20]}")


def cmd_exit(args) -> None:
    svc = build_portfolio_service()
    sym = args.symbol.upper()
    pos = svc.find(sym)
    if not pos:
        raise SystemExit(f"No open position for {sym}")
    price = float(args.price) if args.price else _live_prices([sym]).get(sym, pos.entry_price)
    qty = float(args.qty) if args.qty else (pos.qty_original / 3 if args.tier in (1, 2) else pos.qty_open)
    qty = min(qty, pos.qty_open)

    exit_type = {
        1: ExitType.TIER_1, 2: ExitType.TIER_2,
        None: ExitType.MANUAL,
    }.get(args.tier, ExitType.MANUAL)

    new_stop = None
    if exit_type == ExitType.TIER_1:
        new_stop = pos.entry_price            # break-even
    elif exit_type == ExitType.TIER_2:
        new_stop = pos.entry_price * 1.15     # lock +15%

    sig = ExitSignal(
        symbol=sym, exit_type=exit_type, suggested_qty=qty,
        current_price=price, reason=args.reason or f"manual {exit_type.value}",
        new_stop_price=new_stop,
        pnl_pct=(price / pos.entry_price - 1) * 100,
        pnl_abs=(price - pos.entry_price) * qty,
    )
    svc.execute_exit(sig)
    print(f"✅ Sold {qty:.4f} of {sym} @ {price:.2f} ({exit_type.value})")
    print(f"   Realized: {sig.pnl_abs:+,.2f}")


def cmd_close(args) -> None:
    args.tier = None
    args.qty = None
    args.reason = args.reason or "manual full close"
    cmd_exit(args)


def cmd_history(args) -> None:
    svc = build_portfolio_service()
    state = svc.state()
    trades = state.trades
    if args.symbol:
        sym = args.symbol.upper()
        trades = [t for t in trades if t.symbol.upper() == sym]
    if not trades:
        print("No trades.")
        return
    print(f"   {'Time':<20} {'Symbol':<12} {'Action':<5} {'Qty':>9} "
          f"{'Price':>10} {'P&L':>10}  Reason")
    for t in trades[-50:]:
        print(f"   {t.timestamp:<20} {t.symbol:<12} {t.action:<5} {t.qty:>9.4f} "
              f"{t.price:>10.2f} {t.pnl_abs:>+10.2f}  {t.reason[:40]}")


def cmd_evaluate(args) -> None:
    """Dry-run: show what exit signals would fire today."""
    svc = build_portfolio_service()
    state = svc.state()
    open_pos = state.open_positions()
    if not open_pos:
        print("No open positions.")
        return
    prices = _live_prices([p.symbol for p in open_pos])
    signals = svc.evaluate_all(prices)
    if not signals:
        print("✓ No exit signals today.")
        return
    print(f"\n🚨 {len(signals)} EXIT SIGNAL(S):\n")
    for s in signals:
        print(f"  {s.exit_type.value:<14} {s.symbol:<12} "
              f"qty {s.suggested_qty:>9.4f} @ {s.current_price:>9.2f}  "
              f"P&L {s.pnl_pct:+5.1f}%   — {s.reason}")


# ── Argparse ─────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(prog="portfolio")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add"); a.set_defaults(fn=cmd_add)
    a.add_argument("symbol")
    a.add_argument("--qty", type=float)
    a.add_argument("--rupees", type=float)
    a.add_argument("--price", type=float, help="override entry price (default = today's close)")
    a.add_argument("--score", type=float, default=70.0)
    a.add_argument("--sector", default="")

    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)

    e = sub.add_parser("exit"); e.set_defaults(fn=cmd_exit)
    e.add_argument("symbol")
    e.add_argument("--tier", type=int, choices=[1, 2])
    e.add_argument("--price", type=float)
    e.add_argument("--qty", type=float)
    e.add_argument("--reason", default="")

    c = sub.add_parser("close"); c.set_defaults(fn=cmd_close)
    c.add_argument("symbol")
    c.add_argument("--price", type=float)
    c.add_argument("--reason", default="")

    h = sub.add_parser("history"); h.set_defaults(fn=cmd_history)
    h.add_argument("--symbol")

    ev = sub.add_parser("evaluate"); ev.set_defaults(fn=cmd_evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
