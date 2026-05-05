"""Paper Trading CLI — user-friendly interface for Alpaca paper trading.

Quick start:
  1. Sign up free at https://alpaca.markets (5 minutes, no SSN, $100k paper cash)
  2. Generate API keys → Paper Trading → API Keys
  3. Add to .env:
        ALPACA_API_KEY=PK...
        ALPACA_SECRET_KEY=...
        ALPACA_PAPER=true

Commands:
  python paper_cli.py setup              # verify keys + show account
  python paper_cli.py status             # live account + positions snapshot
  python paper_cli.py sync               # pull broker positions → portfolio.json
  python paper_cli.py preview            # dry-run today's signals (no orders sent)
  python paper_cli.py trade              # ⚠️  ACTUALLY submit orders (paper money)
  python paper_cli.py close SYMBOL       # market-close a position
  python paper_cli.py cancel [--symbol]  # cancel open orders
  python paper_cli.py history            # show recent broker orders

All commands are READ-ONLY except `trade` and `close` and `cancel`.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("paper")


# ── Color helpers (graceful fallback if no TTY) ──────────────────────────────
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")


def c(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def green(s): return c(str(s), "32")
def red(s):   return c(str(s), "31")
def yellow(s): return c(str(s), "33")
def cyan(s):  return c(str(s), "36")
def bold(s):  return c(str(s), "1")


def _money(x: float, currency: str = "$") -> str:
    return f"{currency}{x:,.2f}"


def _pct(x: float) -> str:
    s = f"{x:+.2f}%"
    return green(s) if x >= 0 else red(s)


# ── Broker bootstrap ─────────────────────────────────────────────────────────
def _broker():
    """Build AlpacaBroker with friendly errors."""
    try:
        from brokers.alpaca_broker import AlpacaBroker
    except RuntimeError as e:
        print(red("❌ ") + str(e))
        sys.exit(1)
    try:
        return AlpacaBroker()
    except RuntimeError as e:
        print(red("❌ ") + str(e))
        print()
        print(yellow("💡 To fix:"))
        print("   1. Sign up at https://alpaca.markets (free, 5 min)")
        print("   2. Go to Paper Trading → API Keys → Generate")
        print("   3. Add to .env:")
        print("        ALPACA_API_KEY=PK...")
        print("        ALPACA_SECRET_KEY=...")
        print("        ALPACA_PAPER=true")
        sys.exit(1)


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_setup(args) -> None:
    print(bold("🔧 Alpaca Setup Check\n"))
    broker = _broker()
    try:
        a = broker.account()
    except Exception as e:
        print(red(f"❌ Could not reach Alpaca: {e}"))
        sys.exit(1)

    mode = green("PAPER") if a.is_paper else red("LIVE ⚠️  REAL MONEY")
    print(f"   Mode:           {mode}")
    print(f"   Currency:       {a.currency}")
    print(f"   Cash:           {_money(a.cash)}")
    print(f"   Equity:         {_money(a.equity)}")
    print(f"   Buying power:   {_money(a.buying_power)}")
    print(f"   Day trades:     {a.daytrade_count}/3   (PDT: {a.pattern_day_trader})")
    print(f"   Market open:    {green('YES') if broker.is_market_open() else yellow('NO (afterhours/weekend)')}")
    print()
    print(green("✅ Connection OK. Ready for paper trading."))
    print()
    print(cyan("Next steps:"))
    print(f"   • {bold('python paper_cli.py status')}    — see live positions")
    print(f"   • {bold('python paper_cli.py preview')}   — see what today's run would do (no orders)")
    print(f"   • {bold('python paper_cli.py trade')}     — submit orders for real (paper $$)")


def cmd_status(args) -> None:
    broker = _broker()
    a = broker.account()
    positions = broker.positions()

    print(bold(f"\n💼 Alpaca Account — {datetime.now():%Y-%m-%d %H:%M}"))
    mode_tag = green("[PAPER]") if a.is_paper else red("[LIVE]")
    print(f"   {mode_tag}  Cash {_money(a.cash)}  |  Equity {_money(a.equity)}  "
          f"|  Buying Power {_money(a.buying_power)}")
    print(f"   Open positions: {len(positions)}  |  "
          f"Market: {green('OPEN') if broker.is_market_open() else yellow('CLOSED')}")

    if not positions:
        print(yellow("\n   (no open positions)"))
        return

    print()
    header = (f"   {'Symbol':<8} {'Qty':>10} {'Entry':>10} {'Last':>10} "
              f"{'Mkt Val':>12} {'P&L $':>12} {'P&L %':>10}")
    print(bold(header))
    print("   " + "-" * (len(header) - 3))
    total_pnl = 0.0
    for p in sorted(positions, key=lambda x: -x.unrealized_pnl):
        total_pnl += p.unrealized_pnl
        print(f"   {p.symbol:<8} {p.qty:>10.4f} {p.avg_entry_price:>10.2f} "
              f"{p.current_price:>10.2f} {_money(p.market_value):>12} "
              f"{(green if p.unrealized_pnl>=0 else red)(f'{p.unrealized_pnl:>+11.2f}')} "
              f"{_pct(p.unrealized_pnl_pct):>10}")
    print("   " + "-" * (len(header) - 3))
    print(f"   {bold('Total unrealized P&L'):<40} "
          f"{(green if total_pnl>=0 else red)(_money(total_pnl)):>20}")


def cmd_sync(args) -> None:
    from factories import build_portfolio_service
    from paper_trader import PaperTrader, PaperTraderConfig

    broker = _broker()
    svc = build_portfolio_service()
    pt = PaperTrader(broker, svc, PaperTraderConfig(dry_run=True))
    n = pt.sync()
    print(green(f"✅ Synced {n} broker position(s) into portfolio.json"))
    if n > 0:
        print(cyan("   Run 'python portfolio_cli.py status' to verify."))


def cmd_preview(args) -> None:
    """Dry-run today's signals — show what trade would do, no orders sent."""
    print(bold("🔮 Preview mode — no orders will be submitted\n"))
    _run_trader(dry_run=True, args=args)


def cmd_trade(args) -> None:
    """Real submission to Alpaca paper account."""
    if not args.yes:
        broker = _broker()
        a = broker.account()
        mode = "PAPER" if a.is_paper else red("LIVE — REAL MONEY")
        print(yellow(f"\n⚠️  About to submit orders to Alpaca [{mode}]"))
        print(f"   Account equity: {_money(a.equity)}")
        ans = input("   Type 'yes' to proceed: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            return
    _run_trader(dry_run=False, args=args)


def _run_trader(dry_run: bool, args) -> None:
    """Shared driver for preview + trade."""
    from factories import build_portfolio_service
    from paper_trader import PaperTrader, PaperTraderConfig

    broker = _broker()
    svc = build_portfolio_service()

    # Load picks: prefer reusing daily_runner's pipeline; if not invoked from there,
    # try latest cached picks file.
    picks, exit_signals = _load_today_signals(svc, broker)
    if not picks and not exit_signals:
        print(yellow("⚠️  No picks or exit signals available."))
        print("   Run the daily report first:  python daily_runner.py")
        return

    cfg = PaperTraderConfig(
        dry_run=dry_run,
        require_market_open=not args.force,
        max_orders_per_run=args.max_orders,
        use_trailing_stop=args.trailing,
        max_position_pct=args.max_pos / 100.0,
        min_score_to_buy=args.min_score,
    )
    pt = PaperTrader(broker, svc, cfg)
    result = pt.run(picks=picks, exit_signals=exit_signals)

    print()
    print(bold("📊 Results"))
    print(f"   Mode:            {'DRY-RUN' if result.dry_run else green('LIVE submission')}")
    print(f"   Market open:     {green('YES') if result.market_open else yellow('NO')}")
    print(f"   Account equity:  {_money(result.account_equity)}")
    print(f"   Open positions:  {result.open_positions}")
    print()
    _print_orders("EXITS", result.exits)
    _print_orders("BUYS", result.new_buys)
    _print_orders("PROTECTIVE STOPS", result.new_stops)
    if result.skipped:
        print(yellow(f"\n   Skipped ({len(result.skipped)}):"))
        for s in result.skipped[:20]:
            print(f"     • {s}")


def _print_orders(label: str, orders: list) -> None:
    if not orders:
        return
    print(bold(f"   {label}: {len(orders)}"))
    for o in orders:
        status_color = green if o.status in ("accepted", "filled", "new", "dry_run") else red
        print(f"     {status_color(o.status):<20} id={o.broker_order_id[:16]:<18} "
              f"{('(' + o.error + ')') if o.error else ''}")


def _load_today_signals(svc, broker) -> tuple[list, list]:
    """Generate today's picks + exit signals using existing daily_runner pipeline.

    For now we lazily import daily_runner internals to avoid a heavy startup.
    """
    try:
        from daily_runner import _generate_signals_for_paper_trading
        return _generate_signals_for_paper_trading(svc, broker)
    except (ImportError, AttributeError):
        log.warning("daily_runner._generate_signals_for_paper_trading not found; "
                    "no picks generated. Run daily_runner.py first.")
        return [], []


def cmd_close(args) -> None:
    broker = _broker()
    sym = args.symbol.upper()
    pos = broker.get_position(sym)
    if pos is None:
        print(red(f"No open position for {sym}"))
        return
    print(f"Closing {sym}: qty={pos.qty} @ ~{pos.current_price:.2f} "
          f"(P&L {_pct(pos.unrealized_pnl_pct)})")
    if not args.yes:
        if input("Confirm 'yes': ").strip().lower() != "yes":
            print("Aborted."); return
    result = broker.close_position(sym)
    color = green if result.status not in ("rejected",) else red
    print(color(f"  {result.status}  id={result.broker_order_id}  {result.error}"))


def cmd_cancel(args) -> None:
    broker = _broker()
    n = broker.cancel_open_orders(args.symbol)
    print(green(f"✅ Cancelled {n} open order(s)"))


def cmd_history(args) -> None:
    broker = _broker()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=args.limit)
        orders = broker._trading.get_orders(filter=req)
    except Exception as e:
        print(red(f"Failed to fetch history: {e}"))
        return
    if not orders:
        print(yellow("No order history."))
        return
    print(bold(f"\n📜 Last {len(orders)} orders\n"))
    print(f"   {'Date':<20} {'Symbol':<8} {'Side':<6} {'Qty':>8} {'Type':<8} "
          f"{'Status':<12} {'Filled @':>10}")
    print("   " + "-" * 86)
    for o in orders:
        ts = (o.submitted_at or o.created_at).strftime("%Y-%m-%d %H:%M") if o.submitted_at or o.created_at else ""
        fill = float(o.filled_avg_price) if o.filled_avg_price else 0
        side_color = green if str(o.side).lower() == "buy" else red
        print(f"   {ts:<20} {o.symbol:<8} {side_color(str(o.side).lower()):<15} "
              f"{float(o.qty):>8.2f} {str(o.order_type):<8} {str(o.status):<12} "
              f"{fill:>10.2f}")


# ── argparse plumbing ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Alpaca paper trading CLI — see PAPER_TRADING.md for details",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="Verify Alpaca connection + show account")
    sub.add_parser("status", help="Live account + positions snapshot")
    sub.add_parser("sync", help="Pull broker positions into portfolio.json")

    for name, help_text in [("preview", "Dry-run today's signals (no orders)"),
                            ("trade", "⚠️  Submit orders for real (paper money)")]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--yes", action="store_true", help="skip confirmation")
        sp.add_argument("--force", action="store_true", help="submit even if market closed")
        sp.add_argument("--trailing", action="store_true",
                        help="use Alpaca trailing-stop instead of fixed STOP")
        sp.add_argument("--max-orders", type=int, default=20,
                        help="safety brake on total orders per run (default 20)")
        sp.add_argument("--max-pos", type=float, default=10.0,
                        help="max %% of equity per position (default 10)")
        sp.add_argument("--min-score", type=float, default=65.0,
                        help="minimum score to BUY (default 65)")

    sp = sub.add_parser("close", help="Market-close a position")
    sp.add_argument("symbol")
    sp.add_argument("--yes", action="store_true")

    sp = sub.add_parser("cancel", help="Cancel open orders")
    sp.add_argument("--symbol", default=None)

    sp = sub.add_parser("history", help="Recent broker orders")
    sp.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    handler = {
        "setup": cmd_setup, "status": cmd_status, "sync": cmd_sync,
        "preview": cmd_preview, "trade": cmd_trade,
        "close": cmd_close, "cancel": cmd_cancel, "history": cmd_history,
    }[args.cmd]
    handler(args)


if __name__ == "__main__":
    main()
