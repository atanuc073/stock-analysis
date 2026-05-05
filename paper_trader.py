"""Paper trader — bridges your screener/portfolio engine to Alpaca paper trading.

Workflow each run:
  1. Sync broker positions → portfolio.json (broker is source of truth for fills)
  2. Evaluate exits on existing positions → submit SELL orders
  3. Generate top BUY picks → size them → submit BUY orders + protective stops
  4. Save updated portfolio state

Safety features:
  - Dry-run mode (default ON) prints orders but does not submit
  - Idempotent client_order_id prevents duplicate orders on retry
  - Skips Indian stocks (.NS / .BO) — Alpaca is US-only
  - Max-orders-per-run cap prevents runaway loops
  - Honors market-hours by default (skips submission when market closed)
"""
from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

from brokers.base import (
    Broker, Order, OrderSide, OrderType, TimeInForce, OrderResult,
)
from portfolio.models import (
    Position, PositionStatus, ExitSignal, ExitType, Trade,
)
from portfolio.service import PortfolioService

log = logging.getLogger(__name__)


@dataclass
class PaperTraderConfig:
    dry_run: bool = True                  # default SAFE: print, don't submit
    require_market_open: bool = True      # skip if market closed
    max_orders_per_run: int = 20          # safety brake
    use_trailing_stop: bool = False       # if True, use Alpaca trailing-stop (broker-managed)
    trail_percent: float = 15.0           # only used if use_trailing_stop
    max_position_pct: float = 0.10        # max 10% of equity per position
    cash_floor_pct: float = 0.05          # keep 5% cash buffer
    min_score_to_buy: float = 65.0


@dataclass
class PaperTraderResult:
    market_open: bool
    dry_run: bool
    new_buys: list[OrderResult]
    new_stops: list[OrderResult]
    exits: list[OrderResult]
    skipped: list[str]                    # reasons why some picks were skipped
    account_cash: float = 0.0
    account_equity: float = 0.0
    open_positions: int = 0


class PaperTrader:
    """Stateless orchestrator. Pass in deps; call run()."""

    def __init__(
        self,
        broker: Broker,
        portfolio: PortfolioService,
        config: PaperTraderConfig | None = None,
    ) -> None:
        self.broker = broker
        self.portfolio = portfolio
        self.cfg = config or PaperTraderConfig()

    # ── Public API ───────────────────────────────────────────────────────────
    def sync(self) -> int:
        """Pull broker positions into portfolio.json. Returns count synced.

        This makes the broker the source of truth for fills. We add any
        broker-held position not yet tracked locally; we DO NOT remove local
        positions that broker doesn't have (those represent intent to sell).
        """
        broker_positions = {p.symbol.upper(): p for p in self.broker.positions()}
        state = self.portfolio.state()
        local = {p.symbol.upper(): p for p in state.open_positions()}

        added = 0
        for sym, bp in broker_positions.items():
            if sym in local:
                continue
            try:
                self.portfolio.add_position(
                    symbol=sym,
                    qty=bp.qty,
                    entry_price=bp.avg_entry_price,
                    atr=bp.avg_entry_price * 0.05,   # rough fallback; will refine on next eval
                    sector="",
                    market="US",
                    score=0.0,
                )
                added += 1
                log.info("Synced from broker: %s (qty=%s, entry=%.2f)",
                         sym, bp.qty, bp.avg_entry_price)
            except ValueError:
                pass  # already exists race

        # Warn about local-only positions (they may have been closed at broker)
        only_local = set(local) - set(broker_positions)
        for sym in only_local:
            log.warning("Position %s in portfolio.json but NOT at broker. "
                        "Run reconcile to fix.", sym)
        return added

    def submit_exits(self, exit_signals: list[ExitSignal]) -> list[OrderResult]:
        """Convert ExitSignals to broker SELL orders. Returns submission results."""
        results: list[OrderResult] = []
        for sig in exit_signals[: self.cfg.max_orders_per_run]:
            if sig.suggested_qty <= 0:
                continue
            order = Order(
                symbol=sig.symbol,
                qty=round(sig.suggested_qty, 6),
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=self._client_id("exit", sig.symbol, sig.exit_type.value),
                notes=f"{sig.exit_type.value}: {sig.reason}",
            )
            result = self._submit(order)
            results.append(result)
        return results

    def submit_entries(
        self,
        picks: list,                  # list of objects with .symbol, .market, .score, .price, .atr_value, .sector, .annual_vol
        equity: float,
    ) -> tuple[list[OrderResult], list[OrderResult], list[str]]:
        """Generate BUY market orders + protective STOP orders for new picks.

        Returns: (buy_results, stop_results, skip_reasons)
        """
        buys: list[OrderResult] = []
        stops: list[OrderResult] = []
        skipped: list[str] = []

        # Existing symbols (don't double-buy)
        held = {p.symbol.upper() for p in self.portfolio.state().open_positions()}
        per_position_cap = equity * self.cfg.max_position_pct
        order_budget = self.cfg.max_orders_per_run

        for p in picks:
            if order_budget <= 0:
                skipped.append("order budget exhausted")
                break
            if not self._is_us(p.symbol):
                skipped.append(f"{p.symbol}: not US (Alpaca limitation)")
                continue
            if p.symbol.upper() in held:
                skipped.append(f"{p.symbol}: already held")
                continue
            if p.score < self.cfg.min_score_to_buy:
                skipped.append(f"{p.symbol}: score {p.score:.1f} < {self.cfg.min_score_to_buy}")
                continue
            if p.price <= 0:
                skipped.append(f"{p.symbol}: invalid price")
                continue

            qty = max(round(per_position_cap / p.price, 4), 0.0)
            if qty <= 0:
                skipped.append(f"{p.symbol}: qty rounds to zero")
                continue

            # Submit market BUY
            buy_order = Order(
                symbol=p.symbol,
                qty=qty,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                client_order_id=self._client_id("buy", p.symbol),
                notes=f"score={p.score:.1f}",
            )
            buy_result = self._submit(buy_order)
            buys.append(buy_result)
            order_budget -= 1

            if buy_result.status in ("rejected", ):
                continue

            # Record locally (fill price = current; will reconcile on next run)
            try:
                self.portfolio.add_position(
                    symbol=p.symbol, qty=qty, entry_price=p.price,
                    atr=getattr(p, "atr", None) or getattr(p, "atr_value", p.price * 0.05),
                    sector=getattr(p, "sector", ""),
                    market="US",
                    score=p.score,
                )
            except ValueError:
                pass

            # Submit protective STOP based on the position's stop
            local_pos = self.portfolio.find(p.symbol)
            if local_pos and order_budget > 0:
                if self.cfg.use_trailing_stop:
                    stop_order = Order(
                        symbol=p.symbol, qty=qty, side=OrderSide.SELL,
                        order_type=OrderType.TRAILING_STOP,
                        time_in_force=TimeInForce.GTC,
                        trail_percent=self.cfg.trail_percent,
                        client_order_id=self._client_id("trail", p.symbol),
                    )
                else:
                    stop_order = Order(
                        symbol=p.symbol, qty=qty, side=OrderSide.SELL,
                        order_type=OrderType.STOP,
                        time_in_force=TimeInForce.GTC,
                        stop_price=round(local_pos.stop_price, 2),
                        client_order_id=self._client_id("stop", p.symbol),
                    )
                stop_result = self._submit(stop_order)
                stops.append(stop_result)
                order_budget -= 1

        return buys, stops, skipped

    def run(
        self,
        picks: list,
        exit_signals: list[ExitSignal] | None = None,
    ) -> PaperTraderResult:
        """End-to-end paper trading run.

        picks: top BUY candidates (with .symbol, .price, .score, .atr_value, .sector)
        exit_signals: pre-evaluated exit signals (optional; if None, none submitted)
        """
        market_open = self.broker.is_market_open()
        if self.cfg.require_market_open and not market_open and not self.cfg.dry_run:
            log.warning("Market is closed — skipping submissions.")
            return PaperTraderResult(
                market_open=False, dry_run=self.cfg.dry_run,
                new_buys=[], new_stops=[], exits=[],
                skipped=["market closed"],
            )

        # 1. Sync broker → local
        self.sync()

        # 2. Account snapshot
        account = self.broker.account()

        # 3. Submit exits first (free up cash)
        exit_results = self.submit_exits(exit_signals or [])

        # 4. Submit entries with stops
        buys, stops, skipped = self.submit_entries(picks, account.equity)

        return PaperTraderResult(
            market_open=market_open,
            dry_run=self.cfg.dry_run,
            new_buys=buys,
            new_stops=stops,
            exits=exit_results,
            skipped=skipped,
            account_cash=account.cash,
            account_equity=account.equity,
            open_positions=len(self.broker.positions()),
        )

    # ── Internals ────────────────────────────────────────────────────────────
    def _submit(self, order: Order) -> OrderResult:
        if self.cfg.dry_run:
            log.info("[DRY-RUN] Would submit: %s %s %s @ %s (stop=%s, trail=%s) [%s]",
                     order.side.value, order.qty, order.symbol,
                     order.order_type.value,
                     order.stop_price, order.trail_percent, order.notes)
            return OrderResult(
                broker_order_id=f"dryrun-{order.client_order_id or uuid.uuid4().hex[:8]}",
                client_order_id=order.client_order_id,
                status="dry_run",
            )
        result = self.broker.submit(order)
        log.info("Order %s %s %s -> %s%s",
                 order.side.value, order.qty, order.symbol, result.status,
                 f" ({result.error})" if result.error else "")
        return result

    @staticmethod
    def _is_us(symbol: str) -> bool:
        s = symbol.upper()
        return not (s.endswith(".NS") or s.endswith(".BO"))

    @staticmethod
    def _client_id(*parts: str) -> str:
        """Build a deterministic-per-day client order id for idempotency."""
        today = date.today().isoformat()
        joined = "-".join(p.replace(".", "_") for p in parts)
        return f"copilot-{joined}-{today}"
