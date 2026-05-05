"""Alpaca broker integration — US paper + live trading.

Docs: https://docs.alpaca.markets/
SDK:  https://github.com/alpacahq/alpaca-py

Auth: ALPACA_API_KEY + ALPACA_SECRET_KEY in .env
Mode: ALPACA_PAPER=true (default) → paper.alpaca.markets
      ALPACA_PAPER=false           → live.alpaca.markets ($$$ REAL MONEY)

Limits: US-listed stocks + ETFs only. NO Indian stocks (.NS / .BO).
Free tier: 200 requests/min, IEX market data (delayed ~15min)
"""
from __future__ import annotations
import logging
import os
from typing import Optional

from .base import (
    Broker, Order, OrderResult, OrderSide, OrderType, TimeInForce,
    BrokerPosition, BrokerAccount,
)

log = logging.getLogger(__name__)


def _is_us_symbol(symbol: str) -> bool:
    """Alpaca only handles plain US tickers — no .NS / .BO suffix."""
    s = symbol.upper()
    return not (s.endswith(".NS") or s.endswith(".BO"))


class AlpacaBroker:
    """Concrete Broker for Alpaca Markets (paper or live)."""

    name = "alpaca"

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
    ) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as e:
            raise RuntimeError(
                "alpaca-py not installed. Run:\n"
                "  pip install alpaca-py\n"
                "(use --index-url https://pypi.org/simple if behind corp proxy)"
            ) from e

        api_key = api_key or os.getenv("ALPACA_API_KEY")
        secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in .env (get free keys at https://alpaca.markets)"
            )

        if paper is None:
            paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        self.is_paper = paper

        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)
        log.info("AlpacaBroker initialized (paper=%s)", paper)

    # ── Account ──────────────────────────────────────────────────────────────
    def account(self) -> BrokerAccount:
        a = self._trading.get_account()
        return BrokerAccount(
            cash=float(a.cash),
            equity=float(a.equity),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
            currency=a.currency or "USD",
            is_paper=self.is_paper,
            daytrade_count=int(getattr(a, "daytrade_count", 0) or 0),
            pattern_day_trader=bool(getattr(a, "pattern_day_trader", False)),
        )

    # ── Positions ────────────────────────────────────────────────────────────
    def positions(self) -> list[BrokerPosition]:
        out: list[BrokerPosition] = []
        for p in self._trading.get_all_positions():
            qty = float(p.qty)
            entry = float(p.avg_entry_price)
            cur = float(p.current_price or entry)
            mv = float(p.market_value or qty * cur)
            pnl = float(p.unrealized_pl or (cur - entry) * qty)
            pnl_pct = (cur / entry - 1) * 100 if entry > 0 else 0.0
            out.append(BrokerPosition(
                symbol=p.symbol, qty=qty, avg_entry_price=entry,
                current_price=cur, market_value=mv,
                unrealized_pnl=pnl, unrealized_pnl_pct=pnl_pct,
                side=str(p.side).lower(),
            ))
        return out

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        try:
            p = self._trading.get_open_position(symbol)
        except Exception:
            return None
        qty = float(p.qty)
        entry = float(p.avg_entry_price)
        cur = float(p.current_price or entry)
        return BrokerPosition(
            symbol=p.symbol, qty=qty, avg_entry_price=entry,
            current_price=cur,
            market_value=float(p.market_value or qty * cur),
            unrealized_pnl=float(p.unrealized_pl or 0.0),
            unrealized_pnl_pct=(cur / entry - 1) * 100 if entry > 0 else 0.0,
            side=str(p.side).lower(),
        )

    # ── Orders ───────────────────────────────────────────────────────────────
    def submit(self, order: Order) -> OrderResult:
        if not _is_us_symbol(order.symbol):
            return OrderResult(
                broker_order_id="", client_order_id=order.client_order_id,
                status="rejected",
                error=f"Alpaca cannot trade {order.symbol} (not a US ticker)",
            )

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest, LimitOrderRequest,
                StopOrderRequest, TrailingStopOrderRequest,
            )
            from alpaca.trading.enums import (
                OrderSide as ASide, TimeInForce as ATIF,
            )
        except ImportError as e:
            return OrderResult("", order.client_order_id, "rejected",
                               error=f"alpaca-py import failed: {e}")

        side = ASide.BUY if order.side == OrderSide.BUY else ASide.SELL
        tif = ATIF.GTC if order.time_in_force == TimeInForce.GTC else ATIF.DAY

        common = dict(
            symbol=order.symbol,
            qty=order.qty,
            side=side,
            time_in_force=tif,
            client_order_id=order.client_order_id,
        )

        try:
            if order.order_type == OrderType.MARKET:
                req = MarketOrderRequest(**common)
            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    raise ValueError("limit_price required for LIMIT")
                req = LimitOrderRequest(limit_price=order.limit_price, **common)
            elif order.order_type == OrderType.STOP:
                if order.stop_price is None:
                    raise ValueError("stop_price required for STOP")
                req = StopOrderRequest(stop_price=order.stop_price, **common)
            elif order.order_type == OrderType.TRAILING_STOP:
                if order.trail_percent is None:
                    raise ValueError("trail_percent required for TRAILING_STOP")
                req = TrailingStopOrderRequest(
                    trail_percent=order.trail_percent, **common,
                )
            else:
                return OrderResult("", order.client_order_id, "rejected",
                                   error=f"unsupported order_type {order.order_type}")

            resp = self._trading.submit_order(req)
            return OrderResult(
                broker_order_id=str(resp.id),
                client_order_id=str(resp.client_order_id) if resp.client_order_id else None,
                status=str(resp.status).lower(),
                filled_qty=float(resp.filled_qty or 0),
                filled_avg_price=float(resp.filled_avg_price or 0),
            )
        except Exception as e:
            log.warning("Alpaca submit failed for %s: %s", order.symbol, e)
            return OrderResult("", order.client_order_id, "rejected", error=str(e))

    def cancel_open_orders(self, symbol: Optional[str] = None) -> int:
        try:
            if symbol:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                orders = self._trading.get_orders(filter=req)
                count = 0
                for o in orders:
                    self._trading.cancel_order_by_id(o.id)
                    count += 1
                return count
            else:
                resp = self._trading.cancel_orders()
                return len(resp) if resp else 0
        except Exception as e:
            log.warning("cancel_open_orders failed: %s", e)
            return 0

    def close_position(self, symbol: str, qty: Optional[float] = None) -> OrderResult:
        try:
            if qty is None:
                resp = self._trading.close_position(symbol)
            else:
                from alpaca.trading.requests import ClosePositionRequest
                resp = self._trading.close_position(
                    symbol, ClosePositionRequest(qty=str(qty)),
                )
            return OrderResult(
                broker_order_id=str(resp.id),
                client_order_id=None,
                status=str(resp.status).lower(),
            )
        except Exception as e:
            return OrderResult("", None, "rejected", error=str(e))

    # ── Quotes ───────────────────────────────────────────────────────────────
    def last_price(self, symbol: str) -> Optional[float]:
        if not _is_us_symbol(symbol):
            return None
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            resp = self._data.get_stock_latest_trade(req)
            trade = resp.get(symbol)
            if trade is None:
                return None
            return float(trade.price)
        except Exception as e:
            log.debug("last_price(%s) failed: %s", symbol, e)
            return None

    def is_market_open(self) -> bool:
        try:
            clock = self._trading.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False
