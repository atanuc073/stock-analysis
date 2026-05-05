"""Broker abstraction — vendor-neutral interface for execution.

This is the *only* contract that strategy code should depend on.
Concrete brokers (Alpaca, Zerodha, IBKR, paper-only) implement this Protocol.

Design notes:
  - Pure data classes (no vendor SDK types leak through)
  - All money is float (broker-side), all qty is float (fractional shares ok)
  - Orders are stateless; broker tracks status by client_order_id
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"   # good-til-cancelled (used for stop-losses)


@dataclass
class Order:
    """A single order request — broker-agnostic."""
    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None  # for TRAILING_STOP, e.g. 15.0 for 15%
    client_order_id: Optional[str] = None  # idempotency key
    notes: str = ""


@dataclass
class OrderResult:
    """Returned after submission."""
    broker_order_id: str
    client_order_id: Optional[str]
    status: str            # "accepted" | "filled" | "rejected" | "pending"
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    error: str = ""


@dataclass
class BrokerPosition:
    """A position as the broker sees it (read-only snapshot)."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str = "long"   # "long" | "short"


@dataclass
class BrokerAccount:
    """High-level account snapshot."""
    cash: float
    equity: float
    buying_power: float
    portfolio_value: float
    currency: str = "USD"
    is_paper: bool = True
    daytrade_count: int = 0
    pattern_day_trader: bool = False


@runtime_checkable
class Broker(Protocol):
    """Vendor-neutral broker interface.

    All concrete brokers (Alpaca, Zerodha, IBKR ...) implement this.
    Implementations must be **idempotent on client_order_id** for safe retries.
    """

    name: str
    is_paper: bool

    # ── Account ──────────────────────────────────────────────────────────────
    def account(self) -> BrokerAccount:
        """Snapshot of cash, equity, buying power."""

    # ── Positions ────────────────────────────────────────────────────────────
    def positions(self) -> list[BrokerPosition]:
        """All currently-held positions."""

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Single position lookup, None if flat."""

    # ── Orders ───────────────────────────────────────────────────────────────
    def submit(self, order: Order) -> OrderResult:
        """Submit an order. Idempotent if client_order_id is reused."""

    def cancel_open_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders (optionally for one symbol). Returns count."""

    def close_position(self, symbol: str, qty: Optional[float] = None) -> OrderResult:
        """Close all or part of a position with a market order."""

    # ── Quotes ───────────────────────────────────────────────────────────────
    def last_price(self, symbol: str) -> Optional[float]:
        """Latest trade price. Returns None if unavailable / market closed."""

    def is_market_open(self) -> bool:
        """Whether the primary exchange is currently open."""
