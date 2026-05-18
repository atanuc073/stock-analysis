"""Portfolio domain models — pure data, no logic, no I/O."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Optional


class ExitType(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TIER_1 = "TIER_1"            # +22% — sell 33%, move stop to BE
    TIER_2 = "TIER_2"            # +35% — sell 33%, move stop to +15%
    TRAILING = "TRAILING"        # remainder trailing stop hit
    TIME_STOP = "TIME_STOP"      # 12 months no progress
    THESIS_BREAK = "THESIS_BREAK"  # score dropped below threshold
    RED_FLAG = "RED_FLAG"        # forensic / governance issue
    MANUAL = "MANUAL"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"


@dataclass
class TierLevel:
    """A profit-target tier on a position."""
    pct_gain: float       # e.g. 0.20 for +20%
    sell_fraction: float  # e.g. 0.33 of original qty
    new_stop_pct: Optional[float]  # new stop relative to entry; None = no change
    triggered: bool = False
    triggered_on: Optional[str] = None
    fill_price: Optional[float] = None


@dataclass
class Position:
    symbol: str
    qty_open: float                # remaining quantity
    qty_original: float            # original buy qty
    entry_price: float
    entry_date: str                # ISO date
    atr_at_entry: float
    stop_price: float              # current stop (moves with tiers / trailing)
    initial_stop_price: float
    peak_price: float              # highest close since entry (for trailing)
    tiers: list[TierLevel] = field(default_factory=list)
    sector: str = ""
    market: str = "IN"             # IN | US
    score_at_entry: float = 0.0
    uptrend_score_at_entry: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    realized_pnl: float = 0.0
    notes: str = ""
    # Number of consecutive bars (days) the price has closed at or below
    # stop_price. Used for 2-bar confirmation to avoid flash-crash whipsaws.
    below_stop_streak: int = 0
    # Regime label at the time of entry (BULL/NEUTRAL_BULL/NEUTRAL/CAUTIOUS/BEAR).
    # Used by the backtest reporter to slice P&L by entry regime.
    regime_label_at_entry: str = ""
    # Wide (loose) stop floor that applies only for the first N days after
    # entry. While today <= ``wide_stop_until_date``, the effective stop used
    # by ExitEvaluator is ``min(stop_price, wide_stop_price)`` — i.e. the
    # wider of the two. After that date, the wide stop is ignored and only
    # ``stop_price`` (with its tier/trail updates) governs exits.
    # 0.0 / "" = feature disabled for this position (legacy behaviour).
    wide_stop_price: float = 0.0
    wide_stop_until_date: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["status"] = PositionStatus(d.get("status", "OPEN"))
        d["tiers"] = [TierLevel(**t) for t in d.get("tiers", [])]
        return cls(**d)


@dataclass
class ExitSignal:
    """A recommendation produced by the ExitEvaluator — never executes itself."""
    symbol: str
    exit_type: ExitType
    suggested_qty: float
    current_price: float
    reason: str
    new_stop_price: Optional[float] = None
    pnl_pct: float = 0.0
    pnl_abs: float = 0.0


@dataclass
class Trade:
    """Immutable audit log entry."""
    symbol: str
    action: str                    # BUY | SELL
    qty: float
    price: float
    timestamp: str
    reason: str = ""
    exit_type: Optional[str] = None
    pnl_abs: float = 0.0


@dataclass
class PortfolioState:
    positions: list[Position] = field(default_factory=list)
    cash: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    peak_equity: float = 0.0       # high-water mark for drawdown tracking
    last_updated: str = ""

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status != PositionStatus.CLOSED]
