"""Portfolio service — orchestrates models, repository, and lifecycle.
Depends only on abstractions (PortfolioRepository protocol)."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .models import (
    PortfolioState, Position, PositionStatus, Trade,
    ExitSignal, ExitType, TierLevel,
)
from .repository import PortfolioRepository
from .lifecycle import PositionFactory, ExitEvaluator


@dataclass
class EquitySnapshot:
    cash: float
    invested: float
    market_value: float
    total: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown_pct: float


class PortfolioService:
    """High-level portfolio operations.

    Stateless w.r.t. business logic — state lives in the repository.
    """

    def __init__(
        self,
        repo: PortfolioRepository,
        factory: PositionFactory | None = None,
        evaluator: ExitEvaluator | None = None,
    ) -> None:
        self._repo = repo
        self._factory = factory or PositionFactory()
        self._evaluator = evaluator or ExitEvaluator()

    # ── Reads ────────────────────────────────────────────────────────────────
    def state(self) -> PortfolioState:
        return self._repo.load()

    def find(self, symbol: str) -> Optional[Position]:
        for p in self.state().open_positions():
            if p.symbol.upper() == symbol.upper():
                return p
        return None

    # ── Writes ───────────────────────────────────────────────────────────────
    def add_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        atr: float,
        sector: str = "",
        market: str = "IN",
        score: float = 0.0,
    ) -> Position:
        state = self._repo.load()
        if any(p.symbol.upper() == symbol.upper()
               and p.status != PositionStatus.CLOSED for p in state.positions):
            raise ValueError(f"Position already open for {symbol}")

        pos = self._factory.create(symbol, qty, entry_price, atr, sector, market, score)
        state.positions.append(pos)
        state.cash -= qty * entry_price
        state.trades.append(Trade(
            symbol=symbol, action="BUY", qty=qty, price=entry_price,
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            reason=f"score={score:.1f}",
        ))
        self._repo.save(state)
        return pos

    def execute_exit(
        self,
        signal: ExitSignal,
        actual_price: Optional[float] = None,
        actual_qty: Optional[float] = None,
    ) -> None:
        """Apply an exit signal to portfolio state. Idempotent on re-application."""
        state = self._repo.load()
        pos = next((p for p in state.positions
                    if p.symbol.upper() == signal.symbol.upper()
                    and p.status != PositionStatus.CLOSED), None)
        if pos is None:
            raise ValueError(f"No open position for {signal.symbol}")

        price = actual_price if actual_price is not None else signal.current_price
        qty = actual_qty if actual_qty is not None else signal.suggested_qty
        qty = min(qty, pos.qty_open)
        if qty <= 0:
            return

        proceeds = price * qty
        cost = pos.entry_price * qty
        realized = proceeds - cost

        pos.qty_open = round(pos.qty_open - qty, 6)
        pos.realized_pnl += realized
        if signal.new_stop_price is not None:
            pos.stop_price = max(pos.stop_price, signal.new_stop_price)

        # Mark tier as triggered if applicable
        if signal.exit_type in (ExitType.TIER_1, ExitType.TIER_2):
            tier_idx = 0 if signal.exit_type == ExitType.TIER_1 else 1
            if tier_idx < len(pos.tiers):
                pos.tiers[tier_idx].triggered = True
                pos.tiers[tier_idx].triggered_on = datetime.utcnow().date().isoformat()
                pos.tiers[tier_idx].fill_price = price

        if pos.qty_open <= 1e-6:
            pos.status = PositionStatus.CLOSED
            pos.qty_open = 0.0
        else:
            pos.status = PositionStatus.PARTIALLY_CLOSED

        state.cash += proceeds
        state.trades.append(Trade(
            symbol=pos.symbol, action="SELL", qty=qty, price=price,
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            reason=signal.reason, exit_type=signal.exit_type.value,
            pnl_abs=realized,
        ))
        self._repo.save(state)

    def update_peaks(self, prices: dict[str, float]) -> None:
        state = self._repo.load()
        for p in state.open_positions():
            cp = prices.get(p.symbol)
            if cp is not None:
                self._evaluator.update_peak(p, cp)
        self._repo.save(state)

    # ── Evaluation (read-only) ───────────────────────────────────────────────
    def evaluate_all(
        self,
        prices: dict[str, float],
        scores: dict[str, float] | None = None,
        red_flags: dict[str, int] | None = None,
    ) -> list[ExitSignal]:
        state = self._repo.load()
        scores = scores or {}
        red_flags = red_flags or {}
        signals: list[ExitSignal] = []
        for p in state.open_positions():
            cp = prices.get(p.symbol)
            if cp is None:
                continue
            self._evaluator.update_peak(p, cp)
            signals.extend(self._evaluator.evaluate(
                p, cp,
                current_score=scores.get(p.symbol),
                red_flags=red_flags.get(p.symbol, 0),
            ))
        return signals

    def equity_snapshot(self, prices: dict[str, float]) -> EquitySnapshot:
        state = self._repo.load()
        invested = sum(p.entry_price * p.qty_open for p in state.open_positions())
        mv = sum(prices.get(p.symbol, p.entry_price) * p.qty_open
                 for p in state.open_positions())
        realized = sum(p.realized_pnl for p in state.positions)
        total = state.cash + mv
        # update high-water mark
        if total > state.peak_equity:
            state.peak_equity = total
            self._repo.save(state)
        dd = (total / state.peak_equity - 1) * 100 if state.peak_equity > 0 else 0.0
        return EquitySnapshot(
            cash=state.cash, invested=invested, market_value=mv,
            total=total, unrealized_pnl=mv - invested,
            realized_pnl=realized, drawdown_pct=dd,
        )
