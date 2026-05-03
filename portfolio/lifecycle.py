"""Position lifecycle: creation from signals + exit evaluation. Pure logic — no I/O."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

from .models import Position, TierLevel, ExitSignal, ExitType, PositionStatus


# ── Configuration (overridable via constructor) ──────────────────────────────
DEFAULT_TIERS = (
    # (pct_gain, sell_fraction, new_stop_pct_relative_to_entry)
    (0.20, 1 / 3, 0.0),     # T1 — sell 33%, move stop to break-even
    (0.35, 1 / 3, 0.15),    # T2 — sell 33%, lock +15%
)
DEFAULT_ATR_STOP_MULT = 2.5
DEFAULT_HARD_STOP_PCT = 0.12         # never risk more than 12% per position
DEFAULT_TRAILING_PCT = 0.15          # trail 15% off peak after T2
DEFAULT_TIME_STOP_DAYS = 365
DEFAULT_TIME_STOP_BAND = (-0.05, 0.10)  # if return between -5% and +10% at time-stop, exit
DEFAULT_THESIS_BREAK_SCORE = 50.0


# ── Position factory ─────────────────────────────────────────────────────────
@dataclass
class EntryParameters:
    atr_stop_mult: float = DEFAULT_ATR_STOP_MULT
    hard_stop_pct: float = DEFAULT_HARD_STOP_PCT
    tiers: tuple = DEFAULT_TIERS


class PositionFactory:
    """Builds a fully-formed Position from a buy signal. Single responsibility."""

    def __init__(self, params: EntryParameters | None = None) -> None:
        self.params = params or EntryParameters()

    def create(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        atr: float,
        sector: str = "",
        market: str = "IN",
        score: float = 0.0,
        entry_date: Optional[str] = None,
    ) -> Position:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")

        atr_stop = entry_price - self.params.atr_stop_mult * max(atr, 0.0)
        hard_stop = entry_price * (1 - self.params.hard_stop_pct)
        # take the *higher* (tighter, smaller loss) of the two — but never above entry
        initial_stop = min(entry_price * 0.999, max(atr_stop, hard_stop))

        tiers = [
            TierLevel(
                pct_gain=pct,
                sell_fraction=frac,
                new_stop_pct=new_stop,
            )
            for pct, frac, new_stop in self.params.tiers
        ]

        return Position(
            symbol=symbol,
            qty_open=qty,
            qty_original=qty,
            entry_price=entry_price,
            entry_date=entry_date or date.today().isoformat(),
            atr_at_entry=atr,
            stop_price=initial_stop,
            initial_stop_price=initial_stop,
            peak_price=entry_price,
            tiers=tiers,
            sector=sector,
            market=market,
            score_at_entry=score,
        )


# ── Exit evaluator ───────────────────────────────────────────────────────────
@dataclass
class ExitConfig:
    trailing_pct: float = DEFAULT_TRAILING_PCT
    time_stop_days: int = DEFAULT_TIME_STOP_DAYS
    time_stop_band: tuple = DEFAULT_TIME_STOP_BAND
    thesis_break_score: float = DEFAULT_THESIS_BREAK_SCORE


class ExitEvaluator:
    """Decides what (if anything) should happen to a position on a given day.

    Returns a *recommendation* — execution is the caller's job.
    Order of checks: hard stop → tier targets → thesis break → time stop.
    """

    def __init__(self, config: ExitConfig | None = None) -> None:
        self.cfg = config or ExitConfig()

    def evaluate(
        self,
        position: Position,
        current_price: float,
        current_score: Optional[float] = None,
        red_flags: int = 0,
        today: Optional[date] = None,
    ) -> list[ExitSignal]:
        """Return zero or more exit signals. Tiers stack; stops short-circuit."""
        if position.status == PositionStatus.CLOSED or position.qty_open <= 0:
            return []
        today = today or date.today()
        signals: list[ExitSignal] = []

        # 1) RED FLAG override — always exit fully
        if red_flags >= 2:
            return [self._full_exit(position, current_price, ExitType.RED_FLAG,
                                    f"{red_flags} forensic flags raised")]

        # 2) THESIS BREAK — score crashed
        if current_score is not None and current_score < self.cfg.thesis_break_score:
            return [self._full_exit(position, current_price, ExitType.THESIS_BREAK,
                                    f"Score {current_score:.1f} < {self.cfg.thesis_break_score}")]

        # 3) STOP LOSS — short-circuits everything else
        if current_price <= position.stop_price:
            return [self._full_exit(position, current_price, ExitType.STOP_LOSS,
                                    f"Price {current_price:.2f} ≤ stop {position.stop_price:.2f}")]

        # 4) TIER targets (T1, T2) — partial exits
        for idx, tier in enumerate(position.tiers):
            if tier.triggered:
                continue
            target_price = position.entry_price * (1 + tier.pct_gain)
            if current_price >= target_price:
                qty_to_sell = round(position.qty_original * tier.sell_fraction, 4)
                qty_to_sell = min(qty_to_sell, position.qty_open)
                new_stop = (
                    position.entry_price * (1 + (tier.new_stop_pct or 0.0))
                    if tier.new_stop_pct is not None else None
                )
                signals.append(ExitSignal(
                    symbol=position.symbol,
                    exit_type=ExitType.TIER_1 if idx == 0 else ExitType.TIER_2,
                    suggested_qty=qty_to_sell,
                    current_price=current_price,
                    reason=f"+{tier.pct_gain*100:.0f}% target hit",
                    new_stop_price=new_stop,
                    pnl_pct=(current_price / position.entry_price - 1) * 100,
                    pnl_abs=(current_price - position.entry_price) * qty_to_sell,
                ))

        # 5) TRAILING stop on remainder (only after all tiers triggered)
        all_tiers_done = all(t.triggered for t in position.tiers)
        if all_tiers_done and position.qty_open > 0:
            trail_stop = position.peak_price * (1 - self.cfg.trailing_pct)
            if current_price <= trail_stop:
                signals.append(self._full_exit(
                    position, current_price, ExitType.TRAILING,
                    f"Price {current_price:.2f} ≤ trailing {trail_stop:.2f} ({self.cfg.trailing_pct*100:.0f}% off peak)",
                ))

        # 6) TIME STOP — dead capital
        try:
            entry_dt = date.fromisoformat(position.entry_date)
        except Exception:
            entry_dt = today
        days_held = (today - entry_dt).days
        if days_held >= self.cfg.time_stop_days and position.qty_open > 0:
            ret = current_price / position.entry_price - 1
            lo, hi = self.cfg.time_stop_band
            if lo <= ret <= hi:
                signals.append(self._full_exit(
                    position, current_price, ExitType.TIME_STOP,
                    f"{days_held}d held, return {ret*100:+.1f}% in dead band",
                ))

        return signals

    @staticmethod
    def _full_exit(position: Position, price: float, exit_type: ExitType, reason: str) -> ExitSignal:
        return ExitSignal(
            symbol=position.symbol,
            exit_type=exit_type,
            suggested_qty=position.qty_open,
            current_price=price,
            reason=reason,
            pnl_pct=(price / position.entry_price - 1) * 100,
            pnl_abs=(price - position.entry_price) * position.qty_open,
        )

    @staticmethod
    def update_peak(position: Position, current_price: float) -> None:
        if current_price > position.peak_price:
            position.peak_price = current_price
