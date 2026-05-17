"""Position lifecycle: creation from signals + exit evaluation. Pure logic — no I/O."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

from .models import Position, TierLevel, ExitSignal, ExitType, PositionStatus


# ── Configuration (overridable via constructor) ──────────────────────────────
DEFAULT_TIERS = (
    # (pct_gain, sell_fraction, new_stop_pct_relative_to_entry)
    (0.22, 0.33, 0.0),     # T1 — sell 33%, move stop to break-even
    (0.50, 0.33, 0.25),    # T2 — sell 33%, lock +25%
)
DEFAULT_ATR_STOP_MULT = 3.0
DEFAULT_HARD_STOP_PCT = 0.18         # cap per-position loss at 18% (give trades room)
DEFAULT_TRAILING_PCT = 0.25          # trail 25% off peak after T2
DEFAULT_TIME_STOP_DAYS = 365
DEFAULT_TIME_STOP_BAND = (-0.05, 0.10)  # if return between -5% and +10% at time-stop, exit
DEFAULT_THESIS_BREAK_SCORE = 50.0
DEFAULT_STOP_CONFIRM_BARS = 2  # close ≤ stop on N consecutive bars before firing
DEFAULT_HARD_STOP_BUFFER = 0.025  # if price < hard_stop * (1 - this), fire immediately
DEFAULT_ADAPTIVE_TIERS = True   # re-evaluate strength at each tier and adapt sell size


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
        uptrend_score: float = 0.0,
        entry_date: Optional[str] = None,
    ) -> Position:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")

        atr_stop = entry_price - self.params.atr_stop_mult * max(atr, 0.0)
        hard_stop = entry_price * (1 - self.params.hard_stop_pct)
        # Cap the total risk at the hard stop floor:
        initial_stop = max(atr_stop, hard_stop)
        initial_stop = min(entry_price * 0.999, initial_stop)

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
            uptrend_score_at_entry=uptrend_score,
        )


# ── Exit evaluator ───────────────────────────────────────────────────────────
@dataclass
class ExitConfig:
    trailing_pct: float = DEFAULT_TRAILING_PCT
    time_stop_days: int = DEFAULT_TIME_STOP_DAYS
    time_stop_band: tuple = DEFAULT_TIME_STOP_BAND
    thesis_break_score: float = DEFAULT_THESIS_BREAK_SCORE
    # Stop must be breached on this many consecutive closes before firing.
    # Eliminates 1-day flash-crash whipsaws (2018 Q4, 2020 covid). 1 = legacy
    # behaviour.
    stop_confirm_bars: int = DEFAULT_STOP_CONFIRM_BARS
    # Even with confirmation, fire immediately if price collapses well below
    # the stop on a single bar (real breakdown, not noise).
    hard_stop_buffer: float = DEFAULT_HARD_STOP_BUFFER
    # When True, re-score the position's trend strength when a tier triggers
    # and adapt the sell fraction:
    #   3/3 strong  -> hold full position, raise stop only
    #   2/3 mixed   -> default tier behaviour
    #   0-1/3 weak  -> sell more aggressively + tighter stop
    adaptive_tiers: bool = DEFAULT_ADAPTIVE_TIERS
    trail_stop_pct: Optional[float] = None  # continuous trailing stop % from peak (e.g. 0.15 for 15%)


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
        regime_shock: bool = False,
        history=None,
    ) -> list[ExitSignal]:
        """Return zero or more exit signals. Tiers stack; stops short-circuit.

        Parameters
        ----------
        regime_shock
            If True, the broad market is in a panic-day state (VIX spike +
            index gap-down). Stop checks are skipped — we don't sell into
            a flash crash. Tier targets and hard-deep-breach exits still run.
            Pass via the engine using same-day macro inputs.
        """
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

        # 3) STOP LOSS — with N-bar confirmation + flash-crash protection
        if current_price <= position.stop_price:
            position.below_stop_streak += 1
            # Always fire on a deep breach (genuine breakdown, not noise)
            deep_breach = current_price <= position.stop_price * (1 - self.cfg.hard_stop_buffer)
            confirmed = position.below_stop_streak >= self.cfg.stop_confirm_bars
            # Skip the soft-confirm path during a market-wide panic day,
            # but still honour a deep breach.
            if deep_breach or (confirmed and not regime_shock):
                return [self._full_exit(
                    position, current_price, ExitType.STOP_LOSS,
                    f"Price {current_price:.2f} ≤ stop {position.stop_price:.2f}"
                    + (" (deep breach)" if deep_breach else f" ({position.below_stop_streak} bars)"),
                )]
            # Otherwise: keep the streak counter armed; do NOT exit yet.
        else:
            # Price reclaimed the stop — reset the streak.
            position.below_stop_streak = 0

        # 4) TIER targets (T1, T2) — partial exits
        for idx, tier in enumerate(position.tiers):
            if tier.triggered:
                continue
            target_price = position.entry_price * (1 + tier.pct_gain)
            if current_price >= target_price:
                # Default behaviour
                sell_frac = tier.sell_fraction
                new_stop_pct = tier.new_stop_pct
                strength_label = ""

                # Adaptive: re-score trend and adjust sell size
                if self.cfg.adaptive_tiers and history is not None:
                    strength = self._tier_strength(history, current_price)
                    if strength >= 3:
                        # All 3 trend signals positive — hold full size, just raise stop
                        sell_frac = 0.0
                        # Keep at least breakeven for T1, +15% for T2
                        floor = 0.0 if idx == 0 else 0.15
                        new_stop_pct = max(tier.new_stop_pct or 0.0, floor)
                        strength_label = " [strong 3/3 — hold]"
                    elif strength <= 1:
                        # Weakening — take more off, tighten stop
                        sell_frac = min(0.6, tier.sell_fraction * 1.5)
                        # Lock in roughly half the gain so far
                        new_stop_pct = max(
                            tier.new_stop_pct or 0.0,
                            tier.pct_gain * 0.5,
                        )
                        strength_label = f" [weak {strength}/3 — trim more]"
                    else:
                        strength_label = " [mixed 2/3 — default]"

                qty_to_sell = round(position.qty_original * sell_frac, 4)
                qty_to_sell = min(qty_to_sell, position.qty_open)
                new_stop = (
                    position.entry_price * (1 + (new_stop_pct or 0.0))
                    if new_stop_pct is not None else None
                )
                signals.append(ExitSignal(
                    symbol=position.symbol,
                    exit_type=ExitType.TIER_1 if idx == 0 else ExitType.TIER_2,
                    suggested_qty=qty_to_sell,
                    current_price=current_price,
                    reason=f"+{tier.pct_gain*100:.0f}% target hit{strength_label}",
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
    def update_peak(position: Position, current_price: float, trail_pct: Optional[float] = None) -> None:
        if current_price > position.peak_price:
            position.peak_price = current_price
        
        # Continuous trailing stop-loss from peak price
        if trail_pct is not None and trail_pct > 0:
            new_stop = position.peak_price * (1 - trail_pct)
            if new_stop > position.stop_price:
                position.stop_price = new_stop

    @staticmethod
    def _tier_strength(history, current_price: float) -> int:
        """Score the trend's remaining-room on a 0-3 scale.

        Three independent signals (each 0 or 1):
          1. Trend health    : close > SMA50 AND SMA50 > SMA200
          2. Momentum        : 1M return > 0 AND 3M return > 0
          3. Not parabolic   : RSI(14) < 75 AND price < SMA50 * 1.25

        `history` is an OHLCV DataFrame containing all bars up to and including
        the evaluation day (no look-ahead). Returns 2 (neutral) if data is
        insufficient so the default tier behaviour applies.
        """
        try:
            close = history["Close"]
        except (KeyError, TypeError, AttributeError):
            return 2
        n = len(close)
        if n < 60:
            return 2

        score = 0

        # Signal 1 — trend
        sma50 = close.tail(50).mean()
        sma200 = close.tail(200).mean() if n >= 200 else sma50
        if current_price > sma50 and sma50 > sma200:
            score += 1

        # Signal 2 — momentum (1M ~ 21 trading days, 3M ~ 63)
        try:
            ret_1m = current_price / close.iloc[-21] - 1
            ret_3m = current_price / close.iloc[-63] - 1 if n >= 63 else ret_1m
            if ret_1m > 0 and ret_3m > 0:
                score += 1
        except (IndexError, ZeroDivisionError):
            pass

        # Signal 3 — not parabolic
        try:
            delta = close.diff().tail(15)
            up = delta.clip(lower=0).mean()
            down = (-delta.clip(upper=0)).mean()
            rsi = 100.0 if down == 0 else 100 - 100 / (1 + up / down)
            extension = current_price / sma50 - 1 if sma50 else 0.0
            if rsi < 75 and extension < 0.25:
                score += 1
        except Exception:
            pass

        return score
