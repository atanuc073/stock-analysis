"""Event-driven backtest engine.

Replays the scoring + portfolio + exit logic on historical data, using only
data available as of each rebalance date (no lookahead bias).

Design:
  - Reuses PositionFactory and ExitEvaluator unchanged (LSP/DIP)
  - Uses an in-memory portfolio (no JSON I/O)
  - Rebalances weekly by default — fast and matches position-investing horizon
  - Costs: configurable bps per fill (default: 0.10% one-way)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd
from tqdm import tqdm

from portfolio.models import (
    Position, ExitType, PositionStatus, Trade, TierLevel,
)
from portfolio.lifecycle import (
    PositionFactory, ExitEvaluator, EntryParameters, ExitConfig,
)

from .data_loader import HistoricalData
from .scoring import score_at, price_at, BacktestScore
from . import regime as regime_mod

log = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0       # ₹10 lakh default
    rebalance_freq_days: int = 5               # weekly
    min_score: float = 60.0                    # buy threshold
    max_positions: int = 12                    # concurrent open positions
    base_position_weight: float = 0.10         # 10% of equity per new position
    max_position_weight: float = 0.15
    max_sector_weight: float = 0.30
    transaction_cost_bps: float = 10.0         # 10 bps each side (0.10%)
    slippage_bps: float = 5.0                  # 5 bps slippage
    include_forecast: bool = False             # forecast slow; off by default
    live_weights: bool = True                   # use live SCORE_WEIGHTS (no redistribution)
    use_regime: bool = True                     # enable market regime filter
    regime_skip_below: str = "BEAR"            # skip entries at or below this regime
    regime_check_freq_days: int = 5            # re-evaluate regime every N days
    # Regime-driven de-risking of OPEN positions (not just new entries):
    # When regime label is at or below this floor, trim each open position
    # toward `regime_derisk_target_mult` of its current size on rebalance days.
    regime_derisk_below: Optional[str] = "CAUTIOUS"   # None to disable
    regime_derisk_target_mult: float = 0.5            # keep 50% of size in CAUTIOUS/BEAR
    # Shock-day stop suppression: if VIX day-over-day jumps >= shock_vix_jump
    # AND the broad index drops <= shock_index_drop on the same day, treat
    # it as a panic day and skip soft stop-loss exits (still honours deep
    # breaches via lifecycle.hard_stop_buffer).
    shock_vix_jump: float = 0.30               # +30% one-day VIX jump
    shock_index_drop: float = -0.03            # -3% one-day index drop
    weights: Optional[dict] = None              # per-run override of SCORE_WEIGHTS (None = use config)

    @property
    def cost_per_side(self) -> float:
        return (self.transaction_cost_bps + self.slippage_bps) / 10000.0


# ── Trade record (extended for backtest reporting) ───────────────────────────
@dataclass
class BTTrade:
    symbol: str
    action: str             # BUY | SELL
    qty: float
    price: float
    gross_value: float
    cost: float
    net_value: float
    timestamp: str
    sector: str = ""
    market: str = ""
    reason: str = ""
    exit_type: Optional[str] = None
    pnl_abs: float = 0.0
    pnl_pct: float = 0.0
    days_held: int = 0
    score_at_entry: float = 0.0


@dataclass
class EquityPoint:
    date: pd.Timestamp
    cash: float
    market_value: float
    total: float
    n_open: int


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trades: list[BTTrade] = field(default_factory=list)
    final_positions: list[Position] = field(default_factory=list)
    benchmark_curve: dict[str, list[tuple[pd.Timestamp, float]]] = field(default_factory=dict)
    universe_size: int = 0
    start: str = ""
    end: str = ""


# ── The engine ───────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(
        self,
        data: dict[str, HistoricalData],
        config: Optional[BacktestConfig] = None,
        entry_params: Optional[EntryParameters] = None,
        exit_cfg: Optional[ExitConfig] = None,
        regime_data: Optional[dict[str, dict[str, pd.Series]]] = None,
    ):
        """
        regime_data: { 'IN': {'index': series, 'vix': series}, 'US': {...} }
        Pass None to disable regime-aware sizing entirely.
        """
        self.data = data
        self.cfg = config or BacktestConfig()
        self.factory = PositionFactory(entry_params)
        self.exit_eval = ExitEvaluator(exit_cfg)
        self.regime_data = regime_data or {}

        # Simulation state
        self.cash: float = self.cfg.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[BTTrade] = []
        self.equity_curve: list[EquityPoint] = []

        # Regime cache: avoid recomputing every rebalance
        self._regime_cache: dict[tuple[pd.Timestamp, str], regime_mod.HistoricalRegime] = {}
        self._last_regime_compute: Optional[pd.Timestamp] = None
        self._current_regime: dict[str, regime_mod.HistoricalRegime] = {}

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self, dates: pd.DatetimeIndex) -> BacktestResult:
        if len(dates) == 0:
            raise ValueError("No trading dates provided")

        log.info("Backtest: %s → %s, %d days, %d symbols, capital ₹%s",
                 dates[0].date(), dates[-1].date(), len(dates),
                 len(self.data), f"{self.cfg.initial_capital:,.0f}")

        rebalance_dates = set(dates[::self.cfg.rebalance_freq_days])
        rebalance_dates.add(dates[-1])  # always close on last day

        for asof in tqdm(dates, desc="Simulating"):
            # 1) Update prices on all open positions, update peaks
            current_prices = self._current_prices(asof)
            for sym, pos in list(self.positions.items()):
                cp = current_prices.get(sym)
                if cp is not None:
                    ExitEvaluator.update_peak(pos, cp)

            # 1b) Detect macro shock day (VIX spike + index drop)
            shock = self._is_shock_day(asof)

            # 2) Evaluate exits every day (tighter risk control)
            self._process_exits(asof, current_prices, evaluate_thesis=False,
                                regime_shock=shock)

            # 3) Rebalance entries weekly (or whatever cadence)
            if asof in rebalance_dates:
                self._rebalance(asof, current_prices)
                # Re-process exits with thesis break (using updated scores)
                self._process_exits(asof, current_prices, evaluate_thesis=True,
                                    regime_shock=shock)

            # 4) Record equity
            mv = self._market_value(current_prices)
            self.equity_curve.append(EquityPoint(
                date=asof, cash=self.cash, market_value=mv,
                total=self.cash + mv,
                n_open=sum(1 for p in self.positions.values()
                           if p.status != PositionStatus.CLOSED),
            ))

        # 5) Force close remaining positions at end
        self._close_all(dates[-1], self._current_prices(dates[-1]),
                        reason="END_OF_BACKTEST")

        # Final equity point
        final_prices = self._current_prices(dates[-1])
        self.equity_curve.append(EquityPoint(
            date=dates[-1], cash=self.cash,
            market_value=self._market_value(final_prices),
            total=self.cash + self._market_value(final_prices),
            n_open=0,
        ))

        return BacktestResult(
            config=self.cfg,
            equity_curve=self.equity_curve,
            trades=self.trades,
            final_positions=list(self.positions.values()),
            universe_size=len(self.data),
            start=str(dates[0].date()),
            end=str(dates[-1].date()),
        )

    # ── Internals ────────────────────────────────────────────────────────────
    def _current_prices(self, asof: pd.Timestamp) -> dict[str, float]:
        out: dict[str, float] = {}
        for sym, hd in self.data.items():
            p = price_at(hd, asof)
            if p is not None:
                out[sym] = p
        return out

    def _market_value(self, prices: dict[str, float]) -> float:
        return sum(
            prices.get(p.symbol, p.entry_price) * p.qty_open
            for p in self.positions.values()
            if p.status != PositionStatus.CLOSED
        )

    # ── Regime helpers ───────────────────────────────────────────────────────
    def _refresh_regime(self, asof: pd.Timestamp) -> None:
        """Recompute regime per market if check window expired. Cheap."""
        if not self.cfg.use_regime or not self.regime_data:
            return
        if (self._last_regime_compute is not None
            and (asof - self._last_regime_compute).days < self.cfg.regime_check_freq_days):
            return
        for market, series in self.regime_data.items():
            self._current_regime[market] = regime_mod.detect(
                series.get("index", pd.Series(dtype=float)),
                series.get("vix"),
                asof,
            )
        self._last_regime_compute = asof

    def _regime_multiplier(self, market: str) -> float:
        if not self.cfg.use_regime:
            return 1.0
        r = self._current_regime.get(market)
        return r.allocation_multiplier if r else 1.0

    def _regime_blocks_entry(self, market: str) -> bool:
        if not self.cfg.use_regime:
            return False
        r = self._current_regime.get(market)
        if r is None:
            return False
        order = ["BEAR", "CAUTIOUS", "NEUTRAL", "NEUTRAL_BULL", "BULL"]
        try:
            cur = order.index(r.label)
            floor = order.index(self.cfg.regime_skip_below)
        except ValueError:
            return False
        return cur <= floor

    def _regime_should_derisk(self, market: str) -> bool:
        """True when current regime warrants trimming OPEN positions."""
        if not self.cfg.use_regime or self.cfg.regime_derisk_below is None:
            return False
        r = self._current_regime.get(market)
        if r is None:
            return False
        order = ["BEAR", "CAUTIOUS", "NEUTRAL", "NEUTRAL_BULL", "BULL"]
        try:
            cur = order.index(r.label)
            floor = order.index(self.cfg.regime_derisk_below)
        except ValueError:
            return False
        return cur <= floor

    def _apply_regime_derisk(self, asof: pd.Timestamp,
                              current_prices: dict[str, float]) -> None:
        """Trim open positions when regime label drops to/below derisk floor.

        Sells down each affected position to `regime_derisk_target_mult` of
        its current size. Idempotent: once trimmed, won't keep selling
        further unless regime degrades again or position grows back.
        """
        if not self.cfg.use_regime or self.cfg.regime_derisk_below is None:
            return
        target_mult = self.cfg.regime_derisk_target_mult
        if target_mult >= 1.0 or target_mult < 0.0:
            return
        for sym, pos in list(self.positions.items()):
            if pos.status == PositionStatus.CLOSED or pos.qty_open <= 0:
                continue
            if not self._regime_should_derisk(pos.market):
                continue
            cp = current_prices.get(sym)
            if cp is None or cp <= 0:
                continue
            target_qty = pos.qty_original * target_mult
            trim_qty = pos.qty_open - target_qty
            if trim_qty <= max(pos.qty_original * 0.05, 1e-6):
                continue  # already at/below target
            gross = cp * trim_qty
            cost = gross * self.cfg.cost_per_side
            net = gross - cost
            pnl_abs = (cp - pos.entry_price) * trim_qty
            pnl_pct = (cp / pos.entry_price - 1) * 100
            try:
                entry_dt = date.fromisoformat(pos.entry_date)
            except Exception:
                entry_dt = asof.date()
            days_held = (asof.date() - entry_dt).days

            self.cash += net
            pos.qty_open -= trim_qty
            pos.realized_pnl += pnl_abs
            if pos.qty_open <= 1e-6:
                pos.status = PositionStatus.CLOSED
            else:
                pos.status = PositionStatus.PARTIALLY_CLOSED

            r = self._current_regime.get(pos.market)
            label = r.label if r else "?"
            self.trades.append(BTTrade(
                symbol=pos.symbol, action="SELL", qty=trim_qty, price=cp,
                gross_value=gross, cost=cost, net_value=net,
                timestamp=asof.isoformat(),
                sector=pos.sector, market=pos.market,
                reason=f"REGIME_DERISK ({label})",
                exit_type="REGIME_DERISK",
                pnl_abs=pnl_abs, pnl_pct=pnl_pct,
                days_held=days_held,
                score_at_entry=pos.score_at_entry,
            ))

    # ── Entries ──────────────────────────────────────────────────────────────
    def _rebalance(self, asof: pd.Timestamp, current_prices: dict[str, float]) -> None:
        """Score universe, find candidates, open new positions.

        Capital-driven (not slot-driven): keep deploying as long as cash is
        idle (>5% of equity), up to a hard cap of 2x max_positions to prevent
        fragmentation. This eliminates the cash drag where T1/T2 freed cash
        sat idle until the full position closed.
        """
        self._refresh_regime(asof)
        self._apply_regime_derisk(asof, current_prices)

        equity = self.cash + self._market_value(current_prices)
        cash_floor = equity * 0.05  # don't bother if <5% deployable
        n_open = sum(1 for p in self.positions.values()
                     if p.status != PositionStatus.CLOSED)
        # Hard cap on fragmentation: never exceed 2x max_positions
        hard_cap = self.cfg.max_positions * 2
        if self.cash < cash_floor and n_open >= self.cfg.max_positions:
            return
        if n_open >= hard_cap:
            return

        # Score every symbol with sufficient history
        scores: list[BacktestScore] = []
        for sym, hd in self.data.items():
            if sym in self.positions and self.positions[sym].status != PositionStatus.CLOSED:
                continue  # skip open positions
            if self._regime_blocks_entry(hd.market):
                continue  # regime says: no new entries in this market
            s = score_at(hd, asof,
                         include_forecast=self.cfg.include_forecast,
                         live_weights=self.cfg.live_weights,
                         weights_override=self.cfg.weights)
            if s is None:
                continue
            # Hard block: skip stocks extended >40% above 200DMA (late-cycle
            # blow-off tops). The technical scorer already penalizes these
            # but the score can still clear 65 via fundamentals/momentum.
            ext = s.technical.get("pct_above_sma200", 0.0)
            if ext > 40.0:
                continue
            scores.append(s)
        if not scores:
            return
        # Cross-sectional pass over the full candidate universe (sector-relative
        # valuation + universe momentum/quality z-score rank bonus). This must
        # happen BEFORE the min_score filter so z-scores reflect the whole
        # universe, not just the pre-filtered top tier.
        try:
            from analysis.cross_sectional import apply_to_bt
            apply_to_bt(scores)
        except Exception as e:
            log.warning("cross-sectional pass failed: %s", e)
            for s in scores:
                if not hasattr(s, "adjusted_score") or not s.adjusted_score:
                    s.adjusted_score = s.score

        # Filter on adjusted_score so the backtest matches the live ranking.
        scores = [s for s in scores if s.adjusted_score >= self.cfg.min_score]
        if not scores:
            return
        scores.sort(key=lambda s: s.adjusted_score, reverse=True)

        equity = self.cash + self._market_value(current_prices)
        sector_weights = self._sector_weights(current_prices, equity)
        market_weights = self._market_weights(current_prices, equity)

        # Slots = remaining capacity (up to hard_cap), driven by both
        # position count AND idle cash. After a T1/T2 fires, n_open stays
        # the same but cash rises — we still allow new entries via hard_cap.
        slots = hard_cap - n_open
        for s in scores:
            if slots <= 0:
                break
            if self.cash <= cash_floor:
                break
            # Concentration checks
            if sector_weights.get(s.sector, 0.0) >= self.cfg.max_sector_weight:
                continue
            if market_weights.get(s.market, 0.0) >= 0.70:
                continue

            # Volatility-adjusted sizing × regime allocation multiplier.
            # Use the cross-sectional adjusted score to match the live signal.
            base_w = self.cfg.base_position_weight
            vol_adj = min(0.25 / max(s.annual_vol, 0.05), 1.5)
            score_adj = min(s.adjusted_score / 70.0, 1.3)
            regime_adj = self._regime_multiplier(s.market)
            target_w = min(
                base_w * vol_adj * score_adj * regime_adj,
                self.cfg.max_position_weight,
            )
            target_rupees = equity * target_w

            if target_rupees < equity * 0.02:    # too small to bother
                continue
            if target_rupees > self.cash * 0.95:  # not enough cash
                target_rupees = self.cash * 0.95
            if target_rupees < equity * 0.02:
                continue

            qty = target_rupees / s.price
            if qty <= 0:
                continue

            self._open_position(s, qty, asof, equity)
            # Update running weights so next iteration sees latest
            current_prices[s.symbol] = s.price
            equity = self.cash + self._market_value(current_prices)
            sector_weights = self._sector_weights(current_prices, equity)
            market_weights = self._market_weights(current_prices, equity)
            slots -= 1

    def _open_position(self, s: BacktestScore, qty: float,
                       asof: pd.Timestamp, equity: float) -> None:
        cost = s.price * qty * (1 + self.cfg.cost_per_side)
        if cost > self.cash:
            return
        pos = self.factory.create(
            symbol=s.symbol, qty=qty, entry_price=s.price,
            atr=s.atr_value, sector=s.sector, market=s.market,
            score=s.adjusted_score, entry_date=asof.strftime("%Y-%m-%d"),
        )
        self.positions[s.symbol] = pos
        self.cash -= cost
        self.trades.append(BTTrade(
            symbol=s.symbol, action="BUY", qty=qty, price=s.price,
            gross_value=s.price * qty,
            cost=cost - s.price * qty,
            net_value=cost,
            timestamp=asof.isoformat(),
            sector=s.sector, market=s.market,
            reason=f"adj={s.adjusted_score:.1f} (raw={s.score:.1f})",
            score_at_entry=s.adjusted_score,
        ))

    def _sector_weights(self, prices: dict[str, float], equity: float) -> dict[str, float]:
        if equity <= 0:
            return {}
        out: dict[str, float] = {}
        for p in self.positions.values():
            if p.status == PositionStatus.CLOSED:
                continue
            mv = prices.get(p.symbol, p.entry_price) * p.qty_open
            out[p.sector] = out.get(p.sector, 0.0) + mv / equity
        return out

    def _market_weights(self, prices: dict[str, float], equity: float) -> dict[str, float]:
        if equity <= 0:
            return {}
        out: dict[str, float] = {}
        for p in self.positions.values():
            if p.status == PositionStatus.CLOSED:
                continue
            mv = prices.get(p.symbol, p.entry_price) * p.qty_open
            out[p.market] = out.get(p.market, 0.0) + mv / equity
        return out

    # ── Exits ────────────────────────────────────────────────────────────────
    def _is_shock_day(self, asof: pd.Timestamp) -> bool:
        """True if any tracked market shows a VIX spike + index gap-down today.

        Uses the regime_data series passed at construction. Cheap O(1) lookup.
        """
        if not self.regime_data:
            return False
        for mkt, series in self.regime_data.items():
            idx = series.get("index")
            vix = series.get("vix")
            if idx is None or vix is None or idx.empty or vix.empty:
                continue
            if idx.index.tz is not None:
                idx = idx.copy(); idx.index = idx.index.tz_localize(None)
            if vix.index.tz is not None:
                vix = vix.copy(); vix.index = vix.index.tz_localize(None)
            idx_hist = idx[idx.index <= asof]
            vix_hist = vix[vix.index <= asof]
            if len(idx_hist) < 2 or len(vix_hist) < 2:
                continue
            idx_chg = idx_hist.iloc[-1] / idx_hist.iloc[-2] - 1
            vix_chg = vix_hist.iloc[-1] / vix_hist.iloc[-2] - 1
            if (vix_chg >= self.cfg.shock_vix_jump
                    and idx_chg <= self.cfg.shock_index_drop):
                return True
        return False

    def _process_exits(self, asof: pd.Timestamp, prices: dict[str, float],
                       evaluate_thesis: bool,
                       regime_shock: bool = False) -> None:
        for sym, pos in list(self.positions.items()):
            if pos.status == PositionStatus.CLOSED:
                continue
            cp = prices.get(sym)
            if cp is None:
                continue

            current_score: Optional[float] = None
            if evaluate_thesis:
                hd = self.data.get(sym)
                if hd:
                    s = score_at(hd, asof,
                                 include_forecast=False,
                                 live_weights=self.cfg.live_weights,
                                 weights_override=self.cfg.weights)
                    if s:
                        # Use raw composite for thesis-break (cross-sectional
                        # context is unavailable for a single ticker on exit).
                        current_score = s.score

            signals = self.exit_eval.evaluate(
                pos, cp, current_score=current_score, red_flags=0,
                today=asof.date(), regime_shock=regime_shock,
            )
            for sig in signals:
                self._execute_exit(pos, sig, asof)

    def _execute_exit(self, pos: Position, sig, asof: pd.Timestamp) -> None:
        qty = min(sig.suggested_qty, pos.qty_open)
        if qty <= 0:
            return
        gross = sig.current_price * qty
        cost = gross * self.cfg.cost_per_side
        net = gross - cost

        pnl_abs = (sig.current_price - pos.entry_price) * qty
        pnl_pct = (sig.current_price / pos.entry_price - 1) * 100
        try:
            entry_dt = date.fromisoformat(pos.entry_date)
        except Exception:
            entry_dt = asof.date()
        days_held = (asof.date() - entry_dt).days

        self.cash += net
        pos.qty_open -= qty
        pos.realized_pnl += pnl_abs

        # Apply tier side-effects: mark triggered + bump stop
        if sig.exit_type in (ExitType.TIER_1, ExitType.TIER_2):
            tier_idx = 0 if sig.exit_type == ExitType.TIER_1 else 1
            if tier_idx < len(pos.tiers):
                t = pos.tiers[tier_idx]
                t.triggered = True
                t.triggered_on = asof.isoformat()
                t.fill_price = sig.current_price
            if sig.new_stop_price is not None:
                pos.stop_price = sig.new_stop_price

        # Status
        if pos.qty_open <= 1e-6:
            pos.status = PositionStatus.CLOSED
        elif pos.qty_original > pos.qty_open:
            pos.status = PositionStatus.PARTIALLY_CLOSED

        self.trades.append(BTTrade(
            symbol=pos.symbol, action="SELL", qty=qty, price=sig.current_price,
            gross_value=gross, cost=cost, net_value=net,
            timestamp=asof.isoformat(),
            sector=pos.sector, market=pos.market,
            reason=sig.reason,
            exit_type=sig.exit_type.value,
            pnl_abs=pnl_abs, pnl_pct=pnl_pct,
            days_held=days_held,
            score_at_entry=pos.score_at_entry,
        ))

    def _close_all(self, asof: pd.Timestamp, prices: dict[str, float], reason: str) -> None:
        for sym, pos in list(self.positions.items()):
            if pos.status == PositionStatus.CLOSED or pos.qty_open <= 0:
                continue
            cp = prices.get(sym, pos.entry_price)
            from portfolio.models import ExitSignal
            sig = ExitSignal(
                symbol=sym, exit_type=ExitType.MANUAL,
                suggested_qty=pos.qty_open, current_price=cp,
                reason=reason,
                pnl_pct=(cp / pos.entry_price - 1) * 100,
                pnl_abs=(cp - pos.entry_price) * pos.qty_open,
            )
            self._execute_exit(pos, sig, asof)
