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
    # Late-cycle / top-chase guards (data-driven defaults from May'26 audit):
    #  - >25% above 200DMA: forward-30D returns turn negative & win-rate <50%
    #  - At/within 0% of 52WH (no pullback): forward-30D avg only +0.48%
    # Allow buys at the high ONLY when a confirmed breakout fires today.
    max_extension_pct: float = 25.0            # max % stock can be extended above 200DMA (default 25%)
    # Top-chase audit (May'26): the -5% to 0% "near-high but not breakout"
    # zone is the worst forward-return cohort (62% win, +0.85% avg fwd-30D)
    # while deep pullbacks (-25% to -15%) returned +4.4% avg fwd-30D with
    # 100% win-rate. Tightened from -1.5 → -5.0 to skip the noisy middle
    # band; confirmed breakouts still get through via require_breakout_at_high.
    max_pct_from_52w_high: float = 0.0         # require pullback from 52WH (default 0.0 = disabled)
    require_breakout_at_high: bool = True      # if at the high, demand breakout_today=True from uptrend.compute
    require_uptrend_confirm: bool = True       # demand Stage-2, Trend Template, or U/D volume confirmations
    # Relative-strength entry filter (IBD-style RS percentile rank, 12-1
    # momentum cross-sectional). 0 = disabled (legacy behaviour). 70 = only
    # buy stocks in the top 30% of the candidate universe by momentum.
    # Even when disabled as a hard filter, the RS pass still applies a
    # +10/+7/+4/-5 score bump (see scoring.apply_rs_to_bt), so RS influences
    # ranking either way.
    min_rs_pct: float = 70
    max_market_weight: float = 0.99           # max country concentration weight (default 70%)
    transaction_cost_bps: float = 0.0          # 0 bps each side (0.00%)
    slippage_bps: float = 0.0                  # 0 bps slippage
    include_forecast: bool = False             # forecast slow; off by default
    live_weights: bool = True                   # use live SCORE_WEIGHTS (no redistribution)
    use_regime: bool = True                     # enable market regime filter
    regime_skip_below: str = "NONE"            # skip entries at or below this regime (NONE = disabled)
    regime_check_freq_days: int = 5            # re-evaluate regime every N days
    # Regime-driven de-risking of OPEN positions (not just new entries):
    # When regime label is at or below this floor, trim each open position
    # toward `regime_derisk_target_mult` of its current size on rebalance days.
    regime_derisk_below: Optional[str] = "NONE"   # only trim in CAUTIOUS/BEAR (NONE = disabled)
    regime_derisk_target_mult: float = 0.65           # keep 65% of size in CAUTIOUS/BEAR (gentler trim)
    ignore_cash_floor: bool = False                   # completely bypass cash floors to stay fully invested
    # ── Bear-market capital preservation knobs ─────────────────────────────
    # Raise min_score bar in weak regimes (only A+ setups get through).
    # Bull/sideways: no bump (deploy freely). CAUTIOUS/BEAR: real bar.
    regime_min_score_bumps: dict = field(default_factory=lambda: {
        "BULL": 0.0,
        "NEUTRAL_BULL": 0.0,
        "NEUTRAL": 0.0,
        "CAUTIOUS": 0.0,
        "BEAR": 0.0,
    })
    # Force a minimum cash level (% of equity) per regime label.
    # Policy: full deploy in BULL/NEUTRAL_BULL/NEUTRAL (sideways), light
    # cash in CAUTIOUS, real defense in BEAR.
    regime_cash_floor: dict = field(default_factory=lambda: {
        "BULL": 0.0,
        "NEUTRAL_BULL": 0.0,
        "NEUTRAL": 0.0,
        "CAUTIOUS": 0.0,
        "BEAR": 0.0,
    })
    # Tighten the per-position hard stop in weak regimes (multiplier on
    # configured hard_stop_pct; 1.0 = no change, 0.7 = 30% less room).
    # Relaxed May'26: old 0.65/0.50 caused 49% stop-out rate (shakeouts).
    # Prefer smaller POSITION SIZE (regime_derisk) over tighter stops.
    regime_stop_tighten: dict = field(default_factory=lambda: {
        "BULL": 1.0,
        "NEUTRAL_BULL": 1.0,
        "NEUTRAL": 1.0,
        "CAUTIOUS": 1.0,
        "BEAR": 1.0,
    })
    # Shock-day stop suppression: if VIX day-over-day jumps >= shock_vix_jump
    # AND the broad index drops <= shock_index_drop on the same day, treat
    # it as a panic day and skip soft stop-loss exits (still honours deep
    # breaches via lifecycle.hard_stop_buffer).
    shock_vix_jump: float = 0.30               # +30% one-day VIX jump
    shock_index_drop: float = -0.03            # -3% one-day index drop
    weights: Optional[dict] = None              # per-run override of SCORE_WEIGHTS (None = use config)
    # ── SIP (Systematic Investment Plan) ────────────────────────────────────
    # When sip_amount > 0: inject `sip_amount` cash on the first trading day
    # at/after `sip_day_of_month` each calendar month. Sell proceeds still
    # recycle into the cash pool (no change to existing reinvestment logic).
    sip_amount: float = 0.0                     # 0 disables SIP (lumpsum mode)
    sip_day_of_month: int = 13
    uptrend_mode: bool = False                  # if True, use s.uptrend_score for entries
    warmup_days: int = 0                       # skip new entries for first N days (default 0)
    # Universe label (e.g. "nifty500", "russell1000"). Routes to the right
    # WFO weight schedule when sub_decomp is enabled. Leave blank to use the
    # default Nifty500 schedule.
    universe: str = ""


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
    uptrend_score_at_entry: float = 0.0
    # ── Entry-context diagnostics (populated on BUY only) ───────────────
    pct_above_sma200_at_entry: float = 0.0    # extension above 200DMA
    pct_from_52w_high_at_entry: float = 0.0   # negative; 0 = at high
    rsi_at_entry: float = 0.0
    ret_3m_at_entry: float = 0.0
    ret_6m_at_entry: float = 0.0
    ret_1y_at_entry: float = 0.0
    regime_label_at_entry: str = ""


@dataclass
class EquityPoint:
    date: pd.Timestamp
    cash: float
    market_value: float
    total: float
    n_open: int
    contribution: float = 0.0   # cash injected on this date (SIP), 0 otherwise
    regime: Optional[regime_mod.HistoricalRegime] = None


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
    # SIP cash flows: list of (date, amount) including initial seed.
    # Used by reporter for XIRR + benchmark SIP replay.
    contributions: list[tuple[pd.Timestamp, float]] = field(default_factory=list)


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
        # SIP state
        self.contributions: list[tuple[pd.Timestamp, float]] = []
        self._sip_paid_months: set[tuple[int, int]] = set()  # (year, month) keys

        # Regime cache: avoid recomputing every rebalance
        self._regime_cache: dict[tuple[pd.Timestamp, str], regime_mod.HistoricalRegime] = {}
        self._last_regime_compute: Optional[pd.Timestamp] = None
        self._current_regime: dict[str, regime_mod.HistoricalRegime] = {}
        
        # Re-entry lock: track last stop-loss date per symbol
        self._last_stop_loss_date: dict[str, pd.Timestamp] = {}

        # Dynamic parameter updates
        self.dynamic_weights_csv: Optional[str] = None
        self.dynamic_weights_df: Optional[pd.DataFrame] = None
        self._last_weights_month: Optional[str] = None

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self, dates: pd.DatetimeIndex) -> BacktestResult:
        if len(dates) == 0:
            raise ValueError("No trading dates provided")

        if self.dynamic_weights_csv:
            try:
                self.dynamic_weights_df = pd.read_csv(self.dynamic_weights_csv)
                log.info("Loaded dynamic weights from %s", self.dynamic_weights_csv)
            except Exception as e:
                log.error("Failed to load dynamic weights from %s: %s", self.dynamic_weights_csv, e)

        log.info("Backtest: %s → %s, %d days, %d symbols, capital ₹%s",
                 dates[0].date(), dates[-1].date(), len(dates),
                 len(self.data), f"{self.cfg.initial_capital:,.0f}")
        if self.cfg.sip_amount > 0:
            log.info("SIP enabled: ₹%s/month on day %d (or next trading day)",
                     f"{self.cfg.sip_amount:,.0f}", self.cfg.sip_day_of_month)

        # Record initial seed (if any) as a t0 cash flow for XIRR
        if self.cfg.initial_capital > 0:
            self.contributions.append((dates[0], float(self.cfg.initial_capital)))

        rebalance_dates = set(dates[::self.cfg.rebalance_freq_days])
        rebalance_dates.add(dates[-1])  # always close on last day

        # Warmup cutoff: no new entries before this date (regime detector/indicators need runway)
        warmup_cutoff_idx = min(self.cfg.warmup_days, max(0, len(dates) - 1))
        self._warmup_cutoff = dates[warmup_cutoff_idx] if warmup_cutoff_idx > 0 else dates[0]
        if self.cfg.warmup_days > 0:
            log.info("Warmup gate: no new entries until %s (first %d trading days)",
                     self._warmup_cutoff.date(), self.cfg.warmup_days)

        for asof in tqdm(dates, desc="Simulating"):
            # Update dynamic monthly weights if configured
            if self.dynamic_weights_df is not None:
                month_key = asof.strftime("%Y-%m")
                if self._last_weights_month != month_key:
                    self._last_weights_month = month_key
                    row = self.dynamic_weights_df[self.dynamic_weights_df["test_start"] == month_key]
                    if not row.empty:
                        row_data = row.iloc[0]
                        # Safely load w_technical if it exists in the CSV, otherwise default to 0.0
                        w_t = float(row_data["w_technical"]) if "w_technical" in row_data else 0.0
                        w_q = float(row_data["w_quality"])
                        w_f = float(row_data["w_fundamental"])
                        w_m = float(row_data["w_momentum"])
                        w_ed = float(row_data["w_earnings_drift"])
                        total = w_t + w_q + w_f + w_m + w_ed
                        if total > 0:
                            self.cfg.weights = {
                                "technical": w_t / total,
                                "fundamental": w_f / total,
                                "momentum": w_m / total,
                                "quality": w_q / total,
                                "earnings_drift": w_ed / total,
                                "sentiment": 0.0,
                                "options": 0.0,
                                "forecast": 0.0,
                                "valuation": 0.0,
                            }
                            log.debug("Dynamic weights for %s: T=%.2f%%, Q=%.2f%%, ED=%.2f%%, M=%.2f%%, F=%.2f%%",
                                      month_key, (w_t/total)*100, (w_q/total)*100, (w_ed/total)*100, (w_m/total)*100, (w_f/total)*100)

            # Update regime daily (internally respects frequency checks)
            self._refresh_regime(asof)

            # 0) SIP injection: first trading day of month at/after sip_day
            sip_amount_today = 0.0
            if self.cfg.sip_amount > 0:
                month_key = (asof.year, asof.month)
                if (month_key not in self._sip_paid_months
                        and asof.day >= self.cfg.sip_day_of_month):
                    self.cash += self.cfg.sip_amount
                    sip_amount_today = self.cfg.sip_amount
                    self.contributions.append((asof, float(self.cfg.sip_amount)))
                    self._sip_paid_months.add(month_key)

            # 1) Update prices on all open positions, update peaks
            current_prices = self._current_prices(asof)
            for sym, pos in list(self.positions.items()):
                cp = current_prices.get(sym)
                if cp is not None:
                    ExitEvaluator.update_peak(pos, cp, self.exit_eval.cfg.trail_stop_pct)

            # 1b) Detect macro shock day (VIX spike + index drop)
            shock = self._is_shock_day(asof)

            # 1c) Gap-aware stop check: if today's intraday LOW pierces the stop,
            # close at the worse of (open, stop) instead of waiting for close.
            # This catches MO-style -13% losses where the stop was breached on a gap.
            if not shock:
                self._process_gap_stops(asof, current_prices)

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
                contribution=sip_amount_today,
                regime=self._worst_regime(),
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
            regime=self._worst_regime(),
        ))

        return BacktestResult(
            config=self.cfg,
            equity_curve=self.equity_curve,
            trades=self.trades,
            final_positions=list(self.positions.values()),
            universe_size=len(self.data),
            start=str(dates[0].date()),
            end=str(dates[-1].date()),
            contributions=self.contributions,
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

    def _worst_regime_label(self) -> str:
        """Most defensive regime label across markets (BEAR > CAUTIOUS > ...)."""
        if not self.cfg.use_regime or not self._current_regime:
            return "BULL"
        order = ["BEAR", "CAUTIOUS", "NEUTRAL", "NEUTRAL_BULL", "BULL"]
        worst_idx = len(order) - 1
        for r in self._current_regime.values():
            try:
                worst_idx = min(worst_idx, order.index(r.label))
            except ValueError:
                continue
        return order[worst_idx]

    def _worst_regime(self) -> Optional[regime_mod.HistoricalRegime]:
        """HistoricalRegime object corresponding to the most defensive regime across markets."""
        if not self.cfg.use_regime or not self._current_regime:
            return None
        order = ["BEAR", "CAUTIOUS", "NEUTRAL", "NEUTRAL_BULL", "BULL"]
        worst_r = None
        worst_idx = len(order)
        for r in self._current_regime.values():
            try:
                idx = order.index(r.label)
                if idx < worst_idx:
                    worst_idx = idx
                    worst_r = r
            except ValueError:
                continue
        return worst_r

    def _regime_min_score(self, market: str = "") -> float:
        """min_score bumped up by regime weakness.

        If `market` is given, use that market's own regime (independent).
        Else fall back to worst-case across markets (legacy behaviour).
        """
        if market and self.cfg.use_regime and self._current_regime:
            r = self._current_regime.get(market)
            if r is not None:
                bump = self.cfg.regime_min_score_bumps.get(r.label, 0.0)
                return self.cfg.min_score + bump
        bump = self.cfg.regime_min_score_bumps.get(self._worst_regime_label(), 0.0)
        return self.cfg.min_score + bump

    def _regime_cash_floor_frac(self) -> float:
        """Required cash as % of equity given current regime.

        Uses the AVERAGE cash-floor across markets rather than worst-case so
        a single bear market doesn't force the whole book to cash.
        """
        if self.cfg.ignore_cash_floor:
            return 0.0
        if not self.cfg.use_regime or not self._current_regime:
            return self.cfg.regime_cash_floor.get("BULL", 0.0)
        floors = [
            self.cfg.regime_cash_floor.get(r.label, 0.0)
            for r in self._current_regime.values()
        ]
        return (sum(floors) / len(floors)) if floors else 0.0

    def _regime_stop_mult(self, market: str) -> float:
        r = self._current_regime.get(market)
        if r is None:
            return 1.0
        return self.cfg.regime_stop_tighten.get(r.label, 1.0)

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
                uptrend_score_at_entry=pos.uptrend_score_at_entry,
                regime_label_at_entry=pos.regime_label_at_entry,
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

        # Warmup gate check
        if getattr(self, "_warmup_cutoff", None) is not None and asof < self._warmup_cutoff:
            return

        # Score every symbol with sufficient history
        scores: list[BacktestScore] = []
        for sym, hd in self.data.items():
            if sym in self.positions and self.positions[sym].status != PositionStatus.CLOSED:
                continue  # skip open positions
            
            # Re-entry lock: skip if stopped out in the last 30 days
            last_stop = self._last_stop_loss_date.get(sym)
            if last_stop is not None and (asof - last_stop).days < 30:
                continue
                
            if self._regime_blocks_entry(hd.market):
                continue  # regime says: no new entries in this market
            r_obj = self._current_regime.get(hd.market)
            regime_lbl = r_obj.label if r_obj is not None else "NEUTRAL"
            s = score_at(hd, asof,
                         include_forecast=self.cfg.include_forecast,
                         live_weights=self.cfg.live_weights,
                         weights_override=self.cfg.weights,
                         regime_label=regime_lbl)
            if s is None:
                continue
            # Hard block 1: skip stocks extended too far above 200DMA (late-cycle
            # blow-off tops). Try technical first; fall back to uptrend's own
            # computation if technical didn't expose the metric.
            ext = s.technical.get("pct_above_sma200")
            if ext is None or ext == 0.0:
                up = getattr(s, "uptrend_data", None) or {}
                price = s.price or 0.0
                s200 = (up.get("sma200") if isinstance(up, dict) else None)
                if s200 and price > 0:
                    ext = (price / s200 - 1.0) * 100.0
                else:
                    ext = 0.0
            if ext > self.cfg.max_extension_pct:
                continue

            # Hard block 2: top-chase guard. Skip entries at the 52-week high
            # unless a confirmed breakout (volume + close-in-upper-range) fires.
            up = getattr(s, "uptrend_data", None) or {}
            pct_from_high = up.get("pct_from_52w_high", s.technical.get("pct_from_52w_high", -100.0))
            breakout_today = bool(up.get("breakout_today", False))
            if pct_from_high > self.cfg.max_pct_from_52w_high:
                continue

            # Hard block 3: uptrend quality gate. Require multiple structural
            # confirmations before committing capital (Minervini/O'Neil style).
            # Must satisfy at least 2 of: Stage 2, trend template ≥5, U/D ≥1.0
            uptrend_confirms = 0
            if up.get("stage2"):
                uptrend_confirms += 1
            if (up.get("trend_template") or 0) >= 5:
                uptrend_confirms += 1
            if (up.get("ud_50") or 0) >= 1.0:
                uptrend_confirms += 1
            if breakout_today:
                uptrend_confirms += 1  # breakout bypasses — always strong
            if self.cfg.require_uptrend_confirm and uptrend_confirms < 2:
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

        # Cross-sectional RS percentile + sector strength (mirrors the live
        # screener's uptrend.apply_rs). Updates s.uptrend_score and
        # s.adjusted_score in place with RS decile bump + sector bump.
        try:
            from .scoring import apply_rs_to_bt
            apply_rs_to_bt(scores)
        except Exception as e:
            log.warning("RS percentile pass failed: %s", e)

        # Sub-decomposition pass (12-feature WFO-trained blend). When
        # USE_SUB_DECOMP is enabled this OVERWRITES adjusted_score with the
        # 12-feature blended score so backtest scoring is identical to the
        # live screener path in composite.py. Weights are blended per asof
        # date (look-ahead safe) and routed to the right CSV via
        # cfg.universe (e.g. "nifty500" → reports/regression/sub/...,
        # "russell1000" → reports/regression/russell1000_sub/sub/...).
        # Disable by setting env USE_SUB_DECOMP=0.
        try:
            from config import USE_SUB_DECOMP  # type: ignore
        except Exception:
            USE_SUB_DECOMP = True
        if USE_SUB_DECOMP and self.cfg.weights is None:
            try:
                from analysis.sub_decomp import apply_sub_decomp, get_weights_for_date
                w_today = get_weights_for_date(asof, universe=self.cfg.universe)
                # Score US and IN separately so cross-sectional ranks reflect
                # the cohort each stock actually competes in.
                us_scores = [s for s in scores if getattr(s, "market", "") == "US"]
                in_scores = [s for s in scores if getattr(s, "market", "") != "US"]
                if us_scores:
                    apply_sub_decomp(us_scores, weights=w_today)
                if in_scores:
                    apply_sub_decomp(in_scores, weights=w_today)
            except Exception as e:
                log.warning("sub_decomp pass failed at %s: %s", asof.date(), e)

        # Optional hard filter: require RS percentile >= configured floor.
        # Default 0 = disabled; set min_rs_pct=70 to buy only leaders.
        if self.cfg.min_rs_pct > 0:
            scores = [s for s in scores
                      if float((s.uptrend_data or {}).get("rs_pct", 0.0))
                      >= self.cfg.min_rs_pct]
            if not scores:
                return

        # Min score is bumped up in weak regimes (capital preservation).
        # Per-market so a strong US bull is not dragged down by an IN bear.
        score_attr = "uptrend_score" if self.cfg.uptrend_mode else "adjusted_score"
        scores = [
            s for s in scores
            if getattr(s, score_attr) >= self._regime_min_score(getattr(s, "market", ""))
        ]
        if not scores:
            return
        scores.sort(key=lambda s: getattr(s, score_attr), reverse=True)

        equity = self.cash + self._market_value(current_prices)
        sector_weights = self._sector_weights(current_prices, equity)
        market_weights = self._market_weights(current_prices, equity)

        # Slots = remaining capacity (up to hard_cap), driven by both
        # position count AND idle cash. After a T1/T2 fires, n_open stays
        # the same but cash rises — we still allow new entries via hard_cap.
        slots = hard_cap - n_open

        # Regime cash floor: required cash reserve as % of equity.
        # Stops new entries from breaching the floor (capital preservation).
        regime_cash_required = equity * self._regime_cash_floor_frac()

        for s in scores:
            if slots <= 0:
                break
            if self.cash <= cash_floor:
                break
            # Don't deploy below regime cash floor
            if self.cash <= regime_cash_required:
                break
            # Concentration checks
            if sector_weights.get(s.sector, 0.0) >= self.cfg.max_sector_weight:
                continue
            if market_weights.get(s.market, 0.0) >= self.cfg.max_market_weight:
                continue

            # Cash-buffered slot sizing (Method 1): deploy available cash
            # dynamically based on the number of empty slots.
            # Avoid splitting cash into tiny fragments when cash is scarce (like in SIP mode)
            # or when equity is large. Instead, slot_cash is sized to target the base weight,
            # capped by available deployable cash.
            deployable_cash = max(self.cash - regime_cash_required, 0.0)
            target_base_cash = equity * self.cfg.base_position_weight
            slot_cash = max(deployable_cash / max(slots, 1), min(deployable_cash, target_base_cash))
            slot_w = slot_cash / equity

            vol_adj = min(0.25 / max(s.annual_vol, 0.05), 1.5)
            score_adj = min(s.adjusted_score / 70.0, 1.3)
            regime_adj = self._regime_multiplier(s.market)
            target_w = min(
                slot_w * vol_adj * score_adj * regime_adj,
                self.cfg.max_position_weight,
            )
            target_rupees = equity * target_w

            # Determine a dynamic minimum trade size floor to avoid the cash drag deadlock.
            # In SIP mode or when cash is scarce, we allow smaller absolute sizes (down to Rs. 10k).
            min_size_floor = min(equity * 0.02, 10000.0)
            if self.cfg.sip_amount > 0:
                min_size_floor = min(min_size_floor, self.cfg.sip_amount * 0.5)

            if target_rupees < min_size_floor:    # too small to bother
                continue
            # Cap by available cash above regime floor
            deployable_cash = max(self.cash - regime_cash_required, 0.0)
            if target_rupees > deployable_cash * 0.95:
                target_rupees = deployable_cash * 0.95
            if target_rupees < min_size_floor:
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
        # Capture current regime label BEFORE creating position so it gets
        # persisted on the Position (used later by SELL trades + reporter).
        regime_label = ""
        r = self._current_regime.get(s.market)
        if r is not None:
            regime_label = r.label
        pos = self.factory.create(
            symbol=s.symbol, qty=qty, entry_price=s.price,
            atr=s.atr_value, sector=s.sector, market=s.market,
            score=s.adjusted_score, uptrend_score=s.uptrend_score,
            entry_date=asof.strftime("%Y-%m-%d"),
            regime_label=regime_label,
        )
        
        # Override with "Smart Stop" if available
        if s.suggested_stop and s.suggested_stop < s.price:
            pos.stop_price = s.suggested_stop
            pos.initial_stop_price = s.suggested_stop
            pos.notes = f"Stop via {s.stop_method}"

        # Tighten stop in weak regimes
        # regime stop multiplier. Multiplier < 1.0 means a tighter stop
        # (smaller loss tolerance) when regime is degraded.
        stop_mult = self._regime_stop_mult(s.market)
        if stop_mult < 1.0 and pos.stop_price and pos.stop_price < s.price:
            loss_dist = s.price - pos.stop_price
            new_stop = s.price - loss_dist * stop_mult
            pos.stop_price = new_stop
        self.positions[s.symbol] = pos
        self.cash -= cost

        score_attr = "uptrend_score" if self.cfg.uptrend_mode else "adjusted_score"
        self.trades.append(BTTrade(
            symbol=s.symbol, action="BUY", qty=qty, price=s.price,
            gross_value=s.price * qty,
            cost=cost - s.price * qty,
            net_value=cost,
            timestamp=asof.isoformat(),
            sector=s.sector, market=s.market,
            reason=f"{'UP' if self.cfg.uptrend_mode else 'adj'}={getattr(s, score_attr):.1f}",
            score_at_entry=s.adjusted_score,
            uptrend_score_at_entry=s.uptrend_score,
            pct_above_sma200_at_entry=float(s.technical.get("pct_above_sma200", 0.0) or 0.0),
            pct_from_52w_high_at_entry=float(s.technical.get("pct_from_52w_high", 0.0) or 0.0),
            rsi_at_entry=float(s.technical.get("rsi", 0.0) or 0.0),
            ret_3m_at_entry=float(s.momentum.get("ret_3m", 0.0) or 0.0),
            ret_6m_at_entry=float(s.momentum.get("ret_6m", 0.0) or 0.0),
            ret_1y_at_entry=float(s.momentum.get("ret_1y", 0.0) or 0.0),
            regime_label_at_entry=regime_label,
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
    def _process_gap_stops(self, asof: pd.Timestamp,
                           current_prices: dict[str, float]) -> None:
        """Force a STOP_LOSS exit when today's intraday LOW pierced the stop.

        Fills at the worse of (open_price, stop_price): if the stock gapped
        below the stop, you'd realistically have been filled at the open;
        otherwise (intraday breach), the stop would trigger near the stop.
        Skipped on shock days (handled by `shock_vix_jump` suppression).
        """
        for sym, pos in list(self.positions.items()):
            if pos.status == PositionStatus.CLOSED or pos.qty_open <= 0:
                continue
            stop = float(pos.stop_price or 0.0)
            if stop <= 0:
                continue
            hd = self.data.get(sym)
            if hd is None or hd.history.empty:
                continue
            try:
                # Find today's bar (or last available <= asof)
                idx = hd.history.index
                if idx.tz is not None:
                    bars = hd.history[idx.tz_localize(None) <= asof]
                else:
                    bars = hd.history[idx <= asof]
                if bars.empty:
                    continue
                last = bars.iloc[-1]
                low = float(last.get("Low", last.get("Close", 0.0)) or 0.0)
                open_ = float(last.get("Open", last.get("Close", 0.0)) or 0.0)
            except Exception:
                continue
            if low <= 0 or low > stop:
                continue  # stop not breached today
            # Fill at worse of open (gap down) or stop (intraday)
            fill = stop if open_ >= stop else open_

            from portfolio.models import ExitSignal
            sig = ExitSignal(
                symbol=sym, exit_type=ExitType.STOP_LOSS,
                current_price=fill, suggested_qty=pos.qty_open,
                reason=f"STOP_LOSS (gap-aware fill @ {fill:.2f}, stop {stop:.2f}, low {low:.2f})",
            )
            self._execute_exit(pos, sig, asof)
            # Update price cache so downstream MV/sizing reflects exit
            current_prices[sym] = fill

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
            hist_slice = None
            hd = self.data.get(sym)
            if hd is not None:
                try:
                    hist_slice = hd.history.loc[:asof]
                except Exception:
                    hist_slice = None
            if evaluate_thesis and hd is not None:
                r_obj = self._current_regime.get(hd.market)
                regime_lbl = r_obj.label if r_obj is not None else "NEUTRAL"
                s = score_at(hd, asof,
                             include_forecast=False,
                             live_weights=self.cfg.live_weights,
                             weights_override=self.cfg.weights,
                             regime_label=regime_lbl)
                if s:
                    # Use raw composite for thesis-break (cross-sectional
                    # context is unavailable for a single ticker on exit).
                    current_score = s.score

            signals = self.exit_eval.evaluate(
                pos, cp, current_score=current_score, red_flags=0,
                today=asof.date(), regime_shock=regime_shock,
                history=hist_slice,
            )
            for sig in signals:
                self._execute_exit(pos, sig, asof)

    def _execute_exit(self, pos: Position, sig, asof: pd.Timestamp) -> None:
        qty = min(sig.suggested_qty, pos.qty_open)

        # Adaptive-tier no-sell path: tier triggered, but strength score said
        # "hold full". Mark tier as triggered + raise stop without selling.
        if qty <= 0 and sig.exit_type in (ExitType.TIER_1, ExitType.TIER_2):
            tier_idx = 0 if sig.exit_type == ExitType.TIER_1 else 1
            if tier_idx < len(pos.tiers):
                t = pos.tiers[tier_idx]
                t.triggered = True
                t.triggered_on = asof.isoformat()
                t.fill_price = sig.current_price
            if sig.new_stop_price is not None and sig.new_stop_price > pos.stop_price:
                pos.stop_price = sig.new_stop_price
            return

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

        # Re-label a STOP_LOSS exit when prior tier/trail ratchets had already
        # raised the stop above breakeven. The "stop" being hit now is really
        # the profit-lock from an earlier tier, not a real loss. Reporting it
        # as STOP_LOSS pollutes the loss bucket and inflates stop counts.
        #   - Tier 2 triggered  → label TIER_2
        #   - Tier 1 triggered  → label TIER_1
        #   - Trail active (stop > entry, no tiers yet) → label TRAILING
        effective_exit_type = sig.exit_type
        if sig.exit_type == ExitType.STOP_LOSS:
            t2_hit = len(pos.tiers) >= 2 and pos.tiers[1].triggered
            t1_hit = len(pos.tiers) >= 1 and pos.tiers[0].triggered
            if t2_hit:
                effective_exit_type = ExitType.TIER_2
            elif t1_hit:
                effective_exit_type = ExitType.TIER_1
            elif pos.stop_price > pos.entry_price:
                # Stop was ratcheted above entry without a tier hit ⇒ trailing.
                effective_exit_type = ExitType.TRAILING

        self.trades.append(BTTrade(
            symbol=pos.symbol, action="SELL", qty=qty, price=sig.current_price,
            gross_value=gross, cost=cost, net_value=net,
            timestamp=asof.isoformat(),
            sector=pos.sector, market=pos.market,
            reason=sig.reason,
            exit_type=effective_exit_type.value,
            pnl_abs=pnl_abs, pnl_pct=pnl_pct,
            days_held=days_held,
            score_at_entry=pos.score_at_entry,
            uptrend_score_at_entry=pos.uptrend_score_at_entry,
            regime_label_at_entry=pos.regime_label_at_entry,
        ))

        # Record stop-loss date for re-entry lock — only on REAL losses
        # (i.e. unchanged STOP_LOSS, not a relabeled tier/trail profit-lock).
        if effective_exit_type == ExitType.STOP_LOSS:
            self._last_stop_loss_date[pos.symbol] = asof

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