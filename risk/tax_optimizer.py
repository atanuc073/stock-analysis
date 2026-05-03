"""Tax-aware exit advisor for Indian markets.

Rules (FY26):
  - STCG (< 12 months): 20%
  - LTCG (≥ 12 months): 12.5% (₹1.25L exempt per year)
US: separate logic — short-term taxed as ordinary income, long-term ≥12mo at preferential rates.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class TaxAdvice:
    symbol: str
    days_held: int
    days_to_ltcg: int
    current_pnl_pct: float
    current_pnl_abs: float
    short_term_tax: float
    long_term_tax: float
    tax_savings_if_wait: float
    recommendation: str          # SELL_NOW | DEFER_FOR_LTCG | NO_PREFERENCE
    reasoning: str


@dataclass
class IndianTaxConfig:
    stcg_rate: float = 0.20
    ltcg_rate: float = 0.125
    ltcg_holding_days: int = 365
    annual_ltcg_exemption: float = 125_000.0
    defer_threshold_days: int = 60       # only suggest deferral if within N days of LTCG


class IndianTaxOptimizer:
    """Advises whether to defer an exit to qualify for LTCG."""

    def __init__(self, config: IndianTaxConfig | None = None,
                 ltcg_used_this_year: float = 0.0) -> None:
        self.cfg = config or IndianTaxConfig()
        self._ltcg_used = ltcg_used_this_year

    def advise(
        self,
        symbol: str,
        entry_date_iso: str,
        entry_price: float,
        current_price: float,
        qty: float,
        today: Optional[date] = None,
        is_indian: bool = True,
    ) -> Optional[TaxAdvice]:
        if not is_indian:
            return None  # US logic differs; out of scope for Indian-specific optimizer
        try:
            entry_dt = date.fromisoformat(entry_date_iso)
        except Exception:
            return None
        today = today or date.today()
        days_held = (today - entry_dt).days
        days_to_ltcg = max(0, self.cfg.ltcg_holding_days - days_held)

        gain_abs = (current_price - entry_price) * qty
        gain_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0

        if gain_abs <= 0:
            return TaxAdvice(symbol, days_held, days_to_ltcg, gain_pct, gain_abs,
                             0.0, 0.0, 0.0, "NO_PREFERENCE",
                             "loss — tax neutral; consider harvesting against gains")

        st_tax = gain_abs * self.cfg.stcg_rate
        # LTCG with exemption headroom
        remaining_exempt = max(0.0, self.cfg.annual_ltcg_exemption - self._ltcg_used)
        taxable_lt = max(0.0, gain_abs - remaining_exempt)
        lt_tax = taxable_lt * self.cfg.ltcg_rate
        savings = st_tax - lt_tax

        if days_held >= self.cfg.ltcg_holding_days:
            rec = "SELL_NOW"
            reason = f"Already LTCG-eligible (held {days_held}d). Tax: ₹{lt_tax:,.0f}"
        elif 0 < days_to_ltcg <= self.cfg.defer_threshold_days and savings > 0:
            rec = "DEFER_FOR_LTCG"
            reason = (f"Only {days_to_ltcg} days to LTCG. "
                      f"Defer saves ₹{savings:,.0f} tax (₹{st_tax:,.0f} → ₹{lt_tax:,.0f})")
        else:
            rec = "SELL_NOW"
            reason = f"{days_to_ltcg}d to LTCG — too far to defer; STCG ₹{st_tax:,.0f}"

        return TaxAdvice(
            symbol=symbol, days_held=days_held, days_to_ltcg=days_to_ltcg,
            current_pnl_pct=gain_pct, current_pnl_abs=gain_abs,
            short_term_tax=st_tax, long_term_tax=lt_tax,
            tax_savings_if_wait=savings, recommendation=rec, reasoning=reason,
        )
