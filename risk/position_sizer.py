"""Volatility-adjusted, conviction-weighted position sizer.

Formula: weight = base × (score/70) × (target_vol / stock_vol)
Then capped by hard limits.
"""
from __future__ import annotations
from dataclasses import dataclass

from .interfaces import PositionSizer, SizingDecision, TradeCandidate


@dataclass
class SizerConfig:
    base_weight: float = 0.10           # 10% baseline
    target_volatility: float = 0.25     # 25% annualized
    max_single_weight: float = 0.15     # hard cap per position
    min_single_weight: float = 0.03     # below this, skip (not worth it)
    score_anchor: float = 70.0          # score that maps to 1.0× multiplier


class VolatilityAdjustedSizer:
    """Sizes positions by conviction × inverse volatility."""

    def __init__(self, config: SizerConfig | None = None) -> None:
        self.cfg = config or SizerConfig()

    def size(
        self,
        candidate: TradeCandidate,
        equity: float,
        open_weights: dict[str, float],
    ) -> SizingDecision:
        cfg = self.cfg
        if equity <= 0:
            return SizingDecision(candidate.symbol, 0, 0, 0, "no equity")

        conviction_mult = max(0.5, candidate.score / cfg.score_anchor)
        vol = candidate.annual_volatility or cfg.target_volatility
        vol_mult = cfg.target_volatility / max(vol, 0.05)
        weight = cfg.base_weight * conviction_mult * vol_mult
        weight = min(weight, cfg.max_single_weight)

        # Capacity check — don't overweight cluster of similar names
        existing = open_weights.get(candidate.symbol, 0.0)
        weight = max(0.0, weight - existing)

        if weight < cfg.min_single_weight:
            return SizingDecision(
                candidate.symbol, 0, 0, 0,
                f"below min weight {cfg.min_single_weight*100:.1f}%",
            )

        rupees = equity * weight
        qty = rupees / candidate.price if candidate.price > 0 else 0
        return SizingDecision(
            symbol=candidate.symbol,
            rupees_to_invest=round(rupees, 2),
            suggested_qty=round(qty, 4),
            weight_pct=round(weight * 100, 2),
            reasoning=(
                f"base {cfg.base_weight*100:.0f}% × score {conviction_mult:.2f} "
                f"× vol-adj {vol_mult:.2f} (vol {vol*100:.0f}%)"
            ),
        )
