"""Composable risk checks (Open/Closed: add new checks without editing the gate)."""
from __future__ import annotations
from dataclasses import dataclass

from .interfaces import RiskCheck, RiskResult, TradeCandidate, PortfolioContext


# ── Concrete checks ──────────────────────────────────────────────────────────
@dataclass
class ConcentrationCheck:
    name: str = "Concentration"
    max_single_weight: float = 0.15

    def evaluate(self, c: TradeCandidate, ctx: PortfolioContext) -> RiskResult:
        existing = ctx.weights_by_symbol.get(c.symbol, 0.0)
        proposed_weight = (c.rupees_intended / ctx.equity) if ctx.equity > 0 else 0
        total = existing + proposed_weight
        if total > self.max_single_weight:
            return RiskResult(False, "BLOCK",
                              f"{c.symbol} would be {total*100:.1f}% > cap {self.max_single_weight*100:.0f}%")
        return RiskResult(True, "INFO", f"{total*100:.1f}% within cap")


@dataclass
class SectorCheck:
    name: str = "SectorExposure"
    max_sector_weight: float = 0.25

    def evaluate(self, c: TradeCandidate, ctx: PortfolioContext) -> RiskResult:
        if not c.sector:
            return RiskResult(True, "INFO", "no sector")
        existing = ctx.weights_by_sector.get(c.sector, 0.0)
        proposed_weight = (c.rupees_intended / ctx.equity) if ctx.equity > 0 else 0
        total = existing + proposed_weight
        if total > self.max_sector_weight:
            return RiskResult(False, "BLOCK",
                              f"Sector {c.sector} would be {total*100:.1f}% > cap {self.max_sector_weight*100:.0f}%")
        return RiskResult(True, "INFO", f"Sector {c.sector} {total*100:.1f}% within cap")


@dataclass
class MarketCheck:
    name: str = "MarketExposure"
    max_market_weight: float = 0.70

    def evaluate(self, c: TradeCandidate, ctx: PortfolioContext) -> RiskResult:
        existing = ctx.weights_by_market.get(c.market, 0.0)
        proposed_weight = (c.rupees_intended / ctx.equity) if ctx.equity > 0 else 0
        total = existing + proposed_weight
        if total > self.max_market_weight:
            return RiskResult(False, "WARN",
                              f"Market {c.market} would be {total*100:.0f}% > soft cap {self.max_market_weight*100:.0f}%",
                              suggested_size_multiplier=0.5)
        return RiskResult(True, "INFO", f"Market {c.market} {total*100:.0f}% within cap")


@dataclass
class DrawdownCheck:
    name: str = "Drawdown"
    yellow: float = -5.0
    orange: float = -10.0
    red: float = -15.0

    def evaluate(self, c: TradeCandidate, ctx: PortfolioContext) -> RiskResult:
        dd = ctx.drawdown_pct
        if dd <= self.red:
            return RiskResult(False, "BLOCK", f"Portfolio DD {dd:.1f}% — no new buys")
        if dd <= self.orange:
            return RiskResult(True, "WARN", f"Portfolio DD {dd:.1f}% — half size",
                              suggested_size_multiplier=0.5)
        if dd <= self.yellow:
            return RiskResult(True, "WARN", f"Portfolio DD {dd:.1f}% — review", )
        return RiskResult(True, "INFO", f"DD {dd:.1f}%")


@dataclass
class RegimeCheck:
    """Scales position size by market regime."""
    name: str = "MarketRegime"

    def evaluate(self, c: TradeCandidate, ctx: PortfolioContext) -> RiskResult:
        if ctx.regime is None:
            return RiskResult(True, "INFO", "no regime data")
        mult = ctx.regime.allocation_multiplier
        if mult <= 0.2:
            return RiskResult(False, "BLOCK", f"Regime {ctx.regime.label} — defensive only")
        sev = "INFO" if mult >= 0.85 else "WARN"
        return RiskResult(True, sev,
                          f"Regime {ctx.regime.label} ({ctx.regime.score}/10) → size×{mult:.2f}",
                          suggested_size_multiplier=mult)


@dataclass
class SectorMomentumCheck:
    """Block buys from bottom-quartile sectors (Phase B)."""
    name: str = "SectorMomentum"
    block_quartile: int = 4

    def evaluate(self, c: TradeCandidate, ctx: PortfolioContext) -> RiskResult:
        sr = ctx.sector_rankings.get(c.sector)
        if sr is None:
            return RiskResult(True, "INFO", "sector not ranked")
        if sr.quartile >= self.block_quartile:
            return RiskResult(False, "BLOCK",
                              f"Sector {c.sector} in bottom quartile (rank {sr.rank})")
        if sr.quartile == 1:
            return RiskResult(True, "INFO",
                              f"Sector {c.sector} top quartile — favorable",
                              suggested_size_multiplier=1.1)
        return RiskResult(True, "INFO", f"Sector quartile {sr.quartile}")


@dataclass
class CorrelationClusterCheck:
    """Block if joining a cluster pushes its weight beyond cap (Phase B)."""
    name: str = "CorrelationCluster"
    max_cluster_weight: float = 0.30

    def evaluate(self, c: TradeCandidate, ctx: PortfolioContext) -> RiskResult:
        for cluster in ctx.correlation_clusters:
            if c.symbol in cluster.members:
                proposed_weight = (c.rupees_intended / ctx.equity) if ctx.equity > 0 else 0
                total = cluster.total_weight_pct / 100 + proposed_weight
                if total > self.max_cluster_weight:
                    return RiskResult(False, "BLOCK",
                                      f"Cluster {cluster.members[:3]}... would be {total*100:.0f}% "
                                      f"> cap {self.max_cluster_weight*100:.0f}%")
        return RiskResult(True, "INFO", "no cluster breach")
