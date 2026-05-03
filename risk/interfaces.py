"""Protocols defining the risk module's contracts.

Concrete implementations live in sibling modules; consumers depend only on these.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional


# ── Shared result types ──────────────────────────────────────────────────────
@dataclass
class RiskResult:
    """Result of a risk check on a candidate trade or portfolio state."""
    passed: bool
    severity: str = "INFO"          # INFO | WARN | BLOCK
    message: str = ""
    suggested_size_multiplier: float = 1.0  # checks may scale rather than block


@dataclass
class SizingDecision:
    symbol: str
    rupees_to_invest: float
    suggested_qty: float
    weight_pct: float
    reasoning: str = ""


@dataclass
class RegimeReport:
    score: int                       # 0..10
    label: str                       # BULL | NEUTRAL_BULL | NEUTRAL | CAUTIOUS | BEAR
    allocation_multiplier: float     # 0.0..1.0
    components: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class SectorRanking:
    sector: str
    rs_3m: float
    rs_6m: float
    composite: float
    rank: int
    quartile: int                    # 1 (best) - 4 (worst)


@dataclass
class CorrelationCluster:
    members: list[str]
    avg_correlation: float
    total_weight_pct: float


@dataclass
class RedFlag:
    symbol: str
    code: str
    severity: str                    # INFO | WARN | CRITICAL
    message: str


# ── Candidate trade view ─────────────────────────────────────────────────────
@dataclass
class TradeCandidate:
    symbol: str
    sector: str
    market: str                      # IN | US
    score: float
    price: float
    atr: float
    annual_volatility: float         # e.g. 0.30 = 30%
    rupees_intended: float = 0.0     # populated by sizer


# ── Protocols (DIP — depend on these, not the impls) ─────────────────────────
class PositionSizer(Protocol):
    def size(self, candidate: TradeCandidate, equity: float,
             open_weights: dict[str, float]) -> SizingDecision: ...


class RiskCheck(Protocol):
    """A single guardrail. Pure function over (portfolio, candidate)."""
    name: str
    def evaluate(self, candidate: TradeCandidate,
                 portfolio_context: "PortfolioContext") -> RiskResult: ...


class RegimeDetector(Protocol):
    def detect(self) -> RegimeReport: ...


class SectorRanker(Protocol):
    def rank(self) -> list[SectorRanking]: ...


class RedFlagRule(Protocol):
    """One forensic/governance check. Composable."""
    code: str
    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]: ...


# ── Portfolio context passed to checks (read-only view) ──────────────────────
@dataclass
class PortfolioContext:
    equity: float
    cash: float
    drawdown_pct: float
    weights_by_symbol: dict[str, float]
    weights_by_sector: dict[str, float]
    weights_by_market: dict[str, float]
    correlation_clusters: list[CorrelationCluster] = field(default_factory=list)
    regime: Optional[RegimeReport] = None
    sector_rankings: dict[str, SectorRanking] = field(default_factory=dict)
