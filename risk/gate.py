"""RiskGate — orchestrates an ordered list of RiskChecks (composition over inheritance)."""
from __future__ import annotations
from dataclasses import dataclass

from .interfaces import RiskCheck, RiskResult, TradeCandidate, PortfolioContext


@dataclass
class GateDecision:
    approved: bool
    final_size_multiplier: float
    results: list[tuple[str, RiskResult]]
    blocking_check: str | None = None

    def summary(self) -> str:
        lines = [f"approved={self.approved}, final_mult={self.final_size_multiplier:.2f}"]
        for name, r in self.results:
            tag = "✗" if not r.passed else ("⚠" if r.severity == "WARN" else "✓")
            lines.append(f"  {tag} {name}: {r.message}")
        return "\n".join(lines)


class RiskGate:
    """Aggregates checks. Add/remove checks at construction without modifying logic."""

    def __init__(self, checks: list[RiskCheck]) -> None:
        self._checks = list(checks)

    def with_check(self, check: RiskCheck) -> "RiskGate":
        return RiskGate(self._checks + [check])

    def evaluate(self, candidate: TradeCandidate, ctx: PortfolioContext) -> GateDecision:
        results: list[tuple[str, RiskResult]] = []
        size_mult = 1.0
        approved = True
        blocker: str | None = None

        for check in self._checks:
            r = check.evaluate(candidate, ctx)
            results.append((check.name, r))
            if not r.passed and r.severity == "BLOCK":
                approved = False
                blocker = blocker or check.name
            size_mult *= r.suggested_size_multiplier

        return GateDecision(
            approved=approved,
            final_size_multiplier=max(0.0, min(2.0, size_mult)),
            results=results,
            blocking_check=blocker,
        )
