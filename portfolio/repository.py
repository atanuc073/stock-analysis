"""Persistence layer — depend on protocol, swap implementations freely."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .models import PortfolioState, Position, Trade


class PortfolioRepository(Protocol):
    """Storage abstraction for portfolio state."""
    def load(self) -> PortfolioState: ...
    def save(self, state: PortfolioState) -> None: ...


class JsonPortfolioRepository:
    """File-based JSON persistence — atomic writes."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> PortfolioState:
        if not self.path.exists():
            return PortfolioState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return PortfolioState()
        return PortfolioState(
            positions=[Position.from_dict(p) for p in data.get("positions", [])],
            cash=float(data.get("cash", 0.0)),
            trades=[Trade(**t) for t in data.get("trades", [])],
            peak_equity=float(data.get("peak_equity", 0.0)),
            last_updated=data.get("last_updated", ""),
        )

    def save(self, state: PortfolioState) -> None:
        state.last_updated = datetime.utcnow().isoformat(timespec="seconds")
        payload = {
            "positions": [p.to_dict() for p in state.positions],
            "cash": state.cash,
            "trades": [t.__dict__ for t in state.trades],
            "peak_equity": state.peak_equity,
            "last_updated": state.last_updated,
        }
        # atomic write
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)
