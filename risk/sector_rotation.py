"""Sector rotation ranker — relative strength of sector indices vs broad market."""
from __future__ import annotations
from dataclasses import dataclass
import logging

import pandas as pd
import yfinance as yf

from .interfaces import SectorRanking

log = logging.getLogger(__name__)


# Yahoo symbols for sector indices
INDIA_SECTOR_INDICES = {
    "Financial Services": "^NSEBANK",
    "IT": "^CNXIT",
    "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "Metal": "^CNXMETAL",
    "Energy": "^CNXENERGY",
    "Realty": "^CNXREALTY",
    "Media": "^CNXMEDIA",
    "PSU Bank": "^CNXPSUBANK",
}

US_SECTOR_INDICES = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}

INDIA_BENCHMARK = "^NSEI"
US_BENCHMARK = "^GSPC"


@dataclass
class SectorRankerConfig:
    lookback_short_days: int = 63       # 3 months
    lookback_long_days: int = 126       # 6 months
    weight_short: float = 0.6           # 3M weighted higher (more current)
    weight_long: float = 0.4


class IndexBasedSectorRanker:
    """Ranks sectors by relative strength vs a benchmark."""

    def __init__(
        self,
        sector_map: dict[str, str],
        benchmark: str,
        config: SectorRankerConfig | None = None,
    ) -> None:
        self.sector_map = sector_map
        self.benchmark = benchmark
        self.cfg = config or SectorRankerConfig()

    def rank(self) -> list[SectorRanking]:
        bench_df = self._fetch(self.benchmark)
        if bench_df is None or bench_df.empty:
            log.warning("benchmark %s unavailable — sector ranks blank", self.benchmark)
            return []

        rows: list[SectorRanking] = []
        for sector, symbol in self.sector_map.items():
            df = self._fetch(symbol)
            if df is None or df.empty:
                continue
            rs3 = self._relative_strength(df, bench_df, self.cfg.lookback_short_days)
            rs6 = self._relative_strength(df, bench_df, self.cfg.lookback_long_days)
            if rs3 is None or rs6 is None:
                continue
            composite = rs3 * self.cfg.weight_short + rs6 * self.cfg.weight_long
            rows.append(SectorRanking(
                sector=sector, rs_3m=rs3, rs_6m=rs6,
                composite=composite, rank=0, quartile=0,
            ))

        rows.sort(key=lambda r: r.composite, reverse=True)
        n = len(rows)
        for i, r in enumerate(rows):
            r.rank = i + 1
            r.quartile = min(4, (i * 4) // max(n, 1) + 1)
        return rows

    @staticmethod
    def _fetch(symbol: str) -> pd.DataFrame | None:
        try:
            df = yf.Ticker(symbol).history(period="1y", auto_adjust=False)
            return df if not df.empty else None
        except Exception as e:
            log.debug("fetch %s failed: %s", symbol, e)
            return None

    @staticmethod
    def _relative_strength(
        sector: pd.DataFrame, bench: pd.DataFrame, days: int,
    ) -> float | None:
        if len(sector) < days or len(bench) < days:
            return None
        s_ret = sector["Close"].iloc[-1] / sector["Close"].iloc[-days] - 1
        b_ret = bench["Close"].iloc[-1] / bench["Close"].iloc[-days] - 1
        return float(s_ret - b_ret)


class CompositeSectorRanker:
    """Combines India + US into a single ranking namespace."""

    def __init__(self) -> None:
        self._india = IndexBasedSectorRanker(INDIA_SECTOR_INDICES, INDIA_BENCHMARK)
        self._us = IndexBasedSectorRanker(US_SECTOR_INDICES, US_BENCHMARK)

    def rank(self) -> list[SectorRanking]:
        return self._india.rank() + self._us.rank()

    def as_lookup(self) -> dict[str, SectorRanking]:
        return {r.sector: r for r in self.rank()}
