"""Macro regime detector — combines breadth, trend, volatility into a 0-10 score."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from .interfaces import RegimeReport

log = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    india_index: str = "^NSEI"
    us_index: str = "^GSPC"
    india_vix: str = "^INDIAVIX"
    us_vix: str = "^VIX"
    sma_short: int = 50
    sma_long: int = 200
    vix_calm_threshold: float = 18.0
    vix_stress_threshold: float = 28.0


class IndexRegimeDetector:
    """Detects regime for one index (India OR US)."""

    LABELS = {
        (8, 10): "BULL",
        (6, 7): "NEUTRAL_BULL",
        (4, 5): "NEUTRAL",
        (2, 3): "CAUTIOUS",
        (0, 1): "BEAR",
    }
    ALLOC = {
        "BULL": 1.0, "NEUTRAL_BULL": 0.85, "NEUTRAL": 0.70,
        "CAUTIOUS": 0.45, "BEAR": 0.15,
    }

    def __init__(self, market: str = "IN", config: RegimeConfig | None = None) -> None:
        self.market = market
        self.cfg = config or RegimeConfig()

    def detect(self) -> RegimeReport:
        cfg = self.cfg
        index_sym = cfg.india_index if self.market == "IN" else cfg.us_index
        vix_sym = cfg.india_vix if self.market == "IN" else cfg.us_vix

        idx = self._safe_history(index_sym, period="1y")
        vix_df = self._safe_history(vix_sym, period="3mo")

        components: dict[str, bool] = {}
        notes: list[str] = []
        score = 0

        if idx is not None and len(idx) >= cfg.sma_long:
            close = idx["Close"]
            sma_s = close.rolling(cfg.sma_short).mean().iloc[-1]
            sma_l = close.rolling(cfg.sma_long).mean().iloc[-1]
            price = close.iloc[-1]
            sma_l_prev = close.rolling(cfg.sma_long).mean().iloc[-21] if len(close) >= cfg.sma_long + 21 else sma_l

            components["price_above_200dma"] = price > sma_l
            components["50dma_above_200dma"] = sma_s > sma_l
            components["200dma_rising"] = sma_l > sma_l_prev
            score += sum([
                2 if components["price_above_200dma"] else 0,
                2 if components["50dma_above_200dma"] else 0,
                2 if components["200dma_rising"] else 0,
            ])

            # 1M and 3M index momentum
            ret_1m = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
            ret_3m = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0
            components["positive_1m_return"] = ret_1m > 0
            components["positive_3m_return"] = ret_3m > 0
            score += (1 if ret_1m > 0 else 0) + (1 if ret_3m > 0 else 0)
            notes.append(f"Index 1M {ret_1m*100:+.1f}%, 3M {ret_3m*100:+.1f}%")
        else:
            notes.append(f"Index data unavailable for {index_sym}")

        if vix_df is not None and not vix_df.empty:
            vix = float(vix_df["Close"].iloc[-1])
            components["vix_calm"] = vix < cfg.vix_calm_threshold
            components["vix_not_stressed"] = vix < cfg.vix_stress_threshold
            score += (1 if components["vix_calm"] else 0) + (1 if components["vix_not_stressed"] else 0)
            notes.append(f"VIX {vix:.1f}")

        score = min(10, score)
        label = self._label_for(score)
        return RegimeReport(
            score=score,
            label=label,
            allocation_multiplier=self.ALLOC[label],
            components=components,
            notes=notes,
        )

    @staticmethod
    def _safe_history(symbol: str, period: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
            return df if not df.empty else None
        except Exception as e:
            log.warning("regime fetch failed %s: %s", symbol, e)
            return None

    @classmethod
    def _label_for(cls, score: int) -> str:
        for (lo, hi), lbl in cls.LABELS.items():
            if lo <= score <= hi:
                return lbl
        return "NEUTRAL"


class MultiMarketRegimeDetector:
    """Aggregates per-market regimes into a global view (worst-case allocation)."""

    def __init__(self, markets: tuple[str, ...] = ("IN", "US")) -> None:
        self._detectors = {m: IndexRegimeDetector(m) for m in markets}

    def detect(self) -> RegimeReport:
        reports = {m: d.detect() for m, d in self._detectors.items()}
        # Aggregate label = worst-case (back-compat for callers that ignore market).
        # Real per-market regimes live in `per_market` and should be preferred.
        worst = min(reports.values(), key=lambda r: r.allocation_multiplier)
        notes = []
        components = {}
        for m, r in reports.items():
            notes.append(f"{m}: {r.label} ({r.score}/10) — {'; '.join(r.notes)}")
            components[f"{m}_label"] = r.label
        return RegimeReport(
            score=worst.score,
            label=worst.label,
            allocation_multiplier=worst.allocation_multiplier,
            components=components,
            notes=notes,
            per_market=reports,
        )
