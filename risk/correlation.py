"""Correlation analyzer — finds 'closet bets' (highly correlated holdings)."""
from __future__ import annotations
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from .interfaces import CorrelationCluster

log = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    lookback_days: int = 90
    cluster_threshold: float = 0.70    # corr above this = same cluster
    period: str = "6mo"


class CorrelationAnalyzer:
    """Builds correlation matrix and detects clusters via greedy union-find."""

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self.cfg = config or CorrelationConfig()

    def matrix(self, symbols: list[str]) -> pd.DataFrame:
        if len(symbols) < 2:
            return pd.DataFrame()
        try:
            df = yf.download(symbols, period=self.cfg.period,
                             auto_adjust=True, progress=False, group_by="ticker")
        except Exception as e:
            log.warning("correlation download failed: %s", e)
            return pd.DataFrame()
        # Build close-price frame
        closes = {}
        for s in symbols:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    closes[s] = df[s]["Close"]
                else:
                    closes[s] = df["Close"]
            except Exception:
                continue
        if not closes:
            return pd.DataFrame()
        prices = pd.DataFrame(closes).dropna(how="all")
        rets = prices.pct_change().dropna()
        if len(rets) < 20:
            return pd.DataFrame()
        return rets.tail(self.cfg.lookback_days).corr()

    def clusters(
        self,
        symbols: list[str],
        weights_by_symbol: dict[str, float],
    ) -> list[CorrelationCluster]:
        corr = self.matrix(symbols)
        if corr.empty:
            return []

        # Union-find clustering on (corr > threshold)
        parent = {s: s for s in corr.columns}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v) and v >= self.cfg.cluster_threshold:
                    union(a, b)

        groups: dict[str, list[str]] = {}
        for s in cols:
            groups.setdefault(find(s), []).append(s)

        clusters: list[CorrelationCluster] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            sub = corr.loc[members, members].values
            tri = sub[np.triu_indices(len(members), k=1)]
            avg = float(np.nanmean(tri)) if tri.size else 1.0
            weight = sum(weights_by_symbol.get(m, 0.0) for m in members) * 100
            clusters.append(CorrelationCluster(
                members=members, avg_correlation=avg, total_weight_pct=weight,
            ))
        clusters.sort(key=lambda c: c.total_weight_pct, reverse=True)
        return clusters
