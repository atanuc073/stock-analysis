"""Cross-sectional ranking & sector-relative valuation post-processor.

Operates on a *batch* of `StockReport`s after individual scoring is done.
Adds two universe-aware adjustments that absolute scoring cannot capture:

  1. **Sector-relative valuation** — replaces the unused `valuation` weight.
     Ranks each stock's P/E (and P/B) against its sector peers and turns the
     percentile into a 0..100 score. "Cheap-vs-sector" gets a bonus instead
     of "cheap-vs-everything", which prevents tech stocks from never scoring
     as cheap and utilities from always scoring as cheap.

  2. **Cross-sectional momentum/quality rank** — z-scores each component
     across the full universe and returns a small re-weighting bump for
     stocks in the top quintile. This fixes the cash-drag problem when an
     entire universe is depressed (no stock crosses the absolute 70 floor)
     and the inverse (everyone scores 80 in a euphoric tape).

The original `composite_score` is preserved on `StockReport.composite_score`;
the adjusted score is written to `StockReport.adjusted_score` along with a
detail dict on `StockReport.cross_sectional`.
"""
from __future__ import annotations
import logging
import math
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────
SECTOR_VAL_WEIGHT = 0.04       # weight of sector-relative valuation score
RANK_BONUS_WEIGHT = 0.03       # max bump from cross-sectional rank
TOP_QUINTILE = 0.80            # top 20% gets full bump, linear below


def _percentile_score(value: float | None, peers: list[float],
                      invert: bool = True) -> float | None:
    """Return 0..100 percentile-based score.

    `invert=True` means lower raw values are better (P/E, P/B) — the cheapest
    name in the sector gets 100, the most expensive gets 0.
    """
    if value is None or not np.isfinite(value):
        return None
    cleaned = [p for p in peers if p is not None and np.isfinite(p) and p > 0]
    if len(cleaned) < 3:
        return None
    arr = np.array(cleaned)
    pct = float((arr < value).sum()) / len(arr)   # fraction strictly below
    if invert:
        pct = 1.0 - pct
    return float(np.clip(pct * 100, 0, 100))


def _zscore(value: float, arr: np.ndarray) -> float:
    if arr.size < 5:
        return 0.0
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= 1e-9:
        return 0.0
    return (value - mu) / sd


def apply(reports: Iterable) -> list:
    """Apply cross-sectional adjustments in place. Returns the same list."""
    reps = [r for r in reports if r is not None and getattr(r, "composite_score", 0) > 0]
    if len(reps) < 3:
        # Initialize fields anyway so downstream code is uniform.
        for r in reps:
            r.adjusted_score = r.composite_score
            r.cross_sectional = {}
        return list(reports)

    # ── 1. Sector-relative valuation ──────────────────────────────────
    by_sector_pe: dict[str, list[float]] = {}
    by_sector_pb: dict[str, list[float]] = {}
    for r in reps:
        sec = (r.sector or "Unknown") + "|" + (r.market or "")
        pe = (r.fundamental or {}).get("pe")
        pb = (r.fundamental or {}).get("pb")
        if pe is not None and pe > 0:
            by_sector_pe.setdefault(sec, []).append(pe)
        if pb is not None and pb > 0:
            by_sector_pb.setdefault(sec, []).append(pb)

    # ── 2. Cross-sectional component arrays for z-scores ─────────────
    mom_arr = np.array([(r.momentum or {}).get("score", 50.0) for r in reps])
    qual_arr = np.array([(r.quality or {}).get("score", 50.0) for r in reps if hasattr(r, "quality")])
    if qual_arr.size != len(reps):
        qual_arr = np.full(len(reps), 50.0)

    for i, r in enumerate(reps):
        sec = (r.sector or "Unknown") + "|" + (r.market or "")
        pe = (r.fundamental or {}).get("pe")
        pb = (r.fundamental or {}).get("pb")
        peers_pe = by_sector_pe.get(sec, [])
        peers_pb = by_sector_pb.get(sec, [])

        # Sector-relative P/E (60%) + P/B (40%)
        pe_pct = _percentile_score(pe, peers_pe, invert=True)
        pb_pct = _percentile_score(pb, peers_pb, invert=True)
        components = []
        if pe_pct is not None: components.append((pe_pct, 0.6))
        if pb_pct is not None: components.append((pb_pct, 0.4))
        if components:
            sec_val_score = sum(s * w for s, w in components) / sum(w for _, w in components)
        else:
            sec_val_score = 50.0   # neutral when no peers / no data

        # Cross-sectional rank z-scores
        mom_z = _zscore(mom_arr[i], mom_arr)
        qual_z = _zscore(qual_arr[i], qual_arr)
        # Top-quintile bonus: linear ramp from 0 at z=0 to full at z≥1.0
        ramp = float(np.clip((mom_z + qual_z) / 2.0, 0.0, 1.0))
        rank_bonus = ramp * 100.0   # treat as 0..100 score

        # Adjusted composite: deduct the legacy 'valuation' weight (already
        # 0 in composite.py for live) and add the two new components.
        adj = (
            r.composite_score
            + (sec_val_score - 50.0) * SECTOR_VAL_WEIGHT
            + (rank_bonus - 50.0) * RANK_BONUS_WEIGHT
        )
        # Penalty for outright bottom: bottom quintile of momentum AND quality
        if mom_z < -0.8 and qual_z < -0.8:
            adj -= 2.0

        r.adjusted_score = float(np.clip(adj, 0, 100))
        r.cross_sectional = {
            "sector_val_score": round(sec_val_score, 2),
            "momentum_z": round(mom_z, 2),
            "quality_z": round(qual_z, 2),
            "rank_bonus_score": round(rank_bonus, 2),
            "sector_peers_pe_n": len(peers_pe),
        }
        # Append signals for transparency
        if sec_val_score >= 75:
            r.all_signals.append(f"Cheap vs sector ({sec_val_score:.0f}%ile)")
        elif sec_val_score <= 25:
            r.all_signals.append(f"Expensive vs sector ({sec_val_score:.0f}%ile)")
        if mom_z >= 1.0:
            r.all_signals.append(f"Top momentum (z={mom_z:.1f})")

    return list(reports)
