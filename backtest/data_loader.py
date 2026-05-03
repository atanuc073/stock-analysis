"""Bulk historical data loader with disk caching.

Pre-fetches multi-year OHLCV + sector info for the universe, caches to parquet
to avoid hitting Yahoo on every backtest run.
"""
from __future__ import annotations
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import CACHE_DIR

log = logging.getLogger(__name__)
BACKTEST_CACHE = CACHE_DIR / "backtest"
BACKTEST_CACHE.mkdir(exist_ok=True, parents=True)


@dataclass
class HistoricalData:
    """All data needed for backtesting one symbol."""
    symbol: str
    history: pd.DataFrame          # OHLCV with DatetimeIndex
    sector: str = ""
    market: str = "US"
    info_static: dict | None = None  # fundamentals (latest, frozen)

    @property
    def is_indian(self) -> bool:
        return self.symbol.endswith((".NS", ".BO"))


def _cache_path(symbol: str, start: str, end: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return BACKTEST_CACHE / f"{safe}_{start}_{end}.parquet"


def _info_cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return BACKTEST_CACHE / f"{safe}_info.parquet"


def _fetch_one(symbol: str, start: str, end: str) -> HistoricalData | None:
    cache_file = _cache_path(symbol, start, end)
    info_file = _info_cache_path(symbol)

    history: pd.DataFrame
    if cache_file.exists():
        history = pd.read_parquet(cache_file)
    else:
        for attempt in range(3):
            try:
                t = yf.Ticker(symbol)
                history = t.history(start=start, end=end, auto_adjust=True)
                if history.empty:
                    time.sleep(1.5)
                    continue
                history.to_parquet(cache_file)
                break
            except Exception as e:
                msg = str(e).lower()
                if "rate" in msg or "429" in msg:
                    time.sleep(2 ** attempt + 1)
                    continue
                log.debug("fetch %s failed: %s", symbol, e)
                return None
        else:
            log.debug("fetch %s gave up after retries", symbol)
            return None

    if history.empty:
        return None

    # Static info (fundamentals + sector). Frozen — minor lookahead, acceptable.
    info: dict = {}
    sector = ""
    if info_file.exists():
        try:
            info_df = pd.read_parquet(info_file)
            if not info_df.empty:
                info = info_df.iloc[0].to_dict()
                sector = str(info.get("sector") or "")
        except Exception:
            pass
    if not info:
        try:
            info = yf.Ticker(symbol).info or {}
            sector = str(info.get("sector") or "")
            # Persist a small subset
            keep = {k: info.get(k) for k in (
                "sector", "industry", "marketCap", "trailingPE", "priceToBook",
                "returnOnEquity", "debtToEquity", "earningsGrowth",
                "longName", "shortName",
            ) if info.get(k) is not None}
            if keep:
                pd.DataFrame([keep]).to_parquet(info_file)
                info = keep
        except Exception:
            info = {}

    return HistoricalData(
        symbol=symbol,
        history=history,
        sector=sector,
        market="IN" if symbol.endswith((".NS", ".BO")) else "US",
        info_static=info,
    )


def load_universe(symbols: list[str], start: str, end: str,
                  max_workers: int = 4) -> dict[str, HistoricalData]:
    """Load full history for a universe of symbols. Cached on disk."""
    out: dict[str, HistoricalData] = {}
    
    log.info("Loading %d symbols using %d workers (staggered)", len(symbols), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for s in symbols:
            futures[ex.submit(_fetch_one, s, start, end)] = s
            time.sleep(0.15)  # Stagger to avoid Yahoo 429 Rate Limits
            
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading history"):
            sym = futures[fut]
            try:
                hd = fut.result()
                if hd is not None:
                    out[sym] = hd
            except Exception as e:
                log.warning("loader %s: %s", sym, e)
    log.info("Loaded %d/%d symbols", len(out), len(symbols))
    return out


def trading_dates(data: dict[str, HistoricalData], start: str, end: str) -> pd.DatetimeIndex:
    """Union of all symbols' trading dates within the window."""
    all_dates = set()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for hd in data.values():
        idx = hd.history.index
        # Strip timezone for comparison
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        mask = (idx >= start_ts) & (idx <= end_ts)
        all_dates.update(idx[mask])
    return pd.DatetimeIndex(sorted(all_dates))
