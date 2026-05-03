"""Yahoo Finance data fetcher with caching and parallelism."""
from __future__ import annotations
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import MAX_WORKERS

log = logging.getLogger(__name__)


@dataclass
class TickerData:
    symbol: str
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    info: dict = field(default_factory=dict)
    news: list = field(default_factory=list)
    options_summary: Optional[dict] = None
    error: Optional[str] = None

    @property
    def is_indian(self) -> bool:
        return self.symbol.endswith((".NS", ".BO"))

    @property
    def ok(self) -> bool:
        return self.error is None and not self.history.empty


def _fetch_one(symbol: str, period: str = "1y", max_retries: int = 4) -> TickerData:
    td = TickerData(symbol=symbol)
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(symbol)
            td.history = t.history(period=period, auto_adjust=True)
            if td.history.empty:
                last_err = "no history"
                # empty history can be transient; retry
                time.sleep(1.0 + random.random())
                continue
            try:
                td.info = t.info or {}
            except Exception:
                td.info = {}
            try:
                td.news = (t.news or [])[:5]
            except Exception:
                td.news = []
            if not td.is_indian:
                try:
                    expiries = t.options[:1]
                    if expiries:
                        chain = t.option_chain(expiries[0])
                        calls_vol = int(chain.calls["volume"].fillna(0).sum())
                        puts_vol = int(chain.puts["volume"].fillna(0).sum())
                        td.options_summary = {
                            "expiry": expiries[0],
                            "call_volume": calls_vol,
                            "put_volume": puts_vol,
                            "pcr": round(puts_vol / calls_vol, 2) if calls_vol else None,
                        }
                except Exception:
                    pass
            td.error = None
            return td
        except Exception as e:
            msg = str(e)
            last_err = msg
            # Rate-limit / transient — exponential backoff
            if "rate" in msg.lower() or "429" in msg or "too many" in msg.lower():
                wait = (2 ** attempt) + random.random() * 2
                time.sleep(wait)
                continue
            time.sleep(0.5 + random.random())
    td.error = last_err or "unknown error"
    return td


def fetch_many(symbols: list[str], period: str = "1y", show_progress: bool = True) -> dict[str, TickerData]:
    results: dict[str, TickerData] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, s, period): s for s in symbols}
        it = as_completed(futures)
        if show_progress:
            it = tqdm(it, total=len(futures), desc="Fetching")
        for fut in it:
            sym = futures[fut]
            try:
                results[sym] = fut.result()
            except Exception as e:
                log.warning("fetch failed %s: %s", sym, e)
                results[sym] = TickerData(symbol=sym, error=str(e))
    return results
