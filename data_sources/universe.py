"""Universe builders: Nifty 500 and S&P 500 ticker lists for broad screening."""
from __future__ import annotations
import logging
from functools import lru_cache
import pandas as pd

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def sp500_tickers() -> list[str]:
    """Scrape S&P 500 constituents from Wikipedia."""
    try:
        import requests
        from io import StringIO
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysis/1.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        # Yahoo uses '-' instead of '.' (e.g., BRK.B -> BRK-B)
        return [s.replace(".", "-") for s in df["Symbol"].tolist()]
    except Exception as e:
        log.warning("S&P 500 fetch failed: %s — falling back to static list", e)
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM", "V"]


@lru_cache(maxsize=1)
def nifty500_tickers() -> list[str]:
    """Fetch Nifty 500 constituents from NSE archives CSV."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        return [f"{s.strip()}.NS" for s in df["Symbol"].tolist()]
    except Exception as e:
        log.warning("Nifty 500 fetch failed: %s — falling back to static list", e)
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        ]


def broad_universe() -> list[str]:
    return nifty500_tickers() + sp500_tickers()
