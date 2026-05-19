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
def sp400_tickers() -> list[str]:
    """Scrape S&P MidCap 400 constituents from Wikipedia."""
    try:
        import requests
        from io import StringIO
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysis/1.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        # Yahoo uses '-' instead of '.' (e.g., BRK.B -> BRK-B)
        return [s.replace(".", "-") for s in df["Symbol"].tolist()]
    except Exception as e:
        log.warning("S&P 400 fetch failed: %s — falling back to static list", e)
        return ["AA", "AAL", "AAON", "ACI", "ACM", "ADC", "AEIS", "AFG", "AGCO", "AHR"]



@lru_cache(maxsize=1)
def nifty500_tickers() -> list[str]:
    """Fetch Nifty 500 constituents from NSE archives CSV."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        import requests
        from io import StringIO
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysis/1.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        return [f"{s.strip()}.NS" for s in df["Symbol"].tolist()]
    except Exception as e:
        log.warning("!!! Nifty 500 fetch FAILED: %s — using 10-stock fallback list !!!", e)
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        ]


@lru_cache(maxsize=1)
def russell1000_tickers() -> list[str]:
    """Fetch Russell 1000 constituents from the iShares IWB ETF holdings CSV.

    The file has ~9 metadata rows before the actual header. We auto-detect the
    header row by looking for 'Ticker'. Yahoo-style symbols (e.g., BRK-B).
    """
    url = (
        "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
    )
    # iShares strips dots; Yahoo expects '-'. Map known dual-class tickers.
    DUAL_CLASS = {
        "BRKB": "BRK-B", "BFB": "BF-B", "BFA": "BF-A",
        "GOOGL": "GOOGL", "GOOG": "GOOG",  # already correct
        "HEIA": "HEINY",  # Heineken — Yahoo only has the ADR
        "RDSA": "SHEL", "RDSB": "SHEL",
    }
    # BlackRock internal cash/derivative placeholders — not tradeable on Yahoo.
    BLOCKLIST = {
        "-", "USD", "CASH", "MARGIN", "MARGIN_USD",
        "XTSLA",   # Tesla cash placeholder
        "MMF",     # money-market fund
    }
    try:
        import requests
        from io import StringIO
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysis/1.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Ticker,"))
        df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))

        # Some IWB rows include an "Asset Class" column we can use to keep equities only
        if "Asset Class" in df.columns:
            df = df[df["Asset Class"].astype(str).str.contains("Equity", case=False, na=False)]

        out: list[str] = []
        seen: set[str] = set()
        for raw in df["Ticker"].dropna().astype(str):
            t = raw.strip().upper()
            if not t or t in BLOCKLIST:
                continue
            t = DUAL_CLASS.get(t, t)
            # Yahoo uses '-' instead of '.' for share-class tickers
            t = t.replace(".", "-")
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        if not out:
            raise ValueError("no tickers parsed")
        return out
    except Exception as e:
        log.warning("Russell 1000 fetch failed: %s — falling back to S&P 500 + S&P MidCap 400", e)
        return list(dict.fromkeys(sp500_tickers() + sp400_tickers()))



def russell2000_tickers() -> list[str]:
    """Fetch Russell 2000 constituents from raw GitHub CSV.
    """
    url = "https://raw.githubusercontent.com/ikoniaris/Russell2000/master/russell_2000_components.csv"
    # Map known dual-class tickers.
    DUAL_CLASS = {
        "BRKB": "BRK-B", "BFB": "BF-B", "BFA": "BF-A",
        "GOOGL": "GOOGL", "GOOG": "GOOG",
    }
    BLOCKLIST = {
        "-", "USD", "CASH", "MARGIN", "MARGIN_USD",
        "MMF",
    }
    try:
        import requests
        from io import StringIO
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysis/1.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        out: list[str] = []
        seen: set[str] = set()
        for raw in df["Ticker"].dropna().astype(str):
            t = raw.strip().upper()
            if not t or t in BLOCKLIST:
                continue
            t = DUAL_CLASS.get(t, t)
            t = t.replace(".", "-")
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        if not out:
            raise ValueError("no tickers parsed")
        log.info("Loaded %d Russell 2000 tickers from GitHub", len(out))
        return out
    except Exception as e:
        log.warning("Russell 2000 fetch failed: %s — falling back to S&P MidCap 400", e)
        return sp400_tickers()



def broad_universe() -> list[str]:
    return nifty500_tickers() + sp500_tickers()


@lru_cache(maxsize=1)
def nse_all_tickers() -> list[str]:
    """Fetch the full NSE equity universe (~2000 tickers) from EQUITY_L.csv.

    Includes every actively listed equity on the NSE main board (EQ series),
    filtered to series='EQ' to drop ETFs, debt, and special-purpose listings.
    Falls back to nifty500 if the fetch fails.
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        import requests
        from io import StringIO
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysis/1.0",
            "Accept": "text/csv,application/csv;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Normalize header whitespace
        df.columns = [c.strip() for c in df.columns]
        # Series 'EQ' = main-board equities (drops BE/BZ/IL/SM/T0 etc.)
        if "SERIES" in df.columns:
            df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        symbols = [f"{s.strip()}.NS" for s in df["SYMBOL"].dropna().astype(str)]
        # De-dupe while preserving order
        seen, out = set(), []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                out.append(s)
        log.info("NSE all-equity universe: %d tickers", len(out))
        return out
    except Exception as e:
        log.warning("NSE all-equity fetch failed: %s — falling back to Nifty 500", e)
        return nifty500_tickers()
