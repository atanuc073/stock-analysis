"""Central configuration: watchlists, scoring weights, and runtime settings."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "cache"
REPORTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Runtime ──────────────────────────────────────────────────────────────────
RUN_MODE = os.getenv("RUN_MODE", "watchlist").lower()  # "watchlist" | "broad"
TOP_N = int(os.getenv("TOP_N", "15"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))  # parallel ticker downloads (8 = good speed/rate-limit balance)
FETCH_DELAY_MS = int(os.getenv("FETCH_DELAY_MS", "200"))  # ms stagger per thread to avoid Yahoo 429s

# Forecaster: "linear" (fast, default), "prophet" (medium), "timesfm" (best, needs ~2GB deps)
FORECASTER = os.getenv("FORECASTER", "linear").lower()

# ── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Optional APIs ────────────────────────────────────────────────────────────
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ── Curated starter watchlist ────────────────────────────────────────────────
# Indian tickers use NSE suffix ".NS" for yfinance
WATCHLIST_INDIA = [
    # Index leaders / blue chips
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    # Mid/high-growth picks
    "TATAMOTORS.NS", "ADANIENT.NS", "DMART.NS", "TITAN.NS", "SUNPHARMA.NS",
    "WIPRO.NS", "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
]

WATCHLIST_US = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Broad market leaders
    "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH",
    # Semis / AI plays
    "AMD", "AVGO", "TSM", "ASML", "MU",
    # ETFs for sector signal
    "SPY", "QQQ", "DIA", "IWM",
]

WATCHLIST = WATCHLIST_INDIA + WATCHLIST_US

# ── Composite scoring weights (sum ≈ 1.0) ────────────────────────────────────
SCORE_WEIGHTS = {
    "technical": 0.35,
    "fundamental": 0.25,
    "momentum": 0.15,
    "sentiment": 0.10,
    "forecast": 0.10,
    "options": 0.05,  # US only; auto-redistributed for IN tickers
}

# ── Technical thresholds ─────────────────────────────────────────────────────
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 70
VOLUME_SPIKE_MULT = 1.8  # today's volume vs 20-day avg
