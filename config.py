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

# Forecaster: "prophet" (default), "linear" (lightweight), "timesfm" (best, needs ~2GB deps)
FORECASTER = os.getenv("FORECASTER", "prophet").lower()

# ── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Optional APIs ────────────────────────────────────────────────────────────
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# ── Curated starter watchlist ────────────────────────────────────────────────
# Indian tickers use NSE suffix ".NS" for yfinance
WATCHLIST_INDIA = [
    # # Index leaders / blue chips
    # "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    # "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    # "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    # # Mid/high-growth picks
    # "TATAMOTORS.NS", "ADANIENT.NS", "DMART.NS", "TITAN.NS", "SUNPHARMA.NS",
    "WIPRO.NS", "HCLTECH.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
    "SHARDACROP.NS","AVANTIFEED.NS","TMPV.NS","KPIL.NS","VTL.NS","COROMANDEL.NS","LTFOODS.NS","TECHNOE.NS","TORENTPOWER.NS","GESHIP.NS","HINDZINC.NS","NATIONALUM.NS","CHENNPETRO.NS","LUPIN.NS","COALINDIA.NS","NATIONALUM.NS"
]

WATCHLIST_US = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # # Broad market leaders
    # "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH",
    # # Semis / AI plays
    "AMD", "AVGO", "TSM", "ASML", "MU","LRCX"
    # ETFs for sector signal
    "SPY", "QQQ", "DIA", "IWM",
    "CF","ANGLOLDSHR",
]

WATCHLIST = WATCHLIST_INDIA + WATCHLIST_US

# ── Composite scoring weights (sum ≈ 1.0) ────────────────────────────────────
# Rebalanced 2026-05: backtest analysis showed momentum (12-1 factor) is the
# Rebalanced 2026-05-21: Minervini-style momentum-first weighting.
# Backtest score calibration was FLAT (Q1≈Q5) because quality/fundamental
# fought momentum. Fix: concentrate weight on the two proven alpha factors
# (RS momentum + earnings drift) and demote mean-reversion signals.
# Deployed optimized weights (Candidate 20)
# SCORE_WEIGHTS = {
#     "momentum":       0.021,  # Optimized
#     "earnings_drift": 0.614,  # Optimized (Core PEAD Alpha Engine)
#     "technical":      0.145,  # Optimized
#     "fundamental":    0.083,  # Optimized
#     "quality":        0.137,  # Optimized (Renormalized to sum exactly to 1.0)
#     "sentiment":      0.00,
#     "options":        0.00,
#     "forecast":       0.00,
#     "valuation":      0.00,
# }

# User-requested weights (adjusted from sum=1.01 to sum=1.00 by setting fundamental=0.34)
# SCORE_WEIGHTS = {
#     "technical":       0.036,
#     "fundamental":     0.035,
#     "momentum":        0.155,
#     "quality":         0.321,
#     "earnings_drift":  0.453,
#     "sentiment":       0.00,
#     "options":         0.00,
#     "forecast":        0.00,
#     "valuation":       0.00,
# }

# Optimized weights from Russell 1000 factor regression (adjusted slightly to sum to exactly 1.0)
# SCORE_WEIGHTS = {
#     "technical":       0.023,
#     "fundamental":     0.030,  # Adjusted from 0.031 to sum exactly to 1.0
#     "momentum":        0.051,
#     "quality":         0.585,
#     "earnings_drift":  0.311,
#     "sentiment":       0.00,
#     "options":         0.00,
#     "forecast":        0.00,
#     "valuation":       0.00,
# }

# Deployed optimized weights from Russell 1000 'softcagr' factor regression (adjusted to sum exactly to 1.0)
SCORE_WEIGHTS = {
    "technical":       0.251,
    "fundamental":     0.028,  # Adjusted from 0.027 to sum exactly to 1.0
    "momentum":        0.247,
    "quality":         0.141,
    "earnings_drift":  0.333,
    "sentiment":       0.00,
    "options":         0.00,
    "forecast":        0.00,
    "valuation":       0.00,
}
# Sanity check at import: weights MUST sum to 1.0 or backtest composites
# come out scaled (e.g. sum=0.85 → top scores ~68 instead of 80, blowing
# past the 70 buy threshold and producing zero trades).
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-6, (
    f"SCORE_WEIGHTS must sum to 1.0, got {sum(SCORE_WEIGHTS.values())}"
)
# ── Technical thresholds ─────────────────────────────────────────────────────
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 70
VOLUME_SPIKE_MULT = 1.8  # today's volume vs 20-day avg

# ── Top Picks sector diversification ─────────────────────────────────────────
# Cap how many picks from a single sector can appear in the Top Picks list.
# Prevents the late-cycle concentration problem where momentum+quality factors
# both reward the same in-favor sector (e.g. Basic Materials + Energy). Set to
# 0 or a value >= TOP_N to disable. Picks are still selected by score within
# each sector — the cap only stops over-representation in the final list.
TOP_PICKS_SECTOR_CAP = int(os.getenv("TOP_PICKS_SECTOR_CAP", "3"))
