"""Optional: expose this analyzer as an MCP server so you can ask Copilot Chat
queries like "what's today's top pick from my watchlist?".

Requires: pip install mcp
Run: python mcp_server.py
"""
from __future__ import annotations
import json
from pathlib import Path

from config import REPORTS_DIR, WATCHLIST
from data_sources.yahoo import fetch_many
from analysis.composite import analyze

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise SystemExit("Install mcp first: pip install mcp")

mcp = FastMCP("stock-analysis")


@mcp.tool()
def analyze_ticker(symbol: str) -> dict:
    """Run full analysis on a single ticker. Use NSE suffix for Indian stocks (e.g. RELIANCE.NS)."""
    data = fetch_many([symbol], period="1y", show_progress=False)
    rep = analyze(data[symbol])
    return rep.to_dict()


@mcp.tool()
def todays_top_picks(top_n: int = 10) -> list:
    """Return the top N picks from the most recent daily report."""
    files = sorted(Path(REPORTS_DIR).glob("report_*.json"))
    if not files:
        return [{"error": "no report yet — run main.py first"}]
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    sorted_rs = sorted(data, key=lambda r: r.get("composite_score", 0), reverse=True)
    return sorted_rs[:top_n]


@mcp.tool()
def list_watchlist() -> list[str]:
    """Return the configured watchlist."""
    return WATCHLIST


@mcp.tool()
def quick_score(symbol: str) -> dict:
    """Return composite score + verdict only (faster than full analysis)."""
    data = fetch_many([symbol], period="6mo", show_progress=False)
    rep = analyze(data[symbol])
    return {
        "symbol": rep.symbol,
        "name": rep.name,
        "price": rep.price,
        "composite_score": rep.composite_score,
        "verdict": rep.verdict,
        "top_signals": rep.all_signals[:5],
    }


if __name__ == "__main__":
    mcp.run()
