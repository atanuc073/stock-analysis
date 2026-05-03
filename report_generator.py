"""Generate Markdown + HTML reports from ranked StockReport list."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from config import REPORTS_DIR


def _fmt_money(v):
    if v is None or v == 0:
        return "—"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def _fmt_pct(v, signed=True):
    if v is None:
        return "—"
    return f"{'+' if signed and v >= 0 else ''}{v:.2f}%"


def render_markdown(reports: list, top_n: int = 15) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    sorted_rs = sorted(reports, key=lambda r: r.composite_score, reverse=True)
    top = sorted_rs[:top_n]
    bottom = sorted_rs[-min(5, len(sorted_rs)):]

    lines = []
    lines.append(f"# Daily Stock Analysis — {today}\n")
    lines.append(f"_Universe: {len(reports)} tickers analyzed_\n")

    # Summary table
    lines.append("## Top Picks\n")
    lines.append("| # | Ticker | Mkt | Name | Price | Score | Verdict | 1M | 3M | RSI | P/E | Sector |")
    lines.append("|---|--------|-----|------|-------|-------|---------|------|------|-----|-----|--------|")
    for i, r in enumerate(top, 1):
        pe_val = r.fundamental.get('pe')
        pe_str = f"{pe_val:.1f}" if isinstance(pe_val, float) else (pe_val or '—')
        lines.append(
            f"| {i} | **{r.symbol}** | {r.market} | {(r.name or '')[:24]} | "
            f"{r.price:.2f} | **{r.composite_score:.1f}** | {r.verdict} | "
            f"{_fmt_pct(r.momentum.get('ret_1m'))} | {_fmt_pct(r.momentum.get('ret_3m'))} | "
            f"{r.technical.get('rsi', 0):.0f} | "
            f"{pe_str} | "
            f"{(r.sector or '')[:14]} |"
        )

    lines.append("\n## Detailed Analysis (Top Picks)\n")
    for r in top:
        lines.append(f"### {r.symbol} — {r.name} ({r.market})  →  **{r.verdict}** (score {r.composite_score:.1f})\n")
        lines.append(f"- **Price:** {r.price:.2f}  |  **Sector:** {r.sector or '—'}  |  **Mkt Cap:** {_fmt_money(r.fundamental.get('market_cap'))}")
        lines.append(f"- **Returns:** 1W {_fmt_pct(r.momentum.get('ret_1w'))}, 1M {_fmt_pct(r.momentum.get('ret_1m'))}, 3M {_fmt_pct(r.momentum.get('ret_3m'))}, 1Y {_fmt_pct(r.momentum.get('ret_1y'))}")
        lines.append(f"- **Technical:** RSI {r.technical.get('rsi', 0):.1f} | Above 50DMA: {r.technical.get('above_sma50')} | Above 200DMA: {r.technical.get('above_sma200')} | {_fmt_pct(r.technical.get('pct_from_52w_high'), False)} from 52W high")
        f = r.fundamental
        lines.append(
            f"- **Fundamentals:** P/E {f.get('pe') or '—'} | P/B {f.get('pb') or '—'} | "
            f"ROE {(f.get('roe') or 0)*100:.1f}% | D/E {f.get('debt_to_equity') or '—'} | "
            f"EPS growth {(f.get('eps_growth') or 0)*100:.1f}%"
        )
        fc = r.forecast
        if fc.get("expected_return_pct") is not None:
            lines.append(f"- **Forecast (21d):** {_fmt_pct(fc.get('expected_return_pct'))} (target ~{fc.get('forecast_price'):.2f})")
        if r.options.get("available"):
            lines.append(f"- **Options (US):** PCR {r.options.get('put_call_ratio')} | calls {r.options.get('call_volume'):,} | puts {r.options.get('put_volume'):,} | exp {r.options.get('expiry')}")
        if r.sentiment.get("headlines"):
            lines.append(f"- **Sentiment:** avg {r.sentiment.get('avg_sentiment', 0):+.2f}")
            for h in r.sentiment["headlines"][:3]:
                lines.append(f"  - _{h['title']}_  ({h['sentiment']:+.2f})")
        if r.all_signals:
            lines.append(f"- **Signals:** {' • '.join(r.all_signals[:8])}")
        lines.append("")

    lines.append("## Avoid / Weakest\n")
    lines.append("| Ticker | Score | Verdict | Reason |")
    lines.append("|--------|-------|---------|--------|")
    for r in bottom:
        reason = (r.all_signals[0] if r.all_signals else r.error or "weak signals")
        lines.append(f"| {r.symbol} | {r.composite_score:.1f} | {r.verdict} | {reason} |")

    lines.append("\n---\n_⚠️ Disclaimer: This is automated technical/quantitative analysis, not investment advice. Always do your own due diligence and consider consulting a SEBI-registered advisor (India) or licensed financial advisor (US)._\n")
    return "\n".join(lines)


def write_reports(reports: list, top_n: int = 15) -> tuple[Path, Path]:
    today = datetime.now().strftime("%Y-%m-%d")
    md = render_markdown(reports, top_n)
    md_path = REPORTS_DIR / f"report_{today}.md"
    md_path.write_text(md, encoding="utf-8")

    json_path = REPORTS_DIR / f"report_{today}.json"
    json_path.write_text(
        json.dumps([r.to_dict() for r in reports], default=str, indent=2),
        encoding="utf-8",
    )
    return md_path, json_path


def telegram_summary(reports: list, top_n: int = 10) -> str:
    """Compact message for Telegram (4096 char limit)."""
    today = datetime.now().strftime("%Y-%m-%d")
    sorted_rs = sorted(reports, key=lambda r: r.composite_score, reverse=True)
    lines = [f"📊 *Daily Stock Picks — {today}*", f"_{len(reports)} tickers analyzed_\n"]
    lines.append("*🟢 TOP PICKS*")
    for i, r in enumerate(sorted_rs[:top_n], 1):
        flag = "🇮🇳" if r.market == "IN" else "🇺🇸"
        m1 = r.momentum.get("ret_1m")
        m1s = f"{m1:+.1f}%" if m1 is not None else "—"
        sig = (r.all_signals[0] if r.all_signals else "")[:40]
        lines.append(f"{i}. {flag} `{r.symbol}` — *{r.verdict}* ({r.composite_score:.0f}) | 1M {m1s}")
        if sig:
            lines.append(f"   _{sig}_")
    lines.append("\n*🔴 AVOID*")
    for r in sorted_rs[-3:]:
        flag = "🇮🇳" if r.market == "IN" else "🇺🇸"
        lines.append(f"• {flag} `{r.symbol}` — {r.verdict} ({r.composite_score:.0f})")
    lines.append("\n_⚠️ Not investment advice. Full report saved to `reports/`._")
    return "\n".join(lines)
