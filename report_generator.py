"""Generate Markdown + HTML + Excel reports from ranked StockReport list."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import (
    Alignment, Border, Font, PatternFill, Side, numbers,
)
from openpyxl.utils import get_column_letter

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
    lines.append("| # | Ticker | Mkt | Name | Price | Score | Verdict | 1D | 1M | 3M | RSI | P/E | Sector |")
    lines.append("|---|--------|-----|------|-------|-------|---------|------|------|------|-----|-----|--------|")
    for i, r in enumerate(top, 1):
        pe_val = r.fundamental.get('pe')
        pe_str = f"{pe_val:.1f}" if isinstance(pe_val, float) else (pe_val or '—')
        lines.append(
            f"| {i} | **{r.symbol}** | {r.market} | {(r.name or '')[:24]} | "
            f"{r.price:.2f} | **{r.composite_score:.1f}** | {r.verdict} | "
            f"{_fmt_pct(r.momentum.get('ret_1d'))} | "
            f"{_fmt_pct(r.momentum.get('ret_1m'))} | {_fmt_pct(r.momentum.get('ret_3m'))} | "
            f"{r.technical.get('rsi', 0):.0f} | "
            f"{pe_str} | "
            f"{(r.sector or '')[:14]} |"
        )

    lines.append("\n## Detailed Analysis (Top Picks)\n")
    for r in top:
        lines.append(f"### {r.symbol} — {r.name} ({r.market})  →  **{r.verdict}** (score {r.composite_score:.1f})\n")
        lines.append(f"- **Price:** {r.price:.2f}  |  **Sector:** {r.sector or '—'}  |  **Mkt Cap:** {_fmt_money(r.fundamental.get('market_cap'))}")
        lines.append(f"- **Returns:** 1D {_fmt_pct(r.momentum.get('ret_1d'))}, 1W {_fmt_pct(r.momentum.get('ret_1w'))}, 1M {_fmt_pct(r.momentum.get('ret_1m'))}, 3M {_fmt_pct(r.momentum.get('ret_3m'))}, 1Y {_fmt_pct(r.momentum.get('ret_1y'))}")
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


# ── Excel report ─────────────────────────────────────────────────────────

_THIN = Side(style="thin", color="D0D0D0")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

_HEADER_FONT = Font(name="Calibri", bold=True, size=11, color="FFFFFF")
_HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
_HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)

_DATA_FONT = Font(name="Calibri", size=10)
_CENTER = Alignment(horizontal="center", vertical="center")
_LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
_NUM_FMT_PCT = '0.00"%"'
_NUM_FMT_SCORE = '0.0'
_NUM_FMT_PRICE = '#,##0.00'


def _score_fill(score: float) -> PatternFill:
    """Green (high) → Yellow (mid) → Red (low) gradient for 0-100 scores."""
    if score >= 70:
        return PatternFill(start_color="27AE60", end_color="27AE60", fill_type="solid")  # green
    if score >= 62:
        return PatternFill(start_color="2ECC71", end_color="2ECC71", fill_type="solid")  # light green
    if score >= 55:
        return PatternFill(start_color="F1C40F", end_color="F1C40F", fill_type="solid")  # yellow
    if score >= 45:
        return PatternFill(start_color="F39C12", end_color="F39C12", fill_type="solid")  # orange
    if score >= 35:
        return PatternFill(start_color="E74C3C", end_color="E74C3C", fill_type="solid")  # red
    return PatternFill(start_color="C0392B", end_color="C0392B", fill_type="solid")      # dark red


def _verdict_fill(verdict: str) -> PatternFill:
    colors = {
        "STRONG BUY": "1ABC9C",
        "BUY": "2ECC71",
        "HOLD": "F1C40F",
        "REDUCE": "E67E22",
        "AVOID": "E74C3C",
    }
    c = colors.get(verdict, "FFFFFF")
    return PatternFill(start_color=c, end_color=c, fill_type="solid")


def _auto_width(ws, min_width=8, max_width=30):
    """Auto-fit column widths based on content."""
    for col_cells in ws.columns:
        col_letter = get_column_letter(col_cells[0].column)
        lengths = []
        for cell in col_cells:
            val = str(cell.value) if cell.value is not None else ""
            lengths.append(len(val))
        best = min(max(max(lengths, default=min_width) + 2, min_width), max_width)
        ws.column_dimensions[col_letter].width = best


def _style_header_row(ws, row_num=1):
    for cell in ws[row_num]:
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _HEADER_ALIGN
        cell.border = _BORDER


def render_excel(reports: list, top_n: int = 15) -> Workbook:
    """Build a multi-sheet Excel workbook from StockReport list."""
    wb = Workbook()
    sorted_rs = sorted(reports, key=lambda r: r.composite_score, reverse=True)
    top = sorted_rs[:top_n]
    bottom = sorted_rs[-min(5, len(sorted_rs)):]

    # ── Sheet 1: Dashboard ──────────────────────────────────────────────
    ws = wb.active
    ws.title = "Dashboard"
    headers = [
        "#", "Ticker", "Market", "Name", "Sector", "Price",
        "Score", "Verdict", "RSI",
        "1D %", "1W %", "1M %", "3M %",
        "P/E", "P/B", "ROE %", "D/E",
        "EPS Gr %", "Rev Gr %", "Div Yield %",
        "Forecast 21d %", "Signals",
    ]
    ws.append(headers)
    _style_header_row(ws)

    for i, r in enumerate(sorted_rs, 1):
        f = r.fundamental
        m = r.momentum
        fc = r.forecast
        row = [
            i,
            r.symbol,
            r.market,
            (r.name or "")[:28],
            (r.sector or "")[:20],
            round(r.price, 2),
            round(r.composite_score, 1),
            r.verdict,
            round(r.technical.get("rsi", 0), 1),
            round(m.get("ret_1d") or 0, 2),
            round(m.get("ret_1w") or 0, 2),
            round(m.get("ret_1m") or 0, 2),
            round(m.get("ret_3m") or 0, 2),
            round(f.get("pe") or 0, 1) if f.get("pe") else None,
            round(f.get("pb") or 0, 2) if f.get("pb") else None,
            round((f.get("roe") or 0) * 100, 1),
            round(f.get("debt_to_equity") or 0, 1) if f.get("debt_to_equity") else None,
            round((f.get("eps_growth") or 0) * 100, 1),
            round((f.get("revenue_growth") or 0) * 100, 1),
            round(f.get("dividend_yield") or 0, 2),
            round(fc.get("expected_return_pct") or 0, 2),
            "; ".join(r.all_signals[:5]),
        ]
        ws.append(row)
        row_num = i + 1  # header is row 1
        # Color-code score cell
        score_cell = ws.cell(row=row_num, column=7)
        score_cell.fill = _score_fill(r.composite_score)
        score_cell.font = Font(name="Calibri", bold=True, size=10,
                               color="FFFFFF" if r.composite_score >= 55 or r.composite_score < 35 else "000000")
        # Color-code verdict cell
        verdict_cell = ws.cell(row=row_num, column=8)
        verdict_cell.fill = _verdict_fill(r.verdict)
        verdict_cell.font = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
        # Style all cells in this row
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=row_num, column=col_idx)
            cell.border = _BORDER
            if cell.font == Font():  # only set if not already styled
                cell.font = _DATA_FONT
            cell.alignment = _CENTER if col_idx <= 9 else _LEFT if col_idx == len(headers) else _CENTER

    # Freeze header + auto-filter
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
    _auto_width(ws)

    # ── Sheet 2: Detailed Analysis (Top Picks) ──────────────────────────
    ws2 = wb.create_sheet(title="Top Picks Detail")
    detail_headers = [
        "Ticker", "Name", "Market", "Verdict", "Score",
        "Price", "Sector", "Mkt Cap",
        "RSI", "Above 50DMA", "Above 200DMA", "% from 52W High",
        "P/E", "P/B", "ROE %", "D/E", "EPS Gr %", "Rev Gr %", "Profit Margin %",
        "1D %", "1W %", "1M %", "3M %",
        "Forecast 21d %", "Forecast Price",
        "Sentiment Avg", "Top Headline",
        "Signals",
    ]
    ws2.append(detail_headers)
    _style_header_row(ws2)

    for r in top:
        f = r.fundamental
        m = r.momentum
        fc = r.forecast
        t = r.technical
        s = r.sentiment
        headline = ""
        if s.get("headlines"):
            headline = s["headlines"][0].get("title", "")[:60]
        row = [
            r.symbol,
            (r.name or "")[:28],
            r.market,
            r.verdict,
            round(r.composite_score, 1),
            round(r.price, 2),
            r.sector or "",
            _fmt_money(f.get("market_cap")),
            round(t.get("rsi", 0), 1),
            "Yes" if t.get("above_sma50") else "No",
            "Yes" if t.get("above_sma200") else "No",
            round(t.get("pct_from_52w_high", 0), 2),
            round(f.get("pe") or 0, 1) if f.get("pe") else None,
            round(f.get("pb") or 0, 2) if f.get("pb") else None,
            round((f.get("roe") or 0) * 100, 1),
            round(f.get("debt_to_equity") or 0, 1) if f.get("debt_to_equity") else None,
            round((f.get("eps_growth") or 0) * 100, 1),
            round((f.get("revenue_growth") or 0) * 100, 1),
            round((f.get("profit_margin") or 0) * 100, 1),
            round(m.get("ret_1d") or 0, 2),
            round(m.get("ret_1w") or 0, 2),
            round(m.get("ret_1m") or 0, 2),
            round(m.get("ret_3m") or 0, 2),
            round(fc.get("expected_return_pct") or 0, 2),
            round(fc.get("forecast_price") or 0, 2),
            round(s.get("avg_sentiment", 0), 3),
            headline,
            "; ".join(r.all_signals[:6]),
        ]
        ws2.append(row)

    # Style detail sheet
    for row_cells in ws2.iter_rows(min_row=2, max_row=ws2.max_row, max_col=len(detail_headers)):
        for cell in row_cells:
            cell.font = _DATA_FONT
            cell.border = _BORDER
            cell.alignment = _CENTER
        # Color score
        score_cell = row_cells[4]  # column E = Score
        if score_cell.value is not None:
            score_cell.fill = _score_fill(score_cell.value)
            score_cell.font = Font(name="Calibri", bold=True, size=10,
                                   color="FFFFFF" if score_cell.value >= 55 or score_cell.value < 35 else "000000")
        verdict_cell = row_cells[3]  # column D = Verdict
        if verdict_cell.value:
            verdict_cell.fill = _verdict_fill(verdict_cell.value)
            verdict_cell.font = Font(name="Calibri", bold=True, size=10, color="FFFFFF")

    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = f"A1:{get_column_letter(len(detail_headers))}1"
    _auto_width(ws2)

    # ── Sheet 3: Avoid / Weak ───────────────────────────────────────────
    ws3 = wb.create_sheet(title="Avoid - Weak")
    avoid_headers = ["Ticker", "Name", "Market", "Score", "Verdict", "Price", "RSI", "1M %", "3M %", "Key Reason"]
    ws3.append(avoid_headers)
    _style_header_row(ws3)

    for r in bottom:
        reason = (r.all_signals[0] if r.all_signals else r.error or "weak signals")
        ws3.append([
            r.symbol,
            (r.name or "")[:28],
            r.market,
            round(r.composite_score, 1),
            r.verdict,
            round(r.price, 2),
            round(r.technical.get("rsi", 0), 1),
            round(r.momentum.get("ret_1m") or 0, 2),
            round(r.momentum.get("ret_3m") or 0, 2),
            reason,
        ])

    for row_cells in ws3.iter_rows(min_row=2, max_row=ws3.max_row, max_col=len(avoid_headers)):
        for cell in row_cells:
            cell.font = _DATA_FONT
            cell.border = _BORDER
            cell.alignment = _CENTER

    ws3.freeze_panes = "A2"
    ws3.auto_filter.ref = f"A1:{get_column_letter(len(avoid_headers))}1"
    _auto_width(ws3)

    return wb


def write_reports(reports: list, top_n: int = 15) -> tuple[Path, Path, Path]:
    today = datetime.now().strftime("%Y-%m-%d")
    md = render_markdown(reports, top_n)
    md_path = REPORTS_DIR / f"report_{today}.md"
    md_path.write_text(md, encoding="utf-8")

    json_path = REPORTS_DIR / f"report_{today}.json"
    json_path.write_text(
        json.dumps([r.to_dict() for r in reports], default=str, indent=2),
        encoding="utf-8",
    )

    xlsx_path = REPORTS_DIR / f"report_{today}.xlsx"
    wb = render_excel(reports, top_n=top_n)
    wb.save(str(xlsx_path))

    return md_path, json_path, xlsx_path


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
