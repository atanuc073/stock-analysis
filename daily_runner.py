"""Daily orchestrator — Phase A + B integrated.

Pipeline:
  1. Fetch market data (existing screener)
  2. Score all watchlist stocks (existing composite)
  3. Detect macro regime
  4. Rank sectors
  5. Build portfolio context (weights, drawdown, correlation clusters)
  6. Evaluate open positions for exits (incl. red flags + tax advice)
  7. For top-scored non-held candidates: size + run risk gate
  8. Render report + send Telegram
"""
from __future__ import annotations
import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import yfinance as yf

from config import REPORTS_DIR, RUN_MODE, TOP_N, WATCHLIST
from data_sources.universe import broad_universe, russell1000_tickers, sp500_tickers, nifty500_tickers
from data_sources.yahoo import fetch_many, TickerData
from analysis.composite import analyze
from analysis.indicators import atr, annualized_volatility
from analysis import forecast as forecast_dispatcher

from portfolio.models import ExitSignal
from factories import (
    build_portfolio_service, build_sizer, build_regime_detector,
    build_sector_ranker, build_correlation_analyzer,
    build_red_flag_scanner, build_tax_optimizer, build_default_risk_gate,
)
from risk.interfaces import (
    PortfolioContext, TradeCandidate, SectorRanking, CorrelationCluster,
)
from telegram_bot import send_message, send_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("daily")


# ── Helpers ──────────────────────────────────────────────────────────────────
@dataclass
class HorizonForecast:
    horizon_days: int
    forecast_price: float | None
    expected_return_pct: float | None
    lower: float | None = None
    upper: float | None = None
    model: str = ""


@dataclass
class CandidateBundle:
    candidate: TradeCandidate
    sizing_rupees: float
    weight_pct: float
    sizing_reasoning: str
    gate_summary: str
    approved: bool
    forecasts: list[HorizonForecast] = None  # type: ignore[assignment]


def _multi_horizon_forecast(history, horizons: tuple[int, ...] = (30, 60)
                            ) -> list[HorizonForecast]:
    """Run the configured forecaster at multiple horizons.

    Uses the same dispatcher as composite scoring, so respects FORECASTER env var.
    Falls back to linear-trend forecasts if TimesFM/Prophet unavailable.
    """
    out: list[HorizonForecast] = []
    for h in horizons:
        try:
            r = forecast_dispatcher.compute(history, horizon_days=h)
            out.append(HorizonForecast(
                horizon_days=h,
                forecast_price=r.get("forecast_price"),
                expected_return_pct=r.get("expected_return_pct"),
                lower=r.get("lower"),
                upper=r.get("upper"),
                model=r.get("model", ""),
            ))
        except Exception as e:
            log.debug("Forecast %dd failed for series: %s", h, e)
            out.append(HorizonForecast(
                horizon_days=h, forecast_price=None,
                expected_return_pct=None, model="error",
            ))
    return out


def _build_portfolio_context(svc, prices: dict, regime, sector_lookup,
                             correlation_clusters) -> PortfolioContext:
    state = svc.state()
    snap = svc.equity_snapshot(prices)
    equity = snap.total

    weights_sym: dict[str, float] = {}
    weights_sec: dict[str, float] = {}
    weights_mkt: dict[str, float] = {}
    for p in state.open_positions():
        cp = prices.get(p.symbol, p.entry_price)
        mv = cp * p.qty_open
        w = mv / equity if equity > 0 else 0
        weights_sym[p.symbol] = weights_sym.get(p.symbol, 0) + w
        if p.sector:
            weights_sec[p.sector] = weights_sec.get(p.sector, 0) + w
        weights_mkt[p.market] = weights_mkt.get(p.market, 0) + w

    return PortfolioContext(
        equity=equity, cash=state.cash, drawdown_pct=snap.drawdown_pct,
        weights_by_symbol=weights_sym, weights_by_sector=weights_sec,
        weights_by_market=weights_mkt, regime=regime,
        sector_rankings=sector_lookup,
        correlation_clusters=correlation_clusters,
    )


def _candidate_from(report, ticker: TickerData) -> TradeCandidate:
    return TradeCandidate(
        symbol=report.symbol,
        sector=report.sector or "",
        market=report.market,
        score=report.composite_score,
        price=report.price,
        atr=atr(ticker.history),
        annual_volatility=annualized_volatility(ticker.history),
    )


# ── Main runner ──────────────────────────────────────────────────────────────
def run(mode: str = RUN_MODE, top_n: int = TOP_N, send_tg: bool = True) -> None:
    log.info("=== Daily Run @ %s | mode=%s ===", datetime.now().isoformat(timespec="seconds"), mode)

    # 1) Build services
    svc = build_portfolio_service()
    sizer = build_sizer()
    regime_det = build_regime_detector()
    sector_ranker = build_sector_ranker()
    corr_analyzer = build_correlation_analyzer()
    flag_scanner = build_red_flag_scanner()
    tax_opt = build_tax_optimizer()
    gate = build_default_risk_gate()

    # 2) Universe + screening
    if mode == "broad":
        universe = broad_universe()
    elif mode == "russell1000":
        universe = russell1000_tickers()
    elif mode == "sp500":
        universe = sp500_tickers()
    elif mode == "nifty500":
        universe = nifty500_tickers()
    else:
        universe = WATCHLIST
    log.info("Universe: %d", len(universe))
    data = fetch_many(universe, period="1y")
    try:
        from analysis.composite import analyze_batch
        reports = analyze_batch(list(data.values()))
    except Exception:
        reports = []
        for sym, td in tqdm(data.items(), desc="Analyzing"):
            try:
                reports.append(analyze(td))
            except Exception:
                continue
    reports = [r for r in reports if r.composite_score > 0]
    # Sort by adjusted_score (universe-aware) when available, else composite.
    reports.sort(key=lambda r: getattr(r, "adjusted_score", r.composite_score), reverse=True)

    # 3) Regime + sector ranks (parallel-safe but small enough sequential is fine)
    log.info("Detecting regime + sector rotation ...")
    regime = regime_det.detect()
    sector_lookup = sector_ranker.as_lookup()

    # 4) Live prices for open positions
    state = svc.state()
    open_symbols = [p.symbol for p in state.open_positions()]
    held_prices = _live_prices(open_symbols, data)

    # 5) Correlation clusters of holdings
    held_weights: dict[str, float] = {}
    snap = svc.equity_snapshot(held_prices)
    for p in state.open_positions():
        cp = held_prices.get(p.symbol, p.entry_price)
        held_weights[p.symbol] = (cp * p.qty_open) / snap.total if snap.total else 0
    clusters = corr_analyzer.clusters(open_symbols, held_weights) if len(open_symbols) >= 2 else []

    # 6) Red flags on holdings
    red_flag_counts: dict[str, int] = {}
    flag_details: dict[str, list] = {}
    for p in state.open_positions():
        td = data.get(p.symbol)
        info = td.info if td else {}
        hist = td.history if td else None
        flags = flag_scanner.scan(p.symbol, info or {}, hist)
        if flags:
            flag_details[p.symbol] = flags
            red_flag_counts[p.symbol] = flag_scanner.critical_count(flags)

    # 7) Exit evaluation for held positions
    held_scores = {r.symbol: r.composite_score for r in reports if r.symbol in open_symbols}
    exit_signals: list[ExitSignal] = svc.evaluate_all(
        prices=held_prices, scores=held_scores, red_flags=red_flag_counts,
    )

    # 8) Tax advice on hold positions with gains
    tax_advice = []
    for p in state.open_positions():
        cp = held_prices.get(p.symbol, p.entry_price)
        if cp > p.entry_price:
            advice = tax_opt.advise(
                p.symbol, p.entry_date, p.entry_price, cp, p.qty_open,
                is_indian=(p.market == "IN"),
            )
            if advice and advice.recommendation == "DEFER_FOR_LTCG":
                tax_advice.append(advice)

    # 9) Portfolio context for new candidates
    ctx = _build_portfolio_context(svc, held_prices, regime, sector_lookup, clusters)

    # 10) Run gate on new candidates (top N not currently held)
    candidate_bundles: list[CandidateBundle] = []
    held_set = {s.upper() for s in open_symbols}
    new_pool = [r for r in reports if r.symbol.upper() not in held_set and r.composite_score >= 70][:30]

    for r in new_pool:
        td = data.get(r.symbol)
        if td is None or not td.ok:
            continue
        cand = _candidate_from(r, td)
        sizing = sizer.size(cand, ctx.equity if ctx.equity > 0 else 1_000_000,
                            ctx.weights_by_symbol)
        cand.rupees_intended = sizing.rupees_to_invest
        decision = gate.evaluate(cand, ctx)

        # apply gate's size multiplier
        adjusted_rupees = sizing.rupees_to_invest * decision.final_size_multiplier
        candidate_bundles.append(CandidateBundle(
            candidate=cand,
            sizing_rupees=adjusted_rupees,
            weight_pct=(adjusted_rupees / ctx.equity * 100) if ctx.equity > 0 else 0,
            sizing_reasoning=sizing.reasoning,
            gate_summary=decision.summary(),
            approved=decision.approved,
        ))
        if len([b for b in candidate_bundles if b.approved]) >= top_n:
            break

    # 10b) Multi-horizon forecasts for APPROVED candidates only (expensive)
    log.info("Running 30d/60d forecasts on %d approved candidates ...",
             sum(1 for b in candidate_bundles if b.approved))
    for b in candidate_bundles:
        if not b.approved:
            continue
        td = data.get(b.candidate.symbol)
        if td and td.ok:
            b.forecasts = _multi_horizon_forecast(td.history, horizons=(30, 60))

    # 11) Render report
    md = _render_markdown(
        regime, sector_lookup, snap, state, held_prices,
        exit_signals, flag_details, tax_advice, candidate_bundles, clusters,
    )
    today = datetime.now().strftime("%Y-%m-%d")
    md_path = REPORTS_DIR / f"daily_{today}.md"
    md_path.write_text(md, encoding="utf-8")
    log.info("Wrote %s", md_path)

    # 12) Cache picks + exits for paper trader
    _write_paper_cache(candidate_bundles, exit_signals)

    # 13) Telegram
    if send_tg:
        tg = _render_telegram(
            regime, snap, exit_signals, flag_details, tax_advice, candidate_bundles,
        )
        if send_message(tg):
            send_document(str(md_path), caption=f"Daily report — {today}")
            log.info("Telegram OK")

    log.info("=== Done ===")


# ── Paper-trading bridge ─────────────────────────────────────────────────────
PAPER_CACHE = REPORTS_DIR / ".paper_signals.json"


def _write_paper_cache(bundles: list, exit_signals: list) -> None:
    """Persist today's approved picks + exit signals so paper_cli can replay them
    without re-running the full pipeline (which is slow)."""
    try:
        approved = [b for b in bundles if b.approved]
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "picks": [
                {
                    "symbol": b.candidate.symbol,
                    "sector": b.candidate.sector,
                    "market": b.candidate.market,
                    "score": b.candidate.score,
                    "price": b.candidate.price,
                    "atr": b.candidate.atr,
                    "annual_volatility": b.candidate.annual_volatility,
                    "sizing_rupees": b.sizing_rupees,
                }
                for b in approved
            ],
            "exits": [
                {
                    "symbol": s.symbol,
                    "exit_type": s.exit_type.value,
                    "suggested_qty": s.suggested_qty,
                    "current_price": s.current_price,
                    "reason": s.reason,
                    "new_stop_price": s.new_stop_price,
                    "pnl_pct": s.pnl_pct,
                    "pnl_abs": s.pnl_abs,
                }
                for s in exit_signals
            ],
        }
        PAPER_CACHE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Cached %d picks + %d exits to %s",
                 len(payload["picks"]), len(payload["exits"]), PAPER_CACHE)
    except Exception as e:
        log.warning("Failed to write paper cache: %s", e)


def _generate_signals_for_paper_trading(svc=None, broker=None):
    """Loader called by paper_cli — returns (picks, exit_signals).

    Reads the cache written by run(). If stale (>24h) or missing, returns ([], []).
    """
    from risk.interfaces import TradeCandidate
    from portfolio.models import ExitSignal, ExitType

    if not PAPER_CACHE.exists():
        log.warning("No paper-signals cache. Run: python daily_runner.py")
        return [], []

    try:
        payload = json.loads(PAPER_CACHE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Could not read paper cache: %s", e)
        return [], []

    # Stale check (24h)
    try:
        gen = datetime.fromisoformat(payload["generated_at"])
        age_h = (datetime.now() - gen).total_seconds() / 3600
        if age_h > 24:
            log.warning("Paper-signals cache is %.1f h old. Re-run daily_runner.", age_h)
    except Exception:
        pass

    picks = [
        TradeCandidate(
            symbol=p["symbol"], sector=p.get("sector", ""),
            market=p.get("market", "US"), score=float(p["score"]),
            price=float(p["price"]), atr=float(p.get("atr") or 0),
            annual_volatility=float(p.get("annual_volatility") or 0),
            rupees_intended=float(p.get("sizing_rupees") or 0),
        )
        for p in payload.get("picks", [])
    ]

    exit_signals = [
        ExitSignal(
            symbol=e["symbol"],
            exit_type=ExitType(e["exit_type"]),
            suggested_qty=float(e["suggested_qty"]),
            current_price=float(e["current_price"]),
            reason=e.get("reason", ""),
            new_stop_price=e.get("new_stop_price"),
            pnl_pct=float(e.get("pnl_pct") or 0),
            pnl_abs=float(e.get("pnl_abs") or 0),
        )
        for e in payload.get("exits", [])
    ]
    return picks, exit_signals


def _live_prices(symbols: list[str], cached: dict[str, TickerData]) -> dict[str, float]:
    """Reuse fetched data; only call yfinance for symbols not in cache."""
    out: dict[str, float] = {}
    missing = []
    for s in symbols:
        td = cached.get(s)
        if td and td.ok:
            try:
                out[s] = float(td.history["Close"].iloc[-1])
            except Exception:
                missing.append(s)
        else:
            missing.append(s)
    if missing:
        try:
            df = yf.download(missing, period="5d", auto_adjust=True, progress=False, group_by="ticker")
            for s in missing:
                try:
                    if hasattr(df.columns, "levels"):
                        out[s] = float(df[s]["Close"].dropna().iloc[-1])
                    else:
                        out[s] = float(df["Close"].dropna().iloc[-1])
                except Exception:
                    continue
        except Exception:
            pass
    return out


# ── Renderers ────────────────────────────────────────────────────────────────
def _render_markdown(regime, sector_lookup, snap, state, prices,
                     exit_signals, flag_details, tax_advice,
                     candidate_bundles, clusters) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    L: list[str] = []
    L.append(f"# Daily Investment System — {today}\n")

    # Regime
    L.append(f"## 🌐 Market Regime: **{regime.label}** ({regime.score}/10) — alloc ×{regime.allocation_multiplier:.2f}")
    for n in regime.notes:
        L.append(f"- {n}")
    L.append("")

    # Sector leadership
    if sector_lookup:
        ranks = sorted(sector_lookup.values(), key=lambda r: r.composite, reverse=True)
        L.append("## 📊 Sector Leadership")
        L.append("| Rank | Sector | RS 3M | RS 6M | Composite | Quartile |")
        L.append("|---|---|---|---|---|---|")
        for r in ranks[:5]:
            L.append(f"| {r.rank} | {r.sector} | {r.rs_3m*100:+.1f}% | "
                     f"{r.rs_6m*100:+.1f}% | {r.composite*100:+.2f} | Q{r.quartile} 🟢 |")
        L.append("| ... | ... | | | | |")
        for r in ranks[-3:]:
            L.append(f"| {r.rank} | {r.sector} | {r.rs_3m*100:+.1f}% | "
                     f"{r.rs_6m*100:+.1f}% | {r.composite*100:+.2f} | Q{r.quartile} 🔴 |")
        L.append("")

    # Portfolio
    L.append("## 💼 Portfolio Health")
    L.append(f"- Total: {snap.total:,.0f}  |  Cash: {snap.cash:,.0f}  |  MV: {snap.market_value:,.0f}")
    L.append(f"- Unrealized: {snap.unrealized_pnl:+,.0f}  |  Realized: {snap.realized_pnl:+,.0f}")
    L.append(f"- Drawdown from peak: **{snap.drawdown_pct:+.2f}%**")
    if clusters:
        L.append("\n### Correlation clusters")
        for c in clusters:
            L.append(f"- {' / '.join(c.members)} — corr {c.avg_correlation:.2f}, weight {c.total_weight_pct:.1f}%")
    L.append("")

    # Open positions
    open_pos = state.open_positions()
    if open_pos:
        L.append("## Holdings")
        L.append("| Symbol | Mkt | Qty | Entry | Price | P&L% | Stop | Tiers | Sector |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for p in open_pos:
            cp = prices.get(p.symbol, p.entry_price)
            pnl = (cp / p.entry_price - 1) * 100
            tiers = "".join("●" if t.triggered else "○" for t in p.tiers)
            L.append(f"| `{p.symbol}` | {p.market} | {p.qty_open:.2f} | {p.entry_price:.2f} | {cp:.2f} "
                     f"| {pnl:+.1f}% | {p.stop_price:.2f} | {tiers} | {p.sector[:18]} |")
        L.append("")

    # Exit signals
    if exit_signals:
        L.append(f"## 🚨 Exit Signals ({len(exit_signals)})")
        for s in exit_signals:
            stop_str = f" → new stop {s.new_stop_price:.2f}" if s.new_stop_price else ""
            L.append(f"- **{s.exit_type.value}** `{s.symbol}` — sell {s.suggested_qty:.4f} @ "
                     f"{s.current_price:.2f} ({s.pnl_pct:+.1f}%){stop_str}")
            L.append(f"  - {s.reason}")
        L.append("")
    else:
        L.append("## ✓ No exit signals today.\n")

    # Red flags
    if flag_details:
        L.append("## ⚠️ Red Flags on Holdings")
        for sym, flags in flag_details.items():
            L.append(f"- `{sym}`")
            for f in flags:
                L.append(f"  - [{f.severity}] {f.code}: {f.message}")
        L.append("")

    # Tax advice
    if tax_advice:
        L.append("## 📅 Tax-Aware Suggestions")
        for a in tax_advice:
            L.append(f"- `{a.symbol}` — **DEFER**: {a.days_to_ltcg}d to LTCG, "
                     f"saves ₹{a.tax_savings_if_wait:,.0f}")
        L.append("")

    # New candidates
    approved = [b for b in candidate_bundles if b.approved]
    rejected = [b for b in candidate_bundles if not b.approved]
    L.append("## 🎯 New Buy Candidates (gate approved)")
    if not approved:
        L.append("_No candidates passed all risk checks today._\n")
    else:
        L.append("| Symbol | Mkt | Sector | Score | Price | Stop | T1 (+22%) | T2 (+38%) | 30d Fcst | 60d Fcst | Suggested ₹ | Weight |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in approved:
            c = b.candidate
            # Vol-tiered ATR stop: give volatile stocks more breathing room
            vol = getattr(c, "annual_vol", 0.30) or 0.30
            if vol < 0.25:        # large cap / low vol
                atr_mult, hard_floor = 2.5, 0.90
            elif vol < 0.40:      # mid vol
                atr_mult, hard_floor = 3.0, 0.85
            else:                  # small cap / high vol
                atr_mult, hard_floor = 3.5, 0.82
            stop = c.price - atr_mult * c.atr
            stop = max(stop, c.price * hard_floor)
            f30 = _fmt_horizon(b.forecasts, 30)
            f60 = _fmt_horizon(b.forecasts, 60)
            L.append(f"| `{c.symbol}` | {c.market} | {c.sector[:14]} | {c.score:.0f} | {c.price:.2f} "
                     f"| {stop:.2f} | {c.price*1.22:.2f} | {c.price*1.38:.2f} "
                     f"| {f30} | {f60} "
                     f"| {b.sizing_rupees:,.0f} | {b.weight_pct:.1f}% |")

        # Forecast detail (price + range + model)
        any_fcst = any(b.forecasts for b in approved)
        if any_fcst:
            model_used = next((f.model for b in approved if b.forecasts
                               for f in b.forecasts if f.model), "linear")
            L.append(f"\n### 🔮 Forecast Detail (model: **{model_used}**)")
            L.append("| Symbol | Current | 30d Target | 30d Range | 30d Δ | 60d Target | 60d Range | 60d Δ |")
            L.append("|---|---|---|---|---|---|---|---|")
            for b in approved:
                if not b.forecasts:
                    continue
                c = b.candidate
                f30 = _find_horizon(b.forecasts, 30)
                f60 = _find_horizon(b.forecasts, 60)
                L.append(
                    f"| `{c.symbol}` | {c.price:.2f} "
                    f"| {_price(f30)} | {_range(f30)} | {_pct(f30)} "
                    f"| {_price(f60)} | {_range(f60)} | {_pct(f60)} |"
                )

        L.append("\n### Sizing details")
        for b in approved[:10]:
            L.append(f"- `{b.candidate.symbol}`: {b.sizing_reasoning}")
            for line in b.gate_summary.splitlines():
                L.append(f"  {line}")

    if rejected:
        L.append("\n## ❌ Rejected by Risk Gate")
        for b in rejected[:10]:
            L.append(f"- `{b.candidate.symbol}` (score {b.candidate.score:.0f}) — "
                     f"blocked by **{[r for r in b.gate_summary.splitlines() if '✗' in r]}**")

    L.append("\n---\n_⚠️ Automated analysis. Not investment advice._")
    return "\n".join(L)


# ── Forecast formatting helpers ──────────────────────────────────────────────
def _find_horizon(forecasts, days: int) -> HorizonForecast | None:
    if not forecasts:
        return None
    for f in forecasts:
        if f.horizon_days == days:
            return f
    return None


def _fmt_horizon(forecasts, days: int) -> str:
    """Compact cell: '+5.2%' or '—' for the candidate table."""
    f = _find_horizon(forecasts, days)
    if not f or f.expected_return_pct is None:
        return "—"
    return f"{f.expected_return_pct:+.1f}%"


def _price(f: HorizonForecast | None) -> str:
    return f"{f.forecast_price:.2f}" if f and f.forecast_price else "—"


def _range(f: HorizonForecast | None) -> str:
    if not f or f.lower is None or f.upper is None:
        return "—"
    return f"{f.lower:.2f}–{f.upper:.2f}"


def _pct(f: HorizonForecast | None) -> str:
    if not f or f.expected_return_pct is None:
        return "—"
    return f"{f.expected_return_pct:+.1f}%"


def _render_telegram(regime, snap, exit_signals, flag_details, tax_advice, candidate_bundles) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    L = [f"📊 *Daily Update — {today}*", ""]
    L.append(f"🌐 Regime: *{regime.label}* ({regime.score}/10) — alloc ×{regime.allocation_multiplier:.2f}")
    L.append(f"💼 Total: {snap.total:,.0f}  |  P&L: {snap.unrealized_pnl:+,.0f} ({snap.drawdown_pct:+.1f}% DD)")
    L.append("")

    if exit_signals:
        L.append(f"🚨 *EXIT SIGNALS* ({len(exit_signals)})")
        for s in exit_signals[:8]:
            L.append(f"• *{s.exit_type.value}* `{s.symbol}` — qty {s.suggested_qty:.2f} @ {s.current_price:.2f} ({s.pnl_pct:+.1f}%)")
        L.append("")

    if flag_details:
        L.append("⚠️ *RED FLAGS*")
        for sym, flags in list(flag_details.items())[:5]:
            L.append(f"• `{sym}` — {len(flags)} flag(s): {flags[0].code}")
        L.append("")

    if tax_advice:
        L.append("📅 *TAX DEFERRAL OPPORTUNITY*")
        for a in tax_advice[:3]:
            L.append(f"• `{a.symbol}` — wait {a.days_to_ltcg}d, save ₹{a.tax_savings_if_wait:,.0f}")
        L.append("")

    approved = [b for b in candidate_bundles if b.approved][:8]
    if approved:
        L.append("🎯 *NEW BUY CANDIDATES*")
        for b in approved:
            c = b.candidate
            flag = "🇮🇳" if c.market == "IN" else "🇺🇸"
            f30 = _fmt_horizon(b.forecasts, 30)
            f60 = _fmt_horizon(b.forecasts, 60)
            L.append(f"{flag} `{c.symbol}` ({c.score:.0f}) — {b.weight_pct:.1f}% (₹{b.sizing_rupees:,.0f})")
            L.append(f"   T1 {c.price*1.22:.0f}  Stop {max(c.price - 3.0*c.atr, c.price*0.85):.0f}  | 30d {f30}  60d {f60}")
        L.append("")

    L.append("_⚠️ Not investment advice._")
    return "\n".join(L)


# ── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["watchlist", "broad", "russell1000", "sp500", "nifty500"],
                   default=RUN_MODE)
    p.add_argument("--top", type=int, default=TOP_N)
    p.add_argument("--no-telegram", action="store_true")
    args = p.parse_args()
    run(mode=args.mode, top_n=args.top, send_tg=not args.no_telegram)
