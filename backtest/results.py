"""Performance statistics for backtest results."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult, EquityPoint, BTTrade


@dataclass
class PerformanceStats:
    # Returns
    total_return_pct: float
    cagr_pct: float
    # Risk
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    annual_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # Trade stats
    total_trades: int
    closed_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float
    profit_factor: float
    avg_holding_days: float
    # Exit type distribution
    exits_by_type: dict[str, int]
    # Time
    years: float
    # Final
    initial_capital: float
    final_equity: float
    # SIP / cash-flow aware (0/None when no SIP)
    total_invested: float = 0.0
    n_contributions: int = 0
    xirr_pct: float = 0.0           # money-weighted return (SIP-aware CAGR equivalent)


def compute(result: BacktestResult, risk_free_rate: float = 0.06) -> PerformanceStats:
    if not result.equity_curve:
        raise ValueError("Empty equity curve")

    eq = pd.DataFrame([(p.date, p.total, p.contribution) for p in result.equity_curve],
                      columns=["date", "equity", "contribution"])
    eq = eq.set_index("date").sort_index()

    initial = result.config.initial_capital
    final = float(eq["equity"].iloc[-1])

    # SIP cash-flow accounting
    contribs = result.contributions or []
    total_invested = sum(amt for _, amt in contribs) if contribs else float(initial)
    n_contribs = len(contribs)
    sip_mode = result.config.sip_amount > 0

    # Total return: vs total deployed capital (handles SIP correctly)
    base = total_invested if total_invested > 0 else max(initial, 1.0)
    total_return = (final / base - 1) * 100

    # Time span
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    # Plain CAGR is meaningless for SIP (uses single initial). Compute XIRR
    # for SIP, fall back to CAGR for lumpsum.
    if sip_mode and contribs:
        cash_flows = [(d, -float(amt)) for d, amt in contribs]
        cash_flows.append((eq.index[-1], float(final)))
        xirr = _xirr(cash_flows) * 100
        cagr = xirr   # use XIRR as the headline annualized return for SIP
    else:
        cagr = ((final / max(initial, 1.0)) ** (1 / years) - 1) * 100 if final > 0 else -100.0
        xirr = cagr

    # Trading-day returns (DO NOT resample to calendar days — weekend zeros
    # destroy std/mean and make Sharpe meaningless).
    # Contribution-aware: subtract SIP injection from today's equity before
    # computing pct_change so the cash injection doesn't show as a fake gain.
    daily = eq.copy().sort_index()
    daily["adj_equity"] = daily["equity"] - daily["contribution"]
    prev_eq = daily["equity"].shift(1)
    rets = (daily["adj_equity"] / prev_eq - 1).dropna()

    ann_vol = float(rets.std() * np.sqrt(252) * 100)
    rf_daily = risk_free_rate / 252
    excess = rets - rf_daily
    sharpe = float(excess.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    downside = rets[rets < 0]
    sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0.0

    # Drawdown — use equity series, not the DataFrame
    eq_series = daily["equity"]
    cummax = eq_series.cummax()
    dd = (eq_series / cummax - 1) * 100
    max_dd = float(dd.min())
    in_dd = dd < 0
    if in_dd.any():
        # longest run of in-drawdown days
        groups = (in_dd != in_dd.shift()).cumsum()
        max_dd_dur = int(in_dd.groupby(groups).sum().max())
    else:
        max_dd_dur = 0

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # Trade stats — closed sells only
    sells: list[BTTrade] = [t for t in result.trades if t.action == "SELL"]
    closed = [t for t in sells if t.exit_type and t.exit_type != "END_OF_BACKTEST"]

    wins = [t for t in sells if t.pnl_pct > 0]
    losses = [t for t in sells if t.pnl_pct <= 0]
    win_rate = (len(wins) / len(sells) * 100) if sells else 0.0
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    expectancy = (
        (len(wins) / len(sells)) * avg_win + (len(losses) / len(sells)) * avg_loss
    ) if sells else 0.0
    gross_win = sum(t.pnl_abs for t in wins) or 0.0
    gross_loss = abs(sum(t.pnl_abs for t in losses)) or 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    avg_hold = float(np.mean([t.days_held for t in sells])) if sells else 0.0

    exits_by_type = dict(Counter(t.exit_type or "UNKNOWN" for t in sells))

    return PerformanceStats(
        total_return_pct=total_return,
        cagr_pct=cagr,
        max_drawdown_pct=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        annual_volatility_pct=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=len(result.trades),
        closed_trades=len(sells),
        win_rate_pct=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        expectancy_pct=expectancy,
        profit_factor=profit_factor,
        avg_holding_days=avg_hold,
        exits_by_type=exits_by_type,
        years=years,
        initial_capital=initial,
        final_equity=final,
        total_invested=float(total_invested),
        n_contributions=int(n_contribs),
        xirr_pct=float(xirr),
    )


def yearly_breakdown(result: BacktestResult) -> pd.DataFrame:
    """Calendar year P&L."""
    eq = pd.DataFrame([(p.date, p.total) for p in result.equity_curve],
                      columns=["date", "equity"]).set_index("date").sort_index()
    yearly_end = eq["equity"].resample("YE").last()
    yearly_start = eq["equity"].resample("YE").first()
    rows = []
    for ye_date, end_val in yearly_end.items():
        year = ye_date.year
        start_val = yearly_start.loc[ye_date]
        ret_pct = (end_val / start_val - 1) * 100 if start_val else 0.0
        # trades that year
        year_trades = [t for t in result.trades
                       if t.timestamp.startswith(str(year)) and t.action == "SELL"]
        wins = sum(1 for t in year_trades if t.pnl_pct > 0)
        rows.append({
            "Year": year,
            "Start_Equity": round(start_val, 0),
            "End_Equity": round(end_val, 0),
            "Return_%": round(ret_pct, 2),
            "Trades_Closed": len(year_trades),
            "Wins": wins,
            "Losses": len(year_trades) - wins,
        })
    return pd.DataFrame(rows)


def benchmark_buy_and_hold(prices: pd.Series, capital: float) -> pd.Series:
    """Compute equity curve for naive buy-and-hold of a benchmark series."""
    if prices.empty:
        return pd.Series(dtype=float)
    p0 = prices.iloc[0]
    return capital * (prices / p0)


def _xirr(cash_flows: list[tuple[pd.Timestamp, float]],
          guess: float = 0.1, max_iter: int = 100, tol: float = 1e-7) -> float:
    """Money-weighted (irregular) annualized rate of return.

    Convention (mirrors Excel's XIRR):
      - Outflows (investments) are negative; final value is positive.
      - Returns annualized rate (e.g. 0.18 = 18% / yr).

    Newton-Raphson on:  f(r) = sum( cf_i / (1+r)^((d_i - d_0)/365) ) = 0
    Falls back to bisection if Newton diverges.
    """
    if not cash_flows:
        return 0.0
    cf = sorted(cash_flows, key=lambda x: x[0])
    d0 = cf[0][0]
    years = [(d - d0).days / 365.0 for d, _ in cf]
    amts = [a for _, a in cf]

    def npv(r: float) -> float:
        return sum(a / (1 + r) ** y for a, y in zip(amts, years))

    def dnpv(r: float) -> float:
        return sum(-y * a / (1 + r) ** (y + 1) for a, y in zip(amts, years))

    # Newton-Raphson
    r = guess
    for _ in range(max_iter):
        f = npv(r)
        if abs(f) < tol:
            return r
        df = dnpv(r)
        if df == 0:
            break
        r_new = r - f / df
        if r_new <= -0.999:
            r_new = (r + -0.999) / 2
        if abs(r_new - r) < tol:
            return r_new
        r = r_new

    # Fallback: bisection on a wide range
    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return r if not np.isnan(r) else 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def benchmark_sip(prices: pd.Series,
                  contributions: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    """Equity curve for a SIP into a benchmark, using the SAME cash flows
    as the strategy.

    For each (date, amount) in ``contributions``:
      - Find the next available benchmark price at/after that date
      - Buy fractional units at that price
    Returns daily mark-to-market equity series across the full benchmark range.
    """
    if prices.empty or not contributions:
        return pd.Series(dtype=float)
    if prices.index.tz is not None:
        prices = prices.copy()
        prices.index = prices.index.tz_localize(None)

    units = 0.0
    # Build incremental units series
    units_at = pd.Series(0.0, index=prices.index)
    for d, amt in contributions:
        d = d.tz_localize(None) if getattr(d, "tz", None) is not None else d
        # First trading day at/after d
        idx = prices.index.searchsorted(d, side="left")
        if idx >= len(prices):
            continue
        buy_date = prices.index[idx]
        buy_price = float(prices.iloc[idx])
        if buy_price <= 0:
            continue
        units_at.loc[buy_date] = units_at.loc[buy_date] + (amt / buy_price)

    cum_units = units_at.cumsum()
    return cum_units * prices


def score_calibration(result: BacktestResult,
                      bin_edges: list[float] | None = None) -> pd.DataFrame:
    """Bin closed trades by ``score_at_entry`` and report forward-return stats.

    Tells you whether the scorer is actually predictive — i.e. do trades that
    enter with score 80+ realize better returns than those entering at 70-75?
    A monotonic curve = scorer is signal. A flat curve = scorer is noise above
    the threshold.
    """
    if result.config.uptrend_mode:
        sells = [t for t in result.trades
                 if t.action == "SELL" and t.uptrend_score_at_entry > 0]
    else:
        sells = [t for t in result.trades
                 if t.action == "SELL" and t.score_at_entry > 0]
    if not sells:
        return pd.DataFrame()

    if bin_edges is None:
        bin_edges = [0, 60, 65, 70, 75, 80, 85, 100]

    df = pd.DataFrame([
        {"score": t.uptrend_score_at_entry if result.config.uptrend_mode else t.score_at_entry,
         "pnl_pct": t.pnl_pct,
         "pnl_abs": t.pnl_abs, "days": t.days_held,
         "exit": t.exit_type or ""}
        for t in sells
    ])
    df["bucket"] = pd.cut(df["score"], bins=bin_edges, include_lowest=True,
                          right=False)

    grp = df.groupby("bucket", observed=True).agg(
        Trades=("pnl_pct", "count"),
        AvgPnL_Pct=("pnl_pct", "mean"),
        MedianPnL_Pct=("pnl_pct", "median"),
        WinRate_Pct=("pnl_pct", lambda s: (s > 0).mean() * 100),
        AvgDaysHeld=("days", "mean"),
        TotalPnL=("pnl_abs", "sum"),
    ).round(2).reset_index()
    grp = grp.rename(columns={"bucket": "Score_Bucket"})
    grp["Score_Bucket"] = grp["Score_Bucket"].astype(str)

    # Annualized return per bucket (avg pnl% scaled by (365/avg holding))
    grp["AnnualizedRet_Pct"] = (
        grp["AvgPnL_Pct"] * (365.0 / grp["AvgDaysHeld"].replace(0, 1))
    ).round(2)
    return grp


def _forward_return(history: pd.DataFrame, entry_ts: pd.Timestamp,
                    horizon_days: int) -> Optional[float]:
    """Return %-change from entry close to close ``horizon_days`` ahead.

    Uses next available trading day at/after entry_ts (entry close), and the
    last available trading day at/before entry_ts + horizon_days. Returns None
    if either lookup fails.
    """
    if history is None or history.empty or "Close" not in history.columns:
        return None
    idx = history.index
    if idx.tz is not None:
        entry_ts = entry_ts.tz_localize(None) if entry_ts.tz is not None else entry_ts
        idx = idx.tz_localize(None)
        history = history.copy()
        history.index = idx
    # Entry price: first close on/after entry date
    after = history.loc[history.index >= entry_ts]
    if after.empty:
        return None
    p0 = float(after["Close"].iloc[0])
    target = entry_ts + pd.Timedelta(days=horizon_days)
    upto = history.loc[history.index <= target]
    if upto.empty or len(upto) < 2:
        return None
    p1 = float(upto["Close"].iloc[-1])
    if p0 <= 0:
        return None
    return (p1 / p0 - 1) * 100


def top_chase_diagnostic(result: BacktestResult,
                          data: dict) -> dict:
    """Empirically measure whether the strategy is buying near tops.

    For every BUY trade, joins entry-context (extension above 200DMA, distance
    from 52w high, 6M/1Y returns at entry, RSI, regime label) with the
    realized 30d/90d forward return computed from the same historical data
    the engine ran on.

    Returns a dict with:
      - ``trades``: per-trade DataFrame with entry context + forward returns
      - ``buckets_extension``: aggregate stats binned by pct_above_sma200
      - ``buckets_52w``: aggregate stats binned by pct_from_52w_high
      - ``summary``: high-level top-chase indicators (single row dict)
    """
    buys = [t for t in result.trades if t.action == "BUY"]
    if not buys:
        return {"trades": pd.DataFrame(), "buckets_extension": pd.DataFrame(),
                "buckets_52w": pd.DataFrame(), "summary": {}}

    rows: list[dict] = []
    for t in buys:
        try:
            entry_ts = pd.Timestamp(t.timestamp[:10])
        except Exception:
            continue
        hd = data.get(t.symbol)
        hist = hd.history if hd is not None else None
        fwd_30 = _forward_return(hist, entry_ts, 30) if hist is not None else None
        fwd_90 = _forward_return(hist, entry_ts, 90) if hist is not None else None
        rows.append({
            "Date": entry_ts.date(),
            "Symbol": t.symbol,
            "Sector": t.sector,
            "Score": round(t.uptrend_score_at_entry if result.config.uptrend_mode else t.score_at_entry, 1),
            "Pct_Above_200DMA": round(t.pct_above_sma200_at_entry, 2),
            "Pct_From_52w_High": round(t.pct_from_52w_high_at_entry, 2),
            "RSI": round(t.rsi_at_entry, 1),
            "Ret_3M": round(t.ret_3m_at_entry or 0.0, 1),
            "Ret_6M": round(t.ret_6m_at_entry or 0.0, 1),
            "Ret_1Y": round(t.ret_1y_at_entry or 0.0, 1),
            "Regime": t.regime_label_at_entry,
            "Fwd_30D_%": round(fwd_30, 2) if fwd_30 is not None else None,
            "Fwd_90D_%": round(fwd_90, 2) if fwd_90 is not None else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"trades": df, "buckets_extension": pd.DataFrame(),
                "buckets_52w": pd.DataFrame(), "summary": {}}

    # ── Bucket by extension above 200DMA ────────────────────────────────────
    ext_bins = [-100, 0, 10, 20, 30, 40, 1000]
    ext_labels = ["≤0%", "0-10%", "10-20%", "20-30%", "30-40%", ">40%"]
    df["Ext_Bucket"] = pd.cut(df["Pct_Above_200DMA"], bins=ext_bins,
                               labels=ext_labels, include_lowest=True)
    buckets_ext = df.groupby("Ext_Bucket", observed=True).agg(
        Trades=("Symbol", "count"),
        Avg_Fwd30=("Fwd_30D_%", "mean"),
        Median_Fwd30=("Fwd_30D_%", "median"),
        WinRate30=("Fwd_30D_%", lambda s: (s > 0).mean() * 100),
        Avg_Fwd90=("Fwd_90D_%", "mean"),
        WinRate90=("Fwd_90D_%", lambda s: (s > 0).mean() * 100),
    ).round(2).reset_index()

    # ── Bucket by distance from 52w high ────────────────────────────────────
    high_bins = [-100, -25, -15, -8, -3, 0, 100]
    high_labels = ["<-25% (deep)", "-25 to -15% (pullback)", "-15 to -8% (mid)",
                   "-8 to -3% (near high)", "-3 to 0% (at high)", ">0%"]
    df["High_Bucket"] = pd.cut(df["Pct_From_52w_High"], bins=high_bins,
                                labels=high_labels, include_lowest=True)
    buckets_52w = df.groupby("High_Bucket", observed=True).agg(
        Trades=("Symbol", "count"),
        Avg_Fwd30=("Fwd_30D_%", "mean"),
        Median_Fwd30=("Fwd_30D_%", "median"),
        WinRate30=("Fwd_30D_%", lambda s: (s > 0).mean() * 100),
        Avg_Fwd90=("Fwd_90D_%", "mean"),
        WinRate90=("Fwd_90D_%", lambda s: (s > 0).mean() * 100),
    ).round(2).reset_index()

    # ── High-level top-chase summary ────────────────────────────────────────
    n = len(df)
    near_high = (df["Pct_From_52w_High"] > -5).sum()
    extended = (df["Pct_Above_200DMA"] > 25).sum()
    parabolic = ((df["Ret_6M"] > 50) & (df["Ret_1Y"] > 80)).sum()
    overbought = (df["RSI"] > 65).sum()
    fwd30 = df["Fwd_30D_%"].dropna()
    summary = {
        "total_buys": n,
        "pct_within_5pct_of_52w_high": round(near_high / n * 100, 1),
        "pct_extended_25pct_above_200DMA": round(extended / n * 100, 1),
        "pct_parabolic_6M50_AND_1Y80": round(parabolic / n * 100, 1),
        "pct_overbought_RSI_gt_65": round(overbought / n * 100, 1),
        "avg_fwd_30D_pct": round(fwd30.mean(), 2) if not fwd30.empty else None,
        "median_fwd_30D_pct": round(fwd30.median(), 2) if not fwd30.empty else None,
        "winrate_fwd_30D_pct": round((fwd30 > 0).mean() * 100, 1) if not fwd30.empty else None,
        # top-chase signature: forward return for "near 52w high" entries
        "near_high_avg_fwd30": round(
            df.loc[df["Pct_From_52w_High"] > -5, "Fwd_30D_%"].mean(), 2
        ) if near_high else None,
        "deep_pullback_avg_fwd30": round(
            df.loc[df["Pct_From_52w_High"] < -15, "Fwd_30D_%"].mean(), 2
        ) if (df["Pct_From_52w_High"] < -15).any() else None,
    }

    return {
        "trades": df,
        "buckets_extension": buckets_ext,
        "buckets_52w": buckets_52w,
        "summary": summary,
    }
