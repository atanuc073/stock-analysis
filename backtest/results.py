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


def compute(result: BacktestResult, risk_free_rate: float = 0.06) -> PerformanceStats:
    if not result.equity_curve:
        raise ValueError("Empty equity curve")

    eq = pd.DataFrame([(p.date, p.total) for p in result.equity_curve],
                      columns=["date", "equity"])
    eq = eq.set_index("date").sort_index()

    initial = result.config.initial_capital
    final = float(eq["equity"].iloc[-1])
    total_return = (final / initial - 1) * 100

    # Time span
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 else -100.0

    # Trading-day returns (DO NOT resample to calendar days — weekend zeros
    # destroy std/mean and make Sharpe meaningless)
    daily = eq["equity"].groupby(eq.index).last().sort_index()
    rets = daily.pct_change().dropna()

    ann_vol = float(rets.std() * np.sqrt(252) * 100)
    rf_daily = risk_free_rate / 252
    excess = rets - rf_daily
    sharpe = float(excess.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    downside = rets[rets < 0]
    sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0.0

    # Drawdown
    cummax = daily.cummax()
    dd = (daily / cummax - 1) * 100
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


def score_calibration(result: BacktestResult,
                      bin_edges: list[float] | None = None) -> pd.DataFrame:
    """Bin closed trades by ``score_at_entry`` and report forward-return stats.

    Tells you whether the scorer is actually predictive — i.e. do trades that
    enter with score 80+ realize better returns than those entering at 70-75?
    A monotonic curve = scorer is signal. A flat curve = scorer is noise above
    the threshold.
    """
    sells = [t for t in result.trades
             if t.action == "SELL" and t.score_at_entry > 0]
    if not sells:
        return pd.DataFrame()

    if bin_edges is None:
        bin_edges = [0, 60, 65, 70, 75, 80, 85, 100]

    df = pd.DataFrame([
        {"score": t.score_at_entry, "pnl_pct": t.pnl_pct,
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
