"""Deep analysis of backtest 2021-01-01 to 2026-01-01."""
import pandas as pd

path = r"D:\MY_WORK\stock_analysis\reports\backtest\backtest_2020-01-01_2024-12-31_20260513_223033.xlsx"

# ── 1. Summary sheet ──
try:
    summary = pd.read_excel(path, sheet_name="Summary")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
except Exception as e:
    print(f"No Summary sheet: {e}")

print()

# ── 2. Equity Curve — check for idle cash periods ──
try:
    eq = pd.read_excel(path, sheet_name="Equity_Curve")
    print("=" * 60)
    print("EQUITY CURVE — IDLE CASH ANALYSIS")
    print("=" * 60)
    # Check columns
    print(f"Columns: {eq.columns.tolist()}")
    print(f"Total rows (trading days): {len(eq)}")
    print()
    
    # Find date column
    date_col = [c for c in eq.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_col:
        date_col = date_col[0]
    else:
        date_col = eq.columns[0]  # assume first column is date
    
    # Find cash and equity columns
    cash_col = [c for c in eq.columns if 'cash' in c.lower()]
    equity_col = [c for c in eq.columns if 'equity' in c.lower() or 'total' in c.lower() or 'value' in c.lower()]
    positions_col = [c for c in eq.columns if 'position' in c.lower() or 'invest' in c.lower() or 'market' in c.lower()]
    
    print(f"Date col: {date_col}")
    print(f"Cash cols: {cash_col}")
    print(f"Equity cols: {equity_col}")
    print(f"Position cols: {positions_col}")
    print()
    
    # Show first 20 rows and last 10
    print("FIRST 30 ROWS (start of backtest):")
    print(eq.head(30).to_string(index=False))
    print()
    print("LAST 10 ROWS (end of backtest):")
    print(eq.tail(10).to_string(index=False))
    print()
    
    # Check if cash == equity for extended periods (idle money)
    if cash_col and equity_col:
        cc = cash_col[0]
        ec = equity_col[0]
        eq["cash_pct"] = eq[cc] / eq[ec] * 100
        # Find periods where cash > 90% of equity
        idle = eq[eq["cash_pct"] > 90]
        print(f"Days with >90% cash (idle): {len(idle)} out of {len(eq)} ({len(idle)/len(eq)*100:.1f}%)")
        if not idle.empty:
            print(f"  First idle day: {idle[date_col].iloc[0]}")
            print(f"  Last idle day:  {idle[date_col].iloc[-1]}")
        
        # Find first day where cash drops below 80% (first real investment)
        invested = eq[eq["cash_pct"] < 80]
        if not invested.empty:
            print(f"  First day cash < 80%: {invested[date_col].iloc[0]}")
except Exception as e:
    print(f"Equity curve error: {e}")
    import traceback; traceback.print_exc()

print()

# ── 3. All Trades ──
try:
    trades = pd.read_excel(path, sheet_name="All_Trades")
    print("=" * 60)
    print("TRADES ANALYSIS")
    print("=" * 60)
    print(f"Total trades: {len(trades)}")
    print(f"Columns: {trades.columns.tolist()}")
    print()
    
    buys = trades[trades["Action"] == "BUY"]
    sells = trades[trades["Action"] == "SELL"]
    
    print(f"BUY trades:  {len(buys)}")
    print(f"SELL trades: {len(sells)}")
    print()
    
    # When was the FIRST buy?
    if not buys.empty:
        print(f"First BUY:  {buys['Date'].iloc[0]}")
        print(f"Last BUY:   {buys['Date'].iloc[-1]}")
    if not sells.empty:
        print(f"First SELL: {sells['Date'].iloc[0]}")
        print(f"Last SELL:  {sells['Date'].iloc[-1]}")
    print()
    
    # Exit type distribution
    print("EXIT TYPE DISTRIBUTION:")
    print(sells["Exit_Type"].value_counts())
    print()
    
    # Stop loss analysis
    stop_trades = sells[sells["Exit_Type"].str.contains("STOP", case=False, na=False)]
    print(f"=== STOP LOSS TRADES: {len(stop_trades)} ===")
    if not stop_trades.empty:
        print(f"Avg loss:        {stop_trades['PnL_%'].mean():.2f}%")
        print(f"Median loss:     {stop_trades['PnL_%'].median():.2f}%")
        print(f"Min (worst):     {stop_trades['PnL_%'].min():.2f}%")
        print(f"Max (best):      {stop_trades['PnL_%'].max():.2f}%")
        print(f"Avg days held:   {stop_trades['Days_Held'].mean():.0f}")
        print(f"Median days:     {stop_trades['Days_Held'].median():.0f}")
        
        bins = [-100, -15, -12, -9, -6, -3, 0, 100]
        labels = ["<-15%", "-15 to -12%", "-12 to -9%", "-9 to -6%", "-6 to -3%", "-3 to 0%", ">0%"]
        stop_trades = stop_trades.copy()
        stop_trades["loss_bucket"] = pd.cut(stop_trades["PnL_%"], bins=bins, labels=labels)
        print("\nStop loss severity distribution:")
        print(stop_trades["loss_bucket"].value_counts().sort_index())
    print()
    
    # Tier exits
    tier_trades = sells[sells["Exit_Type"].str.contains("TIER", case=False, na=False)]
    print(f"=== TIER EXITS: {len(tier_trades)} ===")
    if not tier_trades.empty:
        for etype in sorted(tier_trades["Exit_Type"].unique()):
            sub = tier_trades[tier_trades["Exit_Type"] == etype]
            print(f"  {etype}: {len(sub)} trades, avg +{sub['PnL_%'].mean():.1f}%, avg hold {sub['Days_Held'].mean():.0f}d")
    print()
    
    # Overall
    print("=== OVERALL SELL STATS ===")
    winners = sells[sells["PnL_%"] > 0]
    losers = sells[sells["PnL_%"] <= 0]
    print(f"Total sells:  {len(sells)}")
    print(f"Winners:      {len(winners)} ({len(winners)/len(sells)*100:.1f}%)")
    print(f"Losers:       {len(losers)} ({len(losers)/len(sells)*100:.1f}%)")
    if not winners.empty:
        print(f"Avg winner:   +{winners['PnL_%'].mean():.2f}%")
    if not losers.empty:
        print(f"Avg loser:    {losers['PnL_%'].mean():.2f}%")
    
    # Quick stops
    quick_stops = stop_trades[stop_trades["Days_Held"] <= 30]
    print(f"\n=== QUICK STOPS (<=30 days): {len(quick_stops)} ===")
    if not quick_stops.empty:
        print(f"Avg loss: {quick_stops['PnL_%'].mean():.2f}%")
        print(f"These are {len(quick_stops)/len(sells)*100:.1f}% of all sells")

except Exception as e:
    print(f"Trades error: {e}")
    import traceback; traceback.print_exc()

print()

# ── 4. Yearly breakdown ──
try:
    yearly = pd.read_excel(path, sheet_name="Yearly")
    print("=" * 60)
    print("YEARLY BREAKDOWN")
    print("=" * 60)
    print(yearly.to_string(index=False))
except Exception as e:
    print(f"No Yearly sheet: {e}")

print()

# ── 5. By Exit Type ──
try:
    exit_df = pd.read_excel(path, sheet_name="By_Exit_Type")
    print("=" * 60)
    print("BY EXIT TYPE")
    print("=" * 60)
    print(exit_df.to_string(index=False))
except Exception as e:
    print(f"No By_Exit_Type sheet: {e}")
