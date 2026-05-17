import pandas as pd
from pathlib import Path
import numpy as np
import re

def run_audit():
    cache_dir = Path("cache/backtest")
    # Grab all backtest Excel files, filtering out temporary lock files starting with ~$
    excel_files = [f for f in Path("reports/backtest").glob("backtest_*.xlsx") if not f.name.startswith("~$")]
    excel_files = sorted(excel_files)
    
    if not excel_files:
        print("No backtest excel reports found!")
        return
        
    latest_excel = excel_files[-1]
    print(f"Loading latest trade log from: {latest_excel.name}")
    
    # Parse dates from filename (format: backtest_YYYY-MM-DD_YYYY-MM-DD_TIMESTAMP.xlsx)
    match = re.search(r"backtest_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_", latest_excel.name)
    if not match:
        print(f"Could not parse start/end dates from filename: {latest_excel.name}")
        return
        
    start_date, end_date = match.group(1), match.group(2)
    print(f"Parsed backtest window: {start_date} to {end_date}")
    
    # Load all trades from Excel
    trades_df = pd.read_excel(latest_excel, sheet_name="All_Trades")
    
    # Filter for BUY trades to see what was bought
    buys = trades_df[trades_df["Action"] == "BUY"]
    sells = trades_df[trades_df["Action"] == "SELL"]
    
    # Map each symbol to its backtest trading details
    bought_symbols = {}
    for idx, row in buys.iterrows():
        sym = row["Symbol"]
        date = row["Date"]
        price = row["Price"]
        score = row["Score_Entry"]
        
        # Find matching sell if any
        matching_sells = sells[(sells["Symbol"] == sym) & (sells["Date"] > date)]
        pnl = ""
        days = ""
        reason = ""
        if not matching_sells.empty:
            sell_row = matching_sells.iloc[0]
            pnl_val = sell_row["PnL_%"]
            pnl = f"{pnl_val:+.1f}%"
            days = str(sell_row["Days_Held"])
            reason = str(sell_row["Reason"])
        else:
            pnl = "OPEN"
            days = "-"
            reason = "-"
            
        if sym not in bought_symbols:
            bought_symbols[sym] = []
        bought_symbols[sym].append({
            "entry_date": date[:10] if isinstance(date, str) else date.strftime("%Y-%m-%d"),
            "entry_price": price,
            "score": score,
            "pnl": pnl,
            "days": days,
            "reason": reason
        })

    # Now scan cached price files to find top explosive stocks
    parquet_files = list(cache_dir.glob("*.parquet"))
    print(f"\nScanning {len(parquet_files)} cached ticker files to identify high-growth leaders between {start_date} and {end_date}...")
    
    ticker_stats = []
    
    for pf in parquet_files:
        # Match symbol before the two date suffixes (e.g. LITE_2024-01-02_2026-02-01.parquet)
        m = re.match(r"^([A-Za-z0-9_.&/\-]+)_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.parquet$", pf.name)
        if m:
            symbol = m.group(1)
        else:
            continue
            
        # Restore slashes if they were escaped
        symbol = symbol.replace("_", "/")
        
        try:
            df = pd.read_parquet(pf)
            if df.empty:
                continue
                
            # Strip timezone
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            # Slice to parsed backtest window
            df_bt = df.loc[start_date:end_date]
            if len(df_bt) < 40: # Needs a minimal history in the period
                continue
                
            # Find the lowest close price and the subsequent highest close price
            close = df_bt["Close"]
            low_idx = close.idxmin()
            low_price = close.loc[low_idx]
            
            # Slice from low_idx onwards to find peak
            subsequent_close = close.loc[low_idx:]
            if len(subsequent_close) == 0:
                continue
                
            high_idx = subsequent_close.idxmax()
            high_price = subsequent_close.loc[high_idx]
            
            if low_price <= 0:
                continue
                
            max_pnl_pct = (high_price / low_price - 1) * 100.0
            
            # Absolute period return (start to end)
            start_price = close.iloc[0]
            end_price = close.iloc[-1]
            period_ret = (end_price / start_price - 1) * 100.0
            
            ticker_stats.append({
                "Symbol": symbol,
                "Low_Date": low_idx.strftime("%Y-%m-%d"),
                "Low_Price": round(low_price, 2),
                "Peak_Date": high_idx.strftime("%Y-%m-%d"),
                "Peak_Price": round(high_price, 2),
                "Max_Run_%": round(max_pnl_pct, 1),
                "Period_Return_%": round(period_ret, 1)
            })
        except Exception as e:
            pass
            
    if not ticker_stats:
        print("No valid ticker statistics computed.")
        return
        
    # Sort by Max_Run_% descending to find the absolute biggest winners
    ticker_stats_df = pd.DataFrame(ticker_stats)
    ticker_stats_df = ticker_stats_df.drop_duplicates(subset=["Symbol"])
    top_explosive = ticker_stats_df.sort_values(by="Max_Run_%", ascending=False).head(25)
    
    print("\n" + "="*90)
    print(f" TOP 20 MOST EXPLOSIVE STOCKS ({start_date} -> {end_date}) vs BACKTEST AUDIT")
    print("="*90)
    
    cols = ["Symbol", "Max_Run_%", "Period_Return_%", "Low_Date -> Peak_Date", "Did Buy?", "Entry Date(s)", "Return Captured"]
    rows = []
    
    captured_count = 0
    for idx, row in top_explosive.iterrows():
        sym = row["Symbol"]
        max_run = f"{row['Max_Run_%']:.1f}%"
        period_ret = f"{row['Period_Return_%']:+.1f}%"
        run_period = f"{row['Low_Date']} -> {row['Peak_Date']}"
        
        # Check if bought
        bt_data = bought_symbols.get(sym)
        did_buy = "NO"
        entry_dates = "-"
        return_captured = "-"
        
        if bt_data:
            did_buy = "YES"
            captured_count += 1
            entries = []
            returns = []
            for entry in bt_data:
                entries.append(entry["entry_date"])
                returns.append(entry["pnl"])
            entry_dates = ", ".join(entries)
            return_captured = ", ".join(returns)
            
        rows.append([sym, max_run, period_ret, run_period, did_buy, entry_dates, return_captured])
        
    audit_df = pd.DataFrame(rows, columns=cols)
    print(audit_df.to_string(index=False))
    print("\n" + "="*90)
    print(f"AUDIT SUMMARY: Captured {captured_count} out of the top {len(top_explosive)} most explosive market runners ({captured_count/len(top_explosive)*100:.1f}%)")
    print("="*90)

if __name__ == "__main__":
    run_audit()
