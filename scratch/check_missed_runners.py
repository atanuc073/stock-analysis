import sys
import pandas as pd
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from analysis import uptrend
import numpy as np

def analyze_missed():
    cache_dir = Path("cache/backtest")
    
    # We will analyze CVNA and PLTR
    tickers = ["CVNA", "PLTR"]
    
    for ticker in tickers:
        pf = cache_dir / f"{ticker}_2022-01-01_2025-12-31.parquet"
        if not pf.exists():
            print(f"Cache file for {ticker} not found!")
            continue
            
        print("\n" + "="*80)
        print(f" DETAILED SCORE AUDIT FOR {ticker} (2023-2024)")
        print("="*80)
        
        df = pd.read_parquet(pf)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        dates_to_check = []
        if ticker == "CVNA":
            dates_to_check = ["2023-01-15", "2023-03-15", "2023-05-15", "2023-07-15", "2023-09-15", "2024-01-15", "2024-06-15"]
        else:
            dates_to_check = ["2023-01-15", "2023-04-15", "2023-05-15", "2023-06-15", "2023-08-15", "2024-02-15", "2024-11-15"]
            
        for dt_str in dates_to_check:
            dt = pd.Timestamp(dt_str)
            # Find the closest trading date in df index
            valid_idx = df.index[df.index <= dt]
            if len(valid_idx) == 0:
                continue
            closest_dt = valid_idx[-1]
            
            # Slice history up to closest_dt
            hist = df.loc[:closest_dt]
            if len(hist) < 250:
                continue
                
            # Compute uptrend score
            try:
                res = uptrend.compute(hist, regime="Neutral")
                score = res.get("score", 0.0)
                stage2 = res.get("stage2", False)
                signals = res.get("signals", [])
                close_price = hist["Close"].iloc[-1]
                sma200 = hist["Close"].rolling(200).mean().iloc[-1]
                
                print(f"Date: {closest_dt.strftime('%Y-%m-%d')} | Price: ${close_price:.2f} | 200DMA: ${sma200:.2f}")
                print(f"  Stage 2 Uptrend? {'YES' if stage2 else 'NO'} | Uptrend Score: {score:.1f}")
                print(f"  Technical Flags: {signals}")
                print("-" * 50)
            except Exception as e:
                print(f"Error computing score for {ticker} on {dt_str}: {e}")

if __name__ == "__main__":
    analyze_missed()
