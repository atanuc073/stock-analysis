import os, sys, io
import logging
# Add project root to path
sys.path.append(os.getcwd())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_sources.yahoo import fetch_many
from analysis.composite import analyze_batch

logging.basicConfig(level=logging.ERROR)

TICKERS = [
    "NVDA", "GOOGL", "AMD", "KLAC", "ORCL", "CF", "EOG", 
    "HAL", "ANET", "AAPL", "MU", "CRUS", "WDC"
]

def run():
    print(f"Fetching data for {len(TICKERS)} symbols...")
    data = fetch_many(TICKERS, period="1y")
    
    print("Analyzing...")
    reports = analyze_batch(list(data.values()))
    
    # Sort by score
    reports.sort(key=lambda r: r.composite_score, reverse=True)
    
    print("\n" + "="*110)
    print(f"{'Ticker':<8} | {'Score':<6} | {'Verdict':<12} | {'Price':<8} | {'Stop Loss':<10} | {'Stop %':<8} | {'Reason'}")
    print("-" * 110)
    for r in reports:
        atr_val = r.technical.get('atr', 0.0)
        atr_stop = r.price - 3.0 * atr_val if atr_val > 0 else r.price * 0.82
        stop_price = max(atr_stop, r.price * 0.82)
        stop_pct = (stop_price / r.price - 1) * 100 if r.price > 0 else -18.0
        
        reason = (r.all_signals[0] if r.all_signals else "N/A")[:45]
        print(f"{r.symbol:<8} | {r.composite_score:<6.1f} | {r.verdict:<12} | {r.price:<8.2f} | {stop_price:<10.2f} | {stop_pct:<8.1f}% | {reason}")
    print("="*110)

if __name__ == "__main__":
    run()
