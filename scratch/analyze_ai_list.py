import os, sys, io
import logging
# Add project root to path
sys.path.append(os.getcwd())
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_sources.yahoo import fetch_many
from analysis.composite import analyze_batch

logging.basicConfig(level=logging.ERROR)

TICKERS = [
    "CRUS", "GRMN", "TER", "ZM", "LRCX", "APH", "INGM", "FSLR", 
    "AVGO", "WEX", "ANET", "ALAB", "AMAT", "UI", "MPWR"
]

def run():
    print(f"Fetching data for {len(TICKERS)} symbols...")
    data = fetch_many(TICKERS, period="1y")
    
    print("Analyzing...")
    reports = analyze_batch(list(data.values()))
    
    # Sort by score
    reports.sort(key=lambda r: r.composite_score, reverse=True)
    
    print("\n" + "="*80)
    print(f"{'Ticker':<8} | {'Score':<6} | {'Verdict':<12} | {'Reason'}")
    print("-" * 80)
    for r in reports:
        reason = (r.all_signals[0] if r.all_signals else "N/A")[:45]
        print(f"{r.symbol:<8} | {r.composite_score:<6.1f} | {r.verdict:<12} | {reason}")
    print("="*80)

if __name__ == "__main__":
    run()
