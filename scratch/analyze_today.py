import pandas as pd
import json

path = r"D:\MY_WORK\stock_analysis\reports\report_2026-05-13.xlsx"

try:
    xl = pd.ExcelFile(path)
    print("Sheets:", xl.sheet_names)
    
    if "Dashboard" in xl.sheet_names:
        df_dash = xl.parse("Dashboard")
        # Try to find regime or just show top picks
        print("\n--- Top 10 Recommended Stocks ---")
        cols = ['Ticker', 'Market', 'Name', 'Score', 'Adj Score', 'Verdict', 'Price', 'Stop (ATR)']
        top_picks = df_dash[df_dash['Score'] >= 55].head(10)
        print(top_picks[cols])
        
except Exception as e:
    print(f"Error: {e}")
