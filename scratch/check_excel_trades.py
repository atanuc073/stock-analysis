import pandas as pd
from pathlib import Path

# Find the latest backtest Excel report
reports_dir = Path("reports/backtest")
excel_files = sorted(reports_dir.glob("backtest_2023-01-01_2025-12-31_*.xlsx"))

if not excel_files:
    print("No excel reports found!")
    exit(1)

latest_file = excel_files[-1]
print(f"Reading latest report: {latest_file.name}")

# Read All_Trades sheet
df = pd.read_excel(latest_file, sheet_name="All_Trades")

print(f"Total trades in sheet: {len(df)}")
print("\nColumns in sheet:")
print(list(df.columns))

# Action stats
print("\nAction breakdown:")
print(df["Action"].value_counts())

# Look at score columns
score_cols = [c for c in df.columns if "score" in c.lower()]
print(f"\nScore columns found: {score_cols}")

# Sell trades score stats
sells = df[df["Action"] == "SELL"]
print(f"\nTotal SELL trades: {len(sells)}")

for c in score_cols:
    print(f"\nStats for {c} in SELL trades:")
    print(sells[c].describe())
    print(f"Nulls in {c}: {sells[c].isna().sum()}")
    print(f"Zeroes in {c}: {(sells[c] == 0).sum()}")
    print("\nSample rows:")
    print(sells[["Symbol", "Action", c, "PnL_Pct", "Reason"]].head(15))
