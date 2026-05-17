import pandas as pd
import numpy as np

path = r"D:\MY_WORK\stock_analysis\reports\backtest\backtest_2020-01-01_2024-12-31_20260513_223033.xlsx"

print("Loading Equity Curve...")
df = pd.read_excel(path, sheet_name="Equity_Curve")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

# Calculate cash percentage
df['Cash_Pct'] = (df['Cash'] / df['Equity']) * 100
df['Invested_Pct'] = (df['MarketValue'] / df['Equity']) * 100

print("\n=== MONEY UTILIZATION BY YEAR ===")
# Group by year and calculate averages
yearly = df.groupby('Year').agg(
    Avg_Equity=('Equity', 'mean'),
    Avg_Cash=('Cash', 'mean'),
    Avg_Invested=('MarketValue', 'mean'),
    Avg_Cash_Pct=('Cash_Pct', 'mean'),
    Avg_Invested_Pct=('Invested_Pct', 'mean'),
    Max_Drawdown_Pct=('Cash_Pct', 'max'), # Max cash held
    Min_Cash_Pct=('Cash_Pct', 'min'),
    Avg_Open_Positions=('OpenPositions', 'mean'),
    Max_Open_Positions=('OpenPositions', 'max')
).round(2)

print(yearly[['Avg_Equity', 'Avg_Cash_Pct', 'Avg_Invested_Pct', 'Min_Cash_Pct', 'Avg_Open_Positions', 'Max_Open_Positions']])

print("\n=== UTILIZATION BUCKETS (Total Days) ===")
bins = [0, 10, 20, 40, 60, 80, 90, 100]
labels = ['Fully Invested (<10% Cash)', '10-20% Cash', '20-40% Cash', '40-60% Cash', '60-80% Cash', '80-90% Cash', 'Idle (>90% Cash)']
df['Utilization_Bucket'] = pd.cut(df['Cash_Pct'], bins=bins, labels=labels, include_lowest=True)
print(df['Utilization_Bucket'].value_counts().sort_index())
