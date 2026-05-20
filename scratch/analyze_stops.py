import pandas as pd

excel_path = 'reports/backtest/backtest_2023-01-01_2024-12-31_20260518_151703.xlsx'

# Load trades sheet
df = pd.read_excel(excel_path, sheet_name='All_Trades')

# Filter to sells only
sells = df[df['Action'] == 'SELL'].copy()

# Analyze STOP_LOSS exits
stop_losses = sells[sells['Exit_Type'] == 'STOP_LOSS']

print(f"Total Closed Sells: {len(sells)}")
print(f"Total STOP_LOSS trades: {len(stop_losses)}")

# Check if any STOP_LOSS trade had a profit or positive PnL_%
profitable_stops = stop_losses[stop_losses['PnL_%'] > 0]
print(f"STOP_LOSS trades with positive P&L: {len(profitable_stops)}")
if not profitable_stops.empty:
    print(profitable_stops[['Symbol', 'PnL_%', 'Days_Held']])

# Print a breakdown of STOP_LOSS trades by PnL_% buckets to see if they are indeed losses
print("\nP&L distribution of STOP_LOSS trades:")
print(stop_losses['PnL_%'].describe())

# Check how many TIER_1 and TIER_2 exits were registered
print("\nExit Type Counts in All_Trades:")
print(sells['Exit_Type'].value_counts())
