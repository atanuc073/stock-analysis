import pandas as pd

excel_path = 'reports/backtest/backtest_2023-01-01_2024-12-31_20260518_151703.xlsx'
df = pd.read_excel(excel_path, sheet_name='All_Trades')

print("All_Trades Columns:", df.columns.tolist())

# Find symbols that have more than 1 SELL trade
sells = df[df['Action'] == 'SELL']
sell_counts = sells['Symbol'].value_counts()
multi_sell_symbols = sell_counts[sell_counts > 1].index.tolist()

print(f"Symbols with multiple sell actions (partial scale-outs/trims): {len(multi_sell_symbols)}")

# Let's print the full lifecycle for the first 5 symbols with multiple sells
for sym in multi_sell_symbols[:5]:
    print(f"\n--- Lifecycle for {sym} ---")
    sym_trades = df[df['Symbol'] == sym].sort_values('Date')
    for _, t in sym_trades.iterrows():
        # Handle potential missing columns gracefully
        exit_type = t.get('Exit_Type', '')
        reason = t.get('Reason', '')
        print(f"{t['Action']} | Date: {t['Date'][:10]} | Qty: {t['Qty']:.2f} | Price: {t['Price']:.2f} | Exit_Type: {exit_type} | PnL_%: {t['PnL_%']:.2f}% | Reason: {reason}")
