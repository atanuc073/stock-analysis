import pandas as pd
try:
    ts = pd.Timestamp("2023-02-31")
    print(f"Success: {ts}")
except Exception as e:
    print(f"Error: {e}")
