
from data_sources.universe import russell1000_tickers
import logging

logging.basicConfig(level=logging.INFO)
tickers = russell1000_tickers()
print(f"Tickers found: {len(tickers)}")
