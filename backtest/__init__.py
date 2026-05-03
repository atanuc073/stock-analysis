"""Backtest engine — replay strategy on historical data.

Architecture:
    data_loader.py    Pre-fetch + cache OHLCV history for the universe
    scoring.py        Sliced (no-lookahead) composite scoring
    engine.py         Event-driven simulation loop
    results.py        Stats: Sharpe, DD, win rate, expectancy
    reporter.py       Excel + Markdown + matplotlib equity curve
    cli.py            Entry point

Usage:
    python -m backtest.cli --start 2019-01-01 --end 2024-12-31 --capital 1000000
"""
