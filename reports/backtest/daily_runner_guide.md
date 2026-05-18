# Operational Run Guide: Systematic Daily Stock Scanners

This guide documents how to execute the systematic daily scanners within your stock analysis engine. You have two operational pipelines:
1. **Core Daily Orchestrator (`main.py`):** A fast, high-level composite scanner.
2. **Integrated Risk-Gated Runner (`daily_runner.py`):** An advanced end-to-end pipeline that scans, scores, computes sector rotations, evaluates open positions for exits, scans for red flags, optimizes tax outcomes, sizing trades, and builds paper-trading signals.

---

## 🚀 1. The Advanced Daily Runner (`daily_runner.py`)

This is your primary end-to-end execution script. It runs your full multi-market momentum pipeline with advanced capital gates.

### Standard Execution Examples

* **Broad Market Scan (900+ US/IN Large & Midcaps) - Skip Telegram:**
  ```powershell
  .\.venv\Scripts\python.exe daily_runner.py --mode broad --no-telegram
  ```

* **Indian Market Scan (Nifty 500 Tickers) - Send Telegram:**
  ```powershell
  .\.venv\Scripts\python.exe daily_runner.py --mode nifty500
  ```

* **US Market Scan (S&P 500 Tickers) - Custom Threshold:**
  ```powershell
  .\.venv\Scripts\python.exe daily_runner.py --mode sp500 --threshold 65.0
  ```

* **Watchlist Scan (Configured watchlist only) - Top 5 Picks:**
  ```powershell
  .\.venv\Scripts\python.exe daily_runner.py --mode watchlist --top 5
  ```

### CLI Arguments Reference

| Argument | Choices / Type | Default | Description |
|---|---|---|---|
| `--mode` | `watchlist`, `broad`, `russell1000`, `sp500`, `nifty500` | Configured in `config.py` | Defines the basket of tickers to download and scan. |
| `--top` | `int` | Configured in `config.py` | Sets the maximum number of top-ranked candidates to approve and size. |
| `--threshold` | `float` | `70.0` | Minimum composite score to qualify for buy gating (before regime bumps are added). |
| `--no-telegram` | Flag | `False` | Skips generating and sending summary reports to your Telegram bot. |

---

## 📈 2. The Core Daily Orchestrator (`main.py`)

This is your core daily orchestrator, focusing on speed and generating a pure technical scorecard for candidates.

### Standard Execution Examples

* **Watchlist Quick Scorecard:**
  ```powershell
  .\.venv\Scripts\python.exe main.py --mode watchlist --no-telegram
  ```

* **Indian Market Quick Scan:**
  ```powershell
  .\.venv\Scripts\python.exe main.py --mode nifty500
  ```

* **Broad Market Quick Scan - Top 15 Picks:**
  ```powershell
  .\.venv\Scripts\python.exe main.py --mode broad --top 15
  ```

### CLI Arguments Reference

| Argument | Choices / Type | Default | Description |
|---|---|---|---|
| `--mode` | `watchlist`, `broad`, `russell1000`, `nifty500`, `niftytotal` | Configured in `config.py` | Defines target ticker universe. |
| `--top` | `int` | Configured in `config.py` | Sets maximum number of ranked candidates written to reports. |
| `--no-telegram` | Flag | `False` | Skips delivery to Telegram. |

---

## 🛠️ 3. Outputs Generated on Run

Every time either script completes, it outputs a fresh suite of files inside the `reports/` folder:
1. **Markdown Report (`reports/daily_YYYY-MM-DD.md`):** A beautiful, human-readable overview of your portfolio holdings, correlation clusters, exit alerts, tax optimizations, and approved buy picks.
2. **Interactive Excel Workbook (`reports/backtest/backtest_*.xlsx`):** A comprehensive multi-sheet data workbook with detailed technical metrics, binned score calibrations, and sector rotations.
3. **Paper Signals Cache (`reports/.paper_signals.json`):** A machine-readable JSON cache containing today's approved candidates and exits, used by the paper trader to instantly execute actions without re-running the heavy data scan.
