# Factor Regression — Run Instructions

Fast weight optimizer for `SCORE_WEIGHTS` that **skips the backtest loop**.
Builds a (stock × date) factor matrix once, then runs SLSQP on a smooth
ranking objective (Spearman IC + monotonicity) with 200 random restarts.
Three chronological splits detect overfit vs underfit.

| Split | Range | Purpose |
|---|---|---|
| Train | 2014-01-01 .. 2020-12-31 | Fit weights via SLSQP |
| Val   | 2021-01-01 .. 2023-12-31 | Pick best candidate; detect train→val IC gap (overfit) |
| Test  | 2024-01-01 .. 2025-12-31 | Touched once; honest OOS score |

> ⚠️ **Network requirement**: yfinance must be reachable. Don't run on
> the corporate network.

---

## Prerequisites

```powershell
# from project root
.\.venv\Scripts\Activate.ps1
```

---

## Step 1 — First run (slow, builds the cache)

Downloads OHLCV + builds factor matrix + runs SLSQP. One-time cost.

```powershell
.\.venv\Scripts\python.exe -m backtest.factor_regression `
    --universe nifty500 `
    --apply-filters `
    --n-starts 200 `
    --lambda-mono 0.5 `
    --top-k 20 `
    --max-workers 6 `
    --output-dir reports/regression/run1
```

What happens:
1. Downloads OHLCV for Nifty 500 from ~2013 → 2025 (parallel, one-time).
2. Computes `score_at()` for every (stock, month-end) ≈ 72k rows.
3. Caches to `cache/factor_matrix.parquet`.
4. Applies `RS ≥ 70` + `above 200DMA` + `≤ 40% extension` filters (matches `backtest/engine.py`).
5. Runs 200 SLSQP restarts on the 2014–2020 train slice.
6. Evaluates top-20 on val + test, prints diagnostic table.

---

## Step 2 — Fast iterations (cache reused, seconds)

Re-runs skip the download/build phase. Sweep λ:

```powershell
# Lower monotonicity weight (more pure IC)
.\.venv\Scripts\python.exe -m backtest.factor_regression `
    --apply-filters --n-starts 500 --lambda-mono 0.3 `
    --output-dir reports/regression/lambda03

# Higher monotonicity (smoother factor)
.\.venv\Scripts\python.exe -m backtest.factor_regression `
    --apply-filters --n-starts 500 --lambda-mono 1.0 `
    --output-dir reports/regression/lambda10
```

Compare filtered vs unfiltered:

```powershell
.\.venv\Scripts\python.exe -m backtest.factor_regression `
    --output-dir reports/regression/unfiltered
```

Different forward-return horizon (separate cache file required):

```powershell
.\.venv\Scripts\python.exe -m backtest.factor_regression `
    --apply-filters --horizon-days 63 --rebuild-matrix `
    --cache-matrix cache/factor_matrix_63d.parquet `
    --output-dir reports/regression/h63

.\.venv\Scripts\python.exe -m backtest.factor_regression `
    --apply-filters --horizon-days 5 --freq W-FRI --rebuild-matrix `
    --cache-matrix cache/factor_matrix_5d_weekly.parquet `
    --output-dir reports/regression/h5
```

---

## Step 3 — Smoke test (recommended first)

5-minute sanity check on 100 stocks before committing to the full run.

```powershell
.\.venv\Scripts\python.exe -m backtest.factor_regression `
    --universe nifty500 --sample-size 100 `
    --apply-filters --n-starts 50 --freq W-FRI --horizon-days 5 `
    --cache-matrix cache/factor_matrix_smoke.parquet `
    --output-dir reports/regression/smoke
```

If the diagnosis block prints `[OK]` or `[WARN]` (not a crash), proceed
to Step 1.

---

## Step 4 — Verify the winning weights in a real backtest

Paste the top weight row from `candidates.csv` into `config.SCORE_WEIGHTS`,
then run the existing simulator-based optimizer for confirmation:

```powershell
.\.venv\Scripts\python.exe -m backtest.optimize `
    --start 2014-01-01 --end 2025-12-31 --universe nifty500 `
    --strategy random --candidates 5 --walk-forward `
    --objective sharpe_mono --max-workers 6 `
    --output-dir reports/optimize/verify_regression
```

Or run a single backtest with the new weights:

```powershell
.\.venv\Scripts\python.exe -m backtest.cli `
    --start 2018-01-01 --end 2025-12-31 --universe nifty500
```

---

## What you'll see during the run

- `tqdm` progress for date loop + SLSQP starts, with `best=` postfix.
- INFO logs every time SLSQP finds a new best, plus 5% checkpoints.
- Per-candidate `IC train/val/test` printed live during evaluation.
- Final block:
  - Candidate table (sorted by `val_consistent = IC_val − 0.5·|train→val gap|`)
  - Best candidate detail
  - Diagnosis: `[WARN]` flags for overfit / underfit / regime break / monotonicity collapse, or `[OK]`
  - Ready-to-paste `SCORE_WEIGHTS = {…}` block

---

## Outputs

```
reports/regression/<run>/candidates.csv          # sorted by val_consistent
reports/regression/<run>/candidates_by_train.csv # sorted by train J
cache/factor_matrix.parquet                      # reused across runs
```

Each row in `candidates.csv` has:
- `w_technical, w_fundamental, w_momentum, w_quality, w_earnings_drift`
- `IC_train, IC_val, IC_test`
- `Mono_train, Mono_val, Mono_test`
- `Q5Q1_train_pct, Q5Q1_val_pct, Q5Q1_test_pct`
- `gap_ic` (train IC − val IC)
- `val_consistent` (selection metric)

---

## All CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--universe` | `nifty500` | Universe name (any value resolved by `backtest.optimize._resolve_universe`) |
| `--sample-size` | `0` (all) | Random subset for fast tests |
| `--train-start` / `--train-end` | `2014-01-01` / `2020-12-31` | Train window |
| `--val-start` / `--val-end` | `2021-01-01` / `2023-12-31` | Validation window |
| `--test-start` / `--test-end` | `2024-01-01` / `2025-12-31` | Test window (touch once) |
| `--horizon-days` | `21` | Forward-return horizon (5, 21, 63 typical) |
| `--freq` | `ME` | Rebalance frequency (`ME` = month-end, `W-FRI` = Friday-weekly) |
| `--n-starts` | `200` | SLSQP random Dirichlet restarts |
| `--lambda-mono` | `0.5` | Weight of monotonicity in objective |
| `--top-k` | `20` | How many candidates to evaluate on val/test |
| `--cache-matrix` | `cache/factor_matrix.parquet` | Parquet cache path |
| `--rebuild-matrix` | off | Force matrix rebuild |
| `--output-dir` | `reports/regression` | CSV output directory |
| `--max-workers` | `6` | Parallel data download workers |
| `--apply-filters` | off | Gate rows on RS / 200DMA / extension |
| `--min-rs-pct` | `70.0` | RS percentile floor (when `--apply-filters`) |
| `--require-above-sma200` | on | Drop rows below 200DMA |
| `--max-extension-pct` | `40.0` | Drop rows extended > 40% above 200DMA |

---

## Interpreting the diagnosis

| Flag | Meaning | Likely cause |
|---|---|---|
| `[OK]` healthy | All 3 IC > 0, gap ≤ 0.02 | Weights generalize; promote to `config.py` |
| `[WARN]` large train→val gap | gap > 0.03 | Overfit. Use `val_consistent` ranking, not `train` |
| `[WARN]` low val IC | `IC_val < 0.01` | Underfit or factors lack signal in val regime |
| `[WARN]` monotonicity collapse | `Mono_test < 0.3` while `Mono_val ≥ 0.5` | Regime-specific weights; deploy with caution |
| `[WARN]` test IC flipped | `IC_test < 0`, `IC_val > 0` | Regime break in 2024–2025; do NOT deploy |

---

## Why this is NOT a neural network

The model is a **constrained linear combination** with only 5 parameters:

```
score = w₁·technical + w₂·fundamental + w₃·momentum + w₄·quality + w₅·earnings_drift
subject to:  w ≥ 0,  Σw = 1
```

With ~200k training rows / 5 params (ratio ≈ 40,000:1), overfit risk is
near zero. A NN with the same data would memorize noise. Same family as
Fama-French / Barra factor models. The only non-standard bits are:
- Simplex constraint (output is directly usable as `SCORE_WEIGHTS`)
- Rank-based objective (Spearman IC + quintile monotonicity), not MSE
- Multi-start SLSQP because rank objectives have flat plateaus
