# Composite Scoring Workflow

End-to-end documentation of how a ticker is converted into a single
**0–100 composite score** and a verdict (`STRONG BUY` … `AVOID`).

Source files referenced:

- [analysis/composite.py](analysis/composite.py) – aggregator (`analyze`, `analyze_batch`)
- [analysis/technical.py](analysis/technical.py)
- [analysis/fundamental.py](analysis/fundamental.py)
- [analysis/momentum.py](analysis/momentum.py)
- [analysis/sentiment.py](analysis/sentiment.py)
- [analysis/forecast.py](analysis/forecast.py) (+ `forecast_timesfm.py`)
- [analysis/options_flow.py](analysis/options_flow.py)
- [analysis/quality.py](analysis/quality.py) – Novy-Marx quality factor
- [analysis/earnings_drift.py](analysis/earnings_drift.py) – PEAD anomaly
- [analysis/cross_sectional.py](analysis/cross_sectional.py) – sector-relative valuation + universe ranking
- [config.py](config.py) – weights & thresholds
- [backtest/scoring.py](backtest/scoring.py) – no-lookahead variant

---

## 1. High-Level Pipeline

```
            ┌────────────────────────────────────────────┐
            │  data_sources.yahoo.TickerData(symbol)     │
            │   - history (OHLCV), info, news, options   │
            └──────────────────────┬─────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │  analysis.composite.analyze(td) -> StockReport│
            │                                              │
            │   technical.compute(history)   -> 0..100     │
            │   fundamental.compute(info)    -> 0..100     │
            │   momentum.compute(history)    -> 0..100     │
            │   sentiment.compute(news)      -> 0..100     │
            │   forecast.compute(history)    -> 0..100     │
            │   options_flow.compute(opts)   -> 0..100     │
            │                                              │
            │   composite = Σ (sub_score × weight)         │
            │   verdict   = bucket(composite)              │
            └──────────────────────────────────────────────┘
                                   │
                                   ▼
                report_generator / portfolio / backtest
```

Each sub-analyzer is **independent**, returns the same shape:

```python
{
  "score":   float,    # 0..100
  "signals": list[str],# human-readable explanations
  ... domain-specific fields ...
}
```

The composite scorer never inspects the internal logic of a sub-score —
it only multiplies by a weight. This makes individual signals easy to
swap or tune.

---

## 2. Composite Weights

Defined in [config.py](config.py) → `SCORE_WEIGHTS` (rebalanced 2026-05 to
add quality + earnings drift; valuation now sector-relative):

| Component       | Weight | Notes                                              |
|-----------------|-------:|----------------------------------------------------|
| technical       |  0.17  | RSI, MACD, SMAs, Bollinger, base/extension         |
| fundamental     |  0.18  | P/E, P/B, ROE, D/E, growth, margins (smooth scoring) |
| momentum        |  0.28  | **Primary driver** — 12-1 factor (Jegadeesh-Titman)|
| sentiment       |  0.05  | VADER on recent headlines                          |
| forecast        |  0.06  | linear / Prophet / TimesFM 21-day projection       |
| options         |  0.04  | put/call ratio (US only)                           |
| **quality**     |  0.10  | Novy-Marx GPA, FCF, accruals, balance-sheet        |
| **earnings_drift** | 0.08 | PEAD: post-earnings drift anomaly                 |
| valuation       |  0.04  | Sector-relative P/E rank (applied cross-sectionally) |

Sum = 1.00. Momentum is the highest single weight because backtests show
the 12-1 factor is the strongest driver of forward returns. Quality and
earnings-drift were added to filter value traps and capture PEAD —
two of the most persistent anomalies in equity markets.

### 2.0 Per-ticker vs universe-aware scoring

There are now **two scoring entry points**:

- `composite.analyze(td)` — single-ticker, returns a `StockReport` whose
  `composite_score` is the per-ticker weighted blend. The `valuation`
  weight is dropped here and the surviving weights are renormalized so
  the score stays on `[0, 100]`.
- `composite.analyze_batch(td_iterable)` — scores every ticker, then
  runs `cross_sectional.apply(reports)` which adds **sector-relative
  valuation** and **cross-sectional momentum/quality rank bonus**, and
  writes the result to `StockReport.adjusted_score`. Verdicts are
  recomputed from `adjusted_score`.

Use `analyze_batch` whenever you have the full universe — it produces
better rankings because cheap-vs-sector and top-quintile signals can
only be computed in batch.


### 2.1 Options weight redistribution (live)

Indian tickers (and any US ticker without an options chain) have no
options score. In `composite.analyze`:

```python
if not rep.options.get("available"):
    opt_w = weights.pop("options")          # 0.05
    weights["technical"] += opt_w * 0.6     # +0.030
    weights["momentum"]  += opt_w * 0.4     # +0.020
```

Result for IN tickers:

| technical | fundamental | momentum | sentiment | forecast |
|----------:|------------:|---------:|----------:|---------:|
| 0.230     | 0.230       | 0.340    | 0.070     | 0.080    |

### 2.2 Backtest weight handling (no-lookahead)

In [backtest/scoring.py](backtest/scoring.py) `score_at(...)` two modes
exist:

- **`live_weights=True` (default)** — keep all weights, missing
  components (sentiment, options, optionally forecast) score a
  neutral **50 × weight**. This matches live composite behaviour and
  produces apples-to-apples scores across all backtest dates.
- **`live_weights=False` (legacy)** — drop missing components and
  redistribute their weight to technical (60%) and momentum (40%).

The `valuation` weight (0.05) is folded into `fundamental` so the
backtest sums match live.

---

## 3. Verdict Bucketing

[analysis/composite.py](analysis/composite.py#L34-L43) `_verdict(score)`:

| Composite score | Verdict       |
|----------------:|---------------|
| ≥ 75            | `STRONG BUY`  |
| 62 – 74.99      | `BUY`         |
| 45 – 61.99      | `HOLD`        |
| 32 – 44.99      | `REDUCE`      |
| < 32            | `AVOID`       |

If `TickerData.ok == False` the report short-circuits with
`composite=0`, `verdict="N/A"` and an `error` string.

---

## 4. Sub-Score: Technical (weight 0.20)

[analysis/technical.py](analysis/technical.py)

Inputs: daily OHLCV (need ≥ 60 rows or returns neutral 50).

Indicators:

- RSI(14)
- MACD(12,26,9) — line vs signal cross
- SMA50, SMA200 (or `len/2` if < 200 rows)
- Golden / Death cross (50 vs 200)
- Bollinger Bands(20) — `bband_pct`
- Volume spike vs 20-day avg (`config.VOLUME_SPIKE_MULT = 1.8`)
- 52-week high / low context
- 30-day range pullback context

Score builds **additively from a 50 baseline**, then is clipped to
`[0, 100]`:

| Condition                                       | Δ score | Signal |
|-------------------------------------------------|--------:|--------|
| RSI < 35 (oversold)                             |  +8     | `RSI oversold` |
| RSI > 70 (overbought)                           | -10     | `RSI overbought` |
| RSI ≥ 60 (late-stage)                           |  -2     | (silent) |
| MACD bullish cross                              | +10     | `MACD bullish cross` |
| MACD bearish cross                              | -10     | `MACD bearish cross` |
| Price > SMA50                                   |  +4     | – |
| Price > SMA200                                  |  +6     | – |
| Golden cross (50/200)                           | +12     | `Golden cross` |
| Death cross (50/200)                            | -12     | `Death cross` |
| Volume ≥ 1.8× avg(20)                           |  +5     | `Volume spike Nx` |
| Within 3% of 52w high (extended)                | -12     | `⚠️ Extended` |
| In base: 8–25% off high & not collapsing        | +10     | `In base` |
| Deep pullback 25–40% off high but > SMA200      |  +6     | `Deep pullback in uptrend` |
| Pulled back ≥3% from 30d high & > SMA50         |  +4     | `Pulled back in 30d range` |
| BB %B in [0, 0.05] (lower band)                 |  +4     | `At lower Bollinger band` |
| BB %B ≥ 0.95 (upper band)                       |  -6     | `At upper Bollinger band` |

> Design note: "near 52w high" is **penalized**, not rewarded — the
> opposite of the original implementation. Bases and pullbacks are
> rewarded so the strategy buys consolidation, not breakouts.

---

## 5. Sub-Score: Fundamental (weight 0.23)

[analysis/fundamental.py](analysis/fundamental.py)

Inputs: yfinance `info` dict. Empty info → neutral 50 + `error`.

Scoring uses a **smooth sigmoid** (`_smooth_score`) instead of step
thresholds so small changes (e.g. P/E 14.99 → 15.01) cause small score
changes, not cliffs.

```
_smooth_score(value, lo, hi, max_pts, invert) ∈ [-max_pts, +max_pts]
```

`lo` / `hi` define where the curve crosses zero on each side; `~88%`
of the swing happens inside `[lo, hi]`. `invert=True` means lower is
better.

| Metric          | lo / hi   | max_pts | Direction      |
|-----------------|-----------|--------:|----------------|
| `trailingPE` (or `forwardPE`) | 15 / 40    | 9 | invert (lower better) |
| `priceToBook`   | 2 / 6     | 4       | invert         |
| `returnOnEquity` (×100) | 5 / 18 | 8     | higher better  |
| `debtToEquity`  | 50 / 200  | 5       | invert (and a flat -1 penalty for any debt) |
| `earningsGrowth` (×100) | 0 / 20 | 6     | higher better  |
| `revenueGrowth` (×100)  | 0 / 15 | 4     | higher better  |
| `profitMargins` (×100)  | 5 / 20 | 4     | higher better  |
| `dividendYield` > 3%    | – | – | adds a `Dividend` signal only (no points) |

Signals (e.g. `Low P/E (12.4)`, `Strong ROE (22.0%)`, `High debt (D/E 235)`)
are added when raw values cross the user-friendly thresholds shown in
the source. Final score clipped to `[0, 100]`.

---

## 6. Sub-Score: Momentum (weight 0.32 — primary driver)

[analysis/momentum.py](analysis/momentum.py)

Computes returns `r_n = close[-1] / close[-n] - 1` for
`n ∈ {2, 5, 21, 63, 126, 252}`. Need ≥ 30 rows.

Key academic factor:

```
mom_12_1 = r252 - r21      # 12-month return MINUS most recent month
mom_6_1  = r126 - r21      # fallback when < 252d history
```

Why subtract the last month? It rewards stocks with long-term strength
and a recent **pause** (a base) — the entry pattern we want — and
filters out parabolic blow-off tops where pure 1-month momentum is
maximal.

Scoring (50 baseline, clipped to `[0,100]`):

| Condition                          | Δ score | Signal |
|------------------------------------|--------:|--------|
| `mom_12_1`: `clip(x, -30, 40) × 0.5` | -15..+20 | `12-1 mom +X% (strong base)` if >20; downtrend if <-15 |
| Else `mom_6_1`: `clip(x, -25, 30) × 0.4` | -10..+12 | `6-1 mom +X%` |
| `r21 > 25` (1M spike)              |  -8     | `⚠️ Spike +X% in 1M (extended)` |
| `15 < r21 ≤ 25`                    |  -3     | – |
| `5 ≤ r63 ≤ 25`                     |  +5     | `+X% 3M (steady)` |
| `r63 < -15`                        |  -6     | `X% 3M (weak)` |
| `r5 < -8`                          |  -4     | `X% last week (knife-catch risk)` |

---

## 7. Sub-Score: Sentiment (weight 0.07)

[analysis/sentiment.py](analysis/sentiment.py)

- Pulls up to 5 most recent headlines from `TickerData.news`.
- Scores each title with VADER (`compound ∈ [-1, 1]`).
- `avg = mean(compound)`; `score = clip(50 + avg × 50, 0, 100)`.
- Signal: `Positive news flow` if `avg > 0.3`, `Negative news flow` if `avg < -0.3`.
- No headlines → neutral 50.

Backtest: always neutral 50 (no historical news available).

---

## 8. Sub-Score: Forecast (weight 0.08)

[analysis/forecast.py](analysis/forecast.py) — strategy pattern,
selected by `FORECASTER` env (`linear` | `prophet` | `timesfm`).
All variants return the same dict shape with `score`, `signals`,
`expected_return_pct`, `forecast_price`, `horizon_days`, `model`.

Common scoring:

```
score = clip(50 + expected_return_pct × 2, 0, 100)
```

Examples:
- `+5%` projected → score 60
- `+15%` projected → score 80
- `-10%` → score 30

Variants:

| Model    | How it works                                      | Notes |
|----------|---------------------------------------------------|-------|
| `linear` | `LinearRegression` on `log(close)` of last 120d   | Default fallback. Cheap. |
| `prophet`| Facebook Prophet, weekly+yearly seasonality       | Heavier; better on seasonal series. |
| `timesfm`| Google TimesFM foundation model                    | Best, requires ~2 GB deps (`requirements-timesfm.txt`). |

If a higher-tier model fails it gracefully falls back
(`timesfm → prophet → linear`). Insufficient data (< 60 rows) → neutral 50.

---

## 9. Sub-Score: Quality (weight 0.10) — NEW

[analysis/quality.py](analysis/quality.py)

Quality is the strongest filter against value traps (cheap stocks that
stay cheap because the business is dying) and momentum blow-ups (stocks
rising on hype with weak underlying economics).

Inputs (yfinance `info`): `grossMargins`, `operatingMargins`,
`profitMargins`, `returnOnAssets`, `returnOnEquity`, `freeCashflow`,
`operatingCashflow`, `totalRevenue`, `totalAssets`, `totalDebt`,
`totalCash`, `marketCap`, `debtToEquity`, `currentRatio`,
`trailingEps`, `forwardEps`. Empty info → neutral 50.

Smooth-sigmoid scoring (50 baseline, clipped to `[0,100]`):

| Metric                                | Range (lo / hi)   | max_pts | Direction       |
|---------------------------------------|-------------------|--------:|-----------------|
| Gross-profit-to-assets (Novy-Marx)    | 0.10 / 0.40       | 10      | higher better   |
| Operating margin                      | 5% / 25%          | 6       | higher better   |
| Return on assets                      | 3% / 12%          | 6       | higher better   |
| ROIC proxy `ROE × (1 − D/E adj)`      | 4% / 15%          | 6       | higher better   |
| FCF margin (`fcf / revenue`)          | 5% / 20%          | 5       | higher better   |
| FCF yield (`fcf / market_cap`)        | 2% / 8%           | 5       | higher better   |
| Accruals gap > +10% (income >> cash)  | flat              | -6      | red flag        |
| Accruals gap < -5% (cash > income)    | flat              | +2      | conservative    |
| Current ratio < 1.0                   | flat              | -4      | liquidity risk  |
| Net cash > 10% of mcap                | flat              | +4      | fortress balance|
| Net cash < -50% of mcap (heavily levered) | flat          | -4      | balance-sheet risk |
| TTM EPS < 0 (loss)                    | flat              | -6      | quality red flag|
| Forward EPS < 70% of trailing         | flat              | -4      | declining       |

> The accruals check (Sloan 1996) detects earnings being driven by
> non-cash accruals — a strong predictor of future negative surprises.

---

## 10. Sub-Score: Earnings Drift / PEAD (weight 0.08) — NEW

[analysis/earnings_drift.py](analysis/earnings_drift.py)

Post-Earnings Announcement Drift (Bernard & Thomas 1989): stocks that
beat earnings drift in the surprise direction for 30–60 trading days.
The drift is one of the most persistent anomalies in finance.

We approximate the surprise + drift signal from yfinance + price action:

| Component                       | Source                                | Δ score range |
|---------------------------------|---------------------------------------|---------------|
| `earningsQuarterlyGrowth` (yoy) | `info`                                | ±12           |
| `revenueQuarterlyGrowth` (yoy)  | `info`                                | ±6            |
| Forward / trailing EPS guidance | `forwardEps` vs `trailingEps`         | ±6            |
| Largest abs gap in last 95 days | `Open[t] / Close[t-1] - 1`            | ±10 × decay   |
| Mismatch flag: beat + no gap    | conditional                           | -3            |

**Decay schedule** for the price-gap term (PEAD strongest 0–30d):

| Days since gap | Multiplier |
|---------------:|-----------:|
|  0 – 30        | 1.0        |
| 31 – 60        | 0.5        |
|     > 60       | 0.15       |

A gap < 3% in absolute terms is ignored as noise. The mismatch flag
penalizes stocks that report a strong beat but the market doesn't
react — historically, drift in those names tends to fizzle.

Backtest: signal is fully usable (uses static `info_static` snapshot +
slicable price history with no lookahead).

---

## 11. Sub-Score: Options Flow (weight 0.04, US only)

[analysis/options_flow.py](analysis/options_flow.py)

Inputs: `TickerData.options_summary` (nearest-expiry put/call totals).
If absent → `{"score": 50, "available": False}` and the composite
**redistributes the 0.05 weight** (60% to technical, 40% to momentum).

| Condition                                    | Δ score | Signal |
|----------------------------------------------|--------:|--------|
| Put/Call ratio (PCR) < 0.6 (bullish)         | +10     | `Bullish options flow` |
| PCR > 1.3 (bearish)                          | -10     | `Bearish options flow` |
| Total volume `cv + pv > 50,000`              |  0      | `Heavy options activity` (signal only) |

Score clipped to `[0, 100]`.

---

## 12. Cross-Sectional Post-Processor (universe-aware) — NEW

[analysis/cross_sectional.py](analysis/cross_sectional.py) — runs after
all per-ticker scoring is done. Adds two signals that pure single-ticker
scoring cannot capture.

### 12.1 Sector-relative valuation (replaces the old `valuation` slot)

For each `(sector, market)` peer group, compute the percentile rank of
the stock's `pe` (60% weight) and `pb` (40% weight) — **inverted** so
the cheapest peer scores 100 and the most expensive scores 0.

```python
adj += (sector_val_score - 50) × 0.04     # SECTOR_VAL_WEIGHT = 0.04
```

This fixes the old global P/E sigmoid where tech stocks never look
cheap (because the global P/E threshold is `~15`) and utilities always
look cheap. With < 3 peers in a sector the score defaults to neutral
50 (no adjustment).

### 12.2 Cross-sectional rank bonus

Z-scores `momentum` and `quality` across the universe, blends them,
and applies a linear ramp (`0` at z=0, full at z≥1.0):

```python
ramp = clip((mom_z + qual_z) / 2, 0, 1)
rank_bonus_score = ramp × 100
adj += (rank_bonus_score - 50) × 0.03      # RANK_BONUS_WEIGHT = 0.03
```

Top-quintile names (high relative momentum **and** quality) get a
`+1.5` bonus; the rest get nothing or a small drag. A separate `-2`
penalty is applied if **both** momentum and quality are bottom-quintile
(`z < -0.8` on each).

### 12.3 Output fields

After `analyze_batch` runs, each `StockReport` gains:

- `adjusted_score` — `composite_score` + sector-rel adj + rank bonus
- `cross_sectional` — `{sector_val_score, momentum_z, quality_z, rank_bonus_score, sector_peers_pe_n}`
- Verdict is recomputed from `adjusted_score`
- New signals appended to `all_signals`: `Cheap vs sector (Xile)`,
  `Expensive vs sector`, `Top momentum (z=...)`

Use `adjusted_score` for ranking when you have the full universe.

---

## 13. Worked Example (illustrative)

US ticker `AAPL`, hypothetical sub-scores after analysis:

| Component   | Sub-score | Weight | Contribution |
|-------------|----------:|-------:|-------------:|
| technical   | 64        | 0.20   | 12.80        |
| fundamental | 58        | 0.23   | 13.34        |
| momentum    | 71        | 0.32   | 22.72        |
| sentiment   | 55        | 0.07   |  3.85        |
| forecast    | 62        | 0.08   |  4.96        |
| options     | 60        | 0.05   |  3.00        |
| **Composite** |         | **1.00** | **60.67 → HOLD** |

If options were unavailable (e.g. an Indian ticker), the 0.05 weight
would have been redistributed:

```
technical_w = 0.20 + 0.05 × 0.6 = 0.230
momentum_w  = 0.32 + 0.05 × 0.4 = 0.340
```

and the options contribution (3.00) replaced by the redistributed
contributions of the other two.

---

## 14. Live vs Backtest at a Glance

| Aspect                     | Live (`composite.analyze` / `analyze_batch`) | Backtest (`backtest.scoring.score_at`) |
|----------------------------|---------------------------------------------|----------------------------------------|
| History window             | full available                              | sliced to `<= asof` (no lookahead)     |
| Fundamentals               | live yfinance `info`                        | static snapshot (`info_static`)        |
| Sentiment                  | VADER on real news                          | always neutral 50                      |
| Options                    | live chain (US)                             | always neutral 50                      |
| Forecast                   | per `FORECASTER`                            | optional (`include_forecast`); else 50 |
| Quality                    | live `info`                                 | live-equivalent (static `info`)        |
| Earnings drift             | live `info` + history                       | live-equivalent (static `info` + sliced history) |
| Sector-relative valuation  | yes (cross-sectional, in `analyze_batch`)   | not applied per-ticker (defer)         |
| Universe rank bonus        | yes (`analyze_batch` only)                  | not applied (defer)                    |
| Output object              | `StockReport` (with `adjusted_score`)       | `BacktestScore`                        |

---

## 15. Extending the Scorer

To add a new component:

1. Create `analysis/<name>.py` exporting `compute(...) -> dict` with at
   minimum `{"score": float, "signals": list[str]}`.
2. Add a key + weight to `SCORE_WEIGHTS` in [config.py](config.py)
   (re-normalize so weights still sum to ~1.0).
3. In [analysis/composite.py](analysis/composite.py):
   - call `compute(...)` and store on `StockReport`,
   - add the component to the `parts` dict,
   - extend `rep.all_signals` concatenation.
4. (Optional) Mirror in [backtest/scoring.py](backtest/scoring.py) if
   the data is historically available without lookahead.

Verdict thresholds and weights are intentionally centralised
(`_verdict` and `SCORE_WEIGHTS`) — tune them in one place.
