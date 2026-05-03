# Daily Investment System — India 🇮🇳 + US 🇺🇸

A research-driven, risk-managed daily investment system for **position investing** (6–12 month holds, ~30% target). Generates buy candidates, manages your portfolio's exits, and delivers actionable Telegram alerts.

> ⚠️ Analytical tool, **not investment advice**.

---

## What's Inside

### Phase A — Core Portfolio Engine
- **Position lifecycle**: ATR-based stops, 3-tier profit targets (T1 +20%, T2 +35%, trail rest), time-stop, thesis-break
- **Volatility-adjusted sizer**: weight = base × (score/70) × (target_vol / stock_vol)
- **Macro regime detector**: 0-10 score from price/MA/VIX → scales position sizes
- **Risk gate**: composable checks (concentration, sector, market, drawdown, regime)

### Phase B — Pro Risk Layer
- **Sector rotation**: only buy from top-quartile sectors via relative strength
- **Correlation analyzer**: detects "closet bets" (HDFC+ICICI+Kotak = same banking exposure)
- **Red flag scanner**: rule-based forensic checks (high debt, low cash conversion, price collapse, …)
- **Indian tax optimizer**: defers exits within 60 days of LTCG threshold

---

## Architecture (SOLID)

```
analysis/
├── indicators.py            # ATR, annualized volatility helpers
├── technical.py / fundamental.py / momentum.py / sentiment.py / forecast.py / options_flow.py
└── composite.py             # 0-100 composite score → BUY/AVOID

portfolio/                   # PHASE A
├── models.py                # Position, ExitSignal, Trade (pure dataclasses)
├── repository.py            # Repository protocol + JSON impl
├── lifecycle.py             # PositionFactory + ExitEvaluator (pure logic)
└── service.py               # PortfolioService orchestrator

risk/                        # PHASE A + B
├── interfaces.py            # All protocols (Sizer, RiskCheck, Ranker, Rule)
├── position_sizer.py        # VolatilityAdjustedSizer
├── regime.py                # MultiMarketRegimeDetector
├── checks.py                # 7 composable risk checks
├── gate.py                  # RiskGate (aggregator)
├── sector_rotation.py       # CompositeSectorRanker (Phase B)
├── correlation.py           # CorrelationAnalyzer (Phase B)
├── red_flags.py             # RedFlagScanner + 7 rules (Phase B)
└── tax_optimizer.py         # IndianTaxOptimizer (Phase B)

factories.py                 # Dependency-injection wiring
portfolio_cli.py             # add / status / exit / close / history / evaluate
daily_runner.py              # End-to-end daily orchestrator
```

**SOLID applied**:
- **SRP** — every module does one thing
- **OCP** — add new `RiskCheck` / `RedFlagRule` without modifying existing code
- **LSP** — uniform `RiskResult` / `Optional[RedFlag]` returns
- **ISP** — small protocols (`PositionSizer`, `RegimeDetector`, `RedFlagRule`)
- **DIP** — services depend on protocols; concrete classes injected via [factories.py](factories.py)

---

## Quick Start

```powershell
# 1. Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env → add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID

# 2. Add a position you already hold
python portfolio_cli.py add RELIANCE.NS --rupees 100000 --score 78
python portfolio_cli.py add NVDA --rupees 100000 --score 80

# 3. Status
python portfolio_cli.py status

# 4. Run the full daily analysis
python daily_runner.py

# 5. Schedule daily at 8 AM
.\schedule_task.ps1
```

---

## CLI Reference

```powershell
# Add a position (auto-computes ATR, stop, T1, T2)
python portfolio_cli.py add SYMBOL --rupees AMOUNT --score 75
python portfolio_cli.py add SYMBOL --qty 10 --price 1500.50 --score 80

# View portfolio with live prices, P&L, drawdown
python portfolio_cli.py status

# Log a tier exit (T1 = +20% sell 33%)
python portfolio_cli.py exit SYMBOL --tier 1 --price 1850

# Close fully
python portfolio_cli.py close SYMBOL --price 1450 --reason "stop hit"

# Trade history
python portfolio_cli.py history --symbol RELIANCE.NS

# Dry-run today's exit signals
python portfolio_cli.py evaluate
```

---

## Daily Run — What Happens

[`daily_runner.py`](daily_runner.py) executes:

1. Screen universe → composite scores
2. Detect regime for India + US (worst case wins)
3. Rank sectors by relative strength
4. Compute correlation clusters of your holdings
5. Scan red flags on every holding
6. Evaluate exits for every open position (stop / tiers / trailing / time / thesis / red flag)
7. Tax advice — flag positions within 60 days of LTCG
8. Find new candidates scoring ≥ 70 not currently held
9. Size each candidate (volatility-adjusted) → run through the risk gate
10. Render markdown report → send Telegram summary

### Sample Telegram alert

```
📊 Daily Update — 2026-05-03

🌐 Regime: NEUTRAL_BULL (7/10) — alloc ×0.85
💼 Total: 12,45,000  |  P&L: +1,55,000 (-2.1% DD)

🚨 EXIT SIGNALS (2)
• TIER_1 NVDA — qty 33.00 @ 540.00 (+22.5%)
• THESIS_BREAK ITC.NS — qty 200.00 @ 415.00 (-8.1%)

📅 TAX DEFERRAL OPPORTUNITY
• INFY.NS — wait 18d, save ₹15,400

🎯 NEW BUY CANDIDATES
🇮🇳 BAJFINANCE.NS (82) — 11.2% (₹1,40,000)
   T1 9,360  Stop 7,420
🇺🇸 ANET (79) — 9.8% (₹1,22,000)
   T1 462  Stop 343
```

---

## Configuration

All thresholds live in dataclasses; override at construction without touching logic:

| Module | Config class | Key knobs |
|---|---|---|
| [`portfolio/lifecycle.py`](portfolio/lifecycle.py) | `EntryParameters`, `ExitConfig` | ATR mult, hard stop %, tier %s, trailing %, time stop |
| [`risk/position_sizer.py`](risk/position_sizer.py) | `SizerConfig` | base weight, target vol, max single weight |
| [`risk/regime.py`](risk/regime.py) | `RegimeConfig` | indices, VIX thresholds, MA windows |
| [`risk/sector_rotation.py`](risk/sector_rotation.py) | `SectorRankerConfig` | lookbacks, weighting |
| [`risk/correlation.py`](risk/correlation.py) | `CorrelationConfig` | cluster threshold, lookback |
| [`risk/tax_optimizer.py`](risk/tax_optimizer.py) | `IndianTaxConfig` | rates, exemption, defer window |

---

## Extending (OCP)

### Add a new risk check

```python
# risk/checks.py
@dataclass
class MyCheck:
    name: str = "MyCheck"
    def evaluate(self, c, ctx):
        return RiskResult(passed=True, message="ok")

# factories.py
def build_default_risk_gate():
    return RiskGate([..., MyCheck()])
```

### Add a new red flag rule

```python
@dataclass
class MyRule:
    code: str = "MY_RULE"
    def check(self, symbol, info, history):
        return RedFlag(...) if condition else None

RedFlagScanner(rules=[..., MyRule()])
```

No existing code is modified — that's OCP in action.

---

## Storage

Portfolio state lives in `portfolio.json` (atomic writes). To swap to SQLite/Postgres, implement `PortfolioRepository` protocol from [`portfolio/repository.py`](portfolio/repository.py) — nothing else changes.

---

## Defaults (Aligned to Your Strategy)

| Setting | Default | Source |
|---|---|---|
| Initial stop | tighter of `−2.5 × ATR` or `−12%` | `EntryParameters` |
| Tier 1 (T1) | +20% → sell 33%, stop → break-even | `DEFAULT_TIERS` |
| Tier 2 (T2) | +35% → sell 33%, stop → +15% | `DEFAULT_TIERS` |
| Trailing (T3) | 15% off peak after both tiers | `ExitConfig` |
| Time stop | 365 days, return between −5% and +10% | `ExitConfig` |
| Thesis break | composite score < 50 | `ExitConfig` |
| Red flag exit | ≥ 2 critical flags | `ExitEvaluator` |
| Max single position | 15% | `ConcentrationCheck` |
| Max sector exposure | 25% | `SectorCheck` |
| Max correlation cluster | 30% | `CorrelationClusterCheck` |
| Drawdown circuit breakers | -5% / -10% / -15% | `DrawdownCheck` |

---

## Disclaimer

Yahoo Finance data has known limitations. Cross-check Indian fundamentals on [screener.in](https://www.screener.in) and US on [finviz.com](https://finviz.com) before acting. **Not investment advice.**
