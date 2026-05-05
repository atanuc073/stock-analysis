# 📈 Paper Trading with Alpaca

Live-test your screener's recommendations with **fake money** before risking real capital.

---

## ✨ What this gives you

- **$100,000 paper cash** to test trades against real US market prices
- All your top-scored US picks auto-executed each day (or on demand)
- Protective **STOP** orders placed broker-side — no daemon needed
- Full reconciliation: broker positions auto-sync to `portfolio.json`
- **Dry-run mode** lets you preview every order before submission
- Skips Indian stocks (`.NS` / `.BO`) — Alpaca is US-only

> **Note:** Alpaca paper trading uses **real market data** but **fake fills** at NBBO mid-price. It's the closest thing to live trading without putting money at risk.

---

## 🚀 Setup (5 minutes, one-time)

### 1. Create Alpaca account

Go to **https://alpaca.markets** → **Sign Up**.
- No SSN required for paper trading
- Use any email
- Choose "Paper Trading" (default)

### 2. Generate API keys

Dashboard → **Paper Trading** → **API Keys** → **Generate New Key**
- Copy the **API Key** (starts with `PK...`)
- Copy the **Secret Key** (shown once — save it!)

### 3. Install the SDK

```powershell
# In the project venv:
.\.venv\Scripts\Activate.ps1
pip install alpaca-py
```

If you're behind the corp proxy:

```powershell
pip install alpaca-py --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### 4. Add keys to `.env`

```ini
ALPACA_API_KEY=PK...your_key...
ALPACA_SECRET_KEY=...your_secret...
ALPACA_PAPER=true
```

### 5. Verify it works

```powershell
python paper_cli.py setup
```

You should see:

```
🔧 Alpaca Setup Check

   Mode:           PAPER
   Currency:       USD
   Cash:           $100,000.00
   Equity:         $100,000.00
   Buying power:   $200,000.00
   ...
✅ Connection OK. Ready for paper trading.
```

---

## 🎯 Daily workflow

### Step 1 — Run the daily report (gets you fresh picks)

```powershell
python daily_runner.py
```

This generates `reports/.paper_signals.json` with today's approved BUY picks + exit signals.

### Step 2 — Preview what would happen (no orders)

```powershell
python paper_cli.py preview
```

Shows every order with `[DRY-RUN]` prefix — read carefully.

### Step 3 — Execute (when you're ready)

```powershell
python paper_cli.py trade
```

Type `yes` to confirm. Orders go to Alpaca paper. You'll see fills within seconds during market hours.

### Step 4 — Check positions

```powershell
python paper_cli.py status
```

```
💼 Alpaca Account — 2025-01-15 10:42
   [PAPER]  Cash $87,234.10  |  Equity $102,116.50  |  Buying Power $189,350.20
   Open positions: 8  |  Market: OPEN

   Symbol     Qty      Entry      Last     Mkt Val    P&L $    P&L %
   ────────────────────────────────────────────────────────────────
   NVDA   45.0000     142.50    158.20  $7,119.00  +706.50    +11.02%
   AAPL   60.0000     185.10    187.40 $11,244.00  +138.00     +1.24%
   ...
```

---

## 🛠️ All commands

| Command | What it does | Safe? |
|---|---|---|
| `paper_cli.py setup` | Verify API keys + show account | ✅ Read-only |
| `paper_cli.py status` | Live positions + P&L | ✅ Read-only |
| `paper_cli.py history` | Last 20 orders submitted | ✅ Read-only |
| `paper_cli.py sync` | Pull broker positions → `portfolio.json` | ✅ Local writes only |
| `paper_cli.py preview` | Dry-run today's signals | ✅ No orders sent |
| `paper_cli.py trade` | **⚠️  Submit orders to Alpaca** | ⚠️  Confirms first |
| `paper_cli.py close NVDA` | Market-close one position | ⚠️  Confirms first |
| `paper_cli.py cancel` | Cancel all open orders | ⚠️  Confirms first |
| `paper_cli.py cancel --symbol NVDA` | Cancel orders for one symbol | ⚠️  |

### `trade` flags

```
--yes               skip confirmation prompt
--force             submit even when market is closed
--trailing          use Alpaca trailing-stop instead of fixed STOP
--max-orders 20     safety brake on total orders per run
--max-pos 10        max % of equity per position (default 10)
--min-score 65      minimum score to BUY (default 65)
```

Example — conservative trial:

```powershell
python paper_cli.py trade --max-orders 3 --max-pos 5 --min-score 75
```

---

## 🔁 How orders are placed

For each approved pick the trader submits:

1. **Market BUY** for `equity × max_pos%` worth of shares (default 10%)
2. **Stop-loss SELL** at the price computed by `PositionFactory` (ATR-based, capped at -15%) — GTC

For each exit signal (TIER_1, TIER_2, STOP_LOSS, etc.) the trader submits:

- **Market SELL** for the suggested quantity (33% on T1, 33% on T2, 100% on stop hit)

All orders carry an idempotent `client_order_id` like `copilot-buy-NVDA-2025-01-15`, so re-running the same command on the same day is **safe** — Alpaca will reject the duplicate, not double-buy.

---

## 🛡️ Safety rails (built in)

| Rail | Default | Why |
|---|---|---|
| Dry-run preview | always available | See orders before submission |
| Confirmation prompt | required for `trade` | Avoid accidents |
| Max orders per run | 20 | Prevents runaway loops |
| Max % per position | 10 | Single-name blowup cap |
| Indian ticker filter | enforced | Alpaca can't trade `.NS` / `.BO` |
| Min-score gate | 65 | Don't trade weak signals |
| Market-hours check | enforced | Skip after-hours |
| Idempotent client IDs | enforced | Re-runs are no-ops |

---

## 💡 Tips

- **Start tiny.** Use `--max-orders 3 --max-pos 5` for the first week to feel out fills.
- **Always preview first.** `python paper_cli.py preview` shows every order before any submission.
- **Reconcile daily.** After fills, run `python paper_cli.py sync` to pull broker truth into `portfolio.json`.
- **The cache is 24h.** Re-run `daily_runner.py` each morning before `paper_cli.py trade`.
- **Going live? Don't.** Keep `ALPACA_PAPER=true` until your strategy has 6+ months of paper alpha.

---

## ❓ Troubleshooting

**`Missing Alpaca credentials`** — Edit `.env` and add `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`.

**`alpaca-py not installed`** — `pip install alpaca-py`

**`No picks or exit signals available`** — Run `python daily_runner.py` first to populate `reports/.paper_signals.json`.

**`Market is closed — skipping submissions`** — Either wait for 9:30 AM ET or pass `--force` (orders will queue for next open).

**Order rejected with `not a US ticker`** — Expected. Alpaca only handles US stocks. Indian picks are skipped silently in the report.

**Re-run is rejected with `client_order_id already exists`** — Safe! The system prevented a duplicate. Cancel old orders first if you want to resubmit.

---

## 🔗 References

- [Alpaca docs](https://docs.alpaca.markets/)
- [alpaca-py SDK](https://github.com/alpacahq/alpaca-py)
- [Alpaca paper trading FAQ](https://docs.alpaca.markets/docs/paper-trading)
