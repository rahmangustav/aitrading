# CLAUDE.md

Guidance for AI assistants (Claude Code and similar) working in this repository.

## What this project is

`aitrading` is a **cryptocurrency trading research** codebase, not a
production trading bot. It contains two OpenClaw-style "skills" plus a
lightweight learning engine. The infrastructure (backtest harness, cost model,
regime detection, risk manager, testnet sandbox, unit tests) is sound and
worth maintaining; the *trading strategies themselves have been formally
judged to have no edge*.

**Read `VERDICT.md` before doing anything strategy-related.** Its conclusion
(18 July 2026): both strategy families (mean reversion and trend following)
lose across a full parameter sweep on large real-Binance samples — none clears
the ≥60% win-rate funding gate, and the 8-signal live track record (~37.5%)
confirms the backtests. The decision on record:

- **No real money.** The `CRYPTO_DEMO=false` live path exists but must not be
  used for the strategies already vetoed. The ≥60% win-rate gate stands.
- **Daemon and cron are OFF** and must stay off for vetoed strategies.
- **Keep the code as research infrastructure.** Do not tune parameters of the
  two vetoed strategies — that space is swept and uniformly negative. If
  extending, pursue genuinely new edge sources (funding rate, order-flow
  imbalance, on-chain data, cross-sectional momentum, market making) — see
  the "Kalau suatu hari mau dilanjutkan" section of `VERDICT.md`.

## Language convention (important)

Per `MEMORY.md`, all conversation with the repository owner (Rahman) **must be
in Bahasa Indonesia** — no foreign language, including English. Code, comments,
and docstrings in this repo are a mix of Indonesian and English; match the
style of the file you are editing. New docstrings in the `crypto-trader` and
`.learnings` code are typically Indonesian; the `trading-research` skill is in
English.

`MEMORY.md` also encodes the owner's trading rules (size/risk, entry/exit,
circuit breakers, pairs never to scalp — BTC/ETH/XRP show 0% signal accuracy).
Respect these when asked to reason about trading, but remember the VERDICT
supersedes any "go live" impulse.

## Repository layout

```
.
├── VERDICT.md                    # FINAL research verdict — read first
├── MEMORY.md                     # Owner's principles, trading rules, language rule
├── requirements.txt              # Top-level deps (ccxt, httpx, pandas, PyYAML, ...)
├── pytest.ini                    # Test discovery roots
├── .github/workflows/            # CI (python-package.yml) + Claude review bots
├── .learnings/trading/           # Live Learning Engine (signal DB, accuracy, reviews)
├── archive/                      # Retired code, kept for history — DO NOT RUN
└── skills/
    ├── crypto-trader/            # The canonical bot/research harness
    └── trading-research/         # Read-only Binance analysis tools (no execution)
```

### `skills/crypto-trader/` — the main codebase

Automated trading skill with 8 strategies, multi-exchange support, risk
management, backtesting, and a monitoring daemon. Entry point is
`scripts/main.py` (a `--mode` CLI emitting JSON to stdout).

```
scripts/
├── main.py                 # CLI entrypoint: status/balance/start_strategy/
│                           #   stop_strategy/list_strategies/backtest/history/
│                           #   sentiment/monitor/emergency_stop
├── exchange_manager.py     # ccxt wrapper; testnet by default; order safety
├── risk_manager.py         # Risk limits, kill switch, tiered trailing stop
├── strategy_engine.py      # BaseStrategy + strategy lifecycle/state persistence
├── strategies/             # One file per strategy (grid, dca, trend_following,
│                           #   scalping, arbitrage, swing_trading, copy_trading,
│                           #   rebalancing)
├── backtester.py           # Generic strategy backtest engine
├── monitor_daemon.py       # Background loop: orders/portfolio/risk/signals/sentiment
├── notifier.py             # Telegram/Discord/Email alerts
├── sentiment_analyzer.py   # RSS/CryptoPanic/Reddit sentiment (VADER)
├── regime.py               # Trending-vs-ranging classification (ADX/ATR)
├── entry_quality.py        # Entry-quality + 4h trend filter (ported from archive)
├── learning_bridge.py      # Fail-open bridge to the Live Learning Engine
├── sandbox_check.py        # Testnet connectivity/safety preflight
├── cache.py                # Simple TTL cache for market data
├── {mr,tf,csm}_backtest.py # Focused research backtests (mean-rev/trend/cross-sec momentum)
└── validate_{mr,tf,csm}.py # Real-data validation harnesses (ccxt public OHLCV)
config/
├── exchanges.yaml          # Exchange connectivity, sandbox mode, rate limits
├── strategies.yaml         # Per-strategy default params (override via --params)
├── risk_limits.yaml        # Global + per-strategy risk limits
├── notifications.yaml      # Alert routing per event/channel
└── cron_examples.yaml      # Example cron jobs (NOT auto-activated)
tests/                      # 32 pytest files
data/backtests/            # Runtime output (gitignored except committed results)
```

Full usage reference lives in `skills/crypto-trader/SKILL.md` — the source of
truth for CLI modes, strategy params, config, and safety rules.

### `skills/trading-research/` — analysis-only tools

Research/analysis scripts against Binance **public** API (no auth, no order
execution). Each script is a standalone CLI:
`binance_market.py`, `technical_analysis.py`, `dca_calculator.py`,
`position_sizer.py`, `market_scanner.py`, `whale_tracker.py`. Reference docs in
`references/` (`binance-api.md`, `indicators.md`, `strategies.md`). See its
`SKILL.md` for command examples.

### `.learnings/trading/` — Live Learning Engine

`learn_live.py` records live signals, verifies them against real Binance
mainnet prices, and computes accuracy. State/reports live here
(`signal_db.json`, `ct_signal_db.json`, `ACCURACY*.md`, `PATTERNS.md`, dated
`review-*.md`). `crypto-trader/scripts/learning_bridge.py` reuses these
functions but redirects DB paths so it never clobbers legacy bot data. All
learning functions are **fail-open** — a learning error must never block
trading logic.

### `archive/` — do not run

Retired monolithic `binance-spot-trader`. Kept only for history and data. Its
good logic was ported into `crypto-trader` (`entry_quality.py`,
`risk_manager.tiered_trailing_stop`) before retirement. See `archive/README.md`
for why it was retired (hardcoded live, silent order failures, no idempotency,
etc.). **Never point a live API key at archived code.**

## Development workflow

### Environment

```bash
cd skills/crypto-trader
pip install -r requirements.txt      # ccxt, pandas, PyYAML, python-dotenv, vaderSentiment, feedparser, ...
```

- Python **3.11 / 3.12** (CI matrix).
- Config via env vars / `.env` at repo root. Key vars:
  `BINANCE_API_KEY`, `BINANCE_API_SECRET`, and `CRYPTO_DEMO` (defaults to
  `true` = paper/testnet). Optional: other exchanges, `TELEGRAM_*`,
  `DISCORD_WEBHOOK_URL`, `CRYPTOPANIC_API_KEY`.
- **`CRYPTO_DEMO=true` is the safe default. Never set it false without explicit,
  unambiguous instruction — and not for strategies vetoed in `VERDICT.md`.**

### Running things

```bash
# Bot CLI (JSON out):
python3 scripts/main.py --mode status
python3 scripts/main.py --mode backtest --strategy trend_following \
    --params '{"symbol":"BTC/USDT","timeframe":"4h"}' --start 2025-01-01 --end 2025-12-31

# Research validation (public data, no key), from scripts/:
python3 validate_tf.py --top 40 --timeframe 4h --months 24 --grid --json
python3 validate_mr.py --top 40 --timeframe 4h --months 24 --bull-sma 200 --json

# Analysis tools:
python3 skills/trading-research/scripts/technical_analysis.py --symbol BTCUSDT --interval 4h
```

### Testing

```bash
pip install pytest
python3 -m pytest -q               # from repo root; pytest.ini sets the roots
python3 -m pytest skills/crypto-trader/tests -v
```

`pytest.ini` discovers tests under `skills/crypto-trader/tests`,
`skills/trading-research/tests`, and `.learnings/trading/tests` (~560 test
functions across 39 files). Tests use fixtures/mocks — they do **not** hit live
exchanges or place orders. Add a test with any behavioural change.

### CI (`.github/workflows/python-package.yml`)

On push/PR to `main`, across Python 3.11 & 3.12:
1. Install deps.
2. `flake8` — build fails on `E9,F63,F7,F82` (syntax/undefined-name); style
   issues (`--max-line-length=127`, `--max-complexity=10`) are exit-zero warnings.
3. `pytest`.

Keep the syntax/undefined-name flake8 selection green — that gate is hard.
`claude.yml` and `claude-code-review.yml` are the Claude PR-assistant/review bots.

## Conventions

- **JSON I/O.** `main.py` modes print structured JSON to stdout; logs go to
  stderr. Preserve this contract — callers parse stdout.
- **Fail-closed vs fail-open by design.** Regime/entry filters fail *closed*
  (skip the trade on insufficient data). The learning bridge fails *open*
  (never block trading). Respect the intent of each module.
- **Safety is not optional.** Never bypass risk limits or the kill switch,
  even on request — explain the limit instead. Always confirm before starting a
  strategy in live mode; show the full parameter set. API keys should carry
  trade permission only, never withdrawal. Kill-switch/state files live under
  `~/.openclaw/`.
- **Cost model.** Research backtests assume fee 0.1% + slippage 0.05% per side.
  Keep this consistent so results stay comparable — cost is the dominant factor
  in every negative result in `VERDICT.md`.
- **Runtime output** (`skills/crypto-trader/data/`) is gitignored; don't commit
  generated backtest/cache files unless they are intentional evidence.

## Git workflow

- Work happens on feature branches merged to `main` via PR (see `git log` —
  Indonesian branch names like `perbaikan/...`, `tes/...`).
- Push with `git push -u origin <branch>`; retry transient network failures
  with exponential backoff.
- **Do not open a PR unless explicitly asked.** If asked, check for a PR
  template first.
- Do not commit secrets or `.env`. Keep commit messages clear and descriptive.

## Before you change strategy logic — checklist

1. Re-read `VERDICT.md`. Is this a vetoed strategy (mean reversion / trend
   following) or a genuinely new edge source?
2. Big sample first, ≥60% win-rate gate first, *then* consider anything live.
3. Reproduce with the existing validation harnesses; keep the shared cost model.
4. Add/adjust tests. Keep flake8 syntax gate and pytest green.
5. Never flip `CRYPTO_DEMO` to live or re-enable the daemon/cron for vetoed
   strategies.
