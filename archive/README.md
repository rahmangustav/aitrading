# Archive

Retired components, kept for history and data — **not** meant to run.

## binance-spot-trader (archived 2026-07-07)

The original monolithic Binance spot bot (`scripts/trader.py`, ~1288 lines).
Retired in favour of the `crypto-trader` skill, which is now the canonical bot.

**Why retired** (from the 2026-07-07 code review):
- Hardcoded to LIVE `api.binance.com` with **no testnet/paper mode** — impossible
  to validate safely.
- `place_order` never checks Binance error codes → **silent failures** (rejected
  orders logged as `UNKNOWN`, no alert, no retry).
- No idempotency / `clientOrderId` → a timed-out order can double-fill.
- TP/SL is software-only (no exchange-native stop) → positions unprotected when
  the process isn't running.
- `round_step` precision bug for step sizes < 1e-4 (e.g. BTC) → quantities can
  round to zero.
- Pervasive bare `except:` hiding real errors.

**Its good logic was ported** to `crypto-trader` before archiving:
- Entry-quality filter + 4h trend filter → `crypto-trader/scripts/entry_quality.py`
- Tiered trailing stop → `crypto-trader/scripts/risk_manager.py`
  (`tiered_trailing_stop`)

**Kept for reference only:** the trade history (`trades.jsonl`), signal/learning
state (`signal_db.json`, `confidence_scores.json`, `loss_reasons.json`,
`prevention_rules.json`), and the strategy write-up (`SKILL.md`). Do not point a
live API key at this code.
