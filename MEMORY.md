# MEMORY.md

## Prinsip
- WAJIB jujur dan selalu berusaha mengerti Rahman. Gak boleh asal jawab atau pura-pura ngerti.

## Trading — Moovon Fund Rules

### Size & Risk
- Bot auto (cron): TRADE_SIZE_PCT=15%, konservatif, mean reversion
- Scalping manual (perintah Rahman): TRADE_SIZE_PCT=50%
- Risk per trade: max 3% dari balance
- R:R minimum: 1:1 untuk scalp, 1:2 untuk swing

### Entry Rules
- Wajib backtest, win rate ≥60%
- Konfluensi minimal 4 dari 6 layer: Regime, News, Whale, Orderbook, TA, Accuracy
- Jam optimal: 06:00 WITA

### Exit Rules
- SL ditentukan SEBELUM entry, berdasarkan level teknikal
- SL tidak boleh dilonggarkan setelah entry (hanya trailing naik)
- OCO (TP+SL) wajib dipasang segera setelah fill
- No revenge trading

### Circuit Breaker
- Daily loss -10% → stop trading hari itu
- 3 consecutive losses → pause, review, jangan entry

### Pairs yang JANGAN discalp
- BTC: signal accuracy 0%
- ETH: signal accuracy 0%
- XRP: signal accuracy 0%

## Trading Memory
- Full trading profile di `~/trading/memory.md`
- Trade journal di `~/trading/journal.md`
- Learning progress di `~/trading/progress.md`

## Bahasa
- WAJIB menggunakan Bahasa Indonesia dalam semua percakapan, tidak boleh menggunakan bahasa asing (termasuk Inggris).
