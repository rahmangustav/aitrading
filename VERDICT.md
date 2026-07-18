# Vonis Akhir — Proyek aitrading

**Tanggal:** 18 Juli 2026
**Status:** Riset SELESAI. **Jangan pakai uang nyata.**

---

## Ringkasan satu paragraf

Dua keluarga strategi yang dipunyai bot ini — *mean reversion* dan *trend
following* — sekarang sudah diuji pada sampel besar data Binance asli, dengan
model biaya yang sama (fee 0,1% + slippage 0,05% per sisi). **Keduanya kalah.**
Tidak ada satu pun varian parameter yang lolos gate dana (winrate ≥60%); yang
terbaik pun masih di bawah 40% dengan profit factor <1. Track record 8 sinyal
live (winrate 37,5% pada horizon 8 jam) konsisten dengan angka backtest, jadi
ini bukan artefak simulasi. Kesimpulannya: kode infrastrukturnya sehat, tapi
**edge-nya tidak ada**. Menaruh uang nyata di sini adalah membayar biaya
transaksi untuk melempar koin yang berat sebelah ke arah rugi.

---

## Bukti 1 — Mean reversion (Juli 2026, sudah ada sebelumnya)

Backtest 432 koin, lalu validasi v2 dengan regime filter (ADX) + ATR stop:

| Varian | Trades | Winrate | Total return |
|---|---|---|---|
| Tanpa filter (top-40, 4h, 24 bln) | 601 | 43,4% | −549% |
| + master switch SMA200 | 107 | 36,4% | −180% |

Filter memang memotong kerugian, tapi **menurunkan** winrate. Angka +3,9% yang
sempat muncul di sampel 10 trade ternyata cuma derau sampel kecil.

## Bukti 2 — Trend following (BARU, 18 Juli 2026)

Ini celah yang sebelumnya terlewat: **strategi yang benar-benar jalan live
justru trend following** (golden cross EMA 9/21 + RSI<70) — semua 8 sinyal di
`ct_signal_db.json` berasal dari sini — tapi yang pernah divalidasi besar-besaran
malah mean reversion. Sekarang sudah diuji dengan harness setara
(`tf_backtest.py` + `validate_tf.py`), replikasi persis logika live termasuk
urutan exit-nya.

**40 pair paling likuid · timeframe 4h · 24 bulan · 2.876 trade:**

| Varian parameter | Trades | Winrate | PF | Σ return |
|---|---|---|---|---|
| live (SL 5% / TP 10%) | 2.876 | 30,6% | 0,83 | −1.064% |
| close-only (persis evaluasi live) | 2.876 | 30,8% | 0,88 | −867% |
| + master switch SMA200 | 1.161 | 31,4% | 0,84 | −462% |
| stop ketat (SL 3% / TP 9%) | 2.876 | 26,7% | 0,81 | −1.046% |
| stop lebar (SL 8% / TP 16%) | 2.876 | 31,3% | 0,89 | −879% |
| trailing 8% | 2.876 | 30,6% | 0,83 | −1.062% |
| gate ADX>25 (hanya pasar trending) | 754 | 26,4% | 0,63 | −689% |

Tidak ada satu baris pun yang mendekati gate 60%. Yang paling menarik: gate ADX
— yang secara teori *seharusnya* menolong trend following — justru menghasilkan
winrate **terburuk** (26,4%) dan PF terburuk (0,63). Golden cross EMA di pasar
kripto sistematis telat: waktu cross terjadi, gerakannya sudah habis.

## Bukti 3 — Track record live

8 sinyal, diverifikasi ulang setelah bug horizon diperbaiki (commit `6eb1575`):
winrate 4 jam 25%, 8 jam 37,5%, 24 jam 25% — **sebelum** fee. Sejalan dengan
backtest ~31%. Angka lama "50% / −1,7%" adalah artefak bug dan tidak boleh
dikutip lagi.

---

## Keputusan

1. **Uang nyata: TIDAK.** Gate winrate ≥60% tidak terpenuhi, bahkan tidak
   mendekati. Gate ini tetap berlaku.
2. **Daemon & cron: mati.** Sudah dipastikan tidak ada proses yang jalan
   (18 Juli 2026). Jangan dihidupkan lagi untuk strategi yang sudah divonis ini.
3. **Kode dipertahankan sebagai infrastruktur riset,** bukan sebagai bot yang
   siap pakai. Yang bernilai dan sudah terbukti benar: harness backtest, model
   biaya, regime detection, risk manager, sandbox testnet, 82 unit test.

## Kalau suatu hari mau dilanjutkan

Jangan ulangi menyetel-nyetel parameter dari dua strategi yang sudah divonis ini
— ruang parameternya sudah disapu dan seluruhnya negatif. Yang belum diuji dan
punya dasar teori berbeda:

- **Sumber edge non-harga:** funding rate, order flow imbalance, data on-chain.
  Semua strategi yang diuji sejauh ini cuma memakai OHLCV, dan OHLCV murni
  hampir pasti sudah habis diarbitrase.
- **Cross-sectional momentum** (rangking relatif antar-koin), bukan sinyal
  per-koin yang berdiri sendiri.
- **Market making / spread capture**, yang menghasilkan uang dari biaya
  transaksi alih-alih membayarnya — perhatikan bahwa biaya adalah pembunuh
  utama di semua tabel di atas.

Aturan mainnya tetap sama: sampel besar dulu, gate 60% dulu, baru uang.

---

## Cara mereproduksi

```bash
cd ~/aitrading && source .venv/bin/activate
cd skills/crypto-trader/scripts

# Trend following (bukti 2)
python3 validate_tf.py --top 40 --timeframe 4h --months 24 --grid --json

# Mean reversion (bukti 1)
python3 validate_mr.py --top 40 --timeframe 4h --months 24 --bull-sma 200 --json

# Test
cd .. && python3 -m pytest tests/ -q     # 82 passed
```

Hasil mentah tersimpan di `skills/crypto-trader/data/backtests/`.
