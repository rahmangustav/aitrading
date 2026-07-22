"""Safety net for the whole `skills/crypto-trader/tests/` suite: fail loudly
(and auto-restore) if any test writes to the REAL production learning-engine
files under `.learnings/trading/` instead of a mocked/tmp path.

Why this exists: `ct_signal_db.json` backs the live winrate numbers quoted in
VERDICT.md/MEMORY.md that gate real-money trading (winrate >=60%). This exact
class of bug -- a test that forgets to mock `learning_bridge` and lets the
real `learn_live` module run, silently appending fake signals to that file --
has already happened at least twice (see PR #11, and `test_monitor_daemon.py`
before PR #44). Both times it was caught by a human noticing a stray `git
diff`, not by CI. This fixture turns that class of bug into a hard, loud test
failure instead of a silent data-corruption bug that has to be noticed by
hand -- and restores the file so a forgotten mock can't leave production data
corrupted even if nobody looks at the CI log.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEARN_DIR = _REPO_ROOT / ".learnings" / "trading"
_GUARDED_FILES = [
    _LEARN_DIR / "ct_signal_db.json",
    _LEARN_DIR / "ACCURACY_crypto_trader.md",
]


def _snapshot():
    return {p: (p.read_bytes() if p.exists() else None) for p in _GUARDED_FILES}


@pytest.fixture(autouse=True)
def _guard_production_learning_files(request):
    before = _snapshot()
    yield
    polluted = []
    for path, original in before.items():
        current = path.read_bytes() if path.exists() else None
        if current != original:
            polluted.append(path)
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)
    if polluted:
        names = ", ".join(str(p.relative_to(_REPO_ROOT)) for p in polluted)
        pytest.fail(
            f"{request.node.nodeid} menulis ke file data produksi asli: {names} "
            "(sudah dikembalikan otomatis ke isi semula). Mock "
            "`learning_bridge`/`learn_live` atau alihkan path-nya ke tmp_path "
            "-- jangan biarkan writer asli menyentuh .learnings/trading/.",
            pytrace=False,
        )
