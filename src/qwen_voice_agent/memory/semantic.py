"""Semantic memory — durable facts about the user and the world.

A tiny SQLite key/value store.  SQLite rather than JSON because facts are updated
in place (a person moves, a preference changes) and queried by key, and because
the standard library ships it — no dependency.

Facts are *updated*, never appended: the whole point of semantic memory is that
"the user lives in Beijing" is one row that can change, not a growing pile of
contradictory statements.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

__all__ = ["SemanticMemory"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    ts    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_ts_idx ON facts(ts);
"""


class SemanticMemory:
    """SQLite-backed key/value facts.

    The connection is opened with ``check_same_thread=False`` and every access is
    guarded by a lock, because the runtime reads facts on the generation worker
    while the main loop writes them after a turn.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def remember(self, key: str, value: str, ts: Optional[float] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts(key, value, ts) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
                (key, value, time.time() if ts is None else ts),
            )
            self._conn.commit()

    def recall(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def forget(self, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            self._conn.commit()
        return cur.rowcount > 0

    def all_facts(self) -> List[Tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM facts ORDER BY ts DESC").fetchall()
        return [(k, v) for k, v in rows]

    def as_dict(self) -> Dict[str, str]:
        return {k: v for k, v in self.all_facts()}

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
