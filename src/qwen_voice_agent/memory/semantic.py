"""Semantic memory — durable facts about each person and the world.

A tiny SQLite key/value store, keyed by **person** as well as by fact name.
The person dimension matters in a shared room: "the user's name is Ada" and
"the user's name is Bob" must coexist, not overwrite each other.  Voice memory
already knows who spoke; semantic memory now uses that identity too.

Facts are *updated*, never appended: the whole point of semantic memory is that
"Ada lives in Beijing" is one row that can change, not a growing pile of
contradictory statements.  The table is migrated in place from the old
person-less schema, so an existing deployment keeps its facts.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

__all__ = ["SemanticMemory", "UNKNOWN_PERSON"]

UNKNOWN_PERSON = "unknown"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    person_id TEXT NOT NULL DEFAULT 'unknown',
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    ts        REAL NOT NULL,
    PRIMARY KEY (person_id, key)
);
CREATE INDEX IF NOT EXISTS facts_ts_idx ON facts(ts);
"""


class SemanticMemory:
    """SQLite-backed, per-person key/value facts.

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
        self._migrate_person_column()
        self._conn.commit()
        self._lock = threading.Lock()

    def _migrate_person_column(self) -> None:
        """Bring an old ``facts(key PRIMARY KEY)`` database up to the new schema.

        SQLite cannot change a primary key in place, so the old table is renamed,
        the new one is created, and every row is copied under ``unknown``.  The
        index must be dropped before the rename or it would keep the old name and
        the ``CREATE INDEX IF NOT EXISTS`` below would silently skip.
        """
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(facts)")]
        if not cols or "person_id" in cols:
            return
        self._conn.execute("DROP INDEX IF EXISTS facts_ts_idx")
        self._conn.execute("ALTER TABLE facts RENAME TO facts_legacy")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT INTO facts(person_id, key, value, ts) "
            "SELECT ?, key, value, ts FROM facts_legacy",
            (UNKNOWN_PERSON,),
        )
        self._conn.execute("DROP TABLE facts_legacy")
        self._conn.commit()

    # ---- write ---------------------------------------------------------
    def remember(
        self, key: str, value: str, person_id: Optional[str] = None, ts: Optional[float] = None
    ) -> None:
        person = person_id or UNKNOWN_PERSON
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts(person_id, key, value, ts) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(person_id, key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
                (person, key, value, time.time() if ts is None else ts),
            )
            self._conn.commit()

    def forget(self, key: str, person_id: Optional[str] = None) -> bool:
        person = person_id or UNKNOWN_PERSON
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE person_id = ? AND key = ?", (person, key)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def forget_oldest(self, n: int = 1, person_id: Optional[str] = None) -> int:
        """Drop the least recently updated facts; used to bound a 24/7 store."""
        n = max(0, int(n))
        if n == 0:
            return 0
        with self._lock:
            if person_id:
                rows = self._conn.execute(
                    "SELECT person_id, key FROM facts WHERE person_id = ? ORDER BY ts ASC LIMIT ?",
                    (person_id, n),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT person_id, key FROM facts ORDER BY ts ASC LIMIT ?", (n,)
                ).fetchall()
            for person, key in rows:
                self._conn.execute(
                    "DELETE FROM facts WHERE person_id = ? AND key = ?", (person, key)
                )
            self._conn.commit()
        return len(rows)

    # ---- read ----------------------------------------------------------
    def recall(self, key: str, person_id: Optional[str] = None) -> Optional[str]:
        person = person_id or UNKNOWN_PERSON
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM facts WHERE person_id = ? AND key = ?", (person, key)
            ).fetchone()
        return row[0] if row else None

    def recall_any(self, key: str) -> Optional[str]:
        """Return the most recently updated value for ``key`` across all people.

        This is the unscoped fallback used when the caller does not know who is
        speaking; it is not used for a known person, so a shared room still keeps
        each speaker's facts separate.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM facts WHERE key = ? ORDER BY ts DESC LIMIT 1", (key,)
            ).fetchone()
        return row[0] if row else None

    def all_facts(self, person_id: Optional[str] = None) -> List[Tuple[str, str]]:
        """Facts for one person, or every fact when ``person_id`` is None."""
        with self._lock:
            if person_id is None:
                rows = self._conn.execute(
                    "SELECT key, value FROM facts ORDER BY ts DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT key, value FROM facts WHERE person_id = ? ORDER BY ts DESC",
                    (person_id,),
                ).fetchall()
        return [(k, v) for k, v in rows]

    def as_dict(self, person_id: Optional[str] = None) -> Dict[str, str]:
        return {k: v for k, v in self.all_facts(person_id)}

    def count(self, person_id: Optional[str] = None) -> int:
        with self._lock:
            if person_id is None:
                return int(self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE person_id = ?", (person_id,)
                ).fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
