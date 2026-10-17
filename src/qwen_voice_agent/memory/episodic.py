"""Episodic memory — an append-only log of what happened.

Semantic memory says what is true; episodic memory says what *occurred*: "at
14:03 Ada asked about the weather and I answered".  It is append-only and never
mutated, which is what makes it usable as an audit trail and as the raw material
for later consolidation into semantic facts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Episode", "EpisodicMemory"]


@dataclass
class Episode:
    kind: str
    data: Dict[str, Any]
    ts: float
    person: Optional[str] = None


class EpisodicMemory:
    """JSONL-backed append-only event log."""

    def __init__(self, path: str, cache_limit: int = 1000) -> None:
        self.path = path
        self._cache: List[Episode] = []
        self._cache_limit = cache_limit
        if path and os.path.exists(path):
            self._load()

    def append(self, kind: str, data: Optional[Dict[str, Any]] = None, ts: Optional[float] = None, person: Optional[str] = None) -> Episode:
        ep = Episode(
            kind=kind,
            data=data or {},
            ts=time.time() if ts is None else ts,
            person=person,
        )
        self._cache.append(ep)
        if len(self._cache) > self._cache_limit:
            self._cache = self._cache[-self._cache_limit :]
        self._write(ep)
        return ep

    def recent(self, n: int = 10) -> List[Episode]:
        return self._cache[-n:] if n > 0 else []

    def by_kind(self, kind: str, limit: int = 20) -> List[Episode]:
        hits = [e for e in self._cache if e.kind == kind]
        return hits[-limit:] if limit > 0 else []

    def __len__(self) -> int:
        return len(self._cache)

    # ---- persistence ---------------------------------------------------
    def _write(self, ep: Episode) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        rec = {"kind": ep.kind, "data": ep.data, "ts": ep.ts, "person": ep.person}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._cache.append(
                    Episode(
                        kind=rec.get("kind", ""),
                        data=rec.get("data", {}) or {},
                        ts=float(rec.get("ts", 0.0)),
                        person=rec.get("person"),
                    )
                )
        if len(self._cache) > self._cache_limit:
            self._cache = self._cache[-self._cache_limit :]
