"""Working memory — the recent turns of the current conversation.

Two independent bounds keep it from growing with conversation length:

* ``max_items`` — a hard count cap (a ring);
* ``ttl_s`` — items older than this are dropped on read.

Both matter: the count cap bounds a fast exchange, the TTL bounds a slow one where
"a few turns ago" might be an hour ago and no longer relevant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

__all__ = ["WorkingItem", "WorkingMemory"]


@dataclass
class WorkingItem:
    role: str          # "user" | "assistant"
    text: str
    ts: float
    person: Optional[str] = None


class WorkingMemory:
    """In-process TTL ring of recent turns."""

    def __init__(self, max_items: int = 64, ttl_s: float = 900.0) -> None:
        self.max_items = max(1, int(max_items))
        self.ttl_s = float(ttl_s)
        self._items: List[WorkingItem] = []

    def add(self, role: str, text: str, ts: Optional[float] = None, person: Optional[str] = None) -> WorkingItem:
        item = WorkingItem(role=role, text=text, ts=time.time() if ts is None else ts, person=person)
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        return item

    def items(self, now: Optional[float] = None) -> List[WorkingItem]:
        """Live items only: expired ones are dropped, not merely filtered."""
        now = time.time() if now is None else now
        self._items = [it for it in self._items if now - it.ts <= self.ttl_s]
        return list(self._items)

    def clear(self) -> None:
        self._items = []

    def __len__(self) -> int:
        return len(self._items)
