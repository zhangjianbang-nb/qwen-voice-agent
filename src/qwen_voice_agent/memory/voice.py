"""Voice memory — who said what, when.

This is the store that makes the agent *recognise* a returning speaker: each
utterance carries the speaker id and (optionally) the embedding that produced the
identification, so a later session can say "welcome back, Ada" without re-enrolling.

Backed by an append-only JSONL file.  Embeddings are stored base64-encoded so the
log stays text and greppable.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["VoiceUtterance", "VoiceMemory"]


@dataclass
class VoiceUtterance:
    person_id: str
    name: str
    text: str
    ts: float
    embedding: Optional[np.ndarray] = None
    score: Optional[float] = None


class VoiceMemory:
    """Append-only log of identified utterances."""

    def __init__(
        self,
        path: str,
        cache_limit: int = 500,
        max_bytes: int = 8_000_000,
        tail_bytes: int = 2_000_000,
    ) -> None:
        self.path = path
        self._cache: List[VoiceUtterance] = []
        self._cache_limit = cache_limit
        self._max_bytes = max(0, int(max_bytes))
        self._tail_bytes = max(0, int(tail_bytes))
        if path and os.path.exists(path):
            self._load()

    def add(
        self,
        person_id: str,
        name: str,
        text: str,
        ts: Optional[float] = None,
        embedding: Optional[np.ndarray] = None,
        score: Optional[float] = None,
    ) -> VoiceUtterance:
        utt = VoiceUtterance(
            person_id=person_id,
            name=name,
            text=text,
            ts=time.time() if ts is None else ts,
            embedding=embedding,
            score=score,
        )
        self._cache.append(utt)
        if len(self._cache) > self._cache_limit:
            self._cache = self._cache[-self._cache_limit :]
        self._append(utt)
        return utt

    def recent(self, n: int = 10) -> List[VoiceUtterance]:
        return self._cache[-n:] if n > 0 else []

    def utterances_for(self, person_id: str, limit: int = 20) -> List[VoiceUtterance]:
        hits = [u for u in self._cache if u.person_id == person_id]
        return hits[-limit:] if limit > 0 else []

    def known_people(self) -> List[str]:
        return sorted({u.person_id for u in self._cache})

    # ---- persistence ---------------------------------------------------
    def _append(self, utt: VoiceUtterance) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._ensure_newline()
        self._rotate_if_needed()
        rec = {
            "person_id": utt.person_id,
            "name": utt.name,
            "text": utt.text,
            "ts": utt.ts,
            "score": utt.score,
        }
        if utt.embedding is not None:
            rec["embedding_b64"] = base64.b64encode(
                np.asarray(utt.embedding, dtype=np.float32).tobytes()
            ).decode("ascii")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _ensure_newline(self) -> None:
        """Repair a partial tail from a crash before appending.

        A killed process can leave the last JSON object without its newline; the
        next append would then merge two records into one unparseable line.
        """
        try:
            with open(self.path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                last = fh.read(1)
        except OSError:
            return
        if last != b"\n":
            with open(self.path, "ab") as fh:
                fh.write(b"\n")

    def _rotate_if_needed(self) -> None:
        """Keep one backup and start a fresh log once the file is large.

        Append-only is right for durability but wrong for a 24/7 process: without
        a bound the file grows forever and startup has to read all of it.  One
        rotated file is enough to keep the previous era greppable.
        """
        if not self._max_bytes or not os.path.exists(self.path):
            return
        try:
            if os.path.getsize(self.path) < self._max_bytes:
                return
            os.replace(self.path, self.path + ".1")
        except OSError:
            pass

    def _load(self) -> None:
        start = 0
        size = os.path.getsize(self.path)
        if self._tail_bytes and size > self._tail_bytes:
            # Read only the tail: a long-running log must not cost O(file) RAM
            # or startup time.  Drop the first partial line after seeking.
            start = size - self._tail_bytes
        with open(self.path, "r", encoding="utf-8") as fh:
            if start:
                fh.seek(start)
                fh.readline()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                emb = None
                if rec.get("embedding_b64"):
                    emb = np.frombuffer(base64.b64decode(rec["embedding_b64"]), dtype=np.float32).copy()
                self._cache.append(
                    VoiceUtterance(
                        person_id=rec.get("person_id", ""),
                        name=rec.get("name", ""),
                        text=rec.get("text", ""),
                        ts=float(rec.get("ts", 0.0)),
                        embedding=emb,
                        score=rec.get("score"),
                    )
                )
        if len(self._cache) > self._cache_limit:
            self._cache = self._cache[-self._cache_limit :]
