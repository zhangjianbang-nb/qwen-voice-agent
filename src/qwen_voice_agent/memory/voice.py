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

    def __init__(self, path: str, cache_limit: int = 500) -> None:
        self.path = path
        self._cache: List[VoiceUtterance] = []
        self._cache_limit = cache_limit
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
        return self._cache[-n:]

    def utterances_for(self, person_id: str, limit: int = 20) -> List[VoiceUtterance]:
        hits = [u for u in self._cache if u.person_id == person_id]
        return hits[-limit:]

    def known_people(self) -> List[str]:
        return sorted({u.person_id for u in self._cache})

    # ---- persistence ---------------------------------------------------
    def _append(self, utt: VoiceUtterance) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
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
