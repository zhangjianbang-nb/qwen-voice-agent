"""MemoryManager — the single object the runtime talks to.

It owns the five stores and answers one question per turn: *what, out of
everything remembered, actually fits in the prompt?*  The answer is a character
budget (:attr:`MemoryConfig.context_budget_chars`), spent top-down on the blocks
that matter most — identity, then durable facts, then house rules, then recent
turns, then recent events.  A character budget rather than a token budget keeps
the memory layer independent of any tokenizer.

``observe(turn)`` is called once per completed turn and fans the turn out to every
store that should remember it.  The manager never writes to a store it was not
asked to; each store owns its own schema.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, List, Optional, Tuple

from ..config import MemoryConfig
from .episodic import EpisodicMemory
from .procedural import ProceduralMemory
from .semantic import SemanticMemory
from .voice import VoiceMemory
from .working import WorkingMemory

__all__ = ["MemoryManager"]

# Conservative, high-precision fact patterns.  Each is a *whole-phrase* match so
# we only record a fact the speaker stated explicitly, never one inferred from
# surrounding text.  Precision beats recall here: a wrong stored fact poisons
# every later turn.
_FACT_PATTERNS: List[Tuple[str, str]] = [
    (r"\bmy name is ([A-Za-z][\w'\- ]{1,40})", "name"),
    (r"\bi am called ([A-Za-z][\w'\- ]{1,40})", "name"),
    (r"\bi live in ([A-Za-z][\w'\- ]{1,40})", "location"),
    (r"\bi'?m from ([A-Za-z][\w'\- ]{1,40})", "origin"),
    (r"\bi work (?:at|for) ([A-Za-z][\w'\- ]{1,40})", "employer"),
    (r"\bi work as (?:a |an )?([A-Za-z][\w'\- ]{1,40})", "role"),
    (r"\bi (?:like|love|prefer) ([A-Za-z][\w'\- ]{1,40})", "preference"),
    (r"\bmy favou?rite \w+ is ([A-Za-z][\w'\- ]{1,40})", "favorite"),
]


def _overlap(a: str, b: str) -> int:
    """Cheap relevance proxy: how many whitespace tokens the two strings share."""
    ta = {w for w in re.findall(r"\w+", a.lower()) if len(w) > 2}
    tb = {w for w in re.findall(r"\w+", b.lower()) if len(w) > 2}
    return len(ta & tb)


class MemoryManager:
    """Owns the five stores; builds the per-turn context block."""

    def __init__(self, cfg: MemoryConfig, root: Optional[str] = None) -> None:
        self.cfg = cfg
        base = root or cfg.root
        base = os.path.abspath(base)

        voice_path = cfg.voice_utterance_log or os.path.join(base, "voice_utterances.jsonl")
        semantic_path = cfg.semantic_db or os.path.join(base, "semantic.sqlite3")
        episodic_path = cfg.episodic_path or os.path.join(base, "episodic.jsonl")
        procedural_path = cfg.procedural_path or os.path.join(base, "procedural.json")

        self.voice = VoiceMemory(voice_path)
        self.working = WorkingMemory(cfg.working_max_items, cfg.working_ttl_s)
        self.semantic = SemanticMemory(semantic_path)
        self.episodic = EpisodicMemory(episodic_path)
        self.procedural = ProceduralMemory(procedural_path, seed=cfg.seed_skills)
        self._lock = threading.Lock()
        self._closed = False

    # ---- per-turn write ------------------------------------------------
    def observe(self, turn: Any, ts: Optional[float] = None) -> None:
        """Record one completed turn across the stores that should hold it.

        ``turn`` is duck-typed: it needs ``user_text``, ``reply`` and optionally
        ``person_id`` / ``person_name`` / ``embedding`` / ``asr_score``.
        """
        ts = time.time() if ts is None else ts
        person_id = getattr(turn, "person_id", None) or "unknown"
        person_name = getattr(turn, "person_name", None) or "there"
        user_text = getattr(turn, "user_text", "") or ""
        reply = getattr(turn, "reply", "") or ""

        with self._lock:
            if self._closed:
                return
            self.working.add("user", user_text, ts=ts, person=person_name)
            self.working.add("assistant", reply, ts=ts, person=person_name)

            self.voice.add(
                person_id=person_id,
                name=person_name,
                text=user_text,
                ts=ts,
                embedding=getattr(turn, "embedding", None),
                score=getattr(turn, "identification_score", None),
            )

            self.episodic.append(
                "turn",
                {"user": user_text, "assistant": reply, "person": person_name},
                ts=ts,
                person=person_name,
            )

            if self.cfg.extract_facts:
                for key, value in self.extract_facts(user_text):
                    self.semantic.remember(key, value, ts=ts)

    # ---- facts ---------------------------------------------------------
    def remember_fact(self, key: str, value: str) -> None:
        self.semantic.remember(key, value)

    def recall_fact(self, key: str) -> Optional[str]:
        return self.semantic.recall(key)

    def extract_facts(self, text: str) -> List[Tuple[str, str]]:
        """Pull explicitly-stated facts out of a sentence (high precision)."""
        found: List[Tuple[str, str]] = []
        for pattern, key in _FACT_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                value = m.group(1).strip().rstrip(".,!?;:")
                if value and len(value) <= 40:
                    found.append((key, value))
        return found

    # ---- per-turn read -------------------------------------------------
    def context_block(self, person: str, query: str = "") -> str:
        """Assemble the memory context under the configured character budget.

        Blocks are added highest-priority first; a block that would overflow the
        budget is dropped whole rather than truncated mid-sentence, and lower
        blocks are skipped once the budget is spent.
        """
        budget = max(0, int(self.cfg.context_budget_chars))
        if budget == 0:
            return ""
        with self._lock:
            if self._closed:
                return ""
            blocks: List[str] = []

            # 1. identity
            blocks.append(f"Speaking with {person}.")

            # 2. durable facts, most relevant first
            facts = self.semantic.all_facts()
            if facts:
                ranked = sorted(facts, key=lambda kv: -_overlap(kv[0] + " " + kv[1], query))
                lines = [f"- {k}: {v}" for k, v in ranked[:6]]
                blocks.append("Known facts:\n" + "\n".join(lines))

            # 3. house rules (procedural)
            skills = self.procedural.all_skills()
            if skills:
                lines = [f"- {s.description}" for s in skills[:4]]
                blocks.append("House rules:\n" + "\n".join(lines))

            # 4. recent turns (working)
            recent = self.working.items()
            if recent:
                lines = [f"{it.role}: {it.text}" for it in recent[-4:]]
                blocks.append("Recent exchange:\n" + "\n".join(lines))

            # 5. recent episodes
            eps = self.episodic.recent(2)
            if eps:
                lines = [f"- {e.kind}: {e.data.get('user', '')}" for e in eps]
                blocks.append("Recently:\n" + "\n".join(lines))

            out: List[str] = []
            used = 0
            for block in blocks:
                sep = 1 if out else 0
                if used + sep + len(block) > budget:
                    break
                out.append(block)
                used += sep + len(block)
            return "\n".join(out)[:budget]

    # ---- maintenance ---------------------------------------------------
    def prune(self) -> None:
        """Drop expired working items; called by the session maintenance loop."""
        with self._lock:
            self.working.items()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.semantic.close()
        self.procedural.save()
