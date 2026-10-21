"""MemoryManager — the single object the runtime talks to.

It owns the five stores and answers one question per turn: *what, out of
everything remembered, actually fits in the prompt?*  The answer is a character
budget (:attr:`MemoryConfig.context_budget_chars`), spent top-down on the blocks
that matter most — identity, then durable facts, then what this person said
recently, then house rules, then the last exchange, then recent events.  A
character budget rather than a token budget keeps the memory layer independent of
any tokenizer.

``observe(turn)`` is called once per completed turn and fans the turn out to every
store that should remember it.  The manager never writes to a store it was not
asked to; each store owns its own schema.

Two details make the difference between a log and a memory:

* facts are keyed by **person**, because a shared room has more than one speaker;
* the *voice* store is read back into the prompt, not just written.  A store that
  is never read is an audit trail, not a memory.
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
from .semantic import SemanticMemory, UNKNOWN_PERSON
from .voice import VoiceMemory
from .working import WorkingMemory

__all__ = ["MemoryManager"]

# Conservative, high-precision fact patterns.  Each is a *whole-phrase* match so
# we only record a fact the speaker stated explicitly, never one inferred from
# surrounding text.  Precision beats recall here: a wrong stored fact poisons
# every later turn.  The Chinese patterns are equally strict: they match "我叫
# 小明" but not "我叫什么名字" (a question), because the value is rejected when it
# contains an interrogative word.
_FACT_PATTERNS: List[Tuple[str, str]] = [
    (r"\bmy name is ([A-Za-z][\w'\- ]{1,40})", "name"),
    (r"\bi am called ([A-Za-z][\w'\- ]{1,40})", "name"),
    (r"\bi live in ([A-Za-z][\w'\- ]{1,40})", "location"),
    (r"\bi'?m from ([A-Za-z][\w'\- ]{1,40})", "origin"),
    (r"\bi work (?:at|for) ([A-Za-z][\w'\- ]{1,40})", "employer"),
    (r"\bi work as (?:a |an )?([A-Za-z][\w'\- ]{1,40})", "role"),
    (r"\bi (?:like|love|prefer) ([A-Za-z][\w'\- ]{1,40})", "preference"),
    (r"\bmy favou?rite \w+ is ([A-Za-z][\w'\- ]{1,40})", "favorite"),
    (r"(?:我|俺|本人)(?:的)?名字(?:是|叫)\s*([^\s，。！？、；：,.!?;:]{1,40})", "name"),
    (r"我叫\s*([^\s，。！？、；：,.!?;:]{1,40})", "name"),
    (r"我(?:住在|家住在|家在)\s*([^\s，。！？、；：,.!?;:]{1,40})", "location"),
    (r"我(?:在|于)\s*([^\s，。！？、；：,.!?;:]{1,20})\s*(?:工作|上班)", "employer"),
    (r"我的(?:职业|工作)是\s*([^\s，。！？、；：,.!?;:]{1,20})", "role"),
    (r"我(?:喜欢|爱|爱好|偏爱)\s*([^\s，。！？、；：,.!?;:]{1,30})", "preference"),
    (r"我对\s*([^\s，。！？、；：,.!?;:]{1,20})\s*过敏", "allergy"),
    (r"我的生日(?:是|在)\s*([^\s，。！？、；：,.!?;:]{1,20})", "birthday"),
]

# A value containing one of these is a question, not a fact ("我叫什么名字").
_QUESTION_WORDS = ("什么", "哪", "谁", "多少", "几", "吗", "呢", "怎么", "如何")


def _tokens(text: str) -> set:
    """Tokens for the relevance proxy, CJK-aware.

    ``\\w+`` treats a whole Chinese clause as one token, so "北京" never matches
    a longer fact value.  Character bigrams give Chinese a usable overlap score
    without a tokenizer dependency.
    """
    lowered = text.lower()
    out = {w for w in re.findall(r"[a-z0-9]+", lowered) if len(w) > 1}
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        out.add(run)
        out.update(run[i : i + 2] for i in range(len(run) - 1))
    return out


def _overlap(a: str, b: str) -> int:
    return len(_tokens(a) & _tokens(b))


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

        self.voice = VoiceMemory(
            voice_path, max_bytes=cfg.voice_rotate_bytes, tail_bytes=cfg.jsonl_tail_bytes
        )
        self.working = WorkingMemory(cfg.working_max_items, cfg.working_ttl_s)
        self.semantic = SemanticMemory(semantic_path)
        self.episodic = EpisodicMemory(
            episodic_path, max_bytes=cfg.episodic_rotate_bytes, tail_bytes=cfg.jsonl_tail_bytes
        )
        self.procedural = ProceduralMemory(procedural_path, seed=cfg.seed_skills)
        self._lock = threading.Lock()
        self._closed = False
        # A restarted process has an empty working memory, but the episodic log
        # still holds the last exchange.  Prewarming means "what did I just say"
        # survives a restart instead of being answered from a blank slate.
        self._prewarm_working()

    def _prewarm_working(self) -> None:
        n = max(0, int(self.cfg.prewarm_turns))
        if not n:
            return
        for ep in self.episodic.by_kind("turn", limit=n):
            user = ep.data.get("user")
            assistant = ep.data.get("assistant")
            if user:
                self.working.add("user", user, person=ep.person)
            if assistant:
                self.working.add("assistant", assistant, person=ep.person)

    # ---- per-turn write ------------------------------------------------
    def observe(self, turn: Any, ts: Optional[float] = None) -> None:
        """Record one completed turn across the stores that should hold it.

        ``turn`` is duck-typed: it needs ``user_text``, ``reply`` and optionally
        ``person_id`` / ``person_name`` / ``embedding`` / ``identification_score``.
        """
        ts = time.time() if ts is None else ts
        person_id = getattr(turn, "person_id", None) or UNKNOWN_PERSON
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
                    self.semantic.remember(key, value, person_id=person_id, ts=ts)

    def observe_proactive(self, turn: Any, ts: Optional[float] = None) -> None:
        """Record a reminder or scheduled task spoken without being addressed.

        There is no user utterance to store, so voice memory and fact extraction
        are skipped; working memory gets the spoken line (so the next turn can
        refer to it) and episodic memory gets a ``reminder`` or ``task`` event.
        """
        ts = time.time() if ts is None else ts
        person_name = getattr(turn, "person_name", None) or "there"
        reply = getattr(turn, "reply", "") or ""
        schedule_id = getattr(turn, "schedule_id", None)
        kind = getattr(turn, "proactive_kind", "reminder") or "reminder"
        with self._lock:
            if self._closed:
                return
            self.working.add("assistant", reply, ts=ts, person=person_name)
            self.episodic.append(
                kind,
                {"text": reply, "schedule_id": schedule_id, "person": person_name},
                ts=ts,
                person=person_name,
            )

    # ---- facts ---------------------------------------------------------
    def remember_fact(self, key: str, value: str, person_id: Optional[str] = None) -> None:
        self.semantic.remember(key, value, person_id=person_id)

    def recall_fact(self, key: str, person_id: Optional[str] = None) -> Optional[str]:
        """Read a fact.  With no person given, fall back to any person's fact.

        Facts are stored per speaker so two people can each have a ``name``.
        A caller that does not know the speaker (the tool loop before the first
        voiceprint match, or a test) still expects ``recall_fact("location")`` to
        find the fact it just wrote, so an unscoped lookup falls back to the most
        recently updated matching key.
        """
        value = self.semantic.recall(key, person_id=person_id)
        if value is None and person_id is None:
            value = self.semantic.recall_any(key)
        return value

    def extract_facts(self, text: str) -> List[Tuple[str, str]]:
        """Pull explicitly-stated facts out of a sentence (high precision)."""
        found: List[Tuple[str, str]] = []
        seen = set()
        for pattern, key in _FACT_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            value = m.group(1).strip().rstrip(".,!?;:，。！？；：")
            if not value or len(value) > 40:
                continue
            if any(word in value for word in _QUESTION_WORDS):
                continue  # "我叫什么名字" is a question, not a fact
            if (key, value) not in seen:
                seen.add((key, value))
                found.append((key, value))
        return found

    # ---- per-turn read -------------------------------------------------
    def context_block(self, person: str, query: str = "", person_id: Optional[str] = None) -> str:
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

            # 2. durable facts for this person, most relevant first
            facts = self.semantic.all_facts(person_id)
            if facts:
                ranked = sorted(facts, key=lambda kv: -_overlap(kv[0] + " " + kv[1], query))
                lines = [f"- {k}: {v}" for k, v in ranked[:6]]
                blocks.append("Known facts:\n" + "\n".join(lines))

            # 3. what this person said recently (voice memory)
            recent_voice = (
                self.voice.utterances_for(person_id, 3)
                if person_id
                else self.voice.recent(3)
            )
            if recent_voice:
                lines = [f"- {u.name}: {u.text[:60]}" for u in recent_voice]
                blocks.append(f"Recent from {person}:\n" + "\n".join(lines))

            # 4. house rules (procedural)
            skills = self.procedural.all_skills()
            if skills:
                lines = [f"- {s.description}" for s in skills[:4]]
                blocks.append("House rules:\n" + "\n".join(lines))

            # 5. recent turns (working)
            recent = self.working.items()
            if recent:
                lines = [f"{it.role}: {it.text}" for it in recent[-4:]]
                blocks.append("Recent exchange:\n" + "\n".join(lines))

            # 6. recent episodes
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
        """Drop expired working items and bound the semantic store.

        Called by the session maintenance loop.  Voice/episodic bound themselves
        at write time by rotating the log; working memory is bounded on read;
        semantic memory is the one store that needs an explicit capacity policy.
        """
        with self._lock:
            self.working.items()
            cap = max(0, int(self.cfg.semantic_max_facts))
            if cap:
                total = self.semantic.count()
                if total > cap:
                    self.semantic.forget_oldest(total - cap)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.semantic.close()
        self.procedural.save()
