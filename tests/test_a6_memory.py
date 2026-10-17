"""A6 — the five memory stores and the manager's context budget."""

import numpy as np
import pytest

from qwen_voice_agent.config import MemoryConfig
from qwen_voice_agent.memory import (
    EpisodicMemory,
    MemoryManager,
    ProceduralMemory,
    SemanticMemory,
    VoiceMemory,
    WorkingMemory,
)


class FakeTurn:
    person_id = "p1"
    person_name = "Ada"
    user_text = "I live in Beijing"
    reply = "Noted, Beijing."
    embedding = None
    asr_score = None


def test_working_memory_count_cap():
    w = WorkingMemory(max_items=2, ttl_s=10)
    w.add("user", "a")
    w.add("user", "b")
    w.add("user", "c")
    assert len(w.items()) == 2
    assert [i.text for i in w.items()] == ["b", "c"]


def test_working_memory_ttl_drops_expired():
    w = WorkingMemory(max_items=64, ttl_s=1.0)
    w.add("user", "old", ts=0.0)
    w.add("user", "new", ts=100.0)
    live = w.items(now=100.5)
    assert [i.text for i in live] == ["new"]


def test_semantic_updates_in_place(tmp_path):
    s = SemanticMemory(str(tmp_path / "s.sqlite3"))
    s.remember("name", "Ada")
    assert s.recall("name") == "Ada"
    s.remember("name", "Bob")
    assert s.recall("name") == "Bob"
    assert s.count() == 1  # updated, not appended
    assert s.forget("name") is True
    assert s.recall("name") is None


def test_episodic_append_and_query(tmp_path):
    e = EpisodicMemory(str(tmp_path / "e.jsonl"))
    e.append("turn", {"user": "hi"})
    e.append("turn", {"user": "bye"})
    e.append("note", {"x": 1})
    assert len(e.recent()) == 3
    assert [ep.data["user"] for ep in e.by_kind("turn")] == ["hi", "bye"]


def test_voice_memory_records_and_filters(tmp_path):
    v = VoiceMemory(str(tmp_path / "v.jsonl"))
    v.add("p1", "Ada", "hello", embedding=np.zeros(4, np.float32))
    v.add("p2", "Bob", "hi there")
    assert v.known_people() == ["p1", "p2"]
    assert len(v.utterances_for("p1")) == 1


def test_procedural_seeds_defaults(tmp_path):
    p = ProceduralMemory(str(tmp_path / "p.json"), seed=True)
    assert len(p.all_skills()) >= 1
    assert p.get_skill("barge_in") is not None


def test_manager_observe_and_context_block(tmp_path):
    cfg = MemoryConfig(root=str(tmp_path), context_budget_chars=1200)
    m = MemoryManager(cfg)
    m.observe(FakeTurn())
    ctx = m.context_block("Ada", "Beijing")
    assert len(ctx) <= 1200
    # The turn's actual content must reach the context, not just the identity line
    # (which is always present even if observe() wrote nothing).
    assert "Beijing" in ctx            # extracted fact
    assert "Noted, Beijing." in ctx    # recent exchange (assistant reply)
    assert m.recall_fact("location") == "Beijing"


def test_recent_zero_returns_empty(tmp_path):
    e = EpisodicMemory(str(tmp_path / "e.jsonl"))
    e.append("turn", {"user": "a"})
    e.append("turn", {"user": "b"})
    assert e.recent(0) == []
    assert len(e.recent(2)) == 2
    v = VoiceMemory(str(tmp_path / "v.jsonl"))
    v.add("p1", "Ada", "hi")
    assert v.recent(0) == []
    assert len(v.utterances_for("p1", 0)) == 0


def test_context_block_respects_budget(tmp_path):
    cfg = MemoryConfig(root=str(tmp_path), context_budget_chars=50)
    m = MemoryManager(cfg)
    for i in range(20):
        m.remember_fact(f"k{i}", "x" * 20)
    ctx = m.context_block("Ada", "")
    assert len(ctx) <= 50


def test_context_block_zero_budget(tmp_path):
    cfg = MemoryConfig(root=str(tmp_path), context_budget_chars=0)
    m = MemoryManager(cfg)
    m.remember_fact("a", "b")
    assert m.context_block("Ada", "") == ""


def test_extract_facts_precision(tmp_path):
    m = MemoryManager(MemoryConfig(root=str(tmp_path)))
    assert m.extract_facts("My name is Ada.") == [("name", "Ada")]
    assert m.extract_facts("I work at Acme Corp") == [("employer", "Acme Corp")]
    # a sentence with no explicit pattern yields nothing
    assert m.extract_facts("the weather is nice today") == []


def test_manager_persistence_across_reopen(tmp_path):
    cfg = MemoryConfig(root=str(tmp_path))
    m = MemoryManager(cfg)
    m.remember_fact("city", "Beijing")
    m.episodic.append("note", {"x": 1})
    m.close()

    m2 = MemoryManager(MemoryConfig(root=str(tmp_path)))
    assert m2.recall_fact("city") == "Beijing"
    assert len(m2.episodic.recent()) == 1
