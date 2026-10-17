"""Procedural memory — how the agent should behave.

Unlike the other stores, this one is not about the *user*: it holds the agent's
own skills and house rules ("lead with the answer", "never use markdown out loud",
"if the user is upset, acknowledge first").  It is a JSON file so a deployment can
edit the agent's behaviour without touching code.

``seed_skills`` installs a small default set on first run; it never overwrites a
skill that already exists, so operator edits survive restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["Skill", "ProceduralMemory"]

_DEFAULT_SKILLS: Dict[str, Dict[str, object]] = {
    "spoken_style": {
        "description": "Speak like a person, not a document.",
        "steps": [
            "Answer in one or two short sentences.",
            "Never read markdown, bullet lists, or code out loud.",
            "Spell out numbers and units the way they are said.",
        ],
    },
    "lead_with_answer": {
        "description": "Give the answer first, context only if asked.",
        "steps": [
            "State the conclusion in the first sentence.",
            "Offer to elaborate rather than elaborating unprompted.",
        ],
    },
    "acknowledge_emotion": {
        "description": "If the speaker sounds upset, acknowledge before answering.",
        "steps": [
            "Name the feeling briefly.",
            "Then address the request.",
        ],
    },
    "barge_in": {
        "description": "The user always outranks the agent's own voice.",
        "steps": [
            "Stop speaking immediately when the user starts.",
            "Do not resume the interrupted sentence unless asked.",
        ],
    },
}


@dataclass
class Skill:
    name: str
    description: str
    steps: List[str] = field(default_factory=list)


class ProceduralMemory:
    """JSON-backed skill store."""

    def __init__(self, path: str, seed: bool = True) -> None:
        self.path = path
        self._skills: Dict[str, Skill] = {}
        if path and os.path.exists(path):
            self._load()
        elif seed:
            self.seed_defaults()
            if path:
                self.save()

    def add_skill(self, name: str, description: str, steps: Optional[List[str]] = None) -> Skill:
        skill = Skill(name=name, description=description, steps=list(steps or []))
        self._skills[name] = skill
        return skill

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def all_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def remove_skill(self, name: str) -> bool:
        return self._skills.pop(name, None) is not None

    def seed_defaults(self) -> None:
        """Install the default skill set without clobbering existing skills."""
        for name, spec in _DEFAULT_SKILLS.items():
            if name not in self._skills:
                self.add_skill(name, str(spec["description"]), list(spec["steps"]))

    # ---- persistence ---------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "skills": {
                name: {"description": s.description, "steps": s.steps}
                for name, s in self._skills.items()
            },
        }

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for name, spec in data.get("skills", {}).items():
            self._skills[name] = Skill(
                name=name,
                description=spec.get("description", ""),
                steps=list(spec.get("steps", [])),
            )
