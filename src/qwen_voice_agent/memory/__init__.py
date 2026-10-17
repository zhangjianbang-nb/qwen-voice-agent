"""The five memory stores and the manager that decides what reaches the prompt.

The stores are deliberately separate because they answer different questions:

* **voice** — *who* said *what*, when (speaker identity carried across sessions);
* **working** — the last few turns, with a TTL, so "that" and "it" still resolve;
* **semantic** — durable facts ("the user lives in Beijing");
* **episodic** — an append-only event log ("at 14:03 Ada asked about the weather");
* **procedural** — how to behave ("lead with the answer").

:class:`MemoryManager` owns all five and is the only object the runtime talks to;
it decides what fits in the per-turn context block under a character budget.
"""

from .episodic import EpisodicMemory
from .manager import MemoryManager
from .procedural import ProceduralMemory
from .semantic import SemanticMemory
from .voice import VoiceMemory, VoiceUtterance
from .working import WorkingItem, WorkingMemory

__all__ = [
    "VoiceMemory",
    "VoiceUtterance",
    "WorkingMemory",
    "WorkingItem",
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
    "MemoryManager",
]
