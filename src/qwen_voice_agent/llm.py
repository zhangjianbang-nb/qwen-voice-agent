"""LLM backends.

Three implementations behind one protocol.  ``http`` reaches any OpenAI-compatible
``/v1/chat/completions`` server (the production path for a served Qwen model);
``local`` loads a ``transformers`` checkpoint for an on-box model; ``mock`` is a
deterministic stub so the turn loop is testable without any model at all.

The system prompt is a *template* with ``{context}`` and ``{person}`` placeholders,
rendered per turn by the memory manager — the agent's persona is configuration,
not a source edit.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol

from .config import LLMConfig

__all__ = ["LlmBackend", "LlmError", "HttpLlm", "LocalLlm", "MockLlm", "build_llm", "render_system_prompt"]


class LlmError(RuntimeError):
    """Raised when an LLM backend cannot produce a reply."""


class LlmBackend(Protocol):
    def chat(self, messages: List[dict]) -> str:  # pragma: no cover - protocol
        ...


def render_system_prompt(cfg: LLMConfig, person: str, context: str) -> str:
    """Render the persona template with the current person and memory context."""
    return cfg.system_prompt.format(person=person, context=context)


class HttpLlm:
    """OpenAI-compatible ``POST {base_url}/chat/completions``."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    def chat(self, messages: List[dict]) -> str:
        try:
            import httpx  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise LlmError("HttpLlm needs httpx installed") from exc
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        try:
            with httpx.Client(timeout=self.cfg.timeout_s) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LlmError(f"LLM request to {url} failed: {exc}") from exc
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"LLM response shape unexpected: {data!r}") from exc


class LocalLlm:
    """Local generation through ``transformers`` (CPU is fine, just slower)."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            try:
                from transformers import pipeline  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise LlmError("LocalLlm needs the 'transformers' package") from exc
            self._pipe = pipeline("text-generation", model=self.cfg.model)
        return self._pipe

    def chat(self, messages: List[dict]) -> str:
        pipe = self._ensure()
        # A chat-shaped prompt assembled from the messages; keeps the backend
        # dependency-free of any model-specific chat template.
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        prompt += "\nassistant:"
        out = pipe(
            prompt,
            max_new_tokens=self.cfg.max_tokens,
            temperature=max(1e-5, self.cfg.temperature),
            do_sample=self.cfg.temperature > 0,
        )
        text = out[0]["generated_text"]
        return text.split("assistant:")[-1].strip()


class MockLlm:
    """Deterministic reply, stable for a given last user message.

    Short, single-sentence, no markdown — the same shape a voice assistant should
    produce, so pipeline tests exercise realistic text.
    """

    def chat(self, messages: List[dict]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        digest = hashlib.sha1(last_user.encode("utf-8")).hexdigest()[:6]
        if not last_user:
            return "I am listening."
        return f"I heard you say: {last_user[:60]} ({digest})."


def build_llm(cfg: LLMConfig) -> LlmBackend:
    """Instantiate the LLM backend named by ``cfg.backend``."""
    if cfg.backend == "http":
        return HttpLlm(cfg)
    if cfg.backend == "local":
        return LocalLlm(cfg)
    return MockLlm()
