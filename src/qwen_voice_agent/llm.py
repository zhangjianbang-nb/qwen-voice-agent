"""LLM backends.

Three implementations behind one protocol.  ``http`` reaches any OpenAI-compatible
``/v1/chat/completions`` server (the production path for a served Qwen model);
``local`` loads a ``transformers`` checkpoint for an on-box model; ``mock`` is a
deterministic stub so the turn loop is testable without any model at all.

The system prompt is a *template* with ``{context}`` and ``{person}`` placeholders,
rendered per turn by the memory manager — the agent's persona is configuration,
not a source edit.

Tool calling is optional and layered on top: :meth:`HttpLlm.chat_with_tools`
sends an OpenAI ``tools[]`` array and parses ``tool_calls``; backends that do not
support it are still usable through :func:`chat_with_tools`, which falls back to
plain :meth:`chat`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .config import LLMConfig
from .tools import ToolCall

__all__ = [
    "LlmBackend",
    "LlmError",
    "LlmReply",
    "HttpLlm",
    "LocalLlm",
    "MockLlm",
    "ScriptedLlm",
    "build_llm",
    "chat_with_tools",
    "parse_reply",
    "render_system_prompt",
]


class LlmError(RuntimeError):
    """Raised when an LLM backend cannot produce a reply."""


@dataclass
class LlmReply:
    """One model response: spoken text, and any tools it wants executed."""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LlmBackend(Protocol):
    def chat(self, messages: List[dict]) -> str:  # pragma: no cover - protocol
        ...


def render_system_prompt(cfg: LLMConfig, person: str, context: str) -> str:
    """Render the persona template with the current person and memory context."""
    return cfg.system_prompt.format(person=person, context=context)


def parse_reply(message: Dict[str, Any]) -> LlmReply:
    """Parse an OpenAI ``choices[0].message`` into a :class:`LlmReply`.

    ``content`` may be ``null`` when the model emits only tool calls; a missing
    ``function.name`` is a malformed response, not an empty answer, so it raises
    rather than silently producing silence.
    """
    content = message.get("content")
    content = "" if content is None else str(content).strip()
    calls: List[ToolCall] = []
    for i, raw in enumerate(message.get("tool_calls") or []):
        fn = raw.get("function") or {}
        name = fn.get("name")
        if not name:
            raise LlmError(f"tool call without a function name: {raw!r}")
        calls.append(
            ToolCall(
                id=str(raw.get("id") or f"call_{i}"),
                name=str(name),
                arguments=fn.get("arguments", ""),
            )
        )
    return LlmReply(content, calls)


def chat_with_tools(llm, messages: List[dict], tools: List[dict]) -> LlmReply:
    """Call ``llm`` with tools when it supports them; otherwise plain ``chat``.

    This is the compatibility seam: ``mock`` and ``local`` backends predate tool
    calling and only implement :meth:`chat`, and a test double may do the same.
    The pipeline therefore never assumes every backend speaks ``tool_calls``.
    """
    fn = getattr(llm, "chat_with_tools", None)
    if callable(fn):
        return fn(messages, tools)
    return LlmReply(str(llm.chat(messages)).strip(), [])


class HttpLlm:
    """OpenAI-compatible ``POST {base_url}/chat/completions``."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = None

    def _http(self):
        """One client per process: opening a fresh client every turn pays a TCP
        handshake and TLS setup per turn, which is pure added latency in a loop
        that runs every few seconds."""
        if self._client is None:
            try:
                import httpx  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise LlmError("HttpLlm needs httpx installed") from exc
            self._client = httpx.Client(timeout=self.cfg.timeout_s)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _post(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Dict[str, Any]:
        client = self._http()
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
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise LlmError(f"LLM request to {url} failed: {exc}") from exc

    def chat(self, messages: List[dict]) -> str:
        data = self._post(messages)
        try:
            return parse_reply(data["choices"][0]["message"]).content
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"LLM response shape unexpected: {data!r}") from exc

    def chat_with_tools(self, messages: List[dict], tools: List[dict]) -> LlmReply:
        data = self._post(messages, tools=tools)
        try:
            return parse_reply(data["choices"][0]["message"])
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


class ScriptedLlm:
    """Plays a fixed script — the deterministic stand-in for tool-call tests.

    Each script item is either a string (a plain reply) or a dict with
    ``content`` and/or ``tool_calls``::

        [{"tool_calls": [{"name": "get_current_time", "arguments": "{}"}]},
         {"content": "现在是十点二十分。"}]

    The last item repeats once the script is exhausted, which keeps a test from
    hanging if the pipeline asks for one more round than expected.
    """

    def __init__(self, script: List[Any]) -> None:
        if not script:
            raise ValueError("ScriptedLlm needs at least one script item")
        self.script = list(script)
        self._index = 0

    def _next(self) -> Any:
        item = self.script[min(self._index, len(self.script) - 1)]
        self._index += 1
        return item

    def chat(self, messages: List[dict]) -> str:
        item = self._next()
        if isinstance(item, str):
            return item
        return str(item.get("content") or "")

    def chat_with_tools(self, messages: List[dict], tools: List[dict]) -> LlmReply:
        item = self._next()
        if isinstance(item, str):
            return LlmReply(item, [])
        calls = []
        for i, raw in enumerate(item.get("tool_calls") or []):
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call_{i}"),
                    name=str(raw["name"]),
                    arguments=raw.get("arguments", ""),
                )
            )
        return LlmReply(str(item.get("content") or ""), calls)


def build_llm(cfg: LLMConfig) -> LlmBackend:
    """Instantiate the LLM backend named by ``cfg.backend``."""
    if cfg.backend == "http":
        return HttpLlm(cfg)
    if cfg.backend == "local":
        return LocalLlm(cfg)
    return MockLlm()
