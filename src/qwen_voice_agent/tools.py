"""Tool calling — the part that turns a voice chatbot into a voice agent.

An LLM can only *say* things.  A tool loop lets it *do* things: the model emits
``tool_calls`` instead of prose, the runtime executes them, feeds the results
back as ``role: "tool"`` messages, and asks for a final spoken answer.  That
second round is what makes the reply reflect reality ("it is 10:20") instead of
a guess.

Two rules shape this module:

* **The core stays thin.**  Nothing here imports a model, a network client or a
  heavy package — a tool is a Python callable plus a JSON schema.
* **Failures are data, not crashes.**  A bad argument, an unknown tool name or
  an exception inside a tool becomes a ``ToolResult(ok=False)`` that is handed
  back to the model.  The model can apologise or try another approach; the
  24/7 loop must not die because a tool was called with the wrong key.

The action tools (files, browser, apps, desktop input) are deliberately
constrained rather than generic:

* file access is confined to ``tools.workspace_root`` (``..`` and symlink
  escapes are rejected, and both reads and writes are size-bounded);
* the browser accepts only ``http``/``https`` URLs, never ``file:`` or
  ``javascript:``;
* apps come from an explicit allow-list of executable names, launched as argv
  (never through a shell);
* typing, key presses and screenshots are off unless
  ``tools.control_enabled=true``, because synthesising input into whatever
  window has focus is the most dangerous capability here.

Reminder tools write to the durable :class:`~qwen_voice_agent.scheduler.Scheduler`;
the pipeline, not the tool, is what speaks a due reminder.  There is still **no
shell/exec tool**: a voice agent that can run arbitrary commands is a
remote-code-execution endpoint with a microphone.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import ToolsConfig
from .scheduler import KIND_LABELS, REPEAT_LABELS, Scheduler, format_due, parse_when

__all__ = [
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "ToolError",
    "build_tools",
    "builtin_tools",
]


class ToolError(RuntimeError):
    """Raised when a tool cannot be executed (bad arguments, unknown name)."""


@dataclass(frozen=True)
class ToolCall:
    """One ``tool_calls[]`` entry from an OpenAI-compatible response.

    ``arguments`` is the raw JSON string the model emitted.  Keeping it raw lets
    the executor distinguish "invalid JSON" from "valid JSON, wrong keys", and
    lets a caller log exactly what the model said.
    """

    id: str
    name: str
    arguments: Any = ""


@dataclass
class ToolResult:
    """What a tool returns to the model, and whether it worked."""

    call_id: str
    name: str
    content: str
    ok: bool = True


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke, with the schema it needs to call it.

    ``context_kwarg`` names a key in the per-turn execution context that must be
    injected into the call rather than taken from the model.  The memory tools
    use it for ``person_id``: who is speaking is a fact the runtime knows, and a
    model must not be able to write or read another person's facts by inventing
    an id.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    context_kwarg: Optional[str] = None

    def schema(self) -> Dict[str, Any]:
        """The OpenAI ``tools[]`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """The set of tools offered to the model on every turn."""

    def __init__(self, tools: Sequence[Tool] = (), *, max_rounds: int = 3) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool
        self.max_rounds = max(1, int(max_rounds))

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def enabled(self) -> bool:
        return bool(self._tools)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def execute(
        self, call: ToolCall, context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Run one tool call; every failure becomes a result the model can read.

        ``context`` carries runtime facts (``person_id``, ``person_name``) that a
        tool may need but the model must not control.  It is injected through
        ``Tool.context_kwarg`` after parsing, overriding any value the model
        supplied under the same name.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call.id,
                call.name,
                f"error: unknown tool {call.name!r}; available: {', '.join(self.names())}",
                ok=False,
            )
        try:
            args = self._parse_arguments(call.arguments)
        except ToolError as exc:
            return ToolResult(call.id, call.name, f"error: {exc}", ok=False)
        if tool.context_kwarg:
            args[tool.context_kwarg] = (context or {}).get(tool.context_kwarg)
        try:
            value = tool.func(**args)
        except TypeError as exc:
            # A wrong signature is the most common failure: the model invented a
            # parameter or omitted a required one.  Report it verbatim so the
            # next round can correct itself.
            return ToolResult(call.id, call.name, f"error: bad arguments: {exc}", ok=False)
        except Exception as exc:  # a tool must not take the agent down
            return ToolResult(call.id, call.name, f"error: {type(exc).__name__}: {exc}", ok=False)
        return ToolResult(call.id, call.name, _stringify(value), ok=True)

    @staticmethod
    def _parse_arguments(raw: Any) -> Dict[str, Any]:
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolError(f"arguments are not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ToolError("arguments must be a JSON object")
            return parsed
        raise ToolError(f"unsupported arguments type: {type(raw).__name__}")


def _stringify(value: Any) -> str:
    """Tool output is text; JSON keeps structured results readable for the model."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


# ---- shared helpers -----------------------------------------------------


def _workspace_root(cfg: ToolsConfig) -> str:
    root = os.path.realpath(os.path.abspath(os.path.expanduser(cfg.workspace_root)))
    os.makedirs(root, exist_ok=True)
    return root


def _resolve_path(
    cfg: ToolsConfig,
    raw: Any,
    *,
    must_exist: bool = False,
    want_dir: bool = False,
) -> tuple[str, str]:
    """Resolve a model-supplied path inside ``workspace_root`` or reject it.

    ``realpath`` resolves symlinks before the containment check, so a symlink
    created inside the workspace cannot be used to reach a file outside it.
    Relative paths are relative to the workspace root, not the process CWD.
    """
    if not isinstance(raw, str):
        raise ValueError("path must be a string")
    text = raw.strip() or "."
    if "\x00" in text:
        raise ValueError("path contains a NUL byte")
    root = _workspace_root(cfg)
    target = text if os.path.isabs(text) else os.path.join(root, text)
    target = os.path.realpath(target)
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"path escapes the workspace: {raw!r}")
    if must_exist and not os.path.exists(target):
        raise FileNotFoundError(f"no such file or directory: {raw!r}")
    if want_dir and not os.path.isdir(target):
        raise NotADirectoryError(f"not a directory: {raw!r}")
    return root, target


def _relative(root: str, target: str) -> str:
    rel = os.path.relpath(target, root)
    return "." if rel == "." else rel


def _max_file_bytes(cfg: ToolsConfig) -> int:
    return max(1, int(cfg.max_file_bytes))


def _resolve_executable(value: Any, label: str) -> str:
    """Turn a configured command into one executable path, never a shell line."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is empty")
    if any(ch.isspace() for ch in text) or any(ch in text for ch in ";|&$`"):
        raise ValueError(f"{label} must be a single executable name/path, not a shell command")
    path = text if os.path.isabs(text) else shutil.which(text)
    if not path or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"{label} not found or not executable: {text!r}")
    return path


def _launch(argv: List[str]) -> None:
    """Start a detached GUI process without blocking the generation worker."""
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise RuntimeError(f"cannot launch {argv[0]!r}: {exc}") from exc
    # xdg-open and friends normally return immediately.  Give them a moment so
    # a missing browser is reported as an error instead of a silent no-op.
    time.sleep(0.25)
    status = proc.poll()
    if status is not None and status != 0:
        raise RuntimeError(f"{argv[0]!r} exited with status {status}")


def _run(argv: List[str], timeout: float = 20.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"command not found: {argv[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{argv[0]!r} timed out after {timeout:g}s") from exc


def _require_gui() -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("no graphical session (DISPLAY and WAYLAND_DISPLAY are unset)")


def _normalise_url(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("url is empty")
    if len(text) > 2048:
        raise ValueError("url is too long (limit 2048 characters)")
    if any(ord(ch) < 32 or ch.isspace() for ch in text):
        raise ValueError("url contains whitespace or control characters")
    if "://" not in text:
        # A bare host like "example.com" or "example.com:8080" is fine;
        # "javascript:alert(1)" and "data:..." must not be rewritten into an
        # https host.  Only a known dangerous scheme is rejected here — a
        # generic "scheme:" check would misread host:port as a scheme.
        if re.match(
            r"^(?:javascript|data|vbscript|file|ftp|mailto|tel|about|chrome|blob):",
            text,
            re.IGNORECASE,
        ):
            raise ValueError("only http:// and https:// URLs are allowed")
        text = "https://" + text
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("only http:// and https:// URLs are allowed")
    if not parsed.netloc:
        raise ValueError("url has no host")
    return text


# ---- built-in tools -----------------------------------------------------
#
# Deliberately small and constrained.  There is no shell tool: a voice agent
# that can run arbitrary commands is a remote-code-execution endpoint with a
# microphone.  The file tools are sandboxed to `workspace_root`; the desktop
# tools are opt-in.


def _time_tool() -> Tool:
    def get_current_time() -> str:
        now = time.localtime()
        weekdays = "一二三四五六日"
        return time.strftime("%Y-%m-%d %H:%M:%S", now) + f"（星期{weekdays[now.tm_wday]}）"

    return Tool(
        name="get_current_time",
        description="返回当前本地日期和时间。用户问“现在几点”“今天几号”时调用。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=get_current_time,
    )


def _remember_tool(memory) -> Tool:
    def remember_fact(key: str, value: str, person_id: Optional[str] = None) -> str:
        key, value = str(key).strip(), str(value).strip()
        if not key or not value:
            raise ValueError("key and value must both be non-empty")
        memory.remember_fact(key, value, person_id=person_id)
        return f"已记住：{key} = {value}"

    return Tool(
        name="remember_fact",
        description="把一条长期事实写入语义记忆（例如用户的名字、住址、偏好）。用户明确要求“记住”时调用。",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "事实的键，如 name、location、favorite_food"},
                "value": {"type": "string", "description": "事实的值"},
            },
            "required": ["key", "value"],
        },
        func=remember_fact,
        context_kwarg="person_id",
    )


def _recall_tool(memory) -> Tool:
    def recall_fact(key: str, person_id: Optional[str] = None) -> str:
        value = memory.recall_fact(str(key).strip(), person_id=person_id)
        if value is None:
            return f"没有关于 {key} 的记录。"
        return value

    return Tool(
        name="recall_fact",
        description="从语义记忆里按 key 读取一条长期事实。用户问“我叫什么”“我住哪”时调用。",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "要查询的事实键，如 name、location"},
            },
            "required": ["key"],
        },
        func=recall_fact,
        context_kwarg="person_id",
    )


def _write_file_tool(cfg: ToolsConfig) -> Tool:
    def write_file(path: str, content: str) -> str:
        data = str(content).encode("utf-8")
        limit = _max_file_bytes(cfg)
        if len(data) > limit:
            raise ValueError(f"content is {len(data)} bytes; the limit is {limit} bytes")
        root, target = _resolve_path(cfg, path)
        if os.path.isdir(target):
            raise IsADirectoryError(f"is a directory: {path!r}")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # Write-then-rename: a crash mid-write leaves the old file intact rather
        # than a truncated note the user cannot recover.
        tmp = f"{target}.tmp-{os.getpid()}-{time.time_ns()}"
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return f"已写入 {_relative(root, target)}（{len(data)} 字节）"

    return Tool(
        name="write_file",
        description=(
            "把文本写入工作区内的文件，路径相对工作区根目录；子目录会自动创建。"
            "用户要求写文件、记笔记、保存内容时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，如 notes/today.txt"},
                "content": {"type": "string", "description": "要写入的完整文本"},
            },
            "required": ["path", "content"],
        },
        func=write_file,
    )


def _read_file_tool(cfg: ToolsConfig) -> Tool:
    def read_file(path: str) -> str:
        root, target = _resolve_path(cfg, path, must_exist=True)
        if os.path.isdir(target):
            raise IsADirectoryError(f"is a directory: {path!r}")
        limit = _max_file_bytes(cfg)
        with open(target, "rb") as fh:
            data = fh.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"file is larger than {limit} bytes")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("file is not UTF-8 text") from exc
        return text if text else "（空文件）"

    return Tool(
        name="read_file",
        description="读取工作区内一个 UTF-8 文本文件。用户要求查看、读取或总结某个文件时调用。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，如 notes/today.txt"},
            },
            "required": ["path"],
        },
        func=read_file,
    )


def _list_files_tool(cfg: ToolsConfig) -> Tool:
    def list_files(path: str = ".") -> str:
        root, target = _resolve_path(cfg, path, must_exist=True, want_dir=True)
        entries = sorted(os.scandir(target), key=lambda entry: entry.name)
        lines: List[str] = []
        for entry in entries[:200]:
            try:
                if entry.is_dir(follow_symlinks=False):
                    lines.append(entry.name + "/")
                elif entry.is_symlink():
                    lines.append(entry.name + " [link]")
                else:
                    size = entry.stat(follow_symlinks=False).st_size
                    lines.append(f"{entry.name} ({size} B)")
            except OSError:
                lines.append(entry.name + " [unreadable]")
        if not lines:
            return f"{_relative(root, target)} 为空"
        more = f"\n… 共 {len(entries)} 项，只显示前 200 项" if len(entries) > 200 else ""
        return f"{_relative(root, target)}:\n" + "\n".join(lines) + more

    return Tool(
        name="list_files",
        description="列出工作区内某个目录的文件和子目录。用户问“有哪些文件”时调用。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径，默认工作区根目录"},
            },
            "required": [],
        },
        func=list_files,
    )


def _open_browser_tool(cfg: ToolsConfig) -> Tool:
    def open_browser(url: str = "", query: str = "") -> str:
        if url and query:
            raise ValueError("pass either url or query, not both")
        if query:
            query_text = str(query).strip()
            if not query_text:
                raise ValueError("query is empty")
            if len(query_text) > 500:
                raise ValueError("query is too long (limit 500 characters)")
            target = "https://www.baidu.com/s?wd=" + urllib.parse.quote(query_text)
        elif url:
            target = _normalise_url(url)
        else:
            raise ValueError("url or query is required")
        executable = _resolve_executable(cfg.browser_command, "browser_command")
        _launch([executable, target])
        return f"已在浏览器打开：{target}"

    return Tool(
        name="open_browser",
        description=(
            "在浏览器中打开网页。参数 url 与 query 二选一：url 是完整 http/https 地址；"
            "query 是搜索关键词，会用百度搜索。用户要求打开网页或搜索时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "完整网址，如 https://example.com"},
                "query": {"type": "string", "description": "搜索关键词，如 北京天气"},
            },
            "required": [],
        },
        func=open_browser,
    )


def _open_app_tool(cfg: ToolsConfig) -> Tool:
    def open_app(name: str) -> str:
        key = str(name or "").strip().lower()
        apps = {str(k).lower(): str(v) for k, v in cfg.allowed_apps.items()}
        if key not in apps:
            raise ValueError(f"unknown app {name!r}; allowed: {', '.join(sorted(apps))}")
        executable = _resolve_executable(apps[key], f"allowed_apps[{key!r}]")
        _launch([executable])
        return f"已打开 {key}"

    return Tool(
        name="open_app",
        description=(
            "打开允许列表中的一个桌面应用（chrome、firefox、files、terminal）。"
            "用户要求打开某个应用时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "应用名，如 chrome、terminal"},
            },
            "required": ["name"],
        },
        func=open_app,
    )


def _input_backend() -> str:
    if shutil.which("xdotool"):
        return "xdotool"
    if shutil.which("wtype"):
        return "wtype"
    raise RuntimeError(
        "no desktop input backend: install xdotool (X11/XWayland) or wtype (Wayland)"
    )


# X11 keysym names are case-sensitive: `xdotool key return` prints "No such key
# name" but still exits 0.  The model usually writes lowercase, so canonicalise
# the common names before sending them.
_KEYSYM_ALIASES = {
    "return": "Return",
    "enter": "Return",
    "escape": "Escape",
    "esc": "Escape",
    "tab": "Tab",
    "backspace": "BackSpace",
    "delete": "Delete",
    "del": "Delete",
    "space": "space",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "home": "Home",
    "end": "End",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "insert": "Insert",
}


def _normalise_key(value: str) -> str:
    return "+".join(_KEYSYM_ALIASES.get(part.lower(), part) for part in value.split("+"))


def _xdotool_window(pattern: str) -> Optional[str]:
    text = str(pattern or "").strip()
    if not text:
        return None
    if len(text) > 100 or "\x00" in text:
        raise ValueError("window pattern is invalid")
    proc = _run(["xdotool", "search", "--onlyvisible", "--name", text], timeout=5)
    ids = [line.strip() for line in proc.stdout.splitlines() if line.strip().isdigit()]
    if not ids:
        raise RuntimeError(f"no visible window matches {pattern!r}")
    # `search --name` is a regex and can return helper windows too.  Prefer an
    # exact title match, which is what the model means by "the window called X".
    for window_id in ids[:10]:
        try:
            name = _run(["xdotool", "getwindowname", window_id], timeout=2).stdout.strip()
        except RuntimeError:
            continue
        if name == text:
            return window_id
    return ids[0]


def _focus_window(window_id: str) -> bool:
    """Best-effort focus; return True if X input focus is now on the window.

    ``windowactivate`` asks the window manager, which needs EWMH and can hang
    under some Wayland/XWayland combinations, so it is bounded and not ``--sync``.
    ``windowfocus`` sets the X input focus directly and works even without a WM.
    The caller falls back to ``type --window`` (XSendEvent) only if both fail.
    """
    for argv in (
        ["xdotool", "windowactivate", window_id],
        ["xdotool", "windowfocus", "--sync", window_id],
    ):
        try:
            proc = _run(argv, timeout=2)
        except RuntimeError:
            continue
        if proc.returncode == 0:
            return True
    return False


def _type_text_tool(cfg: ToolsConfig) -> Tool:
    def type_text(text: str, window: str = "") -> str:
        _require_gui()
        value = str(text or "")
        if not value:
            raise ValueError("text is empty")
        if len(value) > 4000:
            raise ValueError("text is too long (limit 4000 characters)")
        backend = _input_backend()
        if backend == "xdotool":
            window_id = _xdotool_window(window) if window else None
            if window_id and not _focus_window(window_id):
                # No WM/focus path: XSendEvent is less compatible (some toolkits
                # ignore synthetic events), but it is the only remaining option.
                argv = [
                    "xdotool", "type", "--window", window_id,
                    "--clearmodifiers", "--delay", "12", "--", value,
                ]
            else:
                argv = ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", value]
        elif backend == "wtype":
            argv = ["wtype", "--", value]
        else:  # pragma: no cover - _input_backend only returns the two above
            raise RuntimeError(f"unsupported input backend: {backend}")
        proc = _run(argv, timeout=20)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"{backend} type failed ({proc.returncode})")
        suffix = f"（窗口 {window!r}）" if window else ""
        return f"已输入 {len(value)} 个字符{suffix}"

    return Tool(
        name="type_text",
        description=(
            "把文字输入到当前焦点窗口，或标题匹配 window 的窗口。"
            "需要 tools.control_enabled=true。用户要求在电脑上打字、填写内容时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要输入的文字"},
                "window": {"type": "string", "description": "可选，窗口标题的一部分"},
            },
            "required": ["text"],
        },
        func=type_text,
    )


def _press_key_tool(cfg: ToolsConfig) -> Tool:
    def press_key(key: str, window: str = "") -> str:
        _require_gui()
        value = str(key or "").strip()
        if not value or len(value) > 64 or not re.fullmatch(r"[A-Za-z0-9_+]+", value):
            raise ValueError("key must be a keysym such as Return, ctrl+l or alt+Tab")
        value = _normalise_key(value)
        backend = _input_backend()
        if backend == "xdotool":
            window_id = _xdotool_window(window) if window else None
            argv = ["xdotool", "key"]
            if window_id and not _focus_window(window_id):
                argv += ["--window", window_id]
            argv += ["--clearmodifiers", value]
        elif backend == "wtype":
            argv = ["wtype", "-k", value]
        else:  # pragma: no cover - _input_backend only returns the two above
            raise RuntimeError(f"unsupported input backend: {backend}")
        proc = _run(argv, timeout=10)
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0 or "No such key name" in stderr:
            raise RuntimeError(stderr or f"{backend} key failed ({proc.returncode})")
        suffix = f"（窗口 {window!r}）" if window else ""
        return f"已按键 {value}{suffix}"

    return Tool(
        name="press_key",
        description=(
            "向当前焦点窗口，或标题匹配 window 的窗口发送按键，如 Return、ctrl+l、alt+Tab。"
            "需要 tools.control_enabled=true。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "按键名，如 Return、ctrl+l"},
                "window": {"type": "string", "description": "可选，窗口标题的一部分"},
            },
            "required": ["key"],
        },
        func=press_key,
    )


def _screenshot_argv(target: str) -> List[str]:
    """Pick a screenshot backend that can actually see this session.

    ``scrot`` only works on a real X11 session; on a Wayland session it captures
    the black X root window, so it is deliberately skipped when
    ``WAYLAND_DISPLAY`` is set.
    """
    if shutil.which("gnome-screenshot"):
        return ["gnome-screenshot", "-f", target]
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("grim"):
        return ["grim", target]
    if shutil.which("spectacle"):
        return ["spectacle", "-b", "-n", "-o", target]
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") and shutil.which("scrot"):
        return ["scrot", "-o", target]
    if shutil.which("import"):
        return ["import", "-window", "root", target]
    raise RuntimeError(
        "no screenshot backend: install gnome-screenshot (GNOME/Wayland) or grim (wlroots)"
    )


def _screenshot_tool(cfg: ToolsConfig) -> Tool:
    def screenshot(path: str = "") -> str:
        _require_gui()
        value = str(path or "").strip()
        if not value:
            value = f"screenshot-{time.strftime('%Y%m%d-%H%M%S')}.png"
        if not os.path.splitext(value)[1]:
            value += ".png"
        root, target = _resolve_path(cfg, value)
        if os.path.isdir(target):
            raise IsADirectoryError(f"is a directory: {value!r}")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        proc = _run(_screenshot_argv(target), timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"screenshot failed ({proc.returncode})")
        if not os.path.exists(target):
            raise RuntimeError("screenshot command produced no file")
        size = os.path.getsize(target)
        if size == 0:
            raise RuntimeError("screenshot is empty")
        return f"已截图：{_relative(root, target)}（{size} 字节）"

    return Tool(
        name="screenshot",
        description=(
            "截取当前屏幕并保存为工作区内的图片。"
            "需要 tools.control_enabled=true。用户要求截图时调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "可选，保存路径，如 shots/screen.png"},
            },
            "required": [],
        },
        func=screenshot,
    )


def _set_reminder_tool(cfg: ToolsConfig, scheduler: Scheduler) -> Tool:
    def set_reminder(text: str, when: str, repeat: str = "once") -> str:
        value = str(text or "").strip()
        when_text = str(when or "").strip()
        if not value:
            raise ValueError("reminder text is empty")
        repeat_value = str(repeat or "once").strip().lower()
        if repeat_value not in REPEAT_LABELS:
            raise ValueError(f"unknown repeat {repeat_value!r}; use {', '.join(REPEAT_LABELS)}")
        if repeat_value == "once":
            if when_text.startswith(("每天", "每日", "每晚")):
                repeat_value = "daily"
            elif when_text.startswith(("每周", "每星期", "每礼拜")):
                repeat_value = "weekly"
        due = parse_when(when_text)
        schedule = scheduler.add(value, due, repeat_value)
        return (
            f"已设置提醒 {schedule.id}：{format_due(schedule.due_ts)}"
            f"（{REPEAT_LABELS[repeat_value]}）{schedule.text}"
        )

    return Tool(
        name="set_reminder",
        description=(
            "设置一次性或重复提醒，到点直接播报给定文字。**只用于“到点提醒我一句话”**；"
            "如果用户要求到点执行任务、查询或播报结果（如天气、总结），必须用 "
            "schedule_task，不要用本工具。when 支持“10分钟后”“明天早上8点”“每天8点”"
            "“周一9点”“2026-10-20 09:00”；repeat 为 once/daily/weekdays/weekly。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "提醒的内容"},
                "when": {"type": "string", "description": "提醒时间，如 10分钟后、明天早上8点"},
                "repeat": {
                    "type": "string",
                    "enum": list(REPEAT_LABELS),
                    "description": "重复方式，默认 once",
                },
            },
            "required": ["text", "when"],
        },
        func=set_reminder,
    )


def _schedule_task_tool(cfg: ToolsConfig, scheduler: Scheduler) -> Tool:
    def schedule_task(prompt: str, when: str, repeat: str = "once") -> str:
        value = str(prompt or "").strip()
        when_text = str(when or "").strip()
        if not value:
            raise ValueError("task prompt is empty")
        repeat_value = str(repeat or "once").strip().lower()
        if repeat_value not in REPEAT_LABELS:
            raise ValueError(f"unknown repeat {repeat_value!r}; use {', '.join(REPEAT_LABELS)}")
        if repeat_value == "once":
            if when_text.startswith(("每天", "每日", "每晚")):
                repeat_value = "daily"
            elif when_text.startswith(("每周", "每星期", "每礼拜")):
                repeat_value = "weekly"
        due = parse_when(when_text)
        schedule = scheduler.add(value, due, repeat_value, kind="task")
        return (
            f"已设置定时任务 {schedule.id}：{format_due(schedule.due_ts)}"
            f"（{REPEAT_LABELS[repeat_value]}）{schedule.text}"
        )

    return Tool(
        name="schedule_task",
        description=(
            "在指定时间让助手执行一个任务，并用一句话播报结果。**当用户要求到点"
            "做事（如“每天早上8点播报天气”“每天9点总结昨天的对话”“每晚10点检查"
            "文件”）时，必须用本工具，不要用 set_reminder。** when/repeat 与 "
            "set_reminder 相同。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "到点后要执行的任务描述"},
                "when": {"type": "string", "description": "执行时间，如 每天8点、明天早上8点"},
                "repeat": {
                    "type": "string",
                    "enum": list(REPEAT_LABELS),
                    "description": "重复方式，默认 once",
                },
            },
            "required": ["prompt", "when"],
        },
        func=schedule_task,
    )


def _list_reminders_tool(scheduler: Scheduler) -> Tool:
    def list_reminders() -> str:
        items = scheduler.list()
        if not items:
            return "没有定时任务。"
        lines = [
            f"{s.id} | {KIND_LABELS.get(s.kind, s.kind)} | {format_due(s.due_ts)} | "
            f"{REPEAT_LABELS.get(s.repeat, s.repeat)} | {s.text}"
            for s in items
        ]
        return "\n".join(lines)

    return Tool(
        name="list_reminders",
        description="列出所有定时提醒及其 id。用户问“有哪些提醒/定时任务”时调用。",
        parameters={"type": "object", "properties": {}, "required": []},
        func=list_reminders,
    )


def _cancel_reminder_tool(scheduler: Scheduler) -> Tool:
    def cancel_reminder(reminder_id: str) -> str:
        key = str(reminder_id or "").strip()
        if not key:
            raise ValueError("reminder_id is empty")
        if scheduler.cancel(key):
            return f"已取消提醒 {key}"
        return f"没有找到提醒 {key}"

    return Tool(
        name="cancel_reminder",
        description="按 id 取消一个定时提醒。用户要求取消提醒时调用。",
        parameters={
            "type": "object",
            "properties": {
                "reminder_id": {"type": "string", "description": "提醒的 id，如 a1b2c3d4"},
            },
            "required": ["reminder_id"],
        },
        func=cancel_reminder,
    )


def builtin_tools(
    memory, cfg: ToolsConfig, scheduler: Optional[Scheduler] = None
) -> Dict[str, Tool]:
    """The tools a :class:`ToolsConfig` can name, keyed by their short name."""
    tools = {
        "time": _time_tool(),
        "remember_fact": _remember_tool(memory),
        "recall_fact": _recall_tool(memory),
        "write_file": _write_file_tool(cfg),
        "read_file": _read_file_tool(cfg),
        "list_files": _list_files_tool(cfg),
        "open_browser": _open_browser_tool(cfg),
        "open_app": _open_app_tool(cfg),
    }
    if scheduler is not None:
        tools.update(
            {
                "set_reminder": _set_reminder_tool(cfg, scheduler),
                "schedule_task": _schedule_task_tool(cfg, scheduler),
                "list_reminders": _list_reminders_tool(scheduler),
                "cancel_reminder": _cancel_reminder_tool(scheduler),
            }
        )
    if cfg.control_enabled:
        tools.update(
            {
                "type_text": _type_text_tool(cfg),
                "press_key": _press_key_tool(cfg),
                "screenshot": _screenshot_tool(cfg),
            }
        )
    return tools


def build_tools(
    memory, cfg: ToolsConfig, scheduler: Optional[Scheduler] = None
) -> Optional[ToolRegistry]:
    """Build the registry named by ``cfg``; ``None`` when tools are off or empty.

    ``memory`` is the :class:`~qwen_voice_agent.memory.MemoryManager` the tools
    act on.  Passing it in (rather than importing a global) keeps the registry
    testable with a temporary memory root.  ``scheduler`` is the same instance
    the pipeline polls, so a reminder set in one turn is spoken in a later frame.
    """
    if not cfg.enabled:
        return None
    available = builtin_tools(memory, cfg, scheduler)
    selected: List[Tool] = []
    for name in cfg.builtins:
        tool = available.get(name)
        if tool is None:
            if name in {"set_reminder", "schedule_task", "list_reminders", "cancel_reminder"}:
                raise ValueError(
                    f"unknown built-in tool: {name!r} requires scheduler.enabled=true"
                )
            if name in {"type_text", "press_key", "screenshot"}:
                raise ValueError(
                    f"unknown built-in tool: {name!r} requires tools.control_enabled=true"
                )
            raise ValueError(f"unknown built-in tool: {name!r} (available: {', '.join(available)})")
        selected.append(tool)
    return ToolRegistry(selected, max_rounds=cfg.max_rounds) if selected else None
