"""Tool calling: registry semantics, built-ins, and the pipeline's tool loop."""

import json
import os
import time

import numpy as np
import pytest

from qwen_voice_agent.asr import MockAsr
from qwen_voice_agent.audio import AudioFrame
from qwen_voice_agent.config import SchedulerConfig, ToolsConfig, VoiceAgentConfig
from qwen_voice_agent.llm import ScriptedLlm, parse_reply
from qwen_voice_agent.memory import MemoryManager
from qwen_voice_agent.pipeline import Pipeline, PipelineState
from qwen_voice_agent.scheduler import Scheduler
from qwen_voice_agent.sources import NullSink
from qwen_voice_agent.tools import Tool, ToolCall, ToolRegistry, build_tools
from qwen_voice_agent.vad import EnergyVad, Segmenter

SR = 16000
FRAME = 320


class FastTts:
    def synthesize(self, text: str) -> AudioFrame:
        return AudioFrame(np.zeros(1600, dtype=np.float32), SR)


class SlowTts:
    """Keeps the proactive worker alive long enough to test barge-in."""

    def __init__(self, delay: float = 1.0) -> None:
        self.delay = delay

    def synthesize(self, text: str) -> AudioFrame:
        time.sleep(self.delay)
        return AudioFrame(np.zeros(1600, dtype=np.float32), SR)


def _tone():
    t = np.arange(FRAME, dtype=np.float32) / SR
    return AudioFrame((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)


def _silence():
    return AudioFrame(np.zeros(FRAME, dtype=np.float32), SR)


def _memory(tmp_path):
    return MemoryManager(VoiceAgentConfig().memory, root=str(tmp_path))


def _pipeline(cfg, llm, memory, tools, tmp_path, scheduler=None):
    return Pipeline(
        cfg,
        vad=EnergyVad(cfg.vad),
        segmenter=Segmenter(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_samples),
        asr=MockAsr(),
        tts=FastTts(),
        llm=llm,
        memory=memory,
        sink=NullSink(),
        tools=tools,
        scheduler=scheduler,
    )


def _run_one_turn(p):
    turns = []
    for _ in range(5):
        turns += p.feed(_silence())
    for _ in range(20):
        turns += p.feed(_tone())
    for _ in range(10):
        turns += p.feed(_silence())
    for _ in range(300):
        if turns:
            break
        turns += p.feed(_silence())
        time.sleep(0.005)
    return turns


def _run_silence_until_turn(p, max_frames=400):
    """Drive the pipeline with silence only, for a proactive reminder."""
    turns = []
    for _ in range(max_frames):
        turns += p.feed(_silence())
        if turns:
            return turns
        time.sleep(0.005)
    return turns


def _feed_until_worker(p, max_frames=400):
    """Feed silence until a generation worker starts; return True on success."""
    for _ in range(max_frames):
        p.feed(_silence())
        if any(t.is_alive() for t in p._workers):
            return True
        time.sleep(0.005)
    return False


def _fake_command(bin_dir, name, body):
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


# ---- registry -----------------------------------------------------------


def test_registry_schema_and_execute():
    tool = Tool(
        name="echo",
        description="echo the argument",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        func=lambda x: x,
    )
    reg = ToolRegistry([tool], max_rounds=2)
    assert reg.enabled and len(reg) == 1
    schema = reg.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"

    result = reg.execute(ToolCall("c1", "echo", '{"x": "hi"}'))
    assert result.ok
    assert result.content == "hi"
    assert result.call_id == "c1"


def test_registry_failures_are_data():
    reg = ToolRegistry(
        [
            Tool("ok", "ok", {"type": "object", "properties": {}}, lambda: "fine"),
            Tool("boom", "boom", {"type": "object", "properties": {}}, lambda: 1 / 0),
        ]
    )

    unknown = reg.execute(ToolCall("c1", "nope", "{}"))
    assert not unknown.ok and "unknown tool" in unknown.content

    bad_json = reg.execute(ToolCall("c2", "ok", "{not json"))
    assert not bad_json.ok and "valid JSON" in bad_json.content

    bad_args = reg.execute(ToolCall("c3", "ok", '{"extra": 1}'))
    assert not bad_args.ok and "bad arguments" in bad_args.content

    raised = reg.execute(ToolCall("c4", "boom", "{}"))
    assert not raised.ok and "ZeroDivisionError" in raised.content


def test_builtin_tools_act_on_memory(tmp_path):
    memory = _memory(tmp_path)
    cfg = ToolsConfig(builtins=["time", "remember_fact", "recall_fact"])
    reg = build_tools(memory, cfg)
    assert reg is not None
    assert set(reg.names()) == {"get_current_time", "remember_fact", "recall_fact"}

    remembered = reg.execute(
        ToolCall("c1", "remember_fact", json.dumps({"key": "name", "value": "Ada"}))
    )
    assert remembered.ok
    assert memory.recall_fact("name") == "Ada"

    recalled = reg.execute(ToolCall("c2", "recall_fact", '{"key": "name"}'))
    assert recalled.ok and recalled.content == "Ada"

    missing = reg.execute(ToolCall("c3", "recall_fact", '{"key": "nope"}'))
    assert missing.ok and "没有" in missing.content

    clock = reg.execute(ToolCall("c4", "get_current_time", "{}"))
    assert clock.ok and clock.content[:2].isdigit() and "星期" in clock.content


def test_build_tools_can_be_disabled(tmp_path):
    cfg = ToolsConfig(enabled=False)
    assert build_tools(_memory(tmp_path), cfg) is None


# ---- action tools -------------------------------------------------------


def test_memory_tools_use_runtime_person_not_model_input(tmp_path):
    memory = _memory(tmp_path)
    reg = build_tools(memory, ToolsConfig(builtins=["remember_fact", "recall_fact"]))
    stored = reg.execute(
        ToolCall(
            "c1",
            "remember_fact",
            json.dumps({"key": "name", "value": "Ada", "person_id": "spoofed"}),
        ),
        context={"person_id": "p1"},
    )
    assert stored.ok
    assert memory.recall_fact("name", person_id="p1") == "Ada"
    assert memory.recall_fact("name", person_id="spoofed") is None

    recalled = reg.execute(
        ToolCall("c2", "recall_fact", '{"key": "name"}'),
        context={"person_id": "p1"},
    )
    assert recalled.ok and recalled.content == "Ada"


def test_file_tools_are_sandboxed(tmp_path):
    root = tmp_path / "ws"
    cfg = ToolsConfig(
        builtins=["write_file", "read_file", "list_files"],
        workspace_root=str(root),
        max_file_bytes=64,
    )
    reg = build_tools(_memory(tmp_path), cfg)

    written = reg.execute(
        ToolCall(
            "c1",
            "write_file",
            json.dumps({"path": "notes/a.txt", "content": "你好\nworld"}),
        )
    )
    assert written.ok and "已写入" in written.content
    assert (root / "notes" / "a.txt").read_text(encoding="utf-8") == "你好\nworld"

    read = reg.execute(ToolCall("c2", "read_file", '{"path": "notes/a.txt"}'))
    assert read.ok and read.content == "你好\nworld"

    listing = reg.execute(ToolCall("c3", "list_files", '{"path": "notes"}'))
    assert listing.ok and "a.txt" in listing.content

    escape = reg.execute(
        ToolCall("c4", "write_file", json.dumps({"path": "../escape.txt", "content": "x"}))
    )
    assert not escape.ok and "escapes" in escape.content

    absolute = reg.execute(
        ToolCall(
            "c5",
            "write_file",
            json.dumps({"path": str(tmp_path / "outside.txt"), "content": "x"}),
        )
    )
    assert not absolute.ok and "escapes" in absolute.content

    too_big = reg.execute(
        ToolCall("c6", "write_file", json.dumps({"path": "big.txt", "content": "x" * 100}))
    )
    assert not too_big.ok and "limit" in too_big.content

    missing = reg.execute(ToolCall("c7", "read_file", '{"path": "nope.txt"}'))
    assert not missing.ok and "no such file" in missing.content

    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    linked = reg.execute(ToolCall("c8", "read_file", '{"path": "link.txt"}'))
    assert not linked.ok and "escapes" in linked.content

    (root / "bin.dat").write_bytes(b"\xff\xfe\x00")
    binary = reg.execute(ToolCall("c9", "read_file", '{"path": "bin.dat"}'))
    assert not binary.ok and "UTF-8" in binary.content

    (root / "empty.txt").write_bytes(b"")
    empty = reg.execute(ToolCall("c10", "read_file", '{"path": "empty.txt"}'))
    assert empty.ok and "空文件" in empty.content

    nul = reg.execute(
        ToolCall("c11", "read_file", json.dumps({"path": "a\x00b"}))
    )
    assert not nul.ok and "NUL" in nul.content

    # An absolute path is allowed when it is inside the workspace.
    inside_abs = root / "notes" / "a.txt"
    absolute_inside = reg.execute(
        ToolCall("c12", "read_file", json.dumps({"path": str(inside_abs)}))
    )
    assert absolute_inside.ok and absolute_inside.content == "你好\nworld"

    at_root = reg.execute(
        ToolCall("c13", "write_file", json.dumps({"path": ".", "content": "x"}))
    )
    assert not at_root.ok and "directory" in at_root.content


def test_browser_and_app_tools_validate_input(tmp_path):
    cfg = ToolsConfig(
        builtins=["open_browser", "open_app"],
        browser_command="/bin/true",
        allowed_apps={"true": "/bin/true"},
    )
    reg = build_tools(_memory(tmp_path), cfg)

    searched = reg.execute(
        ToolCall("c1", "open_browser", json.dumps({"query": "北京 天气"}))
    )
    assert searched.ok and "baidu.com/s?wd=" in searched.content

    for bad_url in ("file:///etc/passwd", "ftp://example.com", "javascript:alert(1)",
                    "data:text/html,<b>x</b>", "vbscript:msgbox(1)"):
        rejected = reg.execute(
            ToolCall("c2", "open_browser", json.dumps({"url": bad_url}))
        )
        assert not rejected.ok and "http" in rejected.content

    # A bare host with a port is not a URL scheme; it must still be accepted.
    host_port = reg.execute(
        ToolCall("c2b", "open_browser", json.dumps({"url": "example.com:8080"}))
    )
    assert host_port.ok and "https://example.com:8080" in host_port.content

    both = reg.execute(
        ToolCall("c3", "open_browser", json.dumps({"url": "https://a.com", "query": "x"}))
    )
    assert not both.ok

    empty = reg.execute(ToolCall("c4", "open_browser", "{}"))
    assert not empty.ok

    opened = reg.execute(ToolCall("c5", "open_app", '{"name": "true"}'))
    assert opened.ok

    unknown = reg.execute(ToolCall("c6", "open_app", '{"name": "rm"}'))
    assert not unknown.ok and "allowed" in unknown.content

    # A configured app value is an executable, never a shell line.
    bad_cfg = ToolsConfig(builtins=["open_app"], allowed_apps={"bad": "rm -rf /"})
    bad_reg = build_tools(_memory(tmp_path), bad_cfg)
    shellish = bad_reg.execute(ToolCall("c7", "open_app", '{"name": "bad"}'))
    assert not shellish.ok and "shell command" in shellish.content


def test_control_tools_require_opt_in(tmp_path):
    with pytest.raises(ValueError, match="control_enabled"):
        build_tools(_memory(tmp_path), ToolsConfig(builtins=["type_text"]))


def test_desktop_input_tools_use_xdotool(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    record = tmp_path / "record.txt"
    _fake_command(
        bin_dir,
        "xdotool",
        'if [ "$1" = "search" ]; then echo 4242; exit 0; fi\n'
        'if [ "$1" = "getwindowname" ]; then echo Demo; exit 0; fi\n'
        'printf "%s\\n" "$@" >> "$QVA_XDOTOOL_RECORD"\n'
        "exit 0\n",
    )
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QVA_XDOTOOL_RECORD", str(record))

    cfg = ToolsConfig(builtins=["type_text", "press_key"], control_enabled=True)
    reg = build_tools(_memory(tmp_path), cfg)

    typed = reg.execute(
        ToolCall("c1", "type_text", json.dumps({"text": "hello", "window": "Demo"}))
    )
    assert typed.ok and "已输入" in typed.content
    lines = record.read_text(encoding="utf-8").splitlines()
    assert "windowactivate" in lines and "4242" in lines
    assert "type" in lines and "hello" in lines
    # Focus succeeded, so the tool used XTEST rather than XSendEvent.
    assert "--window" not in lines

    pressed = reg.execute(
        ToolCall("c2", "press_key", json.dumps({"key": "ctrl+l", "window": "Demo"}))
    )
    assert pressed.ok and "已按键" in pressed.content
    lines = record.read_text(encoding="utf-8").splitlines()
    assert "key" in lines and "ctrl+l" in lines

    # Models often write "return"; xdotool only knows "Return".  The tool must
    # canonicalise it (and report an error, not success, if xdotool rejects it).
    normalised = reg.execute(
        ToolCall("c3", "press_key", json.dumps({"key": "return", "window": "Demo"}))
    )
    assert normalised.ok and "已按键 Return" in normalised.content
    assert "Return" in record.read_text(encoding="utf-8").splitlines()

    bad = reg.execute(ToolCall("c4", "press_key", '{"key": "rm -rf /"}'))
    assert not bad.ok and "keysym" in bad.content


def test_press_key_reports_xdotool_rejection(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    _fake_command(
        bin_dir,
        "xdotool",
        'if [ "$1" = "search" ]; then echo 4242; exit 0; fi\n'
        'if [ "$1" = "getwindowname" ]; then echo Demo; exit 0; fi\n'
        'echo "(symbol) No such key name" >&2\n'
        "exit 0\n",
    )
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("DISPLAY", ":0")

    cfg = ToolsConfig(builtins=["press_key"], control_enabled=True)
    reg = build_tools(_memory(tmp_path), cfg)
    result = reg.execute(ToolCall("c1", "press_key", '{"key": "Bogus"}'))
    assert not result.ok and "No such key name" in result.content


def test_screenshot_uses_available_backend(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    _fake_command(bin_dir, "gnome-screenshot", 'printf "PNG" > "$2"\n')
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("DISPLAY", ":0")

    cfg = ToolsConfig(
        builtins=["screenshot"],
        control_enabled=True,
        workspace_root=str(tmp_path / "ws"),
    )
    reg = build_tools(_memory(tmp_path), cfg)
    result = reg.execute(ToolCall("c1", "screenshot", '{"path": "shots/a.png"}'))
    assert result.ok and "已截图" in result.content
    assert (tmp_path / "ws" / "shots" / "a.png").read_bytes() == b"PNG"


def test_screenshot_reports_missing_backend(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    cfg = ToolsConfig(
        builtins=["screenshot"],
        control_enabled=True,
        workspace_root=str(tmp_path / "ws"),
    )
    reg = build_tools(_memory(tmp_path), cfg)
    result = reg.execute(ToolCall("c1", "screenshot", "{}"))
    assert not result.ok and "no screenshot backend" in result.content


def test_reminder_tools_persist_and_cancel(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "schedules.json"), retry_s=1.0)
    scheduler = Scheduler(cfg)
    tools_cfg = ToolsConfig(builtins=["set_reminder", "list_reminders", "cancel_reminder"])
    reg = build_tools(_memory(tmp_path), tools_cfg, scheduler=scheduler)

    created = reg.execute(
        ToolCall("c1", "set_reminder", json.dumps({"text": "喝水", "when": "10分钟后"}))
    )
    assert created.ok and "已设置提醒" in created.content
    assert len(scheduler) == 1

    listing = reg.execute(ToolCall("c2", "list_reminders", "{}"))
    assert listing.ok and "喝水" in listing.content

    schedule_id = scheduler.list()[0].id
    cancelled = reg.execute(
        ToolCall("c3", "cancel_reminder", json.dumps({"reminder_id": schedule_id}))
    )
    assert cancelled.ok and "已取消" in cancelled.content
    assert len(scheduler) == 0

    scheduler.add("关灯", time.time() + 3600, "once")
    reloaded = Scheduler(cfg)
    assert len(reloaded) == 1 and reloaded.list()[0].text == "关灯"


def test_schedule_task_tool_creates_a_task(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "schedules.json"))
    scheduler = Scheduler(cfg)
    tools_cfg = ToolsConfig(builtins=["schedule_task", "list_reminders"])
    reg = build_tools(_memory(tmp_path), tools_cfg, scheduler=scheduler)

    created = reg.execute(
        ToolCall(
            "c1",
            "schedule_task",
            json.dumps({"prompt": "播报天气", "when": "每天8点"}),
        )
    )
    assert created.ok and "已设置定时任务" in created.content
    item = scheduler.list()[0]
    assert item.kind == "task"
    assert item.repeat == "daily", "每天... should imply a daily repeat"

    listing = reg.execute(ToolCall("c2", "list_reminders", "{}"))
    assert listing.ok and "任务" in listing.content and "播报天气" in listing.content


def test_reminder_tools_require_a_scheduler(tmp_path):
    with pytest.raises(ValueError, match="scheduler.enabled"):
        build_tools(_memory(tmp_path), ToolsConfig(builtins=["set_reminder"]))


# ---- proactive delivery -------------------------------------------------


def test_pipeline_speaks_due_reminder(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.scheduler = SchedulerConfig(
        store_path=str(tmp_path / "schedules.json"),
        poll_interval_s=0.0,
        speak_prefix="提醒你：",
    )
    scheduler = Scheduler(cfg.scheduler)
    item = scheduler.add("喝水", time.time() - 1, "once")
    memory = _memory(tmp_path)
    pipeline = _pipeline(
        cfg, ScriptedLlm(["unused"]), memory, None, tmp_path, scheduler=scheduler
    )

    turns = _run_silence_until_turn(pipeline)
    proactive = [t for t in turns if t.proactive]
    assert proactive, "the due reminder was not spoken"
    assert proactive[0].reply == "提醒你：喝水"
    assert proactive[0].proactive_kind == "reminder"
    assert proactive[0].schedule_id == item.id
    assert len(scheduler) == 0, "a delivered one-shot must be removed"
    pipeline.close()


def test_pipeline_repeating_reminder_advances(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.scheduler = SchedulerConfig(
        store_path=str(tmp_path / "schedules.json"), poll_interval_s=0.0
    )
    scheduler = Scheduler(cfg.scheduler)
    scheduler.add("吃药", time.time() - 86400 * 2, "daily")
    memory = _memory(tmp_path)
    pipeline = _pipeline(
        cfg, ScriptedLlm(["unused"]), memory, None, tmp_path, scheduler=scheduler
    )

    turns = _run_silence_until_turn(pipeline)
    assert [t for t in turns if t.proactive]
    remaining = scheduler.list()
    assert len(remaining) == 1 and remaining[0].due_ts > time.time()
    pipeline.close()


def test_pipeline_runs_a_scheduled_task(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.scheduler = SchedulerConfig(
        store_path=str(tmp_path / "schedules.json"), poll_interval_s=0.0
    )
    scheduler = Scheduler(cfg.scheduler)
    scheduler.add("播报天气", time.time() - 1, "once", kind="task")
    memory = _memory(tmp_path)
    pipeline = _pipeline(
        cfg,
        ScriptedLlm([{"content": "今天晴，适合出门。"}]),
        memory,
        None,
        tmp_path,
        scheduler=scheduler,
    )

    turns = _run_silence_until_turn(pipeline)
    proactive = [t for t in turns if t.proactive]
    assert proactive, "the scheduled task did not run"
    assert proactive[0].reply == "今天晴，适合出门。"
    assert proactive[0].proactive_kind == "task"
    assert len(scheduler) == 0
    pipeline.close()


def test_pipeline_scheduled_task_uses_tools(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.scheduler = SchedulerConfig(
        store_path=str(tmp_path / "schedules.json"), poll_interval_s=0.0
    )
    scheduler = Scheduler(cfg.scheduler)
    scheduler.add("现在几点了？", time.time() - 1, "once", kind="task")
    memory = _memory(tmp_path)
    tools = build_tools(memory, ToolsConfig(builtins=["time"]))
    llm = ScriptedLlm(
        [
            {"tool_calls": [{"name": "get_current_time", "arguments": "{}"}]},
            {"content": "现在是十点二十分。"},
        ]
    )
    pipeline = _pipeline(cfg, llm, memory, tools, tmp_path, scheduler=scheduler)

    turns = _run_silence_until_turn(pipeline)
    proactive = [t for t in turns if t.proactive]
    assert proactive, "the scheduled task did not run"
    assert proactive[0].reply == "现在是十点二十分。"
    assert proactive[0].proactive_kind == "task"
    assert [c["name"] for c in proactive[0].tool_calls] == ["get_current_time"]
    assert len(scheduler) == 0
    pipeline.close()


def test_pipeline_speaks_multiple_due_reminders_in_order(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.scheduler = SchedulerConfig(
        store_path=str(tmp_path / "schedules.json"), poll_interval_s=0.0
    )
    scheduler = Scheduler(cfg.scheduler)
    first = scheduler.add("喝水", time.time() - 2, "once")
    second = scheduler.add("关灯", time.time() - 1, "once")
    memory = _memory(tmp_path)
    pipeline = _pipeline(
        cfg, ScriptedLlm(["unused"]), memory, None, tmp_path, scheduler=scheduler
    )

    turns = _run_silence_until_turn(pipeline)
    turns += _run_silence_until_turn(pipeline)
    proactive = [t for t in turns if t.proactive]
    assert [t.schedule_id for t in proactive] == [first.id, second.id]
    assert [t.reply for t in proactive] == ["提醒你：喝水", "提醒你：关灯"]
    assert len(scheduler) == 0
    pipeline.close()


def test_pipeline_defers_reminder_while_user_speaking(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.scheduler = SchedulerConfig(
        store_path=str(tmp_path / "schedules.json"), poll_interval_s=0.0
    )
    scheduler = Scheduler(cfg.scheduler)
    # The reminder becomes due *while* the utterance is open, which is the case
    # the gate must handle; a reminder that was already due before speech starts
    # is allowed to fire and can be barged in.
    scheduler.add("喝水", time.time() + 0.2, "once")
    memory = _memory(tmp_path)
    pipeline = _pipeline(
        cfg, ScriptedLlm(["unused"]), memory, None, tmp_path, scheduler=scheduler
    )

    for _ in range(5):
        pipeline.feed(_silence())
    for _ in range(5):
        pipeline.feed(_tone())
    assert pipeline.segmenter.active, "the test needs an open utterance"
    time.sleep(0.25)
    pipeline.poll_proactive()
    assert len(scheduler) == 1, "a due reminder must not interrupt an open utterance"
    assert not any(t.is_alive() for t in pipeline._workers)
    pipeline.close()


def test_pipeline_proactive_reminder_is_barge_in_interruptible(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.wake.enabled = True
    cfg.wake.wake_word = "小丽同学"
    cfg.scheduler = SchedulerConfig(
        store_path=str(tmp_path / "schedules.json"), poll_interval_s=0.0
    )
    scheduler = Scheduler(cfg.scheduler)
    scheduler.add("喝水", time.time() - 1, "once")
    memory = _memory(tmp_path)
    pipeline = Pipeline(
        cfg,
        vad=EnergyVad(cfg.vad),
        segmenter=Segmenter(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_samples),
        asr=MockAsr(),
        tts=SlowTts(1.0),
        llm=ScriptedLlm(["unused"]),
        memory=memory,
        sink=NullSink(),
        scheduler=scheduler,
    )

    assert _feed_until_worker(pipeline), "the reminder never started"
    assert pipeline.state == PipelineState.GENERATING
    for _ in range(10):
        pipeline.feed(_tone())
    assert pipeline._cancel.is_set(), "speech did not cancel the reminder"

    # The barge-in replays the interrupting frames into the segmenter, so a new
    # (ignored) user worker may start; drain it before checking the state.
    for _ in range(400):
        pipeline.feed(_silence())
        if pipeline.state == PipelineState.LISTENING and not any(
            t.is_alive() for t in pipeline._workers
        ):
            break
        time.sleep(0.005)
    assert pipeline.state == PipelineState.LISTENING
    remaining = scheduler.list()
    assert len(remaining) == 1, "an interrupted reminder must be kept for retry"
    assert remaining[0].next_attempt_ts > time.time()
    pipeline.close()


# ---- LLM parsing --------------------------------------------------------


def test_parse_reply_reads_tool_calls():
    # The exact shape llama-server returns (verified against 8082).
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_current_time", "arguments": "{}"},
                "id": "call_abc",
            }
        ],
    }
    reply = parse_reply(message)
    assert reply.content == ""
    assert reply.has_tool_calls
    assert reply.tool_calls[0].id == "call_abc"
    assert reply.tool_calls[0].name == "get_current_time"
    assert reply.tool_calls[0].arguments == "{}"


def test_parse_reply_rejects_nameless_call():
    from qwen_voice_agent.llm import LlmError

    with pytest.raises(LlmError):
        parse_reply({"content": "", "tool_calls": [{"function": {}}]})


# ---- pipeline loop ------------------------------------------------------


def test_pipeline_executes_tool_then_speaks_result(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    memory = _memory(tmp_path)
    tools = build_tools(memory, cfg.tools)
    llm = ScriptedLlm(
        [
            {"tool_calls": [{"name": "remember_fact", "arguments": '{"key": "name", "value": "Ada"}'}]},
            {"content": "好的，记住了。"},
        ]
    )
    p = _pipeline(cfg, llm, memory, tools, tmp_path)
    turns = _run_one_turn(p)
    assert turns, "pipeline produced no turn"

    turn = turns[0]
    assert turn.reply == "好的，记住了。"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0]["name"] == "remember_fact"
    assert turn.tool_calls[0]["ok"] is True
    # The tool wrote through to semantic memory, which is what makes the next
    # turn's prompt see it.
    assert memory.recall_fact("name") == "Ada"
    p.close()


def test_pipeline_bounds_tool_rounds(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.tools.max_rounds = 2
    memory = _memory(tmp_path)
    tools = build_tools(memory, cfg.tools)
    # Always asks for a tool; the final no-tools request gets the same dict, so
    # its content is empty and the pipeline must still say something.
    llm = ScriptedLlm([{"tool_calls": [{"name": "get_current_time", "arguments": "{}"}]}])
    p = _pipeline(cfg, llm, memory, tools, tmp_path)
    turns = _run_one_turn(p)
    assert turns

    turn = turns[0]
    assert len(turn.tool_calls) == 2, "max_rounds must bound the loop"
    assert turn.reply, "a voice agent must never answer with silence"
    p.close()


def test_task_busy_ignores_ambient_speech(tmp_path):
    cfg = VoiceAgentConfig()
    memory = _memory(tmp_path)
    tools = build_tools(memory, cfg.tools)
    p = _pipeline(cfg, ScriptedLlm(["hi"]), memory, tools, tmp_path)

    assert p._gate("随便说点什么", None) == ("respond", "随便说点什么")
    p._task_busy = True
    assert p._gate("电视里的人声", None) is None
    # An external policy may still let a correction through while busy.
    p.task_gate = lambda text: "停" in text
    assert p._gate("停下", None) == ("respond", "停下")
    p._task_busy = False
    p.close()
