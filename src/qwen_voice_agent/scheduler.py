"""Scheduled reminders and proactive announcements.

A voice agent that only answers when spoken to is a chatbot.  A reminder that
fires at 08:00 while nobody is talking is the difference between an assistant
and a timer.  This module owns the durable schedule and the time parser; the
pipeline owns *speaking* the reminder, because it already has the TTS backend,
the sink and the barge-in machinery.

Design choices:

* **One JSON file, written atomically.**  A 24/7 process must survive a crash
  without losing the user's reminders; write-then-rename means a half-written
  file is never read back.
* **Claims, not deletion on read.**  ``claim_due`` marks a schedule as in flight
  before the pipeline speaks it.  A crash while speaking leaves the claim to
  expire (on restart claims are reset), so the reminder is retried rather than
  silently lost.
* **The parser is deliberately forgiving.**  A 4B model asked for "明天早上八点"
  will pass that string, not an ISO timestamp.  Parsing relative and Chinese
  wall-clock expressions here keeps the tool schema simple and the model honest.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional

from .config import SchedulerConfig

__all__ = ["Schedule", "Scheduler", "parse_when", "format_due", "REPEAT_LABELS", "KIND_LABELS"]

log = logging.getLogger("qwen_voice_agent.scheduler")

REPEAT_LABELS = {
    "once": "一次",
    "daily": "每天",
    "weekdays": "工作日",
    "weekly": "每周",
}

# "reminder" speaks a fixed line; "task" runs a prompt through the LLM at due
# time and speaks the answer (e.g. "每天8点播报天气").
KIND_LABELS = {"reminder": "提醒", "task": "任务"}

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_NUM = r"[零〇一二两三四五六七八九十百]+"
_DAY_WORDS = {"今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3}
_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_PERIODS = ("凌晨", "早上", "早晨", "上午", "中午", "下午", "傍晚", "晚上", "夜里", "夜间")
_EVENING = {"傍晚", "晚上", "夜里", "夜间"}

_REL_RE = re.compile(
    r"(?:(?P<n>\d+|" + _CN_NUM + r")\s*(?:个)?\s*)?"
    r"(?P<half_prefix>半)?"
    r"(?P<unit>秒|分钟|小时|钟头|天|周|星期)"
    r"\s*(?P<half>半)?\s*(?:后|以后|之后)"
)
_DATE_RE = re.compile(
    r"(?P<y>\d{4})\s*[-/年]\s*(?P<m>\d{1,2})\s*[-/月]\s*(?P<d>\d{1,2})\s*日?"
)
_CLOCK_RE = re.compile(r"(?P<h>\d{1,2})\s*[:：]\s*(?P<m>\d{1,2})(?:\s*[:：]\s*(?P<s>\d{1,2}))?")
_POINT_RE = re.compile(
    r"(?P<h>\d{1,2}|" + _CN_NUM + r")\s*(?:点|时)\s*"
    r"(?:(?P<half>半)|(?P<m>\d{1,2}|" + _CN_NUM + r")\s*分?)?"
)


def _cn_to_int(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    if "百" in text:
        raise ValueError(f"不支持的数字: {text!r}")
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _cn_to_int(left) if left else 1
        ones = _cn_to_int(right) if right else 0
        return tens * 10 + ones
    if text in _CN_DIGITS:
        return _CN_DIGITS[text]
    raise ValueError(f"不支持的数字: {text!r}")


def _strip_repeat_prefix(text: str) -> str:
    """Drop the "every ..." prefix but keep the weekday token intact.

    ``每周一9点`` must become ``周一9点``, not ``一9点``, or the weekday parser
    loses the day it needs.
    """
    text = text.strip()
    replacements = (
        ("每个星期", "星期"),
        ("每星期", "星期"),
        ("每礼拜", "礼拜"),
        ("每周", "周"),
        ("每天", ""),
        ("每日", ""),
        ("每晚", ""),
    )
    for prefix, replacement in replacements:
        if text.startswith(prefix):
            return (replacement + text[len(prefix):]).strip()
    return text


def _parse_relative(text: str, now: datetime) -> Optional[float]:
    if "一刻钟后" in text or "一刻钟以后" in text or "一刻钟之后" in text:
        return (now + timedelta(minutes=15)).timestamp()
    if text in {"一会儿", "过一会儿", "等一会儿"}:
        return (now + timedelta(minutes=5)).timestamp()
    m = _REL_RE.search(text)
    if not m:
        return None
    raw_n = m.group("n")
    half = bool(m.group("half") or m.group("half_prefix"))
    unit_seconds = {
        "秒": 1,
        "分钟": 60,
        "分": 60,
        "小时": 3600,
        "钟头": 3600,
        "天": 86400,
        "周": 604800,
        "星期": 604800,
    }[m.group("unit")]
    if raw_n:
        n = _cn_to_int(raw_n)
        total = n * unit_seconds + (unit_seconds / 2.0 if half else 0)
    elif half:
        total = unit_seconds / 2.0
    elif m.group("unit") in {"周", "星期"}:
        total = unit_seconds
    else:
        raise ValueError(f"没有说清多长时间: {text!r}")
    return (now + timedelta(seconds=total)).timestamp()


def _parse_clock(text: str) -> Optional[tuple[int, int, int]]:
    m = _CLOCK_RE.search(text)
    if m:
        return int(m.group("h")), int(m.group("m")), int(m.group("s") or 0)
    m = _POINT_RE.search(text)
    if not m:
        return None
    hour = _cn_to_int(m.group("h"))
    minute = 30 if m.group("half") else (_cn_to_int(m.group("m")) if m.group("m") else 0)
    return hour, minute, 0


def _parse_period(text: str) -> Optional[str]:
    for period in _PERIODS:
        if period in text:
            return period
    return None


def _apply_period(hour: int, period: Optional[str]) -> tuple[int, bool]:
    """Return (24h hour, crossed_midnight) for a wall-clock hour + period."""
    if period in _EVENING or period == "下午":
        if hour == 12 and period in _EVENING:
            return 0, True
        if hour < 12:
            return hour + 12, False
    if period == "凌晨" and hour == 12:
        return 0, False
    if period == "中午" and hour < 11:
        return hour + 12, False
    return hour, False


def parse_when(text: str, now: Optional[float] = None) -> float:
    """Parse a human time expression into an epoch timestamp.

    Supported forms (all local time):

    * ``10分钟后`` / ``半小时后`` / ``两天后`` / ``一周后``
    * ``8点`` / ``8:30`` / ``下午3点半`` / ``晚上8点``
    * ``今天`` / ``明天`` / ``后天`` + a clock
    * ``周一9点`` / ``每周一9点`` (the next occurrence)
    * ``2026-10-20 09:00`` / ``2026-10-20`` (09:00 default)
    * ``现在`` / ``马上``

    Raises ``ValueError`` with a readable message when the expression is
    ambiguous, malformed, or explicitly in the past.  The error text is fed back
    to the model, which can retry with a clearer expression.
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("when is empty")
    current = datetime.fromtimestamp(time.time() if now is None else now)
    text = _strip_repeat_prefix(raw)

    if text in {"现在", "马上", "立刻", "立即"}:
        return (current + timedelta(seconds=5)).timestamp()

    relative = _parse_relative(text, current)
    if relative is not None:
        return relative

    # ISO 8601 first: it is unambiguous and covers both date and time.
    try:
        iso = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if iso.tzinfo is not None:
            return iso.timestamp()
        if len(text.strip()) == 10 and "T" not in text and " " not in text:
            iso = iso.replace(hour=9)
        if iso <= current:
            raise ValueError(f"提醒时间已经过去: {raw!r}")
        return iso.timestamp()
    except ValueError as exc:
        if "已经过去" in str(exc):
            raise
        # Not ISO; fall through to the Chinese wall-clock parser.

    clock = _parse_clock(text)
    date_match = _DATE_RE.search(text)
    if date_match:
        year = int(date_match.group("y"))
        month = int(date_match.group("m"))
        day = int(date_match.group("d"))
        hour, minute, second = clock if clock else (9, 0, 0)
        try:
            candidate = datetime(year, month, day, hour, minute, second)
        except ValueError as exc:
            raise ValueError(f"日期不合法: {raw!r}") from exc
        if candidate <= current:
            raise ValueError(f"提醒时间已经过去: {raw!r}")
        return candidate.timestamp()

    if clock is None:
        # A day word alone ("明天") defaults to 09:00; a weekday alone too.
        if any(word in text for word in _DAY_WORDS) or re.search(r"(?:周|星期|礼拜)[一二三四五六日天]", text):
            clock = (9, 0, 0)
        else:
            raise ValueError(
                f"没有识别出时间: {raw!r}；可用“10分钟后”“明天早上8点”“2026-10-20 09:00”"
            )
    hour, minute, second = clock
    period = _parse_period(text)
    hour, crossed_midnight = _apply_period(hour, period)

    weekday_match = re.search(r"(?:周|星期|礼拜)([一二三四五六日天])", text)
    if weekday_match:
        weekday = _WEEKDAYS[weekday_match.group(1)]
        days_ahead = (weekday - current.weekday()) % 7
        date = current.date() + timedelta(days=days_ahead)
        candidate = datetime.combine(date, dtime(hour, minute, second))
        if candidate <= current:
            candidate += timedelta(days=7)
        return candidate.timestamp()

    day_offset = None
    for word, offset in _DAY_WORDS.items():
        if word in text:
            day_offset = offset
            break
    if day_offset is not None:
        date = current.date() + timedelta(days=day_offset)
        candidate = datetime.combine(date, dtime(hour, minute, second))
        if candidate <= current:
            raise ValueError(f"提醒时间已经过去: {raw!r}")
        return candidate.timestamp()

    candidate = datetime.combine(current.date(), dtime(hour, minute, second))
    if crossed_midnight or candidate <= current:
        candidate += timedelta(days=1)
    return candidate.timestamp()


def format_due(ts: float) -> str:
    """Format a due time with a relative day label when it is close.

    A bare date is technically correct but a 4B model reads "2026-10-19 12:14"
    as "tomorrow" often enough to matter; "今天 12:14" removes the ambiguity.
    """
    due = datetime.fromtimestamp(ts)
    delta_days = (due.date() - datetime.now().date()).days
    label = {0: "今天", 1: "明天", 2: "后天"}.get(delta_days)
    if label is not None:
        return f"{label} {due:%H:%M}"
    return due.strftime("%Y-%m-%d %H:%M")


def _next_occurrence(due_ts: float, repeat: str, now: float) -> float:
    """Advance a repeating schedule past ``now`` (wall-clock preserving)."""
    dt = datetime.fromtimestamp(due_ts)
    while True:
        if repeat == "daily":
            dt += timedelta(days=1)
        elif repeat == "weekly":
            dt += timedelta(days=7)
        elif repeat == "weekdays":
            dt += timedelta(days=1)
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
        else:
            return due_ts
        if dt.timestamp() > now:
            return dt.timestamp()


@dataclass
class Schedule:
    """One durable reminder or recurring task."""

    id: str
    text: str
    due_ts: float
    repeat: str = "once"
    kind: str = "reminder"  # "reminder" | "task"
    created_ts: float = 0.0
    enabled: bool = True
    # Runtime-only fields: never persisted.  A claim expires so a crashed
    # process retries instead of losing the reminder.
    claimed_until: float = 0.0
    next_attempt_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "due_ts": self.due_ts,
            "repeat": self.repeat,
            "kind": self.kind,
            "created_ts": self.created_ts,
            "enabled": self.enabled,
        }


class Scheduler:
    """A JSON-backed set of one-shot and repeating schedules."""

    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg
        self.path = cfg.store_path or ""
        self._lock = threading.Lock()
        self._schedules: List[Schedule] = []
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read schedules %s: %s; starting empty", self.path, exc)
            return
        items = data.get("schedules", []) if isinstance(data, dict) else []
        for item in items[: max(0, int(self.cfg.max_schedules))]:
            if not isinstance(item, dict):
                continue
            try:
                schedule = Schedule(
                    id=str(item["id"]),
                    text=str(item["text"]),
                    due_ts=float(item["due_ts"]),
                    repeat=str(item.get("repeat", "once")),
                    kind=str(item.get("kind", "reminder")),
                    created_ts=float(item.get("created_ts", 0.0)),
                    enabled=bool(item.get("enabled", True)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("skipping malformed schedule %r: %s", item, exc)
                continue
            if schedule.repeat not in REPEAT_LABELS:
                schedule.repeat = "once"
            if schedule.kind not in KIND_LABELS:
                schedule.kind = "reminder"
            # Claims are process-local; a restart means "try again".
            schedule.claimed_until = 0.0
            schedule.next_attempt_ts = 0.0
            self._schedules.append(schedule)

    def _save(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        payload = {"version": 1, "schedules": [s.to_dict() for s in self._schedules]}
        tmp = f"{self.path}.tmp-{os.getpid()}-{time.time_ns()}"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # ---- public --------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._schedules)

    def add(
        self, text: str, due_ts: float, repeat: str = "once", kind: str = "reminder"
    ) -> Schedule:
        text = str(text or "").strip()
        if not text:
            raise ValueError("schedule text is empty")
        if repeat not in REPEAT_LABELS:
            raise ValueError(f"unknown repeat {repeat!r}; use {', '.join(REPEAT_LABELS)}")
        if kind not in KIND_LABELS:
            raise ValueError(f"unknown kind {kind!r}; use {', '.join(KIND_LABELS)}")
        with self._lock:
            if len(self._schedules) >= max(1, int(self.cfg.max_schedules)):
                raise ValueError(f"too many schedules (limit {self.cfg.max_schedules})")
            schedule = Schedule(
                id=uuid.uuid4().hex[:8],
                text=text,
                due_ts=float(due_ts),
                repeat=repeat,
                kind=kind,
                created_ts=time.time(),
            )
            self._schedules.append(schedule)
            self._save()
            return replace(schedule)

    def cancel(self, schedule_id: str) -> bool:
        key = str(schedule_id or "").strip()
        with self._lock:
            before = len(self._schedules)
            self._schedules = [s for s in self._schedules if s.id != key]
            changed = len(self._schedules) != before
            if changed:
                self._save()
            return changed

    def list(self) -> List[Schedule]:
        with self._lock:
            items = sorted(self._schedules, key=lambda s: s.due_ts)
            return [replace(s) for s in items]

    def claim_due(self, now: Optional[float] = None) -> Optional[Schedule]:
        """Claim the earliest due schedule for delivery, or ``None``.

        A claim prevents the next audio frame from starting the same reminder
        again while the pipeline is still speaking it.  ``complete`` or
        ``release`` must follow.
        """
        now = time.time() if now is None else now
        with self._lock:
            for schedule in sorted(self._schedules, key=lambda s: s.due_ts):
                if not schedule.enabled or schedule.due_ts > now:
                    continue
                if schedule.claimed_until > now or schedule.next_attempt_ts > now:
                    continue
                schedule.claimed_until = now + max(30.0, float(self.cfg.retry_s) * 5)
                self._save()
                return replace(schedule)
        return None

    def complete(self, schedule_id: str) -> None:
        """Mark delivery successful: remove a one-shot, advance a repeat."""
        now = time.time()
        with self._lock:
            for schedule in self._schedules:
                if schedule.id != schedule_id:
                    continue
                if schedule.repeat == "once":
                    self._schedules = [s for s in self._schedules if s.id != schedule_id]
                else:
                    schedule.due_ts = _next_occurrence(schedule.due_ts, schedule.repeat, now)
                    schedule.claimed_until = 0.0
                    schedule.next_attempt_ts = 0.0
                self._save()
                return

    def release(self, schedule_id: str, retry_after_s: Optional[float] = None) -> None:
        """Delivery failed or was interrupted; retry after a delay."""
        delay = max(0.0, float(self.cfg.retry_s if retry_after_s is None else retry_after_s))
        now = time.time()
        with self._lock:
            for schedule in self._schedules:
                if schedule.id != schedule_id:
                    continue
                schedule.claimed_until = 0.0
                schedule.next_attempt_ts = now + delay
                self._save()
                return

    def close(self) -> None:
        with self._lock:
            self._save()
