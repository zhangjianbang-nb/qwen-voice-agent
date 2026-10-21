"""Scheduler: time parsing, durable store, and claim/complete semantics."""

import json
import time
from datetime import datetime

import pytest

from qwen_voice_agent.config import SchedulerConfig
from qwen_voice_agent.scheduler import Scheduler, parse_when


def _ts(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute).timestamp()


NOW = _ts(2026, 10, 19, 10, 0)  # a Monday, 10:00 local


# ---- parsing ------------------------------------------------------------


def test_parse_relative_expressions():
    assert parse_when("10分钟后", NOW) == pytest.approx(NOW + 600)
    assert parse_when("十分钟后", NOW) == pytest.approx(NOW + 600)
    assert parse_when("半小时后", NOW) == pytest.approx(NOW + 1800)
    assert parse_when("一个半小时后", NOW) == pytest.approx(NOW + 5400)
    assert parse_when("两小时后", NOW) == pytest.approx(NOW + 7200)
    assert parse_when("三天后", NOW) == pytest.approx(NOW + 3 * 86400)
    assert parse_when("两天半后", NOW) == pytest.approx(NOW + 2.5 * 86400)
    assert parse_when("一刻钟后", NOW) == pytest.approx(NOW + 900)
    assert parse_when("一周后", NOW) == pytest.approx(NOW + 7 * 86400)
    assert parse_when("现在", NOW) == pytest.approx(NOW + 5)


def test_parse_wall_clock_and_day_words():
    assert parse_when("下午3点半", NOW) == pytest.approx(_ts(2026, 10, 19, 15, 30))
    assert parse_when("晚上8点", NOW) == pytest.approx(_ts(2026, 10, 19, 20, 0))
    assert parse_when("中午12点", NOW) == pytest.approx(_ts(2026, 10, 19, 12, 0))
    assert parse_when("下午1点", NOW) == pytest.approx(_ts(2026, 10, 19, 13, 0))
    assert parse_when("明天早上8点", NOW) == pytest.approx(_ts(2026, 10, 20, 8, 0))
    assert parse_when("后天晚上8点", NOW) == pytest.approx(_ts(2026, 10, 21, 20, 0))
    assert parse_when("明天", NOW) == pytest.approx(_ts(2026, 10, 20, 9, 0))
    # A bare clock time that has already passed rolls to tomorrow.
    assert parse_when("8点", NOW) == pytest.approx(_ts(2026, 10, 20, 8, 0))
    assert parse_when("8:30", NOW) == pytest.approx(_ts(2026, 10, 20, 8, 30))
    # Midnight is the next day, not noon.
    assert parse_when("晚上12点", NOW) == pytest.approx(_ts(2026, 10, 20, 0, 0))
    assert parse_when("凌晨1点", NOW) == pytest.approx(_ts(2026, 10, 20, 1, 0))


def test_parse_weekdays_and_iso():
    assert parse_when("周一9点", NOW) == pytest.approx(_ts(2026, 10, 26, 9, 0))
    assert parse_when("周三9点", NOW) == pytest.approx(_ts(2026, 10, 21, 9, 0))
    assert parse_when("每周一9点", NOW) == pytest.approx(_ts(2026, 10, 26, 9, 0))
    assert parse_when("2026-10-20 09:00", NOW) == pytest.approx(_ts(2026, 10, 20, 9, 0))
    assert parse_when("2026-10-20", NOW) == pytest.approx(_ts(2026, 10, 20, 9, 0))
    assert parse_when("2026年10月20日 9点", NOW) == pytest.approx(_ts(2026, 10, 20, 9, 0))


def test_parse_rejects_past_and_garbage():
    with pytest.raises(ValueError, match="过去"):
        parse_when("2026-10-18 09:00", NOW)
    with pytest.raises(ValueError, match="过去"):
        parse_when("今天下午3点", _ts(2026, 10, 19, 16, 0))
    with pytest.raises(ValueError):
        parse_when("", NOW)
    with pytest.raises(ValueError):
        parse_when("随便什么时候", NOW)


# ---- store --------------------------------------------------------------


def test_scheduler_claim_release_complete(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "s.json"), retry_s=1.0)
    scheduler = Scheduler(cfg)
    item = scheduler.add("喝水", time.time() - 1, "once")

    claimed = scheduler.claim_due()
    assert claimed is not None and claimed.id == item.id
    assert scheduler.claim_due() is None, "a claim must prevent duplicate delivery"

    scheduler.release(item.id, retry_after_s=0.0)
    assert scheduler.claim_due() is not None

    scheduler.complete(item.id)
    assert len(scheduler) == 0
    assert scheduler.claim_due() is None


def test_scheduler_persists_across_restart(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "s.json"))
    scheduler = Scheduler(cfg)
    scheduler.add("关灯", time.time() + 3600, "daily")

    reloaded = Scheduler(cfg)
    assert len(reloaded) == 1
    assert reloaded.list()[0].text == "关灯"
    assert reloaded.list()[0].repeat == "daily"
    assert reloaded.list()[0].kind == "reminder"


def test_scheduler_claim_is_reset_on_restart(tmp_path):
    """A crash while speaking must not hide the reminder forever."""
    cfg = SchedulerConfig(store_path=str(tmp_path / "s.json"), retry_s=60.0)
    scheduler = Scheduler(cfg)
    item = scheduler.add("喝水", time.time() - 1, "once")
    assert scheduler.claim_due() is not None

    reloaded = Scheduler(cfg)
    claimed = reloaded.claim_due()
    assert claimed is not None and claimed.id == item.id


def test_scheduler_persists_task_kind(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "s.json"))
    scheduler = Scheduler(cfg)
    scheduler.add("播报天气", time.time() + 60, "daily", kind="task")

    reloaded = Scheduler(cfg)
    assert len(reloaded) == 1
    assert reloaded.list()[0].kind == "task"


def test_scheduler_repeat_advances_past_now(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "s.json"))
    scheduler = Scheduler(cfg)
    item = scheduler.add("吃药", time.time() - 86400 * 2, "daily")
    assert scheduler.claim_due() is not None
    scheduler.complete(item.id)

    remaining = scheduler.list()
    assert len(remaining) == 1
    assert remaining[0].due_ts > time.time()


def test_scheduler_weekday_repeat_skips_weekend(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "s.json"))
    scheduler = Scheduler(cfg)
    item = scheduler.add("上班", time.time() - 86400, "weekdays")
    assert scheduler.claim_due() is not None
    scheduler.complete(item.id)

    due = scheduler.list()[0].due_ts
    assert due > time.time()
    assert datetime.fromtimestamp(due).weekday() < 5


def test_scheduler_limit_and_cancel(tmp_path):
    cfg = SchedulerConfig(store_path=str(tmp_path / "s.json"), max_schedules=2)
    scheduler = Scheduler(cfg)
    first = scheduler.add("a", time.time() + 60)
    scheduler.add("b", time.time() + 120)

    with pytest.raises(ValueError, match="too many"):
        scheduler.add("c", time.time() + 180)
    assert scheduler.cancel(first.id)
    assert not scheduler.cancel("missing")
    assert len(scheduler) == 1


def test_scheduler_recovers_from_corrupt_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json", encoding="utf-8")
    scheduler = Scheduler(SchedulerConfig(store_path=str(path)))
    assert len(scheduler) == 0
    scheduler.add("ok", time.time() + 60)
    assert len(Scheduler(SchedulerConfig(store_path=str(path)))) == 1


def test_scheduler_rejects_unknown_repeat(tmp_path):
    scheduler = Scheduler(SchedulerConfig(store_path=str(tmp_path / "s.json")))
    with pytest.raises(ValueError, match="repeat"):
        scheduler.add("x", time.time() + 60, "hourly")
