"""Regression tests for the semantic gate and Teacher provider contract.

Covers: a semantic-conditional task never dispatches an action from the geometric
candidate alone — only a CONFIRMED in-schema match does (LIVE-001); the Teacher
generation guard blocks a new provider call after disable (LIVE-008); one absolute
deadline so expired waiting work cannot start a call (LIVE-009); strict confidence
and requested-label validation plus scoped cache identity (LIVE-010); decimal dwell
roundtrip; and strict composite-unsupported handling.

All Teacher calls go through a clearly-labeled QA transport — no real GPU/provider.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.live import config, dsl, planner, rooms, teacher, worker
from app.live.planner import PlannerResult
from app.live.teacher import Observation


@pytest.fixture(autouse=True)
def _clean_teacher(monkeypatch):
    """Isolate Teacher in-memory state between tests and neutralize pacing so the
    direct-adapter tests are not throttled by budget/cooldown."""
    teacher._reset_for_tests()
    teacher.transport = None
    monkeypatch.setattr(config, "TEACHER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "TEACHER_ROOM_BUDGET", 100)
    monkeypatch.setattr(config, "TEACHER_BASE_URL", "http://127.0.0.1:65535")  # availability flag
    yield
    teacher._reset_for_tests()
    teacher.transport = None


# --- LIVE-010 : strict typed provider contract -----------------------------


@pytest.mark.parametrize("bad", [True, False, "high", 1.5, -0.1, float("nan"),
                                 float("inf"), None])
def test_confidence_rejects_malformed(bad):
    assert teacher._valid_confidence(bad) is None


@pytest.mark.parametrize("good,expected", [(0.0, 0.0), (0.5, 0.5), (1, 1.0)])
def test_confidence_accepts_valid(good, expected):
    assert teacher._valid_confidence(good) == expected


def test_out_of_schema_label_is_unsupported():
    # Occupancy scoring ("occupied"/"empty") cannot answer an opening question.
    teacher.transport = lambda q, labels, imgs: Observation(
        status="ok", label="occupied", confidence=0.98)
    obs = teacher.observe("r1", "is the bottle open?", ["open", "closed"], [b"img"])
    assert obs.status == "unsupported"
    assert obs.in_schema is False
    teacher.transport = None


def test_in_schema_label_is_ok():
    teacher.transport = lambda q, labels, imgs: Observation(
        status="ok", label="open", confidence=0.7)
    obs = teacher.observe("r1", "is the bottle open?", ["open", "closed"], [b"img"])
    assert obs.status == "ok" and obs.in_schema is True
    assert obs.uncertain is True  # still flagged uncalibrated
    teacher.transport = None


def test_cache_scoped_by_labels_and_room():
    calls = {"n": 0}

    def _t(q, labels, imgs):
        calls["n"] += 1
        return Observation(status="ok", label=labels[0])

    teacher.transport = _t
    teacher.observe("r1", "q", ["open", "closed"], [b"same"])
    teacher.observe("r1", "q", ["open", "closed"], [b"same"])   # cached
    assert calls["n"] == 1
    teacher.observe("r1", "q", ["wet", "dry"], [b"same"])       # different labels
    assert calls["n"] == 2
    teacher.observe("r2", "q", ["open", "closed"], [b"same"])   # different room
    assert calls["n"] == 3
    teacher.transport = None


# --- LIVE-008 / LIVE-009 : generation guard + deadline ---------------------


def test_generation_guard_blocks_provider_call():
    called = {"n": 0}

    def _t(q, labels, imgs):
        called["n"] += 1
        return Observation(status="ok", label="open")

    teacher.transport = _t
    obs = teacher.observe("r1", "q", ["open", "closed"], [b"img"],
                          should_proceed=lambda: False)
    assert obs.status == "stale"
    assert called["n"] == 0  # no provider call started after cancellation
    teacher.transport = None


def test_expired_deadline_starts_no_call():
    # A deadline already in the past must not begin a provider request.
    obs = teacher._call_server("q", ["open", "closed"], [b"img"],
                               deadline=time.monotonic() - 1.0, should_proceed=None)
    assert obs.status == "timeout"


# --- LIVE-001 : semantic gate (no action without confirmation) -------------


def _bottle_semantic_task():
    return {
        "name": "focus if bottle opened", "object_class": "bottle",
        "zone": {"type": "full_frame"}, "trigger": "presence_in_zone",
        "action": "focus_notify", "focus": True,
        "visual_observation": {"question": "is the bottle open",
                               "labels": ["open", "closed"], "temporal": True},
    }


def _room_with_semantic_task(client, flow, pin):
    flow.login(client, "camera", pin)
    created = flow.post(client, "/api/live/rooms", {}).json()
    rid, code = created["room_id"], created["join_code"]
    judge = TestClient(client.app)
    flow.login(judge, "judge", pin)
    flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    flow.approve(judge, f"/api/live/rooms/{rid}/task/activate", {"task": _bottle_semantic_task()})
    return rid, judge


def _drive_frames(room, jpeg):
    with room.lock:
        version, task, engines = room.task_version, room.task, list(room.engines)
    for _ in range(3):
        worker.process_frame(room, jpeg, {"w": 320, "h": 240}, version, True, task, engines)


def test_semantic_unavailable_teacher_does_not_notify(client, test_pin, flow, jpeg_frame, monkeypatch):
    monkeypatch.setattr(config, "TEACHER_BASE_URL", "")  # unavailable
    teacher.transport = None
    rid, judge = _room_with_semantic_task(client, flow, test_pin)
    _drive_frames(rooms.get_room(rid), jpeg_frame)
    state = judge.get(f"/api/live/rooms/{rid}/state").json()
    assert len(state["events"]) >= 1               # recorded as observation
    assert state["events"][0]["action_status"] == "observation_only"
    assert state["notifications"] == []            # NO action authorized


def test_semantic_no_match_does_not_notify(client, test_pin, flow, jpeg_frame, monkeypatch):
    monkeypatch.setattr(config, "TEACHER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "TEACHER_ROOM_BUDGET", 100)
    teacher.transport = lambda q, labels, imgs: Observation(status="ok", label="closed")
    rid, judge = _room_with_semantic_task(client, flow, test_pin)
    _drive_frames(rooms.get_room(rid), jpeg_frame)
    state = judge.get(f"/api/live/rooms/{rid}/state").json()
    assert state["notifications"] == []
    assert state["events"][0]["action_status"] == "observation_only"
    teacher.transport = None


def test_semantic_confirmed_match_notifies(client, test_pin, flow, jpeg_frame, monkeypatch):
    monkeypatch.setattr(config, "TEACHER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "TEACHER_ROOM_BUDGET", 100)
    # Only a genuine in-schema match of the target (first) label confirms.
    teacher.transport = lambda q, labels, imgs: Observation(status="ok", label="open")
    rid, judge = _room_with_semantic_task(client, flow, test_pin)
    _drive_frames(rooms.get_room(rid), jpeg_frame)
    state = judge.get(f"/api/live/rooms/{rid}/state").json()
    assert len(state["notifications"]) >= 1
    assert state["events"][0]["action_status"] == "done"
    teacher.transport = None


def test_pure_geometric_task_still_notifies(client, test_pin, flow, jpeg_frame):
    # The primary automotive entry/exit/dwell flow (no visual_observation) is
    # unaffected and still produces a notification.
    planner.transport = lambda text, rid: PlannerResult(status="ok", proposal={
        "supported": True, "name": "watch car", "object_class": "car",
        "zone": {"type": "full_frame"}, "trigger": "presence_in_zone",
        "action": "notify", "focus": False, "explanation": "qa"})
    flow.login(client, "camera", test_pin)
    created = flow.post(client, "/api/live/rooms", {}).json()
    rid, code = created["room_id"], created["join_code"]
    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    plan = flow.post(judge, f"/api/live/rooms/{rid}/task/plan", {"text": "watch car"}).json()
    flow.approve(judge, f"/api/live/rooms/{rid}/task/activate", {"task": plan["task"]})
    _drive_frames(rooms.get_room(rid), jpeg_frame)
    state = judge.get(f"/api/live/rooms/{rid}/state").json()
    assert len(state["notifications"]) >= 1
    planner.transport = None


# --- DSL : decimal dwell + composite unsupported ---------------------------


def test_decimal_dwell_roundtrip():
    t = dsl.LiveTask(name="dwell", object_class="car", trigger="dwell_in_zone",
                     duration_seconds=1.25)
    dumped = t.model_dump()
    assert dumped["duration_seconds"] == 1.25
    # Canonical stored value feeds the engine unchanged.
    eng = rooms.build_engines(dumped, 320, 240, 1)[0]
    assert eng.dwell_seconds == 1.25


def test_activate_preserves_decimal_dwell(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    created = flow.post(client, "/api/live/rooms", {}).json()
    rid, code = created["room_id"], created["join_code"]
    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    task = {"name": "dwell", "object_class": "car", "zone": {"type": "full_frame"},
            "trigger": "dwell_in_zone", "duration_seconds": 1.25, "action": "notify"}
    r = flow.approve(judge, f"/api/live/rooms/{rid}/task/activate", {"task": task}).json()
    assert r["task"]["duration_seconds"] == 1.25
    state = judge.get(f"/api/live/rooms/{rid}/state").json()
    assert state["task"]["duration_seconds"] == 1.25


def test_composite_request_is_unsupported(client, test_pin, flow):
    # The planner must preserve all conditions or faithfully return unsupported —
    # a two-people-each-5-seconds request can never collapse into occupancy.
    planner.transport = lambda text, rid: PlannerResult(status="ok", proposal={
        "supported": False, "unsupported_reason": "composite per-object dwell not expressible",
        "clarification_needed": ""})
    flow.login(client, "camera", test_pin)
    created = flow.post(client, "/api/live/rooms", {}).json()
    rid, code = created["room_id"], created["join_code"]
    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    r = flow.post(judge, f"/api/live/rooms/{rid}/task/plan",
                  {"text": "alert when two people each stay 5 seconds"}).json()
    assert r["status"] == "unsupported"
    assert r["task"] is None
    planner.transport = None
