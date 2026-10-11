"""Planner typed-validation + budget/cooldown tests.

These exercise the REAL validation/budget logic through an injected deterministic
transport (QA ONLY). The real `claude` CLI is never spawned here — a few genuine
novel-prompt CLI calls are reserved for the human E2E run, not the unit suite.
"""

import pytest

from app.live import config, planner
from app.live.planner import PlannerResult


@pytest.fixture(autouse=True)
def _clean():
    planner._room_attempts.clear()
    planner._room_last_attempt.clear()
    planner.transport = None
    yield
    planner.transport = None


def test_supported_proposal_validates_to_task(monkeypatch):
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    planner.transport = lambda text, rid: PlannerResult(status="ok", proposal={
        "supported": True, "name": "Service bay entry", "object_class": "car",
        "zone": {"type": "full_frame"}, "trigger": "entered_zone",
        "action": "notify", "explanation": "car entering the bay",
    })
    r = planner.plan("alert me when a car enters the service bay", "room1")
    assert r.status == "ok"
    assert r.task["object_class"] == "car"
    assert r.task["trigger"] == "entered_zone"


def _semantic_proposal():
    return PlannerResult(status="ok", proposal={
        "supported": True, "name": "Bottle opening watch", "object_class": "bottle",
        "zone": {"type": "full_frame"}, "trigger": "presence_in_zone",
        "action": "focus_notify", "focus": True,
        "visual_observation": {
            "question": "Is the bottle being opened?",
            "labels": ["opening", "closed"], "temporal": True,
        },
        "explanation": "focus the bottle; opening is a temporal check",
        "clarification_needed": "Opening is detected by a weak VLM and may be uncertain.",
    })


def test_semantic_request_is_unsupported_in_video_mode(monkeypatch):
    """LIVE-VIDEO-001: on the enabled VIDEO surface (camera ingest disabled) a
    semantic/state-change request is rejected fail-closed — never downgraded to a
    presence rule or a visual_observation."""
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", False)
    planner.transport = lambda text, rid: _semantic_proposal()
    r = planner.plan("focus on the bottle if anybody opens it", "room1")
    assert r.status == "unsupported"
    assert r.task is None
    assert "video mode" in (r.unsupported_reason or "")


def test_temporal_semantic_preserves_intent_on_legacy_camera(monkeypatch):
    """Legacy camera path (local QA only, camera ingest enabled) still carries the
    semantic condition as an (uncertain) visual_observation."""
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", True)
    planner.transport = lambda text, rid: _semantic_proposal()
    r = planner.plan("focus on the bottle if anybody opens it", "room1")
    assert r.status == "ok"
    assert r.task["object_class"] == "bottle"
    assert r.task["focus"] is True
    assert r.task["visual_observation"]["temporal"] is True
    assert r.clarification


def test_unsupported_proposal(monkeypatch):
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    planner.transport = lambda text, rid: PlannerResult(status="ok", proposal={
        "supported": False, "name": "n/a", "explanation": "not expressible",
        "unsupported_reason": "audio events are not supported",
        "clarification_needed": "Can you rephrase as a visual condition?",
    })
    r = planner.plan("tell me when it gets loud", "room1")
    assert r.status == "unsupported"
    assert r.task is None
    assert "audio" in r.unsupported_reason


def test_schema_valid_but_bad_class_downgraded(monkeypatch):
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    planner.transport = lambda text, rid: PlannerResult(status="ok", proposal={
        "supported": True, "name": "x", "object_class": "unicorn",
        "zone": {"type": "full_frame"}, "trigger": "presence_in_zone",
        "action": "notify", "explanation": "x",
    })
    r = planner.plan("watch the unicorn", "room1")
    assert r.status == "unsupported"
    assert r.task is None
    assert "unicorn" in (r.unsupported_reason or "")


def test_budget_exhausted(monkeypatch):
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "PLANNER_ROOM_BUDGET", 2)
    planner.transport = lambda text, rid: PlannerResult(status="ok", proposal={
        "supported": True, "name": "x", "object_class": "car",
        "zone": {"type": "full_frame"}, "trigger": "presence_in_zone",
        "action": "notify", "explanation": "x",
    })
    assert planner.plan("a", "room1").status == "ok"
    assert planner.plan("b", "room1").status == "ok"
    assert planner.plan("c", "room1").status == "budget_exhausted"


def test_cooldown(monkeypatch):
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 1000.0)
    monkeypatch.setattr(config, "PLANNER_ROOM_BUDGET", 100)
    planner.transport = lambda text, rid: PlannerResult(status="ok", proposal={
        "supported": True, "name": "x", "object_class": "car",
        "zone": {"type": "full_frame"}, "trigger": "presence_in_zone",
        "action": "notify", "explanation": "x",
    })
    assert planner.plan("a", "room1").status == "ok"
    assert planner.plan("b", "room1").status == "cooldown"


def test_empty_text_unavailable():
    assert planner.plan("   ", "room1").status == "unavailable"
