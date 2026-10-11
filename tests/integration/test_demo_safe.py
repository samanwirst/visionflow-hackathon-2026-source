"""Demo-safe backend regressions (VISIONFLOW_DEMO_SAFE=True).

These prove the demo's safety boundary is enforced in the BACKEND, not just the
UI, independent of UI code:

- the NL compiler is never invoked, even for a known-bad M2-001 prompt;
- a visual-predicate workflow cannot be created or enabled;
- an already-enabled legacy visual workflow is skipped at processing with ZERO
  Teacher admission, even when a VLM endpoint is configured and a counting stub
  Teacher is installed — and it is refused, never coerced to geometric;
- a manual geometric rule stays disabled until an explicit enable (the reviewed
  activation), while the geometric path itself keeps working;
- VLM stats report disabled WITHOUT instantiating the router/Teacher.

The original M2 experimental tests opt out of demo-safe mode explicitly; these
tests run with the production default (True).
"""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, main as main_module, nl_compiler, pipeline
from app.db import VlmCall, Workflow, get_session
from app.main import app
from app.vlm_router import VlmRouter, set_router
from app.teacher import TeacherAdapter, TeacherResult
from visionflow_vision.detector import BaseDetectorTracker
from visionflow_vision.events import TrackedObject


class ScriptedDetector(BaseDetectorTracker):
    """One 'car' track sweeping left->right; enters the right-half zone."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_inference_ms = 0.1

    def reset(self) -> None:
        self.calls = 0

    def track_frame(self, frame, timestamp: float):
        h, w = frame.shape[:2]
        cx = (self.calls / 8.0) * w
        self.calls += 1
        if cx < 10 or cx > w - 10:
            return []
        return [TrackedObject(track_id=1, class_name="car", confidence=0.95,
                              x1=cx - 10, y1=100, x2=cx + 10, y2=140)]


class CountingTeacher(TeacherAdapter):
    """Records every classify() call so we can assert ZERO Teacher admission."""

    provider_name = "counting-stub"
    model = "counting-stub-vlm"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, image_bytes, prompt, labels, on_attempt=None):
        self.calls += 1
        return TeacherResult(status="ok", label="yes", confidence=0.9, model=self.model)


@pytest.fixture(autouse=True)
def _demo_safe_mode():
    """Run with the production demo-safe default and restore config afterwards."""
    prev = (config.DEMO_SAFE, config.VLM_BASE_URL, config.VLM_MODEL)
    config.DEMO_SAFE = True
    try:
        yield
    finally:
        config.DEMO_SAFE, config.VLM_BASE_URL, config.VLM_MODEL = prev
        set_router(None)


@pytest.fixture
def client():
    pipeline.detector_factory = ScriptedDetector
    pipeline._detector = None
    with TestClient(app) as c:
        yield c
    pipeline.detector_factory = None
    pipeline._detector = None
    set_router(None)


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("demovideo") / "d.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (int(i / 60 * 320), 120), 15, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _setup_source_zone(c, video_path, name):
    with open(video_path, "rb") as f:
        source = c.post("/api/sources/upload", files={"file": (name, f, "video/mp4")}).json()
    zone = c.post("/api/zones", json={
        "source_id": source["id"], "name": "Service Bay A",
        "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
    }).json()
    return source, zone


def _process_and_wait(c, source_id):
    assert c.post(f"/api/sources/{source_id}/process").status_code == 202
    deadline = time.time() + 30
    while time.time() < deadline:
        s = c.get(f"/api/sources/{source_id}").json()["status"]
        if s in ("processed", "failed"):
            return s
        time.sleep(0.2)
    return "timeout"


def _visual_dsl(source_id, zone_id):
    return {
        "name": "Legacy hood check",
        "source_id": source_id,
        "object_class": "car",
        "trigger": {"type": "object_entered_zone", "zone_id": zone_id},
        "action": {"type": "create_notification", "config": {"title": "Hood"}},
        "visual_predicate": {
            "prompt": "Is the vehicle hood open?",
            "labels": ["yes", "no"], "expected_label": "yes",
            "description": "hood is open",
            "sample_interval_seconds": 0, "cooldown_seconds": 0,
        },
        "cooldown_seconds": 5,
        "requires_approval": True,
    }


# -- compiler is never invoked -------------------------------------------------

def test_compile_never_invokes_compiler_even_on_bad_prompt(client, video_path, monkeypatch):
    source, _ = _setup_source_zone(client, video_path, "compile1.mp4")
    called = {"n": 0}

    def spy(*a, **k):  # must never run in demo-safe mode
        called["n"] += 1
        raise AssertionError("nl_compiler.compile_text was invoked in demo-safe mode")

    monkeypatch.setattr(nl_compiler, "compile_text", spy)

    # A known-bad M2-001 prompt (data-management action + ambiguity) must not even
    # reach the compiler grammar.
    r = client.post("/api/workflows/compile", json={
        "source_id": source["id"],
        "text": "delete the video when a car does not enter Service Bay A",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["draft"] is None
    assert body["errors"]
    assert called["n"] == 0


# -- visual workflow cannot be created or enabled ------------------------------

def test_create_visual_workflow_rejected(client, video_path):
    source, zone = _setup_source_zone(client, video_path, "vc1.mp4")
    r = client.post("/api/workflows", json=_visual_dsl(source["id"], zone["id"]))
    assert r.status_code == 403, r.text
    assert client.get(f"/api/workflows?source_id={source['id']}").json() == []


def test_enable_legacy_visual_workflow_rejected(client, video_path):
    source, zone = _setup_source_zone(client, video_path, "ve1.mp4")
    # Insert a legacy visual workflow row directly (bypassing the create guard).
    session = get_session()
    try:
        wf = Workflow(source_id=source["id"], name="legacy visual",
                      dsl=_visual_dsl(source["id"], zone["id"]), enabled=False)
        session.add(wf)
        session.commit()
        wf_id = wf.id
    finally:
        session.close()
    r = client.post(f"/api/workflows/{wf_id}/enable")
    assert r.status_code == 403, r.text
    # still disabled
    session = get_session()
    try:
        assert session.get(Workflow, wf_id).enabled is False
    finally:
        session.close()


# -- enabled legacy visual workflow is skipped: zero Teacher admission ---------

def test_enabled_legacy_visual_workflow_skipped_no_teacher(client, video_path):
    # Even with a VLM endpoint "configured" and a counting Teacher installed, a
    # legacy ENABLED visual workflow must not reach the Teacher.
    config.VLM_BASE_URL = "https://configured.invalid/v1"
    config.VLM_MODEL = "configured-vlm"
    teacher = CountingTeacher()
    set_router(VlmRouter(teacher_provider=lambda: teacher, sample_interval_s=0, cooldown_s=0))

    source, zone = _setup_source_zone(client, video_path, "vp1.mp4")
    session = get_session()
    try:
        wf = Workflow(source_id=source["id"], name="legacy enabled visual",
                      dsl=_visual_dsl(source["id"], zone["id"]), enabled=True)
        session.add(wf)
        session.commit()
        wf_id = wf.id
    finally:
        session.close()

    assert _process_and_wait(client, source["id"]) == "processed"

    # No Teacher call, no events, no VlmCall rows for the skipped workflow.
    assert teacher.calls == 0
    events = client.get(f"/api/events?source_id={source['id']}&workflow_id={wf_id}").json()
    assert events == []
    session = get_session()
    try:
        assert session.query(VlmCall).filter(VlmCall.workflow_id == wf_id).count() == 0
    finally:
        session.close()


# -- manual geometric draft stays disabled until explicit activation -----------

def test_manual_geometric_draft_disabled_until_explicit_enable(client, video_path):
    source, zone = _setup_source_zone(client, video_path, "geo1.mp4")
    wf = client.post("/api/workflows", json={
        "name": "Car entered", "source_id": source["id"], "object_class": "car",
        "trigger": {"type": "object_entered_zone", "zone_id": zone["id"]},
        "action": {"type": "create_notification", "config": {"title": "Bay"}},
        "cooldown_seconds": 5,
    }).json()
    assert wf["enabled"] is False  # disabled draft until reviewed + approved

    # Processing WITHOUT enabling arms nothing -> no events.
    assert _process_and_wait(client, source["id"]) == "processed"
    assert client.get(f"/api/events?source_id={source['id']}").json() == []

    # Explicit activation (the reviewed approval) arms the geometric rule.
    assert client.post(f"/api/workflows/{wf['id']}/enable").json()["enabled"] is True
    assert _process_and_wait(client, source["id"]) == "processed"
    events = client.get(f"/api/events?source_id={source['id']}").json()
    enter = [e for e in events if e["type"] == "object_entered_zone"]
    assert enter and enter[0]["action_status"] == "done"
    assert enter[0]["visual_status"] is None  # geometric: no routing


# -- stats report disabled without instantiating the router/Teacher ------------

def test_vlm_stats_disabled_without_instantiating_router(client, monkeypatch):
    def boom():
        raise AssertionError("get_router() must not be called in demo-safe stats")

    # main.py binds get_router at import; patch the name it actually calls.
    monkeypatch.setattr(main_module, "get_router", boom)

    stats = client.get("/api/vlm/stats")
    assert stats.status_code == 200
    assert stats.json()["live"]["status"] == "disabled_demo_safe"
    assert stats.json()["live"]["teacher_available"] is False

    metrics = client.get("/api/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["vlm"]["status"] == "disabled_demo_safe"
