"""M2 integration: event-routed visual reasoning + NL compile + boundaries.

Uses a deterministic STUB detector and a STUB teacher (both clearly test-only).
This verifies the wiring and the honesty guarantees — it does NOT verify real
visual inference, which requires a configured provider (documented blocker).

Covered acceptance:
- M2-AC1: /api/workflows/compile produces a strict editable draft; ambiguous
  text is rejected; the draft saves disabled and needs explicit enable.
- M2-AC2: a geometric-only workflow makes ZERO VLM calls; a visual workflow
  routes the candidate ROI to the teacher.
- M2-AC3: a confirmed teacher result fires the action; unknown/error/missing
  credentials do NOT fire an action and remain visible.
- M2-AC4/AC5: router counters/VlmCall rows reflect calls; stats endpoints work.
"""

import threading
import time

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, pipeline
from app.main import app


@pytest.fixture(autouse=True)
def _experimental_m2_mode():
    """These are the historical M2 experimental tests: they exercise NL compile
    and Teacher-routed visual predicates, which the demo-safe backend disables by
    default. Opt explicitly into the pre-demo behavior (DEMO_SAFE=False) without
    weakening any original assertion."""
    prev = config.DEMO_SAFE
    config.DEMO_SAFE = False
    try:
        yield
    finally:
        config.DEMO_SAFE = prev
from app.teacher import OpenAICompatibleTeacher, TeacherAdapter, TeacherResult
from app.vlm_router import VlmRouter, set_router
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


class StubTeacher(TeacherAdapter):
    provider_name = "stub"
    model = "stub-vlm"

    def __init__(self, status="ok", label="yes"):
        self.status = status
        self.label = label
        self.calls = 0

    def classify(self, image_bytes, prompt, labels, on_attempt=None):
        self.calls += 1
        return TeacherResult(status=self.status, label=self.label, confidence=0.9,
                             model=self.model, latency_ms=1.0, usage={"total_tokens": 7})


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
    path = tmp_path_factory.mktemp("m2video") / "t.mp4"
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


# -- M2-AC1: NL compile -------------------------------------------------------

def test_compile_produces_editable_draft(client, video_path):
    source, zone = _setup_source_zone(client, video_path, "c1.mp4")
    r = client.post("/api/workflows/compile", json={
        "source_id": source["id"],
        "text": "alert me when a car enters Service Bay A",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["draft"]["trigger"]["type"] == "object_entered_zone"
    assert body["draft"]["trigger"]["zone_id"] == zone["id"]
    # draft is not persisted by compile
    assert client.get(f"/api/workflows?source_id={source['id']}").json() == []


def test_compile_rejects_ambiguous(client, video_path):
    source, _ = _setup_source_zone(client, video_path, "c2.mp4")
    r = client.post("/api/workflows/compile", json={
        "source_id": source["id"], "text": "do something with a vehicle",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["errors"]
    assert body["draft"] is None


# -- M2-AC2: geometric path makes zero VLM calls ------------------------------

def test_geometric_workflow_makes_zero_vlm_calls(client, video_path):
    stub = StubTeacher()
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "g1.mp4")
    wf = client.post("/api/workflows", json={
        "name": "Car entered", "source_id": source["id"], "object_class": "car",
        "trigger": {"type": "object_entered_zone", "zone_id": zone["id"]},
        "action": {"type": "create_notification", "config": {"title": "Bay"}},
        "cooldown_seconds": 5,
    }).json()
    client.post(f"/api/workflows/{wf['id']}/enable")
    assert _process_and_wait(client, source["id"]) == "processed"

    events = client.get(f"/api/events?source_id={source['id']}").json()
    enter = [e for e in events if e["type"] == "object_entered_zone"]
    assert enter and enter[0]["action_status"] == "done"
    assert enter[0]["visual_status"] is None  # geometric: no routing
    assert stub.calls == 0
    assert client.get("/api/vlm/stats").json()["live"]["calls"] == 0


# -- M2-AC2/AC3: visual confirmed fires action --------------------------------

def _make_visual_workflow(client, source, zone):
    return client.post("/api/workflows", json={
        "name": "Car entered + hood open", "source_id": source["id"], "object_class": "car",
        "trigger": {"type": "object_entered_zone", "zone_id": zone["id"]},
        "action": {"type": "create_notification", "config": {"title": "Hood open in bay"}},
        "visual_predicate": {
            "prompt": "Is the vehicle hood open?",
            "labels": ["yes", "no"], "expected_label": "yes",
            "description": "hood is open",
            "sample_interval_seconds": 0, "cooldown_seconds": 0,
        },
        "cooldown_seconds": 5,
    })


def test_visual_confirmed_fires_action(client, video_path):
    stub = StubTeacher(status="ok", label="yes")
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "v1.mp4")
    wf = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf['id']}/enable")
    assert _process_and_wait(client, source["id"]) == "processed"

    events = client.get(f"/api/events?source_id={source['id']}").json()
    enter = [e for e in events if e["type"] == "object_entered_zone"]
    assert enter, events
    ev = enter[0]
    assert ev["visual_status"] == "ok"
    assert ev["visual_label"] == "yes"
    assert ev["action_status"] == "done"
    assert stub.calls >= 1
    # a notification was created
    notifs = client.get("/api/notifications").json()
    assert any(n["event_id"] == ev["id"] for n in notifs)


def test_visual_negative_does_not_fire_action(client, video_path):
    stub = StubTeacher(status="ok", label="no")  # confirmed-but-not-expected
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "v2.mp4")
    wf = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf['id']}/enable")
    assert _process_and_wait(client, source["id"]) == "processed"

    events = client.get(f"/api/events?source_id={source['id']}").json()
    ev = [e for e in events if e["type"] == "object_entered_zone"][0]
    assert ev["visual_status"] == "ok"
    assert ev["visual_label"] == "no"
    assert ev["action_status"] == "skipped_visual"
    notifs = client.get("/api/notifications").json()
    assert not any(n["event_id"] == ev["id"] for n in notifs)


def test_missing_credentials_cannot_fire_action(client, video_path):
    from app.teacher import UnavailableTeacher

    set_router(VlmRouter(teacher_provider=lambda: UnavailableTeacher(),
                         sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "v3.mp4")
    wf = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf['id']}/enable")
    assert _process_and_wait(client, source["id"]) == "processed"

    ev = [e for e in client.get(f"/api/events?source_id={source['id']}").json()
          if e["type"] == "object_entered_zone"][0]
    assert ev["visual_status"] == "missing_credentials"
    assert ev["action_status"] == "skipped_visual"


def test_teacher_error_cannot_fire_action(client, video_path):
    stub = StubTeacher(status="error", label=None)
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "v4.mp4")
    wf = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf['id']}/enable")
    assert _process_and_wait(client, source["id"]) == "processed"

    ev = [e for e in client.get(f"/api/events?source_id={source['id']}").json()
          if e["type"] == "object_entered_zone"][0]
    assert ev["visual_status"] == "error"
    assert ev["action_status"] == "skipped_visual"


# -- M2-AC5: stats + provenance ----------------------------------------------

def test_vlm_stats_and_calls_recorded(client, video_path):
    stub = StubTeacher(status="ok", label="yes")
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "s1.mp4")
    wf = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf['id']}/enable")
    assert _process_and_wait(client, source["id"]) == "processed"

    stats = client.get("/api/vlm/stats").json()
    assert stats["total_decisions"] >= 1
    assert stats["live"]["teacher_provider"] == "stub"
    calls = client.get("/api/vlm/calls").json()
    assert calls and calls[0]["roi_hash"]
    assert calls[0]["model"] == "stub-vlm"


# -- M2-006: recover a confirmed-but-undispatched visual action ----------------

def test_resume_confirmed_visual_event_dispatches_once(client, video_path):
    from app.db import ActionExecution, TrackEvent, Workflow, get_session

    stub = StubTeacher(status="ok", label="yes")
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "r1.mp4")
    wf_json = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf_json['id']}/enable")

    session = get_session()
    try:
        wf = session.get(Workflow, wf_json["id"])
        # Simulate an interrupt AFTER the positive visual result was committed and
        # BEFORE action dispatch: a pending event carrying a valid confirmation.
        ev = TrackEvent(
            source_id=source["id"], workflow_id=wf.id, track_id=1,
            type="object_entered_zone", class_name="car", video_timestamp=1.0,
            confidence=0.9, detail="interrupted", dedupe_key="resume-confirmed-1",
            action_status="pending", visual_status="ok", visual_label="yes",
            visual_confidence=0.9,
            visual_routing={"confirmed": True, "expected_label": "yes"},
        )
        session.add(ev)
        session.commit()
        ev_id = ev.id

        # Recovery resumes the confirmed action through the guarded dispatch path.
        assert pipeline._resume_pending_event(session, wf, ev) is True
        session.refresh(ev)
        assert ev.action_status == "done"
        assert (
            session.query(ActionExecution)
            .filter(ActionExecution.event_id == ev_id)
            .count()
            == 1
        )
        # A confirmed visual resume fires no new Teacher call.
        assert stub.calls == 0

        # Idempotent: a second recovery does not re-dispatch.
        assert pipeline._resume_pending_event(session, wf, ev) is True
        assert (
            session.query(ActionExecution)
            .filter(ActionExecution.event_id == ev_id)
            .count()
            == 1
        )
    finally:
        session.close()


def test_resume_unconfirmed_visual_event_marked_interrupted(client, video_path):
    from app.db import ActionExecution, TrackEvent, Workflow, get_session

    stub = StubTeacher(status="ok", label="yes")
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "r2.mp4")
    wf_json = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf_json['id']}/enable")

    session = get_session()
    try:
        wf = session.get(Workflow, wf_json["id"])
        # Pending event that was NEVER validly confirmed (routing never completed).
        ev = TrackEvent(
            source_id=source["id"], workflow_id=wf.id, track_id=1,
            type="object_entered_zone", class_name="car", video_timestamp=1.0,
            confidence=0.9, detail="interrupted", dedupe_key="resume-unconfirmed-1",
            action_status="pending",
        )
        session.add(ev)
        session.commit()
        ev_id = ev.id

        assert pipeline._resume_pending_event(session, wf, ev) is True
        session.refresh(ev)
        assert ev.action_status == "interrupted_unverified"
        assert ev.visual_status == "interrupted"
        # No action may fire for an unverified candidate.
        assert (
            session.query(ActionExecution)
            .filter(ActionExecution.event_id == ev_id)
            .count()
            == 0
        )
        assert stub.calls == 0
    finally:
        session.close()


# -- M2-007: a committed disable prevents provider admission / image disclosure -

def test_disable_before_route_cancels_without_teacher_call(client, video_path):
    from app.db import TrackEvent, Workflow, get_session
    from visionflow_vision.events import TrackedObject

    stub = StubTeacher(status="ok", label="yes")
    set_router(VlmRouter(teacher_provider=lambda: stub, sample_interval_s=0, cooldown_s=0))
    source, zone = _setup_source_zone(client, video_path, "d1.mp4")
    wf_json = _make_visual_workflow(client, source, zone).json()
    client.post(f"/api/workflows/{wf_json['id']}/enable")
    # Disable commits under dispatch_admission_lock and returns 200 before routing.
    assert client.post(f"/api/workflows/{wf_json['id']}/disable").status_code == 200

    session = get_session()
    try:
        wf = session.get(Workflow, wf_json["id"])
        ev = TrackEvent(
            source_id=source["id"], workflow_id=wf.id, track_id=1,
            type="object_entered_zone", class_name="car", video_timestamp=1.0,
            confidence=0.9, detail="candidate", dedupe_key="disable-route-1",
            action_status="pending",
        )
        session.add(ev)
        session.commit()

        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        objs = [TrackedObject(track_id=1, class_name="car", confidence=0.9,
                              x1=10, y1=10, x2=60, y2=60)]
        confirmed = pipeline._route_visual(session, wf, ev, frame, objs)

        assert confirmed is False
        session.refresh(ev)
        assert ev.visual_status == "cancelled"
        assert ev.action_status == "cancelled"
        # The disable prevented the provider request entirely: no image disclosed.
        assert stub.calls == 0
    finally:
        session.close()


# -- M2-007 (exceptional recovery): disable is serialized with the ACTUAL provider
# admission. The reviewed race paused AFTER the authoritative approval read: a
# disable returned 200, then one HTTP POST still started. With the fix the
# approval read and admission are one boundary under dispatch_admission_lock, so a
# disable attempted during that boundary BLOCKS and cannot admit a late request.

def test_disable_cannot_interleave_during_transport_admission(client, video_path, monkeypatch):
    from app.db import TrackEvent, Workflow, get_session

    source, zone = _setup_source_zone(client, video_path, "race1.mp4")
    wf_json = _make_visual_workflow(client, source, zone).json()
    assert client.post(f"/api/workflows/{wf_json['id']}/enable").status_code == 200

    teacher = OpenAICompatibleTeacher(
        "https://qa.invalid/v1", "", "qa-vlm", "v1", overall_timeout=5, max_retries=0
    )
    set_router(VlmRouter(teacher_provider=lambda: teacher, sample_interval_s=0, cooldown_s=0))

    posts = []

    def mock_post(url, **kwargs):
        posts.append(time.monotonic())
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"yes","confidence":0.9}'}}]},
        )

    monkeypatch.setattr(httpx, "post", mock_post)

    at_guard = threading.Event()
    release = threading.Event()
    orig_enabled = pipeline._workflow_enabled
    calls = {"n": 0}

    def paused_enabled(wf_id):
        # call 1: route-admission read. call 2: the transport-boundary guard,
        # held under dispatch_admission_lock by the fix. Pause there (holding the
        # lock) so a concurrent disable must serialize behind the admission.
        calls["n"] += 1
        val = orig_enabled(wf_id)
        if calls["n"] == 2:
            at_guard.set()
            assert release.wait(5)
        return val

    monkeypatch.setattr(pipeline, "_workflow_enabled", paused_enabled)

    routed = []

    def worker():
        session = get_session()
        try:
            wf = session.get(Workflow, wf_json["id"])
            ev = TrackEvent(
                source_id=source["id"], workflow_id=wf.id, track_id=1,
                type="object_entered_zone", class_name="car", video_timestamp=1.0,
                confidence=0.9, detail="candidate", dedupe_key="race-admit-1",
                action_status="pending",
            )
            session.add(ev)
            session.commit()
            frame = np.full((240, 320, 3), 40, dtype=np.uint8)
            objs = [TrackedObject(track_id=1, class_name="car", confidence=0.9,
                                  x1=10, y1=10, x2=60, y2=60)]
            routed.append(pipeline._route_visual(session, wf, ev, frame, objs))
        finally:
            session.close()

    t = threading.Thread(target=worker)
    t.start()
    disable_result = {}
    disable_done = threading.Event()
    try:
        assert at_guard.wait(5)

        def do_disable():
            resp = client.post(f"/api/workflows/{wf_json['id']}/disable")
            disable_result["status"] = resp.status_code
            disable_result["at"] = time.monotonic()
            disable_done.set()

        dt = threading.Thread(target=do_disable)
        dt.start()
        # While the admission boundary holds the lock, disable must NOT complete
        # and no HTTP request may have started.
        time.sleep(0.3)
        assert not disable_done.is_set(), "disable completed during admission boundary"
        assert posts == []
        release.set()
        assert disable_done.wait(5)
        dt.join(5)
    finally:
        release.set()
    t.join(5)
    assert not t.is_alive()

    # Exactly one admitted POST, and it was admitted BEFORE the disable could
    # commit/return — a disable can no longer slip in between the approval read
    # and the request to admit a late image disclosure.
    assert len(posts) == 1
    assert disable_result.get("status") == 200
    assert posts[0] < disable_result["at"]
    assert routed == [True]


# -- M2-008 (exceptional recovery): a boolean confidence must not authorize an
# action across the full provider -> router -> event -> local-action boundary.

def test_boolean_confidence_cannot_fire_action(client, video_path, monkeypatch):
    from app.db import Notification, TrackEvent, Workflow, get_session

    source, zone = _setup_source_zone(client, video_path, "bool1.mp4")
    wf_json = _make_visual_workflow(client, source, zone).json()
    assert client.post(f"/api/workflows/{wf_json['id']}/enable").status_code == 200

    teacher = OpenAICompatibleTeacher(
        "https://qa.invalid/v1", "", "qa-vlm", "v1", max_retries=0
    )
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"label":"yes","confidence":true}'}}]},
        ),
    )
    set_router(VlmRouter(teacher_provider=lambda: teacher, sample_interval_s=0, cooldown_s=0))

    session = get_session()
    try:
        wf = session.get(Workflow, wf_json["id"])
        ev = TrackEvent(
            source_id=source["id"], workflow_id=wf.id, track_id=1,
            type="object_entered_zone", class_name="car", video_timestamp=1.0,
            confidence=0.9, detail="candidate", dedupe_key="bool-action-1",
            action_status="pending",
        )
        session.add(ev)
        session.commit()
        ev_id = ev.id

        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        objs = [TrackedObject(track_id=1, class_name="car", confidence=0.9,
                              x1=10, y1=10, x2=60, y2=60)]
        confirmed = pipeline._route_visual(session, wf, ev, frame, objs)
        if confirmed:  # must not happen; dispatch only on a valid confirmation
            pipeline._dispatch_action(session, wf, ev)

        assert confirmed is False
        session.refresh(ev)
        assert ev.visual_status == "invalid_response"
        assert ev.visual_confidence is None
        assert ev.action_status == "skipped_visual"
        assert (
            session.query(Notification).filter(Notification.event_id == ev_id).count() == 0
        )
    finally:
        session.close()
