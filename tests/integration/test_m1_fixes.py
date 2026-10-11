"""Integration tests for M1 defect fixes M1-003/004/005/007/008/009.

All use clearly-disclosed test-only stub detectors (not real inference).
"""

import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import pipeline
from app.db import VideoSource, Workflow, get_session
from app.main import app
from visionflow_vision.detector import BaseDetectorTracker
from visionflow_vision.events import TrackedObject


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "fix.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(60):  # 2 seconds @30fps -> 10 sampled frames @5fps
        writer.write(np.full((240, 320, 3), 40, dtype=np.uint8))
    writer.release()
    return path


def _upload(client, video_path, name):
    with open(video_path, "rb") as f:
        r = client.post("/api/sources/upload", files={"file": (name, f, "video/mp4")})
    assert r.status_code == 201, r.text
    return r.json()


def _right_half_zone(client, source_id):
    return client.post(
        "/api/zones",
        json={
            "source_id": source_id,
            "name": "bay",
            "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        },
    ).json()


def _enter_workflow(client, source_id, zone_id):
    wf = client.post(
        "/api/workflows",
        json={
            "name": "car enter",
            "source_id": source_id,
            "object_class": "car",
            "trigger": {"type": "object_entered_zone", "zone_id": zone_id},
            "action": {"type": "create_notification", "config": {"title": "bay"}},
            "cooldown_seconds": 1,
        },
    ).json()
    client.post(f"/api/workflows/{wf['id']}/enable")
    return wf


def _wait_status(client, source_id, target=("processed", "failed"), timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/sources/{source_id}").json()["status"]
        if status in target:
            return status
        time.sleep(0.1)
    return client.get(f"/api/sources/{source_id}").json()["status"]


def _wait_not_running(source_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pipeline.is_running(source_id):
            return True
        time.sleep(0.05)
    return not pipeline.is_running(source_id)


def _use_detector(det):
    pipeline.detector_factory = lambda: det
    pipeline._detector = None


def _reset_detector():
    pipeline.detector_factory = None
    pipeline._detector = None


# -------------------------------------------------------------- M1-003


class GatedCarDetector(BaseDetectorTracker):
    """Blocks on the first frame so the test can disable mid-run, then always
    reports one car inside the right-half zone."""

    def __init__(self):
        self.last_inference_ms = 0.1
        self.reached = threading.Event()
        self.release = threading.Event()
        self._first = True

    def reset(self):
        self._first = True

    def track_frame(self, frame, timestamp):
        if self._first:
            self._first = False
            self.reached.set()
            self.release.wait(timeout=10)
        h, w = frame.shape[:2]
        cx = 0.8 * w
        return [TrackedObject(1, "car", 0.95, cx - 10, 100, cx + 10, 140)]


def test_disable_during_processing_revokes_actions(client, video_path):
    det = GatedCarDetector()
    _use_detector(det)
    try:
        source = _upload(client, video_path, "m1003.mp4")
        zone = _right_half_zone(client, source["id"])
        wf = _enter_workflow(client, source["id"], zone["id"])

        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert det.reached.wait(timeout=10), "worker never reached the detector"

        # Disable while the worker is paused before its first result.
        r = client.post(f"/api/workflows/{wf['id']}/disable")
        assert r.status_code == 200 and r.json()["enabled"] is False

        det.release.set()
        assert _wait_status(client, source["id"]) == "processed"

        events = client.get("/api/events", params={"source_id": source["id"]}).json()
        assert events == [], f"disabled workflow still produced events: {events}"
    finally:
        det.release.set()
        _reset_detector()


# -------------------------------------------------------------- M1-004


class SerialProbeDetector(BaseDetectorTracker):
    """Shared across sources; records reset/track ordering and blocks on the
    first track call so the test can prove sessions are serialized, not
    interleaved. Reports no objects (ordering is what matters here)."""

    def __init__(self):
        self.last_inference_ms = 0.1
        self.log: list[str] = []
        self._lock = threading.Lock()
        self.reached = threading.Event()
        self.release = threading.Event()
        self._block_next = True

    def reset(self):
        with self._lock:
            self.log.append("reset")

    def track_frame(self, frame, timestamp):
        with self._lock:
            self.log.append("t")
            first = self._block_next
            if first:
                self._block_next = False
        if first:
            self.reached.set()
            self.release.wait(timeout=10)
        return []


def test_concurrent_sources_are_serialized(client, video_path):
    det = SerialProbeDetector()
    _use_detector(det)
    try:
        a = _upload(client, video_path, "m1004a.mp4")
        b = _upload(client, video_path, "m1004b.mp4")

        assert client.post(f"/api/sources/{a['id']}/process").status_code == 202
        assert det.reached.wait(timeout=10), "source A never reached the detector"

        # Source A holds the session lock (paused in track_frame). Start B.
        assert client.post(f"/api/sources/{b['id']}/process").status_code == 202

        # Give B a chance to (incorrectly) reset the shared detector.
        time.sleep(0.5)
        assert det.log.count("reset") == 1, (
            f"B reset the shared tracker while A was active: {det.log}"
        )

        det.release.set()
        assert _wait_status(client, a["id"]) == "processed"
        assert _wait_status(client, b["id"]) == "processed"

        # Exactly two resets total, one per isolated session.
        assert det.log.count("reset") == 2, det.log
    finally:
        det.release.set()
        _reset_detector()


# -------------------------------------------------------------- M1-005


class NoopDetector(BaseDetectorTracker):
    def __init__(self):
        self.last_inference_ms = 0.1

    def reset(self):
        pass

    def track_frame(self, frame, timestamp):
        return []


def test_interrupted_processing_recovered_and_retryable(client, video_path):
    det = NoopDetector()
    _use_detector(det)
    try:
        source = _upload(client, video_path, "m1005.mp4")

        # Simulate an interrupted run: a persisted 'processing' row, no worker.
        session = get_session()
        try:
            row = session.get(VideoSource, source["id"])
            row.status = "processing"
            session.commit()
        finally:
            session.close()

        # Startup recovery (as runs on restart).
        recovered = pipeline.recover_interrupted_sources()
        assert recovered >= 1

        got = client.get(f"/api/sources/{source['id']}").json()
        assert got["status"] == "failed"
        assert "interrupt" in (got["error"] or "").lower()

        # Retry is now permitted (not 409).
        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert _wait_status(client, source["id"]) == "processed"
    finally:
        _reset_detector()


# -------------------------------------------------------------- M1-003 (two events / one frame)


class TwoCarDetector(BaseDetectorTracker):
    """Reports two distinct cars, both inside the right-half zone every frame, so
    both transition inside and emit enter events on the same sampled frame."""

    def __init__(self):
        self.last_inference_ms = 0.1

    def reset(self):
        pass

    def track_frame(self, frame, timestamp):
        h, w = frame.shape[:2]
        cx = 0.8 * w
        return [
            TrackedObject(1, "car", 0.95, cx - 10, 100, cx + 10, 140),
            TrackedObject(2, "car", 0.95, cx - 10, 150, cx + 10, 190),
        ]


def test_disable_between_events_in_one_frame_suppresses_remainder(client, video_path):
    det = TwoCarDetector()
    _use_detector(det)
    original = pipeline._dispatch_action
    state = {"count": 0, "wf_id": None}

    def wrapped(session, wf, event):
        original(session, wf, event)
        state["count"] += 1
        if state["count"] == 1:
            # Disable (committed, like the API does) immediately after the first
            # action commits, before the second event of the same frame.
            s = get_session()
            try:
                row = s.get(Workflow, state["wf_id"])
                row.enabled = False
                s.commit()
            finally:
                s.close()

    try:
        source = _upload(client, video_path, "m1003b.mp4")
        zone = _right_half_zone(client, source["id"])
        wf = _enter_workflow(client, source["id"], zone["id"])
        state["wf_id"] = wf["id"]
        pipeline._dispatch_action = wrapped

        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert _wait_status(client, source["id"]) == "processed"

        events = client.get("/api/events", params={"source_id": source["id"]}).json()
        # First event (and its action) kept; the second is suppressed by disable.
        assert len(events) == 1, f"expected exactly one event, got {events}"
        assert events[0]["action_status"] == "done"
        assert state["count"] == 1, "a second action was dispatched after disable"
    finally:
        pipeline._dispatch_action = original
        _reset_detector()


# -------------------------------------------------------------- M1-007 (resume undispatched action)


class SingleCarDetector(BaseDetectorTracker):
    def __init__(self):
        self.last_inference_ms = 0.1

    def reset(self):
        pass

    def track_frame(self, frame, timestamp):
        h, w = frame.shape[:2]
        cx = 0.8 * w
        return [TrackedObject(1, "car", 0.95, cx - 10, 100, cx + 10, 140)]


def test_interrupted_before_dispatch_resumes_on_retry(client, video_path):
    det = SingleCarDetector()
    _use_detector(det)
    original = pipeline._dispatch_action
    interrupt = {"armed": True}

    def crashing_dispatch(session, wf, event):
        if interrupt["armed"]:
            interrupt["armed"] = False
            raise SystemExit("simulated crash after event+evidence commit")
        original(session, wf, event)

    try:
        source = _upload(client, video_path, "m1007b.mp4")
        zone = _right_half_zone(client, source["id"])
        wf = _enter_workflow(client, source["id"], zone["id"])

        pipeline._dispatch_action = crashing_dispatch
        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert _wait_not_running(source["id"]), "worker thread did not exit"

        # The event + evidence committed, but the action never dispatched.
        events = client.get("/api/events", params={"source_id": source["id"]}).json()
        assert len(events) == 1, events
        ev_id = events[0]["id"]
        assert events[0]["action_status"] == "pending"
        assert client.get(f"/api/events/{ev_id}/actions").json() == []

        # Restore dispatch, recover the interrupted source, reprocess.
        pipeline._dispatch_action = original
        assert pipeline.recover_interrupted_sources() >= 1
        assert client.get(f"/api/sources/{source['id']}").json()["status"] == "failed"

        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert _wait_status(client, source["id"]) == "processed"

        # No duplicate event; the previously pending action now completes once.
        events2 = client.get("/api/events", params={"source_id": source["id"]}).json()
        assert len(events2) == 1, f"reprocess duplicated events: {events2}"
        assert events2[0]["action_status"] == "done"
        acts = client.get(f"/api/events/{ev_id}/actions").json()
        assert len(acts) == 1 and acts[0]["status"] == "success", acts
    finally:
        pipeline._dispatch_action = original
        _reset_detector()


# -------------------------------------------------------------- M1-008 (invalid notification title)


def test_invalid_notification_title_rejected(client, video_path):
    source = _upload(client, video_path, "m1008.mp4")
    zone = _right_half_zone(client, source["id"])
    r = client.post(
        "/api/workflows",
        json={
            "name": "bad title",
            "source_id": source["id"],
            "object_class": "car",
            "trigger": {"type": "object_entered_zone", "zone_id": zone["id"]},
            "action": {
                "type": "create_notification",
                "config": {"title": {"invalid": "object"}},
            },
        },
    )
    assert r.status_code == 422, r.text


# -------------------------------------------------------------- M1-009 (evidence write failure)


def test_evidence_write_failure_fails_source_and_rolls_back(client, video_path, monkeypatch):
    det = SingleCarDetector()
    _use_detector(det)
    try:
        source = _upload(client, video_path, "m1009.mp4")
        zone = _right_half_zone(client, source["id"])
        _enter_workflow(client, source["id"], zone["id"])

        # Simulate an unwritable/full evidence destination.
        monkeypatch.setattr(pipeline.cv2, "imwrite", lambda *a, **k: False)

        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert _wait_status(client, source["id"]) == "failed"

        got = client.get(f"/api/sources/{source['id']}").json()
        assert "evidence" in (got["error"] or "").lower(), got
        # The incomplete event was rolled back — no dangling event/evidence URL.
        events = client.get("/api/events", params={"source_id": source["id"]}).json()
        assert events == [], f"event persisted despite evidence failure: {events}"
    finally:
        _reset_detector()


# -------------------------------------------------------------- M1-010 (disable completes before dispatch)


def test_disable_before_dispatch_suppresses_queued_action(client, video_path, monkeypatch):
    """The exact M1-010 race: the event is committed, dispatch is paused, the
    workflow is disabled through the API (returns enabled=false), then dispatch
    is released. The queued action must be suppressed (cancelled) explicitly —
    no notification may be created."""
    det = SingleCarDetector()
    _use_detector(det)
    original = pipeline._dispatch_action
    ready, release = threading.Event(), threading.Event()

    def gated(session, wf, event):
        ready.set()
        assert release.wait(10), "dispatch was never released"
        original(session, wf, event)

    try:
        source = _upload(client, video_path, "m1010.mp4")
        zone = _right_half_zone(client, source["id"])
        wf = _enter_workflow(client, source["id"], zone["id"])
        monkeypatch.setattr(pipeline, "_dispatch_action", gated)

        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert ready.wait(10), "worker never reached dispatch"

        # Disable completes (committed) before dispatch is released.
        r = client.post(f"/api/workflows/{wf['id']}/disable")
        assert r.status_code == 200 and r.json()["enabled"] is False
        release.set()

        assert _wait_status(client, source["id"]) == "processed"

        events = client.get("/api/events", params={"source_id": source["id"]}).json()
        assert len(events) == 1, events
        ev_id = events[0]["id"]
        assert events[0]["action_status"] == "cancelled", events[0]

        acts = client.get(f"/api/events/{ev_id}/actions").json()
        assert len(acts) == 1 and acts[0]["status"] == "cancelled", acts
        # The suppressed action created no notification side effect.
        notifs = client.get("/api/notifications").json()
        assert not any(n["event_id"] == ev_id for n in notifs), notifs
    finally:
        release.set()
        _reset_detector()


# -------------------------------------------------------------- M1-011 / M1-012 (webhook removed)


def test_legacy_webhook_action_rejected_no_side_effect(client, video_path, monkeypatch):
    """External webhook delivery was removed. A legacy workflow whose persisted
    DSL still names send_webhook must be rejected at dispatch with no network
    lookup/delivery, so the crash-recovery duplicate-delivery (M1-011) and
    DNS-rebinding (M1-012) scenarios cannot occur. We inject the legacy DSL
    directly because the API now refuses to create such a workflow."""
    import socket

    det = SingleCarDetector()
    _use_detector(det)

    def exploding_lookup(*args, **kwargs):
        raise AssertionError("no DNS lookup may occur for the removed webhook action")

    try:
        source = _upload(client, video_path, "m1011.mp4")
        zone = _right_half_zone(client, source["id"])
        wf = _enter_workflow(client, source["id"], zone["id"])

        # Rewrite the stored DSL to a legacy send_webhook action, then enable.
        s = get_session()
        try:
            row = s.get(Workflow, wf["id"])
            dsl = dict(row.dsl)
            dsl["action"] = {"type": "send_webhook", "config": {"url": "http://approved.example:9876/hook"}}
            row.dsl = dsl
            row.enabled = True
            s.commit()
        finally:
            s.close()

        monkeypatch.setattr(socket, "getaddrinfo", exploding_lookup)

        assert client.post(f"/api/sources/{source['id']}/process").status_code == 202
        assert _wait_status(client, source["id"]) == "processed"

        events = client.get("/api/events", params={"source_id": source["id"]}).json()
        assert len(events) == 1, events
        ev_id = events[0]["id"]
        acts = client.get(f"/api/events/{ev_id}/actions").json()
        assert len(acts) == 1 and acts[0]["status"] == "rejected", acts
        assert "not supported" in (acts[0]["error"] or "").lower()
        notifs = client.get("/api/notifications").json()
        assert not any(n["event_id"] == ev_id for n in notifs), notifs
    finally:
        _reset_detector()


# -------------------------------------------------------------- M1-013 (truncated video)


def test_truncated_video_reported_failed(client, tmp_path):
    """A faststart MP4 truncated to 65% of its bytes decodes far fewer frames
    than its container metadata reports. Processing must surface a failed source
    with an explicit shortfall message, not silently report 'processed'."""
    import subprocess

    det = SingleCarDetector()
    _use_detector(det)
    try:
        # Build a complete 60-frame source, remux with faststart (reliable
        # leading metadata), then truncate the bytes.
        full = tmp_path / "full.mp4"
        writer = cv2.VideoWriter(str(full), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
        for i in range(60):
            writer.write(np.full((240, 320, 3), i % 256, dtype=np.uint8))
        writer.release()

        faststart = tmp_path / "faststart.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(full),
             "-c", "copy", "-movflags", "+faststart", str(faststart)],
            check=True,
        )

        # A complete faststart video must still process cleanly (no false fail).
        ok_src = _upload(client, faststart, "faststart_ok.mp4")
        assert client.post(f"/api/sources/{ok_src['id']}/process").status_code == 202
        assert _wait_status(client, ok_src["id"]) == "processed"

        truncated = faststart.read_bytes()
        truncated = truncated[: int(len(truncated) * 0.65)]
        cut = tmp_path / "cut.mp4"
        cut.write_bytes(truncated)

        bad = client.post(
            "/api/sources/upload",
            files={"file": ("cut.mp4", truncated, "video/mp4")},
        )
        assert bad.status_code == 201, bad.text
        bad_id = bad.json()["id"]
        assert client.post(f"/api/sources/{bad_id}/process").status_code == 202
        assert _wait_status(client, bad_id) == "failed"

        got = client.get(f"/api/sources/{bad_id}").json()
        assert got["frame_count"] == 60, got
        assert "incomplete" in (got["error"] or "").lower(), got
    finally:
        _reset_detector()
