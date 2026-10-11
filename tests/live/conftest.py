"""Fixtures for the live-app tests.

The demo PIN is NEVER hardcoded anywhere (source, tests, or logs). Each test run
generates a throwaway random PIN at runtime and installs it on the config module,
so these tests exercise real PIN auth without embedding the production value.
"""

import secrets
from types import SimpleNamespace

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.live import auth, config, planner, rooms, teacher, video, worker
from visionflow_vision.detector import BaseDetectorTracker
from visionflow_vision.events import TrackedObject


@pytest.fixture
def test_pin(monkeypatch):
    pin = "qa-" + secrets.token_urlsafe(8)  # runtime-only test secret
    monkeypatch.setattr(config, "DEMO_PIN", pin)
    return pin


class ScriptedDetector(BaseDetectorTracker):
    """Deterministic QA stub — NEVER a phone-inference claim. Keeps one 'car' and
    one 'bottle' track centered so a full-frame presence rule fires immediately."""

    def __init__(self):
        self.last_inference_ms = 0.2

    def reset(self):
        pass

    def track_frame(self, frame, timestamp):
        h, w = frame.shape[:2]
        return [
            TrackedObject(track_id=1, class_name="car", confidence=0.9,
                          x1=w*0.4, y1=h*0.4, x2=w*0.6, y2=h*0.9),
            TrackedObject(track_id=2, class_name="bottle", confidence=0.8,
                          x1=w*0.1, y1=h*0.1, x2=w*0.2, y2=h*0.3),
        ]


@pytest.fixture
def jpeg_frame():
    img = np.full((240, 320, 3), 60, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


@pytest.fixture
def qa_mp4(tmp_path):
    """A small, freshly-generated QA_ONLY clip (disclosed synthetic fixture — not
    real footage and never a phone-inference claim). 15 frames at 10 fps."""
    path = tmp_path / "qa_only_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
    assert vw.isOpened()
    for i in range(15):
        frame = np.full((240, 320, 3), 40 + i, dtype=np.uint8)
        vw.write(frame)
    vw.release()
    assert path.exists() and path.stat().st_size > 0
    return path.read_bytes()


@pytest.fixture
def client(test_pin, monkeypatch, tmp_path_factory):
    # Reset all in-memory state between tests.
    auth._reset_for_tests()
    rooms._reset_for_tests()
    worker._reset_for_tests()
    video._reset_for_tests()
    planner._room_attempts.clear()
    planner._room_last_attempt.clear()
    teacher._reset_for_tests()
    planner.transport = None
    teacher.transport = None
    worker.detector_factory = ScriptedDetector
    # Allow Secure cookies to be stored by the test client over http.
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    # Legacy WebSocket/camera hardening tests exercise the (now non-public) camera
    # ingest path; enable it only for the in-process test app. The public default
    # is OFF — a dedicated test asserts the disabled surface. A fresh, disposable
    # upload dir isolates QA video writes.
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", True)
    vdir = tmp_path_factory.mktemp("live_uploads")
    monkeypatch.setattr(rooms, "LIVE_VIDEO_DIR", vdir)
    # Neutralize planner pacing by default so rapid test calls are not throttled;
    # the planner budget/cooldown tests override these explicitly.
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "PLANNER_ROOM_BUDGET", 100)
    from app.live.app import app
    with TestClient(app) as c:
        # Stop the background worker loop so tests drive processing deterministically.
        worker.stop()
        yield c
    worker.detector_factory = None
    worker._reset_for_tests()


@pytest.fixture
def flow():
    """Helpers bound to any client (supports multiple independent contexts)."""
    def _login(c, role, pin):
        return c.post("/api/live/login", json={"role": role, "pin": pin})

    def _csrf(c):
        return c.cookies.get(config.CSRF_COOKIE_NAME)

    def _post(c, path, body):
        return c.post(path, json=body, headers={"X-CSRF-Token": _csrf(c)})

    def _approve(c, path, body):
        draft = _post(c, path.replace("/task/activate", "/task/validate"), body)
        assert draft.status_code == 200, draft.text
        return _post(c, path, draft.json())

    def _ws_path(c, path):
        # WebSocket upgrades carry the double-submit CSRF token as a query param.
        sep = "&" if "?" in path else "?"
        return f"{path}{sep}csrf={_csrf(c)}"

    return SimpleNamespace(login=_login, csrf=_csrf, post=_post, approve=_approve, ws_path=_ws_path)
