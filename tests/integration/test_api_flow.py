"""End-to-end API flow test: upload -> zone -> workflow -> process -> event -> action.

Uses a deterministic STUB detector (clearly test-only) so the flow test is
fast and reproducible. Real-YOLO end-to-end verification lives in
scripts/smoke_e2e.py and is run before each handoff.
"""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import pipeline
from app.main import app
from visionflow_vision.detector import BaseDetectorTracker
from visionflow_vision.events import TrackedObject


class ScriptedDetector(BaseDetectorTracker):
    """Moves one 'car' track left-to-right across a 320x240 frame.

    Anchor crosses into the right-half zone around the middle of the video.
    TEST STUB ONLY — production uses YoloByteTrackDetector.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.last_inference_ms = 0.1

    def reset(self) -> None:
        self.calls = 0

    def track_frame(self, frame, timestamp: float) -> list[TrackedObject]:
        h, w = frame.shape[:2]
        cx = (self.calls / 8.0) * w  # sweeps full width over ~8 sampled frames
        self.calls += 1
        if cx < 10 or cx > w - 10:
            return []
        return [
            TrackedObject(
                track_id=1, class_name="car", confidence=0.95,
                x1=cx - 10, y1=100, x2=cx + 10, y2=140,
            )
        ]


@pytest.fixture(scope="module")
def client():
    pipeline.detector_factory = ScriptedDetector
    pipeline._detector = None
    with TestClient(app) as c:
        yield c
    pipeline.detector_factory = None
    pipeline._detector = None


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    """2-second 320x240 synthetic mp4 (content irrelevant to the stub)."""
    path = tmp_path_factory.mktemp("video") / "test.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
    )
    for i in range(60):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (int(i / 60 * 320), 120), 15, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_pipeline_flow(client, video_path):
    # 1. upload
    with open(video_path, "rb") as f:
        r = client.post(
            "/api/sources/upload", files={"file": ("test.mp4", f, "video/mp4")}
        )
    assert r.status_code == 201, r.text
    source = r.json()
    assert source["status"] == "uploaded"
    assert source["width"] == 320

    # snapshot available for zone drawing
    r = client.get(f"/api/sources/{source['id']}/snapshot")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"

    # 2. zone over the right half of the frame
    r = client.post(
        "/api/zones",
        json={
            "source_id": source["id"],
            "name": "Service Bay A",
            "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        },
    )
    assert r.status_code == 201, r.text
    zone = r.json()

    # 3. workflow: car enters zone -> notification
    r = client.post(
        "/api/workflows",
        json={
            "name": "Car entered bay",
            "source_id": source["id"],
            "object_class": "car",
            "trigger": {"type": "object_entered_zone", "zone_id": zone["id"]},
            "action": {"type": "create_notification", "config": {"title": "Bay alert"}},
            "cooldown_seconds": 5,
        },
    )
    assert r.status_code == 201, r.text
    wf = r.json()
    assert wf["enabled"] is False  # requires explicit enable

    r = client.post(f"/api/workflows/{wf['id']}/enable")
    assert r.status_code == 200
    assert r.json()["enabled"] is True

    # 4. process
    r = client.post(f"/api/sources/{source['id']}/process")
    assert r.status_code == 202

    # 5. wait for background processing
    deadline = time.time() + 30
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/sources/{source['id']}").json()["status"]
        if status in ("processed", "failed"):
            break
        time.sleep(0.3)
    assert status == "processed", f"final status: {status}"

    # 6. event with evidence exists
    events = client.get("/api/events", params={"source_id": source["id"]}).json()
    enter_events = [e for e in events if e["type"] == "object_entered_zone"]
    assert len(enter_events) == 1, events
    ev = enter_events[0]
    assert ev["class_name"] == "car"
    assert ev["evidence_url"]
    assert ev["action_status"] == "done"

    r = client.get(ev["evidence_url"])
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"

    # 7. action executed: notification persisted
    actions = client.get(f"/api/events/{ev['id']}/actions").json()
    assert [a["status"] for a in actions] == ["success"]
    notifications = client.get("/api/notifications").json()
    assert any(n["event_id"] == ev["id"] and n["title"] == "Bay alert" for n in notifications)


def test_webhook_action_rejected_at_creation(client, video_path):
    # External webhook delivery was removed (M1-011/M1-012). Creating a workflow
    # with send_webhook is now rejected at the API boundary (422), so no
    # outbound-delivery code path can ever be reached.
    with open(video_path, "rb") as f:
        source = client.post(
            "/api/sources/upload", files={"file": ("test2.mp4", f, "video/mp4")}
        ).json()
    zone = client.post(
        "/api/zones",
        json={
            "source_id": source["id"],
            "name": "Zone",
            "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        },
    ).json()
    r = client.post(
        "/api/workflows",
        json={
            "name": "Webhook rule",
            "source_id": source["id"],
            "object_class": "car",
            "trigger": {"type": "object_entered_zone", "zone_id": zone["id"]},
            "action": {"type": "send_webhook", "config": {"url": "http://evil.example/hook"}},
            "cooldown_seconds": 5,
        },
    )
    assert r.status_code == 422, r.text


def test_upload_rejects_non_mp4(client):
    r = client.post(
        "/api/sources/upload", files={"file": ("evil.exe", b"MZ....", "application/exe")}
    )
    assert r.status_code == 400


def test_upload_rejects_garbage_mp4(client):
    r = client.post(
        "/api/sources/upload", files={"file": ("fake.mp4", b"not a video", "video/mp4")}
    )
    assert r.status_code == 400


def test_workflow_zone_source_mismatch_rejected(client, video_path):
    with open(video_path, "rb") as f:
        source = client.post(
            "/api/sources/upload", files={"file": ("t3.mp4", f, "video/mp4")}
        ).json()
    r = client.post(
        "/api/workflows",
        json={
            "name": "Bad zone ref",
            "source_id": source["id"],
            "object_class": "car",
            "trigger": {"type": "object_entered_zone", "zone_id": 99999},
            "action": {"type": "create_notification", "config": {}},
        },
    )
    assert r.status_code == 422
