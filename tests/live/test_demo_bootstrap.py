"""Fast behavior regressions for the isolated, no-PIN demo bootstrap."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.live import auth, config, planner, rooms
from app.live.planner import PlannerResult


TASK = {"name": "Car presence", "object_class": "car", "zone": {"type": "full_frame"},
        "trigger": "presence_in_zone", "action": "notify"}


def bootstrap(client, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "true")
    response = client.post("/api/demo/session", json={})
    assert response.status_code == 200, response.text
    return response.json()


def test_bootstrap_explicit_opt_in_and_strict_empty_body(client, monkeypatch):
    monkeypatch.delenv("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", raising=False)
    assert client.post("/api/demo/session", json={}).status_code == 404
    monkeypatch.setenv("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "true")
    for body in ({"role": "camera"}, {"room_id": "shared"}, [], "", True):
        assert client.post("/api/demo/session", json=body).status_code == 422


def test_each_visitor_owns_isolated_room_and_reload_reuses_it(client, monkeypatch):
    first = bootstrap(client, monkeypatch)
    again = client.post("/api/demo/session", json={}).json()
    assert again == first
    other = TestClient(client.app)
    second = bootstrap(other, monkeypatch)
    assert first["room_id"] != second["room_id"]
    assert first["csrf_token"] != second["csrf_token"]
    assert other.get(f"/api/live/rooms/{first['room_id']}/state").status_code == 403
    assert client.get("/api/live/me").json() == {
        "role": "judge", "room_id": first["room_id"], "owns_room": True}


def test_bootstrap_origin_and_secure_cookie(client, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "true")
    assert client.post("/api/demo/session", json={}, headers={"Origin": "http://evil.invalid"}).status_code == 403
    monkeypatch.setattr(config, "COOKIE_SECURE", True)
    response = client.post("/api/demo/session", json={})
    cookie = response.headers.get("set-cookie", "").lower()
    assert response.status_code == 200
    assert "httponly" in cookie and "secure" in cookie and "samesite=strict" in cookie


def test_bootstrap_room_capacity_cleanup(client, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "true")
    monkeypatch.setattr(config, "MAX_ROOMS", 1)
    bootstrap(client, monkeypatch)
    other = TestClient(client.app)
    assert other.post("/api/demo/session", json={}).status_code == 503
    assert auth.active_session_count() == 1


def test_unified_mode_disables_legacy_public_surfaces(client, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "true")
    for path in ("/camera", "/static/camera.html", "/static/judges.html", "/static/upload.html",
                 "/static/landing.html", "/static/common.js"):
        assert client.get(path).status_code == 404
    for path in ("/api/live/login", "/api/live/rooms", "/api/live/rooms/join"):
        assert client.post(path, json={}).status_code == 404


def test_judge_owner_can_upload_and_access_catalog(client, monkeypatch, qa_mp4, tmp_path):
    b = bootstrap(client, monkeypatch)
    source = tmp_path / "Camera1-960x540-cfr24.mp4"
    source.write_bytes(qa_mp4)
    monkeypatch.setenv("VISIONFLOW_DEMO_VIDEO_DIR", str(tmp_path))
    catalog = client.get("/api/live/video-catalog")
    assert catalog.status_code == 200
    assert [item["id"] for item in catalog.json()["inputs"]] == ["Camera1"]
    headers = {"X-CSRF-Token": b["csrf_token"]}
    selected = client.post(f"/api/live/rooms/{b['room_id']}/video/select", json={"id": "Camera1"}, headers=headers)
    assert selected.status_code == 200, selected.text
    assert rooms.get_room(b["room_id"]).video_path is not None


@pytest.mark.parametrize("body", [{"text": True}, {"text": 1}, {"text": []},
                                   {"text": "watch car", "ignored_condition": "never"}])
def test_plan_outer_schema_rejects_types_and_unknown_fields(client, monkeypatch, flow, body):
    b = bootstrap(client, monkeypatch)
    planner.transport = lambda *args: pytest.fail("invalid input reached planner")
    assert flow.post(client, f"/api/live/rooms/{b['room_id']}/task/plan", body).status_code == 422


@pytest.mark.parametrize("change", ["source", "version", "newer_draft", "expired_session"])
def test_inflight_plan_cannot_bind_to_changed_authority(client, monkeypatch, flow, change):
    b = bootstrap(client, monkeypatch)
    room = rooms.get_room(b["room_id"])
    room.video_sha256 = "a" * 64
    def delayed_result(text, rid):
        if change == "source":
            room.video_sha256 = "b" * 64
        elif change == "version":
            room.task_version += 1
        elif change == "newer_draft":
            room.draft_generation += 1
        else:
            auth._sessions[room.owner_token].expires_at = time.time() - 1
        return PlannerResult(status="ok", proposal={"supported": True, **TASK,
                                                     "explanation": "Car enters region"})
    planner.transport = delayed_result
    response = flow.post(client, f"/api/live/rooms/{room.room_id}/task/plan", {"text": planner.SUPPORTED_COMMAND_EXAMPLES[0]})
    assert response.status_code == (401 if change == "expired_session" else 409)
    from app.live.app import _drafts
    assert room.room_id not in _drafts


def test_validate_outer_unknown_fields_and_approval_hash(client, monkeypatch, flow):
    b = bootstrap(client, monkeypatch)
    rid = b["room_id"]
    path = f"/api/live/rooms/{rid}/task/validate"
    assert flow.post(client, path, {"task": TASK, "ignored": True}).status_code == 422
    draft = flow.post(client, path, {"task": TASK})
    assert draft.status_code == 200
    activated = flow.post(client, path.replace("validate", "activate"), draft.json())
    assert activated.status_code == 200
    assert rooms.get_room(rid).approved_hash == draft.json()["draft_hash"]


def test_evidence_requires_committed_registration_and_expiry_purges(client, monkeypatch, tmp_path):
    b = bootstrap(client, monkeypatch)
    room = rooms.get_room(b["room_id"])
    monkeypatch.setattr(rooms, "LIVE_EVIDENCE_DIR", tmp_path)
    path = tmp_path / f"room_{room.room_id}_orphan.jpg"
    path.write_bytes(b"TEST_ONLY_BYTES")
    endpoint = f"/api/live/rooms/{room.room_id}/evidence/{path.name}"
    assert client.get(endpoint).status_code == 404
    room.record_evidence(path)
    assert client.get(endpoint).status_code == 404
    room.events.append(rooms.LiveEvent(id=1, type="test", class_name="car", track_id=1,
        detail="QA only", confidence=.9, capture_ts=1, recv_ts=1, event_ts=1,
        task_version=0, evidence_file=path.name))
    assert client.get(endpoint).content == b"TEST_ONLY_BYTES"
    auth._sessions[room.owner_token].expires_at = time.time() - 1
    assert client.get(endpoint).status_code == 401
    assert rooms.get_room(room.room_id) is None
    assert not path.exists()


def test_busy_admission_is_explicit_and_never_leaves_task_enabled(client, monkeypatch, flow):
    from app.live import video
    b = bootstrap(client, monkeypatch)
    room = rooms.get_room(b["room_id"])
    room.video_path = Path("owned-fixture.mp4")
    room.video_sha256 = "a" * 64
    draft = flow.post(client, f"/api/live/rooms/{room.room_id}/task/validate", {"task": TASK})
    assert draft.status_code == 200
    def busy(_):
        raise RuntimeError("capacity reached")
    monkeypatch.setattr(video, "start_job", busy)
    activation = flow.post(client, f"/api/live/rooms/{room.room_id}/task/activate", draft.json())
    assert activation.status_code == 503 and not room.task_enabled
    assert room.job_status == "busy"
    enabled = flow.post(client, f"/api/live/rooms/{room.room_id}/task/state", {"enabled": True, "paused": False})
    assert enabled.status_code == 503 and not room.task_enabled
