"""Regression tests for the first independent LIVE review (fix pass 2).

Covers: authoritative Origin/Host policy and CSRF on logout + WS (LIVE-003),
spoofed X-Forwarded-For rate-limit evasion (LIVE-004), room lifecycle / capacity
release (LIVE-006), malformed-frame resilience and pre-parse body bounds
(LIVE-005), and strict typed task-state controls with engine rebuild (LIVE-007).

All detectors/providers are clearly-labeled QA stubs; no physical camera, real
provider, or GPU is used, and no demo PIN is hardcoded.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.live import auth, config, rooms, worker


# --- LIVE-003 : authoritative Origin / Host / CSRF -------------------------


def test_origin_allowlist_is_authoritative(client, test_pin, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", ["https://good.example"])
    monkeypatch.setattr(config, "ALLOWED_HOSTS", ["good.example"])
    # With an allowlist configured, a missing Origin is refused on a mutation.
    assert client.post("/api/live/login", json={"role": "judge", "pin": test_pin}).status_code == 403
    # A caller-controlled Host matching an attacker origin cannot bypass it.
    r = client.post("/api/live/login", json={"role": "judge", "pin": test_pin},
                    headers={"Origin": "https://evil.example", "Host": "evil.example"})
    assert r.status_code == 403
    # The exact allowlisted Origin succeeds.
    r = client.post("/api/live/login", json={"role": "judge", "pin": test_pin},
                    headers={"Origin": "https://good.example", "Host": "good.example"})
    assert r.status_code == 200


def test_logout_requires_csrf(client, test_pin, flow):
    flow.login(client, "judge", test_pin)
    # Logout is a mutation: without the CSRF token it is refused.
    assert client.post("/api/live/logout").status_code == 403
    # With the CSRF token it succeeds.
    assert client.post("/api/live/logout",
                       headers={"X-CSRF-Token": flow.csrf(client)}).status_code == 200


def test_ws_requires_csrf_token(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    flow.post(client, "/api/live/rooms", {})
    # No csrf query param on the upgrade -> rejected.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/live/ws/camera"):
            pass
    # Wrong csrf -> rejected.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/live/ws/camera?csrf=wrong"):
            pass


# --- LIVE-004 : spoofed forwarding headers cannot evade the limit ----------


def test_spoofed_xff_cannot_evade_lockout(client, test_pin):
    # TRUSTED_PROXY_HOPS defaults to 0 => XFF is ignored and the peer identity is
    # used, so rotating X-Forwarded-For does not reset the per-client counter.
    for i in range(config.LOGIN_MAX_FAILURES):
        r = client.post("/api/live/login", json={"role": "judge", "pin": "bad"},
                        headers={"X-Forwarded-For": f"10.0.0.{i}"})
        assert r.status_code == 401
    r = client.post("/api/live/login", json={"role": "judge", "pin": test_pin},
                    headers={"X-Forwarded-For": "10.0.0.250"})
    assert r.status_code == 429


def test_trusted_proxy_identity_from_right(monkeypatch):
    monkeypatch.setattr(config, "TRUSTED_PROXY_HOPS", 1)
    # One trusted hop => the rightmost XFF entry is the trusted client identity.
    assert auth.client_identity("127.0.0.1", "1.1.1.1, 2.2.2.2") == "2.2.2.2"
    # Untrusted deployment (0 hops) ignores XFF entirely.
    monkeypatch.setattr(config, "TRUSTED_PROXY_HOPS", 0)
    assert auth.client_identity("9.9.9.9", "1.1.1.1") == "9.9.9.9"


# --- LIVE-006 : room lifecycle / capacity release --------------------------


def test_logout_retires_owner_room(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    rid = flow.post(client, "/api/live/rooms", {}).json()["room_id"]
    assert rooms.get_room(rid) is not None
    client.post("/api/live/logout", headers={"X-CSRF-Token": flow.csrf(client)})
    assert rooms.get_room(rid) is None  # capacity released


def test_retire_orphaned_and_idle_rooms():
    rooms._reset_for_tests()
    orphan = rooms.create_room("owner-token-gone")
    retired = rooms.retire_idle_and_orphaned(active=set(), idle_seconds=10_000)
    assert orphan.room_id in retired
    assert rooms.get_room(orphan.room_id) is None


def test_expired_session_sweep_releases_room(client, test_pin, flow):
    import time
    flow.login(client, "camera", test_pin)
    rid = flow.post(client, "/api/live/rooms", {}).json()["room_id"]
    for s in list(auth._sessions.values()):
        s.expires_at = time.time() - 1
    active = auth.sweep_expired()  # fires the teardown hook
    rooms.retire_idle_and_orphaned(active, config.ROOM_IDLE_TTL_SECONDS)
    assert rooms.get_room(rid) is None


# --- LIVE-005 : malformed frames + body bounds -----------------------------


def _camera_room(client, flow, pin):
    flow.login(client, "camera", pin)
    created = flow.post(client, "/api/live/rooms", {}).json()
    return created["room_id"]


def test_empty_and_nonjpeg_frames_do_not_kill_worker(client, test_pin, flow):
    rid = _camera_room(client, flow, test_pin)
    room = rooms.get_room(rid)
    # These would previously raise inside cv2 and kill the sole worker thread.
    worker.process_frame(room, b"", {"w": 320, "h": 240}, 0, False, None, [])
    worker.process_frame(room, b"not-a-jpeg-at-all", {"w": 320, "h": 240}, 0, False, None, [])
    worker.process_frame(room, b"\x00" * 200, {"w": 320, "h": 240}, 0, False, None, [])
    # Worker code path survived (no exception) and no events were produced.
    assert list(room.events) == []


def test_oversize_decoded_dimensions_dropped(client, test_pin, flow):
    rid = _camera_room(client, flow, test_pin)
    room = rooms.get_room(rid)
    big = np.full((100, 1400, 3), 50, np.uint8)  # long edge 1400 > bound 960
    jpeg = cv2.imencode(".jpg", big)[1].tobytes()
    before = room.frames_dropped
    # Declared metadata lies (320x240); the actual decoded size is inspected.
    worker.process_frame(room, jpeg, {"w": 320, "h": 240}, 0, False, None, [])
    assert room.frames_dropped == before + 1
    assert room.frames_inferred == 0


def test_duplicate_and_stale_sequence_dropped(client, test_pin, flow, jpeg_frame):
    from app.live.app import _ingest_frame
    rid = _camera_room(client, flow, test_pin)
    room = rooms.get_room(rid)
    _ingest_frame(room, jpeg_frame, {"seq": 5, "w": 320, "h": 240})
    _ingest_frame(room, jpeg_frame, {"seq": 5, "w": 320, "h": 240})  # duplicate
    _ingest_frame(room, jpeg_frame, {"seq": 4, "w": 320, "h": 240})  # stale
    _ingest_frame(room, jpeg_frame, {"seq": 6, "w": 320, "h": 240})  # accepted
    assert room.frames_received == 2
    assert room.frames_dropped == 2


def test_nonfinite_capture_ts_sanitized(client, test_pin, flow, jpeg_frame):
    from app.live.app import _ingest_frame
    rid = _camera_room(client, flow, test_pin)
    room = rooms.get_room(rid)
    _ingest_frame(room, jpeg_frame, {"seq": 1, "w": 320, "h": 240, "capture_ts": float("nan")})
    assert room.latest_meta["capture_ts"] is None  # never poisons JSON state
    # The owning camera session reads its own room state without a 500.
    r = client.get(f"/api/live/rooms/{rid}/state")
    assert r.status_code == 200


def test_json_body_pre_parse_limit(client, test_pin, flow):
    flow.login(client, "judge", test_pin)
    huge = {"text": "x" * (config.MAX_JSON_BODY_BYTES + 500)}
    r = client.post("/api/live/rooms/none/task/plan", json=huge,
                    headers={"X-CSRF-Token": flow.csrf(client)})
    assert r.status_code == 413


def test_health_reports_worker_liveness(client):
    h = client.get("/api/live/health").json()
    assert "worker" in h and "alive" in h["worker"]


# --- LIVE-007 : strict typed controls + engine rebuild ---------------------


def _activate(client, flow, rid, judge, task):
    return flow.approve(judge, f"/api/live/rooms/{rid}/task/activate", {"task": task})


def _basic_task(cls="car", trigger="presence_in_zone"):
    return {"name": f"watch {cls}", "object_class": cls, "zone": {"type": "full_frame"},
            "trigger": trigger, "action": "notify"}


def test_task_state_rejects_string_bool(client, test_pin, flow):
    rid = _camera_room(client, flow, test_pin)
    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    code = rooms.get_room(rid).join_code
    flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    _activate(client, flow, rid, judge, _basic_task())
    # The string "false" must NOT enable/flip the task.
    r = flow.post(judge, f"/api/live/rooms/{rid}/task/state", {"enabled": "false"})
    assert r.status_code == 422
    # An unexpected field is rejected too.
    r = flow.post(judge, f"/api/live/rooms/{rid}/task/state", {"enabled": True, "bogus": 1})
    assert r.status_code == 422


def test_reenable_and_pause_rebuild_engines_and_bump_version(client, test_pin, flow):
    rid = _camera_room(client, flow, test_pin)
    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    code = rooms.get_room(rid).join_code
    flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    act = _activate(client, flow, rid, judge, _basic_task()).json()
    room = rooms.get_room(rid)
    v0 = act["task_version"]
    assert room.engines  # active task has engines

    # Disable -> version bumps, engines cleared.
    r = flow.post(judge, f"/api/live/rooms/{rid}/task/state", {"enabled": False}).json()
    assert r["task_version"] > v0
    assert room.engines == []

    # Re-enable -> version bumps again AND engines are rebuilt (not "running with 0").
    r2 = flow.post(judge, f"/api/live/rooms/{rid}/task/state", {"enabled": True}).json()
    assert r2["task_version"] > r["task_version"]
    assert room.engines

    # Pause invalidates the generation so held inference cannot notify post-ACK.
    vpaused = r2["task_version"]
    r3 = flow.post(judge, f"/api/live/rooms/{rid}/task/state", {"paused": True}).json()
    assert r3["task_version"] > vpaused
    assert room.engines == []  # paused is not active
