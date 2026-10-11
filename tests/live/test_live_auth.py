"""Security tests for the live public surface: PIN auth, brute force, roles,
CSRF, Origin, session expiry, and room cross-access / pairing isolation."""

import time

from fastapi.testclient import TestClient

from app.live import auth, config


def test_wrong_pin_rejected(client, test_pin):
    r = client.post("/api/live/login", json={"role": "judge", "pin": test_pin + "x"})
    assert r.status_code == 401
    assert config.SESSION_COOKIE_NAME not in r.cookies


def test_login_sets_hardened_cookie(client, test_pin, monkeypatch):
    monkeypatch.setattr(config, "COOKIE_SECURE", True)
    r = client.post("/api/live/login", json={"role": "judge", "pin": test_pin})
    assert r.status_code == 200
    set_cookie = r.headers.get("set-cookie", "")
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "samesite=strict" in set_cookie.lower()


def test_brute_force_lockout(client, test_pin):
    for _ in range(config.LOGIN_MAX_FAILURES):
        assert client.post("/api/live/login", json={"role": "judge", "pin": "nope"}).status_code == 401
    # Further attempts (even with the CORRECT pin) are rate-limited.
    r = client.post("/api/live/login", json={"role": "judge", "pin": test_pin})
    assert r.status_code == 429


def test_unauthenticated_cannot_create_room(client):
    r = client.post("/api/live/rooms", json={}, headers={"X-CSRF-Token": "x"})
    assert r.status_code == 401


def test_csrf_required_for_mutation(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    r = client.post("/api/live/rooms", json={})  # no CSRF header
    assert r.status_code == 403


def test_origin_rejected_cross_site(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    r = client.post("/api/live/rooms", json={},
                    headers={"X-CSRF-Token": flow.csrf(client), "Origin": "http://evil.example"})
    assert r.status_code == 403


def test_role_separation_camera_cannot_plan(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    room_id = flow.post(client, "/api/live/rooms", {}).json()["room_id"]
    r2 = flow.post(client, f"/api/live/rooms/{room_id}/task/plan", {"text": "watch the car"})
    assert r2.status_code == 403


def test_judge_cannot_create_room(client, test_pin, flow):
    flow.login(client, "judge", test_pin)
    assert flow.post(client, "/api/live/rooms", {}).status_code == 403


def test_pin_alone_cannot_join_or_list_rooms(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    flow.post(client, "/api/live/rooms", {})

    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    # No listing endpoint exists; joining with a bad code fails with a plain 404.
    r = flow.post(judge, "/api/live/rooms/join", {"join_code": "not-a-code"})
    assert r.status_code == 404


def test_room_cross_access_forbidden(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    room_id = flow.post(client, "/api/live/rooms", {}).json()["room_id"]

    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    assert judge.get(f"/api/live/rooms/{room_id}/state").status_code == 403


def test_join_binds_and_grants_access(client, test_pin, flow):
    flow.login(client, "camera", test_pin)
    created = flow.post(client, "/api/live/rooms", {}).json()
    room_id, code = created["room_id"], created["join_code"]

    judge = TestClient(client.app)
    flow.login(judge, "judge", test_pin)
    r = flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    assert r.status_code == 200
    assert judge.get(f"/api/live/rooms/{room_id}/state").status_code == 200


def test_session_expiry(client, test_pin, flow):
    flow.login(client, "judge", test_pin)
    for s in list(auth._sessions.values()):
        s.expires_at = time.time() - 1
    assert client.get("/api/live/me").status_code == 401


def test_ws_rejects_cross_origin(client, test_pin, flow):
    import pytest
    from starlette.websockets import WebSocketDisconnect
    flow.login(client, "camera", test_pin)
    flow.post(client, "/api/live/rooms", {})
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/live/ws/camera",
                                      headers={"Origin": "http://evil.example"}):
            pass


def test_ws_rejects_unauthenticated(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/live/ws/judge"):
            pass


def test_health_reports_pin_configured(client):
    h = client.get("/api/live/health").json()
    assert h["pin_configured"] is True
    assert h["service"] == "visionflow-live"
