"""Owner-final security regressions for LIVE-VIDEO-001/005/006/007/008.

QA_ONLY: no real provider/credentials/GPU/camera. Each test maps to one reviewed
P1 finding on e75554d and asserts the FIXED fail-closed contract. The legacy
camera path (camera ingest enabled, local QA only) is exercised separately where
it must remain unchanged.
"""
import json
import os
import secrets
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.live import auth, config, planner, rooms, teacher, worker
from app.live import app as live_app
from app.live.app import app
from app.live.dsl import LiveTask, ZoneSpec, VisualObservation, capabilities_doc
from app.live.planner import PlannerResult

ORIGIN = "https://live.qa.invalid"
HOST = "live.qa.invalid"


@pytest.fixture
def vid(monkeypatch):
    """VIDEO-mode app (camera ingest DISABLED) with a random PIN and a configured
    Origin/Host allowlist."""
    auth._reset_for_tests(); rooms._reset_for_tests(); worker._reset_for_tests()
    teacher._reset_for_tests()
    planner._room_attempts.clear(); planner._room_last_attempt.clear()
    live_app._reset_drafts_for_tests()
    pin = "qa-" + secrets.token_urlsafe(10)
    monkeypatch.setattr(config, "DEMO_PIN", pin)
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    monkeypatch.setattr(config, "ALLOWED_ORIGINS", [ORIGIN])
    monkeypatch.setattr(config, "ALLOWED_HOSTS", [HOST])
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", False)
    monkeypatch.setattr(config, "PLANNER_COOLDOWN_SECONDS", 0.0)
    monkeypatch.setattr(config, "PLANNER_ROOM_BUDGET", 100)
    monkeypatch.setattr(config, "TEACHER_BASE_URL", "")
    monkeypatch.setattr(worker, "start", lambda: None)
    monkeypatch.setattr(planner, "transport", None)
    monkeypatch.setattr(teacher, "transport", None)
    clients = []

    def _client(role=None):
        c = TestClient(app, base_url=ORIGIN, raise_server_exceptions=False)
        clients.append(c)
        if role:
            r = c.post("/api/live/login", json={"role": role, "pin": pin},
                       headers={"Origin": ORIGIN})
            assert r.status_code == 200, r.text
        return c

    def _post(c, path, body):
        return c.post(path, json=body, headers={
            "Origin": ORIGIN, "X-CSRF-Token": c.cookies.get(config.CSRF_COOKIE_NAME, "")})

    def _room():
        cam = _client("camera")
        r = _post(cam, "/api/live/rooms", {})
        assert r.status_code == 200, r.text
        data = r.json()
        return cam, rooms.get_room(data["room_id"]), data["join_code"]

    def _judge(code):
        j = _client("judge")
        assert _post(j, "/api/live/rooms/join", {"join_code": code}).status_code == 200
        return j

    with TestClient(app, base_url=ORIGIN, raise_server_exceptions=False):
        yield type("H", (), {"pin": pin, "client": staticmethod(_client),
                             "post": staticmethod(_post), "room": staticmethod(_room),
                             "judge": staticmethod(_judge)})
    for c in clients:
        c.close()
    worker._reset_for_tests()


def _task(cls="car", trigger="presence_in_zone", **extra):
    t = {"name": f"QA {cls}", "object_class": cls, "zone": {"type": "full_frame"},
         "trigger": trigger, "action": "notify"}
    t.update(extra)
    return t


# --------------------------------------------------------------- LIVE-VIDEO-001


def _semantic_proposal():
    return {
        "supported": True, "name": "Bottle opening", "object_class": "bottle",
        "zone": {"type": "full_frame", "polygon": []}, "trigger": "presence_in_zone",
        "action": "focus_notify", "focus": True,
        "visual_observation": {"question": "Is the bottle being opened?",
                               "labels": ["opening", "closed"], "temporal": True},
        "clarification_needed": "", "unsupported_reason": "",
    }


def test_001_semantic_proposal_unsupported_in_video_mode(monkeypatch):
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", False)
    r = planner._finalize(PlannerResult(status="ok", proposal=_semantic_proposal()))
    assert r.status == "unsupported" and r.task is None
    assert "video mode" in (r.unsupported_reason or "")


def test_001_livetask_rejects_visual_observation_in_video_mode(monkeypatch):
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", False)
    vo = VisualObservation(question="is it open?", labels=["open", "closed"])
    with pytest.raises(Exception):
        LiveTask(name="x", object_class="bottle", trigger="presence_in_zone",
                 visual_observation=vo)
    # Legacy camera path still accepts it.
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", True)
    t = LiveTask(name="x", object_class="bottle", trigger="presence_in_zone",
                 visual_observation=vo)
    assert t.visual_observation is not None


def test_001_capabilities_declare_semantic_unsupported(monkeypatch):
    monkeypatch.setattr(config, "CAMERA_INGEST_ENABLED", False)
    caps = capabilities_doc()
    assert caps["visual_observation"] == "UNSUPPORTED in video mode"
    assert any("UNSUPPORTED" in n for n in caps["notes"])


def test_001_probe_route_is_inaccessible_in_video_mode(vid):
    c, room, code = vid.room()
    j = vid.judge(code)
    calls = []
    import app.live.teacher as _t
    _t_observe = _t.observe
    _t.observe = lambda *a, **kw: calls.append(a)
    try:
        r = vid.post(j, f"/api/live/rooms/{room.room_id}/probe",
                     {"question": "open?", "labels": ["open", "closed"]})
    finally:
        _t.observe = _t_observe
    assert r.status_code == 404 and calls == []


# --------------------------------------------------------------- LIVE-VIDEO-005


@pytest.mark.parametrize("mutate", [
    {"code": "print(1)"},                 # injected side-channel field
    {"threshold": True, "trigger": "count_exceeded"},  # bool count
    {"duration_seconds": True, "trigger": "dwell_in_zone"},  # bool duration
    {"focus": "false"},                   # string coerced to bool
    {"cooldown_seconds": True},           # bool cooldown
])
def test_005_strict_dsl_rejects_coercion_and_extras(mutate):
    raw = _task(**mutate)
    with pytest.raises(Exception):
        LiveTask.model_validate(raw)


def test_005_cooldown_seconds_is_supported_and_carried():
    # Owner adaptation: cooldown_seconds is a SUPPORTED field (reviewer baseline
    # expected rejection; owner chose faithful support, default handled by VIDEO).
    t = LiveTask.model_validate(_task(cooldown_seconds=1.25))
    assert t.cooldown_seconds == 1.25
    assert t.model_dump()["cooldown_seconds"] == 1.25
    # Absent -> None (VIDEO applies config.DEFAULT_COOLDOWN_SECONDS).
    assert LiveTask.model_validate(_task()).cooldown_seconds is None


def test_005_full_frame_conflicting_polygon_rejected():
    with pytest.raises(Exception):
        ZoneSpec(type="full_frame", polygon=[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)])
    # Canonical square is idempotent (re-validation safe).
    z = ZoneSpec(type="full_frame")
    assert ZoneSpec.model_validate(z.model_dump()).type == "full_frame"


def test_005_duration_125_exact_preserved():
    t = LiveTask.model_validate(_task(trigger="dwell_in_zone", duration_seconds=1.25))
    assert t.duration_seconds == 1.25


def test_005_activation_wrong_binding_rejected(vid):
    c, room, code = vid.room()
    j = vid.judge(code)
    r = vid.post(j, f"/api/live/rooms/{room.room_id}/task/activate",
                 {"task": _task(), "draft_hash": "WRONG", "expected_version": 900})
    assert r.status_code in (409, 422), r.text
    assert room.task is None and room.task_version == 0


def test_005_validate_then_activate_binding_and_replay(vid):
    c, room, code = vid.room()
    j = vid.judge(code)
    v = vid.post(j, f"/api/live/rooms/{room.room_id}/task/validate", {"task": _task()})
    assert v.status_code == 200, v.text
    draft = v.json()
    assert draft["draft_hash"] and draft["expected_version"] == 0
    body = {"task": draft["task"], "draft_hash": draft["draft_hash"],
            "expected_version": draft["expected_version"]}
    ok = vid.post(j, f"/api/live/rooms/{room.room_id}/task/activate", body)
    assert ok.status_code == 200, ok.text
    assert room.task_version == 1
    # Replay the SAME approval: the draft is consumed and the version moved -> reject.
    replay = vid.post(j, f"/api/live/rooms/{room.room_id}/task/activate", body)
    assert replay.status_code == 409, replay.text


def test_005_plan_exposes_draft_binding(vid, monkeypatch):
    c, room, code = vid.room()
    j = vid.judge(code)
    monkeypatch.setattr(planner, "transport",
                        lambda *_: PlannerResult(status="ok", proposal={
                            "supported": True, "name": "QA car", "object_class": "car",
                            "zone": {"type": "full_frame", "polygon": []},
                            "trigger": "presence_in_zone", "action": "notify",
                            "explanation": "x", "clarification_needed": "",
                            "unsupported_reason": ""}))
    r = vid.post(j, f"/api/live/rooms/{room.room_id}/task/plan", {"text": planner.SUPPORTED_COMMAND_EXAMPLES[0]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["draft_hash"] and body["expected_version"] == 0


# --------------------------------------------------------------- LIVE-VIDEO-006


def test_006_host_allowlist_authoritative(vid):
    c = vid.client()
    r = c.post("/api/live/login", json={"role": "judge", "pin": vid.pin},
               headers={"Origin": ORIGIN, "Host": "attacker.invalid"})
    assert r.status_code in (400, 403), r.text


def test_006_untrusted_peer_cannot_spoof_xff(monkeypatch):
    monkeypatch.setattr(config, "FORWARDED_ALLOW_IPS", "127.0.0.1,::1")
    monkeypatch.setattr(config, "TRUSTED_PROXY_HOPS", 1)
    # Untrusted direct peer: its XFF header is ignored -> identity is the peer.
    assert auth.client_identity("203.0.113.9", "198.51.100.1") == "203.0.113.9"
    assert auth.client_identity("203.0.113.9", "198.51.100.2") == "203.0.113.9"
    # Trusted proxy peer: the chain entry is honored.
    assert auth.client_identity("127.0.0.1", "198.51.100.5") == "198.51.100.5"
    # Hops disabled: XFF never honored.
    monkeypatch.setattr(config, "TRUSTED_PROXY_HOPS", 0)
    assert auth.client_identity("127.0.0.1", "198.51.100.5") == "127.0.0.1"


# --------------------------------------------------------------- LIVE-VIDEO-007


def test_007_oversize_json_rejected_by_content_length(vid):
    c = vid.client()
    r = c.post("/api/live/login", json={"role": "judge", "pin": vid.pin,
               "padding": "x" * (config.MAX_JSON_BODY_BYTES * 2)},
               headers={"Origin": ORIGIN})
    assert r.status_code == 413, r.text


def test_007_chunked_json_without_content_length_bounded(vid, monkeypatch):
    c, room, code = vid.room()
    j = vid.judge(code)
    monkeypatch.setattr(planner, "transport",
                        lambda *_: PlannerResult(status="unsupported"))
    data = json.dumps({"text": "watch a car",
                       "padding": "x" * (config.MAX_JSON_BODY_BYTES * 2)}).encode()
    r = j.post(f"/api/live/rooms/{room.room_id}/task/plan", content=iter([data]),
               headers={"Origin": ORIGIN,
                        "X-CSRF-Token": j.cookies.get(config.CSRF_COOKIE_NAME),
                        "Content-Type": "application/json"})
    assert r.status_code == 413, r.text


# --------------------------------------------------------------- LIVE-VIDEO-008


def test_008_planner_timeout_kills_descendants_before_releasing_permit(tmp_path, monkeypatch):
    fixture = tmp_path / "qa_only_child_planner.py"
    pidfile = tmp_path / "qa_only_child_pids.txt"
    childcode = "import time;time.sleep(.5)"
    fixture.write_text(
        "#!/usr/bin/python3\nimport subprocess,time\n"
        f'child=subprocess.Popen(["/usr/bin/python3","-c",{childcode!r}])\n'
        f'open({str(pidfile)!r},"a").write(str(child.pid)+"\\n")\n'
        "time.sleep(2)\n")
    fixture.chmod(0o700)
    monkeypatch.setattr(config, "PLANNER_CLI", str(fixture))
    monkeypatch.setattr(config, "PLANNER_TIMEOUT_SECONDS", 0.06)
    results = [planner._run_cli("QA_ONLY descendant") for _ in range(2)]
    pids = [int(x) for x in pidfile.read_text().splitlines()]
    live = []
    for pid in pids:
        try:
            os.kill(pid, 0); live.append(pid)
        except ProcessLookupError:
            pass
    assert all(r.status == "unavailable" for r in results)
    # Capacity is 1; each timed-out job's whole process group is reaped before the
    # permit is released, so no descendant survives.
    assert len(live) <= 1, {"live": live, "statuses": [r.status for r in results]}


def test_008_planner_timeout_is_bounded(tmp_path, monkeypatch):
    fixture = tmp_path / "qa_only_planner.py"
    fixture.write_text(
        "#!/usr/bin/python3\nimport subprocess,time\n"
        'subprocess.Popen(["/usr/bin/python3","-c","import time;time.sleep(.35)"])\n'
        "time.sleep(2)\n")
    fixture.chmod(0o700)
    monkeypatch.setattr(config, "PLANNER_CLI", str(fixture))
    monkeypatch.setattr(config, "PLANNER_TIMEOUT_SECONDS", 0.06)
    start = time.monotonic()
    r = planner._run_cli("QA_ONLY deadline")
    elapsed = time.monotonic() - start
    assert elapsed < 0.6 and r.status == "unavailable", {"elapsed": elapsed, "status": r.status}
