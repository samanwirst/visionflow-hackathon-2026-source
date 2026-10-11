"""Frame ingest + live YOLO worker pipeline tests.

Uses the deterministic QA stub detector (clearly test-only; NEVER a phone-inference
claim) and owned generated JPEG frames. Exercises size/queue/drop handling, event
-> evidence -> notification, task replacement, disable-while-camera-persists, and
the version guard.
"""

import numpy as np
import cv2

from app.live import config, rooms, worker, planner
from app.live.planner import PlannerResult


def _supported_proposal(cls="car", trigger="presence_in_zone"):
    return PlannerResult(status="ok", proposal={
        "supported": True, "name": f"watch {cls}", "object_class": cls,
        "zone": {"type": "full_frame"}, "trigger": trigger,
        "action": "focus_notify", "focus": True, "explanation": "qa",
    })


def _camera_room(client, flow, pin):
    flow.login(client, "camera", pin)
    created = flow.post(client, "/api/live/rooms", {}).json()
    return created["room_id"], created["join_code"]


def _join_judge(app, flow, pin, code):
    from fastapi.testclient import TestClient
    judge = TestClient(app)
    flow.login(judge, "judge", pin)
    flow.post(judge, "/api/live/rooms/join", {"join_code": code})
    return judge


def test_ws_camera_ingest_and_bounds(client, test_pin, flow, jpeg_frame):
    room_id, _ = _camera_room(client, flow, test_pin)
    with client.websocket_connect(flow.ws_path(client, "/api/live/ws/camera")) as ws:
        ws.send_text('{"seq":1,"w":320,"h":240,"capture_ts":1.0}')
        ws.send_bytes(jpeg_frame)
        # oversize frame is dropped
        ws.send_text('{"seq":2,"w":320,"h":240}')
        ws.send_bytes(b"\x00" * (config.MAX_FRAME_BYTES + 10))
    room = rooms.get_room(room_id)
    assert room.frames_received == 1
    assert room.frames_dropped == 1
    assert room.latest_jpeg == jpeg_frame
    assert room.frame_dirty is True


def test_latest_frame_only(client, test_pin, flow, jpeg_frame):
    room_id, _ = _camera_room(client, flow, test_pin)
    other = cv2.imencode(".jpg", np.full((240, 320, 3), 10, np.uint8))[1].tobytes()
    with client.websocket_connect(flow.ws_path(client, "/api/live/ws/camera")) as ws:
        ws.send_text('{"seq":1,"w":320,"h":240}')
        ws.send_bytes(jpeg_frame)
        ws.send_text('{"seq":2,"w":320,"h":240}')
        ws.send_bytes(other)
    room = rooms.get_room(room_id)
    # Only the newest frame is retained for inference.
    assert room.latest_jpeg == other
    assert room.frames_received == 2


def test_event_evidence_notification_pipeline(client, test_pin, flow, jpeg_frame):
    planner.transport = lambda text, rid: _supported_proposal("car")
    room_id, code = _camera_room(client, flow, test_pin)
    judge = _join_judge(client.app, flow, test_pin, code)

    plan = flow.post(judge, f"/api/live/rooms/{room_id}/task/plan", {"text": "watch the car"}).json()
    assert plan["status"] == "ok"
    act = flow.approve(judge, f"/api/live/rooms/{room_id}/task/activate", {"task": plan["task"]}).json()
    assert act["task_version"] == 1

    # Camera streams a frame; the worker processes several so the debounced
    # presence/enter rule can fire (needs >= 2 consecutive inside frames).
    with client.websocket_connect(flow.ws_path(client, "/api/live/ws/camera")) as ws:
        ws.send_text('{"seq":1,"w":320,"h":240,"capture_ts":5.0}')
        ws.send_bytes(jpeg_frame)
    room = rooms.get_room(room_id)
    with room.lock:
        version, task, engines = room.task_version, room.task, list(room.engines)
    for _ in range(3):
        worker.process_frame(room, jpeg_frame, {"w": 320, "h": 240}, version, True, task, engines)

    state = judge.get(f"/api/live/rooms/{room_id}/state").json()
    assert len(state["events"]) >= 1
    assert len(state["notifications"]) >= 1
    ev = state["events"][0]
    assert ev["evidence_url"]
    # Evidence is retrievable only by a room member.
    assert judge.get(ev["evidence_url"]).status_code == 200
    planner.transport = None


def test_disable_while_camera_persists(client, test_pin, flow, jpeg_frame):
    planner.transport = lambda text, rid: _supported_proposal("car")
    room_id, code = _camera_room(client, flow, test_pin)
    judge = _join_judge(client.app, flow, test_pin, code)
    plan = flow.post(judge, f"/api/live/rooms/{room_id}/task/plan", {"text": "watch car"}).json()
    flow.approve(judge, f"/api/live/rooms/{room_id}/task/activate", {"task": plan["task"]})

    # Disable the task; the camera session stays connected and keeps streaming.
    r = flow.post(judge, f"/api/live/rooms/{room_id}/task/state", {"enabled": False}).json()
    assert r["task_enabled"] is False
    room = rooms.get_room(room_id)
    with client.websocket_connect(flow.ws_path(client, "/api/live/ws/camera")) as ws:
        ws.send_text('{"seq":1,"w":320,"h":240}')
        ws.send_bytes(jpeg_frame)
    worker._process_room_if_ready(room)
    state = judge.get(f"/api/live/rooms/{room_id}/state").json()
    # Frame still ingested + annotated, but no events while disabled.
    assert state["metrics"]["frames_received"] == 1
    assert state["events"] == []
    planner.transport = None


def test_version_guard_drops_stale_events(client, test_pin, flow, jpeg_frame):
    planner.transport = lambda text, rid: _supported_proposal("car")
    room_id, code = _camera_room(client, flow, test_pin)
    judge = _join_judge(client.app, flow, test_pin, code)
    plan = flow.post(judge, f"/api/live/rooms/{room_id}/task/plan", {"text": "watch car"}).json()
    flow.approve(judge, f"/api/live/rooms/{room_id}/task/activate", {"task": plan["task"]})
    room = rooms.get_room(room_id)

    # Snapshot v1 engines, then disable (bumps version) BEFORE processing.
    with room.lock:
        stale_version = room.task_version
        stale_engines = list(room.engines)
        stale_task = room.task
    flow.post(judge, f"/api/live/rooms/{room_id}/task/state", {"enabled": False})

    # Processing a frame under the stale version must emit no events (guard).
    worker.process_frame(room, jpeg_frame, {"w": 320, "h": 240},
                         stale_version, True, stale_task, stale_engines)
    assert list(room.events) == []
    planner.transport = None


def test_task_replacement_new_version(client, test_pin, flow, jpeg_frame):
    seq = iter(["car", "bottle"])
    planner.transport = lambda text, rid: _supported_proposal(next(seq))
    room_id, code = _camera_room(client, flow, test_pin)
    judge = _join_judge(client.app, flow, test_pin, code)

    p1 = flow.post(judge, f"/api/live/rooms/{room_id}/task/plan", {"text": "watch car"}).json()
    a1 = flow.approve(judge, f"/api/live/rooms/{room_id}/task/activate", {"task": p1["task"]}).json()
    p2 = flow.post(judge, f"/api/live/rooms/{room_id}/task/plan", {"text": "watch bottle"}).json()
    a2 = flow.approve(judge, f"/api/live/rooms/{room_id}/task/activate", {"task": p2["task"]}).json()
    assert a2["task_version"] == a1["task_version"] + 1
    room = rooms.get_room(room_id)
    assert room.task["object_class"] == "bottle"
    planner.transport = None


def test_activate_rejects_invalid_task(client, test_pin, flow):
    planner.transport = lambda text, rid: _supported_proposal("car")
    room_id, code = _camera_room(client, flow, test_pin)
    judge = _join_judge(client.app, flow, test_pin, code)
    r = flow.post(judge, f"/api/live/rooms/{room_id}/task/activate",
                  {"task": {"name": "x", "object_class": "unicorn", "trigger": "presence_in_zone"}})
    assert r.status_code == 422
    planner.transport = None


def test_judge_cannot_push_camera_ws(client, test_pin, flow):
    # A judge session must not be able to open the camera WebSocket.
    _, code = _camera_room(client, flow, test_pin)
    judge = _join_judge(client.app, flow, test_pin, code)
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with judge.websocket_connect("/api/live/ws/camera"):
            pass
    planner.transport = None
