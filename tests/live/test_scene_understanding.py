"""Mock transport tests are NOT evidence of real model scene understanding."""
import hashlib
import io
import json
import threading
import base64
import subprocess
import sys

from PIL import Image
import pytest

from app.live import rooms, scene, scene_api_worker


@pytest.fixture
def source_room(tmp_path, monkeypatch):
    path = tmp_path / "opaque.mp4"
    path.write_bytes(b"mock validated video bytes")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    room = rooms.RoomState("fixture-room", "fixture-join", "fixture-owner")
    room.video_path = path
    room.video_sha256 = sha
    room.video_meta = {"fps": 10.0, "frame_count": 11}
    room.task_version = 4
    monkeypatch.setenv("VISIONFLOW_LIVE_SCENE_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-token-not-real")
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(output, format="JPEG")
    jpeg = output.getvalue()
    def sample(_source):
        return [{"frame_id": i, "video_timestamp_seconds": i / 10,
                 "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                 "image_bytes": len(jpeg), "jpeg": jpeg} for i in (0, 5, 10)]
    monkeypatch.setattr(scene, "_sample", sample)
    return room


def envelope(objects=None, support="SUPPORTED"):
    return {"subtype": "success", "model": scene.MODEL,
            "usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
            "structured_output": {"scene": "QA mock observation", "visible_objects": objects or [],
                "answer": "QA mock answer", "answer_support": support,
                "answer_evidence_frame_ids": [0] if support == "SUPPORTED" else [],
                "uncertainty": "QA transport mock, not a real model measurement"}}


def inspect(room, **kwargs):
    return scene.inspect_scene(room, "What is visible?", room.video_sha256, 4,
                               session_is_current=kwargs.get("current", lambda: True))


def test_mock_positive_provenance_without_actions(source_room, monkeypatch):
    monkeypatch.setattr(scene, "_infer", lambda *_, **__: envelope([
        {"name": "test object", "description": "mock detail", "evidence_frame_ids": [5]}]))
    result = inspect(source_room)
    assert result["status"] == "ok" and result["nonactioning"] is True
    assert result["source_sha256"] == source_room.video_sha256
    assert result["workflow_version"] == 4
    assert result["model"] == scene.MODEL
    assert [f["video_timestamp_seconds"] for f in result["evidence"]] == [0, .5, 1]
    assert all(len(f["frame_sha256"]) == 64 and "jpeg" not in f for f in result["evidence"])
    assert not source_room.events and not source_room.notifications and source_room.task is None


def test_mock_negative_abstains_without_actions(source_room, monkeypatch):
    monkeypatch.setattr(scene, "_infer", lambda *_, **__: envelope(support="UNCERTAIN"))
    result = inspect(source_room)
    assert result["observation"]["answer_support"] == "UNCERTAIN"
    assert result["observation"]["visible_objects"] == []
    assert source_room.task is None and not source_room.events


@pytest.mark.parametrize("transition", ["source", "version", "retired", "session"])
def test_stale_results_never_publish_observations(source_room, monkeypatch, transition):
    current = [True]
    def infer(*_, **__):
        if transition == "source":
            source_room.video_sha256 = "0" * 64
        elif transition == "version":
            source_room.task_version += 1
        elif transition == "retired":
            source_room.retired = True
        else:
            current[0] = False
        return envelope()
    monkeypatch.setattr(scene, "_infer", infer)
    result = inspect(source_room, current=lambda: current[0])
    assert result["status"] == "stale"
    assert "observation" not in result and "evidence" not in result
    assert not source_room.events and not source_room.notifications


def test_before_provider_stale_source_does_not_spend(source_room, monkeypatch):
    called = []
    def sample(_):
        source_room.task_version += 1
        return []
    monkeypatch.setattr(scene, "_sample", sample)
    monkeypatch.setattr(scene, "_infer", lambda *_, **__: called.append(True))
    assert inspect(source_room)["status"] == "stale" and not called


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 0, 61])
def test_invalid_source_clock_rejected(source_room, bad):
    source_room.video_meta["fps"] = bad
    assert inspect(source_room)["status"] == "unavailable"


@pytest.mark.parametrize("bad", [True, "11", 0, 5001])
def test_invalid_source_frame_count_rejected(source_room, bad):
    source_room.video_meta["frame_count"] = bad
    assert inspect(source_room)["status"] == "unavailable"


def test_source_byte_tampering_rejected(source_room, monkeypatch):
    source_room.video_path.write_bytes(b"changed")
    called = []
    monkeypatch.setattr(scene, "_infer", lambda *_, **__: called.append(True))
    assert inspect(source_room)["status"] == "unavailable" and not called


@pytest.mark.parametrize("defect", ["invalid_frame", "boolean_frame", "extra", "no_evidence", "too_many_tokens"])
def test_invalid_model_output_rejected(source_room, monkeypatch, defect):
    data = envelope()
    if defect == "invalid_frame":
        data["structured_output"]["answer_evidence_frame_ids"] = [99]
    elif defect == "boolean_frame":
        data["structured_output"]["answer_evidence_frame_ids"] = [True]
    elif defect == "extra":
        data["structured_output"]["action"] = "notify"
    elif defect == "no_evidence":
        data["structured_output"]["answer_evidence_frame_ids"] = []
    else:
        data["usage"] = {"prompt_tokens": 200, "completion_tokens": 701, "total_tokens": 901}
    monkeypatch.setattr(scene, "_infer", lambda *_, **__: data)
    assert inspect(source_room)["status"] == "unavailable"
    assert not source_room.events


def test_provider_failure_has_no_secret_or_event(source_room, monkeypatch):
    def failure(*_):
        raise RuntimeError("private credential and provider details")
    monkeypatch.setattr(scene, "_infer", failure)
    result = inspect(source_room)
    assert result["status"] == "unavailable"
    assert "private credential" not in json.dumps(result) and not source_room.events


def test_decoder_and_inference_actual_concurrency_bounded(source_room, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def blocking(*_, **__):
        entered.set()
        assert release.wait(2)
        return envelope()
    monkeypatch.setattr(scene, "_infer", blocking)
    thread = threading.Thread(target=lambda: inspect(source_room))
    thread.start()
    assert entered.wait(1)
    assert inspect(source_room)["status"] == "busy"
    release.set()
    thread.join(2)
    assert not thread.is_alive()


def test_disabled_or_missing_credential_does_not_decode(source_room, monkeypatch):
    called = []
    monkeypatch.setattr(scene, "_sample", lambda *_, **__: called.append(True))
    monkeypatch.delenv("OPENAI_API_KEY")
    assert inspect(source_room)["status"] == "unavailable" and not called
    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setenv("VISIONFLOW_LIVE_SCENE_ENABLED", "false")
    assert inspect(source_room)["status"] == "unavailable" and not called


def test_actual_ffmpeg_sample_provenance(qa_mp4, tmp_path):
    path = tmp_path / "qa-only.mp4"
    path.write_bytes(qa_mp4)
    source = scene.Source(path, hashlib.sha256(qa_mp4).hexdigest(), 10., 15, 0)
    frames = scene._sample(source)
    assert [f["frame_id"] for f in frames] == [0, 7, 14]
    assert [f["video_timestamp_seconds"] for f in frames] == [0, .7, 1.4]
    assert len({f["frame_sha256"] for f in frames}) > 1
    assert all(0 < f["image_bytes"] <= scene.MAX_IMAGE_BYTES for f in frames)


def test_fixed_provider_rejects_invalid_requests_before_network():
    with pytest.raises(ValueError):
        scene_api_worker.infer({}, "mock-only", 1)
    assert scene_api_worker.ENDPOINT == "https://api.openai.com/v1/chat/completions"
    assert scene_api_worker.MODEL == scene.MODEL


def test_redirects_refused():
    with pytest.raises(Exception):
        scene_api_worker.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.test")


def worker_request():
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(output, format="JPEG")
    return {"question": "What objects are visibly supported?", "system": scene.SYSTEM,
            "schema": scene.SCHEMA, "frames": [{"frame_id": 7,
                "video_timestamp_seconds": .7,
                "jpeg_base64": base64.b64encode(output.getvalue()).decode()}]}


def test_provider_receives_actual_image_bytes_and_strict_schema(monkeypatch):
    captured = []
    data = envelope()
    provider = {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(data["structured_output"])}}], "model": scene.MODEL, "usage": data["usage"]}
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, n): return json.dumps(provider).encode()
    class Opener:
        def open(self, request, timeout):
            captured.append((request, timeout))
            return Response()
    monkeypatch.setattr(scene_api_worker.urllib.request, "build_opener", lambda *_, **__: Opener())
    source = worker_request()
    result = scene_api_worker.infer(source, "qa-only-token", 2)
    request, timeout = captured[0]
    body = json.loads(request.data)
    assert request.full_url == scene_api_worker.ENDPOINT and timeout == 2
    assert body["model"] == scene.MODEL and body["max_completion_tokens"] == 700
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["store"] is False and body["stream"] is False and body["n"] == 1
    image_url = body["messages"][1]["content"][2]["image_url"]["url"]
    assert image_url == "data:image/jpeg;base64," + source["frames"][0]["jpeg_base64"]
    assert result["model"] == scene.MODEL and result["usage"] == data["usage"]


@pytest.mark.parametrize("defect", ["image_budget", "frame_boolean", "timestamp_nan", "extra_frame", "too_many_frames"])
def test_malformed_image_request_never_reaches_network(monkeypatch, defect):
    called = []
    monkeypatch.setattr(scene_api_worker.urllib.request, "build_opener", lambda *_, **__: called.append(True))
    request = worker_request()
    if defect == "image_budget":
        request["frames"][0]["jpeg_base64"] = base64.b64encode(b"\xff\xd8" + b"x" * (256 * 1024)).decode()
    elif defect == "frame_boolean":
        request["frames"][0]["frame_id"] = True
    elif defect == "timestamp_nan":
        request["frames"][0]["video_timestamp_seconds"] = float("nan")
    elif defect == "extra_frame":
        request["frames"][0]["url"] = "https://evil.test"
    else:
        request["frames"] *= 4
    with pytest.raises(ValueError):
        scene_api_worker.infer(request, "qa-only-token", 1)
    assert not called


@pytest.mark.parametrize("reserve", [False, "corrupt"])
def test_shared_budget_failclosed_and_releases_permit(monkeypatch, reserve):
    from app.live import planner
    def reservation():
        if reserve == "corrupt":
            raise ValueError("invalid private ledger")
        return False
    monkeypatch.setenv("OPENAI_API_KEY", "qa-only-token")
    monkeypatch.setattr(planner, "_reserve_api_attempt", reservation, raising=False)
    called = []
    monkeypatch.setattr(scene.subprocess, "Popen", lambda *_args, **_kwargs: called.append(True))
    with pytest.raises(ValueError):
        scene._infer("QA question", [])
    assert not called
    assert planner._planner_lock.acquire(blocking=False)
    planner._planner_lock.release()


def test_timeout_reaps_underlying_request_before_releasing_permit(monkeypatch):
    from app.live import planner
    monkeypatch.setenv("OPENAI_API_KEY", "qa-only-token")
    monkeypatch.setattr(planner, "_reserve_api_attempt", lambda: True, raising=False)
    monkeypatch.setattr(scene, "INFERENCE_TIMEOUT", .08)
    real_popen = subprocess.Popen
    children = []
    def popen(_argv, **kwargs):
        proc = real_popen([sys.executable, "-c", "import sys,time; sys.stdin.read(); time.sleep(30)"], **kwargs)
        children.append(proc)
        return proc
    monkeypatch.setattr(scene.subprocess, "Popen", popen)
    with pytest.raises(ValueError):
        scene._infer("QA question", [])
    assert children and children[0].poll() is not None
    assert planner._planner_lock.acquire(blocking=False)
    planner._planner_lock.release()
