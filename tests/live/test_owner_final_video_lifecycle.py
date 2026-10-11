"""OWNER-FINAL regression tests for the VIDEO lifecycle (LIVE-VIDEO-002/003/004).

These drive the owned modules ``app.live.video`` and ``app.live.rooms`` directly
(fresh detector/tracker/engine state, execution generations, global permit, source
snapshot, ffprobe-based container validation) so each independently-reproduced P1
has a focused, deterministic regression.

Every clip is a freshly-generated QA_ONLY synthetic fixture — never real footage
and never a phone/Teacher inference claim. A ``ScriptedDetector`` subclass that can
be HELD lets a test hold a worker inside inference or an evidence write, acknowledge
a control (disable / logout / expiry / source replace), then release — exactly the
original review reproductions:

  * 004 baseline: probe clamped 0.5/61 fps -> 25 (accepted), accepted a JPEG as a
    1-frame video, and a 65%-truncated clip reported ``done`` after 40/60 frames.
  * 003 baseline: a detector-construction exception left ``status=idle``,
    ``error=None`` and the job registered; with capacity 1 and a held detector,
    4 restarts all started (5 live detectors on one slot).
  * 002 baseline: a disable ACK dropped the event but left an orphan evidence JPEG;
    logout/expiry still committed an event + notification + private JPEG; a source
    replacement still published an old-source event after ACK.
"""

import os
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest

from app.live import config, rooms, video, worker
from app.live.dsl import LiveTask
from visionflow_vision.detector import BaseDetectorTracker
from visionflow_vision.events import TrackedObject

FFMPEG = "/usr/bin/ffmpeg"


# --------------------------------------------------------------------------- fixtures


class ScriptedDetector(BaseDetectorTracker):
    """Deterministic QA stub — never a real-inference claim. Always reports one
    centered 'car' so a full-frame presence rule fires on the first frame."""

    def __init__(self):
        self.last_inference_ms = 0.1

    def reset(self):
        pass

    def track_frame(self, frame, timestamp):
        h, w = frame.shape[:2]
        return [TrackedObject(track_id=1, class_name="car", confidence=0.9,
                              x1=w * 0.4, y1=h * 0.4, x2=w * 0.6, y2=h * 0.9)]


def _write_clip(path, fps, frames, w=320, h=240, noise=False):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    rng = np.random.default_rng(7)
    for i in range(frames):
        if noise:
            # Rich, incompressible frames so a byte-truncation genuinely drops
            # frames (solid colors compress to a near-empty mdat).
            vw.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        else:
            vw.write(np.full((h, w, 3), 40 + i % 120, np.uint8))
    vw.release()
    return path


def _task(cls="car", trigger="presence_in_zone", **extra):
    raw = {"name": "QA_ONLY " + cls, "object_class": cls,
           "zone": {"type": "full_frame"}, "trigger": trigger,
           "action": "notify", **extra}
    return LiveTask.model_validate(raw).model_dump()


@pytest.fixture
def env(tmp_path, monkeypatch):
    rooms._reset_for_tests()
    video._reset_for_tests()
    monkeypatch.setattr(rooms, "LIVE_EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(rooms, "LIVE_VIDEO_DIR", tmp_path / "uploads")
    rooms.LIVE_EVIDENCE_DIR.mkdir(parents=True)
    rooms.LIVE_VIDEO_DIR.mkdir(parents=True)
    monkeypatch.setattr(worker, "detector_factory", ScriptedDetector)
    monkeypatch.setattr(config, "MAX_CONCURRENT_VIDEO_JOBS", 2)

    def make_room(clip_bytes, task=None):
        room = rooms.create_room("owner-token")
        src = rooms.LIVE_VIDEO_DIR / f"room_{room.room_id}_src.mp4"
        src.write_bytes(clip_bytes)
        room.video_path = src
        room.video_sha256 = video.sha256_file(src)
        room.video_meta = video.probe(src)
        if task is not None:
            with room.lock:
                rooms.activate_task(room, task, room.video_meta["w"], room.video_meta["h"])
        return room

    yield type("Env", (), {"tmp_path": tmp_path, "make_room": make_room})
    for job in list(video._jobs.values()):
        job.cancel()
        job.thread.join(3)
    video._reset_for_tests()
    rooms._reset_for_tests()


def _wait_terminal(room, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if room.job_status in ("done", "error", "cancelled"):
            return room.job_status
        time.sleep(0.02)
    return room.job_status


def _clip(env, **kw):
    return _write_clip(env.tmp_path / f"c_{kw.get('fps')}_{kw.get('frames')}.mp4", **kw).read_bytes()


# --------------------------------------------------------------- LIVE-VIDEO-004


@pytest.mark.parametrize("fps", [0.5, 61.0])
def test_owner_final_video_probe_rejects_out_of_range_fps(env, fps):
    # Baseline: both were silently retimed to 25 fps and accepted.
    clip = _write_clip(env.tmp_path / f"fps_{fps}.mp4", fps, 3)
    with pytest.raises(video.VideoError):
        video.probe(clip)


def test_owner_final_video_probe_accepts_supported_fps_with_real_rate(env):
    clip = _write_clip(env.tmp_path / "ok.mp4", 10.0, 6)
    meta = video.probe(clip)
    assert meta["fps"] == 10.0  # real source clock, not an invented default
    assert meta["frame_count_reliable"] is True
    assert meta["w"] == 320 and meta["h"] == 240


def test_owner_final_video_probe_rejects_disguised_jpeg(env):
    # Baseline: a JPEG with an .mp4 name decoded as a 1-frame "video".
    p = env.tmp_path / "fake.mp4"
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 80, np.uint8))
    assert ok
    p.write_bytes(bytes(buf))
    with pytest.raises(video.VideoError):
        video.probe(p)


def test_owner_final_video_truncated_clip_reports_error(env):
    # Baseline: a 65%-truncated faststart MP4 reported done after 40/60 frames.
    full = _write_clip(env.tmp_path / "full.mp4", 30.0, 60, noise=True)
    fast = env.tmp_path / "faststart.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-threads", "2",
                    "-i", str(full), "-c", "copy", "-movflags", "+faststart", str(fast)],
                   check=True, timeout=10)
    complete = fast.read_bytes()

    # Control: the complete clip decodes fully and finishes 'done'.
    room = env.make_room(complete, _task())
    video.start_job(room)
    assert _wait_terminal(room) == "done"
    assert room.job_processed == 60

    # Truncated: the reliable container count (60) exceeds the decodable frames, so
    # the run must fail closed rather than silently report done.
    cut = complete[: int(len(complete) * 0.65)]
    room.video_path.write_bytes(cut)
    # Stronger upload gate: bounded full decoding rejects the corrupt source
    # before it can be attached. Retain the previously validated snapshot here
    # to also reproduce a source corrupted after validation.
    with pytest.raises(video.VideoError):
        video.probe(room.video_path)
    video.start_job(room)
    assert _wait_terminal(room) == "error"
    assert room.job_processed < 60
    assert "truncated" in (room.job_error or "")


# --------------------------------------------------------------- LIVE-VIDEO-003


def test_owner_final_video_detector_startup_failure_reports_error_and_releases_capacity(env, monkeypatch):
    # Baseline: detector construction raised OUTSIDE try/finally -> status idle,
    # no job_error, job still registered, permit leaked.
    def boom():
        raise RuntimeError("QA_ONLY detector startup failure")

    monkeypatch.setattr(worker, "detector_factory", boom)
    room = env.make_room(_clip(env, fps=10.0, frames=6), _task())
    video.start_job(room)
    assert _wait_terminal(room) == "error"
    assert "startup failure" in (room.job_error or "")
    assert room.room_id not in video._jobs
    assert video._active_permits == 0  # capacity released despite the failure


def test_owner_final_video_cancelled_worker_retains_capacity_until_exit(env, monkeypatch):
    # Baseline: with capacity 1 and a held detector, 4 restarts all started and 5
    # detectors ran on one slot. A cancelled/blocked worker must keep its permit.
    monkeypatch.setattr(config, "MAX_CONCURRENT_VIDEO_JOBS", 1)
    entered = []
    release = threading.Event()

    class Held(ScriptedDetector):
        def track_frame(self, frame, timestamp):
            entered.append(threading.current_thread().name)
            assert release.wait(4)
            return super().track_frame(frame, timestamp)

    monkeypatch.setattr(worker, "detector_factory", Held)
    room = env.make_room(_clip(env, fps=10.0, frames=12), _task())
    video.start_job(room)
    end = time.time() + 2
    while not entered and time.time() < end:
        time.sleep(0.005)
    assert entered  # the one permitted worker is inside inference

    results = []
    for _ in range(4):
        try:
            video.start_job(room)
            results.append("started")
        except RuntimeError:
            results.append("busy")
    time.sleep(0.2)
    assert results == ["busy"] * 4
    assert len(entered) == 1  # exactly one detector on the single slot
    assert video._active_permits == 1

    release.set()
    _wait_terminal(room)
    end = time.time() + 3
    while video._active_permits != 0 and time.time() < end:
        time.sleep(0.02)
    assert video._active_permits == 0  # permit freed only after the thread exits


# --------------------------------------------------------------- LIVE-VIDEO-002


def _held_write(paths, entered, release):
    original = video.VideoJob._save_evidence

    def wrapper(self, *args):
        entered.set()
        assert release.wait(4)
        name = original(self, *args)
        if name:
            paths.append(rooms.LIVE_EVIDENCE_DIR / name)
        return name

    return wrapper


def test_owner_final_video_disable_ack_deletes_inflight_evidence(env, monkeypatch):
    # Baseline: disable dropped the event but left the already-written evidence JPEG.
    entered, release, paths = threading.Event(), threading.Event(), []
    monkeypatch.setattr(video.VideoJob, "_save_evidence", _held_write(paths, entered, release))
    room = env.make_room(_clip(env, fps=10.0, frames=12), _task())
    video.start_job(room)
    assert entered.wait(3)

    with room.lock:
        rooms.set_task_state(room, enabled=False)
    video.cancel_job(room.room_id)  # SEC's disable path calls this after the state change

    release.set()
    _wait_terminal(room)
    assert not room.events
    assert not room.notifications
    assert not any(p.exists() for p in paths)  # no orphan artifact survives the ACK


def test_owner_final_video_logout_retire_purges_outputs_and_evidence(env, monkeypatch):
    # Baseline: owner logout/expiry still committed an event + notification + a
    # private JPEG from the in-flight worker. Retirement must purge everything.
    entered, release, paths = threading.Event(), threading.Event(), []
    monkeypatch.setattr(video.VideoJob, "_save_evidence", _held_write(paths, entered, release))
    room = env.make_room(_clip(env, fps=10.0, frames=12), _task())
    video.start_job(room)
    assert entered.wait(3)

    # Real retirement path (owner session gone) — closes and purges the room.
    rooms.close_room(room.room_id)
    assert rooms.get_room(room.room_id) is None

    release.set()
    _wait_terminal(room)
    assert room.retired is True
    assert not room.events
    assert not room.notifications
    assert not any(p.exists() for p in paths)
    assert not room.video_path.exists()  # opaque source removed


def test_owner_final_video_expired_owner_inflight_event_is_suppressed(env, monkeypatch):
    # Baseline: an event already committed before/after retirement survived on the
    # held RoomState. Retirement must clear committed outputs too.
    enter, release = threading.Event(), threading.Event()

    class Held(ScriptedDetector):
        def track_frame(self, frame, timestamp):
            if timestamp >= 0.1:
                enter.set()
                assert release.wait(4)
            return super().track_frame(frame, timestamp)

    monkeypatch.setattr(worker, "detector_factory", Held)
    room = env.make_room(_clip(env, fps=10.0, frames=12), _task())
    video.start_job(room)
    assert enter.wait(3)
    rooms.close_room(room.room_id)
    release.set()
    _wait_terminal(room)
    assert not room.events and not room.notifications
    assert not [p for p in room._evidence_files if p.exists()]


def test_owner_final_video_source_replace_revokes_old_run(env, monkeypatch):
    # Baseline: replacing the source still let the old worker publish an event whose
    # provenance mixed the OLD source hash with the replacement's state.
    enter, release = threading.Event(), threading.Event()

    class Held(ScriptedDetector):
        def track_frame(self, frame, timestamp):
            if timestamp >= 0.1:
                enter.set()
                assert release.wait(4)
            return super().track_frame(frame, timestamp)

    monkeypatch.setattr(worker, "detector_factory", Held)
    room = env.make_room(_clip(env, fps=10.0, frames=12), _task())
    original_sha = room.video_sha256
    video.start_job(room)
    assert enter.wait(3)

    # SEC's upload endpoint calls the central atomic API to swap the source.
    new_clip = _write_clip(env.tmp_path / "replacement.mp4", 12.0, 18)
    new_path = rooms.LIVE_VIDEO_DIR / f"room_{room.room_id}_new.mp4"
    new_path.write_bytes(new_clip.read_bytes())
    new_meta = video.probe(new_path)
    video.replace_source(room, new_path, video.sha256_file(new_path), new_meta)
    assert room.video_sha256 != original_sha

    release.set()
    end = time.time() + 3
    while room.room_id in video._jobs and time.time() < end:
        time.sleep(0.02)
    assert not room.events  # the revoked old run publishes nothing after the ACK
    assert not room.notifications


def test_owner_final_video_rerun_uses_fresh_engines_and_single_notification(env):
    # Baseline: a same-task rerun reused the old engine object and produced 0 alerts.
    room = env.make_room(_clip(env, fps=10.0, frames=12), _task())
    video.start_job(room)
    assert _wait_terminal(room) == "done"
    assert len(room.notifications) == 1
    previous_engine = room.engines[0]

    video.start_job(room)  # owner /video/process rerun on the SAME approved task
    assert _wait_terminal(room) == "done"
    assert room.engines[0] is not previous_engine  # fresh engine, clean temporal state
    assert len(room.notifications) == 1


def test_owner_final_video_happy_path_event_has_snapshot_provenance(env):
    clip = _clip(env, fps=10.0, frames=12)
    room = env.make_room(clip, _task())
    expected_sha = video.sha256_file(room.video_path)
    video.start_job(room)
    assert _wait_terminal(room) == "done"
    assert room.events
    prov = room.events[0].provenance
    assert prov["source_sha256"] == expected_sha
    assert prov["source_fps"] == 10.0  # immutable snapshot fps, not room-mutable
    assert prov["workflow_version"] == room.events[0].task_version
    assert prov["frame_sha256"] and len(prov["frame_sha256"]) == 64
    assert room.events[0].evidence_file
    assert (rooms.LIVE_EVIDENCE_DIR / room.events[0].evidence_file).exists()


def test_owner_final_video_cooldown_uses_task_value_else_default():
    # Default when unspecified is 10s; a supplied value is applied exactly.
    default_engine = rooms.build_engines(_task(), 320, 240, 1)[0]
    assert default_engine.cooldown_seconds == 10.0
    custom = _task()
    custom["cooldown_seconds"] = 3.5
    custom_engine = rooms.build_engines(custom, 320, 240, 2)[0]
    assert custom_engine.cooldown_seconds == 3.5
