"""Pre-recorded VIDEO inference pipeline (OWNER PIVOT).

The public demo ingests an uploaded MP4 rather than a live camera stream. A room
owner uploads a clip (bounded, decode-validated, stored at an opaque path); a
per-room background job then decodes it frame-by-frame, runs the SAME real YOLO
detector + tracker and the SAME temporal ``ZoneEventEngine`` as the live worker,
and publishes annotated frames, events, evidence and local notifications.

Temporal CLOCK: the dwell/cooldown timeline uses the SOURCE video clock
(frame_index / fps), never processing wall-time, so a 1.25s dwell means 1.25s of
footage regardless of how fast or slow the CPU decodes it. Wall latency is
measured separately and never presented as "real-time".

Lifecycle guarantees (LIVE-VIDEO-002/003/004):

* Every run captures an IMMUTABLE source+task snapshot (path, sha, fps, frame
  count, approved task) and builds FRESH detector/tracker/engines from it, so a
  rerun — a new approved request, a re-enable, or an owner ``/video/process`` —
  starts from clean temporal state and never reuses the room's cached engines.
* Each run is stamped with a unique monotonic EXECUTION GENERATION recorded on
  the room. Any source/task/session transition (replace, disable, pause, logout,
  expiry, room retirement, or a newer rerun) atomically revokes that generation.
  A revoked worker — even one blocked inside inference or an evidence write —
  publishes NOTHING further and deletes any artifact it was mid-writing.
* The in-flight bound counts a GLOBAL PERMIT per live worker thread and releases
  it only in an encompassing ``finally`` AFTER the thread exits, so a cancelled
  or dead worker keeps occupying a slot until it is really gone. A fresh run is
  refused with ``RuntimeError`` (busy) until capacity is actually freed, which is
  preferable to an unsafe simultaneous rerun.
* Upload validation uses ``ffprobe`` to require a supported MP4-family container,
  a genuine finite FPS inside the supported band (never an invented clock), and
  real decode of at least one frame; the job compares decoded frame count against
  reliable container metadata and reports a truncated/partial clip as an error.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.live import auth, config, rooms, worker

logger = logging.getLogger("visionflow.live.video")

_jobs: dict[str, "VideoJob"] = {}
_jobs_lock = threading.Lock()
# Count of live worker threads (reserved on start, released only after the thread
# has actually exited). A cancelled-but-not-yet-exited worker still counts, so the
# bound is a REAL in-flight limit (LIVE-VIDEO-003).
_active_permits = 0
# Monotonic execution-generation source (guarded by _jobs_lock).
_generation_counter = 0

# ffprobe integration (installed system binary; no new Python dependency). Output
# is tiny JSON; the call is bounded by a real timeout and the process is killed on
# expiry by subprocess.run.
FFPROBE_BIN = os.environ.get("VISIONFLOW_FFPROBE", "").strip() or "ffprobe"
_FFPROBE_TIMEOUT_SECONDS = float(os.environ.get("VISIONFLOW_LIVE_FFPROBE_TIMEOUT", "15"))

# Supported MP4-family container short names (ffmpeg joins the candidate demuxers
# with commas). ``mj2`` (Motion JPEG 2000) and still-image demuxers are excluded.
_SUPPORTED_CONTAINERS = {"mov", "mp4", "m4a", "3gp", "3g2"}
# Still-image / non-temporal codecs that must not masquerade as a video clip.
_STILL_IMAGE_CODECS = {
    "mjpeg", "png", "bmp", "gif", "tiff", "jpeg2000", "webp", "jpegxl", "apng",
}
# A clip is treated as truncated/incomplete when it decodes fewer than this
# fraction of the frames its reliable container metadata advertises.
_COMPLETE_RATIO = 1.0


class VideoError(ValueError):
    """Raised for an unusable / out-of-bounds uploaded clip (cleanly rejected)."""


# --------------------------------------------------------------------------- probe


def _parse_fps(value) -> float | None:
    """Parse an ffprobe frame-rate field ('num/den' or a plain number) to a finite
    positive float, or None when it is absent / zero / unparseable."""
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        text = str(value)
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            fps = float(num) / den_f
        else:
            fps = float(text)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return fps if math.isfinite(fps) and fps > 0 else None


def _ffprobe(path: Path) -> dict | None:
    """Run ffprobe with a hard timeout; return parsed JSON, None if the binary is
    unavailable (caller falls back), or raise VideoError on timeout/parse failure."""
    try:
        proc = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error", "-threads", "2",
                "-max_pixels", str(config.MAX_VIDEO_LONG_EDGE ** 2),
                "-select_streams", "v:0", "-count_frames", "-print_format", "json",
                "-show_entries", "format=format_name,duration:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames",
                str(path),
            ],
            capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        raise VideoError("video probe timed out")
    except OSError as exc:  # pragma: no cover - defensive
        raise VideoError(f"could not probe video: {exc}")
    if proc.returncode != 0 or proc.stderr.strip():
        raise VideoError("unreadable, corrupt or incomplete video")
    try:
        return json.loads(proc.stdout or "{}")
    except (ValueError, TypeError):
        raise VideoError("could not parse container metadata")


def probe(path: Path) -> dict:
    """Decode-validate an uploaded clip and return its real properties, or raise
    VideoError. Enforces a supported MP4 container, a genuine finite FPS inside the
    supported band (out-of-range is REJECTED, never clamped to an invented clock),
    frame-count / dimension / duration bounds, and that at least one frame really
    decodes. ``frame_count_reliable`` records whether the count came from
    authoritative container metadata (used later for a truncation check)."""
    info = _ffprobe(path)
    if info is None:
        raise VideoError("ffprobe is required for bounded complete video validation")

    fmt = str((info.get("format") or {}).get("format_name") or "")
    container_names = {n.strip() for n in fmt.split(",") if n.strip()}
    if not (container_names & _SUPPORTED_CONTAINERS):
        raise VideoError(f"unsupported container '{fmt or 'unknown'}'")

    video_streams = [s for s in (info.get("streams") or []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise VideoError("no video stream in container")
    stream = video_streams[0]

    codec = str(stream.get("codec_name") or "").lower()
    if codec in _STILL_IMAGE_CODECS:
        raise VideoError(f"unsupported still-image codec '{codec}'")

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoError("invalid video dimensions")
    if max(width, height) > config.MAX_VIDEO_LONG_EDGE or min(width, height) < config.MIN_VIDEO_LONG_EDGE:
        raise VideoError(f"frame dimensions {width}x{height} out of bounds")

    nominal_fps = _parse_fps(stream.get("r_frame_rate"))
    average_fps = _parse_fps(stream.get("avg_frame_rate"))
    fps = nominal_fps or average_fps
    if fps is None:
        raise VideoError("could not determine a finite source frame rate")
    if not (config.MIN_VIDEO_FPS <= fps <= config.MAX_VIDEO_FPS):
        raise VideoError(
            f"frame rate {fps:g} fps outside supported "
            f"{config.MIN_VIDEO_FPS:g}-{config.MAX_VIDEO_FPS:g} fps"
        )
    if nominal_fps and average_fps and not math.isclose(nominal_fps, average_fps, rel_tol=0.00001):
        raise VideoError("variable frame rate is unsupported; supply a constant-frame-rate MP4")

    duration = _coerce_float((info.get("format") or {}).get("duration")) or _coerce_float(stream.get("duration"))

    frame_count = 0
    reliable = False
    nb = stream.get("nb_frames")
    if nb not in (None, "", "N/A"):
        try:
            frame_count = int(nb)
            reliable = frame_count > 0
        except (ValueError, TypeError):
            frame_count = 0
    if not reliable:
        frame_count = max(1, round(duration * fps)) if duration else 1
    try:
        decoded_count = int(stream.get("nb_read_frames") or 0)
    except (ValueError, TypeError):
        raise VideoError("could not verify the complete decoded frame count")
    if decoded_count <= 0:
        raise VideoError("no decodable frames in video")
    if reliable and decoded_count != frame_count:
        raise VideoError(f"incomplete/truncated decode ({decoded_count}/{frame_count} frames)")
    frame_count = decoded_count
    reliable = True
    if frame_count > config.MAX_VIDEO_FRAMES:
        raise VideoError(f"too many frames ({frame_count} > {config.MAX_VIDEO_FRAMES})")

    if duration is None:
        duration = frame_count / fps if fps else 0.0
    if duration > config.MAX_VIDEO_DURATION_SECONDS:
        raise VideoError(f"duration {duration:.1f}s exceeds limit")

    # Actual full decoding ran in a subprocess with a kill/reap deadline, decoder
    # pixel limit and two threads. No unbounded native decoder runs on the ASGI loop.
    return {
        "fps": fps,
        "frame_count": int(frame_count),
        "frame_count_reliable": bool(reliable),
        "w": width, "h": height,
        "duration_seconds": round(float(duration), 3),
    }


def _coerce_float(value) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def _probe_cv2_fallback(path: Path) -> dict:
    """ffprobe-unavailable fallback. Still fail-closed: reject disguised JPEGs and
    non-MP4 blobs by magic bytes, and reject out-of-range FPS rather than clamp."""
    head = b""
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
    except OSError:
        raise VideoError("could not read upload")
    if head[:2] == b"\xff\xd8":
        raise VideoError("still image is not a supported video container")
    if b"ftyp" not in head:
        raise VideoError("unsupported or unrecognized container")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("file is not a decodable video")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("no decodable frames in video")
        h, w = frame.shape[:2]
        if max(w, h) > config.MAX_VIDEO_LONG_EDGE or min(w, h) < config.MIN_VIDEO_LONG_EDGE:
            raise VideoError(f"frame dimensions {w}x{h} out of bounds")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or not (config.MIN_VIDEO_FPS <= fps <= config.MAX_VIDEO_FPS):
            raise VideoError(f"frame rate {fps:g} fps outside supported range")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        reliable = frame_count > 0
        if not reliable:
            frame_count = 1
            while frame_count <= config.MAX_VIDEO_FRAMES and cap.grab():
                frame_count += 1
        if frame_count > config.MAX_VIDEO_FRAMES:
            raise VideoError(f"too many frames ({frame_count} > {config.MAX_VIDEO_FRAMES})")
        duration = frame_count / fps if fps else 0.0
        if duration > config.MAX_VIDEO_DURATION_SECONDS:
            raise VideoError(f"duration {duration:.1f}s exceeds limit")
        return {
            "fps": round(fps, 4), "frame_count": frame_count,
            "frame_count_reliable": bool(reliable),
            "w": w, "h": h, "duration_seconds": round(duration, 3),
        }
    finally:
        cap.release()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- detector


def _make_detector():
    """A FRESH detector/tracker per job (clean tracking identity on every rerun)."""
    if worker.detector_factory is not None:
        return worker.detector_factory()
    from visionflow_vision.detector import YoloByteTrackDetector
    import app.config as app_config

    detector = YoloByteTrackDetector(
        model_name=app_config.DETECTOR_MODEL,
        confidence=app_config.DETECTOR_CONFIDENCE,
        device=app_config.DETECTOR_DEVICE,
    )
    detector.model_name = str(app_config.DETECTOR_MODEL)
    model_path = Path(app_config.DETECTOR_MODEL)
    detector.weights_sha256 = sha256_file(model_path) if model_path.is_file() else None
    return detector


def _model_identity(detector) -> str:
    name = getattr(detector, "model_name", None) or type(detector).__name__
    digest = getattr(detector, "weights_sha256", None)
    return str(name) + ("@sha256:" + digest if digest else "")


# -------------------------------------------------------- immutable source snapshot


@dataclass(frozen=True)
class _SourceSnapshot:
    """An immutable snapshot of the approved source + task taken when a run starts.
    The worker reads ONLY from this, so a later source replacement can never mix an
    old source hash with a new clip's fps or vice versa (LIVE-VIDEO-002/004)."""

    path: Path
    sha256: str
    fps: float
    frame_count: int
    frame_count_reliable: bool
    w: int
    h: int
    task: dict
    workflow_hash: str = ""


def _snapshot_source(room: rooms.RoomState) -> _SourceSnapshot | None:
    with room.lock:
        path = room.video_path
        if path is None or room.task is None:
            return None
        meta = dict(room.video_meta)
        return _SourceSnapshot(
            path=path,
            sha256=room.video_sha256 or "",
            fps=float(meta.get("fps") or 25.0),
            frame_count=int(meta.get("frame_count") or 0),
            frame_count_reliable=bool(meta.get("frame_count_reliable", False)),
            w=int(meta.get("w") or 1),
            h=int(meta.get("h") or 1),
            task=copy.deepcopy(room.task),
            workflow_hash=getattr(room, "approved_hash", None) or hashlib.sha256(
                json.dumps(room.task, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        )


class VideoJob:
    def __init__(self, room: rooms.RoomState, version: int, generation: int,
                 snapshot: _SourceSnapshot):
        self.room = room
        self.version = version
        self.generation = generation
        self.snapshot = snapshot
        self.stop = threading.Event()
        self.last_notification_ts = None
        self._seen_outside: set[int] = set()
        self.thread = threading.Thread(target=self._run, name=f"video-{room.room_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.stop.set()

    # -- authority -----------------------------------------------------------

    def _authoritative(self) -> bool:
        """True only while THIS run is still the room's authorized generation and
        its task is enabled and the room is live. Checked before engine update,
        before evidence I/O and (atomically) at commit, so any acknowledged control
        stops all subsequent outputs — even if an old detector/write was blocked."""
        room = self.room
        with room.lock:
            return (
                not room.retired
                and auth.peek_session(room.owner_token) is not None
                and room.active_generation == self.generation
                and room.task_enabled
                and not room.task_paused
            )

    # -- lifecycle -----------------------------------------------------------

    def _run(self) -> None:
        room = self.room
        snap = self.snapshot
        final = "error"
        error_msg: str | None = None
        cap = None
        try:
            path = snap.path
            if path is None or not path.exists():
                raise VideoError("source video missing")
            # Detector construction and capture setup are INSIDE the try, so a
            # startup failure becomes a visible job error and still releases the
            # permit via the finally (LIVE-VIDEO-003).
            detector = _make_detector()
            model_id = _model_identity(detector)
            engines = rooms.build_engines(snap.task, snap.w, snap.h, self.version)
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise VideoError("could not open source video")

            with room.lock:
                if room.active_generation == self.generation:
                    room.job_status = "running"
                    room.job_processed = 0
                    room.job_total = snap.frame_count
                    room.job_error = None
                    room.job_version = self.version
                    room.frames_received = 0
                    room.frames_inferred = 0
                    room.events.clear()
                    room.notifications.clear()
                    # Publish the FRESH engines for this run so state/reruns reflect
                    # clean temporal objects (never the room's cached engines).
                    room.engines = engines

            fps = snap.fps
            idx = 0
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if idx >= config.MAX_VIDEO_FRAMES:
                    raise VideoError("decoded frame limit exceeded")
                video_ts = idx / fps if fps else float(idx)
                self._process_frame(frame, idx, video_ts, snap, model_id, detector, engines)
                idx += 1
                with room.lock:
                    if room.active_generation == self.generation:
                        room.job_processed = idx
                        room.frames_received = idx

            if self.stop.is_set():
                final = "cancelled"
            elif snap.frame_count_reliable and idx < snap.frame_count * _COMPLETE_RATIO:
                final = "error"
                error_msg = f"incomplete/truncated decode ({idx}/{snap.frame_count} frames)"
            else:
                final = "done"
        except VideoError as exc:
            final = "error"
            error_msg = str(exc)[:200]
        except Exception as exc:  # noqa: BLE001 — a bad frame must not crash the server
            logger.warning("video job room=%s failed: %s", room.room_id, exc)
            final = "error"
            error_msg = str(exc)[:200]
        finally:
            if cap is not None:
                cap.release()
            self._finish(final, error_msg)

    def _finish(self, final: str, error_msg: str | None) -> None:
        room = self.room
        global _active_permits
        with _jobs_lock:
            with room.lock:
                # Revocation clears authority, but this worker still owns its
                # terminal state until a genuinely newer worker replaces it.
                if _jobs.get(room.room_id) is self and room.job_version == self.version:
                    room.job_status = final
                    if error_msg is not None:
                        room.job_error = error_msg
            _active_permits = max(0, _active_permits - 1)
            if _jobs.get(room.room_id) is self:
                _jobs.pop(room.room_id, None)

    # -- per-frame processing ------------------------------------------------

    def _process_frame(self, frame, idx, video_ts, snap, model_id, detector, engines) -> None:
        room = self.room
        recv_ts = time.time()
        frame_h, frame_w = frame.shape[:2]
        t0 = time.perf_counter()
        objects = detector.track_frame(frame, video_ts)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Authority #1 (post-inference, pre-engine): a revoked run publishes nothing
        # further and recreates no private annotated frame.
        if not self._authoritative():
            self.stop.set()
            return

        emitted = []
        for engine in engines:
            engine.frame_w = frame_w
            engine.frame_h = frame_h
            # Initial presence is not an observed crossing. Entry requires this
            # same tracked identity to have been visibly outside the polygon.
            for obj in objects:
                if obj.class_name == snap.task["object_class"] and not engine.zone.contains(
                        obj.anchor, frame_w, frame_h):
                    self._seen_outside.add(obj.track_id)
            candidates = engine.update(objects, video_ts)
            if snap.task["trigger"] == "entered_zone":
                candidates = [ev for ev in candidates if ev.track_id in self._seen_outside]
            emitted.extend(candidates)

        annotated = worker._annotate(frame, objects, snap.task, emitted, None)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with room.lock:
            if self._authoritative():
                room.frames_inferred += 1
                room.last_inference_ms = inference_ms
                room.latest_meta = {"w": frame_w, "h": frame_h}
                if ok:
                    room.annotated_jpeg = bytes(buf)
                    room.annotated_meta = {
                        "ts": time.time(), "frame_index": idx,
                        "video_ts": round(video_ts, 3), "object_count": len(objects),
                    }

        if not emitted:
            return
        for ev in emitted:
            self._handle_event(ev, frame, objects, snap, model_id, idx,
                               video_ts, recv_ts, inference_ms)

    def _handle_event(self, ev, frame, objects, snap, model_id, idx, video_ts,
                      recv_ts, inference_ms) -> None:
        room = self.room
        # Authority #2 (pre evidence I/O).
        if not self._authoritative():
            return
        cooldown = snap.task.get("cooldown_seconds")
        cooldown = float(cooldown) if cooldown is not None else 10.0
        if self.last_notification_ts is not None and video_ts - self.last_notification_ts < cooldown:
            return
        ok, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_hash = hashlib.sha256(bytes(raw)).hexdigest() if ok else None
        # Write the evidence artifact but DO NOT register it until the final
        # authority check passes; if the run was revoked while the (possibly
        # blocked) write ran, delete it so no private media survives an ACK.
        fname = self._save_evidence(frame, objects, snap.task, ev, idx)
        evidence_path = (rooms.LIVE_EVIDENCE_DIR / ".pending" / fname) if fname else None
        now = time.time()
        provenance = {
            "source_sha256": snap.sha256,
            "frame_index": idx,
            "video_ts_seconds": round(video_ts, 3),
            "source_fps": snap.fps,
            "recv_ts": recv_ts,
            "processing_ts": now,
            "processing_latency_ms": round(inference_ms, 2),
            "frame_sha256": frame_hash,
            "detector_model": model_id,
            "workflow_version": self.version,
            "workflow_hash": snap.workflow_hash,
            "execution_generation": self.generation,
        }
        committed = False
        with room.lock:
            # Authority #3: atomic with the commit under the room lock.
            if self._authoritative():
                if evidence_path is not None:
                    committed_path = rooms.LIVE_EVIDENCE_DIR / evidence_path.name
                    os.replace(evidence_path, committed_path)
                    evidence_path = committed_path
                eid = room.next_event_id()
                live_ev = rooms.LiveEvent(
                    id=eid, type=ev.type.value, class_name=ev.class_name, track_id=ev.track_id,
                    detail=ev.detail, confidence=ev.confidence,
                    capture_ts=round(video_ts, 3), recv_ts=recv_ts, event_ts=now,
                    task_version=self.version, evidence_file=fname,
                    visual_observation=None, action_status="done", provenance=provenance,
                )
                room.events.append(live_ev)
                task = snap.task
                title = (task.get("name") or "VisionFlow") if task else "VisionFlow"
                room.notifications.append({
                    "id": eid, "event_id": eid, "title": title[:120],
                    "message": ev.detail, "created_at": now,
                    "video_ts_seconds": round(video_ts, 3),
                })
                if evidence_path is not None:
                    room.record_evidence(evidence_path)
                committed = True
                self.last_notification_ts = video_ts
        if not committed:
            if evidence_path is not None:
                evidence_path.unlink(missing_ok=True)
            return
        logger.info("room %s video event %s @frame %s (t=%.3fs): %s",
                    room.room_id, eid, idx, video_ts, ev.detail)

    def _save_evidence(self, frame, objects, task, ev, idx) -> str | None:
        annotated = worker._annotate(frame, objects, task, [ev], ev.track_id)
        rooms.LIVE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"room_{self.room.room_id}_ev{self.version}_f{idx}_{int(time.time() * 1000)}.jpg"
        # Writes remain outside the served directory until publication is
        # authorized and the artifact is atomically committed under room.lock.
        staging = rooms.LIVE_EVIDENCE_DIR / ".pending"
        staging.mkdir(mode=0o700, exist_ok=True)
        path = staging / fname
        if not cv2.imwrite(str(path), annotated):
            logger.warning("failed to write live evidence %s", path)
            return None
        return fname


# --------------------------------------------------------------- public controls


def start_job(room: rooms.RoomState) -> None:
    """Start a FRESH run (fresh detector/tracker/engines, unique generation,
    immutable snapshot) from frame 0 for the room's current approved task.

    Capacity is a REAL in-flight bound: a cancelled/replaced worker keeps its
    permit until its thread exits, so this raises ``RuntimeError`` (busy) rather
    than overwrite accounting or run two detectors on one slot. A newly admitted
    run revokes any prior run by stamping a new authoritative generation."""
    global _active_permits, _generation_counter
    with _jobs_lock:
        if _active_permits >= config.MAX_CONCURRENT_VIDEO_JOBS:
            raise RuntimeError("video job capacity reached")
        snap = _snapshot_source(room)
        if snap is None:
            raise RuntimeError("no approved source/task to process")
        old = _jobs.get(room.room_id)
        if old is not None:
            # Signal the prior run to stop; it keeps its permit until it exits, and
            # the new generation below immediately revokes its publish authority.
            old.cancel()
        _generation_counter += 1
        gen = _generation_counter
        with room.lock:
            version = room.task_version
            room.active_generation = gen
            # Flip to 'running' synchronously on admission so a prior run's terminal
            # status is cleared the instant a new generation is authorized (no stale
            # 'done'/'error' is observable between admission and the worker start).
            room.job_status = "running"
            room.job_error = None
        job = VideoJob(room, version, gen, snap)
        _active_permits += 1
        _jobs[room.room_id] = job
    try:
        job.start()
    except BaseException:
        with _jobs_lock:
            _active_permits = max(0, _active_permits - 1)
            if _jobs.get(room.room_id) is job:
                _jobs.pop(room.room_id, None)
        with room.lock:
            if room.active_generation == gen:
                room.active_generation = 0
        raise


def cancel_job(room_id: str) -> None:
    """Safely invalidate a room's generation and signal its worker to stop. The
    worker keeps its permit until it exits, but loses all publish authority at once
    (used by disable/pause, source replacement and room retirement)."""
    with _jobs_lock:
        job = _jobs.get(room_id)
        if job is not None:
            job.cancel()
        room = rooms.get_room(room_id)
        if room is not None:
            with room.lock:
                room.active_generation = 0


def replace_source(room: rooms.RoomState, path: Path, sha256: str, meta: dict,
                   *, authorize=None) -> None:
    """Atomically swap a room's uploaded source and revoke any in-flight run.

    This is the central API SEC's upload endpoint should call: under the job + room
    locks it cancels the old worker, resets the generation to 0 (so a still-running
    old worker cannot publish an event carrying the OLD source hash after the new
    clip is accepted), swaps the source metadata and deletes the previous file."""
    old_path = None
    with _jobs_lock:
        with rooms._rooms_lock:
            with room.lock:
                if room.retired or rooms._rooms.get(room.room_id) is not room:
                    raise VideoError("room retired during upload")
                if authorize is not None and not authorize():
                    raise VideoError("upload authority revoked")
                job = _jobs.get(room.room_id)
                if job is not None:
                    job.cancel()
                old_path = room.video_path
                room.video_path = path
                room.video_sha256 = sha256
                room.video_meta = dict(meta)
                # Approval is source-bound. A new source requires a fresh exact
                # draft and approval; an older enabled task cannot be re-enabled.
                room.task = None
                room.task_enabled = False
                room.task_paused = False
                room.approved_hash = None
                room.task_version += 1
                room.engines = []
                room.events.clear()
                room.notifications.clear()
                room.annotated_jpeg = None
                room.annotated_meta = {}
                for evidence in tuple(room._evidence_files):
                    evidence.unlink(missing_ok=True)
                room._evidence_files.clear()
                room.job_status = "idle"
                room.job_processed = 0
                room.job_total = int(meta.get("frame_count") or 0)
                room.job_error = None
                room.active_generation = 0
    if old_path is not None and old_path != path:
        try:
            old_path.unlink(missing_ok=True)
        except OSError:
            pass


def job_running(room_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(room_id)
    return bool(job and not job.stop.is_set() and job.thread.is_alive())


def _reset_for_tests() -> None:
    global _active_permits
    with _jobs_lock:
        for job in list(_jobs.values()):
            job.cancel()
        _jobs.clear()
        _active_permits = 0
