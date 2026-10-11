"""Single-capacity live inference worker.

One background thread serves every room: it picks each room's LATEST frame
(dropping stale ones), runs REAL YOLO detection + tracking, advances the active
task's temporal engine, writes event evidence + a local notification, and
publishes an annotated frame for the Judges view. CPU inference is serialized
(global capacity one) with a bounded thread count, so no unbounded backlog can
form — only the most recent frame per room is ever processed.

A version guard ensures that an inference started under task version N cannot
produce an event/action after the task was replaced or disabled (version != N).
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from app.live import auth, config, rooms, teacher
from visionflow_vision.detector import BaseDetectorTracker

logger = logging.getLogger("visionflow.live.worker")

# Worker liveness, surfaced in /health so a dead loop cannot masquerade as ok.
_last_loop_ts: float = 0.0
_loop_errors: int = 0
_last_sweep_ts: float = 0.0


def liveness() -> dict:
    alive = _worker_thread is not None and _worker_thread.is_alive()
    age = (time.time() - _last_loop_ts) if _last_loop_ts else None
    return {
        "alive": alive,
        "last_loop_age_s": round(age, 2) if age is not None else None,
        "loop_errors": _loop_errors,
    }

try:  # keep CPU inference sequential on the limited-RAM host
    cv2.setNumThreads(config.THREAD_COUNT)
except Exception:  # noqa: BLE001
    pass

_detector: BaseDetectorTracker | None = None
_detector_lock = threading.Lock()
_last_room: str | None = None
_seen_dedupe: dict[str, set] = {}

# Tests inject a deterministic stub detector here; production builds real YOLO.
detector_factory = None

_worker_thread: threading.Thread | None = None
_stop = threading.Event()


def _get_detector() -> BaseDetectorTracker:
    global _detector
    with _detector_lock:
        if _detector is None:
            if detector_factory is not None:
                _detector = detector_factory()
            else:
                from visionflow_vision.detector import YoloByteTrackDetector

                import app.config as app_config

                _detector = YoloByteTrackDetector(
                    model_name=app_config.DETECTOR_MODEL,
                    confidence=app_config.DETECTOR_CONFIDENCE,
                    device=app_config.DETECTOR_DEVICE,
                )
        return _detector


def start() -> None:
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _stop.clear()
    config.THREAD_COUNT  # touch for clarity
    rooms.LIVE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    _worker_thread = threading.Thread(target=_loop, name="live-worker", daemon=True)
    _worker_thread.start()
    logger.info("live worker started")


def stop() -> None:
    _stop.set()


def _loop() -> None:
    global _last_loop_ts, _loop_errors, _last_sweep_ts
    interval = 1.0 / max(0.5, config.TARGET_PROCESS_FPS)
    while not _stop.is_set():
        _last_loop_ts = time.time()
        did_work = False
        for room in rooms.iter_rooms():
            if _stop.is_set():
                break
            try:
                if _process_room_if_ready(room):
                    did_work = True
            except Exception as exc:  # noqa: BLE001 — one bad frame must never kill the worker
                _loop_errors += 1
                logger.warning("room %s: frame processing error (worker survives): %s",
                               getattr(room, "room_id", "?"), exc)
        # Periodically retire expired sessions' rooms and idle/orphaned rooms so
        # capacity is released for later users (LIVE-006).
        now = time.time()
        if now - _last_sweep_ts > 5.0:
            _last_sweep_ts = now
            try:
                active = auth.sweep_expired()
                rooms.retire_idle_and_orphaned(active, config.ROOM_IDLE_TTL_SECONDS)
            except Exception as exc:  # noqa: BLE001
                logger.warning("room sweep error: %s", exc)
        # If nothing to do, back off; otherwise pace to the target FPS.
        time.sleep(0.01 if did_work else 0.05)
        if did_work:
            time.sleep(max(0.0, interval - 0.01))


def _process_room_if_ready(room: rooms.RoomState) -> bool:
    with room.lock:
        if not room.frame_dirty or room.latest_jpeg is None:
            return False
        jpeg = room.latest_jpeg
        meta = dict(room.latest_meta)
        room.frame_dirty = False
        version = room.task_version
        enabled = room.task_enabled and not room.task_paused
        task = room.task
        engines = list(room.engines)
    process_frame(room, jpeg, meta, version, enabled, task, engines)
    return True


def process_frame(room, jpeg, meta, version, enabled, task, engines) -> None:
    """Decode + infer + evaluate one frame. Separated out so tests can drive it
    directly with a stub detector."""
    global _last_room
    if not jpeg or len(jpeg) < 100 or jpeg[:2] != b"\xff\xd8":
        # Empty / too small / not a JPEG (SOI marker FF D8). Never reaches cv2.
        logger.warning("room %s: non-JPEG or empty frame dropped", room.room_id)
        return
    try:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        logger.warning("room %s: cv2 decode error dropped: %s", room.room_id, exc)
        return
    if frame is None:
        logger.warning("room %s: undecodable frame dropped", room.room_id)
        return
    frame_h, frame_w = frame.shape[:2]
    # Inspect the ACTUAL decoded dimensions (declared metadata is untrusted): a
    # frame whose real long edge exceeds the bound never reaches inference.
    if (max(frame_h, frame_w) > config.MAX_FRAME_LONG_EDGE
            or min(frame_h, frame_w) < config.MIN_FRAME_LONG_EDGE):
        logger.warning("room %s: decoded dims %dx%d out of bounds, dropped",
                       room.room_id, frame_w, frame_h)
        with room.lock:
            room.frames_dropped += 1
        return
    now = time.time()

    detector = _get_detector()
    # Reset tracker identity when the active streaming room changes so one room's
    # tracks cannot contaminate another's (single shared detector).
    if _last_room != room.room_id:
        detector.reset()
        _last_room = room.room_id
    t0 = time.perf_counter()
    objects = detector.track_frame(frame, now)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    emitted = []
    focus_track = None
    if enabled and engines and task is not None:
        for engine in engines:
            # The task may have been activated before any frame arrived (unknown
            # resolution) or the stream resolution may differ; bind the engine to
            # the ACTUAL decoded frame size so normalized zones map correctly.
            engine.frame_w = frame_w
            engine.frame_h = frame_h
            for ev in engine.update(objects, now):
                emitted.append(ev)

    # Annotate for the Judges view (zone + boxes + focus highlight).
    annotated = _annotate(frame, objects, task, emitted, focus_track)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    annotated_jpeg = bytes(buf) if ok else None

    with room.lock:
        room.frames_inferred += 1
        room.last_inference_ms = inference_ms
        if annotated_jpeg is not None:
            room.annotated_jpeg = annotated_jpeg
            room.annotated_meta = {
                "ts": now,
                "object_count": len(objects),
                "seq": meta.get("seq"),
                "capture_ts": meta.get("capture_ts"),
                "recv_ts": meta.get("recv_ts"),
            }
        # Version guard: a task replaced/disabled during this inference must not
        # produce events or actions.
        current_version = room.task_version
        current_enabled = room.task_enabled and not room.task_paused

    if not emitted:
        return
    if current_version != version or not current_enabled:
        logger.info("room %s: dropping %d event(s) from stale task v%s (now v%s enabled=%s)",
                    room.room_id, len(emitted), version, current_version, current_enabled)
        return

    for ev in emitted:
        _handle_event(room, ev, frame, objects, task, version, meta, now, inference_ms)


def _handle_event(room, ev, frame, objects, task, version, meta, now, inference_ms) -> None:
    seen = _seen_dedupe.setdefault(room.room_id, set())
    key = f"v{version}:{ev.dedupe_key}"
    if key in seen:
        return
    seen.add(key)
    if len(seen) > 1000:
        seen.clear()

    evidence_file = _save_evidence(room, frame, objects, task, ev, version)

    # A generation guard carried into Teacher admission: no new provider request
    # may start once this task version has been replaced/disabled/paused.
    def _still_current() -> bool:
        with room.lock:
            return room.task_version == version and room.task_enabled and not room.task_paused

    observation = None
    vo = task.get("visual_observation") if task else None
    if vo:
        if teacher.available():
            try:
                images = _collect_images(room, frame, vo.get("temporal"))
                obs = teacher.observe(room.room_id, vo["question"], vo["labels"], images,
                                      should_proceed=_still_current)
                observation = {
                    "status": obs.status, "label": obs.label, "confidence": obs.confidence,
                    "uncertain": obs.uncertain, "in_schema": obs.in_schema,
                    "image_hash": obs.image_hash, "image_count": obs.image_count,
                    "model": obs.model, "revision": obs.revision,
                    "latency_ms": obs.latency_ms, "reason": obs.reason,
                }
            except Exception as exc:  # noqa: BLE001
                observation = {"status": "error", "reason": str(exc)[:150], "uncertain": True}
        else:
            observation = {"status": "unavailable", "uncertain": True,
                           "reason": "Teacher endpoint not configured"}

    # Semantic gate: a task carrying a visual_observation is a SEMANTIC-CONDITIONAL
    # task. The geometric candidate alone (presence/entry) is only ADMISSION to
    # verification — it must NOT authorize the action. Only a CONFIRMED in-schema
    # match of the requested condition dispatches the local notification. Unknown,
    # unavailable, uncertain, out-of-schema, timeout or stale => the event is
    # recorded as an informational observation only, with NO notification.
    confirmed = _semantic_confirmed(vo, observation) if vo else True
    action_status = "done" if confirmed else "observation_only"

    with room.lock:
        # Re-check version under lock before committing the event/action.
        if room.task_version != version or not (room.task_enabled and not room.task_paused):
            return
        eid = room.next_event_id()
        capture_ts = meta.get("capture_ts")
        recv_ts = meta.get("recv_ts") or now
        live_ev = rooms.LiveEvent(
            id=eid, type=ev.type.value, class_name=ev.class_name, track_id=ev.track_id,
            detail=ev.detail, confidence=ev.confidence,
            capture_ts=capture_ts if isinstance(capture_ts, (int, float)) else recv_ts,
            recv_ts=recv_ts, event_ts=now, task_version=version,
            evidence_file=evidence_file, visual_observation=observation,
            action_status=action_status,
        )
        room.events.append(live_ev)
        if confirmed:
            # Local notification action (side-effect-free, no outbound delivery).
            title = (task.get("name") or "VisionFlow") if task else "VisionFlow"
            room.notifications.append({
                "id": eid, "event_id": eid, "title": title[:120],
                "message": ev.detail, "created_at": now,
            })
    logger.info("room %s event %s (%s): %s", room.room_id, eid, action_status, ev.detail)


def _semantic_confirmed(vo: dict | None, observation: dict | None) -> bool:
    """A semantic condition is confirmed ONLY when the provider returned a valid,
    in-schema answer whose label matches the requested target state. The target is
    the first requested label by convention. Uncertain/unavailable/out-of-schema/
    unknown/timeout/stale never confirms (fail-closed). A self-reported confidence
    score alone never confirms — the geometric candidate is the corroboration."""
    if not vo:
        return True
    if not observation or observation.get("status") != "ok":
        return False
    if not observation.get("in_schema"):
        return False
    labels = [str(x).strip().lower() for x in (vo.get("labels") or [])]
    target = labels[0] if labels else None
    return bool(target) and (observation.get("label") == target)


def _collect_images(room, frame, temporal) -> list[bytes]:
    """Current frame JPEG, plus one earlier frame from the bounded deque when a
    temporal (before/after) observation is requested and one exists."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cur = bytes(buf) if ok else b""
    images = []
    if temporal:
        with room.lock:
            if room.temporal:
                before = room.temporal[0].get("jpeg")
                if before:
                    images.append(before)
    if cur:
        images.append(cur)
    return images


def _save_evidence(room, frame, objects, task, ev, version) -> str | None:
    annotated = _annotate(frame, objects, task, [ev], ev.track_id)
    rooms.LIVE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"room_{room.room_id}_ev{version}_{int(time.time()*1000)}.jpg"
    path = rooms.LIVE_EVIDENCE_DIR / fname
    if not cv2.imwrite(str(path), annotated):
        logger.warning("failed to write live evidence %s", path)
        return None
    room.record_evidence(path)
    return fname


def _annotate(frame, objects, task, events, focus_track):
    img = frame.copy()
    h, w = img.shape[:2]
    event_tracks = {e.track_id for e in events if getattr(e, "track_id", None) is not None}
    # Draw the task zone when it is not the full frame.
    if task and task.get("zone", {}).get("type") == "polygon":
        pts = np.array([[int(x * w), int(y * h)] for x, y in task["zone"]["polygon"]],
                       dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 200, 255), 2)
    # A persistent "focus" highlight is a FULFILLED focus action; for a semantic-
    # conditional task (one carrying a visual_observation) it must NOT be applied
    # by mere class/ROI selection — only confirmed event tracks are highlighted.
    persistent_focus = bool(task and task.get("focus") and not task.get("visual_observation"))
    for obj in objects:
        focus = (persistent_focus and obj.class_name == task.get("object_class")) \
            or obj.track_id in event_tracks
        color = (0, 0, 255) if focus else (0, 255, 0)
        thickness = 3 if focus else 2
        cv2.rectangle(img, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), color, thickness)
        cv2.putText(img, f"{obj.class_name} #{obj.track_id}",
                    (int(obj.x1), max(15, int(obj.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def _reset_for_tests() -> None:
    global _detector, _last_room
    with _detector_lock:
        _detector = None
    _last_room = None
    _seen_dedupe.clear()
