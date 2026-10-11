"""Background video-processing pipeline.

Video -> frame sampling -> YOLO detection + ByteTrack -> zone event engines
(one per enabled workflow) -> event persistence -> evidence JPEG -> action.

Runs in a daemon thread per source; one run per source at a time. All
timestamps are video time (frame_idx / fps) so results are reproducible.
"""

from __future__ import annotations

import logging
import threading

import cv2

from app import config
from app.actions import execute_action
from app.db import (
    ActionExecution,
    TrackEvent,
    VideoSource,
    VlmCall,
    Workflow,
    Zone as ZoneRow,
    get_session,
)
from app.vlm_router import get_router
from visionflow_vision.detector import BaseDetectorTracker
from visionflow_vision.events import EventType, ZoneEventEngine
from visionflow_vision.zones import Zone

logger = logging.getLogger("visionflow.pipeline")

# Keep CPU inference sequential on the limited-RAM host.
try:  # pragma: no cover - environment dependent
    cv2.setNumThreads(config.THREAD_COUNT)
except Exception:  # noqa: BLE001
    pass

_running: set[int] = set()
_running_lock = threading.Lock()

_detector: BaseDetectorTracker | None = None
_detector_lock = threading.Lock()

# Serializes a complete processing session (detector reset + every frame call)
# so concurrent sources cannot reset or contaminate each other's shared tracker
# state. This also keeps CPU/RAM use sequential, which suits the M1 target.
_session_lock = threading.Lock()

# Serializes a workflow disable (API) against an action's dispatch admission
# (worker) so a disable that completes before dispatch is always observed and
# the queued action is suppressed, closing the check-commit-dispatch race
# (M1-010). The API disable handler acquires this lock around its commit; the
# worker acquires it around the authoritative enabled recheck + dispatch.
dispatch_admission_lock = threading.Lock()

# Tests may replace this factory to inject a deterministic stub detector.
# Production always builds the real YOLO detector.
detector_factory = None


def _get_detector() -> BaseDetectorTracker:
    global _detector
    with _detector_lock:
        if _detector is None:
            if detector_factory is not None:
                _detector = detector_factory()
            else:
                from visionflow_vision.detector import YoloByteTrackDetector

                _detector = YoloByteTrackDetector(
                    model_name=config.DETECTOR_MODEL,
                    confidence=config.DETECTOR_CONFIDENCE,
                    device=config.DETECTOR_DEVICE,
                )
        return _detector


def start_processing(source_id: int) -> bool:
    """Start processing a source in a background thread. False if already running."""
    with _running_lock:
        if source_id in _running:
            return False
        _running.add(source_id)
    thread = threading.Thread(target=_process_source, args=(source_id,), daemon=True)
    thread.start()
    return True


def is_running(source_id: int) -> bool:
    with _running_lock:
        return source_id in _running


def recover_interrupted_sources() -> int:
    """Mark sources stuck in 'processing' (from a crash/restart) as failed so
    they can be retried. Single-worker model: on startup nothing is running, so
    any persisted 'processing' row represents an interrupted run. Returns the
    number of rows recovered."""
    session = get_session()
    try:
        stale = (
            session.query(VideoSource)
            .filter(VideoSource.status == "processing")
            .all()
        )
        count = 0
        for source in stale:
            if is_running(source.id):
                continue
            source.status = "failed"
            source.error = "processing interrupted by restart; retry to reprocess"
            count += 1
        if count:
            session.commit()
            logger.info("recovered %d interrupted processing source(s)", count)
        return count
    finally:
        session.close()


def _workflow_enabled(workflow_id: int) -> bool:
    """Authoritative, fresh read of a workflow's current enabled state.

    Uses its own short-lived session so a disable committed by the API during
    an active run is observed immediately (no stale worker-session snapshot)."""
    session = get_session()
    try:
        return bool(
            session.query(Workflow.enabled)
            .filter(Workflow.id == workflow_id)
            .scalar()
        )
    finally:
        session.close()


def _build_engines(session, source: VideoSource) -> list[tuple[Workflow, ZoneEventEngine]]:
    engines: list[tuple[Workflow, ZoneEventEngine]] = []
    workflows = (
        session.query(Workflow)
        .filter(Workflow.source_id == source.id, Workflow.enabled.is_(True))
        .all()
    )
    for wf in workflows:
        dsl = wf.dsl
        # Demo-safe guard: a Teacher-routed visual-predicate workflow is NEVER
        # armed in demo mode, even if it was enabled through a pre-demo NL path or
        # inserted as a legacy row. It is refused (skipped) — not silently coerced
        # into a geometric rule — so zero Teacher admission can occur (M2-AC2).
        if config.DEMO_SAFE and dsl.get("visual_predicate"):
            logger.info(
                "demo-safe: skipping visual-predicate workflow %s (no Teacher admission)",
                wf.id,
            )
            continue
        zone_row = session.get(ZoneRow, dsl["trigger"]["zone_id"])
        if zone_row is None or zone_row.source_id != source.id:
            logger.warning("workflow %s references missing zone; skipping", wf.id)
            continue
        zone = Zone(
            id=zone_row.id,
            name=zone_row.name,
            polygon=tuple((float(x), float(y)) for x, y in zone_row.polygon),
        )
        engine = ZoneEventEngine(
            zone=zone,
            object_class=dsl["object_class"],
            event_type=EventType(dsl["trigger"]["type"]),
            frame_w=source.width or 1,
            frame_h=source.height or 1,
            dwell_seconds=dsl["trigger"].get("duration_seconds") or 0.0,
            count_threshold=dsl["trigger"].get("threshold") or 0,
            cooldown_seconds=dsl.get("cooldown_seconds", 10.0),
            rule_key=f"src{source.id}-wf{wf.id}",
        )
        engines.append((wf, engine))
    return engines


def _save_evidence(frame, objects, zone: Zone, event, frame_w: int, frame_h: int) -> str:
    """Draw zone + boxes on the frame and save a JPEG. Returns relative path."""
    img = frame.copy()
    import numpy as np

    pts = np.array(
        [[int(x * frame_w), int(y * frame_h)] for x, y in zone.polygon], dtype=np.int32
    )
    cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
    for obj in objects:
        color = (0, 0, 255) if obj.track_id == event.track_id else (0, 255, 0)
        cv2.rectangle(img, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), color, 2)
        cv2.putText(
            img,
            f"{obj.class_name} #{obj.track_id}",
            (int(obj.x1), max(15, int(obj.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cv2.putText(
        img, event.detail[:80], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    fname = f"event_{event.dedupe_key.replace(':', '_').replace('/', '_')}.jpg"
    out_path = config.EVIDENCE_DIR / fname
    # cv2.imwrite returns False (never raises) on an unwritable/full destination.
    # Surface it so we never publish an evidence URL that 404s (M1-009).
    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f"failed to write evidence image: {out_path}")
    return f"evidence/{fname}"


def _record_source_failure(session, source_id: int, message: str) -> None:
    """Mark a source failed after an exception. The worker's session may be in a
    broken/aborted transaction (e.g. SQLite rejected a bad value at commit), so
    roll it back first; if it still cannot persist, use a fresh session. This
    guarantees an interrupted run never leaves the source stuck in 'processing'
    with no visible error (M1-008)."""
    try:
        session.rollback()
    except Exception:  # noqa: BLE001
        pass
    try:
        source = session.get(VideoSource, source_id)
        if source is not None:
            source.status = "failed"
            source.error = message
            session.commit()
            return
    except Exception:  # noqa: BLE001
        logger.exception("failed to record source failure on worker session; retrying fresh")
        try:
            session.rollback()
        except Exception:  # noqa: BLE001
            pass
    fresh = get_session()
    try:
        source = fresh.get(VideoSource, source_id)
        if source is not None:
            source.status = "failed"
            source.error = message
            fresh.commit()
    except Exception:  # noqa: BLE001
        logger.exception("could not record source failure even with a fresh session")
    finally:
        fresh.close()


def _encode_roi(frame, objects, event) -> bytes:
    """Encode the triggering object's ROI as JPEG bytes for the Teacher.

    Uses the bounding box of the object whose track_id matches the event; for
    aggregate (track-less) events the whole frame is used. The ROI is the only
    pixels disclosed to the provider (M2 privacy boundary)."""
    import numpy as np

    img = frame
    if event.track_id is not None:
        match = next((o for o in objects if o.track_id == event.track_id), None)
        if match is not None:
            h, w = frame.shape[:2]
            x1 = max(0, int(match.x1)); y1 = max(0, int(match.y1))
            x2 = min(w, int(match.x2)); y2 = min(h, int(match.y2))
            if x2 > x1 and y2 > y1:
                img = frame[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("failed to encode ROI for visual routing")
    return bytes(buf)


def _route_visual(session, wf: Workflow, event: TrackEvent, frame, objects) -> bool:
    """Route a candidate event's ROI through the VLM router (M2-AC2/AC3/AC5).

    Returns True if the visual predicate was CONFIRMED (teacher status ok and
    label == expected), meaning the action may fire. Any suppression, timeout,
    error, unknown, invalid response or missing credential returns False and the
    event is recorded as not actioned — a failure can never confirm an action.

    A VlmCall row and the event's visual_* provenance are always persisted."""
    vp = wf.dsl["visual_predicate"]
    labels = [str(l) for l in vp["labels"]]
    expected = str(vp["expected_label"]).strip().lower()
    task_key = f"src{event.source_id}-wf{wf.id}"
    # Cooldown scope is stable per (source, workflow) so repeated candidates for
    # the same rule are throttled across distinct events (M2-AC4). The per-event
    # ``dedupe_key`` is unique and never repeats, so keying cooldown on it would
    # silently disable cooldown entirely (the reviewed defect). Keeping it at the
    # task/source level means the documented cooldown actually suppresses the
    # next eligible candidate of the same workflow within the window.
    scope = task_key
    router = get_router()

    # M2-007: serialize provider-request admission with workflow disable. A
    # disable that has already committed (under ``dispatch_admission_lock``) must
    # prevent any new Teacher call and the associated image disclosure. We recheck
    # the authoritative enabled state under the same lock before admitting the
    # route; if revoked we record the candidate as cancelled WITHOUT any Teacher
    # call. (The real adapter additionally rechecks ``admit`` at the transport
    # boundary for a disable that lands mid-route.)
    with dispatch_admission_lock:
        admitted = _workflow_enabled(wf.id)
    if not admitted:
        logger.info(
            "workflow %s disabled before visual routing; cancelling event %s with no Teacher call",
            wf.id, event.id,
        )
        event.visual_status = "cancelled"
        event.action_status = "cancelled"
        event.visual_routing = {
            "route_status": "cancelled",
            "reason": "workflow disabled before provider admission",
        }
        session.add(
            VlmCall(
                event_id=event.id, source_id=event.source_id, workflow_id=wf.id,
                task_key=task_key, status="cancelled",
                reason="workflow disabled before provider admission",
            )
        )
        session.add(
            ActionExecution(
                event_id=event.id, action_type=wf.dsl["action"]["type"],
                status="cancelled", attempt_count=0,
                error="workflow disabled before provider admission",
            )
        )
        session.commit()
        return False

    try:
        image_bytes = _encode_roi(frame, objects, event)
        decision = router.route(
            task_key=task_key,
            scope=scope,
            prompt=vp["prompt"],
            labels=labels,
            image_bytes=image_bytes,
            admit=lambda: _workflow_enabled(wf.id),
            sample_interval_s=vp.get("sample_interval_seconds"),
            cooldown_s=vp.get("cooldown_seconds"),
            budget=vp.get("budget"),
        )
    except Exception as exc:  # noqa: BLE001 — never crash the pipeline on routing
        logger.warning("visual routing failed for event %s: %s", event.id, exc)
        event.visual_status = "error"
        event.visual_routing = {"error": str(exc)[:300]}
        event.action_status = "skipped_visual"
        session.add(
            VlmCall(
                event_id=event.id, source_id=event.source_id, workflow_id=wf.id,
                task_key=task_key, status="failed", reason=str(exc)[:300],
            )
        )
        session.commit()
        return False

    result = decision.result
    confirmed = bool(result and result.confirmed and result.label == expected)
    # Visual status reflects the teacher result when a call/cache/dedup produced
    # one, otherwise the suppression reason.
    if result is not None:
        event.visual_status = result.status
        event.visual_label = result.label
        event.visual_confidence = result.confidence
    else:
        event.visual_status = decision.status  # suppressed_*
    event.visual_routing = {
        "route_status": decision.status,
        "reason": decision.reason,
        "roi_hash": decision.roi_hash,
        "model": result.model if result else None,
        "prompt_version": result.prompt_version if result else None,
        "latency_ms": result.latency_ms if result else None,
        "expected_label": expected,
        "confirmed": confirmed,
    }
    session.add(
        VlmCall(
            event_id=event.id,
            source_id=event.source_id,
            workflow_id=wf.id,
            task_key=task_key,
            status=decision.status,
            teacher_status=result.status if result else None,
            model=result.model if result else None,
            prompt_version=result.prompt_version if result else None,
            label=result.label if result else None,
            confidence=result.confidence if result else None,
            latency_ms=result.latency_ms if result else None,
            roi_hash=decision.roi_hash,
            tokens_total=(result.usage or {}).get("total_tokens") if result else None,
            reason=decision.reason,
        )
    )
    if not confirmed:
        # A mid-route disable revoked at the transport boundary is recorded as
        # cancelled (M2-007); every other non-confirmation is skipped_visual.
        event.action_status = "cancelled" if decision.status == "cancelled" else "skipped_visual"
        session.commit()
        logger.info(
            "event %s visual predicate NOT confirmed (route=%s teacher=%s label=%s); no action",
            event.id, decision.status, result.status if result else "-",
            result.label if result else "-",
        )
    else:
        session.commit()
    return confirmed


def _dispatch_action(session, wf: Workflow, event: TrackEvent) -> None:
    """Run the workflow's action for one persisted event.

    Admission is serialized against workflow disable (M1-010): under
    ``dispatch_admission_lock`` we re-read the authoritative enabled state and,
    if the workflow was disabled before this dispatch, suppress the queued
    action explicitly (recorded as a ``cancelled`` ActionExecution and a
    ``cancelled`` event action_status) instead of running it. Because the API
    disable handler commits inside the same lock, a disable that completed
    before dispatch is always observed here."""
    with dispatch_admission_lock:
        if not _workflow_enabled(wf.id):
            logger.info(
                "workflow %s disabled before dispatch; suppressing action for event %s",
                wf.id, event.id,
            )
            session.add(
                ActionExecution(
                    event_id=event.id,
                    action_type=wf.dsl["action"]["type"],
                    status="cancelled",
                    attempt_count=0,
                    error="workflow disabled before action dispatch",
                )
            )
            event.action_status = "cancelled"
            session.commit()
            return
        execute_action(
            session, event, wf.dsl["action"]["type"], wf.dsl["action"].get("config", {})
        )


def _resume_pending_event(session, wf: Workflow, existing: TrackEvent) -> bool:
    """If a previously committed event never had its action dispatched (e.g. the
    process was interrupted between the event commit and action dispatch),
    complete it now instead of silently skipping (M1-007).

    Returns True if the event was handled as a resume (completed or already
    complete), so the caller treats this dedupe key as done. Completed events
    (action_status != 'pending' or an existing ActionExecution row) are left
    untouched to preserve idempotent deduplication."""
    has_execution = (
        session.query(ActionExecution.id)
        .filter(ActionExecution.event_id == existing.id)
        .first()
        is not None
    )
    if existing.action_status != "pending" or has_execution:
        return True  # already dispatched; keep dedupe, do not re-run
    # A visual-predicate workflow must never dispatch an action without a VALID,
    # persisted confirmation (M2-006). Two interrupted cases:
    #   1. The Teacher already confirmed the predicate (visual_status == 'ok' with
    #      the still-matching expected label) but the process died before action
    #      dispatch -> resume exactly once through the guarded idempotent dispatch
    #      path (which rechecks current approval).
    #   2. The candidate was never validly confirmed (routing never completed, was
    #      suppressed, errored or returned a non-matching label) -> we have no
    #      frame to safely re-route during recovery, so record an explicit
    #      interrupted/error state instead of leaving it silently pending or
    #      firing an unverified action.
    if wf.dsl.get("visual_predicate"):
        vp = wf.dsl["visual_predicate"]
        expected = str(vp.get("expected_label", "")).strip().lower()
        routing = existing.visual_routing or {}
        confirmed = (
            existing.visual_status == "ok"
            and existing.visual_label is not None
            and str(existing.visual_label).strip().lower() == expected
            and routing.get("confirmed") is True
        )
        if confirmed:
            logger.info(
                "resuming confirmed visual event %s through guarded dispatch", existing.id
            )
            _dispatch_action(session, wf, existing)
            return True
        logger.info(
            "visual event %s lacks a valid confirmation at recovery; marking interrupted "
            "(no auto-dispatch, no re-route without a frame)",
            existing.id,
        )
        existing.action_status = "interrupted_unverified"
        if existing.visual_status is None:
            existing.visual_status = "interrupted"
        session.commit()
        return True
    logger.info("resuming undispatched action for event %s", existing.id)
    _dispatch_action(session, wf, existing)
    return True


def _process_source(source_id: int) -> None:
    session = get_session()
    session_locked = False
    try:
        source = session.get(VideoSource, source_id)
        if source is None:
            return
        source.status = "processing"
        source.frames_processed = 0
        source.error = None
        session.commit()

        cap = cv2.VideoCapture(source.path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {source.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        source.fps, source.width, source.height, source.frame_count = fps, frame_w, frame_h, total
        session.commit()

        engines = _build_engines(session, source)
        if not engines:
            logger.info("source %s: no enabled workflows; detection-only pass", source_id)

        # Serialize the whole session: reset + every frame call must be
        # isolated from any other source so tracker identities cannot be
        # contaminated by a concurrent reset (M1-004).
        _session_lock.acquire()
        session_locked = True
        detector = _get_detector()
        detector.reset()

        step = max(1, round(fps / config.PROCESS_FPS))
        frame_idx = 0
        processed = 0
        inference_ms_total = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            timestamp = frame_idx / fps
            objects = detector.track_frame(frame, timestamp)
            inference_ms_total += getattr(detector, "last_inference_ms", 0.0)
            processed += 1

            for wf, engine in engines:
                # Advance the engine exactly once per frame to keep debounce /
                # cooldown state consistent.
                emitted = engine.update(objects, timestamp)
                for ev in emitted:
                    # Re-check authoritative enabled state before EACH event so a
                    # disable committed by the API mid-frame (e.g. between two
                    # events emitted on the same frame) revokes the remaining
                    # events and actions (M1-003).
                    if not _workflow_enabled(wf.id):
                        break
                    existing = (
                        session.query(TrackEvent)
                        .filter(TrackEvent.dedupe_key == ev.dedupe_key)
                        .first()
                    )
                    if existing is not None:
                        _resume_pending_event(session, wf, existing)
                        continue
                    row = TrackEvent(
                        source_id=source.id,
                        workflow_id=wf.id,
                        track_id=ev.track_id,
                        type=ev.type.value,
                        class_name=ev.class_name,
                        video_timestamp=ev.timestamp,
                        confidence=ev.confidence,
                        detail=ev.detail,
                        dedupe_key=ev.dedupe_key,
                    )
                    session.add(row)
                    session.flush()
                    # Evidence must be written before we commit the event so an
                    # unwritable destination rolls back the whole event instead
                    # of publishing a dangling evidence URL (M1-009).
                    row.evidence_path = _save_evidence(
                        frame, objects, engine.zone, row, frame_w, frame_h
                    )
                    session.commit()
                    # Geometric rules dispatch immediately and make ZERO VLM
                    # calls. A workflow carrying a visual_predicate instead routes
                    # the candidate's ROI to the Teacher and only dispatches the
                    # action if the predicate is confirmed (M2-AC2/AC3).
                    if wf.dsl.get("visual_predicate"):
                        if _route_visual(session, wf, row, frame, objects):
                            _dispatch_action(session, wf, row)
                    else:
                        _dispatch_action(session, wf, row)
                    logger.info("event %s: %s", row.id, ev.detail)

            if processed % 25 == 0:
                source.frames_processed = processed
                session.commit()
            frame_idx += 1

        cap.release()
        # frame_idx counted every successfully decoded frame (sampled or not).
        frames_read = frame_idx
        source.frames_processed = processed
        source.avg_inference_ms = inference_ms_total / processed if processed else None

        # Distinguish a clean EOF from a premature decode termination (M1-013).
        # If the container reports a reliable frame count and we decoded
        # substantially fewer frames, the upload was truncated/corrupt: surface
        # a failed source instead of silently reporting "processed". Events
        # emitted before the truncation point remain committed (visible partial
        # progress); the source row makes the shortfall explicit.
        if total and frames_read < total * config.DECODE_MIN_COMPLETENESS:
            source.status = "failed"
            source.error = (
                f"incomplete video: decoded {frames_read} of {total} frames reported "
                "by container metadata (truncated or corrupt upload)"
            )
            session.commit()
            logger.warning(
                "source %s incomplete: decoded %d of %d frames (min %.0f%%)",
                source_id, frames_read, total, config.DECODE_MIN_COMPLETENESS * 100,
            )
            return

        source.status = "processed"
        session.commit()
        logger.info(
            "source %s done: %d frames analyzed (%d decoded), avg inference %.1f ms",
            source_id, processed, frames_read, source.avg_inference_ms or 0,
        )
    except Exception as exc:  # noqa: BLE001 — surface failure on the source row
        logger.exception("processing failed for source %s", source_id)
        _record_source_failure(session, source_id, str(exc)[:1000])
    finally:
        if session_locked:
            _session_lock.release()
        session.close()
        with _running_lock:
            _running.discard(source_id)
