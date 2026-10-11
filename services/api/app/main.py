"""VisionFlow AI — FastAPI application (M1: video -> event -> action)."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app import config, nl_compiler, pipeline
from app.db import (
    ActionExecution,
    Notification,
    TrackEvent,
    VideoSource,
    VlmCall,
    Workflow,
    Zone,
    get_session,
    init_db,
)
from app.schemas import (
    ActionOut,
    CompileRequest,
    CompileResponse,
    EventOut,
    NotificationOut,
    SourceOut,
    WorkflowDSL,
    WorkflowOut,
    ZoneCreate,
    ZoneOut,
)
from app.vlm_router import get_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="VisionFlow AI", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    init_db()
    # Recover rows left in 'processing' by a crash/restart so they can be retried.
    pipeline.recover_interrupted_sources()
    app.mount("/media/evidence", StaticFiles(directory=str(config.EVIDENCE_DIR)), name="evidence")


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def db() -> Session:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _source_or_404(session: Session, source_id: int) -> VideoSource:
    source = session.get(VideoSource, source_id)
    if source is None:
        raise HTTPException(404, "source not found")
    return source


def _event_out(e: TrackEvent) -> EventOut:
    return EventOut(
        id=e.id,
        source_id=e.source_id,
        workflow_id=e.workflow_id,
        track_id=e.track_id,
        type=e.type,
        class_name=e.class_name,
        video_timestamp=e.video_timestamp,
        confidence=e.confidence,
        detail=e.detail,
        evidence_url=f"/media/{e.evidence_path}" if e.evidence_path else None,
        action_status=e.action_status,
        visual_status=e.visual_status,
        visual_label=e.visual_label,
        visual_confidence=e.visual_confidence,
        visual_routing=e.visual_routing,
        created_at=e.created_at,
    )


# ---------------------------------------------------------------- health


@app.get("/api/health")
def health():
    # Expose demo_safe so the frontend can verify the connected backend actually
    # enforces the demo's safety boundary before offering any demo mutation/run.
    return {
        "status": "ok",
        "service": "visionflow-api",
        "version": "0.1.0",
        "demo_safe": config.DEMO_SAFE,
    }


# ---------------------------------------------------------------- sources


@app.post("/api/sources/upload", response_model=SourceOut, status_code=201)
def upload_source(file: UploadFile = File(...), session: Session = Depends(db)):
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "only .mp4 uploads are supported")
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.mp4"
    size = 0
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file exceeds {config.MAX_UPLOAD_MB} MB limit")
            out.write(chunk)

    cap = cv2.VideoCapture(str(dest))
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "file is not a readable video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    source = VideoSource(
        type="upload",
        name=file.filename,
        path=str(dest),
        status="uploaded",
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )
    session.add(source)
    session.commit()

    snapshot = config.SNAPSHOT_DIR / f"source_{source.id}.jpg"
    cv2.imwrite(str(snapshot), first_frame)
    return source


@app.get("/api/sources", response_model=list[SourceOut])
def list_sources(session: Session = Depends(db)):
    return session.query(VideoSource).order_by(VideoSource.id.desc()).all()


@app.get("/api/sources/{source_id}", response_model=SourceOut)
def get_source(source_id: int, session: Session = Depends(db)):
    return _source_or_404(session, source_id)


@app.get("/api/sources/{source_id}/snapshot")
def get_snapshot(source_id: int, session: Session = Depends(db)):
    _source_or_404(session, source_id)
    path = config.SNAPSHOT_DIR / f"source_{source_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/sources/{source_id}/process", status_code=202)
def process_source(source_id: int, session: Session = Depends(db)):
    source = _source_or_404(session, source_id)
    if source.status == "processing" or pipeline.is_running(source_id):
        raise HTTPException(409, "source is already being processed")
    if not Path(source.path).exists():
        raise HTTPException(410, "video file no longer exists")
    started = pipeline.start_processing(source_id)
    if not started:
        raise HTTPException(409, "source is already being processed")
    return {"status": "processing", "source_id": source_id}


# ---------------------------------------------------------------- zones


@app.post("/api/zones", response_model=ZoneOut, status_code=201)
def create_zone(body: ZoneCreate, session: Session = Depends(db)):
    _source_or_404(session, body.source_id)
    zone = Zone(source_id=body.source_id, name=body.name, polygon=[list(p) for p in body.polygon])
    session.add(zone)
    session.commit()
    return ZoneOut(id=zone.id, source_id=zone.source_id, name=zone.name, polygon=body.polygon)


@app.get("/api/zones", response_model=list[ZoneOut])
def list_zones(source_id: int | None = None, session: Session = Depends(db)):
    q = session.query(Zone)
    if source_id is not None:
        q = q.filter(Zone.source_id == source_id)
    return [
        ZoneOut(id=z.id, source_id=z.source_id, name=z.name, polygon=[tuple(p) for p in z.polygon])
        for z in q.order_by(Zone.id).all()
    ]


# ---------------------------------------------------------------- workflows


@app.post("/api/workflows/compile", response_model=CompileResponse)
def compile_workflow(body: CompileRequest, session: Session = Depends(db)):
    """Compile a natural-language instruction into a strict, editable DSL draft
    (M2-AC1). Deterministic rule-based compiler (no text-model credentials).
    Unsupported/ambiguous requests are rejected; the draft is never saved or
    enabled here and no code is generated or executed."""
    # Demo-safe: return a visible unavailable response WITHOUT invoking
    # nl_compiler. The module is imported at startup but its grammar is never
    # called here, so no prompt (including the unresolved M2-001 cases) can ever
    # reach the compiler in this demo.
    if config.DEMO_SAFE:
        return CompileResponse(
            ok=False,
            draft=None,
            errors=[config.DEMO_SAFE_COMPILE_MESSAGE],
            explanation=config.DEMO_SAFE_COMPILE_MESSAGE,
        )
    _source_or_404(session, body.source_id)
    zones = [
        {"id": z.id, "name": z.name}
        for z in session.query(Zone).filter(Zone.source_id == body.source_id).all()
    ]
    result = nl_compiler.compile_text(body.text, body.source_id, zones)
    draft_model = None
    if result.ok and result.draft is not None:
        try:
            # Final strict validation: the draft must satisfy the same DSL the
            # save endpoint enforces, or it is downgraded to an error.
            draft_model = WorkflowDSL.model_validate(result.draft)
        except Exception as exc:  # noqa: BLE001
            return CompileResponse(
                ok=False,
                errors=[f"draft failed strict validation: {exc}"],
                warnings=result.warnings,
                explanation=result.explanation,
            )
    return CompileResponse(
        ok=result.ok and draft_model is not None,
        draft=draft_model,
        errors=result.errors,
        warnings=result.warnings,
        explanation=result.explanation,
    )


@app.post("/api/workflows", response_model=WorkflowOut, status_code=201)
def create_workflow(dsl: WorkflowDSL, session: Session = Depends(db)):
    _source_or_404(session, dsl.source_id)
    if config.DEMO_SAFE and dsl.visual_predicate is not None:
        raise HTTPException(403, config.DEMO_SAFE_VISUAL_MESSAGE)
    zone = session.get(Zone, dsl.trigger.zone_id)
    if zone is None or zone.source_id != dsl.source_id:
        raise HTTPException(422, "trigger.zone_id does not belong to source_id")
    wf = Workflow(source_id=dsl.source_id, name=dsl.name, dsl=dsl.model_dump(), enabled=False)
    session.add(wf)
    session.commit()
    return WorkflowOut(
        id=wf.id, source_id=wf.source_id, name=wf.name, dsl=dsl, enabled=wf.enabled,
        created_at=wf.created_at,
    )


@app.get("/api/workflows", response_model=list[WorkflowOut])
def list_workflows(source_id: int | None = None, session: Session = Depends(db)):
    q = session.query(Workflow)
    if source_id is not None:
        q = q.filter(Workflow.source_id == source_id)
    return [
        WorkflowOut(
            id=w.id, source_id=w.source_id, name=w.name,
            dsl=WorkflowDSL.model_validate(w.dsl), enabled=w.enabled, created_at=w.created_at,
        )
        for w in q.order_by(Workflow.id).all()
    ]


@app.post("/api/workflows/{workflow_id}/enable", response_model=WorkflowOut)
def enable_workflow(workflow_id: int, session: Session = Depends(db)):
    wf = session.get(Workflow, workflow_id)
    if wf is None:
        raise HTTPException(404, "workflow not found")
    # Demo-safe: a visual-predicate workflow (e.g. a legacy row created before the
    # demo) can never be armed, so it cannot admit a Teacher call at processing.
    if config.DEMO_SAFE and (wf.dsl or {}).get("visual_predicate"):
        raise HTTPException(403, config.DEMO_SAFE_VISUAL_MESSAGE)
    WorkflowDSL.model_validate(wf.dsl)  # re-validate before arming
    wf.enabled = True
    session.commit()
    return WorkflowOut(
        id=wf.id, source_id=wf.source_id, name=wf.name,
        dsl=WorkflowDSL.model_validate(wf.dsl), enabled=True, created_at=wf.created_at,
    )


@app.post("/api/workflows/{workflow_id}/disable", response_model=WorkflowOut)
def disable_workflow(workflow_id: int, session: Session = Depends(db)):
    wf = session.get(Workflow, workflow_id)
    if wf is None:
        raise HTTPException(404, "workflow not found")
    # Serialize the disable commit against the worker's dispatch admission so a
    # disable that returns here is guaranteed to suppress any not-yet-dispatched
    # queued action for this workflow (M1-010).
    with pipeline.dispatch_admission_lock:
        wf.enabled = False
        session.commit()
    return WorkflowOut(
        id=wf.id, source_id=wf.source_id, name=wf.name,
        dsl=WorkflowDSL.model_validate(wf.dsl), enabled=False, created_at=wf.created_at,
    )


# ---------------------------------------------------------------- events


@app.get("/api/events", response_model=list[EventOut])
def list_events(
    source_id: int | None = None,
    workflow_id: int | None = None,
    limit: int = 100,
    session: Session = Depends(db),
):
    q = session.query(TrackEvent)
    if source_id is not None:
        q = q.filter(TrackEvent.source_id == source_id)
    if workflow_id is not None:
        q = q.filter(TrackEvent.workflow_id == workflow_id)
    rows = q.order_by(TrackEvent.id.desc()).limit(min(limit, 500)).all()
    return [_event_out(e) for e in rows]


@app.get("/api/events/{event_id}", response_model=EventOut)
def get_event(event_id: int, session: Session = Depends(db)):
    e = session.get(TrackEvent, event_id)
    if e is None:
        raise HTTPException(404, "event not found")
    return _event_out(e)


@app.get("/api/events/{event_id}/actions", response_model=list[ActionOut])
def get_event_actions(event_id: int, session: Session = Depends(db)):
    rows = (
        session.query(ActionExecution)
        .filter(ActionExecution.event_id == event_id)
        .order_by(ActionExecution.id)
        .all()
    )
    return rows


@app.get("/api/notifications", response_model=list[NotificationOut])
def list_notifications(limit: int = 100, session: Session = Depends(db)):
    return (
        session.query(Notification)
        .order_by(Notification.id.desc())
        .limit(min(limit, 500))
        .all()
    )


# ---------------------------------------------------------------- metrics


@app.get("/api/metrics")
def metrics(session: Session = Depends(db)):
    sources = session.query(VideoSource).all()
    events = session.query(TrackEvent).count()
    actions = session.query(ActionExecution).count()
    processed = [s for s in sources if s.avg_inference_ms]
    return {
        "sources_total": len(sources),
        "events_total": events,
        "actions_total": actions,
        "avg_inference_ms": (
            sum(s.avg_inference_ms for s in processed) / len(processed) if processed else None
        ),
        "detector_model": config.DETECTOR_MODEL,
        "detector_device": config.DETECTOR_DEVICE,
        "process_fps": config.PROCESS_FPS,
        # Demo-safe: report disabled without instantiating the router/Teacher.
        "vlm": (
            {
                "status": "disabled_demo_safe",
                "teacher_available": False,
                "reason": config.DEMO_SAFE_VISUAL_MESSAGE,
            }
            if config.DEMO_SAFE
            else get_router().stats()
        ),
    }


@app.get("/api/vlm/stats")
def vlm_stats(session: Session = Depends(db)):
    """Live router counters plus persisted per-decision aggregates (M2-AC5).

    In demo-safe mode the live block reports an explicit disabled state and the
    real router/Teacher is never instantiated; persisted historical rows (if any)
    are still summarized read-only."""
    rows = session.query(VlmCall).all()
    persisted: dict[str, int] = {}
    for r in rows:
        persisted[r.status] = persisted.get(r.status, 0) + 1
    teacher_statuses: dict[str, int] = {}
    for r in rows:
        if r.teacher_status:
            teacher_statuses[r.teacher_status] = teacher_statuses.get(r.teacher_status, 0) + 1
    if config.DEMO_SAFE:
        live = {
            "status": "disabled_demo_safe",
            "teacher_provider": "disabled",
            "teacher_available": False,
            "reason": config.DEMO_SAFE_VISUAL_MESSAGE,
        }
    else:
        live = get_router().stats()
    return {
        "live": live,
        "persisted_by_route_status": persisted,
        "persisted_by_teacher_status": teacher_statuses,
        "total_decisions": len(rows),
    }


@app.get("/api/vlm/calls")
def vlm_calls(limit: int = 50, session: Session = Depends(db)):
    rows = (
        session.query(VlmCall)
        .order_by(VlmCall.id.desc())
        .limit(min(limit, 500))
        .all()
    )
    return [
        {
            "id": r.id,
            "event_id": r.event_id,
            "source_id": r.source_id,
            "workflow_id": r.workflow_id,
            "status": r.status,
            "teacher_status": r.teacher_status,
            "model": r.model,
            "prompt_version": r.prompt_version,
            "label": r.label,
            "confidence": r.confidence,
            "latency_ms": r.latency_ms,
            "roi_hash": r.roi_hash,
            "tokens_total": r.tokens_total,
            "reason": r.reason,
            "created_at": r.created_at,
        }
        for r in rows
    ]
