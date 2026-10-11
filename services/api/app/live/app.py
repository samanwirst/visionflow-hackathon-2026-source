"""Public LIVE Camera + Judges FastAPI sub-application.

This is an ISOLATED public surface. It contains ONLY authenticated live-demo
endpoints and never imports or re-exposes the legacy unauthenticated upload /
control / media endpoints from ``app.main``. Deploy this app (not the private
fallback) behind the public HTTPS tunnel.

Security model (see auth.py): shared PIN -> role; opaque per-room join code ->
room membership; HttpOnly+Secure+SameSite session cookie; double-submit CSRF +
Origin allowlist on every mutation and WebSocket upgrade; bounded sessions,
rooms, login attempts, and frame sizes.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import stat
import threading
import time
from pathlib import Path

from fastapi import (
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import secrets

from app.live import auth, config, planner, rooms, scene, teacher, video, worker
from app.live.dsl import LiveTask, TaskStateRequest

logger = logging.getLogger("visionflow.live.app")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Argus Video Analytics", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    rooms.LIVE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    rooms.LIVE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    # Wire session teardown to room cleanup (logout / expiry) — LIVE-006.
    auth.on_session_dropped = _drop_membership
    worker.start()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------- middleware


async def _send_simple_json(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})


class _BodyBoundMiddleware:
    """Pure-ASGI pre-parse body bound.

    Counts the ACTUAL bytes received from the ASGI ``receive`` channel and rejects
    an oversize JSON body with 413 BEFORE any JSON buffering/parsing, independent of
    whether a (possibly lying/absent) Content-Length is present — so a chunked
    request cannot bypass the memory/parse bound (LIVE-VIDEO-007). A bounded receive
    deadline prevents a trickle body from holding a worker. The large streamed video
    upload has its OWN byte-by-byte cap in its route and is passed straight through.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if os.environ.get("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "").lower() == "true":
            path = scope.get("path", "")
            if scope.get("type") == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            public_static = {"/static/demo.html", "/static/demo.js", "/static/style.css", "/static/argus-logo.jpg"}
            legacy_api = {"/api/live/login", "/api/live/rooms", "/api/live/rooms/join", "/camera"}
            if path in legacy_api or (path.startswith("/static/") and path not in public_static):
                await _send_simple_json(send, 404, "not found")
                return
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "")
        path = scope.get("path", "")
        is_video_upload = method == "POST" and path.endswith("/video")
        if method not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        cl = headers.get("content-length")

        if is_video_upload:
            # Streamed under the route's own cap; only an early Content-Length reject
            # here (never buffer the large body).
            if cl is not None:
                try:
                    if int(cl) > config.MAX_VIDEO_BYTES:
                        await _send_simple_json(send, 413, "video exceeds size limit")
                        return
                except ValueError:
                    await _send_simple_json(send, 400, "invalid content-length")
                    return
            await self.app(scope, receive, send)
            return

        cap = config.MAX_JSON_BODY_BYTES
        if cl is not None:
            try:
                if int(cl) > cap:
                    await _send_simple_json(send, 413, "request body too large")
                    return
            except ValueError:
                await _send_simple_json(send, 400, "invalid content-length")
                return

        # Buffer the body with a hard byte cap + receive deadline, then replay it.
        deadline = time.monotonic() + config.REQUEST_RECEIVE_DEADLINE_SECONDS
        chunks: list[bytes] = []
        total = 0
        more = True
        while more:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await _send_simple_json(send, 408, "request receive timeout")
                return
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except (asyncio.TimeoutError, TimeoutError):
                await _send_simple_json(send, 408, "request receive timeout")
                return
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            if body:
                total += len(body)
                if total > cap:
                    await _send_simple_json(send, 413, "request body too large")
                    return
                chunks.append(body)
            more = message.get("more_body", False)

        buffered = b"".join(chunks)
        served = False

        async def replay():
            nonlocal served
            if not served:
                served = True
                return {"type": "http.request", "body": buffered, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app.add_middleware(_BodyBoundMiddleware)


# ---------------------------------------------------------------- helpers


def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else None
    return auth.client_identity(peer, request.headers.get("x-forwarded-for"))


def _set_session_cookies(resp: Response, sess: auth.Session) -> None:
    resp.set_cookie(
        config.SESSION_COOKIE_NAME, sess.token,
        max_age=config.SESSION_TTL_SECONDS, httponly=True,
        secure=config.COOKIE_SECURE, samesite="strict", path="/",
    )
    # CSRF token is readable by JS (double-submit), so NOT HttpOnly.
    resp.set_cookie(
        config.CSRF_COOKIE_NAME, sess.csrf,
        max_age=config.SESSION_TTL_SECONDS, httponly=False,
        secure=config.COOKIE_SECURE, samesite="strict", path="/",
    )


def require_session(request: Request) -> auth.Session:
    token = request.cookies.get(config.SESSION_COOKIE_NAME)
    sess = auth.get_session(token)
    if sess is None:
        raise HTTPException(401, "authentication required")
    return sess


def require_mutation(request: Request) -> auth.Session:
    """Session + Origin + CSRF for any state-changing request."""
    if not auth.origin_allowed(request.headers.get("origin"), request.headers.get("host")):
        raise HTTPException(403, "origin not allowed")
    sess = require_session(request)
    if not auth.csrf_valid(sess, request.headers.get("x-csrf-token")):
        raise HTTPException(403, "invalid CSRF token")
    return sess


def _require_room_member(sess: auth.Session, room_id: str) -> rooms.RoomState:
    room = rooms.get_room(room_id)
    if room is None:
        raise HTTPException(404, "room not found")
    if sess.room_id != room_id:
        # A valid session bound to a different room (or none) cannot cross over.
        raise HTTPException(403, "not a member of this room")
    if room.retired or not auth.session_current(sess):
        raise HTTPException(401, "session no longer valid")
    return room


class DemoSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=1000)


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: dict


# ------------------------------------------------- canonical approval drafts
# Server-bound, room-scoped canonical task drafts for the plan -> (edit) ->
# validate -> activate contract (LIVE-VIDEO-005). Kept here under an app-level
# lock/dict (NOT in rooms.py). A draft binds the canonical task bytes to the
# room id, the uploaded source sha256 and the task version it was approved
# against, so a stale/replayed/tampered approval cannot activate.
_drafts: dict[str, dict] = {}
_drafts_lock = threading.Lock()


def _canonical_task_json(task: dict) -> str:
    return json.dumps(task, sort_keys=True, separators=(",", ":"))


def _draft_hash(task: dict, source_sha: str | None, room_id: str, version: int) -> str:
    payload = f"{_canonical_task_json(task)}|{source_sha or ''}|{room_id}|{int(version)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _store_draft(room_id: str, task: dict, source_sha: str | None, version: int) -> tuple[str, int]:
    digest = _draft_hash(task, source_sha, room_id, version)
    with _drafts_lock:
        _drafts[room_id] = {
            "hash": digest, "version": version,
            "source_sha": source_sha, "task": task,
        }
    return digest, version


def _pop_draft(room_id: str) -> None:
    with _drafts_lock:
        _drafts.pop(room_id, None)


def _drop_membership(token: str, room_id: str, owns: bool) -> None:
    rooms.evict_session(token, room_id, owns)
    if owns:
        _pop_draft(room_id)


def _reset_drafts_for_tests() -> None:
    with _drafts_lock:
        _drafts.clear()


# ---------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    unified = STATIC_DIR / "demo.html"
    if os.environ.get("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "").lower() == "true" and not unified.is_file():
        raise HTTPException(404, "unified frontend unavailable")
    return HTMLResponse((unified if unified.is_file() else STATIC_DIR / "landing.html").read_text())


@app.get("/upload", response_class=HTMLResponse)
def upload_page() -> HTMLResponse:
    if os.environ.get("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "").lower() == "true":
        return landing()
    return HTMLResponse((STATIC_DIR / "upload.html").read_text())


@app.get("/camera", response_class=HTMLResponse)
def camera_page() -> HTMLResponse:
    # OWNER PIVOT: the physical-camera path is unfinished and disabled on the
    # public surface. The page is only reachable when camera ingest is explicitly
    # enabled (local QA); publicly it is a plain 404.
    if not config.CAMERA_INGEST_ENABLED:
        raise HTTPException(404, "not found")
    return HTMLResponse((STATIC_DIR / "camera.html").read_text())


@app.get("/judges", response_class=HTMLResponse)
def judges_page() -> HTMLResponse:
    if os.environ.get("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "").lower() == "true":
        return landing()
    return HTMLResponse((STATIC_DIR / "judges.html").read_text())


@app.get("/api/live/health")
def health():
    return {
        "status": "ok",
        "service": "visionflow-live",
        "pin_configured": auth.pin_configured(),
        "planner_enabled": config.PLANNER_ENABLED,
        "teacher_available": teacher.available(),
        "cookie_secure": config.COOKIE_SECURE,
        "camera_ingest_enabled": config.CAMERA_INGEST_ENABLED,
        "primary_source": "video_upload",
        "worker": worker.liveness(),
    }


# ---------------------------------------------------------------- auth


@app.post("/api/live/login")
def login(request: Request, response: Response, body: dict = Body(...)):
    if not auth.origin_allowed(request.headers.get("origin"), request.headers.get("host")):
        raise HTTPException(403, "origin not allowed")
    ip = _client_ip(request)
    if auth.login_locked(ip):
        raise HTTPException(429, "too many attempts; try again later")
    role = str(body.get("role", "")).strip().lower()
    pin = str(body.get("pin", ""))
    if role not in {"camera", "judge"}:
        raise HTTPException(400, "role must be 'camera' or 'judge'")
    if not auth.verify_pin(pin):
        auth.record_failure(ip)
        raise HTTPException(401, "invalid PIN")
    auth.clear_failures(ip)
    try:
        sess = auth.create_session(role)
    except RuntimeError:
        raise HTTPException(503, "session capacity reached")
    _set_session_cookies(response, sess)
    return {"role": sess.role, "csrf": sess.csrf}


@app.post("/api/demo/session")
def demo_session(request: Request, response: Response, body: DemoSessionRequest):
    # Explicit deployment opt-in: this demo surface must sit behind its configured
    # access barrier. It never accepts a caller-supplied "authenticated" header.
    if os.environ.get("VISIONFLOW_DEMO_BOOTSTRAP_ENABLED", "").lower() != "true":
        raise HTTPException(404, "not found")
    if not auth.origin_allowed(request.headers.get("origin"), request.headers.get("host")):
        raise HTTPException(403, "origin not allowed")
    existing = auth.get_session(request.cookies.get(config.SESSION_COOKIE_NAME))
    if existing is not None and existing.role == "judge" and existing.owns_room:
        room = _require_room_member(existing, existing.room_id)
        _set_session_cookies(response, existing)
        return {"room_id": room.room_id, "csrf_token": existing.csrf,
                "task_version": room.task_version}
    ip = _client_ip(request)
    if auth.login_locked(ip):
        raise HTTPException(429, "too many demo sessions; try again later")
    auth.record_failure(ip)  # count successful allocation attempts as well
    try:
        sess = auth.create_session("judge")
    except RuntimeError:
        raise HTTPException(503, "session capacity reached")
    try:
        room = rooms.create_room(sess.token)
        auth.bind_room(sess.token, room.room_id, owns=True)
        with room.lock:
            room.judges.add(sess.token)
    except RuntimeError:
        auth.destroy_session(sess.token)
        raise HTTPException(503, "room capacity reached")
    _set_session_cookies(response, sess)
    return {"room_id": room.room_id, "csrf_token": sess.csrf, "task_version": 0}


@app.post("/api/live/logout")
def logout(request: Request, response: Response):
    # Logout is a state-changing mutation: guard it with the Origin allowlist and
    # (for a live session) the CSRF token, so it cannot be forced cross-site.
    if not auth.origin_allowed(request.headers.get("origin"), request.headers.get("host")):
        raise HTTPException(403, "origin not allowed")
    token = request.cookies.get(config.SESSION_COOKIE_NAME)
    sess = auth.get_session(token)
    if sess is not None and not auth.csrf_valid(sess, request.headers.get("x-csrf-token")):
        raise HTTPException(403, "invalid CSRF token")
    # destroy_session fires the membership teardown hook (room retirement / judge
    # eviction), which also closes the associated sockets' room (LIVE-002/006).
    auth.destroy_session(token)
    response.delete_cookie(config.SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(config.CSRF_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/live/me")
def me(sess: auth.Session = Depends(require_session)):
    return {"role": sess.role, "room_id": sess.room_id, "owns_room": sess.owns_room}


# ---------------------------------------------------------------- rooms


@app.post("/api/live/rooms")
def create_room(request: Request, sess: auth.Session = Depends(require_mutation)):
    if sess.role != "camera":
        raise HTTPException(403, "only a camera session can create a room")
    if sess.room_id:
        raise HTTPException(409, "session already bound to a room")
    try:
        room = rooms.create_room(sess.token)
    except RuntimeError:
        raise HTTPException(503, "room capacity reached")
    auth.bind_room(sess.token, room.room_id, owns=True)
    # The join code is returned ONCE to the authenticated owner and never logged
    # or placed in any URL. The owner shares it manually with a judge.
    return {"room_id": room.room_id, "join_code": room.join_code}


@app.post("/api/live/rooms/join")
def join_room(request: Request, sess: auth.Session = Depends(require_mutation),
              body: dict = Body(...)):
    if sess.role != "judge":
        raise HTTPException(403, "only a judge session can join a room")
    join_code = str(body.get("join_code", "")).strip()
    if not join_code:
        raise HTTPException(400, "join_code required")
    room = rooms.room_by_join_code(join_code)
    if room is None:
        # The shared PIN alone must not reveal rooms; an unknown code is a plain 404.
        raise HTTPException(404, "no room for that join code")
    with room.lock:
        if sess.token not in room.judges and len(room.judges) >= config.MAX_JUDGES_PER_ROOM:
            raise HTTPException(503, "room is full")
        room.judges.add(sess.token)
    prev = auth.bind_room(sess.token, room.room_id, owns=False)
    if prev and prev != room.room_id:
        # Rebinding to a new room evicts the judge from the old room so its open
        # socket stops receiving private bytes (LIVE-002).
        rooms.evict_session(sess.token, prev, owns=False)
    return {"room_id": room.room_id}


def _state_payload(room: rooms.RoomState) -> dict:
    with room.lock:
        return {
            "room_id": room.room_id,
            "task": room.task,
            "task_version": room.task_version,
            "task_enabled": room.task_enabled,
            "task_paused": room.task_paused,
            "history": list(room.history),
            "video": {
                "ready": room.video_path is not None,
                "sha256": room.video_sha256,
                "meta": dict(room.video_meta),
                "job_status": room.job_status,
                "job_processed": room.job_processed,
                "job_total": room.job_total,
                "job_error": room.job_error,
            },
            "metrics": {
                "frames_received": room.frames_received,
                "frames_dropped": room.frames_dropped,
                "frames_inferred": room.frames_inferred,
                "observed_fps": room.observed_fps(),
                "last_inference_ms": round(room.last_inference_ms, 1),
                "last_recv_age_s": round(time.time() - room.last_recv_ts, 2) if room.last_recv_ts else None,
                "has_frame": room.latest_jpeg is not None or room.annotated_jpeg is not None,
                "judges": len(room.judges),
            },
            "events": [
                {
                    "id": e.id, "type": e.type, "class_name": e.class_name,
                    "track_id": e.track_id, "detail": e.detail,
                    "confidence": round(e.confidence, 3),
                    "capture_ts": e.capture_ts, "recv_ts": e.recv_ts, "event_ts": e.event_ts,
                    "task_version": e.task_version,
                    "evidence_url": (
                        f"/api/live/rooms/{room.room_id}/evidence/{e.evidence_file}"
                        if e.evidence_file else None
                    ),
                    "visual_observation": e.visual_observation,
                    "action_status": e.action_status,
                    "provenance": e.provenance,
                }
                for e in list(room.events)[-80:]
            ],
            "notifications": list(room.notifications)[-80:],
            "planner_attempts_used": planner.room_attempts_used(room.room_id),
            "planner_budget": config.PLANNER_ROOM_BUDGET,
        }


@app.get("/api/live/rooms/{room_id}/state")
def room_state(room_id: str, sess: auth.Session = Depends(require_session)):
    room = _require_room_member(sess, room_id)
    return _state_payload(room)


@app.get("/api/live/rooms/{room_id}/frame")
def room_frame(room_id: str, sess: auth.Session = Depends(require_session)):
    room = _require_room_member(sess, room_id)
    with room.lock:
        jpeg = room.annotated_jpeg or room.latest_jpeg
        meta = dict(room.annotated_meta or room.latest_meta)
    if jpeg is None:
        raise HTTPException(404, "no frame yet")
    headers = {"Cache-Control": "no-store"}
    if meta.get("ts"):
        headers["X-Frame-Ts"] = str(meta["ts"])
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)


@app.get("/api/live/rooms/{room_id}/evidence/{filename}")
def room_evidence(room_id: str, filename: str, sess: auth.Session = Depends(require_session)):
    room = _require_room_member(sess, room_id)
    # Prevent path traversal and cross-room access: filename must belong to room.
    if "/" in filename or ".." in filename or not filename.startswith(f"room_{room_id}_"):
        raise HTTPException(404, "evidence not found")
    path = rooms.LIVE_EVIDENCE_DIR / filename
    with room.lock:
        if (room.retired or not auth.session_current(sess)
                or path not in room._evidence_files
                or not any(e.evidence_file == filename for e in room.events)):
            raise HTTPException(404, "evidence not found")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
                    raise HTTPException(404, "evidence not found")
                content = handle.read(2 * 1024 * 1024 + 1)
        except OSError:
            raise HTTPException(404, "evidence not found")
    return Response(content, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


# ---------------------------------------------------------------- video source (owner)


@app.get("/api/live/rooms/{room_id}/video/source")
def room_video_source(room_id: str, sha: str | None = None,
                      sess: auth.Session = Depends(require_session)):
    """Private playback of the room's current opaque uploaded source only."""
    room = _require_room_member(sess, room_id)
    with room.lock:
        path = room.video_path
        source_sha = room.video_sha256
        if room.retired or path is None or not path.is_file():
            raise HTTPException(404, "no video uploaded")
        resolved = path.resolve()
        if not resolved.is_relative_to(rooms.LIVE_VIDEO_DIR.resolve()) or not resolved.name.startswith(f"room_{room_id}_"):
            raise HTTPException(404, "source not found")
        if sha is not None and sha != source_sha:
            raise HTTPException(409, "source changed; refresh the current source")
    return FileResponse(resolved, media_type="video/mp4", filename="source.mp4",
                        content_disposition_type="inline",
                        headers={"Cache-Control":"private, no-store", "Pragma":"no-cache", "Vary":"Cookie"})


_uploads_lock = threading.Lock()
_upload_owners: set[str] = set()


def _claim_upload(owner: str) -> None:
    with _uploads_lock:
        if owner in _upload_owners:
            raise HTTPException(429, "one active upload per owner")
        if len(_upload_owners) >= max(1, config.MAX_CONCURRENT_VIDEO_JOBS):
            raise HTTPException(503, "upload capacity reached")
        _upload_owners.add(owner)


def _release_upload(owner: str) -> None:
    with _uploads_lock:
        _upload_owners.discard(owner)


@app.post("/api/live/rooms/{room_id}/video")
async def upload_video(room_id: str, request: Request):
    """Authenticated, bounded, decode-validated video upload (room owner only).

    The body is streamed to an opaque per-room path and the hard byte cap is
    enforced BYTE-BY-BYTE while streaming, so a lying or absent Content-Length can
    never bypass it. Invalid / oversize / undecodable input is rejected cleanly
    and the partial file removed; the worker/thread never dies."""
    # Mutation guards: Origin allowlist + session + CSRF, then role/ownership.
    if not auth.origin_allowed(request.headers.get("origin"), request.headers.get("host")):
        raise HTTPException(403, "origin not allowed")
    sess = require_session(request)
    if not auth.csrf_valid(sess, request.headers.get("x-csrf-token")):
        raise HTTPException(403, "invalid CSRF token")
    if sess.role not in {"camera", "judge"} or not sess.owns_room:
        raise HTTPException(403, "only the room owner can upload a video")
    room = _require_room_member(sess, room_id)
    if not sess.owns_room:
        raise HTTPException(403, "only the room owner can upload a video")

    _claim_upload(sess.token)
    tmp = rooms.LIVE_VIDEO_DIR / f"room_{room_id}_{secrets.token_urlsafe(10)}.mp4"
    handed_to_worker = False
    cancelled = threading.Event()

    def authorized() -> bool:
        # Called INSIDE replace_source's job/registered-room/room lock scope.
        # Do not reacquire the non-reentrant rooms registry lock here.
        return (not cancelled.is_set() and auth.session_current(sess)
                and sess.role in {"camera", "judge"} and sess.owns_room
                and sess.room_id == room_id and room.owner_token == sess.token)

    def finish_upload() -> tuple[dict, str]:
        attached = False
        try:
            # Root's VIDEO probe is a bounded, full-count decoder subprocess.
            # This worker owns its upload permit until REAL completion, even
            # if the awaiting HTTP task is cancelled/disconnected.
            meta = video.probe(tmp)
            sha = video.sha256_file(tmp)
            if not authorized():
                raise HTTPException(401, "upload owner session no longer valid")
            video.replace_source(room, tmp, sha, meta, authorize=authorized)
            attached = True
            _pop_draft(room_id)
            return meta, sha
        finally:
            try:
                if not attached:
                    tmp.unlink(missing_ok=True)
            finally:
                _release_upload(sess.token)

    try:
        rooms.LIVE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        total = 0
        deadline = time.monotonic() + config.REQUEST_RECEIVE_DEADLINE_SECONDS
        stream = request.stream().__aiter__()
        with tmp.open("wb") as fh:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HTTPException(408, "video receive timeout")
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    raise HTTPException(408, "video receive timeout")
                total += len(chunk)
                if total > config.MAX_VIDEO_BYTES:
                    raise HTTPException(413, "video exceeds size limit")
                fh.write(chunk)
        if not total:
            raise HTTPException(400, "empty upload")
        future = asyncio.get_running_loop().run_in_executor(None, finish_upload)
        handed_to_worker = True
        # Retrieve late worker exceptions without logging private paths/data.
        future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        meta, sha = await asyncio.shield(future)
        return {"ok": True, "sha256": sha, "meta": meta}
    except asyncio.CancelledError:
        cancelled.set()
        raise
    except HTTPException:
        raise
    except video.VideoError as exc:
        if not authorized():
            raise HTTPException(401, "upload owner session no longer valid")
        raise HTTPException(422, f"invalid video: {exc}")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "could not process upload")
    finally:
        if not handed_to_worker:
            try:
                tmp.unlink(missing_ok=True)
            finally:
                _release_upload(sess.token)


_DEMO_VIDEO_FILES = {f"Camera{i}": f"Camera{i}-960x540-cfr24.mp4" for i in range(1, 6)}
_DEMO_VIDEO_LABEL = "owner-provided recorded video; visual interpretation requires validated inference"


def _demo_video_path(input_id: str) -> Path | None:
    directory = os.environ.get("VISIONFLOW_DEMO_VIDEO_DIR", "").strip()
    if not directory or input_id not in _DEMO_VIDEO_FILES:
        return None
    root = Path(directory).resolve()
    path = root / _DEMO_VIDEO_FILES[input_id]
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        return None
    return path


@app.get("/api/live/video-catalog")
def video_catalog(sess: auth.Session = Depends(require_session)):
    if sess.role not in {"camera", "judge"} or not sess.owns_room or not sess.room_id:
        raise HTTPException(403, "only a room owner can select prepared videos")
    _require_room_member(sess, sess.room_id)
    inputs = [{"id": key, "label": _DEMO_VIDEO_LABEL}
              for key in _DEMO_VIDEO_FILES if _demo_video_path(key) is not None]
    return JSONResponse({"available": bool(inputs), "inputs": inputs},
                        headers={"Cache-Control": "private, no-store", "Vary": "Cookie"})


@app.post("/api/live/rooms/{room_id}/video/select")
async def select_video(room_id: str, request: Request, body: dict = Body(...),
                       sess: auth.Session = Depends(require_mutation)):
    if sess.role not in {"camera", "judge"} or not sess.owns_room:
        raise HTTPException(403, "only the room owner can select a video")
    _require_room_member(sess, room_id)
    if set(body) != {"id"} or not isinstance(body.get("id"), str) or body["id"] not in _DEMO_VIDEO_FILES:
        raise HTTPException(422, "select one fixed Camera1–Camera5 ID; other fields forbidden")
    path = _demo_video_path(body["id"])
    if path is None:
        raise HTTPException(404, "prepared video unavailable")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise HTTPException(404, "prepared video unavailable")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= config.MAX_VIDEO_BYTES:
            raise HTTPException(422, "prepared video is not a supported bounded file")

        class PreparedVideoRequest:
            headers = request.headers
            cookies = request.cookies

            async def stream(self):
                # Fixed private file -> same opaque upload path. No precomputed
                # labels/events: reuse actual decoder, admission and authority checks.
                while chunk := os.read(fd, 64 * 1024):
                    yield chunk
                    await asyncio.sleep(0)

        result = await upload_video(room_id, PreparedVideoRequest())
        return {**result, "id": body["id"], "label": _DEMO_VIDEO_LABEL}
    finally:
        os.close(fd)


@app.post("/api/live/rooms/{room_id}/video/process")
def process_video(room_id: str, sess: auth.Session = Depends(require_mutation)):
    """Start (or restart from frame 0) processing the uploaded video with the
    room's currently approved task. Requires an approved/active task."""
    room = _require_room_member(sess, room_id)
    with room.lock:
        if room.video_path is None:
            raise HTTPException(409, "no video uploaded")
        if room.task is None or not room.task_enabled:
            raise HTTPException(409, "approve and enable a task first")
    try:
        video.start_job(room)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return {"ok": True, "job_status": "running"}


# Scene observations are deliberately separate from executable workflows.
@app.post("/api/live/rooms/{room_id}/scene")
def inspect_video_scene(room_id: str, sess: auth.Session = Depends(require_mutation),
                        body: dict = Body(...)):
    if sess.role != "judge":
        raise HTTPException(403, "only a judge can inspect a scene")
    room = _require_room_member(sess, room_id)
    if (set(body) != {"question", "source_sha256", "expected_version"}
            or type(body["question"]) is not str or not 1 <= len(body["question"].strip()) <= 1000
            or type(body["source_sha256"]) is not str or len(body["source_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in body["source_sha256"])
            or type(body["expected_version"]) is not int or body["expected_version"] < 0):
        raise HTTPException(422, "invalid scene request")

    def session_is_current():
        current = auth.peek_session(sess.token)
        return current is sess and current.role == "judge" and current.room_id == room_id

    return scene.inspect_scene(room, body["question"], body["source_sha256"],
                               body["expected_version"], session_is_current=session_is_current)


# ---------------------------------------------------------------- task control (judges)


@app.post("/api/live/rooms/{room_id}/task/plan")
def plan_task(room_id: str, sess: auth.Session = Depends(require_mutation),
              body: PlanRequest = Body(...)):
    if sess.role != "judge":
        raise HTTPException(403, "only a judge can plan a task")
    room = _require_room_member(sess, room_id)
    text = body.text
    with room.lock:
        if room.retired or not auth.session_current(sess):
            raise HTTPException(401, "session no longer valid")
        room.draft_generation += 1
        generation = room.draft_generation
        src, path, ver = room.video_sha256, room.video_path, room.task_version
        _pop_draft(room_id)
    result = planner.plan(text, room.room_id)
    # A successful plan issues a server-bound canonical draft so the judge can
    # approve the EXACT task that was previewed (LIVE-VIDEO-005).
    draft_hash = None
    expected_version = None
    with room.lock:
        if not auth.session_current(sess) or sess.room_id != room_id or room.retired:
            raise HTTPException(401, "planner session no longer valid")
        if (room.video_sha256 != src or room.video_path != path
                or room.task_version != ver or room.draft_generation != generation):
            raise HTTPException(409, "source, task or draft changed during planning; plan again")
        if result.status == "ok" and result.task:
            draft_hash, expected_version = _store_draft(room.room_id, result.task, src, ver)
    return {
        "status": result.status,
        "original_text": text,
        "task": result.task,
        "draft_hash": draft_hash,
        "expected_version": expected_version,
        "proposal": result.proposal,
        "clarification": result.clarification,
        "unsupported_reason": result.unsupported_reason,
        "error": result.error,
        "provenance": {
            "model": result.model, "cost_usd": result.cost_usd,
            "duration_ms": result.duration_ms, "usage": result.usage,
            "attempt": result.attempt,
        },
    }


@app.post("/api/live/rooms/{room_id}/task/validate")
def validate_task(room_id: str, sess: auth.Session = Depends(require_mutation),
                  body: ValidateRequest = Body(...)):
    """Validate an edited task against the backend-owned DSL and issue a server-bound
    canonical draft (hash + version) the judge can then approve verbatim. This
    supports manual JSON edits of a planned task (LIVE-VIDEO-005)."""
    if sess.role != "judge":
        raise HTTPException(403, "only a judge can validate a task")
    room = _require_room_member(sess, room_id)
    raw_task = body.task
    if not isinstance(raw_task, dict):
        raise HTTPException(400, "task object required")
    try:
        task = LiveTask.model_validate(raw_task)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"task failed validation: {exc}")
    canonical = task.model_dump()
    with room.lock:
        if room.retired or not auth.session_current(sess) or sess.room_id != room_id:
            raise HTTPException(401, "session no longer valid")
        room.draft_generation += 1
        src = room.video_sha256
        ver = room.task_version
        digest, expected = _store_draft(room.room_id, canonical, src, ver)
    return {"task": canonical, "draft_hash": digest, "expected_version": expected}


@app.post("/api/live/rooms/{room_id}/task/activate")
def activate_task(room_id: str, sess: auth.Session = Depends(require_mutation),
                  body: dict = Body(...)):
    if sess.role != "judge":
        raise HTTPException(403, "only a judge can activate a task")
    room = _require_room_member(sess, room_id)
    raw_task = body.get("task")
    if not isinstance(raw_task, dict):
        raise HTTPException(400, "task object required")
    # Re-validate against the backend-owned DSL: the UI is never trusted.
    try:
        task = LiveTask.model_validate(raw_task)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"task failed validation: {exc}")
    canonical = task.model_dump()

    # Every activation requires the server-issued canonical approval identity.
    # No direct task-only path may bypass human approval of a validated draft.
    if set(body) != {"task", "draft_hash", "expected_version"}:
        raise HTTPException(422, "task, draft_hash and expected_version are required; unknown fields forbidden")
    draft_hash = body["draft_hash"]
    expected_version = body["expected_version"]
    if not isinstance(draft_hash, str) or not draft_hash:
        raise HTTPException(422, "draft_hash must be a nonempty string")
    if not isinstance(expected_version, int) or isinstance(expected_version, bool):
        raise HTTPException(422, "expected_version must be an integer")
    # Validation, one-shot draft consumption and version transition are atomic.
    with room.lock:
        if room.retired:
            raise HTTPException(409, "room retired")
        if not auth.session_current(sess) or sess.room_id != room_id:
            raise HTTPException(401, "approval session no longer valid")
        src = room.video_sha256
        if room.task_version != expected_version:
            raise HTTPException(409, "stale approval: task version changed; re-validate")
        with _drafts_lock:
            stored = _drafts.get(room_id)
            if stored is None:
                raise HTTPException(409, "no validated draft for this room; validate first")
            if stored["source_sha"] != src or stored["version"] != expected_version:
                raise HTTPException(409, "source or task version changed; re-validate")
            want = _draft_hash(canonical, src, room_id, expected_version)
            if stored["task"] != canonical or not hmac.compare_digest(draft_hash, want) or not hmac.compare_digest(stored["hash"], want):
                raise HTTPException(422, "approval does not match the validated draft")
            fw = room.latest_meta.get("w") or 1
            fh = room.latest_meta.get("h") or 1
            version = rooms.activate_task(room, canonical, fw, fh)
            room.approved_hash = want
            has_video = room.video_path is not None
            _drafts.pop(room_id, None)
    # A new approved request on the SAME video reruns the analysis from frame 0
    # with fresh detector/tracker/engine state. Any in-flight job is superseded;
    # its stale events are already dropped by the version guard.
    video.cancel_job(room_id)
    rerun = False
    if has_video:
        try:
            video.start_job(room)
            rerun = True
        except RuntimeError:
            with room.lock:
                if room.task_version == version:
                    room.task_enabled = False
                    room.job_status = "busy"
                    room.job_error = "Inference capacity busy; enable after the previous worker exits."
            raise HTTPException(503, "workflow approved but execution not admitted; task disabled, retry enable when idle")
    return {"ok": True, "task_version": version, "task": canonical, "rerun": rerun}


@app.post("/api/live/rooms/{room_id}/task/state")
def task_state(room_id: str, sess: auth.Session = Depends(require_mutation),
               body: dict = Body(...)):
    if sess.role != "judge":
        raise HTTPException(403, "only a judge can control the task")
    room = _require_room_member(sess, room_id)
    try:
        req = TaskStateRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"invalid task state request: {exc}")
    with room.lock:
        rooms.set_task_state(room, enabled=req.enabled, paused=req.paused)
        active = room.task_enabled and not room.task_paused
        has_video = room.video_path is not None
        version = room.task_version
        enabled = room.task_enabled
        paused = room.task_paused
    # Any state transition bumped the version, so a running job's events are now
    # dropped by the guard. Cancel it so stale work stops immediately; a re-enable
    # /resume on a video rebuilds engines and reruns from frame 0 with reset timing.
    video.cancel_job(room_id)
    if has_video and active:
        try:
            video.start_job(room)
        except RuntimeError:
            with room.lock:
                if room.task_version == version:
                    room.task_enabled = False
                    room.job_status = "busy"
                    room.job_error = "Inference capacity busy; enable after the previous worker exits."
            raise HTTPException(503, "execution not admitted; task remains disabled, retry when idle")
    return {"ok": True, "task_version": version,
            "task_enabled": enabled, "task_paused": paused}


@app.post("/api/live/rooms/{room_id}/probe")
def probe_teacher(room_id: str, sess: auth.Session = Depends(require_mutation),
                  body: dict = Body(...)):
    # The Teacher observation path belongs to the disabled camera/semantic flow and
    # is not verifiable on the enabled VIDEO surface. Keep it inaccessible
    # (fail-closed 404) unless camera ingest is explicitly enabled (LIVE-VIDEO-001).
    if not config.CAMERA_INGEST_ENABLED:
        raise HTTPException(404, "not found")
    if sess.role != "judge":
        raise HTTPException(403, "only a judge can probe")
    room = _require_room_member(sess, room_id)
    question = str(body.get("question", "")).strip()
    labels = body.get("labels") or ["yes", "no"]
    if len(question) < 3:
        raise HTTPException(400, "question too short")
    with room.lock:
        jpeg = room.latest_jpeg
    if jpeg is None:
        raise HTTPException(409, "no frame to probe")
    obs = teacher.observe(room.room_id, question, [str(x) for x in labels][:6], [jpeg])
    return {
        "status": obs.status, "label": obs.label, "confidence": obs.confidence,
        "uncertain": obs.uncertain, "image_hash": obs.image_hash, "model": obs.model,
        "latency_ms": obs.latency_ms, "raw": obs.raw, "reason": obs.reason,
    }


# ---------------------------------------------------------------- websockets


def _ws_token(ws: WebSocket) -> str | None:
    return ws.cookies.get(config.SESSION_COOKIE_NAME)


def _ws_origin_ok(ws: WebSocket) -> bool:
    return auth.origin_allowed(ws.headers.get("origin"), ws.headers.get("host"))


def _ws_csrf_ok(ws: WebSocket, sess: auth.Session) -> bool:
    """WebSocket upgrades cannot carry custom headers, so the double-submit CSRF
    token is supplied as a query parameter and must match the session's CSRF
    cookie (LIVE-003)."""
    return auth.csrf_valid(sess, ws.query_params.get("csrf"))


@app.websocket("/api/live/ws/camera")
async def ws_camera(ws: WebSocket):
    # OWNER PIVOT: the physical-camera WebSocket ingest path is unfinished and
    # DISABLED on the public surface. Deny the upgrade (fail-closed) unless camera
    # ingest is explicitly enabled for local QA.
    if not config.CAMERA_INGEST_ENABLED:
        await ws.close(code=4404)
        return
    if not _ws_origin_ok(ws):
        await ws.close(code=4403)
        return
    token = _ws_token(ws)
    sess = auth.get_session(token)
    if sess is None or sess.role != "camera" or not sess.owns_room or not sess.room_id:
        await ws.close(code=4401)
        return
    if not _ws_csrf_ok(ws, sess):
        await ws.close(code=4403)
        return
    room_id = sess.room_id
    room = rooms.get_room(room_id)
    if room is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    pending_meta: dict = {}
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            # Re-validate the session/role/room membership on EVERY message so a
            # logged-out, expired or rebound camera cannot keep ingesting (LIVE-002).
            cur = auth.get_session(token)
            if cur is None or cur.role != "camera" or not cur.owns_room or cur.room_id != room_id:
                await ws.close(code=4401)
                return
            room = rooms.get_room(room_id)
            if room is None:
                await ws.close(code=4404)
                return
            if msg.get("text") is not None:
                text = msg["text"]
                if len(text) > config.MAX_WS_TEXT_BYTES:
                    pending_meta = {}
                    continue
                try:
                    pending_meta = json.loads(text)
                    if not isinstance(pending_meta, dict):
                        pending_meta = {}
                except (ValueError, TypeError):
                    pending_meta = {}
                continue
            data = msg.get("bytes")
            if data is None:
                continue
            if len(data) > config.MAX_FRAME_BYTES or len(data) < 100 or data[:2] != b"\xff\xd8":
                # Too big, empty, or not a JPEG (SOI marker). Dropped before ingest.
                with room.lock:
                    room.frames_dropped += 1
                pending_meta = {}
                continue
            _ingest_frame(room, data, pending_meta)
            pending_meta = {}
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("camera ws error room=%s: %s", room_id, exc)


def _finite_number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _ingest_frame(room: rooms.RoomState, data: bytes, meta: dict) -> None:
    now = time.time()
    w = meta.get("w"); h = meta.get("h")
    # Enforce the long-edge bound using declared dimensions when present.
    if isinstance(w, int) and isinstance(h, int) and max(w, h) > config.MAX_FRAME_LONG_EDGE:
        with room.lock:
            room.frames_dropped += 1
        return
    # Sequence must strictly increase; duplicate/stale frames are dropped. A
    # non-finite capture_ts is discarded (stored as None) so it can never poison
    # the JSON state payload (LIVE-005).
    seq = meta.get("seq")
    seq = seq if isinstance(seq, int) and not isinstance(seq, bool) else None
    capture_ts = _finite_number(meta.get("capture_ts"))
    with room.lock:
        if seq is not None and seq <= room.last_seq:
            room.frames_dropped += 1
            return
        if seq is not None:
            room.last_seq = seq
        room.frames_received += 1
        room.last_recv_ts = now
        room.recv_times.append(now)
        # Keep ONLY the latest frame for inference (drop stale); bounded temporal
        # deque retains a few recent raw frames for before/after probes.
        room.latest_jpeg = data
        room.latest_meta = {"w": w, "h": h, "seq": seq,
                            "capture_ts": capture_ts, "recv_ts": now}
        room.temporal.append({"jpeg": data, "ts": now, "seq": seq})
        room.frame_dirty = True


@app.websocket("/api/live/ws/judge")
async def ws_judge(ws: WebSocket):
    # OWNER PIVOT: the WebSocket push channel belongs to the disabled two-device
    # camera path. Judges watch via authenticated HTTP polling of the annotated
    # frame + state endpoints instead. Deny the upgrade unless explicitly enabled.
    if not config.CAMERA_INGEST_ENABLED:
        await ws.close(code=4404)
        return
    if not _ws_origin_ok(ws):
        await ws.close(code=4403)
        return
    token = _ws_token(ws)
    sess = auth.get_session(token)
    if sess is None or sess.role != "judge" or not sess.room_id:
        await ws.close(code=4401)
        return
    if not _ws_csrf_ok(ws, sess):
        await ws.close(code=4403)
        return
    room_id = sess.room_id
    room = rooms.get_room(room_id)
    if room is None or token not in room.judges:
        await ws.close(code=4404)
        return
    await ws.accept()
    last_sent_ts = None
    interval = 1.0 / max(1.0, config.TARGET_PROCESS_FPS)
    try:
        while True:
            # Re-validate authority before EACH private send: a logged-out, expired
            # or rebound judge must stop receiving private bytes (LIVE-002).
            cur = auth.get_session(token)
            if cur is None or cur.role != "judge" or cur.room_id != room_id:
                await ws.close(code=4401)
                return
            room = rooms.get_room(room_id)
            if room is None or token not in room.judges:
                await ws.close(code=4404)
                return
            with room.lock:
                jpeg = room.annotated_jpeg
                ts = room.annotated_meta.get("ts")
            if jpeg is not None and ts != last_sent_ts:
                await ws.send_bytes(jpeg)
                last_sent_ts = ts
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.info("judge ws closed room=%s: %s", room_id, exc)
