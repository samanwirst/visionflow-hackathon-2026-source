"""In-memory live room registry: pairing, frame buffers, task state, evidence.

A room is created by a camera session and paired with judges through an opaque
join code (never placed in URLs or logs). Each room keeps only VOLATILE media —
the latest frame plus a short bounded temporal deque — and persists ONLY event
evidence JPEGs, bounded by count. There is no automatic raw recording.

A room holds at most one ACTIVE task. Switching/pausing/disabling a task never
touches the camera stream. A monotonically increasing ``task_version`` lets the
worker drop any inference/action that belongs to a task that was already
replaced or disabled (version guard).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from app import config as app_config
from app.live import config
from visionflow_vision.events import EventType, ZoneEventEngine
from visionflow_vision.zones import Zone

logger = logging.getLogger("visionflow.live.rooms")

LIVE_EVIDENCE_DIR = app_config.DATA_DIR / "live_evidence"
LIVE_VIDEO_DIR = (
    Path(config.VIDEO_UPLOAD_DIR) if config.VIDEO_UPLOAD_DIR
    else app_config.DATA_DIR / "live_uploads"
)


@dataclass
class LiveEvent:
    id: int
    type: str
    class_name: str | None
    track_id: int | None
    detail: str
    confidence: float
    capture_ts: float          # wall-clock seconds (camera capture) when known
    recv_ts: float             # server receive wall-clock
    event_ts: float            # server event wall-clock
    task_version: int
    evidence_file: str | None = None
    visual_observation: dict | None = None
    action_status: str = "pending"
    provenance: dict | None = None   # source sha / frame index / video ts / model id


@dataclass
class RoomState:
    room_id: str
    join_code: str
    owner_token: str
    created_at: float = field(default_factory=time.time)
    judges: set[str] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # volatile media
    latest_jpeg: bytes | None = None
    latest_meta: dict = field(default_factory=dict)      # w,h,seq,capture_ts,recv_ts
    annotated_jpeg: bytes | None = None
    annotated_meta: dict = field(default_factory=dict)
    temporal: deque = field(default_factory=lambda: deque(maxlen=config.TEMPORAL_DEQUE_LEN))
    frame_dirty: bool = False                            # worker has a new frame to process
    last_seq: int = -1                                   # highest accepted sequence number

    # video source (OWNER PIVOT: pre-recorded video inference, not camera stream)
    video_path: Path | None = None          # opaque stored upload path (owner-only)
    video_sha256: str | None = None         # sha256 of the uploaded source bytes
    video_meta: dict = field(default_factory=dict)   # fps, frame_count, w, h, duration
    job_status: str = "idle"                # idle|running|done|cancelled|error
    job_processed: int = 0                  # frames processed so far
    job_total: int = 0                      # total frames to process
    job_error: str | None = None
    job_version: int = 0                    # task_version the running job was started for
    # The execution generation currently AUTHORIZED to publish frames/events/
    # evidence for this room. ``video.start_job`` stamps each run with a unique
    # monotonic id; any source/task/session transition resets it to 0, which
    # atomically revokes a still-running (possibly blocked) worker so it can never
    # publish or persist private media after an acknowledged control (LIVE-VIDEO-002).
    active_generation: int = 0
    # Set once when the room is retired (owner logout / expiry / idle). A retired
    # room never recreates private media; its outputs are purged under the lock.
    retired: bool = False

    # task state
    task: dict | None = None                             # validated LiveTask.model_dump()
    task_version: int = 0
    draft_generation: int = 0  # newer plan/edit requests revoke older in-flight plans
    approved_hash: str | None = None
    task_enabled: bool = False
    task_paused: bool = False
    engines: list = field(default_factory=list)          # [(EventType engine)] for current version
    history: deque = field(default_factory=lambda: deque(maxlen=40))

    # outputs (bounded)
    events: deque = field(default_factory=lambda: deque(maxlen=200))
    notifications: deque = field(default_factory=lambda: deque(maxlen=200))
    _event_seq: int = 0
    _evidence_files: deque = field(default_factory=lambda: deque())

    # metrics
    frames_received: int = 0
    frames_dropped: int = 0
    frames_inferred: int = 0
    last_recv_ts: float = 0.0
    last_inference_ms: float = 0.0
    recv_times: deque = field(default_factory=lambda: deque(maxlen=20))

    def next_event_id(self) -> int:
        self._event_seq += 1
        return self._event_seq

    def observed_fps(self) -> float | None:
        if len(self.recv_times) < 2:
            return None
        span = self.recv_times[-1] - self.recv_times[0]
        if span <= 0:
            return None
        return (len(self.recv_times) - 1) / span

    def record_evidence(self, path: Path) -> None:
        self._evidence_files.append(path)
        while len(self._evidence_files) > config.EVIDENCE_MAX_PER_ROOM:
            old = self._evidence_files.popleft()
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass


_rooms: dict[str, RoomState] = {}
_by_join_code: dict[str, str] = {}   # join_code -> room_id
_rooms_lock = threading.Lock()


def create_room(owner_token: str) -> RoomState:
    with _rooms_lock:
        if len(_rooms) >= config.MAX_ROOMS:
            raise RuntimeError("room capacity reached")
        room_id = secrets.token_urlsafe(12)
        join_code = secrets.token_urlsafe(9)
        room = RoomState(room_id=room_id, join_code=join_code, owner_token=owner_token)
        _rooms[room_id] = room
        _by_join_code[join_code] = room_id
        logger.info("room created id=%s (join code not logged)", room_id)
        return room


def get_room(room_id: str | None) -> RoomState | None:
    if not room_id:
        return None
    with _rooms_lock:
        return _rooms.get(room_id)


def room_by_join_code(join_code: str) -> RoomState | None:
    if not join_code:
        return None
    with _rooms_lock:
        rid = _by_join_code.get(join_code)
        return _rooms.get(rid) if rid else None


def iter_rooms() -> list[RoomState]:
    with _rooms_lock:
        return list(_rooms.values())


def close_room(room_id: str) -> None:
    # 1) Mark the room retired and revoke its execution generation FIRST, under the
    #    room lock, so an in-flight (even blocked) video worker immediately loses
    #    publish authority before any teardown and cannot recreate private media.
    existing = get_room(room_id)
    if existing is not None:
        with existing.lock:
            existing.retired = True
            existing.active_generation = 0
    # 2) Signal any running video job to stop and revoke its generation again.
    try:
        from app.live import video as _video
        _video.cancel_job(room_id)
    except Exception:  # noqa: BLE001
        pass
    with _rooms_lock:
        room = _rooms.pop(room_id, None)
        if room is not None:
            _by_join_code.pop(room.join_code, None)
    if room is not None:
        with room.lock:
            # Fail-closed purge: a retired room keeps no private outputs, so an
            # already-committed event/notification/annotated frame cannot survive
            # the owner's logout/expiry (LIVE-VIDEO-002).
            room.retired = True
            room.active_generation = 0
            room.events.clear()
            room.notifications.clear()
            room.annotated_jpeg = None
            room.latest_jpeg = None
            evidence = list(room._evidence_files)
            room._evidence_files.clear()
        for p in evidence:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        # Delete the opaque uploaded source so no private media outlives the room.
        if room.video_path is not None:
            try:
                room.video_path.unlink(missing_ok=True)
            except OSError:
                pass


def evict_session(token: str, room_id: str, owns: bool) -> None:
    """Tear down a session's room membership (logout / expiry / rebind). The
    room owner leaving retires the whole room; a judge leaving is just removed."""
    if owns:
        close_room(room_id)
        return
    room = get_room(room_id)
    if room is None:
        return
    with room.lock:
        room.judges.discard(token)


def retire_idle_and_orphaned(active: set[str], idle_seconds: float) -> list[str]:
    """Retire rooms whose owner session is gone, or that have had no frames and
    no judges past the idle TTL. Also prune dead judge memberships. Returns the
    retired room ids. Called periodically by the worker (LIVE-006)."""
    now = time.time()
    retire: list[str] = []
    for room in iter_rooms():
        with room.lock:
            room.judges = {t for t in room.judges if t in active}
            owner_gone = room.owner_token not in active
            last = room.last_recv_ts or room.created_at
            idle = (now - last) >= idle_seconds
            orphaned = owner_gone and not room.judges
            stale_idle = idle and not room.judges
        if owner_gone or orphaned or stale_idle:
            retire.append(room.room_id)
    for rid in retire:
        close_room(rid)
    return retire


def _reset_for_tests() -> None:
    with _rooms_lock:
        _rooms.clear()
        _by_join_code.clear()


# --- task engine construction ---------------------------------------------


def build_engines(task: dict, frame_w: int, frame_h: int, version: int) -> list[ZoneEventEngine]:
    """Build the temporal event engine(s) for a validated task. One task => one
    engine today, but kept as a list for parity with the M1 pipeline."""
    zone_poly = tuple((float(x), float(y)) for x, y in task["zone"]["polygon"])
    zone = Zone(id=version, name=task["name"][:60] or "zone", polygon=zone_poly)
    from app.live.dsl import TRIGGER_TO_EVENT

    event_type = EventType(TRIGGER_TO_EVENT[task["trigger"]])
    # Cooldown comes from the approved task when supplied (exact value), else the
    # documented default of 10s. The DSL layer owns whether a supplied cooldown is
    # accepted or rejected; here we faithfully apply whatever survived validation.
    cooldown = task.get("cooldown_seconds")
    cooldown = float(cooldown) if cooldown is not None else 10.0
    engine = ZoneEventEngine(
        zone=zone,
        object_class=task["object_class"],
        event_type=event_type,
        frame_w=frame_w or 1,
        frame_h=frame_h or 1,
        dwell_seconds=task.get("duration_seconds") or 0.0,
        count_threshold=task.get("threshold") or 0,
        cooldown_seconds=cooldown,
        rule_key=f"room-v{version}",
    )
    return [engine]


def activate_task(room: RoomState, task: dict, frame_w: int, frame_h: int) -> int:
    """Install a new active task (replacing any previous one), bump the version,
    and build fresh engines. Returns the new version. Caller holds room.lock."""
    room.task = task
    room.task_version += 1
    room.task_enabled = True
    room.task_paused = False
    try:
        room.engines = build_engines(task, frame_w, frame_h, room.task_version)
    except Exception as exc:  # noqa: BLE001
        room.engines = []
        logger.warning("failed to build engines for room %s: %s", room.room_id, exc)
    room.history.append({
        "ts": time.time(),
        "action": "activated",
        "version": room.task_version,
        "name": task.get("name"),
    })
    return room.task_version


def _rebuild_engines_locked(room: RoomState) -> None:
    """Bump the version (invalidating any in-flight inference) and rebuild fresh
    engines when the task is ACTIVE (enabled and not paused), else clear them.
    Caller holds room.lock. Every enable/disable/pause/resume transition flows
    through here so no stale generation can publish an event after the control
    is acknowledged."""
    room.task_version += 1
    if room.task is not None and room.task_enabled and not room.task_paused:
        fw = room.latest_meta.get("w") or 1
        fh = room.latest_meta.get("h") or 1
        try:
            room.engines = build_engines(room.task, fw, fh, room.task_version)
        except Exception as exc:  # noqa: BLE001
            room.engines = []
            logger.warning("failed to rebuild engines for room %s: %s", room.room_id, exc)
    else:
        room.engines = []


def set_task_state(room: RoomState, *, enabled: bool | None = None, paused: bool | None = None) -> None:
    """Pause/disable/re-enable the active task WITHOUT touching the camera.

    Any transition bumps the version so in-flight inference/actions for the old
    generation are dropped; re-enabling or resuming rebuilds fresh engines with
    explicit timing (new monotonic baseline), so an old inference held across the
    control cannot notify after the acknowledgment (LIVE-007)."""
    if room.task is None:
        return
    changed = False
    if enabled is not None and enabled != room.task_enabled:
        room.task_enabled = enabled
        room.history.append({"ts": time.time(), "action": "enabled" if enabled else "disabled",
                             "version": room.task_version + 1})
        changed = True
    if paused is not None and paused != room.task_paused:
        room.task_paused = paused
        room.history.append({"ts": time.time(), "action": "paused" if paused else "resumed",
                             "version": room.task_version + 1})
        changed = True
    if changed:
        _rebuild_engines_locked(room)
