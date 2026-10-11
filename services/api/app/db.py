"""SQLAlchemy models and session management (SQLite, WAL mode)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app import config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class VideoSource(Base):
    __tablename__ = "video_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String(16), default="upload")
    name: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    # uploaded | processing | processed | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    avg_inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("video_sources.id"))
    name: Mapped[str] = mapped_column(String(255))
    polygon: Mapped[list] = mapped_column(JSON)  # [[x, y], ...] normalized 0..1
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Workflow(Base):
    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("video_sources.id"))
    name: Mapped[str] = mapped_column(String(255))
    dsl: Mapped[dict] = mapped_column(JSON)  # validated WorkflowDSL dump
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrackEvent(Base):
    __tablename__ = "track_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("video_sources.id"))
    workflow_id: Mapped[int] = mapped_column(ForeignKey("workflows.id"))
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type: Mapped[str] = mapped_column(String(64))
    class_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    video_timestamp: Mapped[float] = mapped_column(Float)  # seconds into video
    confidence: Mapped[float] = mapped_column(Float)
    detail: Mapped[str] = mapped_column(Text, default="")
    dedupe_key: Mapped[str] = mapped_column(String(255), unique=True)
    evidence_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_status: Mapped[str] = mapped_column(String(32), default="pending")
    # pending | done | failed | cancelled | skipped_visual | interrupted_unverified
    # Visual routing provenance (M2). NULL for geometric-only events.
    visual_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # ok | unknown | low_confidence | error | timeout | invalid_response
    # | missing_credentials | cancelled | interrupted
    # | suppressed_budget | suppressed_cooldown | suppressed_sampling
    visual_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    visual_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_routing: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ActionExecution(Base):
    __tablename__ = "action_executions"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("track_events.id"))
    action_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # pending | success | failed | rejected | cancelled
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VlmCall(Base):
    """One VLM router decision for a candidate event (M2-AC5). Persisted for
    auditable routing provenance and counters; every decision is recorded,
    including cache hits, suppressed requests and failures."""

    __tablename__ = "vlm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("track_events.id"), nullable=True)
    source_id: Mapped[int] = mapped_column(Integer)
    workflow_id: Mapped[int] = mapped_column(Integer)
    task_key: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32))
    # called | cache_hit | deduped | cancelled | suppressed_budget
    # | suppressed_cooldown | suppressed_sampling | failed | unavailable
    teacher_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tokens_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("track_events.id"))
    title: Mapped[str] = mapped_column(String(255))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_engine = None
SessionLocal: sessionmaker | None = None


def init_db(db_path: str | None = None) -> None:
    """Create engine, enable WAL, create tables. Idempotent."""
    global _engine, SessionLocal
    path = db_path or str(config.DB_PATH)
    config.ensure_dirs()
    _engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    Base.metadata.create_all(_engine)
    SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_session():
    assert SessionLocal is not None, "init_db() not called"
    return SessionLocal()
