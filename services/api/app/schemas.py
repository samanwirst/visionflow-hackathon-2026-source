"""Pydantic API and workflow-DSL schemas.

The workflow DSL is the validated contract between user intent and the event
engine. Unsupported trigger/action types fail validation with a clear error —
they are never silently coerced.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Workflow DSL (PRD section 8). M1 supports the four zone triggers; the
# visual_condition_verified trigger arrives with AutoLearn (M3).
# ---------------------------------------------------------------------------

TriggerType = Literal[
    "object_entered_zone",
    "object_exited_zone",
    "dwell_in_zone",
    "object_count_exceeded",
]

# M1 ships only the local, side-effect-free notification action. Arbitrary
# outbound webhooks were removed (M1-011/M1-012): the SSRF/DNS-rebinding guard
# could not safely protect the delivery connection without a risky custom
# DNS/TLS egress transport, so the capability is disabled rather than shipped
# insecurely. A validated-egress integration can be reintroduced later.
ActionType = Literal["create_notification"]

# COCO classes relevant to the automotive vertical, plus "person".
SUPPORTED_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}


class Trigger(BaseModel):
    type: TriggerType
    zone_id: int
    duration_seconds: float | None = Field(default=None, ge=1, le=86400)
    threshold: int | None = Field(default=None, ge=1, le=1000)

    @model_validator(mode="after")
    def check_params(self) -> "Trigger":
        if self.type == "dwell_in_zone" and not self.duration_seconds:
            raise ValueError("dwell_in_zone requires duration_seconds >= 1")
        if self.type == "object_count_exceeded" and not self.threshold:
            raise ValueError("object_count_exceeded requires threshold >= 1")
        return self


class Action(BaseModel):
    type: ActionType
    config: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_config(self) -> "Action":
        if self.type == "create_notification":
            title = self.config.get("title")
            # Optional: falls back to a generated title. If present it must be a
            # bounded string — a dict/list would be rejected by SQLite at commit
            # time and strand the source in 'processing' (M1-008).
            if title is not None and (not isinstance(title, str) or len(title) > 255):
                raise ValueError(
                    "create_notification config.title must be a string of at most 255 characters"
                )
        return self


class VisualPredicate(BaseModel):
    """An optional event-routed VLM check (M2). When present, the geometric
    trigger produces a *candidate*; the triggering object's ROI is sent to the
    Teacher and the action fires only if the Teacher confirms ``expected_label``.
    Geometric-only workflows (``visual_predicate`` absent) make zero VLM calls.

    Routing settings are per-workflow overrides of the server defaults. The
    prompt/labels are fixed server-side vocabulary, never executable code."""

    prompt: str = Field(min_length=3, max_length=500)
    labels: list[str] = Field(min_length=2, max_length=8)
    expected_label: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=255)
    # Per-workflow routing overrides (optional). None -> use server config.
    sample_interval_seconds: float | None = Field(default=None, ge=0, le=3600)
    cooldown_seconds: float | None = Field(default=None, ge=0, le=86400)
    budget: int | None = Field(default=None, ge=1, le=100000)

    @model_validator(mode="after")
    def check_labels(self) -> "VisualPredicate":
        norm = [l.strip().lower() for l in self.labels if l.strip()]
        if len(norm) < 2:
            raise ValueError("visual predicate needs at least two distinct labels")
        if "unknown" in norm:
            raise ValueError("'unknown' is reserved; do not include it in labels")
        if self.expected_label.strip().lower() not in norm:
            raise ValueError("expected_label must be one of labels")
        return self


class WorkflowDSL(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source_id: int
    object_class: str
    trigger: Trigger
    action: Action
    visual_predicate: VisualPredicate | None = None
    cooldown_seconds: float = Field(default=10.0, ge=0, le=86400)
    requires_approval: bool = True

    @field_validator("object_class")
    @classmethod
    def check_class(cls, v: str) -> str:
        if v not in SUPPORTED_CLASSES:
            raise ValueError(
                f"unsupported object_class '{v}'; supported: {sorted(SUPPORTED_CLASSES)}"
            )
        return v


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------


class ZoneCreate(BaseModel):
    source_id: int
    name: str = Field(min_length=1, max_length=255)
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=50)

    @field_validator("polygon")
    @classmethod
    def check_normalized(cls, v):
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("polygon coordinates must be normalized to 0..1")
        return v


class ZoneOut(BaseModel):
    id: int
    source_id: int
    name: str
    polygon: list[tuple[float, float]]


class SourceOut(BaseModel):
    id: int
    type: str
    name: str
    status: str
    error: str | None
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    frames_processed: int
    avg_inference_ms: float | None
    created_at: datetime


class WorkflowOut(BaseModel):
    id: int
    source_id: int
    name: str
    dsl: WorkflowDSL
    enabled: bool
    created_at: datetime


class CompileRequest(BaseModel):
    source_id: int
    text: str = Field(min_length=1, max_length=2000)


class CompileResponse(BaseModel):
    ok: bool
    draft: WorkflowDSL | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    explanation: str = ""


class EventOut(BaseModel):
    id: int
    source_id: int
    workflow_id: int
    track_id: int | None
    type: str
    class_name: str | None
    video_timestamp: float
    confidence: float
    detail: str
    evidence_url: str | None
    action_status: str
    # Visual routing provenance (M2-AC5): None for geometric-only events.
    visual_status: str | None = None
    visual_label: str | None = None
    visual_confidence: float | None = None
    visual_routing: dict | None = None
    created_at: datetime


class ActionOut(BaseModel):
    id: int
    event_id: int
    action_type: str
    status: str
    attempt_count: int
    error: str | None
    executed_at: datetime | None


class NotificationOut(BaseModel):
    id: int
    event_id: int
    title: str
    message: str
    created_at: datetime
