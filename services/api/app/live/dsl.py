"""Backend-owned task DSL and declared capabilities for the live agent.

The Claude CLI planner proposes a task; the BACKEND owns the authoritative schema
and allowlists here. A planner (or an injected instruction inside a judge prompt)
can never escape this typed validation: anything outside the allowlisted classes,
triggers, actions, and canonical zone geometry is rejected, not silently coerced.

No arbitrary URLs, code, file paths, or tools can appear in a valid task.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator
import math

from app.live import config

# COCO classes the YOLO detector genuinely produces and that are useful for the
# automotive + general demo. These are REAL detectable classes, not canned
# phrases. "bottle" is included because YOLO11n (COCO) detects it — the bottle
# example must travel the same general path, never a hardcoded branch.
SUPPORTED_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "bottle",
    "cup",
    "cell phone",
    "laptop",
    "backpack",
    "handbag",
    "suitcase",
    "chair",
}

# Geometric temporal primitives, all backed by the existing ZoneEventEngine.
TriggerType = Literal[
    "presence_in_zone",   # object of class present inside the zone (debounced)
    "entered_zone",
    "exited_zone",
    "dwell_in_zone",
    "count_exceeded",
]

# Only local, side-effect-free actions. No outbound delivery (see M1 actions.py).
ActionType = Literal["notify", "focus_notify"]

# Map our live trigger names onto the shared vision EventType values.
TRIGGER_TO_EVENT = {
    "presence_in_zone": "object_entered_zone",  # presence == first entry, debounced
    "entered_zone": "object_entered_zone",
    "exited_zone": "object_exited_zone",
    "dwell_in_zone": "dwell_in_zone",
    "count_exceeded": "object_count_exceeded",
}


# Canonical full-frame polygon (unit square). Kept as a module constant so the
# zone validator is idempotent across re-validation.
_FULL_FRAME_SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


class ZoneSpec(BaseModel):
    """Canonical zone geometry. ``full_frame`` expands to the unit square so the
    planner can say "watch the bottle" without drawing a polygon."""

    model_config = {"extra": "forbid"}

    type: Literal["full_frame", "polygon"] = "full_frame"
    polygon: list[tuple[float, float]] = Field(default_factory=list, max_length=50)

    @field_validator("polygon", mode="before")
    @classmethod
    def _strict_coordinates(cls, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("polygon must be an array of numeric coordinate pairs")
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("polygon vertices must contain exactly two coordinates")
            for coordinate in point:
                if type(coordinate) not in (int, float):
                    raise ValueError("coordinates must be JSON numbers, never booleans or strings")
                try:
                    finite = math.isfinite(coordinate)
                except (TypeError, OverflowError):
                    finite = False
                if not finite or not 0 <= coordinate <= 1:
                    raise ValueError("coordinates must be finite and normalized to 0..1")
        return value

    @model_validator(mode="after")
    def _canonicalize(self) -> "ZoneSpec":
        if self.type == "full_frame":
            # A full_frame zone and an explicit DIFFERENT polygon are CONFLICTING
            # intents. Rejecting (instead of silently discarding the polygon) means a
            # drawn region can never be silently lost (LIVE-VIDEO-005). The canonical
            # unit square is allowed so re-validating an already-canonical zone is
            # idempotent.
            if self.polygon and [tuple(p) for p in self.polygon] != _FULL_FRAME_SQUARE:
                raise ValueError(
                    "full_frame zone must not carry a polygon; use type='polygon' to draw a region"
                )
            object.__setattr__(self, "polygon", list(_FULL_FRAME_SQUARE))
            return self
        if len(self.polygon) < 3:
            raise ValueError("polygon zone needs at least 3 vertices")
        for x, y in self.polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("polygon coordinates must be normalized to 0..1")
        return self


class VisualObservation(BaseModel):
    """An OPTIONAL non-actioning semantic/temporal check routed to the Teacher VLM
    at event time (e.g. "is the bottle being opened"). It is recorded as an
    observation with explicit uncertainty; a weak/uncalibrated Teacher result can
    NEVER by itself authorize an action. This preserves intent for requests like
    "focus on the bottle when somebody opens it" without faking a temporal event
    from a single frame."""

    question: str = Field(min_length=3, max_length=300)
    labels: list[str] = Field(min_length=2, max_length=6)
    temporal: bool = False  # True => use a before/after pair when available

    @field_validator("labels")
    @classmethod
    def _norm(cls, v: list[str]) -> list[str]:
        norm = [x.strip().lower() for x in v if x.strip()]
        if len(set(norm)) < 2:
            raise ValueError("visual observation needs at least two distinct labels")
        return norm


class LiveTask(BaseModel):
    """The authoritative, validated live monitoring task.

    Strict and extra-forbidden (recursively): an unknown field (an injected
    ``code``/``url``/``command`` side channel) is rejected, never silently
    dropped, and numeric constraints are never coerced from JSON booleans or
    strings (LIVE-VIDEO-005)."""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=120)
    object_class: str
    zone: ZoneSpec = Field(default_factory=ZoneSpec)
    trigger: TriggerType
    duration_seconds: float | None = Field(default=None, ge=1, le=3600)
    threshold: int | None = Field(default=None, ge=1, le=100, strict=True)
    # Supported event cooldown/debounce in SOURCE-video seconds. When None the
    # VIDEO engine applies config.DEFAULT_COOLDOWN_SECONDS. Faithfully carried,
    # never silently discarded (LIVE-VIDEO-005).
    cooldown_seconds: float | None = Field(default=None, ge=0, le=3600)
    action: ActionType = "notify"
    focus: StrictBool = False
    visual_observation: VisualObservation | None = None
    explanation: str = Field(default="", max_length=800)

    @field_validator("duration_seconds", "threshold", "cooldown_seconds", mode="before")
    @classmethod
    def _reject_bool_numbers(cls, v):
        # A JSON boolean is NEVER a valid count/duration/cooldown. Reject before
        # type coercion so ``true``/``false`` cannot become 1/0.
        if v is not None:
            if type(v) not in (int, float):
                raise ValueError("must be a JSON number, never a boolean or string")
            try:
                finite = math.isfinite(v)
            except (OverflowError, TypeError):
                finite = False
            if not finite:
                raise ValueError("must be a finite number")
        return v

    @field_validator("object_class")
    @classmethod
    def _class_allowed(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SUPPORTED_CLASSES:
            raise ValueError(
                f"unsupported object_class '{v}'; detectable classes: {sorted(SUPPORTED_CLASSES)}"
            )
        return v

    @model_validator(mode="after")
    def _trigger_params(self) -> "LiveTask":
        if self.trigger == "dwell_in_zone" and not self.duration_seconds:
            raise ValueError("dwell_in_zone requires duration_seconds >= 1")
        if self.trigger == "count_exceeded" and not self.threshold:
            raise ValueError("count_exceeded requires threshold >= 1")
        if self.action == "focus_notify":
            object.__setattr__(self, "focus", True)
        # Semantic / state-change conditions are NOT supported on the enabled
        # VIDEO surface: there is no Teacher verification and no closest-geometry
        # fallback. Reject fail-closed so a presence rule can never stand in for a
        # "when opened/unattended" request (LIVE-VIDEO-001). The legacy camera
        # path (local QA only, camera ingest enabled) still carries it.
        if self.visual_observation is not None and not config.CAMERA_INGEST_ENABLED:
            raise ValueError(
                "semantic/state-change visual observations are not supported in video mode; "
                "only geometric zone rules are available"
            )
        return self


class TaskStateRequest(BaseModel):
    """Strictly-typed pause/disable/re-enable control. Only real JSON booleans are
    accepted; the string "false" (or any non-bool) is rejected, not coerced, so a
    control request can never silently flip to the wrong state (LIVE-007)."""

    model_config = {"extra": "forbid"}
    enabled: StrictBool | None = None
    paused: StrictBool | None = None


def capabilities_doc() -> dict:
    """Machine-readable capability declaration handed to the planner. Keeps the
    planner honest about exactly what the backend can execute."""
    doc = {
        "object_classes": sorted(SUPPORTED_CLASSES),
        "triggers": [
            "presence_in_zone",
            "entered_zone",
            "exited_zone",
            "dwell_in_zone",
            "count_exceeded",
        ],
        "actions": ["notify", "focus_notify"],
        "zone": ["full_frame", "polygon"],
        "duration_seconds": "dwell duration in SOURCE-video seconds (>= 1), e.g. 1.25",
        "cooldown_seconds": "optional event debounce in SOURCE-video seconds (default 10)",
        "focus": "highlight / enlarge the tracked bounding box in the Judges UI (NO physical camera pan/tilt)",
        "natural_language_commands": {
            "scope": "Complete supported-command grammar; not arbitrary action understanding",
            "examples": [
                "Notify me locally when a car is visible in the full frame.",
                "Raise a local notification when a car remains in the full frame for exactly 1.25 seconds; set the cooldown to 2.5 seconds.",
                "Alert me locally when a car exits the full frame, with a cooldown of 3.5 seconds.",
            ],
            "limits": "One object class, whole-frame presence/exit/dwell, local notify only. "
                      "Full-frame entry is not verified crossing; polygon entry requires reviewed JSON. "
                      "Unknown qualifiers, negation, comparisons, composites and external actions reject before model calls.",
        },
    }
    if config.CAMERA_INGEST_ENABLED:
        doc["visual_observation"] = (
            "optional non-actioning semantic/temporal check routed to a weak local VLM; "
            "reports uncertainty and never by itself triggers an action"
        )
        doc["notes"] = [
            "Opening/unattended/semantic conditions are not directly detectable by the geometric engine.",
            "Express them as a visual_observation (uncertain) plus the closest geometric trigger, or ask for clarification.",
            "Never silently drop a requested condition.",
        ]
    else:
        # Enabled VIDEO surface: geometric rules only, fail-closed (LIVE-VIDEO-001).
        doc["visual_observation"] = "UNSUPPORTED in video mode"
        doc["notes"] = [
            "Opening/unattended/loitering-intent/damage and any state-change or semantic condition are UNSUPPORTED.",
            "There is no visual_observation, no Teacher verification and NO closest-geometry fallback for a semantic request.",
            "If the request depends on such a condition, set supported=false with unsupported_reason; never approximate it with a presence/dwell rule.",
        ]
    return doc
