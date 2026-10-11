"""Deterministic temporal event engine over tracked detections and zones.

Operates on video time (seconds derived from frame index / fps), not wall
clock, so results are reproducible and testable.

Correctness rules implemented (PRD section 9):
- Track-based identity (one state per track_id).
- Debounce: N consecutive frames inside/outside before a transition counts.
- One-shot events per zone visit: ENTER/EXIT/DWELL fire once per visit.
- Cooldown window per (event type, track) against flapping re-entries.
- Tolerance for brief tracking loss (grace period before a lost track is
  treated as exited).
- Full state reset when the object truly leaves.
- Deterministic dedupe keys suitable for idempotent action execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from visionflow_vision.zones import Zone, bbox_anchor


class EventType(str, Enum):
    ENTER = "object_entered_zone"
    EXIT = "object_exited_zone"
    DWELL = "dwell_in_zone"
    COUNT_EXCEEDED = "object_count_exceeded"


@dataclass(frozen=True)
class TrackedObject:
    """One tracked detection in a frame (absolute pixel bbox)."""

    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def anchor(self) -> tuple[float, float]:
        return bbox_anchor(self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class ZoneEvent:
    type: EventType
    track_id: int | None  # None for aggregate events (count)
    class_name: str | None
    timestamp: float  # video time, seconds
    confidence: float
    dedupe_key: str
    detail: str = ""


@dataclass
class _TrackState:
    consecutive_in: int = 0
    consecutive_out: int = 0
    inside: bool = False
    entered_at: float | None = None
    dwell_emitted: bool = False
    last_seen: float = 0.0
    last_confidence: float = 0.0


class ZoneEventEngine:
    """Evaluates one (zone, object class, event type) rule over a frame stream.

    One engine instance per enabled workflow. Call update() once per sampled
    frame with every tracked object of the frame; it returns the events that
    fired on that frame.
    """

    def __init__(
        self,
        zone: Zone,
        object_class: str,
        event_type: EventType,
        *,
        frame_w: int,
        frame_h: int,
        dwell_seconds: float = 0.0,
        count_threshold: int = 0,
        cooldown_seconds: float = 10.0,
        debounce_frames: int = 2,
        lost_track_grace_seconds: float = 1.0,
        rule_key: str = "rule",
    ) -> None:
        if event_type == EventType.DWELL and dwell_seconds <= 0:
            raise ValueError("dwell_in_zone requires dwell_seconds > 0")
        if event_type == EventType.COUNT_EXCEEDED and count_threshold <= 0:
            raise ValueError("object_count_exceeded requires count_threshold > 0")
        self.zone = zone
        self.object_class = object_class
        self.event_type = event_type
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.dwell_seconds = dwell_seconds
        self.count_threshold = count_threshold
        self.cooldown_seconds = cooldown_seconds
        self.debounce_frames = max(1, debounce_frames)
        self.lost_track_grace_seconds = lost_track_grace_seconds
        self.rule_key = rule_key
        self._tracks: dict[int, _TrackState] = {}
        self._last_event_at: dict[str, float] = {}  # cooldown per dedupe scope
        self._count_above: bool = False
        self._visit_counter: dict[int, int] = {}
        # Deterministic episode counter for aggregate (track-less) events; it
        # advances on every rising threshold crossing so that each distinct
        # crossing produces a unique dedupe key instead of a fixed "visit0".
        self._count_episode: int = 0

    # ------------------------------------------------------------------

    def _cooldown_ok(self, scope: str, now: float) -> bool:
        last = self._last_event_at.get(scope)
        if last is not None and (now - last) < self.cooldown_seconds:
            return False
        self._last_event_at[scope] = now
        return True

    def _emit(
        self,
        etype: EventType,
        track_id: int | None,
        class_name: str | None,
        now: float,
        confidence: float,
        detail: str = "",
        *,
        episode: int | None = None,
    ) -> ZoneEvent | None:
        # Track-based events are keyed by their per-track visit counter;
        # aggregate (track_id is None) events pass an explicit episode so each
        # rising crossing is distinct.
        if episode is None:
            episode = self._visit_counter.get(track_id, 0) if track_id is not None else 0
        scope = f"{self.rule_key}:{etype.value}:{track_id}"
        if not self._cooldown_ok(scope, now):
            return None
        dedupe_key = f"{scope}:visit{episode}"
        return ZoneEvent(
            type=etype,
            track_id=track_id,
            class_name=class_name,
            timestamp=now,
            confidence=confidence,
            dedupe_key=dedupe_key,
            detail=detail,
        )

    # ------------------------------------------------------------------

    def update(self, objects: Iterable[TrackedObject], now: float) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        seen_ids: set[int] = set()

        for obj in objects:
            if obj.class_name != self.object_class:
                continue
            seen_ids.add(obj.track_id)
            st = self._tracks.setdefault(obj.track_id, _TrackState())
            st.last_seen = now
            st.last_confidence = obj.confidence
            in_zone = self.zone.contains(obj.anchor, self.frame_w, self.frame_h)

            if in_zone:
                st.consecutive_in += 1
                st.consecutive_out = 0
                if not st.inside and st.consecutive_in >= self.debounce_frames:
                    st.inside = True
                    st.entered_at = now
                    st.dwell_emitted = False
                    self._visit_counter[obj.track_id] = (
                        self._visit_counter.get(obj.track_id, 0) + 1
                    )
                    if self.event_type == EventType.ENTER:
                        ev = self._emit(
                            EventType.ENTER, obj.track_id, obj.class_name, now,
                            obj.confidence,
                            detail=f"{obj.class_name} #{obj.track_id} entered {self.zone.name}",
                        )
                        if ev:
                            events.append(ev)
                if (
                    st.inside
                    and self.event_type == EventType.DWELL
                    and not st.dwell_emitted
                    and st.entered_at is not None
                    and (now - st.entered_at) >= self.dwell_seconds
                ):
                    st.dwell_emitted = True
                    ev = self._emit(
                        EventType.DWELL, obj.track_id, obj.class_name, now,
                        obj.confidence,
                        detail=(
                            f"{obj.class_name} #{obj.track_id} stayed in {self.zone.name} "
                            f">= {self.dwell_seconds:.0f}s"
                        ),
                    )
                    if ev:
                        events.append(ev)
            else:
                st.consecutive_out += 1
                st.consecutive_in = 0
                if st.inside and st.consecutive_out >= self.debounce_frames:
                    events.extend(self._handle_true_exit(obj.track_id, st, now))

        # Lost-track handling: tracks not seen for longer than the grace
        # period are treated as having left the zone.
        for tid, st in list(self._tracks.items()):
            if tid in seen_ids:
                continue
            if (now - st.last_seen) > self.lost_track_grace_seconds:
                if st.inside:
                    events.extend(self._handle_true_exit(tid, st, now))
                del self._tracks[tid]

        # Aggregate count rule: fires on the rising edge above threshold.
        if self.event_type == EventType.COUNT_EXCEEDED:
            inside_count = sum(1 for s in self._tracks.values() if s.inside)
            if inside_count > self.count_threshold and not self._count_above:
                self._count_above = True
                self._count_episode += 1
                ev = self._emit(
                    EventType.COUNT_EXCEEDED, None, self.object_class, now, 1.0,
                    detail=(
                        f"{inside_count} x {self.object_class} in {self.zone.name} "
                        f"(threshold {self.count_threshold})"
                    ),
                    episode=self._count_episode,
                )
                if ev:
                    events.append(ev)
            elif inside_count <= self.count_threshold:
                self._count_above = False

        return events

    def _handle_true_exit(self, track_id: int, st: _TrackState, now: float) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        st.inside = False
        st.entered_at = None
        st.dwell_emitted = False
        if self.event_type == EventType.EXIT:
            ev = self._emit(
                EventType.EXIT, track_id, self.object_class, now,
                st.last_confidence,
                detail=f"{self.object_class} #{track_id} exited {self.zone.name}",
            )
            if ev:
                events.append(ev)
        return events
