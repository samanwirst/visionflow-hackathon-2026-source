"""Event-triggered VLM router (M2-AC2/AC4/AC5).

The router sits between a *cheap* geometric candidate event and an *expensive*
Teacher call. Geometric rules without a visual predicate never reach it, so they
make zero Teacher calls. For a visual predicate it enforces, per (task, source):

- a bounded **versioned cache** keyed by task/prompt + prompt/preprocessing
  version + model + ROI identity + scope, with TTL and LRU eviction so changed
  task/model/ROI inputs miss and stale results are never reused;
- **sampling** (minimum seconds between real calls) and a per-scope **cooldown**;
- a per-task **request budget**;
- **in-flight deduplication** so concurrent identical ROIs share one call.

Every decision is recorded as a ``RouteDecision`` with an explicit status, and
counters aggregate actual calls, cache hits/misses, suppressed requests (by
reason) and failures. A suppressed or failed route never yields a confirmed
visual condition.

There is no per-frame loop here: ``route()`` is invoked once per admitted
candidate event.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from app import config
from app.teacher import TeacherAdapter, TeacherResult, get_teacher

RouteStatus = str  # called | cache_hit | deduped | cancelled | suppressed_budget
#                    | suppressed_cooldown | suppressed_sampling | failed | unavailable


@dataclass
class RouteDecision:
    status: RouteStatus
    result: TeacherResult | None
    cache_key: str
    roi_hash: str
    reason: str = ""

    @property
    def confirmed(self) -> bool:
        return self.result is not None and self.result.confirmed


@dataclass
class _CacheEntry:
    result: TeacherResult
    expires_at: float


class VersionedTTLCache:
    """Bounded LRU cache with per-entry TTL. The key already encodes task,
    prompt/preprocessing version, model and ROI identity, so a changed version
    produces a different key (natural invalidation) rather than a stale hit."""

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._lock = threading.Lock()
        self.evictions = 0

    def get(self, key: str, now: float) -> TeacherResult | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry.result

    def put(self, key: str, result: TeacherResult, now: float) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(result=result, expires_at=now + self.ttl)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


@dataclass
class Counters:
    # The four distinct quantities the review (M2-004) asked to separate:
    adapter_invocations: int = 0  # times the adapter was invoked after admission
    outbound_attempts: int = 0    # actual outbound HTTP attempts (incl. retries)
    completed: int = 0            # adapter invocations that reached the provider
    unavailable: int = 0          # adapter invocations that made NO request
    calls: int = 0           # alias of ``completed`` (kept for API/UI compat)
    cache_hits: int = 0
    cache_misses: int = 0
    deduped: int = 0         # shared an in-flight identical call
    suppressed_budget: int = 0
    suppressed_cooldown: int = 0
    suppressed_sampling: int = 0
    failed: int = 0          # Teacher returned a non-confirming failure status
    confirmed: int = 0
    unknown: int = 0
    by_status: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "adapter_invocations": self.adapter_invocations,
            "outbound_attempts": self.outbound_attempts,
            "completed": self.completed,
            "unavailable": self.unavailable,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "deduped": self.deduped,
            "suppressed_budget": self.suppressed_budget,
            "suppressed_cooldown": self.suppressed_cooldown,
            "suppressed_sampling": self.suppressed_sampling,
            "failed": self.failed,
            "confirmed": self.confirmed,
            "unknown": self.unknown,
            "by_status": dict(self.by_status),
        }


def roi_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:32]


def cache_key(task_key: str, prompt: str, prompt_version: str, model: str, roi_digest: str) -> str:
    raw = "\x1f".join([task_key, prompt, prompt_version, model or "none", roi_digest])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class VlmRouter:
    """One router per process, shared across sources. Thread-safe."""

    def __init__(
        self,
        teacher_provider: Callable[[], TeacherAdapter] | None = None,
        *,
        sample_interval_s: float | None = None,
        cooldown_s: float | None = None,
        budget_per_task: int | None = None,
        cache: VersionedTTLCache | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._teacher_provider = teacher_provider or get_teacher
        self.sample_interval_s = (
            config.VLM_SAMPLE_INTERVAL_SECONDS if sample_interval_s is None else sample_interval_s
        )
        self.cooldown_s = config.VLM_COOLDOWN_SECONDS if cooldown_s is None else cooldown_s
        self.budget_per_task = (
            config.VLM_BUDGET_PER_TASK if budget_per_task is None else budget_per_task
        )
        self.cache = cache or VersionedTTLCache(
            config.VLM_CACHE_MAX_ENTRIES, config.VLM_CACHE_TTL_SECONDS
        )
        self._clock = clock
        self.counters = Counters()
        self._lock = threading.Lock()
        self._budget_used: dict[str, int] = {}
        self._last_call_at: dict[str, float] = {}      # per task key (sampling)
        self._cooldown_until: dict[str, float] = {}    # per scope (cooldown)
        self._inflight: dict[str, threading.Event] = {}
        self._inflight_result: dict[str, TeacherResult] = {}

    # -- helpers ---------------------------------------------------------------

    def _bump_status(self, status: str) -> None:
        self.counters.by_status[status] = self.counters.by_status.get(status, 0) + 1

    # -- main entry ------------------------------------------------------------

    def route(
        self,
        *,
        task_key: str,
        scope: str,
        prompt: str,
        labels: list[str],
        image_bytes: bytes,
        admit: Callable[[], bool] | None = None,
        sample_interval_s: float | None = None,
        cooldown_s: float | None = None,
        budget: int | None = None,
    ) -> RouteDecision:
        """Resolve one candidate ROI. Checks, in order: cache, budget, sampling,
        cooldown, in-flight dedup, then a real Teacher call. Per-workflow
        overrides fall back to the router defaults when None.

        ``admit``, when supplied, is re-checked at the transport boundary (just
        before each outbound attempt) by the real adapter's guard: if it returns
        False the attempt is revoked (e.g. the workflow was disabled) and the
        decision is ``cancelled`` with no outbound request (M2-007)."""
        sample_interval_s = self.sample_interval_s if sample_interval_s is None else sample_interval_s
        cooldown_s = self.cooldown_s if cooldown_s is None else cooldown_s
        budget = self.budget_per_task if budget is None else budget
        teacher = self._teacher_provider()
        model = getattr(teacher, "model", None) or teacher.provider_name
        prompt_version = config.TEACHER_PROMPT_VERSION
        digest = roi_hash(image_bytes)
        key = cache_key(task_key, prompt, prompt_version, model, digest)
        now = self._clock()

        # 1) Cache hit (identical ROI already classified for this task/model/ver)
        cached = self.cache.get(key, now)
        if cached is not None:
            with self._lock:
                self.counters.cache_hits += 1
                self._bump_status("cache_hit")
            return RouteDecision("cache_hit", cached, key, digest, "served from versioned cache")

        with self._lock:
            self.counters.cache_misses += 1
            # 2) In-flight dedup FIRST (M2-AC4): an identical ROI already being
            # computed is shared without passing it through the budget, sampling
            # and cooldown gates again (which the owner already paid), and
            # without reserving a second slot.
            event = self._inflight.get(key)
            if event is not None:
                dedup_wait = event
            else:
                dedup_wait = None
                # 3) Budget: is there ANY budget left to start a call? Per-attempt
                # consumption happens at the transport boundary (``_attempt_guard``).
                used = self._budget_used.get(task_key, 0)
                if used >= budget:
                    self.counters.suppressed_budget += 1
                    self._bump_status("suppressed_budget")
                    return RouteDecision(
                        "suppressed_budget", None, key, digest,
                        f"task budget exhausted ({used}/{budget})",
                    )
                # 4) Sampling (min interval between real calls for this task)
                last = self._last_call_at.get(task_key)
                if last is not None and (now - last) < sample_interval_s:
                    self.counters.suppressed_sampling += 1
                    self._bump_status("suppressed_sampling")
                    return RouteDecision(
                        "suppressed_sampling", None, key, digest,
                        f"within sampling interval ({sample_interval_s}s)",
                    )
                # 5) Cooldown (per scope)
                cd = self._cooldown_until.get(scope)
                if cd is not None and now < cd:
                    self.counters.suppressed_cooldown += 1
                    self._bump_status("suppressed_cooldown")
                    return RouteDecision(
                        "suppressed_cooldown", None, key, digest,
                        f"scope in cooldown for {cd - now:.1f}s",
                    )
                # Commit: reserve the in-flight slot + sampling + cooldown now that
                # we will call. Budget is consumed per outbound attempt below.
                own_event = threading.Event()
                self._inflight[key] = own_event
                self._last_call_at[task_key] = now
                self._cooldown_until[scope] = now + cooldown_s

        if dedup_wait is not None:
            dedup_wait.wait(timeout=config.VLM_TIMEOUT_SECONDS + 5)
            shared = self._inflight_result.get(key)
            with self._lock:
                self.counters.deduped += 1
                self._bump_status("deduped")
            if shared is not None:
                return RouteDecision("deduped", shared, key, digest, "shared in-flight call")
            # the owner failed to publish a result; treat as a miss without a call
            return RouteDecision(
                "deduped", None, key, digest, "in-flight owner produced no result"
            )

        # 6) Real Teacher call (we own the in-flight slot for this key).
        # The attempt guard enforces and accounts the shared per-task budget at
        # the transport boundary: every outbound attempt (including retries)
        # consumes one unit and is counted; when the budget is gone the guard
        # returns False so the adapter stops before sending (M2-AC4).
        guard_state = {"invoked": False, "granted": 0, "revoked": False}

        def _attempt_guard() -> bool:
            # M2-007: the authoritative approval recheck and the admission of an
            # outbound attempt must be ONE atomic boundary, serialized with a
            # workflow disable. The API disable handler commits ``enabled=False``
            # under ``dispatch_admission_lock``; by taking the same lock here
            # around the ``admit()`` read AND the budget grant, a disable can only
            # land fully before this boundary (admit reads False -> revoked, no
            # request) or fully after it (the attempt was already admitted). The
            # previous code read approval OUTSIDE any lock, so a disable could
            # commit and return 200 between the read and the grant, then still
            # admit one HTTP POST (the reviewed race). The lock is NOT held during
            # the actual inference/transport, only across this decision.
            from app.pipeline import dispatch_admission_lock  # lazy: avoids cycle

            with dispatch_admission_lock:
                if admit is not None and not admit():
                    guard_state["invoked"] = True
                    guard_state["revoked"] = True
                    return False
                with self._lock:
                    guard_state["invoked"] = True
                    used_now = self._budget_used.get(task_key, 0)
                    if used_now >= budget:
                        return False
                    self._budget_used[task_key] = used_now + 1
                    self.counters.outbound_attempts += 1
                    guard_state["granted"] += 1
            return True

        try:
            result = teacher.classify(image_bytes, prompt, labels, on_attempt=_attempt_guard)
        except Exception as exc:  # noqa: BLE001 — never propagate into the pipeline
            result = TeacherResult(status="error", model=model, error=f"adapter raised: {exc}"[:500])

        # M2-007: admission was revoked at the transport boundary (workflow
        # disabled); no outbound request/image disclosure occurred. Record it as
        # cancelled — not a provider failure — consuming no budget and counting no
        # completed call. Any in-flight duplicate wakes with no shared result.
        if guard_state["revoked"] and guard_state["granted"] == 0:
            with self._lock:
                self._bump_status("cancelled")
                self._inflight_result.pop(key, None)
                own = self._inflight.pop(key, None)
            if own is not None:
                own.set()
            return RouteDecision(
                "cancelled", None, key, digest,
                "workflow disabled before provider admission",
            )

        reports = bool(getattr(teacher, "reports_attempts", False))
        with self._lock:
            self.counters.adapter_invocations += 1
            self._bump_status(result.status)
            if result.status == "missing_credentials":
                # Unavailable provider: no request was made, no budget consumed.
                self.counters.unavailable += 1
            elif reports:
                # Real adapter: an outbound request happened iff the guard granted
                # at least one attempt. A preflight/queue failure (oversize image,
                # semaphore timeout, budget-exhausted retry) never granted the
                # guard, so it consumed no budget and is not a completed provider
                # call — it is only recorded as a failure below (M2-004).
                if guard_state["granted"] > 0:
                    self.counters.completed += 1
                    self.counters.calls += 1
            else:
                self.counters.completed += 1
                self.counters.calls += 1
                # Adapters that do not report attempts via the guard (e.g. test
                # stubs) count as a single outbound attempt that consumes one
                # budget unit, so the shared budget still bounds them.
                if not guard_state["invoked"]:
                    self._budget_used[task_key] = self._budget_used.get(task_key, 0) + 1
                    self.counters.outbound_attempts += 1
            if result.confirmed:
                self.counters.confirmed += 1
            elif result.status == "unknown":
                self.counters.unknown += 1
            elif result.status != "ok":
                self.counters.failed += 1
            # Only cache deterministic, non-transient outcomes. Transient
            # failures (timeout/error) and soft non-confirmations
            # (low_confidence) are not cached so a later candidate retries.
            if result.status in ("ok", "unknown"):
                self.cache.put(key, result, now)
            self._inflight_result[key] = result
            own = self._inflight.pop(key, None)
        if own is not None:
            own.set()
        return RouteDecision("called", result, key, digest, f"teacher status={result.status}")

    def stats(self) -> dict:
        teacher = self._teacher_provider()
        snap = self.counters.snapshot()
        snap.update(
            {
                "teacher_provider": teacher.provider_name,
                "teacher_available": teacher.available,
                "cache_size": len(self.cache),
                "cache_evictions": self.cache.evictions,
                "sample_interval_s": self.sample_interval_s,
                "cooldown_s": self.cooldown_s,
                "budget_per_task": self.budget_per_task,
            }
        )
        return snap


# Process-wide router (tests may replace it).
_router: VlmRouter | None = None
_router_lock = threading.Lock()


def get_router() -> VlmRouter:
    global _router
    with _router_lock:
        if _router is None:
            _router = VlmRouter()
        return _router


def set_router(router: VlmRouter | None) -> None:
    global _router
    with _router_lock:
        _router = router
