"""Clean, non-actioning Teacher observation adapter for the live app.

Routes a SELECTED event frame (or a before/after pair) to a configured private
OpenAI-compatible VLM endpoint for a semantic/temporal observation. The deployed
model is a weak, UNCALIBRATED local VLM: its output is recorded as an observation
with explicit uncertainty and can NEVER by itself authorize an action. There is
no per-frame call — only an event-triggered or explicit manual probe, bounded by
a per-room budget + cooldown, with a small result cache scoped by room, model,
question, labels and image identity.

Hard properties enforced here:
- Strict typed provider contract: confidence must be a finite number in [0,1]
  (booleans rejected); the returned label must be one of the REQUESTED labels or
  the observation is ``unsupported`` (the provider cannot answer that question).
- ONE absolute monotonic deadline spans admission (concurrency) AND transport;
  expired waiting work never starts a provider call, and a stalling/trickling
  response is terminated. Concurrency capacity is held until the underlying work
  actually ends.
- A generation guard (``should_proceed``) is re-checked after admission, right
  before the first byte is written, so no new provider request starts after a
  room/task disable acknowledgment.

The server base URL/token/model are configuration, never client-supplied. This
module does not modify the Teacher server or its contract.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass

from app.live import config

logger = logging.getLogger("visionflow.live.teacher")

_room_calls: dict[str, int] = {}
_room_last: dict[str, float] = {}
_cache: dict[str, "Observation"] = {}
_lock = threading.Lock()
# Concurrency capacity one: acquired INSIDE the worker thread and released only
# when the real provider call ends, so an abandoned (timed-out) call still holds
# capacity until it truly completes.
_concurrency = threading.Semaphore(1)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-teacher")


@dataclass
class Observation:
    status: str                 # ok|unavailable|budget|cooldown|error|unsupported|stale|timeout
    label: str | None = None
    confidence: float | None = None
    uncertain: bool = True      # always true: the model is uncalibrated
    in_schema: bool = False     # label is one of the requested labels
    image_hash: str | None = None
    image_count: int = 0
    model: str | None = None
    revision: str | None = None
    latency_ms: float | None = None
    raw: dict | None = None     # bounded provenance from the server contract
    reason: str | None = None


# Tests inject a deterministic transport here (QA ONLY) so no real GPU call runs.
transport = None


def available() -> bool:
    return bool(config.TEACHER_BASE_URL) or transport is not None


def _norm_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for x in labels or []:
        s = str(x).strip().lower()
        if s and s not in out:
            out.append(s)
    return out[:6]


def _cache_key(room_id: str, question: str, labels: list[str], images: list[bytes]) -> str:
    h = hashlib.sha256()
    for img in images:
        h.update(img)
        h.update(b"|")
    h.update(question.strip().lower().encode())
    h.update(b"|")
    h.update(" ".join(sorted(labels)).encode())
    h.update(b"|")
    h.update((config.TEACHER_MODEL or "").encode())
    h.update(b"|")
    h.update(room_id.encode())
    return h.hexdigest()


def _reserve(room_id: str) -> tuple[bool, str | None]:
    now = time.monotonic()
    with _lock:
        used = _room_calls.get(room_id, 0)
        if used >= config.TEACHER_ROOM_BUDGET:
            return False, "budget"
        last = _room_last.get(room_id)
        if last is not None and (now - last) < config.TEACHER_COOLDOWN_SECONDS:
            return False, "cooldown"
        _room_calls[room_id] = used + 1
        _room_last[room_id] = now
        return True, None


def _reset_for_tests() -> None:
    with _lock:
        _room_calls.clear()
        _room_last.clear()
        _cache.clear()


def _valid_confidence(v) -> float | None:
    """Accept only a finite real number in [0,1]. Booleans and NaN/inf/out-of-range
    are rejected (returned as None), never coerced."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    if not math.isfinite(f) or not (0.0 <= f <= 1.0):
        return None
    return f


def observe(room_id: str, question: str, labels: list[str], images: list[bytes],
            *, should_proceed=None) -> Observation:
    """Produce a non-actioning observation for one or two JPEG frames.

    ``should_proceed`` (optional) is a generation guard re-checked after admission
    and immediately before the provider call starts; if it returns False the call
    is cancelled with a ``stale`` observation and NO provider request is sent."""
    if not images:
        return Observation(status="error", reason="no image", uncertain=True)
    images = images[:2]  # provider supports at most two ordered images
    labels = _norm_labels(labels)
    if len(labels) < 2:
        return Observation(status="unsupported", reason="need >= 2 distinct labels",
                           uncertain=True)
    image_hash = _cache_key(room_id, question, labels, images)

    with _lock:
        cached = _cache.get(image_hash)
    if cached is not None:
        return cached

    if not available():
        return Observation(status="unavailable", image_hash=image_hash,
                           reason="Teacher endpoint not configured", uncertain=True)

    if should_proceed is not None and not should_proceed():
        return Observation(status="stale", image_hash=image_hash,
                           reason="cancelled before admission", uncertain=True)

    ok, reason = _reserve(room_id)
    if not ok:
        return Observation(status=reason or "unavailable", image_hash=image_hash,
                           reason=f"teacher refused: {reason}", uncertain=True)

    deadline = time.monotonic() + config.TEACHER_TIMEOUT_SECONDS
    if transport is not None:
        if should_proceed is not None and not should_proceed():
            return Observation(status="stale", image_hash=image_hash,
                               reason="cancelled before provider call", uncertain=True)
        obs = transport(question, labels, images)
        obs = _validate_against_schema(obs, labels)
    else:
        obs = _call_server(question, labels, images, deadline, should_proceed)
    obs.image_hash = image_hash
    obs.image_count = len(images)
    # Only cache a conclusive provider answer; transient refusals/timeouts/stale
    # must be retryable.
    if obs.status in ("ok", "unsupported"):
        with _lock:
            _cache[image_hash] = obs
    return obs


def _validate_against_schema(obs: "Observation", labels: list[str]) -> "Observation":
    """Enforce the typed contract on any ok observation: confidence must be valid
    and the label must be one of the requested labels, else it is unsupported."""
    if obs.status != "ok":
        obs.uncertain = True
        return obs
    obs.confidence = _valid_confidence(obs.confidence)
    label = (obs.label or "").strip().lower() or None
    obs.label = label
    obs.in_schema = label in labels if label else False
    obs.uncertain = True  # uncalibrated model is always flagged uncertain
    if not obs.in_schema:
        obs.status = "unsupported"
        obs.reason = (obs.reason or "") + " | label not in requested schema"
    return obs


def _do_post(url: str, payload: dict, headers: dict, deadline: float) -> "Observation":
    """Runs in a worker thread. Acquires concurrency capacity (held until the real
    call ends) and performs the bounded HTTP call under the shared deadline."""
    import httpx

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return Observation(status="timeout", reason="deadline before admission", uncertain=True)
    if not _concurrency.acquire(timeout=remaining):
        return Observation(status="error", reason="teacher busy", uncertain=True)
    t0 = time.monotonic()
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Observation(status="timeout", reason="deadline after admission", uncertain=True)
        timeout = httpx.Timeout(remaining, connect=remaining, read=remaining,
                                write=remaining, pool=remaining)
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("teacher call failed: %s", exc)
        return Observation(status="error", reason=str(exc)[:200], uncertain=True)
    finally:
        _concurrency.release()
    latency_ms = (time.monotonic() - t0) * 1000.0

    if resp.status_code != 200:
        return Observation(status="error", latency_ms=latency_ms,
                           reason=f"teacher HTTP {resp.status_code}", uncertain=True)
    try:
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
        metrics = body.get("experimental_metrics", {}) or {}
    except Exception as exc:  # noqa: BLE001
        return Observation(status="error", latency_ms=latency_ms,
                           reason=f"teacher bad response: {exc}", uncertain=True)

    label, confidence = None, None
    try:
        import json as _json
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            label = str(parsed.get("label")) if parsed.get("label") is not None else None
            confidence = parsed.get("confidence")
    except (ValueError, TypeError, AttributeError):
        label = (text or "").strip()[:40] or None

    return Observation(
        status="ok",
        label=label,
        confidence=confidence,  # validated downstream by _validate_against_schema
        uncertain=True,
        model=body.get("model"),
        revision=metrics.get("revision"),
        latency_ms=latency_ms,
        raw={
            "scoring": metrics.get("scoring"),
            "inference_seconds": metrics.get("inference_seconds"),
            "image_count": metrics.get("image_count"),
            "output_is_generated_json": (metrics.get("scoring") or {}).get("output_is_generated_json"),
            "usage": body.get("usage"),
        },
        reason="uncalibrated observation (non-actioning)",
    )


def _call_server(question: str, labels: list[str], images: list[bytes],
                 deadline: float, should_proceed) -> "Observation":
    content: list[dict] = [
        {"type": "text",
         "text": (question[:300] + "\nChoose exactly one: " + ", ".join(labels))},
    ]
    for img in images:
        b64 = base64.b64encode(img).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    payload = {
        "model": config.TEACHER_MODEL,
        "messages": [{"role": "user", "content": content}],
        "labels": labels,  # requested-label schema is actually sent to the provider
        "max_tokens": 96,
    }
    headers = {"Content-Type": "application/json"}
    if config.TEACHER_TOKEN:
        headers["Authorization"] = f"Bearer {config.TEACHER_TOKEN}"
    url = config.TEACHER_BASE_URL.rstrip("/") + "/v1/chat/completions"

    # Final generation guard immediately before any socket write.
    if should_proceed is not None and not should_proceed():
        return Observation(status="stale", reason="cancelled before provider call", uncertain=True)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return Observation(status="timeout", reason="deadline exhausted", uncertain=True)
    fut = _executor.submit(_do_post, url, payload, headers, deadline)
    try:
        obs = fut.result(timeout=remaining)
    except FutureTimeout:
        # The underlying call keeps running (and keeps concurrency) until it ends;
        # we stop waiting and report a bounded timeout.
        return Observation(status="timeout", reason="teacher deadline exceeded", uncertain=True)
    return _validate_against_schema(obs, labels)
