"""Teacher VLM adapter (M2-AC3).

A Teacher classifies a single selected frame or object ROI against a supported
visual predicate (e.g. "the vehicle hood is open"). It returns a *validated*
result: one of the predicate's allowed labels or an explicit ``unknown``, plus
model/version, timing and billable usage where the provider exposes it.

Honesty rules enforced here:

- A timeout, HTTP error, malformed/unparseable response, a label outside the
  declared set, or missing credentials NEVER becomes a positive visual
  condition. Each maps to a non-confirming status (``error``/``timeout``/
  ``invalid_response``/``missing_credentials``) that the router and pipeline
  treat as "not verified".
- The real provider is selected by configuration. With ``TEACHER_PROVIDER=none``
  (the default in this repo, no key configured) the adapter is
  ``UnavailableTeacher`` and every call returns ``missing_credentials``. This is
  a documented blocker, not a fabricated success. Stub adapters used in tests
  are clearly test-only and do not verify real visual inference.

No generated code is executed and there is no per-frame loop: the adapter only
runs when the router admits a single candidate ROI.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Literal, Sequence

import threading

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import config

logger = logging.getLogger("visionflow.teacher")

TeacherStatus = Literal[
    "ok",
    "unknown",
    "low_confidence",
    "error",
    "timeout",
    "invalid_response",
    "missing_credentials",
]


class TeacherResult(BaseModel):
    """Validated Teacher response. ``status == 'ok'`` is the only state that may
    ever confirm a visual action; ``label`` is then guaranteed to be one of the
    predicate's declared labels."""

    status: TeacherStatus
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = None
    prompt_version: str = config.TEACHER_PROMPT_VERSION
    latency_ms: float = 0.0
    usage: dict = Field(default_factory=dict)  # provider-billable tokens if exposed
    error: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.status == "ok" and self.label is not None


class _ProviderResponse(BaseModel):
    """Strict schema the provider's JSON content must satisfy. Unexpected fields
    are rejected and confidence must be finite and bounded — the model cannot
    smuggle a free-form or out-of-range answer into a confirmed action.

    ``confidence`` is optional *here* because an explicit ``unknown`` answer may
    omit it; a *classified* label, however, must carry a confidence or it is
    downgraded to ``invalid_response`` in :meth:`OpenAICompatibleTeacher._parse`
    (M2-AC3).

    ``confidence`` must be an *actual* JSON number (M2-008). Pydantic's default
    lax coercion would turn ``true`` into ``1.0`` and ``"0.9"`` into ``0.9``,
    letting a malformed provider answer authorize an action. A ``mode="before"``
    validator rejects booleans (``True``/``False`` are ``int`` subclasses in
    Python), strings, NaN and infinities *before* the numeric bounds are applied,
    so only a finite JSON number in ``[0, 1]`` can ever confirm."""

    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("confidence", mode="before")
    @classmethod
    def _require_json_number(cls, v):
        if v is None:
            return v
        # bool is a subclass of int: reject it explicitly so ``true`` cannot be
        # coerced to 1.0. Reject strings (incl. numeric strings) and any other
        # non-numeric type, and reject non-finite numbers.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("confidence must be a JSON number (not a boolean or string)")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("confidence must be a finite number")
        return f


# A router-supplied guard invoked immediately before every outbound HTTP attempt
# (M2-AC4). It consumes/accounts the shared per-task request budget at the
# transport boundary and returns False when no budget remains, so a retry that
# would exceed the budget is never sent.
AttemptGuard = Callable[[], bool]


class _ConcurrencySlot:
    """A single-owner handle on one bounded-semaphore permit (M2-003).

    The permit is released exactly once. When an outbound request is abandoned
    at the wall-clock deadline while its transport worker is still active, the
    slot is *deferred*: ``classify`` must NOT release it, and a reaper releases
    it only once the worker has truly finished. This keeps the number of active
    transport workers/body-reads bounded by ``max_concurrency`` even across
    timeouts — a returned ``timeout`` never frees capacity while its request is
    still in flight."""

    def __init__(self, semaphore: "threading.BoundedSemaphore") -> None:
        self._semaphore = semaphore
        self._lock = threading.Lock()
        self._released = False
        self._deferred = False

    def release(self) -> None:
        with self._lock:
            if not self._released:
                self._released = True
                self._semaphore.release()

    def defer(self) -> None:
        with self._lock:
            self._deferred = True

    @property
    def deferred(self) -> bool:
        with self._lock:
            return self._deferred


class TeacherAdapter(ABC):
    """Classifies one image ROI against a visual predicate."""

    provider_name: str = "base"
    # When True, the adapter accounts every real outbound attempt through the
    # router-supplied ``on_attempt`` guard, so the router must NOT add its own
    # fallback attempt/budget unit (M2-004). Stubs that do not call the guard
    # leave this False and are charged one fallback unit so the budget still
    # bounds them.
    reports_attempts: bool = False

    @abstractmethod
    def classify(
        self,
        image_bytes: bytes,
        prompt: str,
        labels: Sequence[str],
        on_attempt: AttemptGuard | None = None,
    ) -> TeacherResult:
        """Return a validated TeacherResult for one ROI. Never raises for
        provider/transport failures — those become non-confirming statuses.

        ``on_attempt``, when supplied by the router, is called once before each
        outbound request; if it returns False the adapter must not send that
        request (budget exhausted)."""

    @property
    def available(self) -> bool:
        return True


class UnavailableTeacher(TeacherAdapter):
    """Default when no provider credentials are configured. Always returns
    ``missing_credentials`` so a visual predicate can never be confirmed and the
    limitation is explicit in events and counters."""

    provider_name = "none"

    def __init__(self, reason: str = "no Teacher provider configured (TEACHER_PROVIDER=none)"):
        self._reason = reason

    @property
    def available(self) -> bool:
        return False

    def classify(
        self,
        image_bytes: bytes,
        prompt: str,
        labels: Sequence[str],
        on_attempt: AttemptGuard | None = None,
    ) -> TeacherResult:
        # Makes no outbound request, so ``on_attempt`` is never invoked and no
        # budget/attempt is consumed: an unavailable provider contributes zero
        # actual requests (M2-AC4).
        return TeacherResult(
            status="missing_credentials",
            model=None,
            latency_ms=0.0,
            error=self._reason,
        )


class OpenAICompatibleTeacher(TeacherAdapter):
    """Calls an OpenAI-style chat/completions vision endpoint.

    The request forces a strict JSON answer constrained to the declared labels;
    the response content is parsed and validated. Any deviation is downgraded to
    a non-confirming status. Requires a real base URL + API key via env.
    """

    provider_name = "openai_compatible"
    # The real adapter calls ``on_attempt`` before every outbound request, so the
    # router accounts attempts/budget exclusively through that guard (M2-004).
    reports_attempts = True

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        prompt_version: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 20.0,
        overall_timeout: float = 25.0,
        max_retries: int = 1,
        max_concurrency: int = 1,
        max_image_bytes: int = 4 * 1024 * 1024,
        max_tokens: int = 200,
        min_confidence: float = 0.5,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.prompt_version = prompt_version
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.overall_timeout = overall_timeout
        self.max_retries = max(0, max_retries)
        self.max_image_bytes = max_image_bytes
        self.max_tokens = max_tokens
        self.min_confidence = min_confidence
        # Bounded concurrency so a burst of candidates cannot overwhelm the
        # self-hosted server (AC3a semaphore/queue limit).
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrency))

    def _build_messages(self, image_bytes: bytes, prompt: str, labels: Sequence[str]) -> list:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        label_list = ", ".join(f'"{l}"' for l in labels)
        instruction = (
            "You are a strict visual classifier for an automotive service "
            "operations system. Examine ONLY the provided image region. "
            f"Question: {prompt}\n"
            f"Answer with a single JSON object: "
            f'{{"label": one of [{label_list}, "unknown"], '
            '"confidence": a number 0..1}. '
            "Use \"unknown\" if the image does not clearly show the answer. "
            "Do not output any text outside the JSON object."
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ]

    def classify(
        self,
        image_bytes: bytes,
        prompt: str,
        labels: Sequence[str],
        on_attempt: AttemptGuard | None = None,
    ) -> TeacherResult:
        import httpx

        if len(image_bytes) > self.max_image_bytes:
            return TeacherResult(
                status="error", model=self.model, prompt_version=self.prompt_version,
                error=f"ROI {len(image_bytes)}B exceeds max_image_bytes {self.max_image_bytes}",
            )
        payload = {
            "model": self.model,
            "messages": self._build_messages(image_bytes, prompt, labels),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:  # optional: omitted for a private authenticated tunnel
            headers["Authorization"] = f"Bearer {self.api_key}"

        # A single monotonic deadline bounds EVERYTHING: queue (semaphore)
        # acquisition, each HTTP attempt, and every backoff (M2-003). No phase
        # may run past ``deadline``; when the remaining budget is exhausted the
        # call returns a ``timeout`` result instead of overrunning.
        t0 = time.perf_counter()
        deadline = t0 + self.overall_timeout

        def _elapsed_ms() -> float:
            return (time.perf_counter() - t0) * 1000.0

        def _remaining() -> float:
            return deadline - time.perf_counter()

        # Bound semaphore acquisition by the remaining deadline. A slot that is
        # still held by an abandoned-but-active worker (M2-003) keeps capacity
        # occupied, so a burst while one request is in flight correctly fails to
        # acquire here and returns ``timeout`` without starting a second worker.
        acquire_budget = _remaining()
        if acquire_budget <= 0 or not self._semaphore.acquire(timeout=acquire_budget):
            return TeacherResult(
                status="timeout", model=self.model, prompt_version=self.prompt_version,
                latency_ms=_elapsed_ms(),
                error="overall deadline exceeded before acquiring Teacher concurrency slot",
            )
        slot = _ConcurrencySlot(self._semaphore)

        last: TeacherResult | None = None
        # Bounded attempts: each outbound attempt consults the router budget
        # guard first, and retries only cover transient failures
        # (timeout / connection / 5xx). Auth (401/403) and schema failures fail
        # visibly without retrying.
        try:
            for attempt in range(self.max_retries + 1):
                remaining = _remaining()
                if remaining <= 0:
                    last = last or TeacherResult(
                        status="timeout", model=self.model,
                        prompt_version=self.prompt_version, latency_ms=_elapsed_ms(),
                        error=f"overall deadline exceeded before attempt {attempt + 1}",
                    )
                    break
                # Budget guard at the transport boundary: a retry that would
                # exceed the shared per-task budget is never sent (M2-AC4).
                if on_attempt is not None and not on_attempt():
                    last = last or TeacherResult(
                        status="error", model=self.model,
                        prompt_version=self.prompt_version, latency_ms=_elapsed_ms(),
                        error=f"request budget exhausted before attempt {attempt + 1}",
                    )
                    break
                # Clamp each phase timeout to the remaining deadline.
                attempt_timeout = httpx.Timeout(
                    min(self.overall_timeout, remaining),
                    connect=min(self.connect_timeout, remaining),
                    read=min(self.read_timeout, remaining),
                )
                try:
                    resp = self._post_with_deadline(
                        f"{self.api_base}/chat/completions",
                        payload, headers, attempt_timeout, deadline, slot,
                    )
                except httpx.TimeoutException as exc:
                    last = TeacherResult(
                        status="timeout", model=self.model,
                        prompt_version=self.prompt_version, latency_ms=_elapsed_ms(),
                        error=f"provider timeout (attempt {attempt + 1}): {exc}"[:500],
                    )
                except httpx.HTTPError as exc:
                    last = TeacherResult(
                        status="error", model=self.model,
                        prompt_version=self.prompt_version, latency_ms=_elapsed_ms(),
                        error=f"transport error (attempt {attempt + 1}): {exc}"[:500],
                    )
                else:
                    latency_ms = _elapsed_ms()
                    if resp.status_code in (401, 403):
                        return TeacherResult(
                            status="error", model=self.model,
                            prompt_version=self.prompt_version, latency_ms=latency_ms,
                            error=f"provider auth failed HTTP {resp.status_code}",
                        )
                    if resp.status_code >= 500:
                        last = TeacherResult(
                            status="error", model=self.model,
                            prompt_version=self.prompt_version, latency_ms=latency_ms,
                            error=f"provider HTTP {resp.status_code}: {resp.text[:200]}",
                        )
                    elif resp.status_code >= 400:
                        return TeacherResult(
                            status="error", model=self.model,
                            prompt_version=self.prompt_version, latency_ms=latency_ms,
                            error=f"provider HTTP {resp.status_code}: {resp.text[:200]}",
                        )
                    else:
                        return self._parse(resp, labels, latency_ms)
                # Backoff only if another attempt remains AND it fits the deadline.
                if attempt < self.max_retries:
                    remaining = _remaining()
                    if remaining <= 0:
                        break
                    backoff = min(2.0, 0.5 * (2 ** attempt), remaining)
                    if backoff > 0:
                        time.sleep(backoff)
        finally:
            # Release the concurrency slot — UNLESS an abandoned worker is still
            # active (M2-003): its reaper owns the release so capacity is not
            # freed while a transport request is still in flight.
            if not slot.deferred:
                slot.release()
        return last or TeacherResult(
            status="error", model=self.model, prompt_version=self.prompt_version,
            error="no response",
        )

    def _post_with_deadline(self, url, payload, headers, attempt_timeout, deadline, slot):
        """Execute one POST bounded by the overall wall-clock ``deadline`` (M2-003).

        httpx's phase timeouts reset per read, so a provider that trickles small
        chunks within each read timeout can run far past the overall deadline. We
        run the (blocking) request — connection, transfer and full-body read — in
        a daemon worker and wait only until the remaining wall-clock budget. If it
        has not finished by the deadline the request is abandoned and a late
        response is rejected (raised as a timeout), never consumed.

        Crucially, when the worker is abandoned it is STILL ACTIVE. We must not
        free the concurrency slot yet, or a burst of callers would each start a
        fresh worker while the previous ones keep reading bodies — exceeding the
        configured ``max_concurrency`` (the reviewed defect). Instead we ``defer``
        the slot and hand its release to a small reaper that joins the worker (so
        the body read finishes or its connection/read-timeout fires), drops the
        response to bound retention, and only then releases the permit. The
        bounded semaphore therefore caps the number of outstanding transport
        workers and in-flight body reads even across timeouts. Retries/backoff in
        :meth:`classify` are preserved; a deadline hit leaves no remaining budget,
        so no retry starts another worker behind an abandoned one."""
        import httpx

        box: dict = {}

        def _call() -> None:
            try:
                box["resp"] = httpx.post(
                    url, json=payload, headers=headers, timeout=attempt_timeout
                )
            except BaseException as exc:  # noqa: BLE001 — surfaced to the caller
                box["exc"] = exc

        worker = threading.Thread(target=_call, daemon=True)
        worker.start()
        remaining = deadline - time.perf_counter()
        worker.join(timeout=max(0.0, remaining))
        if worker.is_alive():
            # Deadline hit while the request/body was still in flight. Keep the
            # concurrency slot occupied until the worker truly finishes: a reaper
            # joins it, discards any late response (never consumed), and releases
            # the permit. This bounds active workers/body-reads to max_concurrency.
            slot.defer()

            def _reap() -> None:
                try:
                    worker.join()
                finally:
                    resp = box.get("resp")
                    close = getattr(resp, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:  # noqa: BLE001
                            pass
                    slot.release()

            threading.Thread(target=_reap, daemon=True).start()
            raise httpx.TimeoutException(
                "overall wall-clock deadline exceeded during request/response"
            )
        if "exc" in box:
            raise box["exc"]
        return box["resp"]

    def _parse(self, resp, labels: Sequence[str], latency_ms: float) -> TeacherResult:
        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {}) or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            return TeacherResult(
                status="invalid_response", model=self.model,
                prompt_version=self.prompt_version, latency_ms=latency_ms,
                error=f"unparseable provider envelope: {exc}"[:500],
            )

        parsed = _extract_json_object(content)
        if parsed is None:
            return TeacherResult(
                status="invalid_response", model=self.model,
                prompt_version=self.prompt_version, latency_ms=latency_ms,
                usage=_billable(usage),
                error=f"no JSON object in content: {content[:200]!r}",
            )
        try:
            validated = _ProviderResponse.model_validate(parsed)
        except ValidationError as exc:
            return TeacherResult(
                status="invalid_response", model=self.model,
                prompt_version=self.prompt_version, latency_ms=latency_ms,
                usage=_billable(usage),
                error=f"schema violation: {exc}"[:500],
            )

        allowed = {l.lower() for l in labels}
        label = validated.label.strip().lower()
        # ``unknown`` is an explicit, separate non-confirming outcome; confidence
        # is optional for it.
        if label == "unknown":
            return TeacherResult(
                status="unknown", label=None, confidence=validated.confidence,
                model=self.model, prompt_version=self.prompt_version,
                latency_ms=latency_ms, usage=_billable(usage),
            )
        if label not in allowed:
            return TeacherResult(
                status="invalid_response", model=self.model,
                prompt_version=self.prompt_version, latency_ms=latency_ms,
                usage=_billable(usage),
                error=f"label {label!r} not in declared set {sorted(allowed)}",
            )
        # A *classified* label must carry a confidence — a positive answer with
        # no confidence cannot confirm an action (M2-AC3).
        if validated.confidence is None:
            return TeacherResult(
                status="invalid_response", model=self.model,
                prompt_version=self.prompt_version, latency_ms=latency_ms,
                usage=_billable(usage),
                error=f"classified label {label!r} missing required confidence",
            )
        # Below the configured minimum the classification stands but is
        # non-confirming (distinct from ``unknown``): the label/confidence stay
        # visible for provenance, yet ``confirmed`` is False.
        if validated.confidence < self.min_confidence:
            return TeacherResult(
                status="low_confidence", label=label, confidence=validated.confidence,
                model=self.model, prompt_version=self.prompt_version,
                latency_ms=latency_ms, usage=_billable(usage),
                error=(
                    f"confidence {validated.confidence:.3f} below minimum "
                    f"{self.min_confidence:.3f}"
                ),
            )
        return TeacherResult(
            status="ok", label=label, confidence=validated.confidence,
            model=self.model, prompt_version=self.prompt_version,
            latency_ms=latency_ms, usage=_billable(usage),
        )


def _extract_json_object(content: str) -> dict | None:
    """Parse the first JSON object from a model's text content."""
    if not isinstance(content, str):
        return None
    content = content.strip()
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _billable(usage: dict) -> dict:
    """Keep only billable token counts from a provider usage block."""
    out = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if k in usage:
            out[k] = usage[k]
    return out


# Factory ----------------------------------------------------------------------

# Tests inject a deterministic stub here (clearly test-only). Production builds
# the configured provider.
teacher_factory = None
_teacher: TeacherAdapter | None = None


def get_teacher() -> TeacherAdapter:
    global _teacher
    if teacher_factory is not None:
        return teacher_factory()
    if _teacher is not None:
        return _teacher
    # Enabled by presence of the operator-set base URL + model (key optional for
    # a private authenticated tunnel). Missing config -> unavailable blocker.
    if config.VLM_BASE_URL and config.VLM_MODEL:
        _teacher = OpenAICompatibleTeacher(
            api_base=config.VLM_BASE_URL,
            api_key=config.VLM_API_KEY,
            model=config.VLM_MODEL,
            prompt_version=config.TEACHER_PROMPT_VERSION,
            connect_timeout=config.VLM_CONNECT_TIMEOUT,
            read_timeout=config.VLM_READ_TIMEOUT,
            overall_timeout=config.VLM_TIMEOUT_SECONDS,
            max_retries=config.VLM_MAX_RETRIES,
            max_concurrency=config.VLM_MAX_CONCURRENCY,
            max_image_bytes=config.VLM_MAX_IMAGE_BYTES,
            max_tokens=config.VLM_MAX_TOKENS,
            min_confidence=config.VLM_MIN_CONFIDENCE,
        )
    elif config.VLM_BASE_URL and not config.VLM_MODEL:
        _teacher = UnavailableTeacher("VLM_BASE_URL set but VLM_MODEL is unset")
    else:
        _teacher = UnavailableTeacher()
    logger.info("Teacher adapter: %s (available=%s)", _teacher.provider_name, _teacher.available)
    return _teacher


def reset_teacher() -> None:
    global _teacher
    _teacher = None
