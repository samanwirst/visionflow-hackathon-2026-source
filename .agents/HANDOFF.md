# HANDOFF — M2 exceptional recovery

- **Milestone:** M2 (one authorized exceptional recovery; P1-only).
- **Implementation SHA:** `7096ab3dd3908467c79ba2de75ddd70970995cce`
- **Base / prior review target:** `2127d207a7bf2e1401d40e5ad82d926eb0f66a57`
- **Frozen snapshot where the four P1s reproduced:**
  `a93d983dd89b31e329689b09726c5a1cfbf4dbb5`
- **Branch:** `feature/visionflow`
- **Scope:** ONLY M2-001, M2-003, M2-007, M2-008 and their regressions. No M1
  changes, no M3 work, no P2 cleanup, no new product functionality. M2-009 (P2)
  is explicitly out of scope.

## Implemented fixes

- **M2-001 (NL compiler, `services/api/app/nl_compiler.py`):** replaced the
  heuristic fallthrough with an anchored grammar plus a residual-token backstop.
  New rejections: negated triggers (`_NEG_TRIGGER`, e.g. "does not enter"),
  upper-bound dwell durations (`_UPPER_BOUND_DURATION`, e.g. "less than 10
  minutes"), data-management actions (`_DATA_ACTION`, e.g. "delete the video"),
  and more than one named zone (`_match_zone`). Added `if its/if the/…` condition
  markers. `_residual_tokens` blanks every consumed grammar element and rejects
  any meaningful leftover token. Supported class/zone/temporal and a single
  positive visual condition are preserved unchanged.
- **M2-003 (Teacher transport, `services/api/app/teacher.py`):** a returned
  `timeout` no longer frees the concurrency permit. `_ConcurrencySlot` owns one
  bounded-semaphore permit; on an over-deadline abandon the slot is **deferred**
  and a reaper thread joins the still-active worker, drops the late response, and
  only then releases the permit. The semaphore now caps active transport
  workers/body-reads to `max_concurrency` even across timeouts. Local preflight
  rejection still consumes zero HTTP budget.
- **M2-007 (router admission, `services/api/app/vlm_router.py`):** `_attempt_guard`
  now performs the authoritative `admit()` approval read AND the budget grant as
  one atomic boundary under `dispatch_admission_lock` (the lock the API disable
  commits under). A disable either lands fully before the boundary (→ revoked, no
  request) or fully after it (→ already admitted). The lock is not held during
  inference/transport.
- **M2-008 (strict confidence, `services/api/app/teacher.py`):** `_ProviderResponse`
  gained a `mode="before"` validator requiring an actual JSON number for
  `confidence` — booleans (an `int` subclass), strings, NaN and infinities are
  rejected before the `[0,1]` bounds apply. Valid numeric/int confidence and the
  optional-for-`unknown` path are preserved.

## Changed files

- `services/api/app/nl_compiler.py`
- `services/api/app/teacher.py`
- `services/api/app/vlm_router.py`
- `tests/unit/test_nl_compiler.py` (+8 regressions)
- `tests/unit/test_teacher.py` (+6 M2-008 regressions)
- `tests/unit/test_transport.py` (+1 M2-003 concurrency regression)
- `tests/integration/test_m2_visual.py` (+2: M2-007 interleaving, M2-008 boundary)
- `docs/TRACEABILITY.md`, `.agents/STATE.md`

## Commands executed and actual results

- `services/api/.venv/bin/python -m pytest tests/ -q --ignore=tests/orchestrator`
  → **140 passed, 4 warnings** (was 123; +17 new regressions). The 4 warnings are
  pre-existing (Starlette/FastAPI deprecations + one intentional-crash recovery
  test in `test_m1_fixes.py`).
- `pytest tests/unit/test_nl_compiler.py` → 35 passed.
- `pytest tests/unit/test_teacher.py tests/unit/test_transport.py` → 29 passed.
- `pytest tests/integration/test_m2_visual.py` → 13 passed.
- Concurrency-sensitive tests (M2-003, M2-007 interleaving, M2-008 boundary) run
  three consecutive times → stable, no flakiness observed.

## Known limitations

- Real Teacher visual inference, foveated gains and VLM cost savings remain
  **UNVERIFIED** (no `VLM_BASE_URL`/GPU). Mocked-transport/stub tests do not
  verify real inference.
- M2-003 cancellation is cooperative: the reaper joins the abandoned worker
  (bounded in production by the connection `read_timeout` for a stalled socket).
  A perfectly-paced infinite trickle would hold one permit until its body ends;
  this is documented, not a silent unbounded resource.
- No frontend changes this cycle. No public deployment performed. No credentials,
  hardware, GPU rental, force-push, reset, or main merge.

## Questions for the reviewer

1. M2-001: is the residual-token backstop (reject any unconsumed meaningful
   token) the preferred shape, vs. a fully enumerated template set? The current
   `_FILLER` set is the scaffolding allow-list; please flag any supported phrasing
   it would wrongly reject.
2. M2-007: the admission boundary is atomic under `dispatch_admission_lock`; the
   physical POST trails the grant by a lock-release. Confirm this satisfies
   "once disable returns, no NEW request starts" (admission decision is ordered
   before disable; the socket write is the already-granted admission).
3. M2-003: acceptable that cancellation is a bounded join rather than a hard
   transport cancel, given `httpx.post` is the monkeypatch-able entry point used
   by the existing tests?
