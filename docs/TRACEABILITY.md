# Матрица трассируемости / Traceability

**Official brief status:** VERIFIED — the official Final Task text is present at
`docs/source/README.md` (the PDF `FINAL_TASK_WIUT_HACKATHON.pdf` itself is not in
the repo; the text version is treated as the source of truth). No conflicts with
`docs/PRD.md` found for M1 scope.

Statuses: planned | in_progress | implemented | verified | blocked.
M1 is **accepted with documented P2 risks** after the exceptional review of
`390b337`; the four P1 defects in the previous `cc59523` review were resolved.
Accepted report: `.agents/reviews/M1.md`, pushed on `review/M1-r3`.

## M1 — vision-to-action pipeline

| ID | Requirement | Source | Component / file | Check | Status |
| --- | --- | --- | --- | --- | --- |
| R1 | Prototype runs end to end on new input | Brief "prototype runs end to end, live" | whole pipeline | `scripts/smoke_e2e.py` (real YOLO) — PASS 2026-10-10 | verified |
| R2 | MP4 upload with validation | PRD P0.1 | `services/api/app/main.py` upload_source + `app/pipeline.py` decode-completeness guard | Happy-path/type/size tests pass; truncated decode now surfaces failed source (M1-013) | verified |
| R3 | Real object detection | PRD P0.3; brief "no faked demos" | `packages/vision/visionflow_vision/detector.py` (YOLO11n) | smoke test: bus detected conf 0.76 | verified |
| R4 | Object tracking | PRD P0.3 | ByteTrack via `model.track(persist=True)` | smoke test: stable track id in events | verified |
| R5 | Polygonal zones | PRD P0.4 | `visionflow_vision/zones.py`; UI canvas drawing | `tests/unit/test_zones.py`; browser flow 2026-10-10 | verified |
| R6 | Temporal events (enter/exit/dwell/count) | PRD P0.5 | `visionflow_vision/events.py` | `tests/unit/test_events.py` (debounce, cooldown, track loss, one-shot) | verified |
| R7 | Validated workflow DSL | PRD §8 | `services/api/app/schemas.py` | `tests/unit/test_dsl.py` (unsupported trigger/action/class rejected) | verified |
| R8 | Explicit enable before workflow is armed | PRD P0.7 | workflows create disabled; `/enable` endpoint | integration test asserts enabled=false on create | verified |
| R9 | Real authorized action | PRD P0.8 | `services/api/app/actions.py` (local `create_notification`; outbound webhook removed, M1-011/M1-012) | Action tests pass; disable-before-dispatch suppressed (M1-010); legacy webhook rejected with no network | verified |
| R10 | Event persistence with evidence | PRD P0.9 | `TrackEvent` + evidence JPEG in `app/pipeline.py` | smoke + integration: evidence URL serves image/jpeg | verified |
| R11 | Working web interface | PRD P0.10 | `apps/web/app/page.tsx` | manual browser run 2026-10-10: full flow exercised | verified |
| R12 | Automated tests | PRD P0.11 | `tests/` | `pytest tests/` — 66 passed 2026-10-10 (recovery cycle) | verified |
| R13 | NL → workflow compilation | PRD P0.6 | — | planned for M2 | planned |
| R14 | Live camera source | PRD P0.2 | — | planned (M2+, hardware-dependent) | planned |
| R15 | Public deployment | Brief "publicly available" | — | planned for M4 | planned |
| R16 | AutoLearn | PRD P1 | `packages/autolearn` | planned for M3 | planned |

## M1 review fixes (cycle 1, 2026-10-10)

| ID | Severity | Resolution | File | Test |
| --- | --- | --- | --- | --- |
| M1-001 | P1 | FIXED | `services/api/app/actions.py` — URL canonicalized; userinfo rejected; scheme/host/port exact match; path-prefix enforced; internal-resolution guard | `tests/unit/test_webhook_auth.py` (userinfo never reaches MockTransport; allowlisted delivered) |
| M1-002 | P1 | FIXED | `packages/vision/visionflow_vision/events.py` — aggregate episode counter advances per rising crossing, included in dedupe key | `tests/unit/test_events.py::test_count_rearms_with_distinct_dedupe_keys` |
| M1-003 | P1 | FIXED | `services/api/app/pipeline.py` — authoritative `enabled` re-checked per frame via fresh session before emitting/acting | `tests/integration/test_m1_fixes.py::test_disable_during_processing_revokes_actions` |
| M1-004 | P1 | FIXED | `services/api/app/pipeline.py` — complete session serialized by `_session_lock` around reset + all frame calls | `tests/integration/test_m1_fixes.py::test_concurrent_sources_are_serialized` |
| M1-005 | P1 | FIXED | `services/api/app/{pipeline,main}.py` — startup marks interrupted `processing` rows failed and permits retry | `tests/integration/test_m1_fixes.py::test_interrupted_processing_recovered_and_retryable` |

## M1 review fixes (cycle 2, 2026-10-10)

| ID | Severity | Resolution | File | Test |
| --- | --- | --- | --- | --- |
| M1-006 | P1 | FIXED | `services/api/app/actions.py` — authorize and deliver one canonical URL; `'.'/'..'` path segments resolved and prefix enforced after normalization; percent/backslash-encoded traversal rejected | `tests/unit/test_webhook_auth.py` (`test_dotdot_traversal_rejected`, `test_encoded_traversal_rejected`, `test_traversal_never_reaches_admin_via_transport`, `test_authorized_path_is_exactly_what_is_delivered`) |
| M1-003 | P1 | FIXED (re-hardened) | `services/api/app/pipeline.py` — authoritative `enabled` re-checked before EACH event, not once per frame, so a disable between two events in one frame suppresses the remainder | `tests/integration/test_m1_fixes.py::test_disable_between_events_in_one_frame_suppresses_remainder` |
| M1-007 | P1 | FIXED | `services/api/app/pipeline.py` — a committed event with no dispatched action (pending + zero ActionExecution) is resumed on reprocess/recovery; completed events stay deduped | `tests/integration/test_m1_fixes.py::test_interrupted_before_dispatch_resumes_on_retry` |
| M1-008 | P1 | FIXED | `services/api/app/schemas.py` (bounded-string title → 422) + `services/api/app/pipeline.py` (`_record_source_failure` rolls back the aborted transaction, fresh-session fallback) | `tests/integration/test_m1_fixes.py::test_invalid_notification_title_rejected` |
| M1-009 | P2 | FIXED | `services/api/app/pipeline.py` — `_save_evidence` checks `cv2.imwrite` return and raises; event work is rolled back and the source recorded failed | `tests/integration/test_m1_fixes.py::test_evidence_write_failure_fails_source_and_rolls_back` |

## M1 review fixes (exceptional recovery cycle, 2026-10-10)

Authorized single focused M1 recovery pass (base `b131c50`). Only M1-010..013.

| ID | Severity | Resolution | File | Test |
| --- | --- | --- | --- | --- |
| M1-010 | P1 | FIXED | `services/api/app/{pipeline,main}.py` — `dispatch_admission_lock` serializes API disable commit against the worker's authoritative enabled recheck + dispatch; a disable completed before dispatch suppresses the queued action explicitly (`cancelled` execution + event `action_status`) | `tests/integration/test_m1_fixes.py::test_disable_before_dispatch_suppresses_queued_action` |
| M1-011 | P1 | FIXED (by removal) | `services/api/app/actions.py` — outbound webhook delivery removed, eliminating the interrupted/duplicate-external-delivery ambiguity; the retained local action commits atomically with its event status so recovery re-dispatches exactly once | `tests/integration/test_m1_fixes.py::{test_legacy_webhook_action_rejected_no_side_effect,test_interrupted_before_dispatch_resumes_on_retry}` |
| M1-012 | P1 | FIXED (by removal) | `services/api/app/{actions,schemas,config}.py` — `send_webhook` removed from `ActionType`; no DNS lookup or delivery can occur; legacy rows rejected at dispatch; preferred over a risky custom DNS/TLS egress transport | `tests/unit/test_webhook_auth.py`, `tests/unit/test_dsl.py::test_webhook_action_now_rejected`, `tests/integration/test_api_flow.py::test_webhook_action_rejected_at_creation` |
| M1-013 | P1 | FIXED | `services/api/app/{pipeline,config}.py` — decoded-frame count compared to reliable container frame count with `DECODE_MIN_COMPLETENESS` tolerance; substantial shortfall surfaces a failed source instead of silent `processed` | `tests/integration/test_m1_fixes.py::test_truncated_video_reported_failed` |

## Discrepancies / notes

- Cycle-1/2 fix entries above record the developer's changes and passing
  regressions at the time. Final independent review (`cc59523`) found related
  remaining P1 defects M1-010 through M1-013; those are resolved in the
  exceptional recovery cycle above. M1-001/M1-006 (webhook SSRF/traversal)
  became moot once outbound webhooks were removed; those tests were replaced.
- M1-014 (P2, CI omits httpx/SQLAlchemy) is out of scope for this focused pass.
  Removing httpx from `actions.py` reduces but does not formally close it.

- The PDF named in CLAUDE.md is absent; the official task exists as text in
  `docs/source/README.md`. Treated as the official brief (per operator message).
- Demo video provenance: built by `scripts/make_demo_video.py` from the
  Ultralytics sample photo (real photographic content, synthetic motion).
  Declared in `docs/LICENSES.md`; detection on it is genuine YOLO inference.

## Cost-efficient visual analytics — operator requirement (2026-10-10)

These gates extend the PRD; they are not attributed to the official brief.
Full audit: `docs/VISUAL_ANALYTICS_AUDIT.md`. Planned is not implemented.

| ID | Acceptance requirement | Component / evidence required | Status |
| --- | --- | --- | --- |
| M2-AC1 | Strict NL DSL, rejection, manual approval | `app/nl_compiler.py`, `/api/workflows/compile`, UI compile box; `tests/unit/test_nl_compiler.py`, `test_m2_visual.py` | implemented |
| M2-AC2 | Cheap temporal candidate events select Teacher frame/ROI; zero calls for geometric rules | `app/pipeline._route_visual`, `app/vlm_router.py`; `test_geometric_workflow_makes_zero_vlm_calls` | implemented |
| M2-AC3 | Real validated Teacher adapter; visible unknown/error/budget states | `app/teacher.py` strict validation; `tests/unit/test_teacher.py`; not-confirmed integration tests | implemented (real provider **blocked**: no VLM_BASE_URL configured) |
| M2-AC4 | Sampling, budgets, in-flight dedupe, bounded versioned cache | `app/vlm_router.py` (`VersionedTTLCache`, budget/cooldown/sampling/dedup); `tests/unit/test_vlm_router.py` | implemented |
| M2-AC5 | Actual routing counters, evidence, safe approval/action/data boundaries | `VlmCall` rows, `/api/vlm/stats`, `/api/vlm/calls`, `/api/metrics.vlm`, event `visual_*`; UI section 6 | implemented |
| M3-AC1 | Real Teacher labels, human verification, dataset provenance | Reviewed sample manifest with both classes | planned |
| M3-AC2 | Session-disjoint splits and no near-duplicate leakage | Split manifest and leakage checks | planned |
| M3-AC3 | Frozen pretrained embeddings and actually trained saved Student | Reproducible train/save/reload artifacts | planned |
| M3-AC4 | Computed held-out quality/latency/coverage | Raw predictions, ground truth, sample counts | planned |
| M3-AC5 | Action-free shadow mode, quality gate, manual activation | Integration checks including rejected activation | planned |
| M3-AC6 | Budgeted Teacher fallback, working rollback and durable history | Low-confidence/restart/rollback checks | planned |
| M4-AC1 | Paired baseline VLM / routed Teacher / Student runs | Same held-out inputs/task and real provider calls | planned |
| M4-AC2 | Raw usage, timing, predictions, provenance and comparable quality | Reproducible benchmark artifacts | planned |
| M4-AC3 | Cost derived from actual usage; honest cold/warm and savings claims | Dated rates/bill, denominators, no invented percentages | planned |
| M4-AC4 | Clean deployment, unseen-input smoke, official deliverables, privacy/licenses | Startup smoke, reports and pitch; public deploy needs approval | planned |

## Self-hosting and foveated experiment — operator update (2026-10-10)

PRD version 1.2 and `docs/VLM_ARCHITECTURE.md` define these gates. No external
provider API key is required at this stage. M1 is accepted; current M2 work
continues without interruption or additional calls for this experiment.

| ID | Requirement | Evidence required | Status |
| --- | --- | --- | --- |
| M2-AC3a | Configurable OpenAI-compatible self-hosted Teacher, VLM env vars, strict labels, bounded timeout/retries/concurrency, secure setup | `app/teacher.OpenAICompatibleTeacher` (VLM_BASE_URL/MODEL/API_KEY, connect/read/overall timeouts, bounded retries skipping auth/schema, concurrency semaphore, max image bytes, extra=forbid, data-URL images); `tests/unit/test_teacher.py` | implemented (adapter + boundaries); real vLLM/GPU smoke **UNVERIFIED** (no instance connected) |
| M2-EXP1 | Full frame, ROI, foveated context-plus-ROI interface; full-frame fallback and strategy-aware cache | Input provenance/selection tests; visibly unavailable unfinished strategies | partial — ROI crop (default) with full-frame fallback for track-less/aggregate candidates and roi_hash in cache key; `foveated`/`full_frame` strategy selector NOT implemented (planned) |
| M3-AC7 | Real GPU Teacher labels followed by verified-data Student evaluation | Actual labels/model identity and held-out predictions; currently UNVERIFIED | planned |
| M3-EXP1 | Compare input strategies on the same held-out videos/events | Accuracy/recall, tokens or actual cost, latency, failure cases; unrun results UNVERIFIED | planned |
| M4-AC3a | Actual Vast.ai rental accounting and GPU/latency/throughput/cost per event | Rental records, charged interval, GPU telemetry and paired input manifest; currently UNVERIFIED | planned |

The CVPR 2025 paper is research motivation, not VisionFlow measurement.
ROI-only crops discard scene context and are not labeled foveated. Pixel
reduction is never assumed to scale linearly with tokens or monetary cost.

## M2 review fixes (2026-10-10)

Addresses the five reproduced P1 defects from the frozen M2 review target.

| ID | Defect | Fix | Evidence |
| --- | --- | --- | --- |
| M2-001 | NL compiler accepted unconsumed condition/action clauses and multiple classes; a visual condition was silently dropped | `app/nl_compiler.py`: reject >1 object class, reject unsupported actions (email/SMS/phone/code execution), reject unrecognised condition clauses, and preserve a whitelist of supported visual conditions (hood/trunk/door open, lights on) | `tests/unit/test_nl_compiler.py::test_with_condition_preserves_visual_predicate`, `test_unsupported_visual_condition_rejected`, `test_email_action_rejected`, `test_code_execution_rejected`, `test_multiple_classes_rejected` |
| M2-002 | Positive Teacher response could confirm with missing or zero confidence | `app/teacher.py`: classified label requires confidence (else `invalid_response`); below `VLM_MIN_CONFIDENCE` → new non-confirming `low_confidence` status (distinct from `unknown`) | `tests/unit/test_teacher.py::test_missing_confidence_on_positive_rejected`, `test_zero_confidence_not_confirmed`, `test_below_minimum_confidence_not_confirmed`, `test_at_minimum_confidence_confirms` |
| M2-003 | Retry/backoff and semaphore acquisition ignored the overall timeout | `app/teacher.py`: one monotonic deadline bounds semaphore acquisition, each attempt's phase timeouts, and every backoff | `tests/unit/test_teacher.py::test_backoff_bounded_by_deadline`, `test_semaphore_acquire_bounded_by_deadline` |
| M2-004 | Budget counted adapter invocations not outbound attempts; unavailable provider incremented calls | `app/vlm_router.py`: per-attempt budget guard at the transport boundary; new counters `adapter_invocations`/`outbound_attempts`/`completed`/`unavailable`; unavailable provider makes zero requests | `tests/unit/test_vlm_router.py::test_budget_consumes_every_outbound_attempt`, `test_budget_allows_retry_within_budget`, `test_unavailable_contributes_zero_requests` |
| M2-005 | Cooldown keyed on the unique per-event dedupe_key never suppressed; concurrent duplicate hit the sampling gate instead of sharing | `app/pipeline.py` cooldown scope = stable `(source, workflow)`; `app/vlm_router.py` in-flight dedup moved ahead of the budget/sampling/cooldown gates | `tests/unit/test_vlm_router.py::test_inflight_duplicate_shares_before_sampling_gate`; cooldown scope exercised via existing `test_cooldown_suppresses_same_scope` |

## M2 review fixes — cycle 2 (2026-10-10)

Addresses the five re-opened/new P1 defects from the frozen review target
`2127d207a7bf2e1401d40e5ad82d926eb0f66a57`. Full suite: **123 passed**
(`services/api/.venv/bin/python -m pytest tests/ -q --ignore=tests/orchestrator`),
+18 tests vs. the prior pass; no frontend changes this cycle.

| ID | Defect | Fix | Evidence |
| --- | --- | --- | --- |
| M2-001 | Negation became a positive predicate; unsupported actions (unlock gate) silently became notifications; alternative triggers / extra conditions dropped; "at least 3" compiled to strictly-`>3` | `app/nl_compiler.py`: reject negated/closed-state visual conditions and >1 visual condition; reject physical/control & external-dispatch actions; reject >1 trigger family (enter/exit/dwell/count); reject unsupported appearance attributes (colour/tint/…); convert inclusive counts (`at least N` → `>N-1`, reject upper-bound & too-low) | `tests/unit/test_nl_compiler.py::test_negated_visual_condition_rejected`, `test_closed_state_condition_rejected`, `test_unsupported_physical_action_rejected`, `test_open_the_gate_action_rejected`, `test_alternative_triggers_rejected`, `test_inclusive_count_converted_to_exclusive_threshold`, `test_upper_bound_count_rejected`, `test_inclusive_count_too_low_rejected`, `test_unsupported_colour_attribute_rejected`, `test_plain_more_than_count_still_exclusive` |
| M2-003 | Per-read HTTP timeouts reset on every chunk, so a compliant trickle ran past the overall deadline and returned `confirmed=true` | `app/teacher.py` `_post_with_deadline`: the blocking request + full-body read runs in a daemon worker bounded by the remaining wall-clock deadline; a late response is abandoned and rejected as `timeout`. Bounded queue, retries and backoff preserved | `tests/unit/test_transport.py::test_phase_compliant_trickle_exceeds_overall_deadline` (genuine httpx + MockTransport streaming), `test_fast_streamed_response_still_confirms` |
| M2-004 | Real-adapter local preflight rejection (oversize image) still counted `outbound_attempts`/`completed` and consumed budget, suppressing the next admissible image | `app/teacher.py` `reports_attempts=True` on the real adapter; `app/vlm_router.py` accounts real adapters exclusively via the transport guard (`granted>0`), removing the stub-only fallback for them; preflight/queue failures consume no budget and are only counted as `failed` | `tests/unit/test_vlm_router.py::test_real_adapter_preflight_consumes_no_budget` (+ existing `test_budget_*`, `test_unavailable_contributes_zero_requests`) |
| M2-006 | A confirmed-but-undispatched visual event stayed `pending` forever: recovery skipped all visual workflows | `app/pipeline.py` `_resume_pending_event`: resume a valid persisted confirmation (`visual_status==ok` + matching expected label + `routing.confirmed`) through the guarded idempotent dispatch; mark an unconfirmed interrupted candidate `interrupted_unverified`/`interrupted` instead of firing or silently pending | `tests/integration/test_m2_visual.py::test_resume_confirmed_visual_event_dispatches_once`, `test_resume_unconfirmed_visual_event_marked_interrupted` |
| M2-007 | A committed/acknowledged workflow disable did not re-check before provider admission; the Teacher still received one image | `app/pipeline.py` `_route_visual`: recheck authoritative approval under `dispatch_admission_lock` before routing and record `cancelled` with no Teacher call; `app/vlm_router.py` `admit` guard rechecks at the transport boundary for a disable landing mid-route | `tests/integration/test_m2_visual.py::test_disable_before_route_cancels_without_teacher_call`, `tests/unit/test_vlm_router.py::test_admit_false_cancels_without_outbound` |

Limitations unchanged: real Teacher visual inference remains **UNVERIFIED** (no
`VLM_BASE_URL`); mocked/stub Teacher tests do not verify real inference. An
unconfirmed interrupted visual candidate is marked interrupted rather than
re-routed (no frame is available during recovery). The abandoned over-deadline
request is a bounded daemon worker whose late response is never consumed.

## M2 exceptional recovery (2026-10-10)

Addresses the four P1 defects re-reproduced at the frozen final-review snapshot
`a93d983dd89b31e329689b09726c5a1cfbf4dbb5` (base
`2127d207a7bf2e1401d40e5ad82d926eb0f66a57`). The controller replayed the
reviewer's original reproductions at that commit: all four reproduced. The
assertions below characterize the **desired** behaviour. Full suite: **140
passed** (`services/api/.venv/bin/python -m pytest tests/ -q
--ignore=tests/orchestrator`), +17 tests vs. the prior pass; no frontend changes.

| ID | Defect (reproduced at `a93d983`) | Fix | Evidence |
| --- | --- | --- | --- |
| M2-001 | Heuristic fallthrough still accepted unconsumed semantics: `does not enter`→entry, `less than 10 minutes`→dwell≥600s, `delete the video`→notification, a second named zone / an extra visual condition (`and leaking oil`) / an unknown `if its …` clause silently dropped | `app/nl_compiler.py`: reject negated triggers (`_NEG_TRIGGER`), upper-bound dwell durations (`_UPPER_BOUND_DURATION`), data-management actions (`_DATA_ACTION`), and >1 named zone (`_match_zone`); added `if its/if the/…` condition markers; added a **residual-token filter** (`_residual_tokens`) that rejects any meaningful token not consumed by a recognised grammar element. Supported class/zone/temporal/positive-visual intent preserved | `tests/unit/test_nl_compiler.py::test_delete_video_action_rejected`, `test_negated_trigger_rejected`, `test_upper_bound_dwell_duration_rejected`, `test_alternative_zones_rejected`, `test_extra_visual_condition_rejected`, `test_unknown_if_condition_rejected`, `test_single_named_zone_among_many_still_ok`, `test_longer_than_dwell_still_ok` |
| M2-003 | A returned `timeout` released the concurrency slot while the abandoned daemon worker was still reading the body: at concurrency 1, four sequential trickling requests produced four active workers | `app/teacher.py`: `_ConcurrencySlot` holds the bounded-semaphore permit; on an over-deadline abandon the slot is **deferred** and a reaper releases it only after the worker truly finishes (and drops the response to bound retention). The semaphore therefore caps active transport workers/body-reads even across timeouts; preflight rejection still consumes zero HTTP budget | `tests/unit/test_transport.py::test_timeout_does_not_free_capacity_for_concurrent_workers` (genuine httpx + MockTransport trickle), existing `test_phase_compliant_trickle_exceeds_overall_deadline`, `test_fast_streamed_response_still_confirms` |
| M2-007 | The transport-boundary approval read ran OUTSIDE any lock, so a disable could commit/return 200 between the read and the grant, then still admit one HTTP POST | `app/vlm_router.py` `_attempt_guard`: the authoritative `admit()` read AND the budget grant are now one atomic boundary under `dispatch_admission_lock` (the same lock the API disable commits under). A disable lands fully before (→ revoked, no request) or fully after (→ already admitted); the lock is NOT held during inference | `tests/integration/test_m2_visual.py::test_disable_cannot_interleave_during_transport_admission` (exact interleaving: disable BLOCKS during the admission boundary), existing `test_disable_before_route_cancels_without_teacher_call` |
| M2-008 | Pydantic lax coercion turned `confidence: true`→`1.0` and `"0.9"`→`0.9`, authorizing an action | `app/teacher.py` `_ProviderResponse`: a `mode="before"` validator requires an actual JSON number — rejects booleans (an `int` subclass), strings, NaN and infinities before the `[0,1]` bounds apply. Valid numeric/int confidence and the optional-for-`unknown` path preserved | `tests/unit/test_teacher.py::test_boolean_confidence_rejected`, `test_false_confidence_rejected`, `test_numeric_string_confidence_rejected`, `test_nan_confidence_rejected`, `test_infinity_confidence_rejected`, `test_integer_confidence_still_valid`; boundary: `tests/integration/test_m2_visual.py::test_boolean_confidence_cannot_fire_action` |

Limitations: real Teacher visual inference, foveated gains and VLM cost savings
remain **UNVERIFIED** (no `VLM_BASE_URL`/GPU). M2-003 cancellation is cooperative
— the abandoned worker is joined by its reaper (bounded by the production
`read_timeout` for a stalled connection); a perfectly-paced infinite trickle
would hold one permit until its body ends. No credentials were needed for this
mocked-transport/security recovery.

### Exceptional recovery acceptance outcome

Fresh exact-snapshot review of `0250feb7de89a6bc6df06a3d293dbc3b1cd93595`
returned **CHANGES_REQUIRED**. The implementation entries above record attempted
fixes, not milestone acceptance. M2-001 still accepts unsupported meaning;
M2-003 still leaves expired transports active and can confirm after an
admission wait exceeds the overall deadline. M2-008 strict confidence is verified.
The independent reviewer verified M2-007 read/grant serialization; supplementary
controller QA still reproduces a new HTTP send after disable acknowledgment in
the separate grant/send gap. M2 remains BLOCKED; M1 is accepted, M3 has not started.
See `.agents/reviews/M2.md` and `.agents/review-evidence/M2-0250feb/README.md`.
No further automatic recovery cycle is authorized.
