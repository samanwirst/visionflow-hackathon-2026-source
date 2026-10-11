# Visual analytics architecture audit

Audit date: 2026-10-10. Application snapshot: `cc595239c2d361c714a996f23712643981209b95`.
Scope: the operator's six cost-efficient visual analytics requirements,
existing README/PRD, and the relevant implementation. This focused audit
does not replace the independent M1 correctness review.

| Requirement | Implementation evidence | Gap and milestone |
| --- | --- | --- |
| Cheap continuous detection/tracking and temporal rules | `app/pipeline.py` samples at `PROCESS_FPS`; real YOLO/ByteTrack in `packages/vision`; polygon enter/exit/dwell/count state machines | Present on uploaded video. Live camera and wider held-out quality remain unverified. Preserve this path throughout M2–M4. |
| Event-triggered VLM on selected frames/ROIs | No Teacher adapter or visual-condition routing in the current runtime | M2-AC2/3: add explicit candidate-event selection, ROI/frame provenance, validated Teacher responses, and failure/budget states. |
| Deduplication, sampling, caching | Frame sampling, event cooldowns and persisted dedupe keys exist | No VLM in-flight dedupe, call budget, or versioned result cache. M2-AC4/5 add these and actual routing counters. M1 external-action recovery remains defective. |
| Teacher labels → verified dataset → frozen embeddings → lightweight Student | `packages/autolearn/README.md` is a scaffold; no training implementation | M3-AC1/2/3: real Teacher labeling, human review, traceable split manifests, frozen encoder, actual fitting and saved artifacts. |
| Held-out evaluation, shadow, manual activation, rollback | Not implemented for models; workflow creation is already disabled by default | M3-AC4/5/6: computed evaluation, side-effect-free shadow comparisons, quality and human gates, Teacher fallback, tested rollback/restart. |
| Measured baseline/optimized cost, latency, quality | M1 smoke reports frame inference time; `docs/METRICS.md` was an empty template | M4-AC1/2/3: paired real-provider experiments over the same held-out task/input; actual usage, timing, quality, rates, denominators and cache conditions. No savings measurement exists. |

README previously described a detector dashboard and deferred learning,
without making the inference hierarchy explicit. PRD previously mentioned
selected frames, frozen embeddings, and held-out evaluation, but did not
enforce event-triggered Teacher calls, cache validity, request budgets, or a
paired baseline experiment per milestone. README and PRD now state these
requirements and distinguish them from current functionality. Their source
is the operator's architectural requirement, not an invented hackathon rule.

No VLM API credential is configured in the inspected environment or project
`.env`. Claude/Codex development-agent authentication does not configure a
production Teacher provider. Provider adapters and deterministic boundary
tests can be implemented, but a real Teacher smoke and cost/quality comparison
cannot be claimed from mocked responses or manual labels. Configure the
selected provider securely before those acceptance checks; never commit keys.

M1 review `review/M1-r2` found four P1 defects after two authorized fix cycles
(disable/dispatch race, ambiguous webhook delivery, DNS rebinding, truncated
input). The controller is BLOCKED at its configured limit, with no active
M1 fix interrupted. The milestone requirements are queued in the controller;
M2 must not start by bypassing M1 acceptance.
