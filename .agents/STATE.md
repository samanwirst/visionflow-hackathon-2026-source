# VisionFlow execution state

## M2 BLOCKED — exceptional recovery exhausted

- Implementation branch: `feature/visionflow`.
- Original final normal snapshot: `a93d983dd89b31e329689b09726c5a1cfbf4dbb5`.
- Exceptional implementation: `7096ab3dd3908467c79ba2de75ddd70970995cce`.
- Exact tested/reviewed handoff: `0250feb7de89a6bc6df06a3d293dbc3b1cd93595`.
- Base: `2127d207a7bf2e1401d40e5ad82d926eb0f66a57`.
- Fresh independent verdict: **CHANGES_REQUIRED / FAIL**, completed in 818s.
- Review branch: `review/M2-r4`.
- One exceptional Claude pass and ONE fresh independent Codex review consumed.
  Two normal fixes plus this exception: fix counter 3. No further exception
  cycles or automatic M2 fixes are authorized. Total agent calls: 16.
- M1 remains accepted at `390b337cc1f58504864ba27a23a96c1e811b358d`.
  M1 application code and regression tests were not changed by this recovery.
- M3/M4 implementation has not started. M2 was never accepted.

## Remaining material defects

| ID | Gate result | Minimal scope |
| --- | --- | --- |
| M2-001 | Independent P1: token stripping still discards unknown action/visual meaning and ambiguous duration | Complete supported-command grammar; reject unconsumed clauses and ambiguity |
| M2-003 | Independent P1: expired worker keeps reading unbounded data; admission wait can exceed deadline then send/confirm | Owned cancellable streaming transport, bounded bytes/cleanup, deadline through admission and parsing |
| M2-007 | Supplementary controller P1: after grant but before HTTP, disable acknowledges; releasing pause starts HTTP afterward | Couple approval to actual request start and revoke unsent attempts |
| M2-008 | Verified fixed: malformed confidence produces no action; valid numeric controls work | No further change needed |
| M2-009 | Existing P2: completed `_inflight_result` storage bypasses bounded primary cache | Deferred outside this exceptional scope |

The independent reviewer verified M2-007's original read/grant race fixed.
The controller preserved that judgment verbatim and appended a distinct
actual-send reproduction required by the human's stronger guarantee.
See [the consolidated report](reviews/M2.md) and
[safe executable evidence](review-evidence/M2-0250feb/README.md).

## Executed evidence

- Before implementation, all four original reviewer reproductions ran on a clean
  frozen `a93d983` worktree and reproduced the defects. Their four passing
  characterization assertions describe bugs, not acceptance passes.
- Claude's single focused pass completed in 1328s/73 turns and committed code
  and exact handoff. It added 17 regression cases. No extra implementation call.
- Full `make check`: **160 tests passed**, four existing warnings; frontend
  production build passed. This is the exact `0250feb` snapshot.
- `make smoke`: real YOLO processed 30 frames; mean inference **70.1 ms**; bus
  entry at 4.2s, confidence 0.76, JPEG evidence and local notification persisted.
  Synthetic motion of photographic content; not natural held-out accuracy.
- Independent suite: **160 passed**, four warnings, 22.88s, using disclosed
  temporary runner-only 10ms asyncio heartbeat after sandbox startup stalls.
  Frontend build and UI fixture handlers passed; full browser/network not tested.
- Independent 25 focused cases included bug-characterization assertions and
  reported remaining defects; these are not milestone acceptance passes.
- Supplemental desired-behavior QA: **2 failed**, three warnings, 0.68s, on the
  exact snapshot; no application changes.
- Ten fake-CLI/temp-Git recovery-policy checks passed: single-call enforcement,
  timeouts, failure stops, committed-handoff resume and normal M3-only-after-PASS.

The controller was held while its single reviewer continued under a separate
25-minute watchdog, preventing premature M3 despite confirmed supplementary
failures. After validating CHANGES_REQUIRED for the exact SHA, the hold was
released; the controller published the report and stopped at its fix limit.
No implementation/review subprocess remains active.

## Smallest safe demo scope

Recommended reduction: manually reviewed JSON geometric workflows with real
uploaded-video detection/tracking, temporal events, evidence and local
notifications. Keep the disclosed unavailable Teacher fallback; disable or avoid
NL compilation and Teacher-backed automation until their gates are satisfied.
This recommendation is not another implementation cycle or an M2 PASS.

## Architecture and honest limitations

Preserve committed PRD/VLM_ARCHITECTURE/METRICS requirements: cheap continuous
detection/tracking and temporal rules; selective event/ROI Teacher routing;
sampling, deduplication, caching and call budgets; secure self-hosted compatible
provider; Teacher labels -> verified data -> frozen embeddings -> lightweight
Student; session-disjoint held-out/shadow/manual activation/rollback; identical
held-out full-frame/ROI/foveated comparisons and actual Vast.ai/GPU cost evidence.

No GPU instance is connected, no external provider API key is required now.
Real Teacher labels/inference/quality, foveated gains, Student evaluation and
unexecuted latency/GPU/throughput/rental-cost/savings comparisons remain
**UNVERIFIED**. No savings percentage or fabricated result is claimed.
No main merge, force push, hardware operation, purchase, GPU rental or public
deployment occurred.
