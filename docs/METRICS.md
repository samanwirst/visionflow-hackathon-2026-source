# Measurements and benchmark protocol

Real self-hosted Teacher labeling, foveated-input comparisons, Student
held-out results, and Vast.ai cost comparisons are **UNVERIFIED** until
executed. A prepared adapter, passing mock tests, or a research paper is not
measurement evidence. No savings percentage is established.

## Existing M1 smoke evidence

Controller gate for `390b337`: 66 tests passed, frontend production build
passed, and real YOLO analyzed 30 frames from 180 decoded frames. It produced
a bus-enter event at 4.2 s with confidence 0.76, served evidence, and persisted
a local notification. Mean detector inference in that run: 63.3 ms.

Source: `.agents/reviews/M1.md` and the controller's retained gate output.
This is a development-CPU smoke using real photographic content with
synthetic motion, not a natural-video accuracy study or VLM savings benchmark.
Unit/integration detector/provider stubs are disclosed and do not establish
production model quality. No broad robustness or percentile claim follows
from this mean.

## M3 input-strategy experiment

Freeze one session-disjoint held-out video/event manifest, verified ground
truth, Teacher model/revision, prompt/schema, candidate schedule and thresholds.
Compare full frame, high-resolution ROI crop, and foveated high-resolution
ROIs plus low-resolution scene context. Keep ROI-selection errors, small
objects and context-dependent questions in the evaluation. Preserve a
full-frame fallback with its reason and effective strategy recorded.

| Strategy | Event accuracy / precision / recall / F1 | Input tokens / actual cost | p50 / p95 latency | Status |
| --- | --- | --- | --- | --- |
| Full frame | UNVERIFIED | UNVERIFIED | UNVERIFIED | Not executed |
| ROI crop | UNVERIFIED | UNVERIFIED | UNVERIFIED | Not executed |
| Foveated context + ROI | UNVERIFIED | UNVERIFIED | UNVERIFIED | Not executed |

Also report unknown/failure coverage, confusion matrix, missed/clipped ROIs,
tracking failures, spatial-context loss, small-object misses, and server
multi-image/format limits. Ground truth is independent of Teacher/Student
agreement. Encoded pixel/byte counts and actual server token usage are
separate quantities; missing token telemetry stays UNVERIFIED.

## M4 matched pipeline experiment

Use identical held-out inputs/tasks and ground truth across these variants:

| Pipeline | Real Teacher calls / attempts | Quality and coverage | Latency / throughput | Rental cost / event |
| --- | --- | --- | --- | --- |
| Baseline: VLM on each sampled frame | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| Optimized: temporal routing + sampling + cache | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| Student/hybrid with budgeted Teacher fallback | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |

Fix and publish the sampled-frame schedule, task/prompt/model, input hashes,
code/config versions and evaluation definitions. Strategy ablation uses the
same candidate events to isolate preprocessing; pipeline comparison also
measures event recall including missed candidates. Do not omit unknowns or
failed requests from coverage/error accounting. Separate cold/warm model and
cache runs; record repetitions, sample counts, fallback calls and cache hits.

Measure frame/input-preparation, Teacher/Student, and event-to-action latency
with monotonic timing and explicit boundaries, then compute p50/p95 from raw
observations. Report completed events per wall-second and processed video
duration per wall-second with denominators. GPU activity time/utilization,
where telemetry is available, is distinct from charged rental wall time.

## Actual Vast.ai cost accounting

Use the operator's actual rental records: currency, GPU/instance description,
charged interval, price/rate history and charges. Include or explicitly
delimit model load, idle time, benchmark/training time, storage/network and
local compute. No assumed external VLM token-price schedule is used for the
self-hosted experiment.

Allocated experiment cost/event = attributed actual rental expense divided
by stated processed-event count. Show true/false/failed event counts as well;
more false alerts cannot be presented as economic improvement. If allocating
cost by elapsed time and an observed rental rate, label it an estimate and
reconcile it with the actual bill. A faster run can leave a fixed rental
window's bill unchanged; report this honestly.

Savings = (measured baseline cost − measured optimized cost) / measured
positive baseline cost, only when scope, inputs, billing allocation and
quality definitions match. Pixel reduction and lower request/token counts
do not imply linear monetary savings. Paper gains are never project results.

## Reproducibility and limitations

Preserve safe raw predictions, ground truth and split hashes, commands,
model/prompt/preprocessing versions, requested/effective strategy, ROI mapping,
fallbacks, call/usage/timing records, hardware/runtime configuration, and
cost allocation inputs. Keep credentials, private endpoints and unconsented
images out of public reports. Document all missing telemetry and conditions
not represented by the dataset. Until the GPU and held-out evaluation run,
the comparison tables above remain **UNVERIFIED**.

Acceptance gates: PRD M3-EXP1, M3-AC7 and M4-AC1..3a. Architecture:
[VLM_ARCHITECTURE.md](VLM_ARCHITECTURE.md). Research reference: PRD section 21.
