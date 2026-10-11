# VisionFlow AI — Product & Technical Requirements

**Version:** 1.2
**Project type:** Automotive Computer Vision / Visual AI Agents  
**Duration:** 24-hour hackathon  
**Team:** 3 students  
**Primary vertical:** Automotive service operations  
**Product inspiration:** OpenVector  
**Differentiator:** Cost-efficient visual analytics and AutoLearn specialization

## 1. Mission

Build a deployed, working visual AI automation platform that allows a user to connect a camera or upload a video, describe a visual condition in natural language, detect matching events, and execute an authorized workflow.

The platform must additionally demonstrate how a frequently repeated visual task can be transferred from a general-purpose Vision-Language Model (Teacher) to a smaller specialized model (Student), subject to validation and human approval.

The GVERA robot is an optional physical output device, not a dependency of the core product.

The primary innovation is an OpenVector-inspired inference hierarchy:
cheap continuous detection/tracking and temporal rules → event-selected
frames/ROIs → deduplicated, budgeted Teacher VLM requests → verified labels
→ frozen embeddings and a lightweight Student. Detection-only dashboards
do not satisfy this architectural requirement. Savings are an experimental
question; never claim a percentage without measured comparable runs.

## 2. Hackathon alignment

The team understands the judging criteria as:

| Criterion | Weight |
|---|---:|
| Technology | 25% |
| Innovation | 25% |
| Impact | 25% |
| Presentation | 15% |
| Business | 10% |

Expected deliverables, subject to verification against the original Final Task:

- A working frontend and backend, accessible during judging.
- Real end-to-end execution rather than static mockups.
- Ability to accept new inputs provided by judges.
- Clear data provenance and licensing.
- Measured accuracy, latency, and approximate operating cost.
- Privacy and consent considerations.
- A credible three-month pilot proposal.
- A seven-minute pitch and live demonstration.

**Source-of-truth rule:** If the official Final Task contradicts any item above, the official document takes precedence. Record all discrepancies in `docs/TRACEABILITY.md`.

## 3. Product scope

### P0 — Required MVP

1. Accept an uploaded MP4 video.
2. Support at least one live camera source where hardware allows.
3. Detect and track supported objects.
4. Let the user define a polygonal region of interest.
5. Support visual events:
   - Object enters zone.
   - Object exits zone.
   - Object remains in zone longer than a threshold.
   - Object count exceeds a threshold.
6. Accept a natural-language instruction and compile it into a validated, editable workflow.
7. Require user confirmation before enabling the workflow.
8. Execute a real authorized action when an event occurs.
9. Save timestamp, camera ID, rule ID, event type, and evidence image.
10. Display events and actions in a functional web interface.
11. Run automated tests and support reproducible deployment.

### P1 — Main innovation: AutoLearn

1. Identify recurring visual classification tasks.
2. Collect representative, non-duplicate image examples.
3. Use a real Teacher VLM to propose labels and preserve their provenance.
   Manual labeling is a disclosed development fallback; it does not verify
   the teacher-label stage or satisfy the full AutoLearn acceptance gate.
4. Provide a UI for correcting labels.
5. Extract visual embeddings using a pretrained model.
6. Train a small Student classifier on those embeddings.
7. Evaluate Student on held-out video sessions.
8. Compare baseline VLM, event-routed Teacher, and approved Student pipelines
   using measured quality, latency, usage, and documented cost calculations.
9. Run Student in shadow mode before activation.
10. Activate only after explicit approval and a passed quality gate.
11. Support fallback to Teacher and model rollback.

### P2 — Stretch goals

- RTSP support.
- More complex VLM-based visual conditions.
- Telegram notifications.
- GVERA integration via documented and approved API.
- ONNX export.
- Multi-camera dashboards.
- Advanced foveated vision or adaptive frame sampling.

## 4. Primary use case

A Toyota or BYD service center wants to know when a vehicle enters a service bay, how long it remains there, and when it exits.

The user connects a camera, draws the service bay polygon, and writes:

"When a car stays in Service Bay A for more than ten minutes, create an alert for the service advisor."

The system compiles this instruction into a rule, tracks vehicles, detects the dwell event, stores evidence, and creates an alert.

A second use case demonstrates AutoLearn:

"Determine whether a vehicle's hood is open."

Initially, a Teacher VLM helps classify selected images. The platform then collects verified examples, trains a specialized classifier, evaluates it, and demonstrates lower-cost repeated inference where quality permits.

## 5. User interface

### Page A — Dashboard

- Active camera and uploaded-video sources.
- Active workflows.
- Recent events.
- Processing status.
- Teacher/Student inference counters.
- Error and health indicators.

### Page B — Camera Workspace

- Video player or live feed.
- Bounding boxes and track IDs.
- Editable polygon zones.
- Playback controls for uploaded videos.
- Evidence snapshot viewer.

### Page C — Workflow Builder

- Natural-language instruction input.
- Generated structured workflow.
- Editable trigger parameters.
- Camera and zone selection.
- Action selection.
- Validation errors.
- Explicit enable/disable control.

### Page D — Event Timeline

- Timestamp.
- Event name and confidence.
- Source camera.
- Evidence image.
- Action status.
- Filtering by rule, source, and time.

### Page E — AutoLearn

- Recurring task candidates.
- Teacher-labeled dataset with review controls.
- Positive/negative example counts.
- Train Student button.
- Training job status.
- Held-out evaluation metrics.
- Shadow-mode comparison.
- Activation, fallback, and rollback controls.

### Page F — Settings / Integrations

- Authorized webhook destinations.
- Optional Telegram configuration.
- Optional GVERA integration.
- Data retention settings.
- Consent and privacy information.

## 6. Core architecture

Pipeline:

Camera/Video → Configured Frame Sampling → Cheap Object Detection + Tracking
→ Zone/Temporal Rules → Candidate Event → Validated Workflow → Authorized
Action → Evidence & Event History.

For visual tasks that cannot be resolved by object detection alone:

Candidate Event → Selected Frame/Object ROI → Deduplication + Versioned Cache
→ Approved Student or Budgeted Teacher VLM → Strict Visual Result Validation
→ Confirmed Event → Authorized Action.

The cheap pipeline remains the continuous path. Expensive VLM calls occur
only for supported visual predicates on eligible candidate events or
explicit AutoLearn labeling requests, never as an unconditional per-frame
processing step. A separate, explicit benchmark mode may call the VLM on
every frame at a documented sampling rate to measure the baseline.

The router must enforce configurable per-task/source sampling intervals,
in-flight deduplication, cooldowns, and request limits. Its decision records
must expose sampled candidates, suppressed duplicates, cache hits/misses,
actual Teacher calls, Student decisions, fallback reasons, and unknown/error
results. A provider timeout, invalid response, exhausted budget, or missing
credential cannot be converted to a positive visual condition.

Cache keys include the visual task/prompt, model and preprocessing versions,
image/ROI identity, and applicable scope. TTL/invalidation and bounded
storage must prevent stale reuse after task/model changes; near-duplicate
suppression is distinct from asserting that two images have equal labels.
Cache entries are never treated as newly verified training labels. ROI
selection and cloud transmission must be disclosed and consented.

The frontend must not contain authoritative business logic. All event decisions and actions are validated server-side.

### 6.1 Self-hosted Teacher contract (M2–M3)

Target a configurable **OpenAI-compatible self-hosted VLM**, for example
vLLM serving `Qwen/Qwen2.5-VL-7B-Instruct` on an operator-rented Vast.ai GPU.
Protocol compatibility does not require an OpenAI account or external
provider API key. Read `VLM_BASE_URL`, `VLM_MODEL`, and optional `VLM_API_KEY`
only from server environment/ignored configuration; never send credentials
to the browser or commit them. Missing endpoint/GPU access selects the
disclosed unavailable/manual fallback, not invented Teacher predictions.

The provider interface accepts task/schema, selected encoded images, ROI
coordinates, source/event identity and preprocessing metadata. It returns
strictly validated Teacher labels or unknown/error, actual usage where
available, model identity and measured timing. Bound request size, timeout,
concurrency and total attempts; retry only classified transient failures
with backoff inside the same call budget. Validation/auth errors must not
retry indefinitely. Keys, image bytes and unnecessary personal data stay out
of logs. Prefer a private SSH tunnel or an authenticated TLS gateway; never
publish an unauthenticated inference server. See
[the self-hosted/input-router plan](VLM_ARCHITECTURE.md).

### 6.2 Foveated-input experiment (M2 design, M3 evaluation)

Behind one configurable input-strategy interface support:

| Strategy | Visual information retained |
| --- | --- |
| `full_frame` | Whole scene with a documented resolution cap. |
| `roi_crop` | A detailed crop around selected object(s); scene context outside the crop is discarded. |
| `foveated` | High-resolution important region(s) **plus lower-resolution whole-scene context**, with preserved spatial mapping. |

Foveation is not simply cropping. YOLO boxes, track identity, workflow zones
and temporal candidate events select important regions; per-frame VLM calls
are still prohibited outside explicit benchmark mode. An MVP may send a
low-resolution context image plus labeled high-resolution crops as one
multimodal request. This is an engineering approximation inspired by the
paper in section 21, not a claim to reproduce its retinal sampling method.

Preserve an explicit `full_frame` fallback for spatial/context-dependent
questions, uncertain or invalid ROIs, small objects that downsampling could
erase, and unsupported multi-image inputs. Record requested/effective
strategy, fallback reason, dimensions, crop-to-scene coordinates and
preprocessing version; cache keys must distinguish these inputs.

M3 compares all three strategies on the **same held-out videos/events**,
ground truth, model/prompt and candidate schedule. Measure event accuracy,
precision/recall/F1, unknown coverage, actual input tokens where exposed,
latency and failure cases. Lower pixel counts do not imply proportional
token reductions or rental/API cost reductions. The paper's reported gains
are not VisionFlow results. This remains an optimization experiment and
does not reopen or block the completed M1 recovery.

## 7. Recommended technology stack

- Frontend: Next.js, TypeScript, Tailwind CSS.
- Backend: FastAPI, Python, Pydantic.
- Video: OpenCV and FFmpeg.
- Detection: a suitable pretrained detector with verified model and code licenses.
- Tracking: ByteTrack or a simpler tested tracker.
- Event logic: deterministic Python state machines.
- LLM: structured JSON generation with schema validation.
- VLM: image-capable model for selected visual classifications.
- Embeddings: DINOv2, SigLIP, or another suitable pretrained encoder.
- Student: Logistic Regression or small MLP.
- Storage: SQLite for MVP; PostgreSQL optional.
- Updates: WebSocket or Server-Sent Events.
- Deployment: Docker Compose or an equivalent reproducible setup.

The team must verify that selected pretrained weights support the required object classes. Do not assume a model detects an automotive-specific object without testing it.

## 8. Workflow DSL

A workflow has:

- `id`
- `name`
- `source_id`
- `enabled`
- `object_class`
- `trigger.type`
- `trigger.zone_id`
- `trigger.duration_seconds`, where applicable
- `trigger.threshold`, where applicable
- `action.type`
- `action.config`
- `cooldown_seconds`
- `requires_approval`

Supported trigger types:

- `object_entered_zone`
- `object_exited_zone`
- `dwell_in_zone`
- `object_count_exceeded`
- `visual_condition_verified`

Supported actions:

- `create_notification` (the shipped M1 action: local, side-effect-free)
- `create_service_event` (planned)
- `send_webhook` — **removed in M1** (M1-011/M1-012). Arbitrary outbound
  delivery was disabled because the SSRF/allowlist guard could not safely
  protect the actual delivery connection against DNS rebinding without a risky
  custom DNS/TLS egress transport, and an interrupted delivery could not be
  distinguished from a completed one on recovery. A validated-egress
  integration can be reintroduced later; until then the API rejects this action
  at creation and any legacy workflow naming it is rejected at dispatch with no
  network activity.
- `robot_preset`, only when explicitly configured and authorized

The LLM must produce only a structured workflow. It must never generate arbitrary code that is executed by the backend.

Unsupported requests must produce a clear validation error rather than a fabricated workflow.

## 9. Event correctness

The event engine must implement:

- Track-based object identity where available.
- Polygon inclusion using a consistent object anchor point.
- Debounce or temporal smoothing.
- Cooldown to prevent duplicate alerts.
- Tolerance for brief tracking loss.
- State reset when an object truly leaves.
- Event IDs suitable for idempotent action execution.
- Explicit unknown/error states when confidence is insufficient.

A single object remaining in a zone must not trigger the same one-shot event repeatedly unless the workflow explicitly allows repetition.

## 10. AutoLearn design

### 10.1 Eligibility

A task may be proposed for AutoLearn when:

- It is sufficiently repetitive.
- It can be expressed as a stable visual classification.
- There are enough diverse positive and negative examples.
- The expected future inference cost justifies training.
- A reliable evaluation set can be constructed.

The system must not assume that repeated prompts automatically provide valid training labels.

### 10.2 Dataset collection

- Sample from multiple time intervals or videos.
- Avoid near-identical consecutive frames.
- Store image references, timestamps, source, and label provenance.
- Distinguish Teacher-generated labels from human-verified labels.
- Collect both positive and negative examples.
- Obtain permission to use recorded footage.
- Prevent train/test leakage by splitting at the video/session level.

### 10.3 Student training

- Keep the pretrained embedding encoder frozen in MVP.
- Train a small classifier on extracted embeddings.
- Save model weights, preprocessing configuration, class names, and dataset version.
- Save training duration and environment information.
- Ensure inference uses the exact same preprocessing as training.

### 10.4 Evaluation and activation

Report:

- Precision.
- Recall.
- F1.
- Confusion matrix.
- p50/p95 inference latency.
- Estimated cost per 1,000 checks.
- Teacher/Student disagreement rate.

Activation must require a predefined quality gate and explicit user approval. The system must remain on Teacher or hybrid mode if Student fails.

Shadow mode must not execute production actions based solely on Student output.

Support rollback to the previous active version.

### 10.5 Anti-drift controls

- Do not automatically retrain on Student's own unverified predictions.
- Sample uncertain cases for review.
- Monitor disagreement and quality changes.
- Preserve a known-good model version.
- Record all model activations and rollbacks.

## 11. API contract

Required endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service readiness |
| POST | `/api/sources/upload` | Upload video |
| GET | `/api/sources` | List sources |
| POST | `/api/zones` | Create or update zone |
| POST | `/api/workflows/compile` | Text → validated draft |
| POST | `/api/workflows` | Save workflow |
| POST | `/api/workflows/{id}/enable` | Enable after validation |
| GET | `/api/workflows` | List workflows |
| GET | `/api/events` | Event timeline |
| GET | `/api/events/{id}` | Event and evidence |
| POST | `/api/autolearn/jobs` | Create dataset/training job |
| GET | `/api/autolearn/jobs/{id}` | Job status |
| POST | `/api/autolearn/jobs/{id}/labels` | Review labels |
| POST | `/api/autolearn/jobs/{id}/train` | Train Student |
| POST | `/api/autolearn/jobs/{id}/evaluate` | Evaluate Student |
| POST | `/api/autolearn/jobs/{id}/activate` | Approved activation |
| GET | `/api/metrics` | Processing metrics |

All API payloads must have Pydantic models. The frontend and backend must agree on shared field names and status values.

For the hackathon, asynchronous processing may be implemented with a simple background worker rather than a distributed queue.

## 12. Database entities

Minimum entities:

**VideoSource:** id, type, name, URI/path, status, created_at.

**Zone:** id, source_id, name, normalized polygon coordinates.

**Workflow:** id, source_id, name, compiled DSL, enabled, created_at.

**TrackEvent:** id, source_id, workflow_id, track_id, type, timestamp, confidence, evidence_path, action_status.

**ActionExecution:** id, event_id, action_type, status, attempt_count, error, executed_at.

**AutoLearnTask:** id, source_id, visual_prompt, status, teacher_model, student_model_version.

**TrainingSample:** id, task_id, image_path, label, label_source, verified, session_id.

**ModelVersion:** id, task_id, artifact_path, metrics, status, created_at.

## 13. Security and privacy

- Store secrets only in environment variables.
- Commit `.env.example`, never `.env`.
- Validate upload types and enforce file-size limits.
- Restrict file and evidence access.
- Validate webhook destinations; do not permit arbitrary internal-network requests. (In M1 this is enforced by removing outbound webhooks entirely — M1-011/M1-012 — rather than shipping an allowlist guard that DNS rebinding could bypass.)
- Require authorization for workflow actions and model activation.
- Provide visible notice and consent for recorded people.
- Minimize retention and support deletion.
- Do not claim that video is processed locally if frames are sent to a cloud VLM.
- Keep logs free of API keys and unnecessary personal data.

## 14. GVERA integration

The robot is optional.

If a documented and authorized control interface exists, implement a narrow adapter exposing only approved preset actions.

Requirements:

- Explicit enable switch.
- Allowlisted commands.
- No direct LLM-to-actuator commands.
- Safe workspace and human supervision.
- Emergency stop procedure.
- No dependency on the robot for the core demo.

If hardware or API access is unavailable, mark the integration as unavailable rather than simulating a successful robot command.

## 15. Quality and performance evaluation

Prepare held-out test videos not used to tune thresholds.

Measure:

- Event-level precision, recall, and F1.
- False events per hour.
- End-to-end event-to-action latency.
- p50/p95 frame-processing latency.
- Effective processing throughput.
- Teacher requests and token usage.
- Student inference time.
- Estimated processing cost per 1,000 checks.
- AutoLearn training time.
- Performance on changed lighting, occlusion, and tracking loss.

Do not invent performance numbers.

Where the sample size is small, state that results are preliminary.

## 16. Acceptance criteria

### Core MVP is accepted when:

1. A judge can upload a previously unseen supported video.
2. The backend genuinely processes the video.
3. The user can define a zone and enable a supported workflow.
4. At least one correct event is generated.
5. A real authorized action occurs.
6. The event is visible with timestamp and evidence.
7. The system can be restarted using documented instructions.
8. Automated tests run successfully.
9. Failures are visible and not silently disguised.

### AutoLearn is accepted when:

1. Real labeled examples exist.
2. A Student model is genuinely fitted and saved.
3. Held-out evaluation runs.
4. Metrics are displayed.
5. Shadow mode compares real Teacher and Student outputs without actions.
6. Activation is gated.
7. Rollback works.

A decorative progress bar or hardcoded Student predictions do not count as AutoLearn.

### M2 — Event-routed visual reasoning and workflow product

- **M2-AC1:** Natural language compiles to a strict editable DSL. Unsupported
  or ambiguous requests are rejected; drafts remain disabled until explicit
  validation and approval. Generated code is never executed.
- **M2-AC2:** The real detector/tracker and temporal rules remain the cheap
  continuous path. A supported visual predicate selects a timestamped frame
  or object ROI only after an eligible candidate event. Ordinary geometric
  rules make zero Teacher calls. API and UI expose the visual predicate and
  its routing settings consistently.
- **M2-AC3:** A real image-capable Teacher adapter returns a validated label,
  confidence/unknown result, model/version, timing, and actual usage where
  exposed. Timeout, invalid response, missing credentials, or request-budget
  exhaustion remain visible and cannot trigger a confirmed visual action.
  Stub adapter tests alone do not verify real Teacher operation.
- **M2-AC3a:** The adapter targets an operator-configured OpenAI-compatible
  self-hosted endpoint using `VLM_BASE_URL`, `VLM_MODEL`, optional
  `VLM_API_KEY`; supports selected images/ROIs, strict structured-label
  validation, timeouts, bounded transient retries/concurrency and redacted
  errors. Secure Vast.ai connection instructions are included. No external
  API key is required for interface implementation. Without GPU access the
  interface/fallback can be verified separately; real Teacher labeling is
  explicitly **UNVERIFIED**.
- **M2-AC4:** Configurable sampling, per-task/source budgets, cooldowns,
  in-flight deduplication, and a bounded versioned cache are exercised.
  Tests prove repeated candidates suppress unnecessary calls, changed
  task/model/ROI inputs invalidate reuse, and action deduplication remains
  correct. No ordinary processing loop calls the VLM on every frame.
- **M2-AC5:** Events expose evidence and routing provenance; counters reflect
  actual calls, hits, misses, suppressed requests, and failures. Webhooks,
  enable/disable, and provider-bound data retain their authorization and
  privacy boundaries. The new-user workflow runs without source edits.
- **M2-EXP1:** The input-router design has configurable `full_frame`,
  `roi_crop`, and `foveated` strategies selected from detector/tracker/event
  regions. Foveated inputs retain coarse scene context and spatial mapping;
  cropping alone must not be labeled foveation. Full-frame fallback and
  strategy-aware caching/provenance are designed and tested when implemented.
  An unfinished strategy is visibly unavailable, never silently simulated.
  This experiment does not authorize extra implementation/review calls.

### M3 — Verified teacher-to-student AutoLearn

- **M3-AC1:** A real Teacher labels selected examples; a user reviews/corrects
  them before training. Each sample stores task, image/ROI reference or hash,
  source/session/timestamp, Teacher model/prompt/result, label provenance,
  verification decision, and dataset version. Both classes and diverse
  sessions are present. Manual-only data is disclosed as a partial fallback.
- **M3-AC2:** Deduplication and source/session-level train/validation/test
  partitioning prevent adjacent-frame and near-duplicate leakage. The held-out
  set is never used for fitting, threshold selection, or label propagation.
  The split manifest and sample counts are inspectable.
- **M3-AC3:** A real pretrained embedding encoder stays frozen; an actual
  lightweight classifier is fitted on verified training embeddings. Save
  encoder/version, preprocessing, class mapping, dataset/split hashes,
  trained weights, duration, and environment. Reloaded inference uses the
  same preprocessing; a progress bar is not training evidence.
- **M3-AC4:** Held-out precision, recall, F1, confusion matrix, counts,
  coverage, and latency are computed from predictions against verified
  ground truth. Teacher/Student disagreement is distinct from accuracy.
  Small samples and unsupported conditions are disclosed.
- **M3-AC5:** Shadow mode records Teacher/Student comparisons without
  executing actions or changing the active model. Activation requires a
  predefined quality/sample-count gate plus an explicit human decision;
  missing or failed evaluation cannot be activated.
- **M3-AC6:** Low-confidence/unsupported Student outputs use a tested Teacher
  fallback within M2 budgets, otherwise an explicit unknown state. Rollback
  restores a working previous model plus encoder/preprocessing/configuration.
  Activation/rollback history persists across restart; unverified Student
  predictions cannot silently become training labels.
- **M3-AC7:** Once the self-hosted GPU is available, execute real Teacher
  labeling through M2's provider interface, verify labels, retrain as needed,
  and evaluate the Student on the untouched held-out inputs. Until executed,
  Teacher labels/comparisons are **UNVERIFIED**; manual labels preserve their
  actual provenance.
- **M3-EXP1:** An executable comparison protocol evaluates full frame, ROI,
  and foveated inputs on the identical held-out event/video manifest, fixed
  Teacher model/prompt, candidate schedule and ground truth. Record accuracy,
  recall/F1, unknown coverage, input tokens if exposed, actual incurred cost
  if available, latency and failures (context loss, small objects, missed/
  clipped ROIs, tracking loss, multi-image limitations). Keep warm/cold cache
  conditions explicit. Unexecuted runs are **UNVERIFIED**, with no borrowed
  paper results or linear pixel-to-cost assumptions.

### M4 — Measured efficiency and deliverable evidence

- **M4-AC1:** A reproducible benchmark runs a VLM-per-sampled-frame baseline,
  the M2 event-routed/cached Teacher pipeline, and the M3 Student/hybrid
  pipeline on the same held-out videos, visual task, ground truth, sample
  schedule, and documented settings. Real provider calls are required for
  provider performance claims; unavailable access is marked NOT TESTED.
- **M4-AC2:** Preserve input/split hashes, code/model/prompt versions,
  hardware, commands, repetitions, raw predictions, call/usage records,
  cache state, and actual timing. Report Teacher call counts, input/output
  usage, cache-hit rate, sampling/drop counts, Student/fallback coverage,
  frame and event-to-action p50/p95 latency, and verified quality for every
  variant. Compare the same semantic task and disclose latency boundaries.
- **M4-AC3:** Derive provider cost from measured billable usage and a dated,
  cited rate source or actual bill. Report local compute/training separately
  with its measurement method; unknown usage/cost remains unknown. Normalize
  by video duration and/or checks with denominators stated. Report quality
  tradeoffs and cold/warm-cache conditions. Savings percentages may be
  calculated only from actual comparable positive baseline measurements;
  no hardcoded savings or substituting fewer calls for measured dollar cost.
- **M4-AC3a:** For the planned self-hosted deployment use actual Vast.ai
  rental charges/rates and measured charged wall time, rather than assumed
  external API token prices. Measure inference p50/p95 latency, GPU activity
  time/utilization where observable, throughput, processing duration, and
  cost per processed event with denominators stated. Compare baseline and
  optimized pipelines on identical held-out inputs and include idle/model-
  load/training/network/storage costs or explicitly delimit them. Distinguish
  observed rental cost from an allocated per-run estimate; a fixed rental
  window may have equal actual cost even if optimized inference is faster.
  No GPU/rental measurement yet means **UNVERIFIED**, not zero cost.
- **M4-AC4:** Clean startup/deployment checks, unseen supported-video smoke,
  real authorized actions, evidence/logs, secrets/privacy/license checks,
  and the official frontend/backend/repository/pitch deliverables pass.
  README, UI, deck, and metrics distinguish implemented, measured, pending,
  and untested behavior. Optional robot hardware is not a demo dependency.
  Public deployment follows the operator's explicit approval requirement.

## 17. Scope and fallback policy

If time is limited:

1. Protect the working video → event → action pipeline.
2. Protect support for new input videos.
3. Protect deployment and reproducibility.
4. Reduce the number of event types if necessary.
5. Use manual labeling as an explicitly disclosed development fallback;
   keep the real Teacher-label acceptance criterion unverified until run.
6. Defer RTSP, Telegram, GVERA, and advanced retinal-model replication.
   Keep the configurable context-plus-ROI foveated experiment in the
   M2–M3 plan, with unexecuted comparisons marked UNVERIFIED.
7. Never replace real functionality with an undisclosed mock.

## 18. Business and pilot

Target customer: service center operator or dealership operations manager.

Proposed value:

- Reduce manual monitoring.
- Improve event documentation.
- Provide auditable process timelines.
- Lower recurring AI inference cost through AutoLearn.

Three-month pilot:

**Month 1:** Deploy on one consented service bay, collect baseline data, define target events and evaluate reliability.

**Month 2:** Expand to several workflows, compare automated events against manual records, and measure false positives and operating cost.

**Month 3:** Run AutoLearn on a repeated visual task, evaluate Student against Teacher, measure operational impact, and decide whether expansion is justified.

Pricing is a hypothesis, not a verified market fact. Validate willingness to pay during pilot interviews.

## 19. Pitch requirements

Seven-minute pitch:

- 0:00–0:45 — Problem.
- 0:45–1:30 — Product and target customer.
- 1:30–3:30 — Live end-to-end demo.
- 3:30–4:30 — AutoLearn and technical differentiation.
- 4:30–5:15 — Real evaluation results.
- 5:15–6:00 — Business model and three-month pilot.
- 6:00–7:00 — Impact, limitations, and conclusion.

The live demo must remain functional without GVERA.

## 20. Definition of Done

A feature is complete only when:

- Code is committed.
- Tests relevant to the feature pass.
- API/UI integration works.
- Errors are handled.
- No secret is committed.
- Documentation reflects reality.
- The feature has been demonstrated on a real input.
- Any limitations are explicitly recorded.

A successful build without a working end-to-end demo is not sufficient.

## 21. Research references and attribution

**Andrey Gizdov, Shimon Ullman, Daniel Harari (CVPR 2025).**
*Seeing More with Less: Human-like Representations in Vision Models*,
pp. 4408–4417.
[CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/33296) ·
[CVF proceedings entry](https://openaccess.thecvf.com/content/CVPR2025/html/Gizdov_Seeing_More_with_Less_Human-like_Representations_in_Vision_Models_CVPR_2025_paper.html) ·
[Open-access paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gizdov_Seeing_More_with_Less_Human-like_Representations_in_Vision_Models_CVPR_2025_paper.pdf).

The work investigates variable-resolution visual inputs inspired by human
foveation. VisionFlow uses it to motivate retaining important-region detail
alongside coarse scene context. Our router experiment is a separate design,
not a reproduced model architecture or measured replication. Neither the
paper's accuracy/pixel-budget results nor its gains may be presented as
VisionFlow measurements. All project quality/latency/cost claims require the
matched held-out experiments defined above.
