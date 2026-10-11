# Self-hosted Teacher and input-router plan — M2–M3

Status: architecture/acceptance plan. M1 passed at `390b337`; M2 is active.
No instance has been rented or connected by the controller. Real Teacher
labels, strategy quality, GPU performance, and savings are **UNVERIFIED**.

## Interfaces and boundaries

`VisualInputStrategy.prepare(frame, regions, task)` produces encoded image(s),
dimensions, region-to-scene mappings, preprocessing version, requested/effective
strategy and fallback reason. Strategy choices are `full_frame`, `roi_crop`,
and `foveated`. Regions come from YOLO boxes, tracked objects, zones and
temporal candidates; expensive inference remains event-triggered.

- Full frame retains the whole scene at a documented resolution cap.
- ROI crop retains detailed target pixels and discards outside context.
- Foveated input retains high-resolution target region(s) and a low-resolution
  whole-scene image, with explicit spatial mapping. A bounded multi-image
  request is an MVP approximation; it is not retinal-model replication.

Use full-frame fallback when context is required, ROIs are missing/clipped/
uncertain, small-object visibility is at risk, or the server cannot consume
the chosen representation. Never silently call an ROI crop "foveated".
Record selection/fallback and include the effective strategy, encoded input
identity, model/prompt and preprocessing versions in cache keys.

`TeacherProvider.classify(task, prepared_input)` targets an operator-set
OpenAI-compatible `/v1/chat/completions` endpoint. Send image bytes as bounded
data URLs rather than asking the server to fetch arbitrary user URLs. Request
schema-constrained output where supported and always validate locally:
finite bounded confidence, permitted labels, required fields, no unexpected
fields, explicit unknown/error. A JSON-looking response is not validation.

Configuration is server-only `VLM_BASE_URL`, `VLM_MODEL`, optional `VLM_API_KEY`.
Operator configuration is separate from user workflow input. Set explicit
connect/read/overall limits, semaphore/queue limits, bounded retries/backoff,
maximum images/bytes/tokens and budgets. Retries consume actual attempt
budgets and remain observable; auth/schema failures fail visibly. Missing
server configuration returns unavailable and makes no positive classification.
Manual verification remains distinct from a Teacher-generated label.

## Secure connection when the operator's GPU is available

The model example is `Qwen/Qwen2.5-VL-7B-Instruct`. Pin and record the chosen
model revision and vLLM/container version. Actual GPU memory requirements and
performance depend on model/quantization/context/image limits and must be
validated on the rented hardware. The operator selects/rents the instance;
this plan authorizes no purchase or account changes.

Run inference on remote loopback and do not publish its HTTP port. Obtain
the assigned SSH host/port/user from the operator's instance connection
details, verify its host key, and use your own SSH identity. Bind the local
forward to loopback too. Example commands are templates, not executed setup:

Vast.ai documents key-based connections and local forwarding in its
[official SSH guide](https://docs.vast.ai/guides/instances/connect/ssh).
Use instance-specific connection details; never copy an example IP/port.

```sh
# Remote GPU shell; VLLM_API_KEY is set securely out of source control.
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":3,"video":0}'

# Local terminal; replace connection placeholders with instance details.
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18080:127.0.0.1:8000 \
  -p SSH_PORT SSH_USER@SSH_HOST
```

Ignored environment configuration for the VisionFlow backend:

```dotenv
VLM_BASE_URL=http://127.0.0.1:18080/v1
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VLM_API_KEY=
```

The optional key may be omitted only for a private authenticated tunnel or
equivalent private boundary. If `VLLM_API_KEY` is used remotely, configure the
matching value locally without printing it. The backend's configuration
loading must be documented; merely writing `.env` is not proof it is loaded.
For Docker, loopback belongs to the container: place the tunnel in the same
network namespace or configure a private reachable tunnel/gateway, never a
public unauthenticated port. A TLS gateway must authenticate all exposed
paths, restrict access, and protect metrics/admin paths as well.

Current vLLM documentation warns that API-key checks cover selected API
prefixes, with other routes not protected. An API key alone is not a safe
public-server boundary. [vLLM server documentation](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).
The sample model/serving interface is documented in the
[vLLM Qwen2.5-VL recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen2.5-VL.html).
Loopback forwarding follows the [OpenSSH manual](https://man.openbsd.org/ssh.1).

Once connected, verify availability/model identity and one consent-safe real
image response; exercise malformed JSON, timeout, authentication failure,
request limits, and missing-token fields. Preserve safe timing/usage/error
metadata without keys, private endpoints or image bytes in public reports.
Do not invent a successful health check or Teacher label before this runs.

## Held-out experiment and actual rental accounting

M3 uses one immutable session-disjoint held-out video/event manifest, manual
ground truth, fixed model/prompt, selected candidates, and preprocessing
parameters to compare full frame, ROI and foveated inputs. Keep threshold
tuning outside the test set. Include spatial/context-dependent conditions,
small objects, multiple objects, clipped/incorrect boxes and tracking loss.
For each strategy save raw predictions, failures/unknowns, accuracy/recall/F1,
sample counts, input tokens if the server reports them, encoded size, actual
timing, and fallback/coverage. Isolate preprocessing effects with the same
candidate schedule and separately evaluate full pipeline event recall.

M4 compares VLM-per-sampled-frame baseline, event-routed/cached Teacher, and
Student/hybrid pipelines on identical held-out inputs. Measure wall duration,
p50/p95 inference/event latency, throughput, actual request/attempt counts,
cache conditions, GPU activity time/utilization when observable, and actual
Vast.ai rental charges with currency and billed interval. GPU activity time
is not the same as charged rental wall time. Include or clearly delimit
model loading, idle time, training, storage/network charges and local compute.

Allocated cost/event = attributed actual rental expense / stated processed
event count; report failed attempts and true/false event counts too. If rates
are used to allocate an experiment's time, distinguish that estimate from
the bill actually paid. A shorter optimized run may leave a fixed rented
window's actual charge unchanged. Pixel count, token count and fewer requests
are measurements, not substitutes for measured monetary savings.

Unrun experiments, missing usage telemetry, unavailable GPU/billing records,
and paper gains remain **UNVERIFIED**. No percentage is reported without a
comparable positive measured baseline.

## Research attribution

Gizdov, Ullman and Harari, *Seeing More with Less: Human-like Representations
in Vision Models*, CVPR 2025, pp. 4408–4417.
[Poster](https://cvpr.thecvf.com/virtual/2025/poster/33296) ·
[CVF paper entry](https://openaccess.thecvf.com/content/CVPR2025/html/Gizdov_Seeing_More_with_Less_Human-like_Representations_in_Vision_Models_CVPR_2025_paper.html).
This motivates the separate context-plus-ROI experiment; no reported paper
gain is a VisionFlow result or a linear pixel-to-token/cost guarantee.
