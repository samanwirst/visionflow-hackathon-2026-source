# Argus — WIUT Hackathon Final 2026

## Run Argus from zero

The production entry point is the standalone `app.live` backend, which also
serves the unified Argus frontend at `/`. Each browser gets its own opaque
session and private room. There is no in-app PIN, operator/judge choice or
shared workspace. Select a prepared video or upload a new MP4, inspect real
sampled images, describe a supported rule, review its exact JSON and explicitly
approve it before geometric processing. Local notifications contain actual
event evidence; scene descriptions alone never trigger events.

M2/M3 remain **BLOCKED** and Student activation is **OFF**. The earlier exact
review of `f49bf098528fcdd0f42024cef30013c3791b6627` remains
**CHANGES_REQUIRED**. The owner explicitly waived further independent review
for the final release and authorized fast checks; this is not an independent
PASS. The protected fallback is `723a4a73acf33e31843c59d86f942cd5247273c0`.

### 1. Install the source and system dependencies

Ubuntu 24.04/Python 3.12 is supported by the deployment setup. Python 3.14 was
also used in local checks. Run these commands from a fresh checkout:

```sh
git clone https://github.com/samanwirst/visionflow-hackathon-2026-source.git argus
cd argus
sudo apt-get update
sudo apt-get install -y python3-venv ffmpeg curl
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
(cd services/api && ../../.venv/bin/python -m pip install -r requirements.txt)
.venv/bin/python -m pip install ultralytics==8.4.175
```

`ffmpeg` supplies both `ffmpeg` and `ffprobe`; both are required for bounded,
validated video decoding. Python requirements use version ranges rather than a
complete lock, so record resolved versions when reproducing elsewhere. The
published release archive can alternatively be checksum-verified and extracted;
its `visionflow-source/` directory is the same application root.

### 2. Download and verify the detector

```sh
mkdir -p private-models private-runtime
chmod 700 private-models private-runtime
curl --fail --location --proto '=https' --tlsv1.2 \
  https://github.com/samanwirst/visionflow-hackathon-2026-source/releases/download/hackathon-2026-ui1/yolo11n.pt \
  --output private-models/yolo11n.pt
printf '%s  %s\n' 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 \
  private-models/yolo11n.pt | sha256sum -c -
```

The weight file retains its Ultralytics upstream license; it is not a model
trained by this team. Preserve its notices and the accompanying upstream source.

### 3. Prepare a private catalog, configure API access and start locally

The current UI bootstrap requires **at least one valid, owner-authorized MP4**
in its private catalog. Prepare one before launching; the actual environment
variable and lookup filename used by this version are shown below:

```sh
mkdir -p private-demo-videos
chmod 700 private-demo-videos
cp /absolute/path/to/your-authorized-valid-clip.mp4 private-demo-videos/Camera1-960x540-cfr24.mp4
```

Use a supported constant-frame-rate MP4 within the decoder's size, duration and
frame limits (at most 64 MiB, 120 seconds and 5,000 frames). The filename is a
catalog lookup key; it does not bypass content validation. Uploaded videos are
not source assets. After this catalog admits the workspace, **Upload MP4** can
process a different supported input.

Copy `.env.example` to `.env` and run `chmod 600 .env`. Configure your own
`OPENAI_API_KEY` in that ignored server-only file. Do not paste credentials into
frontend configuration, URLs, logs or commits. Load it from the application root:

```sh
cp .env.example .env
chmod 600 .env
# Edit .env privately to configure OPENAI_API_KEY before continuing.
set -a
. ./.env
set +a
export VISIONFLOW_DATA_DIR="$PWD/private-runtime"
export VISIONFLOW_DEMO_VIDEO_DIR="$PWD/private-demo-videos"
export DETECTOR_MODEL="$PWD/private-models/yolo11n.pt"
export DETECTOR_DEVICE=cpu
export VISIONFLOW_THREADS=2
export VISIONFLOW_DEMO_SAFE=true
export VISIONFLOW_DEMO_BOOTSTRAP_ENABLED=true
export VISIONFLOW_LIVE_ENABLE_CAMERA=false
export VISIONFLOW_LIVE_PLANNER_PROVIDER=openai
export VISIONFLOW_LIVE_PLANNER_MODEL=gpt-4.1-mini-2025-04-14
export VISIONFLOW_LIVE_SCENE_ENABLED=true
export VISIONFLOW_LIVE_OPENAI_MAX_CALLS=50
export VISIONFLOW_LIVE_MAX_VIDEO_JOBS=1
export VLM_BASE_URL=
export LIVE_TEACHER_BASE_URL=
export VISIONFLOW_LIVE_ALLOWED_ORIGINS=http://127.0.0.1:19113
export VISIONFLOW_LIVE_ALLOWED_HOSTS=127.0.0.1:19113
export VISIONFLOW_LIVE_TRUSTED_PROXY_HOPS=0
export VISIONFLOW_LIVE_FORWARDED_ALLOW_IPS=127.0.0.1,::1
export VISIONFLOW_LIVE_INSECURE_COOKIE=true
cd services/api
../../.venv/bin/python -m app.live --host 127.0.0.1 --port 19113
```

Open **http://127.0.0.1:19113/**. The final command serves frontend and backend;
no separate Next.js process is required. Stop it with Ctrl-C. The insecure-cookie
setting above is exclusively for loopback HTTP. Missing credentials, provider
errors and exhausted budgets return unavailable without fabricated results;
there is no public subscription-CLI fallback. The shared, durable OpenAI attempt
ledger lives in `VISIONFLOW_DATA_DIR` and caps planning plus scene requests.

To populate all five selections, add authorized, supported files named
`Camera2-960x540-cfr24.mp4` through `Camera5-960x540-cfr24.mp4` to the same directory.
At least one configured input is required: an empty or unset catalog leaves the
current workspace unavailable, including its upload control. These private videos
are not distributed as application source; never invent missing catalog entries.
Scene inspection samples at most three real 512-pixel images and reports source
and frame hashes, timestamps, model identity and token usage. Basic unfamiliar
object descriptions do not establish precise box tracking, aisle blockage,
removal, bottle opening or other unvalidated temporal actions.

### 4. Deploy securely

Run the same backend on loopback behind an HTTPS proxy/tunnel and the outer
access-control gate. Set `VISIONFLOW_LIVE_INSECURE_COOKIE=false`, exact HTTPS
`VISIONFLOW_LIVE_ALLOWED_ORIGINS` and matching `VISIONFLOW_LIVE_ALLOWED_HOSTS`.
Keep trusted proxy hops at zero and forwarding peers restricted to loopback;
never expose the backend port, legacy `app.main`, camera ingest or WebSockets.
Different visitors retain separate session cookies, rooms, prompts and workflows.

Inference runs wherever this backend is deployed: YOLO runs on its configured
CPU/GPU; planning and sampled-image understanding use the authorized OpenAI API.
If a VPS receives a provider region/access refusal, it cannot supply those
capabilities. Use a deployment location and provider account explicitly supported
by the provider, or leave inference unavailable; do not route around restrictions
or claim a successful local test proves VPS inference. Uploaded video, evidence,
API credentials and runtime data remain private and excluded from source.

### 5. Verify your installation

```sh
make check
node tests/frontend/argus_single_visitor.cjs
make smoke
```

`make check` runs backend tests and the separate historical Next.js build (Node
22 and `npm ci` in `apps/web` are needed for that build). `make smoke` executes
real YOLO; it is not proof of the complete public browser workflow. A new
installation still needs an actual browser upload, novel prompt, exact approval,
real processing, evidence and visitor-isolation test. Prepared commands are not
executed-test evidence. Foveation, temporal Teacher quality and savings remain
**UNVERIFIED** until measured. See [docs/LICENSES.md](docs/LICENSES.md).

## License and corresponding source

Original team-authored software in the current hackathon release is licensed
under **AGPL-3.0-only**. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the recorded
[owner consent attestation](.agents/CONTRIBUTOR_LICENSE_CONSENT.md). Third-party
software, weights, data and footage retain their separate upstream licenses.

The corresponding-source download for the final deployed version is:

[https://github.com/samanwirst/visionflow-hackathon-2026-source/releases/download/hackathon-2026-ui1/visionflow-source.tar.gz](https://github.com/samanwirst/visionflow-hackathon-2026-source/releases/download/hackathon-2026-ui1/visionflow-source.tar.gz)

Before public service launch the controller must publish and anonymously verify the exact accepted
snapshot at that URL, including build/run source, notices and dependency
information, with a manifest identifying its frozen commit. The private origin
and Git history remain private. No secrets, user uploads or private experiments
belong in the release. A source link is not a claim of technical acceptance or
complete third-party licensing verification.

Visual AI automation for automotive operations: upload a recorded video, draw a
zone, configure a supported geometric condition, review its JSON and approve
real events, evidence and local notifications. Track: Computer Vision.

The product's innovation is **cost-efficient visual analytics inspired by
OpenVector**: cheap detection and tracking run continuously at a configured
sampling rate; temporal events select frames or object ROIs for expensive
visual reasoning. Deduplication and versioned caches limit repeated teacher
calls. AutoLearn transfers verified teacher labels to a lightweight student
over frozen visual embeddings, with held-out evaluation, shadow comparison,
manual activation, fallback, and rollback.

These are implementation requirements, not a claim that all stages are
already available. Cost savings require an actual paired baseline experiment.

The official task is in [docs/source/README.md](docs/source/README.md);
product requirements in [docs/PRD.md](docs/PRD.md);
architecture in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Accepted geometric engine (M1)

Working end-to-end pipeline: **MP4 upload → YOLO11n detection + ByteTrack
tracking → polygon-zone temporal events (enter / exit / dwell / count) →
SQLite persistence + evidence JPEG → authorized local action
(create_notification) → web UI** (upload, canvas zone drawing, workflow
builder, event timeline).

The exceptional M1 recovery cycle resolved the four P1 defects from the
`cc59523` review: dispatch-after-disable is now serialized and suppressed
(M1-010); a truncated video is reported as a failed source (M1-013); and
arbitrary outbound webhooks were **removed** (M1-011/M1-012) — the SSRF/
DNS-rebinding guard could not safely protect the delivery connection without a
risky custom egress transport, so only the local `create_notification` action
ships. See [.agents/reviews/M1.md](.agents/reviews/M1.md) and
[docs/TRACEABILITY.md](docs/TRACEABILITY.md).

M2's experimental VLM routing/compiler implementation remains BLOCKED.
The isolated ten-hour experiment has executed actual private GPU Teacher
labeling and applied the user's verification of 24 exact synthetic images.
Twelve biased Teacher proposals needed correction. A real Student was fitted,
saved, reloaded and compared in shadow on six held-out synthetic images:
Student 6/6 correct, Teacher 3/6. The independent review of that experimental
snapshot returned CHANGES_REQUIRED. The final correction at `ff48ca9` passed249
tests/build/realYOLO but independently failed QA with6P1; the experiment is
BLOCKED, closed and not integrated. See the [final review](.agents/reviews/AUTOLEARN-TEN-HOUR-FINAL.md)
and [executed pilot evidence](.agents/autolearn-evidence/synthetic-5938b6d/VALIDATION.md)
for exact inputs, timings and limits. No model was activated; this is not M3
acceptance. The verified fallback
is protected on `feature/demo-safe`.
The [architecture audit](docs/VISUAL_ANALYTICS_AUDIT.md) and
[milestone acceptance criteria](docs/PRD.md#16-acceptance-criteria) define
the remaining work. No cost savings or natural-video quality result is established.

Current experimental verdicts and limits are in [the execution status](docs/TEN_HOUR_STATUS.md).

The delivery candidate uses [a private reproducible container deployment](deploy/README.md)
and an optional authenticated same-origin gateway. [Executed evidence](.agents/delivery-evidence/private-749d27b/VALIDATION.md)
records a real YOLO event, evidence and notification in the fresh unprivileged
container. No public tunnel is active. Use the [seven-minute defence script](docs/DEFENSE_SCRIPT.md)
and [actual recorded fallback](docs/presentation/VisionFlow-fallback-demo.mp4).

## Demo-safe build (current deliverable)

The branch `feature/demo-safe` ships a polished, demonstrable **geometric**
control platform for recorded video with the experimental M2 paths locked off
in the backend. See the [Demo Guide](docs/DEMO_GUIDE.md) for the exact startup,
data provenance, the seven-minute script, and the supported/disabled/unverified
matrix.

The [independent demo evidence](.agents/demo-evidence/99e8670/VALIDATION.md)
records 195 passing tests, the production build, real YOLO smoke and the actual
desktop approval/event/evidence journey. Use a desktop viewport of at least
1024px; a known phone layout issue is documented there.

Demo-safe mode (`VISIONFLOW_DEMO_SAFE=true`, the default) is enforced
server-side, not just in the UI:

- the natural-language compiler is **never invoked** (M2-001 remains unresolved);
- Teacher-routed **visual-predicate** workflows cannot be created, enabled, or
  processed — zero Teacher admission even with a VLM endpoint configured or a
  legacy visual row enabled, and they are refused rather than silently coerced
  into geometric rules;
- VLM stats report an explicit disabled state **without** instantiating a Teacher.

**M1 is accepted; M2 remains BLOCKED.** This demo is not an M2 acceptance, an M3
AutoLearn implementation, or a measured-cost result. The fallback does not expose
Teacher or Student features. Foveated input and GPU cost savings remain UNVERIFIED;
no savings percentage is claimed.
The guided UI builds a rule, shows the exact server-validated JSON, and requires
a deliberate approval before the real detector → tracker → temporal-rule
pipeline is armed.

## Historical geometric fallback startup

The previous `app.main` plus Next.js workflow is the protected geometric fallback,
not the production entry point. Keep it private and use the isolated setup in
[deploy/README.md](deploy/README.md) or [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md).
The unified production frontend/backend starts with `python -m app.live` as above.

## Tests

```sh
# unit + integration (fast; integration uses a disclosed test-stub detector)
services/api/.venv/bin/python -m pytest tests/ -q

# end-to-end smoke with REAL YOLO inference (starts its own server)
services/api/.venv/bin/python scripts/smoke_e2e.py

# demo video from real photographic content (Ultralytics sample photo)
services/api/.venv/bin/python scripts/make_demo_video.py data/demo/demo_bus.mp4

# repository structure check
make check
```

Configuration variables: see [.env.example](.env.example).
Licenses and data provenance: [docs/LICENSES.md](docs/LICENSES.md).

## Structure

```text
apps/web/              Next.js UI (M1: single-page flow)
services/api/          FastAPI backend, SQLite, pipeline worker
packages/vision/       detection, tracking, zones, temporal event engine
packages/workflows/    (M2) NL → DSL compilation
packages/autolearn/    (M3) teacher/student specialization
packages/integrations/ (M2+) webhook/CRM/GVERA adapters
tests/                 unit + integration
scripts/               structure check, demo video builder, e2e smoke
docs/                  task source, PRD, architecture, traceability, licenses
.agents/               agent protocol, state, handoffs, reviews
```

## Agents

Instructions: [AGENTS.md](AGENTS.md) · Protocol: [.agents/PROTOCOL.md](.agents/PROTOCOL.md) ·
State: [.agents/STATE.md](.agents/STATE.md) · Handoff: [.agents/HANDOFF.md](.agents/HANDOFF.md)

The sequential Claude/Codex controller is documented in
[.agents/AUTOPILOT.md](.agents/AUTOPILOT.md). With both CLIs authenticated:

```sh
python3 scripts/orchestrate.py --dry-run
make autopilot
python3 scripts/orchestrate.py --resume
python3 scripts/orchestrate.py --stop
```

`make check` runs the structure check, tests and frontend build. `make smoke`
runs real YOLO inference. The controller reviews frozen commits, saves local
checkpoints, caps fix cycles, pushes validated feature work and never merges
into main or publishes without approval.
