# VisionFlow AI — Demo Guide (demo-safe build)

This guide covers the `feature/demo-safe` deliverable: a polished, demonstrable
**geometric** visual-analytics control platform for recorded video, with the
experimental M2 paths locked off in the backend.

> **Status honesty.** M1 is accepted; **M2 remains BLOCKED**. This demo is *not*
> an M2 acceptance, an M3 AutoLearn implementation, or a measured-cost result.
> Teacher VLM, Student, foveated input and any GPU savings are **UNVERIFIED** and
> no savings percentage is claimed. The zero Teacher calls shown reflect the
> geometric-only path, not a paired cost benchmark.

## 1. What runs vs. what is disabled

| Capability | Status in this demo |
| --- | --- |
| MP4 upload → frame sampling → **YOLO11n detection + ByteTrack** | **Supported (real)** |
| Polygon zones; temporal rules: enter / exit / **dwell ≥ s** / **count > n** | **Supported (real)** |
| Event persistence, evidence JPEG, **local `create_notification`** action | **Supported (real)** |
| Guided rule build → **server-canonical JSON review** → deliberate approval | **Supported (real)** |
| Edit / disable / re-arm with renewed JSON review | **Supported (real)** |
| Natural-language → DSL compiler | **Disabled** (M2-001 unresolved; never invoked) |
| Teacher-routed **visual-predicate** workflows (create/enable/process) | **Disabled** (zero Teacher admission) |
| Outbound webhook / external delivery | **Removed in M1** (M1-011/M1-012) |
| Live camera / RTSP | **Unavailable** |
| AutoLearn (Teacher → verified dataset → Student), foveated input, GPU savings | **Planned / UNVERIFIED** |

Rule-label semantics match the real engine exactly: **count** fires when the
in-zone count **strictly exceeds** the threshold (`count > n`); **dwell** fires
when the object has been in the zone for **≥** the configured seconds.

## 2. Demo-safe enforcement (backend, not just UI)

With `VISIONFLOW_DEMO_SAFE=true` (default):

- The guard is **fail-closed**: it defaults ON and only an explicitly recognized
  false value (`0`/`false`/`no`/`off`) turns it off. A typo (e.g. `treu`) or any
  other unrecognized value keeps the guard **ON**.
- `GET /api/health` reports `demo_safe: true|false`. The frontend verifies this
  and **refuses** uploads, rule arming and runs if the connected API is not in
  safe mode, or if the capability cannot be verified (e.g. an older backend).
- `POST /api/workflows/compile` returns a visible unavailable error **without
  invoking** the NL compiler. The module is imported at startup but its grammar
  is never called, so no prompt reaches it.
- `POST /api/workflows` and `POST /api/workflows/{id}/enable` **reject**
  (`403`) any workflow carrying a `visual_predicate`.
- Processing **skips** any enabled `visual_predicate` workflow (even a legacy row
  or one enabled before the demo) — it is refused, never coerced to geometric —
  so no Teacher call can be admitted, even with `VLM_BASE_URL`/`VLM_MODEL` set.
- `GET /api/metrics` and `GET /api/vlm/stats` report an explicit
  `disabled_demo_safe` state **without instantiating** the router or a Teacher.

Set `VISIONFLOW_DEMO_SAFE=false` only to exercise the historical M2 experimental
paths; the original M2 tests opt out with an explicit fixture.

## 3. Local startup (loopback only, isolated data)

No credentials and no GPU are required. Run everything on loopback with an
isolated data directory so the demo never touches real data, and **without**
the autopilot/orchestrator. Ports 8000 and 3000 must be free for these commands.

Use a desktop browser at least **1024px wide**; 1024/1280/1440px were checked.
At 390px, selecting a real source still expands the zone workspace horizontally
(known P2). Phone readiness is not claimed. The delivery branch now includes an
[editable deck and real recorded backup](presentation/README.md), a
[current seven-minute defence script](DEFENSE_SCRIPT.md), and
[executed private deployment evidence](../.agents/delivery-evidence/private-749d27b/VALIDATION.md).
Public access remains unexecuted under the user's restriction; see the
[prepared publication proposal](PUBLICATION_PROPOSAL.md).

### Backend

```sh
# from the repo root
# preflight also downloads/caches the weights used below
make smoke
export VISIONFLOW_DATA_DIR="$(mktemp -d /tmp/visionflow-demo.XXXX)"
export VISIONFLOW_DEMO_SAFE=true
export DETECTOR_MODEL="$PWD/services/api/yolo11n.pt"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
services/api/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 \
  --app-dir services/api
```

The preflight smoke downloads YOLO11n weights automatically if missing. Keep
the cached model ready before the timed presentation.
API docs: http://127.0.0.1:8000/docs

### Frontend

```sh
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm --prefix apps/web run dev
# open http://localhost:3000
```

### Demo clip (bus-only crop, synthetic motion)

```sh
services/api/.venv/bin/python scripts/make_demo_safe_video.py \
  data/demo/demo_bus_private_crop.mp4
```

This clip crops the Ultralytics sample photo to the **bus only** (crop
`[350,220,755,665]`, excluding the foreground pedestrians) and sweeps that crop
left→right across a flat canvas. **Detection is genuine YOLO inference on real
image content; only the motion is synthetic** — never the prediction. The
foreground pedestrians are excluded because their subject rights are unverified
(see [LICENSES.md](LICENSES.md)); the historical M1 smoke clip
(`scripts/make_demo_video.py` → `demo_bus.mp4`) is left unchanged and not used
here. Upload the generated file through the UI (or use any MP4 a judge provides).
There is no invented camera feed and no hardcoded prediction.

## 4. Seven-minute demo script (exact timing)

The live demo defines the rule **before** running anything — a run requires an
approved, enabled geometric rule on the selected source, and the UI guides you
there if one is missing.

Name **Startup mode** on the first slide. Suggested opening: “VisionFlow is a
recorded-video control platform for service-centre managers. Today we prove
zone events, evidence and approved local notifications; the learning pipeline
is planned.” The pilot below is a proposal, not a partner commitment.

| Time | Beat | On screen |
| --- | --- | --- |
| 0:00–0:45 (45s) | **Startup mode; problem** — manual service-bay monitoring is slow and undocumented. | First slide naming the mode, then dashboard; "Demo-safe · verified" + capability chips. |
| 0:45–1:30 (45s) | **Product & customer** — recorded-video control platform for service ops. | Lede + architecture note. |
| 1:30–2:00 (30s) | **Upload / select** the demo clip (`demo_bus_private_crop.mp4`). | Video sources card. |
| 2:00–2:40 (40s) | **Draw the zone** — a "Service Bay A" polygon on the first frame. | Zone workspace. |
| 2:40–3:20 (40s) | **Build the rule** explicitly — `bus · enters zone`. | Build & review a rule. |
| 3:20–3:50 (30s) | **Review the canonical JSON** the server validated, then **Approve & activate**. | Review pane → armed rule. |
| 3:50–4:40 (50s) | **Run once** — real YOLO → event + evidence frame + one local notification appear. | Event timeline, Notifications. |
| 4:40–5:10 (30s) | **Edit / disable / re-enable** — Edit disables the prior rule (events/evidence preserved) and creates a new reviewed version; disable then **Review & arm** re-activates from the stored JSON. | Rules table. |
| 5:10–6:00 (50s) | **Technical differentiation** — OpenVector-inspired hierarchy: cheap detection/tracking/temporal rules now; Teacher → verified dataset → Student planned. Honest JSON review + server-side validation; fail-closed demo-safe guard. | Architecture & honest status card. |
| 6:00–6:30 (30s) | **Honest evaluation** — real per-frame inference latency from `make smoke`; events reproducible on video time. No savings % claimed. | Smoke/terminal output. |
| 6:30–7:00 (30s) | **Pilot & limits** — the three-month pilot proposal below; demo-safe boundary; M2 BLOCKED; AutoLearn/Teacher/foveated UNVERIFIED. | This guide (matrix + pilot). |

Run the live beat **once**; do not re-trigger processing on the same clip.

### Proposed three-month service-bay pilot (proposal only — no results claimed)

A proposal to validate the product on a partner's own recorded footage. Numbers
below are **targets to measure during the pilot**, not achieved results:

- **Month 1 — Baseline & geometric rules.** Deploy on recorded clips from one
  service bay; configure zones + enter/exit/dwell/count rules; capture events,
  evidence and local notifications. Measure real per-frame latency and
  operator-confirmed event correctness on the partner's footage.
- **Month 2 — Teacher-assisted labeling (gated).** Only if an approved, secured
  VLM endpoint is available: route selected candidate events for visual
  predicates; build a human-reviewed labeled set. (Disabled in this demo-safe
  build; M2 is BLOCKED.)
- **Month 3 — Student + shadow mode.** Train a lightweight Student on the
  verified set; run shadow-only and compare against held-out labels before any
  activation; document rollback.

No ROI, cost, or savings figures are asserted — any such numbers would be
produced *from the pilot's own measurements*.

## 5. Commands (verify before showing)

```sh
# full check: structure + unit/integration tests + frontend production build
make check

# real-YOLO end-to-end smoke (starts its own server on an isolated data dir)
make smoke

# demo-safe backend regressions only
services/api/.venv/bin/python -m pytest tests/integration/test_demo_safe.py -q
```

Record actual results; do not claim a pass that was not run. See
[the independent validation](../.agents/demo-evidence/99e8670/VALIDATION.md)
for exact commands, receipts, screenshots and the desktop browser journey at
`99e8670`. The implementation history is in
[.agents/DEMO_HANDOFF.md](../.agents/DEMO_HANDOFF.md).

## 6. Pre-show checklist

- [ ] Backend on `127.0.0.1:8000` with `VISIONFLOW_DEMO_SAFE=true` and an
      isolated `VISIONFLOW_DATA_DIR`.
- [ ] Frontend on `127.0.0.1:3000` pointed at the loopback API.
- [ ] `data/demo/demo_bus_private_crop.mp4` generated (or a judge's MP4 ready).
- [ ] Header shows **Demo-safe · verified** (the backend reported `demo_safe:true`).
- [ ] `make check` and `make smoke` pass on the target machine.
- [ ] Desktop viewport is at least 1024px; weights cached before the presentation.
- [ ] No autopilot/orchestrator running; no public deployment.
