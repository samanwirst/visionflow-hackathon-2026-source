# Argus hackathon corresponding source

**Exact frozen implementation source SHA: `f9d74f024d87a56ae66263dc1d9244a1049d9dc8`**

[Download the complete corresponding source archive](https://github.com/samanwirst/visionflow-hackathon-2026-source/releases/download/hackathon-2026/visionflow-source.tar.gz).
The release also contains its SHA256 checksum, the verified unmodified
Ultralytics 8.4.175 source distribution and original license, and unchanged
YOLO11n pretrained weights under their upstream AGPL-3.0 terms.

Original team-authored release software is **AGPL-3.0-only**; see LICENSE and
NOTICE. Upstream software, model weights and footage retain separate licenses.
The source archive includes exact tracked integration/build/test files and
provenance, with declared non-source metadata additions. This is a NEW snapshot
repository and contains no private project Git history, user uploads or secrets.

**M2 and M3 remain BLOCKED. Student activation is OFF.** Teacher temporal
understanding, foveation performance and savings remain unverified; this release
does not certify the complete adaptive pipeline. The owner explicitly waived
further independent review for this release (**OWNER_WAIVED_INDEPENDENT_REVIEW**).
The preceding exact review of `f49bf098528fcdd0f42024cef30013c3791b6627` remains
**CHANGES_REQUIRED**. This release uses the actual owner-authorized fast test
gates recorded in SOURCE_RELEASE.json; it is not an independent PASS verdict.
Owner directive record SHA256: `5b9ee60576f57d31f7df210f893919089bf937e7e5a6ee2365c656654213ecbc`.

## Reproduce the owner-authorized video application

On Ubuntu 24.04 with Python 3.12, download the source archive and checksum above,
verify the checksum, then install the system decoders (`ffmpeg` and `ffprobe`) and Python dependencies:

```sh
sha256sum -c visionflow-source.tar.gz.sha256
tar -xzf visionflow-source.tar.gz
cd visionflow-source
sudo apt-get update
sudo apt-get install -y python3-venv ffmpeg curl
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
(cd services/api && ../../.venv/bin/python -m pip install -r requirements.txt)
.venv/bin/python -m pip install ultralytics==8.4.175
mkdir -p private-models private-runtime
chmod 700 private-models private-runtime
curl --fail --location --proto '=https' --tlsv1.2   'https://github.com/samanwirst/visionflow-hackathon-2026-source/releases/download/hackathon-2026/yolo11n.pt'   --output private-models/yolo11n.pt
printf '%s  %s\n' '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'   'private-models/yolo11n.pt' | sha256sum -c -
```

Create an ignored, server-private `.env` containing your own `OPENAI_API_KEY` and
`VISIONFLOW_DEMO_PIN`; run `chmod 600 .env`. Do not put credentials in URLs,
frontend configuration, shell command arguments or source commits. The example
file contains no working credentials. Load it and launch only the recorded-video
`app.live` application, which serves both its frontend and backend:

```sh
set -a
. ./.env
set +a
export VISIONFLOW_DATA_DIR="$PWD/private-runtime"
export DETECTOR_MODEL="$PWD/private-models/yolo11n.pt"
export DETECTOR_DEVICE=cpu
export VISIONFLOW_THREADS=2
export VISIONFLOW_DEMO_SAFE=true
export VISIONFLOW_LIVE_ENABLE_CAMERA=false
export VISIONFLOW_LIVE_INSECURE_COOKIE=false
export VISIONFLOW_LIVE_PLANNER_PROVIDER=openai
export VISIONFLOW_LIVE_PLANNER_MODEL=gpt-4.1-mini-2025-04-14
export VISIONFLOW_LIVE_SCENE_ENABLED=true
export VISIONFLOW_LIVE_OPENAI_MAX_CALLS=50
export VISIONFLOW_LIVE_MAX_VIDEO_JOBS=1
export LIVE_TEACHER_BASE_URL=
export VLM_BASE_URL=
export VISIONFLOW_LIVE_ALLOWED_ORIGINS=https://argus.example.org
export VISIONFLOW_LIVE_ALLOWED_HOSTS=argus.example.org
export VISIONFLOW_LIVE_TRUSTED_PROXY_HOPS=0
export VISIONFLOW_LIVE_FORWARDED_ALLOW_IPS=127.0.0.1,::1
cd services/api
../../.venv/bin/python -m app.live --host 127.0.0.1 --port 19113
```

Replace the example HTTPS origin/host with your exact deployment hostname.
Place a trusted HTTPS reverse proxy/tunnel on the same host in front of the
loopback backend; preserve the external Host and Origin, and forward to
`127.0.0.1:19113`. Do not expose that backend port publicly, allow wildcard
origins/proxies, mount the legacy `app.main`, or enable camera/WebSocket ingest.
Secure cookies require HTTPS. Visit `/` for the Argus frontend, `/upload` for the
operator and `/judges` for task review/approval. The PIN grants role access; the
operator shares a private, one-use pairing code. An uploaded video is geometrically processed only after explicit human approval
of its exact generated workflow. Scene
inspection is nonactioning and does not prove object tracking or temporal
understanding. Missing API credentials return unavailable, without fake results.

The published source includes the separate historical Next.js frontend; its
build can be reproduced with Node.js 22 using `cd apps/web`, `npm ci` and
`npm run build`. It is not the frontend served by this recorded-video application.
Do not expose blocked M2/M3 paths. The recorded-video application needs no separate
frontend process. Dependencies have version ranges; record resolved versions
when reproducing on another machine. The exact detector asset is hash-checked
above; upstream source/license and weight notices accompany the release.

See SOURCE_RELEASE.json for exact source/asset identity and docs/LICENSES.md for
third-party terms. The source is freely downloadable without login. No warranty
is provided. These are reproducible launch instructions, not evidence that a
new operator deployment has executed its acceptance tests.
