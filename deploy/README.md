# Private deployment check

This package binds host ports to loopback. It does not publish an unauthenticated
API. The accepted geometric demo remains the fallback; Teacher workflow actions
are disabled. Experimental AutoLearn can appear only after its reviewed commit
is explicitly integrated. Missing Teacher configuration is an honest unavailable
provider, not fabricated labels.

From the selected delivery checkout:

```bash
docker compose -f deploy/compose.yaml build
docker compose -f deploy/compose.yaml up -d
curl --fail http://127.0.0.1:18085/api/health
```

Open `http://127.0.0.1:13085`. Upload a permitted MP4, draw a polygon, save a
geometric rule, inspect the exact JSON, approve it, run analysis, and inspect
the event evidence and local notification. Editing and re-enabling require
review again. Uploaded files, evidence and database persist in `demo-data`.

YOLO weights must exist at `/app/data/models/yolo11n.pt`. Provision the actual
model from the tested local checkout without committing its binary:

```bash
docker compose -f deploy/compose.yaml exec api mkdir -p /app/data/models
docker compose -f deploy/compose.yaml cp services/api/yolo11n.pt api:/app/data/models/yolo11n.pt
```

An encoder used by an integrated Student experiment may require its real
pretrained weights to be downloaded once into `model-cache`. A missing artifact
must remain an explicit error. No synthetic success is acceptable.

For another machine, forward these private ports through authenticated SSH:

```bash
ssh -N -L 13085:127.0.0.1:13085 -L 18085:127.0.0.1:18085 user@private-host
```

The browser API URL is fixed at build time, and both forwarded local ports must
be available. No server credential belongs in frontend build arguments. A
public release requires an authenticated TLS boundary and separate authorization
for the concrete prepared deployment.

Stop only this compose project:

```bash
docker compose -f deploy/compose.yaml down
```

Do not add `--volumes` when preserving evidence or trained artifacts. The actual
build, startup and inference results are recorded separately; this document
alone is not deployment verification.

## Authenticated gateway candidate

The optional `compose.gateway.yaml` serves frontend, API and evidence under one
origin, with HTTP Basic authentication on every path and a 25 MiB upload limit.
Keep both the password and its hash in ignored runtime storage. Configure only
the hash-file path as `VISIONFLOW_DEMO_AUTH_FILE`. Do not commit either file.

```bash
docker compose -f deploy/compose.yaml -f deploy/compose.gateway.yaml build web
docker compose -f deploy/compose.yaml -f deploy/compose.gateway.yaml up -d
```

Open `http://127.0.0.1:18088` and authenticate. The same-origin frontend build is
for this gateway; direct port 13085 does not proxy the API. Local HTTP is suitable
for loopback or an authenticated SSH connection. Public exposure must use TLS
and separately approved hosting/tunneling; this overlay starts no public tunnel.
