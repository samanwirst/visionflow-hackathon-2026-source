# VisionFlow LIVE Camera + Judges — setup, security, scope

An **additive, isolated** product: a real phone camera streams live frames; a judge
on another device watches the feed and creates/activates AI monitoring tasks in
natural language, executed against the live stream with real YOLO detection,
tracking, temporal events, evidence, and local notifications.

It reuses the accepted M1 engine (`visionflow_vision` detector/tracker, zone event
engine, evidence JPEGs, local notifications). It does **not** modify the main app,
the geometric fallback, or the M2/M3 worktrees. It is served as a **separate public
sub-app** so the legacy unauthenticated upload/control/media endpoints are never
exposed on the public tunnel.

## Run

```bash
cd services/api
# REQUIRED: set the shared demo PIN in the environment (never in source).
export VISIONFLOW_DEMO_PIN='<pin>'
# For a public HTTPS deployment also set the exact origin allowlist:
export VISIONFLOW_LIVE_ALLOWED_ORIGINS='https://your-demo.example'
.venv/bin/python -m app.live --host 127.0.0.1 --port 8001
```

- Landing: `/` (exactly two choices: **Live Camera**, **Judges**).
- Camera page `/camera`, Judges page `/judges` (static HTML served by the app).
- Health: `/api/live/health`.

Local HTTP QA only (cookies won't be Secure): `VISIONFLOW_LIVE_INSECURE_COOKIE=true`.
Never use that flag on the public deployment.

## Public HTTPS proxy (cloudflared) — run by root after independent review

`getUserMedia` requires a secure context, so the camera only works over HTTPS.
Point the existing cloudflared tunnel at the live port and nothing else:

```bash
cloudflared tunnel --url http://127.0.0.1:8001
```

Proxy requirements (already handled by the app / entrypoint):
- WebSocket Upgrade is forwarded (cloudflared does this by default).
- `uvicorn --proxy-headers` honors `X-Forwarded-For` for per-IP rate limiting.
- Set `VISIONFLOW_LIVE_ALLOWED_ORIGINS` to the tunnel's https origin so the
  Origin/CSRF checks accept the real site and reject cross-site requests.
- No `CORS: *` is ever set; the live app has no permissive CORS middleware.

## Security model

- **Shared PIN → role only.** The PIN authenticates `camera` or `judge`. It does
  NOT grant room access, cannot list rooms (no listing endpoint exists), and cannot
  operate a workflow by itself.
- **Opaque per-room join code** is the pairing secret. The camera creates a room and
  receives the join code once over the authenticated channel; it is never placed in a
  URL or logged. The judge joins by POSTing the code; their session is then bound to
  that one room.
- **Session cookie** is HttpOnly + Secure + SameSite=strict, short TTL. A separate
  readable CSRF cookie backs a double-submit `X-CSRF-Token` header on every mutation.
- **Origin allowlist** (no wildcard) guards every mutation and both WebSocket
  upgrades.
- **Role/room isolation:** camera sessions cannot plan/activate tasks; judge sessions
  cannot push camera frames; a session bound to one room cannot read another room's
  frames/events/evidence. Evidence is served only to room members (path-scoped, no
  traversal).
- **Bounds:** sessions, rooms, judges/room, login failures, frame bytes (≤256 KiB)
  and long edge (≤960) are all capped. Only the latest frame per room is processed
  (stale frames dropped); CPU inference is serialized (global capacity one).

## Dynamic agent (planner)

The judge's free text is planned by the **real tool-less `claude` CLI** run as an
isolated typed planner: `--print --output-format json --tools "" --permission-mode
dontAsk --strict-mcp-config --mcp-config '{"mcpServers":{}}' --no-session-persistence
--safe-mode --system-prompt <caps+honesty> --json-schema <strict>`. The prompt is
passed on stdin, `shell=False`, in a throwaway cwd outside the repo, with a bounded
timeout, global concurrency 1, a per-room attempt budget + cooldown, and NO automatic
retry. Auth failure / timeout / invalid JSON → an explicit *unavailable* result;
there is **no keyword/heuristic fallback**. The backend re-validates every proposal
against its own allowlists (`app.live.dsl`), so an injected instruction cannot escape
the typed schema. Real model/usage/cost/duration are preserved for audit.

Observed CLI cost in verification: ~$0.03–0.09 per plan, ~7–15 s (model
`claude-fable-5`). The structured-output tool requires every schema field to be
required (strict mode); "not applicable" uses sentinels (`""`/`0`/`[]`).

## Supported primitives (declared, honest)

- Object classes: real COCO classes the detector produces —
  `person, car, truck, bus, motorcycle, bicycle, bottle, cup, cell phone, laptop,
  backpack, handbag, suitcase, chair`.
- Triggers: `presence_in_zone, entered_zone, exited_zone, dwell_in_zone,
  count_exceeded` (full-frame or polygon zone).
- Actions: `notify`, `focus_notify` (highlight / enlarge the tracked bbox in the
  Judges UI — **no physical camera pan/tilt**; no hardware PTZ exists).
- Optional `visual_observation`: a non-actioning semantic/temporal check routed to
  the weak local VLM (e.g. "is the bottle being opened"). It is **uncalibrated and
  always flagged uncertain**, and can never by itself trigger an action.

## Unsupported / honest limitations

- Semantic/temporal conditions (opening, unattended, damage) are not directly
  detectable; the planner preserves intent as an uncertain `visual_observation` plus
  the closest geometric trigger, or asks for clarification — it never silently drops
  a condition and never fakes a temporal event from a still frame.
- The Teacher VLM is weak/uncalibrated; observations are informational only.
- Multiple simultaneous rooms share one detector; switching the actively streaming
  room resets tracker identities (designed for one active camera per demo).
- Phone background/lock may pause capture; a screen wake lock is requested where
  supported and the limitation is disclosed in the UI.
- No automatic raw recording: only the volatile latest frame, a short bounded
  temporal deque, and bounded event evidence JPEGs are kept.
- This is a demo PIN, not production-grade auth.

## Tests

`tests/live/` — auth/role/CSRF/Origin/brute-force/session-expiry/room-cross-access
/WS rejection; JPEG size/queue/latest-only/drop; worker event→evidence→notification;
task replacement; disable-while-camera-persists; version guard; planner strict typed
validation + budget/cooldown (deterministic injected transport, QA only); Teacher
non-actioning/uncertain/cached/budget. Real YOLO and real CLI planner are verified
out-of-band (not in the unit suite) and by the human two-device E2E.
