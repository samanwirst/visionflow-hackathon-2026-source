"""Runtime configuration for the LIVE Camera + Judges sub-application.

This is a SEPARATE public surface from the main M1 API. Nothing here is imported
by ``app.main``; the live app is mounted on its own port so the public tunnel
never exposes the legacy unauthenticated upload/control/media endpoints.

Secrets (the demo PIN) come ONLY from the environment. They are never hardcoded
in source, tests, or logs. Session cookies default to Secure so an accidental
HTTP deployment cannot leak them; local QA may opt out with an explicit flag.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger("visionflow.live.config")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    _log.warning("Unrecognized %s=%r; using default %s", name, raw, default)
    return default


# --- authentication --------------------------------------------------------
# The shared demo PIN authenticates a ROLE (camera or judge). It is NOT
# production-grade security; the per-room join code is the real pairing secret.
# An unset PIN means the live app refuses every login (fail-closed) rather than
# accepting an empty password.
DEMO_PIN = os.environ.get("VISIONFLOW_DEMO_PIN", "")

SESSION_TTL_SECONDS = int(os.environ.get("VISIONFLOW_LIVE_SESSION_TTL", "1800"))  # 30 min
SESSION_COOKIE_NAME = "vf_live_session"
CSRF_COOKIE_NAME = "vf_live_csrf"

# Secure cookie default TRUE. Only an explicit opt-out (local HTTP QA) sets
# VISIONFLOW_LIVE_INSECURE_COOKIE=true; a public HTTPS deployment keeps it on.
COOKIE_SECURE = not _flag("VISIONFLOW_LIVE_INSECURE_COOKIE", False)

# --- bounds (fail-closed resource limits) ----------------------------------
MAX_SESSIONS = int(os.environ.get("VISIONFLOW_LIVE_MAX_SESSIONS", "64"))
MAX_ROOMS = int(os.environ.get("VISIONFLOW_LIVE_MAX_ROOMS", "8"))
MAX_JUDGES_PER_ROOM = int(os.environ.get("VISIONFLOW_LIVE_MAX_JUDGES", "8"))

# Login rate limiting (per client IP): N failures within WINDOW -> locked out.
LOGIN_MAX_FAILURES = int(os.environ.get("VISIONFLOW_LIVE_LOGIN_MAX_FAILURES", "5"))
LOGIN_WINDOW_SECONDS = int(os.environ.get("VISIONFLOW_LIVE_LOGIN_WINDOW", "300"))

# --- frame ingest bounds ---------------------------------------------------
MAX_FRAME_BYTES = int(os.environ.get("VISIONFLOW_LIVE_MAX_FRAME_BYTES", str(256 * 1024)))
MAX_FRAME_LONG_EDGE = int(os.environ.get("VISIONFLOW_LIVE_MAX_LONG_EDGE", "960"))
MIN_FRAME_LONG_EDGE = 64
# Bounded short temporal history kept in memory per room for before/after probes.
TEMPORAL_DEQUE_LEN = int(os.environ.get("VISIONFLOW_LIVE_TEMPORAL_DEQUE", "8"))
# Event evidence retention (bounded TTL + count) per room.
EVIDENCE_MAX_PER_ROOM = int(os.environ.get("VISIONFLOW_LIVE_EVIDENCE_MAX", "60"))

# --- origin / CSRF ---------------------------------------------------------
# Comma-separated allowlist of exact origins permitted for state-changing HTTP
# and WebSocket upgrades. NO wildcard. When empty, same-origin requests (no
# Origin header, or Origin matching an allowlisted Host) are allowed; cross-origin
# is refused. Populate this for the deployed HTTPS origin.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("VISIONFLOW_LIVE_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
# Trusted deployment host[:port] values. The ``Host`` header is CALLER-controlled,
# so same-origin matching is only honored when the request Host is one of these
# trusted values. Derived from the origins' netlocs by default; override with
# VISIONFLOW_LIVE_ALLOWED_HOSTS. Empty + no origins => dev same-origin (Host used
# only to compare against Origin, never to widen a configured allowlist).
_derived_hosts = []
for _o in ALLOWED_ORIGINS:
    try:
        from urllib.parse import urlparse as _up
        _nl = _up(_o).netloc
        if _nl:
            _derived_hosts.append(_nl)
    except ValueError:
        pass
ALLOWED_HOSTS = [
    h.strip() for h in os.environ.get("VISIONFLOW_LIVE_ALLOWED_HOSTS", "").split(",") if h.strip()
] or _derived_hosts

# --- trusted proxy / client identity ---------------------------------------
# Number of trusted reverse-proxy hops in front of the app (e.g. 1 for a single
# cloudflared tunnel). X-Forwarded-For is only honored when > 0, and the client
# identity is taken as the (count)-th address from the RIGHT, so an attacker
# cannot spoof it by prepending addresses. 0 => ignore XFF, use the peer address.
TRUSTED_PROXY_HOPS = int(os.environ.get("VISIONFLOW_LIVE_TRUSTED_PROXY_HOPS", "0"))
# Comma-separated peer IPs allowed to supply forwarding headers (passed to
# uvicorn's forwarded_allow_ips). Default loopback only — never "*".
FORWARDED_ALLOW_IPS = os.environ.get(
    "VISIONFLOW_LIVE_FORWARDED_ALLOW_IPS", "127.0.0.1,::1"
).strip()
# Hard cap on distinct rate-limit keys retained (prevents unbounded growth).
MAX_RATE_LIMIT_KEYS = int(os.environ.get("VISIONFLOW_LIVE_MAX_RL_KEYS", "4096"))

# --- request body bounds (pre-parse) ---------------------------------------
# Maximum JSON body for any live HTTP endpoint and maximum WebSocket text-frame
# (metadata) size. Bounded well below the frame byte cap.
MAX_JSON_BODY_BYTES = int(os.environ.get("VISIONFLOW_LIVE_MAX_JSON_BYTES", str(16 * 1024)))
MAX_WS_TEXT_BYTES = int(os.environ.get("VISIONFLOW_LIVE_MAX_WS_TEXT_BYTES", str(2 * 1024)))
# Hard wall-clock bound on how long the server will spend RECEIVING a JSON body
# in the pre-parse ASGI guard. A trickle/slowloris body cannot hold a worker
# indefinitely (LIVE-VIDEO-007).
REQUEST_RECEIVE_DEADLINE_SECONDS = float(
    os.environ.get("VISIONFLOW_LIVE_RECEIVE_DEADLINE", "30")
)

# --- room lifecycle --------------------------------------------------------
# A room with no received frames and no judges for this many seconds is retired
# (pairing code invalidated, evidence deleted, capacity released).
ROOM_IDLE_TTL_SECONDS = float(os.environ.get("VISIONFLOW_LIVE_ROOM_IDLE_TTL", "900"))

# --- Claude CLI planner ----------------------------------------------------
PLANNER_ENABLED = _flag("VISIONFLOW_LIVE_PLANNER", True)
PLANNER_CLI = os.environ.get("VISIONFLOW_LIVE_CLAUDE_BIN", "claude")
PLANNER_TIMEOUT_SECONDS = float(os.environ.get("VISIONFLOW_LIVE_PLANNER_TIMEOUT", "75"))
PLANNER_MODEL = os.environ.get("VISIONFLOW_LIVE_PLANNER_MODEL", "").strip()  # "" -> CLI default
# Per-room planner attempt budget (whole process lifetime) and cooldown between
# attempts. Protects the real subscription from a flood of judge prompts.
PLANNER_ROOM_BUDGET = int(os.environ.get("VISIONFLOW_LIVE_PLANNER_BUDGET", "10"))
PLANNER_COOLDOWN_SECONDS = float(os.environ.get("VISIONFLOW_LIVE_PLANNER_COOLDOWN", "3"))
# Hard cap on bytes captured from the planner subprocess stdout/stderr. A flooding
# CLI cannot exhaust memory; excess output is dropped while the pipe keeps draining
# so the process can never deadlock (LIVE-VIDEO-008).
PLANNER_MAX_OUTPUT_BYTES = int(
    os.environ.get("VISIONFLOW_LIVE_PLANNER_MAX_OUTPUT", str(4 * 1024 * 1024))
)

# --- Teacher (non-actioning observation adapter) ---------------------------
# Reuses the existing private loopback SmolVLM endpoint. Unset -> unavailable.
TEACHER_BASE_URL = os.environ.get("LIVE_TEACHER_BASE_URL", "").strip()
TEACHER_MODEL = os.environ.get("LIVE_TEACHER_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct").strip()
TEACHER_TOKEN = os.environ.get("LIVE_TEACHER_TOKEN", "").strip()
TEACHER_TIMEOUT_SECONDS = float(os.environ.get("LIVE_TEACHER_TIMEOUT", "60"))
TEACHER_ROOM_BUDGET = int(os.environ.get("LIVE_TEACHER_ROOM_BUDGET", "20"))
TEACHER_COOLDOWN_SECONDS = float(os.environ.get("LIVE_TEACHER_COOLDOWN", "4"))
# Maximum confidence treated as a usable signal (still uncalibrated). Values
# outside [0,1], non-finite, or boolean are rejected as malformed, never coerced.
# Semantic confirmation requires corroboration, never a score alone.

# --- worker ----------------------------------------------------------------
THREAD_COUNT = int(os.environ.get("VISIONFLOW_THREADS", "2"))
TARGET_PROCESS_FPS = float(os.environ.get("VISIONFLOW_LIVE_PROCESS_FPS", "4"))

# --- input source mode (OWNER PIVOT) ---------------------------------------
# The primary public demo is PRE-RECORDED VIDEO inference. The physical-camera /
# two-device WebSocket ingest path is UNFINISHED and DISABLED on the public
# surface by default (fail-closed): its routes deny upgrades / return 404 and the
# landing page hides its links. Local QA that still wants to exercise the legacy
# (non-public) WebSocket hardening flips this flag explicitly. Never enable it on
# the public tunnel.
CAMERA_INGEST_ENABLED = _flag("VISIONFLOW_LIVE_ENABLE_CAMERA", False)

# --- video upload bounds (fail-closed resource limits) ---------------------
# Hard cap on the streamed upload size, enforced byte-by-byte during the stream
# (a lying/absent Content-Length can never bypass it). Plus bounds on the decoded
# clip so a crafted container cannot exhaust CPU/RAM.
MAX_VIDEO_BYTES = int(os.environ.get("VISIONFLOW_LIVE_MAX_VIDEO_BYTES", str(64 * 1024 * 1024)))
MAX_VIDEO_DURATION_SECONDS = float(os.environ.get("VISIONFLOW_LIVE_MAX_VIDEO_SECONDS", "120"))
MAX_VIDEO_FRAMES = int(os.environ.get("VISIONFLOW_LIVE_MAX_VIDEO_FRAMES", "5000"))
MAX_VIDEO_FPS = float(os.environ.get("VISIONFLOW_LIVE_MAX_VIDEO_FPS", "60"))
MIN_VIDEO_FPS = float(os.environ.get("VISIONFLOW_LIVE_MIN_VIDEO_FPS", "1"))
MAX_VIDEO_LONG_EDGE = int(os.environ.get("VISIONFLOW_LIVE_MAX_VIDEO_LONG_EDGE", "1920"))
MIN_VIDEO_LONG_EDGE = int(os.environ.get("VISIONFLOW_LIVE_MIN_VIDEO_LONG_EDGE", "64"))
# Only one processing job per room at a time; total concurrent jobs bounded.
MAX_CONCURRENT_VIDEO_JOBS = int(os.environ.get("VISIONFLOW_LIVE_MAX_VIDEO_JOBS", "2"))
# Supported event cooldown (debounce) for a geometric rule, in SOURCE-video
# seconds. A task may carry an explicit ``cooldown_seconds``; when absent the VIDEO
# engine uses this default (LIVE-VIDEO-005). Consumed by rooms.build_engines.
DEFAULT_COOLDOWN_SECONDS = float(os.environ.get("VISIONFLOW_LIVE_DEFAULT_COOLDOWN", "10"))
VIDEO_UPLOAD_DIR = app_env_dir = os.environ.get("VISIONFLOW_LIVE_VIDEO_DIR", "").strip()
