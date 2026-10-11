"""Authentication, sessions, rate limiting, and CSRF/Origin checks for the live app.

The shared demo PIN authenticates a ROLE only. It deliberately does NOT grant
access to any room: joining a room requires the opaque per-room join code, so a
PIN holder cannot enumerate or join rooms or operate a workflow.

Sessions are opaque server-side tokens delivered in an HttpOnly, Secure,
SameSite cookie with a short TTL. A double-submit CSRF token plus an Origin
allowlist protects every state-changing request and WebSocket upgrade.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.live import config

Role = str  # "camera" | "judge"


@dataclass
class Session:
    token: str
    role: Role
    csrf: str
    created_at: float
    expires_at: float
    room_id: str | None = None  # the single room this session is bound to
    owns_room: bool = False     # camera that created the room

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()

# Per-IP failed-login tracking for rate limiting.
_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()


def pin_configured() -> bool:
    return bool(config.DEMO_PIN)


def verify_pin(pin: str) -> bool:
    """Constant-time PIN check. An unconfigured PIN rejects everything."""
    if not config.DEMO_PIN:
        return False
    return hmac.compare_digest(str(pin or ""), config.DEMO_PIN)


# --- rate limiting ---------------------------------------------------------


def _prune_failures(ip: str, now: float) -> list[float]:
    window = config.LOGIN_WINDOW_SECONDS
    fails = [t for t in _failures.get(ip, []) if now - t < window]
    if fails:
        _failures[ip] = fails
    else:
        _failures.pop(ip, None)
    # Bound total distinct keys: drop the oldest-touched keys if the map grows
    # past the cap (an attacker rotating identities cannot exhaust memory).
    if len(_failures) > config.MAX_RATE_LIMIT_KEYS:
        for k in list(_failures.keys())[: len(_failures) - config.MAX_RATE_LIMIT_KEYS]:
            _failures.pop(k, None)
    return fails


def login_locked(ip: str) -> bool:
    now = time.time()
    with _failures_lock:
        fails = _prune_failures(ip, now)
        return len(fails) >= config.LOGIN_MAX_FAILURES


def record_failure(ip: str) -> None:
    now = time.time()
    with _failures_lock:
        _prune_failures(ip, now)
        _failures.setdefault(ip, []).append(now)


def clear_failures(ip: str) -> None:
    with _failures_lock:
        _failures.pop(ip, None)


# --- sessions --------------------------------------------------------------


def _sweep(now: float) -> list[Session]:
    dropped = [s for s in _sessions.values() if s.expired(now)]
    dead = [t for t, s in _sessions.items() if s.expired(now)]
    for t in dead:
        _sessions.pop(t, None)
    return dropped


def create_session(role: Role) -> Session:
    now = time.time()
    sweep_expired()
    with _sessions_lock:
        if len(_sessions) >= config.MAX_SESSIONS:
            raise RuntimeError("session capacity reached")
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        sess = Session(
            token=token,
            role=role,
            csrf=csrf,
            created_at=now,
            expires_at=now + config.SESSION_TTL_SECONDS,
        )
        _sessions[token] = sess
    return sess


def get_session(token: str | None) -> Session | None:
    if not token:
        return None
    now = time.time()
    with _sessions_lock:
        sess = _sessions.get(token)
        if sess is None:
            return None
        expired = sess.expired(now)
        if expired:
            _sessions.pop(token, None)
    if expired:
        _notify_dropped([sess])
        return None
    return sess


def session_current(sess: Session) -> bool:
    """Check authority without a teardown callback while holding a room lock."""
    with _sessions_lock:
        return _sessions.get(sess.token) is sess and not sess.expired()


def peek_session(token: str | None) -> Session | None:
    """Non-callback authority lookup for callers already holding room.lock."""
    with _sessions_lock:
        sess = _sessions.get(token)
        return sess if sess is not None and not sess.expired() else None


# Optional hook (set by the app layer) invoked when a session's membership must
# be torn down: on logout and on expiry sweep. Receives (token, room_id, owns).
# Kept as a callback to avoid an auth -> rooms import cycle.
on_session_dropped = None


def _notify_dropped(dropped: list[Session]) -> None:
    cb = on_session_dropped
    if cb is None:
        return
    for s in dropped:
        if s.room_id:
            try:
                cb(s.token, s.room_id, s.owns_room)
            except Exception:  # noqa: BLE001 — cleanup must never crash auth
                pass


def destroy_session(token: str | None) -> Session | None:
    """Remove a session and return it (so the caller can tear down memberships)."""
    if not token:
        return None
    with _sessions_lock:
        sess = _sessions.pop(token, None)
    if sess is not None:
        _notify_dropped([sess])
    return sess


def bind_room(token: str, room_id: str, owns: bool) -> str | None:
    """Bind a session to a room; returns the PREVIOUS room_id if the session was
    re-bound (so the caller can evict it from the old room)."""
    with _sessions_lock:
        sess = _sessions.get(token)
        if sess is None:
            return None
        prev = sess.room_id if sess.room_id and sess.room_id != room_id else None
        sess.room_id = room_id
        sess.owns_room = owns
        return prev


def active_tokens() -> set[str]:
    now = time.time()
    with _sessions_lock:
        return {t for t, s in _sessions.items() if not s.expired(now)}


def active_session_count() -> int:
    with _sessions_lock:
        dropped = [s for t, s in _sessions.items() if s.expired(time.time())]
        _sweep(time.time())
    _notify_dropped(dropped)
    with _sessions_lock:
        return len(_sessions)


def sweep_expired() -> set[str]:
    """Drop expired sessions, firing membership teardown, and return the tokens
    that remain active. Called periodically by the worker (LIVE-006)."""
    now = time.time()
    with _sessions_lock:
        dropped = [s for t, s in list(_sessions.items()) if s.expired(now)]
        for s in dropped:
            _sessions.pop(s.token, None)
        alive = set(_sessions.keys())
    _notify_dropped(dropped)
    return alive


def _trusted_proxy_peers() -> set[str]:
    raw = config.FORWARDED_ALLOW_IPS or ""
    return {p.strip() for p in raw.split(",") if p.strip()}


def client_identity(peer: str | None, forwarded_for: str | None) -> str:
    """Resolve a trustworthy client identity for rate limiting.

    X-Forwarded-For is honored ONLY when (a) ``TRUSTED_PROXY_HOPS`` > 0 AND (b) the
    DIRECT peer is one of the explicitly trusted proxy peers
    (``FORWARDED_ALLOW_IPS``). The identity is then the (hops)-th address from the
    RIGHT of the chain — the entry inserted by the closest trusted proxy — so a
    client cannot spoof it by prepending addresses. A caller connecting directly
    (untrusted peer) can never supply its own forwarding header to dodge the PIN
    lockout (LIVE-VIDEO-006)."""
    hops = config.TRUSTED_PROXY_HOPS
    if hops > 0 and forwarded_for and peer is not None:
        trusted = _trusted_proxy_peers()
        if "*" in trusted or peer in trusted:
            chain = [a.strip() for a in forwarded_for.split(",") if a.strip()]
            if len(chain) >= hops:
                return chain[-hops]
    return peer or "unknown"


def _reset_for_tests() -> None:
    """Clear all in-memory auth state (used by unit tests only)."""
    with _sessions_lock:
        _sessions.clear()
    with _failures_lock:
        _failures.clear()


# --- origin / CSRF ---------------------------------------------------------


def host_trusted(host: str | None) -> bool:
    """A ``Host`` value is trusted only when it matches the configured deployment
    host allowlist. When no allowlist is configured (local dev) any Host is
    accepted, but it can never be used to widen a configured origin allowlist."""
    if not config.ALLOWED_HOSTS:
        return True
    return bool(host) and host in config.ALLOWED_HOSTS


def origin_allowed(origin: str | None, host: str | None) -> bool:
    """Authoritative Origin policy for mutations and WebSocket upgrades.

    When an origin allowlist is configured it is the ONLY authority: the request
    must carry an ``Origin`` in that list. The configured ``Host`` allowlist is ALSO
    authoritative — a request with an allowed Origin but a Host outside the
    allowlist (a Host-header attack) is refused (LIVE-VIDEO-006). Only in
    unconfigured local dev do we fall back to same-origin matching."""
    if config.ALLOWED_ORIGINS:
        # Strict: a configured deployment demands an explicit, allowlisted Origin AND
        # a Host within the configured host allowlist.
        return bool(origin) and origin in config.ALLOWED_ORIGINS and host_trusted(host)
    # Unconfigured (dev): accept a missing Origin (non-browser / same-origin
    # navigation) or an Origin whose netloc matches a trusted Host.
    if not origin:
        return True
    try:
        oh = urlparse(origin).netloc
    except ValueError:
        return False
    return bool(oh) and host_trusted(host) and oh == host


def csrf_valid(sess: Session, header_token: str | None) -> bool:
    if not header_token:
        return False
    return hmac.compare_digest(sess.csrf, header_token)
