// Shared helpers for the live camera / judges pages.
// The CSRF token is stored in a readable cookie by the server at login and sent
// back on every mutating request (double-submit). The session is an HttpOnly
// cookie the browser attaches automatically.

function getCookie(name) {
  const m = document.cookie.match("(^|;)\\s*" + name + "\\s*=\\s*([^;]+)");
  return m ? decodeURIComponent(m.pop()) : "";
}

const CSRF_COOKIE = "vf_live_csrf";

async function api(path, { method = "GET", body = null } = {}) {
  const headers = {};
  if (body) headers["Content-Type"] = "application/json";
  if (method !== "GET") headers["X-CSRF-Token"] = getCookie(CSRF_COOKIE);
  const resp = await fetch(path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : null,
    credentials: "same-origin",
  });
  let data = null;
  try { data = await resp.json(); } catch { /* non-json */ }
  if (!resp.ok) {
    const detail = data && data.detail ? data.detail : `request failed (${resp.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function login(role, pin) {
  return api("/api/live/login", { method: "POST", body: { role, pin } });
}

function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  // WebSocket upgrades cannot send custom headers, so the double-submit CSRF token
  // is carried as a query parameter and validated server-side against the cookie.
  const csrf = encodeURIComponent(getCookie(CSRF_COOKIE));
  const sep = path.includes("?") ? "&" : "?";
  return `${proto}//${location.host}${path}${sep}csrf=${csrf}`;
}

function setBanner(id, kind, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!msg) { el.classList.add("hidden"); return; }
  el.className = `banner ${kind}`;
  el.textContent = msg;
  el.classList.remove("hidden");
}
