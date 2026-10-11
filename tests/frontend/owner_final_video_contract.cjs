// Owner-final VIDEO frontend contract tests (LIVE-VIDEO-009 + frontend portion of
// LIVE-VIDEO-005). Executes the committed production scripts in a simulated
// DOM/HTTP VM — NOT a browser. No rendering/codec/network claim is made; this
// asserts the script behaviour and served markup only.
//
// Canonical approval contract (coordinated with SEC, see coordination-VFUI.json):
//   plan/validate success -> { task, draft_hash, expected_version }
//   activate request      -> { task, draft_hash, expected_version }
// The original review probe (frontend-video-contract.cjs) guessed the version
// field name as `base_version`; this file asserts the canonical `expected_version`
// used by the owned pages — an explicit, documented contract adaptation.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");

const staticRoot =
  __dirname + "/../../services/api/app/live/static/";

class Element {
  constructor(id = "") {
    Object.assign(this, {
      id, value: "", textContent: "", innerHTML: "",
      children: [], listeners: {}, classes: new Set(), src: "",
    });
  }
  get classList() {
    return {
      add: (c) => this.classes.add(c),
      remove: (c) => this.classes.delete(c),
      contains: (c) => this.classes.has(c),
    };
  }
  addEventListener(t, f) { this.listeners[t] = f; }
  appendChild(c) { this.children.push(c); }
  dispatch(t) { this.listeners[t]?.(); }
}

function page(file) {
  const elements = new Map();
  const intervals = [];
  const document = {
    cookie: "vf_live_csrf=QA_ONLY_CSRF",
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, new Element(id));
      return elements.get(id);
    },
    createElement: (t) => new Element(t),
  };
  const context = {
    document, console, Date, JSON, XMLHttpRequest: class {},
    location: { protocol: "https:", host: "live.qa.invalid" },
    fetch: async () => ({ ok: true, json: async () => ({ teacher_available: false }) }),
    setInterval: (fn) => { intervals.push(fn); return intervals.length; },
    clearInterval() {},
    setTimeout: (fn) => fn && 1,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(staticRoot + "common.js", "utf8"), context);
  const html = fs.readFileSync(staticRoot + file, "utf8");
  vm.runInContext(
    [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).join("\n"),
    context,
  );
  const calls = [];
  context.api = async (path, opts = {}) => { calls.push({ path, opts }); return { ok: true, task_version: 1 }; };
  return { context, html, calls, intervals, get: (id) => document.getElementById(id), eval: (s) => vm.runInContext(s, context) };
}

const results = [];
async function test(name, fn) {
  try { await fn(); results.push({ name, status: "PASS" }); }
  catch (e) { results.push({ name, status: "FAIL", reason: e.message }); }
}

const task = {
  name: "QA_ONLY dwell 1.25", object_class: "car",
  zone: { type: "full_frame", polygon: [[0, 0], [1, 0], [1, 1], [0, 1]] },
  trigger: "dwell_in_zone", duration_seconds: 1.25, threshold: null,
  action: "notify", focus: false, visual_observation: null, explanation: "QA_ONLY",
};

(async () => {
  // ---- LIVE-VIDEO-009: judge progress / job status / source clock / error ----
  await test("judge served markup exposes a playable source video element", async () => {
    const p = page("judges.html");
    assert.ok(/<video\b/.test(p.html), "no <video> source/replay element served");
    assert.ok(/\/video\/source/.test(p.html), "source player not bound to the owned /video/source route");
  });

  await test("judge renders real job status, count/total and source elapsed time", async () => {
    const p = page("judges.html");
    p.context.renderState({
      task, task_version: 1, task_enabled: true, task_paused: false, metrics: {},
      events: [], notifications: [],
      video: { job_status: "running", job_processed: 14, job_total: 32, meta: { fps: 10 } },
    });
    const info = p.get("jobInfo").textContent;
    assert.ok(/running/.test(info), "job_status not shown: " + info);
    assert.ok(/14\/32/.test(info), "frame count/total not shown: " + info);
    assert.ok(/1\.4s/.test(info), "source elapsed time not shown: " + info);
  });

  await test("job status is distinct from task enablement (enabled != running)", async () => {
    const p = page("judges.html");
    p.context.renderState({
      task, task_version: 1, task_enabled: true, task_paused: false, metrics: {},
      events: [], notifications: [],
      video: { job_status: "done", job_processed: 32, job_total: 32, meta: { fps: 10 } },
    });
    // Task pill still reports enabled; the job line separately reports the real state.
    assert.ok(/running/.test(p.get("activeTask").innerHTML), "task enablement pill missing");
    assert.ok(/done/.test(p.get("jobInfo").textContent), "job completion not distinguished from enablement");
  });

  await test("judge surfaces the actual job error", async () => {
    const p = page("judges.html");
    p.context.renderState({
      task, task_version: 1, task_enabled: true, task_paused: false, metrics: {},
      events: [], notifications: [],
      video: { job_status: "error", job_processed: 3, job_total: 32, job_error: "decode_failed", meta: { fps: 10 } },
    });
    assert.ok(/decode_failed/.test(p.get("jobInfo").textContent), "job error not displayed");
  });

  await test("judge binds the source player to the authenticated same-room route", async () => {
    const p = page("judges.html");
    p.eval('roomId="QA_ONLY"');
    p.context.renderState({
      task, task_version: 1, task_enabled: true, task_paused: false, metrics: {},
      events: [], notifications: [],
      video: { sha256: "a".repeat(64), job_status: "running", job_processed: 1, job_total: 32, meta: { fps: 10 } },
    });
    assert.equal(p.get("sourceVideo").src, "/api/live/rooms/QA_ONLY/video/source?sha=" + "a".repeat(64));
  });

  await test("judge 401/403 polling visibly revokes stale running state and requests sign-in", async () => {
    const p = page("judges.html");
    p.eval('roomId="QA_ONLY"');
    // seed a stale running display
    p.get("feed").src = "/api/live/rooms/QA_ONLY/frame?t=1";
    p.get("main").classList.remove("hidden");
    p.get("authCard").classList.add("hidden");
    p.context.api = async () => { throw new Error("request failed (401)"); };
    p.context.startPolling();
    await new Promise((r) => setImmediate(r));
    assert.ok(/auth|expired|sign ?in/i.test(p.get("banner").textContent), "401 swallowed; no sign-in prompt");
    assert.equal(p.get("feed").src, "", "stale feed not cleared on expiry");
    assert.ok(p.get("main").classList.contains("hidden"), "main not hidden on expiry");
    assert.ok(!p.get("authCard").classList.contains("hidden"), "sign-in not re-shown on expiry");
  });

  await test("judge non-auth polling errors keep last state (no spurious logout)", async () => {
    const p = page("judges.html");
    p.eval('roomId="QA_ONLY"');
    p.get("main").classList.remove("hidden");
    p.context.api = async () => { throw new Error("network timeout"); };
    p.context.startPolling();
    await new Promise((r) => setImmediate(r));
    assert.ok(!p.get("main").classList.contains("hidden"), "transient error wrongly forced logout");
  });

  // ---- LIVE-VIDEO-005 (frontend): canonical digest + version approval binding ----
  await test("approval binds canonical draft_hash and expected_version from plan", async () => {
    const p = page("judges.html");
    p.context.showDraft({
      status: "ok", original_text: "QA_ONLY", task,
      draft_hash: "QA_ONLY_hash", expected_version: 7, provenance: { model: "QA_ONLY" },
    });
    await p.get("activateBtn").onclick();
    const body = p.calls[0].opts.body;
    assert.equal(body.draft_hash, "QA_ONLY_hash", "draft_hash not sent on approval");
    assert.equal(body.expected_version, 7, "expected_version not sent on approval");
    assert.equal(body.task.duration_seconds, 1.25, "decimal dwell duration mutated");
  });

  await test("manual JSON edit invalidates pending approval until re-validated", async () => {
    const p = page("judges.html");
    p.context.showDraft({ status: "ok", original_text: "x", task, draft_hash: "h1", expected_version: 1, provenance: {} });
    // user edits the JSON editor
    p.get("editJson").value = JSON.stringify({ ...task, name: "edited" });
    p.get("editJson").dispatch("input");
    assert.equal(p.eval("currentDraft"), null, "edit did not clear the pending draft");
    assert.equal(p.eval("currentDraftHash"), null, "edit did not clear the bound hash");
    assert.ok(p.get("activateBtn").classList.contains("hidden"), "approve stayed visible after edit");
    // approving now must be a no-op (no stale request)
    const before = p.calls.length;
    await p.get("activateBtn").onclick();
    assert.equal(p.calls.length, before, "stale/edited draft was approved without re-validation");
  });

  await test("validate re-binds approval to the server canonical task + digest", async () => {
    const p = page("judges.html");
    p.context.showDraft({ status: "ok", original_text: "x", task, draft_hash: "old", expected_version: 1, provenance: {} });
    p.get("editJson").value = JSON.stringify({ ...task, name: "edited" });
    p.get("editJson").dispatch("input");
    // server returns the canonical task + fresh identity
    const canonical = { ...task, name: "edited" };
    p.context.api = async (path, opts = {}) => {
      p.calls.push({ path, opts });
      if (path.endsWith("/task/validate")) return { task: canonical, draft_hash: "NEWHASH", expected_version: 2 };
      return { ok: true, task_version: 2 };
    };
    await p.get("validateBtn").onclick();
    assert.ok(!p.get("activateBtn").classList.contains("hidden"), "approve not re-exposed after validation");
    await p.get("activateBtn").onclick();
    const activate = p.calls.find((c) => c.path.endsWith("/task/activate"));
    assert.ok(activate, "no activate call after validation");
    assert.equal(activate.opts.body.draft_hash, "NEWHASH", "approval not bound to re-validated digest");
    assert.equal(activate.opts.body.expected_version, 2, "approval not bound to re-validated version");
    assert.equal(activate.opts.body.task.name, "edited", "approved task is not the canonical validated task");
  });

  await test("validate error shows reason and does not expose approval", async () => {
    const p = page("judges.html");
    p.get("editJson").value = JSON.stringify({ ...task, object_class: "bottle" });
    p.get("editJson").dispatch("input");
    p.context.api = async () => { throw new Error("unsupported object_class"); };
    await p.get("validateBtn").onclick();
    assert.ok(/unsupported/i.test(p.get("banner").textContent), "server rejection reason not shown");
    assert.ok(p.get("activateBtn").classList.contains("hidden"), "approve exposed after failed validation");
  });

  await test("unknown planner cost renders 'unavailable', never $0.0000", async () => {
    const p = page("judges.html");
    p.context.showDraft({ status: "ok", original_text: "x", task, provenance: { model: "QA_ONLY", cost_usd: null } });
    assert.ok(/unavailable/i.test(p.get("planMeta").textContent), "unknown cost not labelled unavailable");
    assert.ok(!/\$0\.0000/.test(p.get("planMeta").textContent), "unknown cost fabricated as $0.0000");
  });

  // ---- LIVE-VIDEO-009 (operator/upload page) ----
  await test("operator page shows job status, progress and source clock", async () => {
    const p = page("upload.html");
    p.eval('roomId="QA_ONLY"');
    p.context.api = async () => ({ video: { job_status: "running", job_processed: 7, job_total: 20, meta: { fps: 10 } } });
    p.context.startPolling();
    await new Promise((r) => setImmediate(r));
    const info = p.get("jobInfo").textContent;
    assert.ok(/running/.test(info) && /7\/20/.test(info) && /0\.7s/.test(info), "operator progress incomplete: " + info);
  });

  await test("operator page surfaces the actual job error", async () => {
    const p = page("upload.html");
    p.eval('roomId="QA_ONLY"');
    p.context.api = async () => ({ video: { job_status: "error", job_processed: 0, job_total: 20, job_error: "bad_codec", meta: { fps: 10 } } });
    p.context.startPolling();
    await new Promise((r) => setImmediate(r));
    assert.ok(/bad_codec/.test(p.get("jobInfo").textContent), "operator job error not shown");
  });

  await test("operator 401/403 polling revokes stale state and requests sign-in", async () => {
    const p = page("upload.html");
    p.eval('roomId="QA_ONLY"');
    p.get("feed").src = "/api/live/rooms/QA_ONLY/frame?t=1";
    p.get("main").classList.remove("hidden");
    p.get("authCard").classList.add("hidden");
    p.context.api = async () => { throw new Error("request failed (403)"); };
    p.context.startPolling();
    await new Promise((r) => setImmediate(r));
    assert.ok(/auth|expired|sign ?in/i.test(p.get("banner").textContent), "operator 401/403 swallowed");
    assert.equal(p.get("feed").src, "", "operator stale feed not cleared");
    assert.ok(!p.get("authCard").classList.contains("hidden"), "operator sign-in not re-shown");
  });

  await test("operator page exposes a source video replay element", async () => {
    const p = page("upload.html");
    assert.ok(/<video\b/.test(p.html) && /\/video\/source/.test(p.html), "operator source player missing");
  });

  // Actual production-script source identity and asynchronous control boundaries.
  const state = (version, sha, extra = {}) => ({
    task, task_version: version, task_enabled: true, task_paused: false,
    metrics: {}, events: [], notifications: [],
    video: { sha256: sha, job_status: "running", job_processed: 1, job_total: 32, meta: { fps: 10 } },
    ...extra,
  });
  await test("judge reloads the source player after a same-room source SHA changes", async () => {
    const p = page("judges.html"); p.eval('roomId="QA_ONLY"');
    p.context.renderState(state(1, "a".repeat(64)));
    const first = p.get("sourceVideo").src;
    p.context.renderState(state(2, "b".repeat(64)));
    assert.notEqual(p.get("sourceVideo").src, first);
    assert.equal(p.get("sourceVideo").src, "/api/live/rooms/QA_ONLY/video/source?sha=" + "b".repeat(64));
    p.context.renderState(state(3, null));
    assert.equal(p.get("sourceVideo").src, "", "removed source remains playable");
  });
  await test("operator reloads a same-room changed source and clears a removed source", async () => {
    const p = page("upload.html"); p.eval('roomId="QA_ONLY"');
    let current = state(1, "a".repeat(64)); p.context.api = async () => current;
    p.context.startPolling(); await new Promise(r => setImmediate(r));
    const first = p.get("sourceVideo").src;
    current = state(2, "b".repeat(64)); await p.intervals[0]();
    assert.notEqual(p.get("sourceVideo").src, first);
    assert.equal(p.get("sourceVideo").src, "/api/live/rooms/QA_ONLY/video/source?sha=" + "b".repeat(64));
    current = state(3, null); await p.intervals[0]();
    assert.equal(p.get("sourceVideo").src, "");
  });
  await test("judge ignores lower-version state and source after a disable acknowledgement", async () => {
    const p = page("judges.html"); p.eval('roomId="QA_ONLY"');
    p.context.renderState(state(4, "b".repeat(64)));
    let release; p.context.api = () => new Promise(r => { release = r; });
    p.context.startPolling();
    p.context.api = async () => ({ok: true, task_version: 5, task_enabled: false, task_paused: false});
    await p.context.setState({enabled: false});
    release(state(4, "a".repeat(64), { notifications: [{id: 1, title: "STALE", message: "old"}] }));
    await new Promise(r => setImmediate(r));
    assert.ok(/disabled/.test(p.get("activeTask").innerHTML));
    assert.ok(!/STALE/.test(p.get("notifs").innerHTML));
    assert.equal(p.get("sourceVideo").src, "/api/live/rooms/QA_ONLY/video/source?sha=" + "b".repeat(64));
    p.context.renderState(state(4, "a".repeat(64)));
    assert.ok(/disabled/.test(p.get("activeTask").innerHTML));
  });
  await test("operator rejects out-of-order lower-version source polling", async () => {
    const p = page("upload.html"); p.eval('roomId="QA_ONLY"');
    let replies = []; p.context.api = () => new Promise(r => replies.push(r));
    p.context.startPolling(); const newer = p.intervals[0]();
    replies[1](state(7, "b".repeat(64))); await newer;
    replies[0](state(6, "a".repeat(64))); await new Promise(r => setImmediate(r));
    assert.equal(p.get("sourceVideo").src, "/api/live/rooms/QA_ONLY/video/source?sha=" + "b".repeat(64));
  });
  await test("late polling cannot restore private sources after authentication revocation", async () => {
    for (const file of ["judges.html", "upload.html"]) {
      const p = page(file); p.eval('roomId="QA_ONLY"');
      let release; p.context.api = () => new Promise(r => { release = r; });
      p.context.startPolling(); p.context.requireLogin(); release(state(1, "a".repeat(64)));
      await new Promise(r => setImmediate(r));
      assert.equal(p.get("sourceVideo").src, "");
      assert.equal(p.get("feed").src, "");
      assert.ok(p.get("main").classList.contains("hidden"));
    }
  });

  await test("activation acknowledgement prevents a late old task from replacing the approved task", async () => {
    const p = page("judges.html"); p.eval('roomId="QA_ONLY"');
    p.context.renderState(state(3, "a".repeat(64)));
    let release; p.context.api = () => new Promise(r => { release = r; });
    p.context.startPolling();
    const replacement = {...task, name: "QA_ONLY replacement truck", object_class: "truck"};
    p.context.showDraft({status: "ok", original_text: "monitor trucks", task: replacement,
      draft_hash: "new-approved", expected_version: 3, provenance: {}});
    p.context.api = async () => ({ok: true, task_version: 4, task: replacement});
    await p.get("activateBtn").onclick();
    release(state(3, "a".repeat(64))); await new Promise(r => setImmediate(r));
    assert.ok(/replacement truck/.test(p.get("activeTask").innerHTML));
    assert.ok(/version 4/.test(p.get("activeTask").innerHTML));
  });
  await test("upload acknowledgement invalidates old-source polling before the next tick", async () => {
    const p = page("upload.html"); p.eval('roomId="QA_ONLY"');
    let release; p.context.api = () => new Promise(r => { release = r; });
    p.context.startPolling();
    p.context.XMLHttpRequest = class {
      constructor(){this.upload = {}; this.status = 200;}
      open(){} setRequestHeader(){}
      send(){this.responseText = JSON.stringify({sha256: "b".repeat(64),
        meta:{w:320,h:240,fps:10,frame_count:32,duration_seconds:3.2}}); this.onload();}
    };
    p.get("file").files = [{name: "QA_ONLY.mp4"}];
    await p.get("upload").onclick();
    release(state(1, "a".repeat(64))); await new Promise(r => setImmediate(r));
    assert.equal(p.get("sourceVideo").src, "/api/live/rooms/QA_ONLY/video/source?sha=" + "b".repeat(64));
    assert.equal(p.get("feed").src, "", "old source frame returned after new upload acknowledgement");
  });
  await test("same-version out-of-order polling cannot replace a newer source snapshot", async () => {
    for (const file of ["judges.html", "upload.html"]) {
      const p = page(file); p.eval('roomId="QA_ONLY"');
      let replies=[]; p.context.api = () => new Promise(r => replies.push(r));
      p.context.startPolling(); const newer = p.intervals[0]();
      replies[1](state(8, "b".repeat(64))); await newer;
      replies[0](state(8, "a".repeat(64))); await new Promise(r => setImmediate(r));
      assert.equal(p.get("sourceVideo").src, "/api/live/rooms/QA_ONLY/video/source?sha=" + "b".repeat(64));
    }
  });

  const failed = results.filter((r) => r.status === "FAIL");
  console.log(JSON.stringify(results, null, 2));
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  process.exitCode = failed.length ? 1 : 0;
})();
