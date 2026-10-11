// Single-visitor recorded-video workspace. No credentials are shared or
// embedded, no room is read from URL/storage. Bootstrap creates a server-owned
// opaque session and a private room; failed authentication clears private DOM
// and invalidates every pending callback. Deployment acceptance is server-owned.
(() => {
  "use strict";
  const el = (id) => document.getElementById(id);
  const controls = ["inputVideo", "selectVideo", "juryVideo", "uploadVideo", "inspectScene", "planTask", "validateTask", "approveTask", "runVideo", "disableTask", "enableTask"];
  let room = null, source = null, version = -1, draft = null, capabilities = null;
  let epoch = 0, latestPoll = 0, timer = null, csrf = null;
  const pending = new Set();
  async function api(path, {method="GET",body=null,raw=null} = {}) {
    const headers = {};
    if (method !== "GET" && csrf) headers["X-CSRF-Token"] = csrf;
    if (body) headers["Content-Type"] = "application/json";
    if (raw) headers["Content-Type"] = "video/mp4";
    const response = await fetch(path,{method,headers,body:raw || (body ? JSON.stringify(body) : null),credentials:"same-origin"});
    let result; try { result = await response.json(); } catch { result = null; }
    if (!response.ok) {
      const error = new Error(typeof result?.detail === "string" ? result.detail : "Request failed (" + response.status + ")");
      error.auth = response.status === 401 || response.status === 403; throw error;
    }
    return result;
  }
  const validVersion = (v) => Number.isInteger(v) && v >= 0;
  const validSha = (v) => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
  const owned = (suffix) => "/api/live/rooms/" + encodeURIComponent(room) + suffix;
  function message(s) { el("banner").textContent = s; el("banner").classList.remove("hidden"); }
  function setPlanningBusy(busy) {
    el("planTask").disabled = busy || !room || !source || !capabilities?.workflow;
    el("planTask").textContent = busy ? "Generating workflow…" : "Generate workflow";
    if (busy) el("planningStatus").classList.remove("hidden");
    else el("planningStatus").classList.add("hidden");
  }
  function clearDraft() {
    draft = null;
    el("approveTask").disabled = true;
    el("draftIdentity").textContent = "No current validated approval.";
  }
  function invalidate() {
    epoch++;
    clearDraft();
    el("draft").classList.add("hidden");
    el("sceneObservation").textContent = "No inspection executed for the current source/workflow.";
    el("eventList").textContent = "No events for the current workflow.";
    el("notificationList").textContent = "No notifications for the current workflow.";
    el("processedFrame").src = "";
    el("processedFrame").classList.add("hidden");
  }
  function unavailable(reason) {
    invalidate(); latestPoll++; room = null; source = null; version = -1; csrf = null; capabilities = null;
    setPlanningBusy(false);
    el("prompt").value = ""; el("editJson").value = ""; el("draftJson").textContent = "";
    el("sceneQuestion").value = ""; el("sourceIdentity").textContent = "No source selected.";
    el("activeWorkflow").textContent = "No approved workflow.";
    el("jobStatus").textContent = "No active private workspace."; el("progress").value = 0;
    el("juryVideo").value = ""; el("inputVideo").textContent = "";
    if (timer) clearInterval(timer);
    for (const id of controls) el(id).disabled = true;
    const video = el("sourceVideo");
    if (video.pause) video.pause(); video.src = "";
    if (video.removeAttribute) video.removeAttribute("src"); if (video.load) video.load();
    el("sessionStatus").textContent = "Workspace unavailable";
    message(reason);
  }
  function showDraft(r, expectedEpoch) {
    if (expectedEpoch !== epoch || !r || r.status && r.status !== "ok") return;
    if (!r.task || !validSha(r.draft_hash) || !validVersion(r.expected_version) || r.expected_version !== version) {
      clearDraft(); throw new Error("No valid approval identity returned.");
    }
    draft = { task: r.task, draft_hash: r.draft_hash, expected_version: r.expected_version, source, epoch };
    el("draftJson").textContent = JSON.stringify(r.task, null, 2);
    el("editJson").value = JSON.stringify(r.task, null, 2);
    el("draftIdentity").textContent = "Workflow version " + r.expected_version + " · " + r.draft_hash;
    el("draft").classList.remove("hidden");
    el("approveTask").disabled = !capabilities?.workflow;
    el("validateTask").disabled = !capabilities?.workflow;
  }
  function bindSource(sha, metadata) {
    if (!validSha(sha)) throw new Error("Invalid source provenance.");
    source = sha;
    el("sourceVideo").src = owned("/video/source?sha=" + source);
    el("sourceIdentity").textContent = JSON.stringify({source_sha256: source, metadata}, null, 2);
    el("inspectScene").disabled = !capabilities?.scene;
    el("planTask").disabled = !capabilities?.workflow || pending.has("planTask");
  }
  function listText(target, items, empty) {
    const container = el(target); container.textContent = "";
    if (!items.length) { container.textContent = empty; return; }
    for (const item of items) {
      const row = document.createElement("pre"); row.className = "code";
      row.textContent = JSON.stringify(item, null, 2); container.appendChild(row);
      if (target === "eventList" && typeof item.evidence_url === "string" &&
          item.evidence_url.startsWith(owned("/evidence/")) && !item.evidence_url.includes("?")) {
        const link = document.createElement("a");
        link.href = item.evidence_url; link.textContent = "Open timestamped evidence";
        link.target = "_blank"; link.rel = "noopener noreferrer"; container.appendChild(link);
      }
    }
  }
  async function poll() {
    if (!room) return;
    const request = ++latestPoll, requestEpoch = epoch, requestedRoom = room;
    try {
      const state = await api(owned("/state"));
      if (!room || request !== latestPoll || requestEpoch !== epoch) return;
      if (!validVersion(state.task_version) || state.task_version < version) return;
      if (state.video?.sha256 !== source) return;
      if (state.task_version > version) { invalidate(); version = state.task_version; }
      const v = state.video || {};
      const sourceTime = typeof v.meta?.fps === "number" && Number.isFinite(v.meta.fps) && v.meta.fps > 0 ? "source t=" + (v.job_processed / v.meta.fps).toFixed(2) + "s" : "";
      el("jobStatus").textContent = [v.job_status || "idle", v.job_processed + "/" + v.job_total + " frames", sourceTime, v.job_error || ""].filter(Boolean).join(" · ");
      el("progress").value = v.job_total > 0 ? 100 * v.job_processed / v.job_total : 0;
      el("runVideo").disabled = !capabilities?.workflow || !state.task_enabled || v.job_status === "running";
      el("disableTask").disabled = !capabilities?.workflow || !state.task_enabled;
      el("enableTask").disabled = !capabilities?.workflow || !state.task || state.task_enabled;
      el("activeWorkflow").textContent = state.task ? state.task.name + " · " + (state.task_paused ? "paused" : state.task_enabled ? "enabled" : "disabled") + " · version " + version : "No approved workflow.";
      const currentEvents = (state.events || []).filter(e => e.task_version === version);
      const eventIds = new Set(currentEvents.map(e => e.id));
      listText("eventList", currentEvents, "No events for this workflow.");
      listText("notificationList", (state.notifications || []).filter(n => eventIds.has(n.event_id)), "No notifications for this workflow.");
      if (state.metrics?.has_frame && capabilities?.workflow) {
        el("processedFrame").src = owned("/frame?t=" + Date.now());
        el("processedFrame").classList.remove("hidden");
      }
    } catch (e) { if (room !== requestedRoom || requestEpoch !== epoch || request !== latestPoll) return; if (e.auth) unavailable("Private workspace expired. Reload to request a new isolated workspace."); else message(e.message || "Analysis status unavailable; retrying."); }
  }
  function handler(id, fn) {
    el(id).onclick = async () => {
      if (pending.has(id)) return; pending.add(id);
      try { await fn(); } catch (e) { if (e.auth) unavailable("Private workspace expired. Reload to request a new isolated workspace."); else message(e.message || "Operation unavailable."); }
      finally { pending.delete(id); }
    };
  }
  handler("selectVideo", async () => {
    if (!capabilities?.catalog || !room) return;
    invalidate(); const requestEpoch = epoch;
    source = null; el("sourceVideo").src = ""; el("sourceIdentity").textContent = "Validating selected input…";
    el("planTask").disabled = true; el("inspectScene").disabled = true;
    const r = await api(owned("/video/select"), {method:"POST", body:{id:el("inputVideo").value}});
    if (requestEpoch !== epoch || !room) return;
    bindSource(r.sha256, r.meta);
    const s = await api(owned("/state"));
    if (requestEpoch !== epoch || !validVersion(s.task_version) || s.video?.sha256 !== source) return;
    version = s.task_version;
    await poll();
  });
  handler("uploadVideo", async () => {
    if (!capabilities?.catalog || !room) return;
    const file = el("juryVideo").files?.[0];
    if (!file || !/\.mp4$/i.test(file.name) || file.size <= 0 || file.size > 64 * 1024 * 1024) throw new Error("Select an MP4 up to 64 MiB; the backend validates its actual content.");
    invalidate(); const requestEpoch = epoch;
    source = null; el("sourceVideo").src = ""; el("sourceIdentity").textContent = "Validating uploaded input…";
    el("planTask").disabled = true; el("inspectScene").disabled = true;
    const r = await api(owned("/video"), {method:"POST",raw:file});
    if (requestEpoch !== epoch || !room) return;
    bindSource(r.sha256,r.meta);
    const state = await api(owned("/state"));
    if (requestEpoch !== epoch || state.video?.sha256 !== source || !validVersion(state.task_version)) return;
    version = state.task_version; await poll();
  });
  handler("inspectScene", async () => {
    if (!capabilities?.scene || !source || !validVersion(version)) return;
    const requestEpoch = epoch, expectedSource = source, expectedVersion = version;
    el("sceneObservation").textContent = "Inspecting actual sampled images…";
    const r = await api(owned("/scene"), {method:"POST", body:{question:el("sceneQuestion").value, source_sha256:source, expected_version:version}});
    if (requestEpoch !== epoch || expectedSource !== source || expectedVersion !== version) return;
    el("sceneObservation").textContent = JSON.stringify(r, null, 2);
  });
  handler("planTask", async () => {
    if (!capabilities?.workflow || !source || !validVersion(version)) return;
    clearDraft(); const requestEpoch = ++epoch;
    setPlanningBusy(true);
    try {
      const r = await api(owned("/task/plan"), {method:"POST", body:{text:el("prompt").value}});
      if (requestEpoch !== epoch) return;
      if (r.status !== "ok") { message(r.unsupported_reason || r.clarification || r.error || r.explanation || "This request is unsupported or needs clarification."); return; }
      showDraft(r, requestEpoch);
    } finally {
      setPlanningBusy(false);
    }
  });
  el("prompt").addEventListener("input", () => { epoch++; clearDraft(); el("draft").classList.add("hidden"); });
  el("inputVideo").addEventListener("change", () => { epoch++; clearDraft(); el("draft").classList.add("hidden"); });
  el("editJson").addEventListener("input", () => { epoch++; clearDraft(); });
  handler("validateTask", async () => {
    if (!capabilities?.workflow || !source) return;
    const requestEpoch = ++epoch; clearDraft();
    const r = await api(owned("/task/validate"), {method:"POST", body:{task:JSON.parse(el("editJson").value)}});
    showDraft(r, requestEpoch);
  });
  handler("approveTask", async () => {
    if (!capabilities?.workflow || !draft || draft.epoch !== epoch || draft.source !== source || draft.expected_version !== version) return;
    const approved = draft, requestEpoch = epoch, requestedRoom = room; clearDraft();
    const r = await api(owned("/task/activate"), {method:"POST", body:{task:approved.task, draft_hash:approved.draft_hash, expected_version:approved.expected_version}});
    if (requestEpoch !== epoch || requestedRoom !== room) return;
    if (!validVersion(r.task_version)) throw new Error("Activation returned invalid version.");
    invalidate(); version = r.task_version; await poll();
  });
  handler("discardTask", () => { epoch++; clearDraft(); el("draft").classList.add("hidden"); });
  handler("runVideo", async () => { if (capabilities?.workflow && source) { await api(owned("/video/process"), {method:"POST"}); await poll(); } });
  for (const [id, enabled] of [["disableTask",false],["enableTask",true]]) {
    handler(id, async () => {
      if (!capabilities?.workflow || !source) return;
      invalidate(); const requestEpoch = epoch, requestedRoom = room;
      const r = await api(owned("/task/state"), {method:"POST",body:{enabled,paused:false}});
      if (requestEpoch !== epoch || requestedRoom !== room) return;
      if (!validVersion(r.task_version)) throw new Error("State change returned invalid version.");
      version = r.task_version; await poll();
    });
  }
  async function bootstrap() {
    try {
      // Admission failure never falls back to a shared credential or room.
      const r = await api("/api/demo/session", {method:"POST",body:{}});
      if (!r || typeof r.room_id !== "string" || !/^[A-Za-z0-9_-]{8,128}$/.test(r.room_id) ||
          !validVersion(r.task_version) || typeof r.csrf_token !== "string" || r.csrf_token.length < 16) {
        throw new Error("Private workspace admission unavailable.");
      }
      room = r.room_id; version = r.task_version; csrf = r.csrf_token;
      capabilities = {catalog:true,scene:true,workflow:true};
      el("sessionStatus").textContent = "Your private workspace";
      const catalog = await api("/api/live/video-catalog");
      if (!room || !capabilities.catalog || !Array.isArray(catalog.inputs)) throw new Error("Private prepared inputs unavailable.");
      const select = el("inputVideo"); select.textContent = "";
      for (const item of catalog.inputs) {
        if (typeof item.id !== "string" || !/^Camera[1-5]$/.test(item.id)) continue;
        if ([...select.children].some(c => c.value === item.id)) continue;
        const option = document.createElement("option"); option.value = item.id; option.textContent = item.id;
        select.appendChild(option);
      }
      if (!select.children.length) throw new Error("No actual prepared videos are configured.");
      select.disabled = false; el("selectVideo").disabled = false;
      el("juryVideo").disabled = false; el("uploadVideo").disabled = false;
      timer = setInterval(poll, 1000);
    } catch (e) {
      unavailable("Private workspace unavailable. No processing or model call was executed. Reload to try again.");
    }
  }
  bootstrap();
})();
