"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ----------------------------------------------------------------- types */

type Source = {
  id: number;
  type: string;
  name: string;
  status: string;
  error: string | null;
  width: number | null;
  height: number | null;
  frame_count: number | null;
  frames_processed: number;
  avg_inference_ms: number | null;
};

type Zone = { id: number; source_id: number; name: string; polygon: [number, number][] };

type Trigger = { type: string; zone_id: number; duration_seconds?: number | null; threshold?: number | null };

type Dsl = {
  name: string;
  source_id: number;
  object_class: string;
  trigger: Trigger;
  action: { type: string; config: Record<string, unknown> };
  visual_predicate?: unknown;
  cooldown_seconds: number;
  requires_approval: boolean;
};

type Workflow = {
  id: number;
  source_id: number;
  name: string;
  enabled: boolean;
  dsl: Dsl;
};

type TrackEventT = {
  id: number;
  source_id: number;
  workflow_id: number;
  type: string;
  class_name: string | null;
  track_id: number | null;
  video_timestamp: number;
  confidence: number;
  detail: string;
  evidence_url: string | null;
  action_status: string;
  visual_status: string | null;
  created_at: string;
};

type NotificationT = { id: number; event_id: number; title: string; message: string; created_at: string };

type ReviewTarget = { workflowId: number; sourceId: number; dsl: Dsl; kind: "new" | "reenable" };

const TRIGGERS = [
  { value: "object_entered_zone", label: "Object enters the zone" },
  { value: "object_exited_zone", label: "Object exits the zone" },
  { value: "dwell_in_zone", label: "Object stays in the zone (dwell ≥ seconds)" },
  { value: "object_count_exceeded", label: "Count in zone strictly exceeds a threshold" },
];

const CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"];

/* ----------------------------------------------------------------- icons */

const I = {
  logo: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ),
  upload: (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M17 8l-5-5-5 5" /><path d="M12 3v12" />
    </svg>
  ),
  video: (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="5" width="15" height="14" rx="2" /><path d="m17 9 5-3v12l-5-3" />
    </svg>
  ),
  zone: (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7l9-4 9 4-9 4-9-4Z" /><path d="M3 7v10l9 4 9-4V7" />
    </svg>
  ),
  play: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
  ),
  check: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
  ),
  edit: (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9" /><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" /></svg>
  ),
  info: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>
  ),
  alert: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></svg>
  ),
  close: (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>
  ),
  shield: (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" /></svg>
  ),
  bell: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /></svg>
  ),
  list: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M8 6h13" /><path d="M8 12h13" /><path d="M8 18h13" /><path d="M3 6h.01" /><path d="M3 12h.01" /><path d="M3 18h.01" /></svg>
  ),
};

/* ----------------------------------------------------------------- component */

export default function Home() {
  const [sources, setSources] = useState<Source[]>([]);
  const [zones, setZones] = useState<Zone[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [events, setEvents] = useState<TrackEventT[]>([]);
  const [notifications, setNotifications] = useState<NotificationT[]>([]);
  const [selectedSource, setSelectedSource] = useState<number | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [busy, setBusy] = useState(false);
  // Verified backend demo-safe capability: true (enforced), false (NOT safe),
  // null (could not verify, e.g. older backend / unreachable). Only `true`
  // allows demo mutations and runs.
  const [demoSafe, setDemoSafe] = useState<boolean | null>(null);
  // A run requested from the UI but not yet observed as "processing"; keeps the
  // run/arm/edit controls disabled during the pending window too.
  const [runPending, setRunPending] = useState<{ id: number; from: string } | null>(null);
  // Read-only JSON inspection of a workflow (used for unavailable visual rules).
  const [jsonView, setJsonView] = useState<Workflow | null>(null);
  // True only once the source-scoped lists (zones/workflows/events) have finished
  // loading for the CURRENTLY selected source. Reset to false on every switch so
  // a previous recording's data can never be rendered or treated as eligible
  // while the new source's requests are still in flight.
  const [scopeReady, setScopeReady] = useState(false);
  // Marks an in-flight form transition (save draft / approve / edit) during which
  // the builder fields must stay locked so a late server copy cannot clobber an
  // instant edit (e.g. the dwell field disappearing mid-fill).
  const [formPending, setFormPending] = useState(false);
  // Tracks the source a source-scoped refresh must still be relevant for, so a
  // stale in-flight response cannot overwrite a newly selected source.
  const selectedSourceRef = useRef<number | null>(null);

  // zone drawing
  const [drawing, setDrawing] = useState(false);
  const [points, setPoints] = useState<[number, number][]>([]);
  const [zoneName, setZoneName] = useState("Service Bay A");
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // rule builder
  const [wfName, setWfName] = useState("Car entered Service Bay A");
  const [wfClass, setWfClass] = useState("car");
  const [wfTrigger, setWfTrigger] = useState("object_entered_zone");
  const [wfZone, setWfZone] = useState<number | null>(null);
  const [wfDuration, setWfDuration] = useState(10);
  const [wfThreshold, setWfThreshold] = useState(2);
  const [review, setReview] = useState<ReviewTarget | null>(null);
  const [editNote, setEditNote] = useState<string | null>(null);

  // evidence modal
  const [modalEvent, setModalEvent] = useState<TrackEventT | null>(null);

  const builderRef = useRef<HTMLDivElement>(null);

  const current = sources.find((s) => s.id === selectedSource) || null;
  const isProcessing = current?.status === "processing";
  // "Analyzing" covers the pending request window plus the actual processing
  // run; run/arming/edit are unavailable during it, Disable stays available.
  const analyzing = isProcessing || runPending?.id === selectedSource;
  const canMutate = demoSafe === true;

  // Source-scoped views. A record only belongs to the current workspace when its
  // source_id matches the selected source AND the scoped lists have finished
  // loading for that source. This keeps a previous recording's zones, rules and
  // events (and the notifications derived from those events) from ever appearing
  // under a newly selected source while its requests are still in flight.
  const scopeReadyNow = scopeReady && selectedSource != null;
  const scopeLoading = selectedSource != null && !scopeReady;
  const scopedZones = useMemo(
    () => (scopeReadyNow ? zones.filter((z) => z.source_id === selectedSource) : []),
    [zones, selectedSource, scopeReadyNow],
  );
  const scopedWorkflows = useMemo(
    () => (scopeReadyNow ? workflows.filter((w) => w.source_id === selectedSource) : []),
    [workflows, selectedSource, scopeReadyNow],
  );
  const scopedEvents = useMemo(
    () => (scopeReadyNow ? events.filter((e) => e.source_id === selectedSource) : []),
    [events, selectedSource, scopeReadyNow],
  );
  // Run requires THIS source to have an approved ENABLED geometric rule whose
  // data is ready — never one inherited from the previously selected source.
  const approvedGeometric = useMemo(
    () => scopedWorkflows.some((w) => w.enabled && w.dsl.visual_predicate == null),
    [scopedWorkflows],
  );

  // A list-returning GET that fails or returns a non-array is treated as a
  // recoverable error (keep prior state) rather than crashing the render.
  const getArray = async (url: string) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    const body = await r.json();
    if (!Array.isArray(body)) throw new Error(`${url} did not return a list`);
    return body;
  };

  const apiError = async (r: Response) => {
    let detail = `request failed (${r.status})`;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep default */
    }
    setError(detail);
  };

  const refresh = useCallback(async () => {
    const sid = selectedSource;
    try {
      // Verify the backend actually enforces demo-safe mode. A missing field
      // (older backend) or non-ok response leaves it unverified (null).
      const health = await fetch(`${API}/api/health`)
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
      setDemoSafe(health && typeof health.demo_safe === "boolean" ? health.demo_safe : null);

      const [s, n] = await Promise.all([
        getArray(`${API}/api/sources`),
        getArray(`${API}/api/notifications?limit=100`),
      ]);
      setSources(s as Source[]);
      setNotifications(n as NotificationT[]);

      if (sid) {
        try {
          const [z, w, e] = await Promise.all([
            getArray(`${API}/api/zones?source_id=${sid}`),
            getArray(`${API}/api/workflows?source_id=${sid}`),
            getArray(`${API}/api/events?source_id=${sid}&limit=100`),
          ]);
          // Drop the response if the user switched sources (or cleared the
          // selection) while it was in flight — a stale/null-source callback must
          // never install another recording's data.
          if (selectedSourceRef.current === sid) {
            setZones(z as Zone[]);
            setWorkflows(w as Workflow[]);
            setEvents(e as TrackEventT[]);
            setScopeReady(true);
          }
        } catch {
          // Source-scoped data (e.g. workflows) failed to load — surface it and
          // keep the previous lists rather than crashing on a non-array body.
          if (selectedSourceRef.current === sid) {
            setError("Could not refresh this source's workflows/events (the API returned an error).");
          }
        }
      } else {
        setZones([]);
        setWorkflows([]);
        setEvents([]);
        setScopeReady(false);
      }
      setLoaded(true);
      setError((prev) => (prev && prev.startsWith("Cannot reach") ? null : prev));
    } catch {
      setError(`Cannot reach the API at ${API}. Is the backend running?`);
      setDemoSafe(null);
      setLoaded(true);
    }
  }, [selectedSource]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, [refresh]);

  // Switching source clears the previous recording's scoped lists and every
  // dependent surface (draft/approval, drawing, edit note, read-only JSON and
  // evidence modals) so nothing from the old source survives under the new one.
  // scopeReady stays false until the new source's data loads.
  useEffect(() => {
    selectedSourceRef.current = selectedSource;
    setScopeReady(false);
    setZones([]);
    setWorkflows([]);
    setEvents([]);
    setReview(null);
    setEditNote(null);
    setFormPending(false);
    setJsonView(null);
    setModalEvent(null);
    setPoints([]);
    setDrawing(false);
    setWfZone(null);
  }, [selectedSource]);

  // Clear the pending-run marker once the run has actually started (or the
  // source otherwise left the state it was in when Run was pressed).
  useEffect(() => {
    if (runPending && current && current.id === runPending.id && current.status !== runPending.from) {
      setRunPending(null);
    }
  }, [current, runPending]);

  /* -------------------------------------------------- canvas overlay */

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.clientWidth) return;
    const w = img.clientWidth;
    const h = img.clientHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);

    for (const z of scopedZones) {
      ctx.strokeStyle = "#4f8cff";
      ctx.fillStyle = "rgba(79, 140, 255, 0.14)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      z.polygon.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x * w, y * h) : ctx.lineTo(x * w, y * h)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      const [lx, ly] = z.polygon[0];
      ctx.fillStyle = "#4f8cff";
      ctx.font = "600 12px system-ui";
      ctx.fillText(z.name, lx * w + 6, ly * h + 16);
    }

    if (points.length > 0) {
      ctx.strokeStyle = "#a78bfa";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x * w, y * h) : ctx.lineTo(x * w, y * h)));
      ctx.stroke();
      ctx.setLineDash([]);
      for (const [x, y] of points) {
        ctx.fillStyle = "#a78bfa";
        ctx.beginPath();
        ctx.arc(x * w, y * h, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [scopedZones, points]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  // Keep the overlay aligned with the snapshot on any container/viewport resize.
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const ro = new ResizeObserver(() => drawOverlay());
    ro.observe(img);
    window.addEventListener("resize", drawOverlay);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", drawOverlay);
    };
  }, [drawOverlay, selectedSource]);

  const canvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setPoints((p) => [...p, [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))]]);
  };

  /* -------------------------------------------------- actions */

  const upload = async (file: File) => {
    if (!canMutate) return;
    setUploading(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await fetch(`${API}/api/sources/upload`, { method: "POST", body: form });
      if (!r.ok) return apiError(r);
      const s = await r.json();
      setSelectedSource(s.id);
      await refresh();
    } catch {
      setError("Upload failed — the API may be unreachable.");
    } finally {
      setUploading(false);
    }
  };

  const saveZone = async () => {
    if (!canMutate || !selectedSource || points.length < 3) return;
    const r = await fetch(`${API}/api/zones`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: selectedSource, name: zoneName, polygon: points }),
    });
    if (!r.ok) return apiError(r);
    setPoints([]);
    setDrawing(false);
    refresh();
  };

  // Any builder edit invalidates ALL pending review kinds (a new draft OR a
  // re-enable) so no stale approval can be submitted against changed parameters.
  const touchBuilder = () => setReview(null);

  const saveDraft = async () => {
    if (!canMutate || formPending) return;
    if (!selectedSource || !wfZone) {
      setError("Pick a zone for this rule first.");
      return;
    }
    const sid = selectedSource;
    setError(null);
    setBusy(true);
    setFormPending(true);
    const trigger: Record<string, unknown> = { type: wfTrigger, zone_id: wfZone };
    if (wfTrigger === "dwell_in_zone") trigger.duration_seconds = wfDuration;
    if (wfTrigger === "object_count_exceeded") trigger.threshold = wfThreshold;
    const payload = {
      name: wfName,
      source_id: selectedSource,
      object_class: wfClass,
      trigger,
      action: { type: "create_notification", config: { title: wfName } },
      cooldown_seconds: 10,
    };
    try {
      const r = await fetch(`${API}/api/workflows`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) return apiError(r);
      const wf: Workflow = await r.json();
      // If the user switched sources mid-save, do not install a review that
      // belongs to the previous source.
      if (selectedSourceRef.current !== sid || wf.source_id !== sid) return;
      setReview({ workflowId: wf.id, sourceId: wf.source_id, dsl: wf.dsl, kind: "new" });
      setEditNote(null);
      await refresh();
    } finally {
      setBusy(false);
      setFormPending(false);
    }
  };

  const approve = async () => {
    if (!review || !canMutate || formPending) return;
    // Never approve a review that no longer belongs to the selected source (e.g.
    // a switch happened before the user clicked approve).
    if (review.sourceId !== selectedSource) {
      setReview(null);
      return;
    }
    // Defense-in-depth: never activate a rule carrying a visual predicate from
    // the UI (the backend also rejects this). These are view/disable only.
    if (review.dsl.visual_predicate != null) {
      setError("Teacher-routed visual rules cannot be armed in this demo.");
      setReview(null);
      return;
    }
    const sid = selectedSource;
    setBusy(true);
    setFormPending(true);
    try {
      const r = await fetch(`${API}/api/workflows/${review.workflowId}/enable`, { method: "POST" });
      if (!r.ok) return apiError(r);
      // A delayed enable must never install an old-source approval.
      if (selectedSourceRef.current !== sid) {
        setReview(null);
        return;
      }
      setReview(null);
      await refresh();
    } finally {
      setBusy(false);
      setFormPending(false);
    }
  };

  const disableWorkflow = async (wf: Workflow) => {
    const r = await fetch(`${API}/api/workflows/${wf.id}/disable`, { method: "POST" });
    if (!r.ok) return apiError(r);
    setReview((rt) => (rt?.workflowId === wf.id ? null : rt));
    refresh();
  };

  const reviewEnable = (wf: Workflow) => {
    // Visual-predicate rules are unavailable: never offer re-arming for them.
    // Only re-arm a rule that belongs to the selected source.
    if (wf.dsl.visual_predicate != null || !canMutate || analyzing || formPending) return;
    if (wf.source_id !== selectedSource) return;
    setReview({ workflowId: wf.id, sourceId: wf.source_id, dsl: wf.dsl, kind: "reenable" });
    builderRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const editWorkflow = async (wf: Workflow) => {
    // Never copy a visual rule into the geometric builder (that silently drops
    // its visual predicate). Edit is unavailable during analysis, during another
    // form transition, or for a rule that is not bound to the selected source.
    if (wf.dsl.visual_predicate != null || !canMutate || analyzing || formPending) return;
    if (wf.source_id !== selectedSource) return;
    const sid = selectedSource;
    // Lock the builder for the whole transition so an instant edit cannot be
    // clobbered by the awaited copy (e.g. the dwell field disappearing mid-fill).
    setFormPending(true);
    // Guide the user while the previous rule is being disabled and reloaded.
    setEditNote(
      `Editing “${wf.name}”. The previous rule (#${wf.id}) has been disabled; its past ` +
        `events and evidence are preserved. Saving creates a new reviewed version.`,
    );
    try {
      if (wf.enabled) {
        const r = await fetch(`${API}/api/workflows/${wf.id}/disable`, { method: "POST" });
        if (!r.ok) {
          setEditNote(null);
          return apiError(r);
        }
      }
      // Abort cleanly if the user switched sources during the disable round-trip;
      // never install an old-source rule into the new source's builder.
      if (selectedSourceRef.current !== sid) {
        setEditNote(null);
        return;
      }
      setWfName(wf.name);
      setWfClass(wf.dsl.object_class);
      setWfTrigger(wf.dsl.trigger.type);
      setWfZone(wf.dsl.trigger.zone_id);
      // Preserve the exact valid dwell value (fractional seconds too); do not round.
      if (wf.dsl.trigger.duration_seconds) setWfDuration(wf.dsl.trigger.duration_seconds);
      if (wf.dsl.trigger.threshold) setWfThreshold(wf.dsl.trigger.threshold);
      setReview(null);
      await refresh();
      if (selectedSourceRef.current !== sid) return;
      builderRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } finally {
      setFormPending(false);
    }
  };

  const processSource = async (s: Source) => {
    if (!canMutate || analyzing || formPending) return; // never start a second run on top of one
    // Only run the selected source, and only once its own data is ready — never
    // inherit another recording's approval while this source is still loading.
    if (s.id !== selectedSource || !scopeReady) return;
    if (!approvedGeometric) {
      setError("Approve an enabled geometric rule for this source before running it.");
      return;
    }
    setError(null);
    setRunPending({ id: s.id, from: s.status });
    const r = await fetch(`${API}/api/sources/${s.id}/process`, { method: "POST" });
    if (!r.ok) {
      setRunPending(null);
      return apiError(r);
    }
    refresh();
  };

  /* -------------------------------------------------- derived */

  // Notifications are scoped through the source's own events; during loading
  // scopedEvents is empty, so no previous source's notifications leak through.
  const sourceEventIds = useMemo(() => new Set(scopedEvents.map((e) => e.id)), [scopedEvents]);
  const scopedNotifications = useMemo(
    () => (selectedSource ? notifications.filter((n) => sourceEventIds.has(n.event_id)) : notifications),
    [notifications, selectedSource, sourceEventIds],
  );

  const ruleSummary = (dsl: Dsl) => {
    const t = dsl.trigger;
    let s = `${dsl.object_class} · `;
    if (t.type === "object_entered_zone") s += "enters zone";
    else if (t.type === "object_exited_zone") s += "exits zone";
    else if (t.type === "dwell_in_zone") s += `dwell ≥ ${t.duration_seconds ?? "?"}s`;
    else if (t.type === "object_count_exceeded") s += `count > ${t.threshold ?? "?"}`;
    else s += t.type;
    return s;
  };

  /* -------------------------------------------------- render */

  return (
    <>
      <header className="appbar">
        <div className="appbar-inner">
          <div className="brand">
            <div className="brand-mark">{I.logo}</div>
            <div className="brand-text">
              <h1>VisionFlow AI</h1>
              <p>Visual analytics control platform · recorded video</p>
            </div>
          </div>
          <div className="chips">
            {demoSafe === true ? (
              <span className="chip on">{I.shield} Demo-safe · verified</span>
            ) : demoSafe === false ? (
              <span className="chip danger">{I.alert} Not demo-safe</span>
            ) : (
              <span className="chip warn">{I.alert} Demo-safe · unverified</span>
            )}
            <span className="chip planned"><span className="dot" /> NL agent · planned</span>
            <span className="chip planned"><span className="dot" /> Teacher VLM · unavailable</span>
            <span className="chip planned"><span className="dot" /> AutoLearn · planned</span>
            <span className="chip unavailable"><span className="dot" /> Foveated · unverified</span>
          </div>
        </div>
      </header>

      <main className="shell">
        <p className="lede" style={{ marginTop: 0, marginBottom: 20 }}>
          Select or upload a recorded clip, draw a monitored zone, build a detection rule, review the
          exact server-validated JSON, then approve it to arm the real detector → tracker → temporal-rule
          pipeline. Events, evidence frames and local notifications appear below.
        </p>

        {error && (
          <div className="banner error">
            {I.alert}
            <span>{error}</span>
            <button className="ghost sm" onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}

        {loaded && demoSafe !== true && (
          <div className="banner warn">
            {I.alert}
            <span>
              {demoSafe === false
                ? "The connected API reports it is NOT in demo-safe mode. Demo uploads, rule arming and runs are disabled so this UI never drives the experimental backend paths. Restart the backend with VISIONFLOW_DEMO_SAFE=true."
                : "Could not verify demo-safe mode on the connected API (the backend did not report the capability). Demo uploads, rule arming and runs are disabled until it is verified."}
            </span>
          </div>
        )}

        {/* 1 · Sources ------------------------------------------------ */}
        <section className="card">
          <div className="card-head">
            <span className="idx">1</span>
            <h2>Video sources</h2>
            <span className="sub">recorded clips only — live camera unavailable</span>
          </div>

          <div className="row" style={{ marginBottom: 14 }}>
            <label className="secondary" style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "8px 15px", borderRadius: 8, cursor: uploading || !canMutate ? "not-allowed" : "pointer", border: "1px solid var(--border-strong)", opacity: canMutate ? 1 : 0.55 }}>
              {uploading ? <span className="spinner" /> : I.upload}
              <span>{uploading ? "Uploading…" : "Upload MP4"}</span>
              <input
                type="file"
                accept="video/mp4"
                disabled={uploading || !canMutate}
                style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
              />
            </label>
            <span className="faint">
              Upload a recorded MP4. A prepared demo clip (real photo, synthetic motion, analyzed with
              real YOLO) and the setup commands are described in the demo guide.
            </span>
          </div>

          {!loaded ? (
            <div className="empty"><span className="spinner" /><div className="desc">Loading sources…</div></div>
          ) : sources.length === 0 ? (
            <div className="empty">
              {I.video}
              <div className="title">No video sources yet</div>
              <div className="desc">Upload an MP4 clip to begin. Detection runs genuine YOLO inference on sampled frames.</div>
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>#</th><th>Name</th><th>Status</th>
                  <th title="Frames sampled and analyzed by YOLO so far (not a completion target)">Samples analyzed</th>
                  <th title="Total frames reported by the clip's container metadata">Clip frames</th>
                  <th>Inference</th><th />
                </tr>
              </thead>
              <tbody>
                {sources.map((s) => (
                  <tr key={s.id} className={`selectable ${s.id === selectedSource ? "selected" : ""}`} onClick={() => setSelectedSource(s.id)}>
                    <td>{s.id}</td>
                    <td>{s.name}</td>
                    <td>
                      <span className={`badge ${s.status}`}><span className="dot" />{s.status}</span>
                      {s.error && <div className="faint" style={{ marginTop: 3 }}>{s.error}</div>}
                    </td>
                    <td>{s.frames_processed || "—"}</td>
                    <td>{s.frame_count ? `~${s.frame_count}` : "—"}</td>
                    <td>{s.avg_inference_ms ? `${s.avg_inference_ms.toFixed(0)} ms` : "—"}</td>
                    <td className="row" style={{ justifyContent: "flex-end" }} onClick={(e) => e.stopPropagation()}>
                      <button className="secondary sm" onClick={() => setSelectedSource(s.id)}>Select</button>
                      {s.id === selectedSource && (
                        <button
                          className="sm"
                          disabled={analyzing || !approvedGeometric || !canMutate || formPending || !scopeReady}
                          title={
                            scopeLoading
                              ? "Loading this source's rules…"
                              : !approvedGeometric
                                ? "Approve an enabled geometric rule for this source first"
                                : undefined
                          }
                          onClick={() => processSource(s)}
                        >
                          {analyzing ? <><span className="spinner" />Analyzing</> : <>{I.play} Run</>}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {current && (
            <div className="hint" style={{ marginTop: 14 }}>
              {I.info}
              <span>
                <strong>{current.name}</strong> selected ({current.width}×{current.height}). Status:{" "}
                <strong>{current.status}</strong>.{" "}
                {analyzing
                  ? "Analysis in progress — rule editing, arming and re-running are paused until it completes."
                  : "Detection samples frames and runs real YOLO; only the demo clip's motion is synthetic."}
              </span>
            </div>
          )}

          {current && scopeLoading && (
            <div className="hint" style={{ marginTop: 10 }}>
              <span className="spinner" />
              <span>Loading this source&apos;s zones, rules and events…</span>
            </div>
          )}

          {current && !scopeLoading && !approvedGeometric && !analyzing && (
            <div className="hint warn" style={{ marginTop: 10 }}>
              {I.info}
              <span>Build, review and <strong>approve</strong> a geometric rule below before running this source.</span>
            </div>
          )}

          {current && analyzing && (
            <div style={{ marginTop: 14 }}>
              <div className="progress-track"><div className="progress-bar indeterminate" /></div>
              <p className="faint" style={{ marginTop: 6 }}>
                Analyzing on real YOLO — {current.frames_processed || 0} frame
                {current.frames_processed === 1 ? "" : "s"} sampled so far. Total sampled frames depend on the
                clip length and backend sampling rate, so completion is shown as it progresses rather than as a
                fixed percentage.
              </p>
            </div>
          )}
        </section>

        {current ? (
          <div className="grid-2">
            {/* 2 · Workspace ----------------------------------------- */}
            <section className="card" style={{ marginBottom: 0 }}>
              <div className="card-head">
                <span className="idx">2</span>
                <h2>Zone workspace</h2>
                <span className="sub">normalized polygon on first frame</span>
              </div>

              <div className={`canvas-wrap ${drawing ? "drawing" : ""}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={imgRef}
                  src={`${API}/api/sources/${current.id}/snapshot`}
                  alt="first frame of the selected video"
                  onLoad={drawOverlay}
                />
                <canvas ref={canvasRef} onClick={canvasClick} />
                {drawing && <div className="canvas-overlay-hint">Click to add points · {points.length} placed</div>}
              </div>

              <div className="row" style={{ marginTop: 12 }}>
                {!drawing ? (
                  <>
                    <input value={zoneName} onChange={(e) => setZoneName(e.target.value)} placeholder="Zone name" style={{ flex: 1, minWidth: 140 }} />
                    <button className="secondary" onClick={() => { setDrawing(true); setPoints([]); }}>{I.zone} Draw zone</button>
                  </>
                ) : (
                  <>
                    <button className="success" disabled={points.length < 3} onClick={saveZone}>{I.check} Save zone ({points.length})</button>
                    <button className="ghost" onClick={() => { setPoints([]); }}>Clear points</button>
                    <button className="ghost" onClick={() => { setDrawing(false); setPoints([]); }}>Cancel</button>
                  </>
                )}
              </div>

              {scopedZones.length > 0 && (
                <div className="tag-list" style={{ marginTop: 12 }}>
                  {scopedZones.map((z) => (
                    <span key={z.id} className="badge neutral">{z.name} · #{z.id}</span>
                  ))}
                </div>
              )}
            </section>

            {/* 3 · Rule builder -------------------------------------- */}
            <section className="card" style={{ marginBottom: 0 }} ref={builderRef}>
              <div className="card-head">
                <span className="idx">3</span>
                <h2>Build & review a rule</h2>
              </div>

              <div className="hint" style={{ marginBottom: 14 }}>
                {I.info}
                <span>
                  Natural-language compilation is <strong>planned</strong> and disabled in demo-safe mode.
                  Compose the rule below; the server returns the canonical JSON for you to review before activation.
                </span>
              </div>

              {editNote && <div className="hint warn" style={{ marginBottom: 14 }}>{I.edit}<span>{editNote}</span></div>}

              <div className="form-grid">
                <label className="field">Rule name
                  <input value={wfName} onChange={(e) => { setWfName(e.target.value); touchBuilder(); }} disabled={analyzing || !canMutate || formPending} />
                </label>
                <label className="field">Object class
                  <select value={wfClass} onChange={(e) => { setWfClass(e.target.value); touchBuilder(); }} disabled={analyzing || !canMutate || formPending}>
                    {CLASSES.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <label className="field">Zone
                  <select value={wfZone ?? ""} onChange={(e) => { setWfZone(Number(e.target.value) || null); touchBuilder(); }} disabled={analyzing || !canMutate || formPending}>
                    <option value="">— select zone —</option>
                    {scopedZones.map((z) => <option key={z.id} value={z.id}>{z.name}</option>)}
                  </select>
                </label>
                <label className="field" style={{ gridColumn: "1 / -1" }}>Trigger
                  <select value={wfTrigger} onChange={(e) => { setWfTrigger(e.target.value); touchBuilder(); }} disabled={analyzing || !canMutate || formPending}>
                    {TRIGGERS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                  </select>
                </label>
                {wfTrigger === "dwell_in_zone" && (
                  <label className="field">Dwell seconds (≥)
                    <input type="number" min={1} step="any" value={wfDuration} onChange={(e) => { setWfDuration(Number(e.target.value)); touchBuilder(); }} disabled={analyzing || !canMutate || formPending} />
                  </label>
                )}
                {wfTrigger === "object_count_exceeded" && (
                  <label className="field">Threshold (count &gt;)
                    <input type="number" min={1} value={wfThreshold} onChange={(e) => { setWfThreshold(Number(e.target.value)); touchBuilder(); }} disabled={analyzing || !canMutate || formPending} />
                  </label>
                )}
              </div>

              <div className="row end" style={{ marginTop: 14 }}>
                <button onClick={saveDraft} disabled={analyzing || busy || !wfZone || !canMutate || formPending}>
                  {busy ? <span className="spinner" /> : null} Save draft for review
                </button>
              </div>

              {scopeLoading && (
                <p className="faint" style={{ marginTop: 10, textAlign: "right" }}>Loading this source&apos;s zones and rules…</p>
              )}
              {!scopeLoading && scopedZones.length === 0 && (
                <p className="faint" style={{ marginTop: 10, textAlign: "right" }}>Draw a zone first to enable rule creation.</p>
              )}
              {formPending && !analyzing && (
                <p className="faint" style={{ marginTop: 10, textAlign: "right" }}>Applying the change — the builder is locked until it loads.</p>
              )}
              {analyzing && (
                <p className="faint" style={{ marginTop: 10, textAlign: "right" }}>Paused while this source is being analyzed (the frame snapshot would be stale).</p>
              )}

              {review && review.sourceId === selectedSource && (
                <div className="review">
                  <div className="review-head" style={{ color: "var(--accent-hover)" }}>
                    {I.shield}
                    {review.kind === "new" ? "Review the server-validated draft" : "Review before re-arming"}
                  </div>
                  <p className="faint" style={{ marginTop: 0, marginBottom: 10 }}>
                    This is exactly what the backend validated and will execute — no client logic is trusted.
                    Approve to arm it (workflow #{review.workflowId}, currently disabled).
                  </p>
                  <pre className="code">{JSON.stringify(review.dsl, null, 2)}</pre>
                  <div className="row end" style={{ marginTop: 12 }}>
                    <button className="ghost" onClick={() => setReview(null)}>Discard</button>
                    <button className="success" onClick={approve} disabled={busy || analyzing || !canMutate || formPending}>
                      {busy ? <span className="spinner" /> : I.check} Approve &amp; activate
                    </button>
                  </div>
                </div>
              )}

              {scopedWorkflows.length > 0 && (
                <>
                  <div className="divider" />
                  <table>
                    <thead><tr><th>#</th><th>Rule</th><th>State</th><th /></tr></thead>
                    <tbody>
                      {scopedWorkflows.map((w) => {
                        const isVisual = w.dsl.visual_predicate != null;
                        return (
                          <tr key={w.id}>
                            <td>{w.id}</td>
                            <td>
                              {w.name}
                              <div className="faint">{ruleSummary(w.dsl)}</div>
                              {isVisual && (
                                <div className="faint" style={{ color: "var(--warn)" }}>
                                  Teacher-routed visual rule — unavailable in demo-safe mode (it never runs and
                                  cannot be armed or edited here).
                                </div>
                              )}
                            </td>
                            <td>
                              {isVisual ? (
                                <span className="badge visual"><span className="dot" />unavailable</span>
                              ) : (
                                <span className={`badge ${w.enabled ? "enabled" : "disabled"}`}><span className="dot" />{w.enabled ? "armed" : "disabled"}</span>
                              )}
                            </td>
                            <td className="row" style={{ justifyContent: "flex-end" }}>
                              {isVisual ? (
                                <>
                                  {w.enabled && <button className="danger sm" onClick={() => disableWorkflow(w)}>Disable</button>}
                                  <button className="ghost sm" onClick={() => setJsonView(w)}>{I.info} View JSON</button>
                                </>
                              ) : (
                                <>
                                  {w.enabled ? (
                                    <button className="danger sm" onClick={() => disableWorkflow(w)}>Disable</button>
                                  ) : (
                                    <button className="secondary sm" disabled={analyzing || !canMutate || formPending} onClick={() => reviewEnable(w)}>Review &amp; arm</button>
                                  )}
                                  <button className="ghost sm" disabled={analyzing || !canMutate || formPending} onClick={() => editWorkflow(w)}>{I.edit} Edit</button>
                                </>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </>
              )}
            </section>
          </div>
        ) : (
          loaded && sources.length > 0 && (
            <section className="card">
              <div className="empty">
                {I.zone}
                <div className="title">Select a source to open the workspace</div>
                <div className="desc">Pick a clip above to draw zones and build rules.</div>
              </div>
            </section>
          )
        )}

        {/* 4 · Events ------------------------------------------------ */}
        <section className="card" style={{ marginTop: 18 }}>
          <div className="card-head">
            <span className="idx">4</span>
            <h2>Event timeline</h2>
            <span className="sub">{current ? `source #${current.id}` : "select a source"}</span>
          </div>
          {!current ? (
            <div className="empty">{I.list}<div className="title">No source selected</div><div className="desc">Events are scoped to the selected source.</div></div>
          ) : scopeLoading ? (
            <div className="empty"><span className="spinner" /><div className="desc">Loading this source&apos;s events…</div></div>
          ) : scopedEvents.length === 0 ? (
            <div className="empty">{I.list}<div className="title">No events yet</div><div className="desc">Draw a zone, build and approve a rule, then run the source.</div></div>
          ) : (
            <table>
              <thead>
                <tr><th>#</th><th>Event</th><th>Video time</th><th>Conf.</th><th>Evidence</th><th>Action</th></tr>
              </thead>
              <tbody>
                {scopedEvents.map((e) => (
                  <tr key={e.id}>
                    <td>{e.id}</td>
                    <td>{e.detail || e.type}</td>
                    <td>{e.video_timestamp.toFixed(1)}s</td>
                    <td>{(e.confidence * 100).toFixed(0)}%</td>
                    <td>
                      {e.evidence_url ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img className="evidence-thumb" src={`${API}${e.evidence_url}`} alt="evidence frame" onClick={() => setModalEvent(e)} />
                      ) : <span className="faint">—</span>}
                    </td>
                    <td>
                      <span className={`badge ${e.action_status === "done" ? "done" : "neutral"}`}>
                        {e.action_status === "done" && <span className="dot" />}{e.action_status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* 5 · Notifications ----------------------------------------- */}
        <section className="card">
          <div className="card-head">
            <span className="idx">5</span>
            <h2>Local notifications</h2>
            <span className="sub">executed actions — local only, no outbound delivery</span>
          </div>
          {scopedNotifications.length === 0 ? (
            <div className="empty">{I.bell}<div className="title">No notifications yet</div><div className="desc">A confirmed event fires the authorized local notification action.</div></div>
          ) : (
            <table>
              <thead><tr><th>#</th><th>Title</th><th>Message</th><th>At</th></tr></thead>
              <tbody>
                {scopedNotifications.map((n) => (
                  <tr key={n.id}>
                    <td>{n.id}</td>
                    <td>{n.title}</td>
                    <td className="muted">{n.message}</td>
                    <td className="faint">{new Date(n.created_at).toLocaleTimeString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* 6 · Architecture note ------------------------------------- */}
        <section className="card">
          <div className="card-head">
            <span className="idx">6</span>
            <h2>Architecture &amp; honest status</h2>
          </div>
          <p className="muted" style={{ marginTop: 0 }}>
            VisionFlow follows an OpenVector-inspired hierarchy:{" "}
            <strong>cheap continuous detection → tracking → temporal zone rules</strong> (the path running
            in this demo) → a selected future <strong>Teacher VLM</strong> on eligible candidate events →
            a verified dataset and a lightweight <strong>Student</strong>.
          </p>
          <div className="tag-list" style={{ marginTop: 6 }}>
            <span className="badge enabled"><span className="dot" />Detector + tracker + temporal rules · active</span>
            <span className="badge visual">NL agent · planned</span>
            <span className="badge visual">Teacher VLM · unavailable</span>
            <span className="badge visual">AutoLearn / Student · planned</span>
            <span className="badge visual">Foveated input · unverified</span>
          </div>
          <p className="faint" style={{ marginTop: 12 }}>
            Zero Teacher calls here reflect the geometric-only path, not a measured cost benchmark. Teacher,
            Student, foveated inputs and any GPU savings are <strong>UNVERIFIED</strong>; no percentage savings
            is claimed. Demo-safe mode is enforced server-side: the NL compiler is never invoked and visual
            predicates cannot reach a provider.
          </p>
        </section>
      </main>

      {/* read-only workflow JSON (e.g. an unavailable visual rule) */}
      {jsonView && (
        <div className="modal-backdrop" onClick={() => setJsonView(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>Workflow #{jsonView.id} · read-only</h3>
              <button className="ghost sm" onClick={() => setJsonView(null)}>{I.close}</button>
            </div>
            <div className="modal-body">
              <p className="faint" style={{ marginTop: 0 }}>
                Stored definition. This rule carries a Teacher-routed visual predicate and is unavailable in
                demo-safe mode — it is shown for inspection only and cannot be armed or edited here.
              </p>
              <pre className="code">{JSON.stringify(jsonView.dsl, null, 2)}</pre>
            </div>
          </div>
        </div>
      )}

      {/* evidence modal */}
      {modalEvent && (
        <div className="modal-backdrop" onClick={() => setModalEvent(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>Evidence · event #{modalEvent.id}</h3>
              <button className="ghost sm" onClick={() => setModalEvent(null)}>{I.close}</button>
            </div>
            <div className="modal-body">
              {modalEvent.evidence_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={`${API}${modalEvent.evidence_url}`} alt="full evidence frame" />
              )}
              <div className="meta-grid">
                <div><div className="k">Event</div><div className="v">{modalEvent.detail || modalEvent.type}</div></div>
                <div><div className="k">Class</div><div className="v">{modalEvent.class_name ?? "—"}</div></div>
                <div><div className="k">Track</div><div className="v">{modalEvent.track_id ?? "—"}</div></div>
                <div><div className="k">Video time</div><div className="v">{modalEvent.video_timestamp.toFixed(2)}s</div></div>
                <div><div className="k">Confidence</div><div className="v">{(modalEvent.confidence * 100).toFixed(0)}%</div></div>
                <div><div className="k">Action</div><div className="v">{modalEvent.action_status}</div></div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
