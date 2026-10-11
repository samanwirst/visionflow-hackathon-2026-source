# VisionFlow AI — Web UI

Next.js (App Router) + TypeScript single-page control platform for the
demo-safe build. No external image assets or web fonts are used; icons are
inline SVG and the system font stack is applied.

## What it does

A calm, guided operator workflow over the real geometric pipeline:

1. **Sources** — upload an MP4 or select an existing recorded clip; run
   processing with honest status/progress and failure surfacing. Live camera is
   shown as unavailable.
2. **Zone workspace** — draw a normalized polygon on the first-frame snapshot.
   The overlay canvas stays aligned with the image on viewport/container resize.
3. **Build & review a rule** — compose a rule with structured controls
   (class · zone · trigger: enter / exit / dwell ≥ s / count > n). Saving stores
   a **disabled** draft and returns the server's canonical JSON; you review that
   exact JSON and must **deliberately approve** it to arm the rule. Editing
   disables the prior rule and creates a new reviewed version, preserving old
   events and evidence. Re-arming a disabled rule shows its JSON for review
   again. Editing and arming are paused while a source is processing (the
   snapshot would be stale); disabling always remains available.
4. **Event timeline** — source-scoped events with video timestamps, tracks,
   confidence and a full evidence-frame viewer.
5. **Local notifications** — persisted actions (local only; no outbound delivery).

Natural-language compilation, Teacher VLM, AutoLearn and foveated input are
shown as **planned / unavailable / unverified** and are disabled server-side in
demo-safe mode. No fabricated events, metrics or savings percentages are shown.

## Develop

```sh
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev   # http://localhost:3000
npm run build                                            # production build (CI)
```

`NEXT_PUBLIC_API_URL` points the UI at the FastAPI backend. UI code holds no
authoritative business logic — every event decision and action is validated and
executed server-side. See the repository [Demo Guide](../../docs/DEMO_GUIDE.md).
