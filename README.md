# 🎥 EYEWITNESS — Physics-First Collision Fault Analysis

> **Fault isn't who hit whom — it's who could have stopped and didn't.**
> Eyewitness reconstructs who is at fault in a vehicle collision from ordinary dashcam
> video by separating **deterministic facts** from **AI judgment**. Built by **Team Black Box**
> for Beta Hack · Physical AI 2026.

## 🔗 Live URLs

| What | URL |
|------|-----|
| 🟣 **Live dashboard** (Butterbase-hosted) | https://eyewitness.butterbase.dev |
| 🟢 **Interactive analyzer** (Render) | https://eyewitness-pc9a.onrender.com |
| 🎬 **Demo video** (Loom) | https://www.loom.com/share/635c6d482f734a0daa581d3fbd3add52 |
| 📊 **Pitch deck** (Gamma) | https://gamma.app/docs/EYEWITNESS-qomnwxmoib6txbh |
| 🧠 **Data API** (Butterbase) | `https://api.butterbase.ai/v1/app_46yxrt8czo59` |
| ⚙️ **Aggregation function** | `…/v1/app_46yxrt8czo59/fn/get_runs` |
| 💻 **Source** | https://github.com/harsh543/eyewitness |

## 🏗️ Architecture — Render (compute) + Butterbase (backend)

The whole design flows from one decision: **YOLO11 + OpenCV + PyTorch is heavy Python that
can't run on Butterbase functions (TypeScript/Deno).** So **compute lives on Render**, and
**Butterbase owns data + serving.** The two planes are fully decoupled — the dashboard keeps
serving every past case even if compute is down, and your laptop can run the *same* pipeline
into the *same* backend.

```
                        ┌──────────────────────────────────────────┐
                        │              YOU / JUDGES                  │
                        └───────┬───────────────────────┬───────────┘
                                │ upload video          │ view results
                                ▼                       ▼
      ╔═════════════════════════════════════╗   ╔══════════════════════════════╗
      ║          RENDER  (compute)          ║   ║      BUTTERBASE  (backend)    ║
      ║  ───────────────────────────────    ║   ║  ──────────────────────────   ║
      ║  Docker · 2 GB · Python             ║   ║  • Postgres DB (6 tables)     ║
      ║                                     ║   ║  • get_runs serverless fn     ║
      ║   Gradio UI (app.py)                ║   ║  • Static dashboard hosting   ║
      ║      │                              ║   ║  • Auto REST API              ║
      ║      ▼                              ║   ╚══════════════════════════════╝
      ║   analyze_clip()                    ║          ▲                  │
      ║   ① YOLO11 + ByteTrack  (CV facts)  ║          │ writes           │ reads
      ║   ② avoidability physics            ║          │ (REST + API key) │ (get_runs)
      ║   ③ Claude VLM verdict ──────────┐  ║          │                  │
      ║                                  │  ║──────────┘                  │
      ╚══════════════════════════════════╪══╝                            │
                                         │                                ▼
                                ┌────────▼─────────┐         ┌────────────────────────┐
                                │  ANTHROPIC API   │         │  eyewitness.butterbase  │
                                │  (Claude verdict)│         │  .dev  — live dashboard │
                                └──────────────────┘         └────────────────────────┘

  end-to-end:  upload ─▶ ①YOLO facts ─▶ ②physics ─▶ ③Claude ─▶ write to Butterbase
                                                          dashboard ◀─ get_runs() ◀─┘
```

| Plane | Runs on | Owns |
|-------|---------|------|
| **Compute** | Render (Docker, 2 GB) | YOLO tracking, avoidability physics, Claude call, writes to Butterbase |
| **Backend** | Butterbase | Postgres (6 append-only tables), `get_runs` function, hosted dashboard, REST API |
| **LLM** | Anthropic | Claude `claude-sonnet-4-6` verdict (retry + fallback, never blocks) |

## What it does

Upload a dashcam clip → **① YOLO11 + ByteTrack** track vehicles (speed, heading, TTC, braking)
→ **② Field-of-Safe-Motion** physics computes whether each vehicle *could have stopped*
(the fault counterfactual) → **③ Claude** corroborates the verdict with confidence + severity
→ all evidence is appended to a **Butterbase** trail with full chain of custody → humans can
override without ever erasing the original AI verdict.

## Stack

| Layer | Tech |
|-------|------|
| Tracking | YOLO11x via ultralytics |
| CV facts | OpenCV + NumPy (two-pass, memory-safe) |
| VLM | Claude claude-sonnet-4-6 via Anthropic SDK |
| Persistence | Butterbase REST (append-only, 6 tables) |
| UI | Gradio Blocks (file upload + paste-a-URL via yt-dlp) |
| Deploy | Docker → Render (Standard, 2 GB) |

## Setup

```bash
pip install -r requirements.txt
```

### Required env vars

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BUTTERBASE_API_KEY=<your Butterbase service key>   # get from butterbase.ai dashboard
```

### Optional env vars

| Var | Default | Purpose |
|-----|---------|---------|
| `YOLO_MODEL` | `yolo11n.pt` | CPU-friendly default; switch to `yolo11x.pt` on GPU for accuracy |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Override VLM model |
| `GRADIO_PORT` | `7860` | UI port |
| `GRADIO_SHARE` | `false` | Set `true` for a public Gradio link |
| `USE_BUTTERBASE_GATEWAY` | `false` | **Deferred post-demo.** Route VLM calls through Butterbase AI gateway instead of direct Anthropic API |

## Butterbase backend

App: `eyewitness` (`app_46yxrt8czo59`)  
API: `https://api.butterbase.ai/v1/app_46yxrt8czo59`

Tables (all append-only, every row carries `run_id` + `model_version`):
- `claims` — one row per analysis run
- `facts` — one row per tracked vehicle (CV kinematics)
- `avoidability` — one row per vehicle (Field-of-Safe-Motion physics)
- `frames` — 4 keyframe rows per run
- `fault_analyses` — VLM verdict + human overrides as separate rows
- `monitoring_events` — per-stage latency / cost / tokens / fallback metrics

## Demo clips & test footage

The clips below were used to develop and demo Eyewitness. Each was trimmed to the
incident window with `yt-dlp --download-sections` so the impact detector locks onto
the right vehicles. Third-party dashcam footage — used here for research/demo only.

| Local file | Source video | Window | Notes |
|------------|--------------|--------|-------|
| `clip_uk2.mp4` | [Idiot UK Drivers Exposed #5](https://www.youtube.com/watch?v=SyESL5NNgAg) | `6:04–6:12` | ⭐ **hero clip** — verdict: Vehicle #1, 72%, SEVERE |
| `clip_rei.mp4` | [Child's Near-Miss at Intersection](https://www.youtube.com/watch?v=REqJBtkEWkA) | `8:34–8:47` | busy/slow scene — weak verdict |
| `clip_uk.mp4` | [Idiot UK Drivers Exposed #5](https://www.youtube.com/watch?v=SyESL5NNgAg) | `8:34–8:47` | alternate segment |
| _(n/a)_ | [JRS Cars — Close Calls Compilation](https://www.youtube.com/watch?v=PvEpACLg-Zk) | — | video is only 4:25 — chosen window out of range |
| _(browse)_ | [Ultimate Near Miss Playlist](https://www.youtube.com/playlist?list=PLtIavcna3Ct3GYBEVN7NyDYktBSVg8Of7) | — | source playlist for more clips |

Fetch a clip yourself (trim to the incident — pick a window with two clearly
separating vehicles):

```bash
# example: the hero clip
yt-dlp -f "mp4[height<=720]/mp4/best" \
  --download-sections "*6:04-6:12" --force-keyframes-at-cuts \
  -o clip_uk2.mp4 "https://www.youtube.com/watch?v=SyESL5NNgAg"
```

## Run

```bash
python app.py
# open http://localhost:7860
# upload clip.mp4 → Analyze → review report → submit override
```

## Architecture notes

- **Two-pass CV**: pass-1 tracks without storing frames (memory-safe for long clips);
  pass-2 seeks to the 4 keyframe positions.
- **VLM fallback**: if Claude output cannot be parsed or is missing required fields,
  `FALLBACK_HYPOTHESIS` is returned and `fallback_used=True` is recorded.
- **Append-only evidence trail**: human overrides write a new `fault_analyses` row
  with `override_reason` set; the original VLM row is never modified.
- **Butterbase writes are async**: a daemon thread handles persistence so the UI
  returns immediately.
- **PlanGEN untouched**: this package lives entirely under `eyewitness/` with no
  shared imports or side-effects on other projects in this workspace.
