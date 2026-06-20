# Eyewitness — ML Engineering Handoff

A deep-dive for the next engineer. Honest about what's solid, what's a heuristic,
and what will bite you. Structured around the four production-ML concerns:
**pipeline → LLM integration → deployment → monitoring.**

> TL;DR — Eyewitness reconstructs vehicle-collision fault from dashcam video by
> **separating deterministic facts from generative judgment**. A CV layer measures
> kinematics, a physics layer computes whether each vehicle *could have stopped*
> (the fault counterfactual), and only then does a vision LLM corroborate and
> narrate. Everything is logged to an append-only evidence trail on Butterbase
> and rendered on a live forensic dashboard.

---

## 0. System map

```
video ─▶ cv_pipeline ─▶ avoidability ─▶ anthropic_client ─▶ butterbase_client
          (YOLO11)       (physics)         (Claude VLM)        (append-only DB)
            │                │                  │                    │
            └────────────── video_pipeline.analyze_clip orchestrates ┘
                                   │                         │
                              monitor.py                 Butterbase
                          (per-stage metrics)         get_runs() function
                                                             │
                                                   web/index.html dashboard
                                                   (eyewitness.butterbase.dev)
```

| File | Responsibility |
|------|----------------|
| `config.py` | All constants + env binding (single source of truth) |
| `schemas.py` | Frozen dataclasses: `VehicleFact`, `AvoidabilityResult`, `KeyFrame`, `FaultHypothesis`, `AnalysisResult` |
| `cv_pipeline.py` | YOLO11+ByteTrack tracking, kinematics, keyframe extraction (two-pass) |
| `avoidability.py` | Field-of-Safe-Motion stopping-distance counterfactual |
| `prompts.py` | Hardened VLM system/user prompts, fallback constant, JSON encoder |
| `anthropic_client.py` | LLM call: retry, Pydantic validation, cost tracking, fallback |
| `butterbase_client.py` | Append-only REST adapter (claims/facts/avoidability/frames/fault/monitoring) |
| `monitor.py` | `RunMonitor` — per-stage latency/cost/token/fallback metrics |
| `video_pipeline.py` | Orchestration; `analyze_clip(persist_sync=…)` |
| `app.py` | Gradio UI (file upload + URL fetch via yt-dlp) |
| `web/index.html` | Static forensic dashboard (reads `get_runs`) |
| `scripts/` | `health_check`, `live_demo`, `run_demo`, `monitoring_report` |
| `Dockerfile`, `render.yaml` | Containerized deploy to Render |

**Butterbase app:** `app_46yxrt8czo59` · API `https://api.butterbase.ai/v1/app_46yxrt8czo59`
**Live surfaces:** dashboard `eyewitness.butterbase.dev` · analyzer `eyewitness-pc9a.onrender.com`

---

## 1. Pipeline — Layer 1: deterministic CV facts (`cv_pipeline.py`)

**Model:** Ultralytics YOLO11 (`yolo11n.pt` for CPU/live, `yolo11x.pt` for accuracy/GPU),
tracker = ByteTrack (`bytetrack.yaml`), classes `[1,2,3,5,7]` (bicycle, car, motorcycle, bus, truck).

**Two-pass design (deliberate):**
- **Pass 1** runs `model.track(persist=True)` over every `SAMPLE_N`-th frame, accumulating
  per-track centroid history (`deque(maxlen=TRAIL_LEN)`) and per-frame pairwise min-distance.
  Frames are **not** retained → memory-safe on long clips.
- **Impact frame** = `argmin` of pairwise centroid distance across the clip (cheap proxy
  for "moment of closest approach / collision").
- **Pass 2** re-opens the video and `seek`s to 4 keyframe positions
  (scene_overview, pre_impact, impact, post_impact) and JPEG-encodes only those.

**Per-vehicle facts derived near the impact frame:**
- `speed_px_per_frame` = ‖mean Δcentroid‖ over a window around impact
- `speed_kph_est` = px/frame → m/s → km/h via `_speed_kph` (see calibration caveat §6)
- `heading_deg` = `atan2(-vy, vx)` (0°=right, 90°=up)
- `ttc_ms` = pairwise closest-approach time (constant-velocity model), `-1` if none within horizon
- `had_safe_stop` = ≥30% speed drop from peak→tail before impact (a "did they brake?" signal)

**Key params** (`config.py`): `INPUT_W=1280`, `SAMPLE_N=2`, `TRAIL_LEN=20`,
`MIN_TRACK_FRAMES=5`, `HORIZON_SEC=2.0`, `COLL_THRESH_PX=120`.

---

## 2. Pipeline — Layer 2: avoidability physics (`avoidability.py`)

The credibility core. For each vehicle at impact:

```
v          = speed_kph / 3.6                      # m/s
react_dist = v × REACTION_S                       # REACTION_S = 1.5 s (NHTSA)
stop_dist  = v² / (2 × A_MAX)                      # A_MAX = 7.0 m/s² (hard braking, dry)
needed     = react_dist + stop_dist
gap        = nearest-vehicle pixel distance × (SCENE_W_M / frame_width_px)
avoidable  = gap > needed
```

- `avoidable = True`  → vehicle **had room to stop and didn't** → fault signal.
- `avoidable = False` → **no safe stop existed** → physics *exculpates* it.

This counterfactual is the fault determination; the LLM corroborates it rather than
inventing it. Constants live at the top of `avoidability.py` — tune `REACTION_S`/`A_MAX`
for wet/night conditions if you add weather detection.

---

## 3. LLM Integration (`anthropic_client.py`, `prompts.py`)

Built to the skill's LLM-Integration checklist:

| Concern | Implementation |
|---|---|
| Provider abstraction | Single `_call_api(content)` — swap providers in one place |
| Retry / backoff | `@retry(stop_after_attempt(3), wait_exponential(1,10))` on `APIError`/`APIConnectionError` |
| Structured output | Pydantic `_FaultOutput` (confidence ∈ [0,1], severity `Literal`, non-empty factors) |
| Fence stripping | `_strip_fences` removes ```` ```json ```` wrappers before parse |
| Cost tracking | `_compute_cost(usage)` × rates in `config` (`COST_INPUT/OUTPUT_PER_1K_USD`) |
| Fallback | Any parse/validation/exception → `FALLBACK_HYPOTHESIS`, `fallback_used=True`, cost 0 |
| Grounding | System prompt tells Claude to treat physics `avoidable` as ground truth and flag disagreement |

**Model:** `claude-sonnet-4-6`. Multimodal: 4 keyframes (base64 JPEG) + structured facts/avoidability JSON.
**Cost observed:** ~$0.023/run (~6.2k in / ~285 out tokens). **The LLM never blocks or breaks the
pipeline** — `run_hypothesis` always returns a valid `FaultHypothesis`.

⚠️ **Production bug we already hit and fixed (read this):** the CV/physics values are NumPy
scalars (`float32`, `bool_`). `json.dumps` cannot serialize those, which silently broke both
(a) the VLM prompt build → forced fallback with 0 tokens, and (b) Butterbase avoidability inserts.
Fix = a NumPy-aware `default=_json_native` (`o.item()`) at **both** serialization boundaries
(`prompts.py`, `butterbase_client.py`). **Lesson: cast to native types at every json.dumps boundary.**

---

## 4. Data layer — append-only evidence trail (`butterbase_client.py`)

Backend = Butterbase (`app_46yxrt8czo59`). Six tables, all carrying `run_id` + `model_version`:

| Table | Rows per run | Purpose |
|---|---|---|
| `claims` | 1 | run header (clip, status) |
| `facts` | N vehicles | CV kinematics |
| `avoidability` | N vehicles | physics counterfactual |
| `frames` | 4 | keyframes (base64) |
| `fault_analyses` | ≥1 | VLM verdict; **human overrides are new rows, never updates** |
| `monitoring_events` | 2 (cv, vlm) | per-stage metrics |

**Forensic integrity:** writes are append-only. `append_override()` writes a *new*
`fault_analyses` row with `override_reason` set — the original AI verdict is preserved
(chain of custody). The serverless `get_runs` TS function joins all six tables into clean
JSON for the dashboard (anonymous + CORS, runs as service role).

The adapter is **fail-soft**: no API key or a network error logs and returns `False` rather
than raising — analysis still completes locally.

---

## 5. Deployment & serving (`Dockerfile`, `render.yaml`, `app.py`)

- **Analyzer:** Gradio app, containerized, on Render **Standard (2 GB)**. Critical detail:
  `demo.launch(server_name="0.0.0.0", server_port=$PORT)` — localhost bind fails Render health checks.
- **OpenCV:** `opencv-python-headless` (no GUI deps on server); `ffmpeg` in the image for yt-dlp.
- **URL ingest:** `app.py:fetch_video()` runs yt-dlp with `--download-sections` for in-app
  trimming, **with graceful fallback** (YouTube bot-blocks datacenter IPs → tells user to upload).
- **Dashboard:** static `web/index.html`, deployed to Butterbase Pages, reads `get_runs`.
- **Deploy validation gate:** `scripts/health_check.py` checks env keys, all 6 tables,
  Anthropic latency SLA, and the MLOps stack before you run anything (the skill's preflight step).

**`persist_sync` flag** (`analyze_clip`): Gradio uses the async daemon-thread write; the CLI
demo uses `persist_sync=True` so the process blocks until Butterbase writes finish (otherwise
the daemon thread dies on exit and evidence never lands). Know which path you're on.

---

## 6. Known limitations & tech debt (be honest with stakeholders)

1. **Monocular px→m calibration is a fixed assumption** (`SCENE_W_M = 20 m`). Speeds and
   distances are **relative, not metrically accurate**. Avoidability *comparisons between
   vehicles in a scene* are meaningful; absolute meters are rough. Real fix: camera intrinsics
   + homography or a learned depth model.
2. **Impact-frame heuristic** (min pairwise distance) misfires on crowded/parking/low-speed
   scenes — observed on a slow parking clip where 10 near-stationary vehicles produced
   sub-meter gaps and noisy avoidability. Pick clips with a clear 2-vehicle interaction, or
   add an event detector (sudden deceleration / bbox-overlap spike).
3. **Constant-velocity TTC** — no acceleration modeling; fine for ~1.5 s horizons, weak for
   longer or turning trajectories.
4. **No automated tests.** Add unit tests for `avoidability.check_vehicle` and the JSON
   encoders first (highest bug-density per §3).
5. **CPU-bound** (~25–55 s/clip on Render Standard with yolo11n). `yolo11x` is far more
   accurate but needs a GPU instance for acceptable latency.
6. **Single-camera, single-incident** per clip. No multi-camera fusion, no multi-event.

---

## 7. Runbook

```bash
# env (never commit .env — it's gitignored)
ANTHROPIC_API_KEY=sk-ant-…
BUTTERBASE_API_KEY=bb_sk_…

# preflight
python scripts/health_check.py

# analyze a clip end-to-end (writes to Butterbase, opens dashboard)
./demo.sh clip.mp4
# or, explicit:
python scripts/live_demo.py --clip clip.mp4 --model yolo11n.pt --no-open

# observability — aggregate metrics + drift/cost flags across recent runs
python scripts/monitoring_report.py --limit 50
```

Python: use the env with deps installed (`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`).

---

## 8. Monitoring & next steps (the skill's Model-Monitoring pillar)

**Tracked per run** (`monitor.py` → `monitoring_events`): stage latency (p50/p95 via
`monitoring_report.py`), token counts, $ cost, VLM confidence, and **fallback rate** —
the primary **drift signal**. Rising fallback rate = the model is returning malformed/uncertain
output and needs attention (prompt or model-version regression).

**Recommended next moves for whoever picks this up:**
1. Tests for `avoidability` + JSON encoders (cheap, high value).
2. Replace the impact heuristic with a real event detector (decel spike + bbox IoU).
3. Metric calibration (homography or monocular depth) → real meters → defensible speeds.
4. GPU instance + `yolo11x` for accuracy; or batch/offline mode for throughput.
5. Confidence/fallback alerting (the skill's alert thresholds: fallback >10% warn, >25% critical).
6. Ground-truth eval set of labeled fault cases → measure verdict accuracy vs. human adjudicators.
