#!/usr/bin/env python3
"""
Eyewitness — LIVE DEMO  (your own video → Butterbase → live dashboard)

Applies the Senior ML Engineer workflow end to end:
  1. Deployment validation  — preflight health gate before running
  2. LLM Integration         — run pipeline, track tokens + cost + latency
  3. Model Monitoring        — print per-stage metrics, flag drift/fallback
  4. Persistence verify      — confirm the run landed in Butterbase
  5. Hand-off                — open the live dashboard so you SEE it appear

Usage:
    python scripts/live_demo.py --clip clip.mp4
    python scripts/live_demo.py --clip clip.mp4 --no-open      # don't auto-open browser
    python scripts/live_demo.py --clip clip.mp4 --model yolo11n.pt
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import config

DASHBOARD = "https://eyewitness.butterbase.dev"
RUNS_FN   = f"{config.BB_API_URL}/fn/get_runs"

SEP  = "─" * 62
SEP2 = "═" * 62
P = "✅"; F = "❌"; W = "⚠️ "


def bar(frac: float, width: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


# ── 1 · deployment validation gate ──────────────────────────────────────────
def preflight() -> bool:
    print(f"\n{SEP2}\n  STEP 1 · PREFLIGHT  (deployment validation gate)\n{SEP2}")
    ok = True

    keys = {"ANTHROPIC_API_KEY": config.ANTHROPIC_API_KEY,
            "BUTTERBASE_API_KEY": config.BB_API_KEY}
    for name, val in keys.items():
        good = bool(val) and not val.startswith("paste_")
        print(f"  {P if good else F}  {name}")
        ok = ok and good

    try:
        req = urllib.request.Request(
            f"{config.BB_API_URL}/claims?limit=1",
            headers={"Authorization": f"Bearer {config.BB_API_KEY}"},
        )
        urllib.request.urlopen(req, timeout=8)
        print(f"  {P}  Butterbase reachable")
    except Exception as exc:
        print(f"  {F}  Butterbase: {exc}")
        ok = False

    if not ok:
        print(f"\n  {F}  Preflight failed — fix the above, then re-run.\n")
    return ok


# ── 4 · verify persistence ───────────────────────────────────────────────────
def verify_persisted(run_id: str, retries: int = 6) -> bool:
    for i in range(retries):
        try:
            with urllib.request.urlopen(RUNS_FN, timeout=8) as r:
                data = json.loads(r.read())
            if any(run.get("run_id") == run_id for run in data.get("runs", [])):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Eyewitness live demo on your own video")
    ap.add_argument("--clip", required=True, help="Path to a dashcam / collision video")
    ap.add_argument("--model", default=None, help="Override YOLO model (e.g. yolo11n.pt)")
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open the dashboard")
    args = ap.parse_args()

    if args.model:
        config.YOLO_MODEL = args.model

    clip = Path(args.clip)
    if not clip.exists():
        print(f"\n{F}  Clip not found: {clip}\n")
        sys.exit(1)

    print(f"\n{SEP2}")
    print("  EYEWITNESS · LIVE DEMO")
    print(f"  clip:  {clip.name}  ({clip.stat().st_size / 1e6:.1f} MB)")
    print(f"  model: {config.YOLO_MODEL}   dashboard: {DASHBOARD}")
    print(SEP2)

    if not preflight():
        sys.exit(1)

    # ── 2 · run the pipeline (LLM integration) ───────────────────────────────
    print(f"\n{SEP2}\n  STEP 2 · ANALYZE  (CV + avoidability + VLM)\n{SEP2}")
    print("  Running… first run downloads YOLO weights, please wait.\n")
    from video_pipeline import analyze_clip

    t0 = time.perf_counter()
    result = analyze_clip(str(clip), persist_sync=True)   # block until persisted
    wall = (time.perf_counter() - t0) * 1000

    if result.error:
        print(f"  {F}  Pipeline error: {result.error}\n")
        sys.exit(1)

    # ── facts ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  📐  FACTS — {len(result.vehicle_facts)} vehicle(s) tracked\n{SEP}")
    for f in result.vehicle_facts:
        ttc = f"{f.ttc_ms:.0f}ms" if f.ttc_ms >= 0 else "—"
        print(f"   #{f.vehicle_id}  {f.speed_kph_est:6.1f} km/h  "
              f"hdg {f.heading_deg:3.0f}°  ttc {ttc:>7}  "
              f"braked {'yes' if f.had_safe_stop else 'no'}")

    # ── avoidability (signature) ──────────────────────────────────────────────
    print(f"\n{SEP}\n  ⚖️   AVOIDABILITY — Field of Safe Motion\n{SEP}")
    for a in result.avoidability:
        verdict = "✅ AVOIDABLE  (had room — at fault)" if a.avoidable \
                  else "⛔ UNAVOIDABLE (no safe stop — cleared)"
        print(f"   #{a.vehicle_id}  need {a.total_needed_m:6.1f}m   "
              f"gap {a.available_gap_m:6.1f}m   {verdict}")

    # ── verdict (VLM) ─────────────────────────────────────────────────────────
    h = result.hypothesis
    print(f"\n{SEP}\n  🤖  VERDICT — vision-model corroboration\n{SEP}")
    if h:
        fault = f"VEHICLE #{h.fault_vehicle_id}" if h.fault_vehicle_id is not None else "SHARED"
        print(f"   Fault:      {fault}")
        print(f"   Reason:     {h.fault_reason}")
        print(f"   Confidence: [{bar(h.confidence)}] {h.confidence*100:.0f}%")
        print(f"   Severity:   {h.severity.upper()}")
        for factor in h.contributing_factors:
            print(f"               • {factor}")
        if h.fallback_used:
            print(f"   {W} VLM fallback used — verdict from physics only")

    # ── 3 · model monitoring ──────────────────────────────────────────────────
    print(f"\n{SEP}\n  📊  MONITORING — ML Engineer metrics\n{SEP}")
    print(f"   Wall-clock latency:  {wall:,.0f} ms")
    print(f"   Video FPS:           {result.fps:.0f}")
    print(f"   Vehicles / avoidable:{len(result.vehicle_facts)} / "
          f"{sum(1 for a in result.avoidability if a.avoidable)}")
    if h:
        print(f"   Tokens (in/out):     {h.input_tokens:,} / {h.output_tokens:,}")
        print(f"   LLM cost:            ${h.cost_usd:.5f}")
        print(f"   Fallback rate:       {'1/1 ⚠️' if h.fallback_used else '0/1 ✅'}")

    # ── 4 · verify it reached Butterbase ──────────────────────────────────────
    print(f"\n{SEP2}\n  STEP 3 · VERIFY PERSISTENCE\n{SEP2}")
    if verify_persisted(result.run_id):
        print(f"  {P}  Evidence trail confirmed in Butterbase")
        print(f"      run_id: {result.run_id}")
    else:
        print(f"  {W}  Could not confirm the run via get_runs (it may still be writing).")

    # ── 5 · hand-off to live dashboard ────────────────────────────────────────
    print(f"\n{SEP2}\n  STEP 4 · SEE IT LIVE\n{SEP2}")
    print(f"  Open the dashboard — your new case appears at the top:")
    print(f"      {DASHBOARD}")
    if not args.no_open:
        try:
            subprocess.Popen(["open", DASHBOARD])
            print(f"  {P}  Opening in your browser…")
        except Exception:
            pass
    print(f"\n{SEP2}\n  ✅  DEMO COMPLETE  ·  run {result.run_id[:8]}\n{SEP2}\n")


if __name__ == "__main__":
    main()
