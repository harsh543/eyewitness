#!/usr/bin/env python3
"""
Eyewitness — Demo Runner  (no Gradio, terminal output)
ML Engineer LLM Integration Workflow · end-to-end validation

Runs the full pipeline on a clip and prints a structured report:
  CV facts  →  avoidability physics  →  VLM hypothesis  →  MLOps metrics

Usage:
    python scripts/run_demo.py --clip clip.mp4
    python scripts/run_demo.py --clip clip.mp4 --save-frames
    python scripts/run_demo.py --clip clip.mp4 --save-frames --open
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

SEP  = "─" * 60
SEP2 = "═" * 60


def _bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Eyewitness pipeline on a clip")
    ap.add_argument("--clip",        required=True, help="Path to dashcam video")
    ap.add_argument("--save-frames", action="store_true", help="Save 4 keyframes as JPEGs")
    ap.add_argument("--open",        action="store_true", help="Open saved frames after run")
    args = ap.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        print(f"❌  Clip not found: {clip}")
        sys.exit(1)

    print(f"\n{SEP2}")
    print("  EYEWITNESS  ·  Demo Runner")
    print(f"  Clip:  {clip.name}  ({clip.stat().st_size / 1e6:.1f} MB)")
    print(SEP2)

    # ── run pipeline ──────────────────────────────────────────────────────────
    print("\n⚡  Running pipeline...\n")
    from video_pipeline import analyze_clip

    t0     = time.perf_counter()
    result = analyze_clip(str(clip))
    total  = (time.perf_counter() - t0) * 1000

    if result.error:
        print(f"❌  Pipeline error: {result.error}")
        sys.exit(1)

    # ── panel 1: CV facts ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  📐  VEHICLE FACTS  —  {len(result.vehicle_facts)} tracked  (deterministic CV)")
    print(SEP)
    print(f"  {'ID':>4}  {'Speed':>10}  {'Heading':>8}  {'TTC':>10}  {'Safe Stop':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}")
    for f in result.vehicle_facts:
        ttc  = f"{f.ttc_ms:.0f} ms" if f.ttc_ms >= 0 else "      —"
        stop = "✅ yes" if f.had_safe_stop else "❌ no "
        print(f"  #{f.vehicle_id:>3}  {f.speed_kph_est:>7.1f} kph  "
              f"{f.heading_deg:>6.0f}°  {ttc:>10}  {stop}")

    # ── panel 2: avoidability ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  ⚖️   AVOIDABILITY  —  Field of Safe Motion")
    print(SEP)
    print(f"  {'ID':>4}  {'Speed':>10}  {'Need':>10}  {'Gap':>10}  Verdict")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*15}")
    for a in result.avoidability:
        print(f"  #{a.vehicle_id:>3}  {a.speed_kph:>7.1f} kph  "
              f"{a.total_needed_m:>7.2f} m  {a.available_gap_m:>7.2f} m  {a.verdict}")

    unavoidable = [a for a in result.avoidability if not a.avoidable]
    avoidable   = [a for a in result.avoidability if a.avoidable]
    if unavoidable:
        ids = ", ".join(f"#{a.vehicle_id}" for a in unavoidable)
        print(f"\n  ⛔  Physics EXCULPATES {ids}  — no safe stop existed")
    if avoidable:
        ids = ", ".join(f"#{a.vehicle_id}" for a in avoidable)
        print(f"  ✅  Physics IMPLICATES {ids}  — avoidance was possible")

    # ── panel 3: VLM hypothesis ───────────────────────────────────────────────
    h = result.hypothesis
    print(f"\n{SEP}")
    print("  🤖  VLM FAULT HYPOTHESIS  —  Claude corroboration")
    print(SEP)
    if h:
        fault = f"Vehicle #{h.fault_vehicle_id}" if h.fault_vehicle_id is not None else "Shared / indeterminate"
        conf_bar = _bar(h.confidence)
        sev_map  = {"minor": "🟡", "moderate": "🟠", "severe": "🔴", "critical": "⛔", "unknown": "⚪"}
        print(f"  Fault:      {fault}")
        print(f"  Reason:     {h.fault_reason}")
        print(f"  Confidence: [{conf_bar}] {h.confidence * 100:.0f}%")
        print(f"  Severity:   {sev_map.get(h.severity, '⚪')} {h.severity.upper()}")
        print(f"  Factors:")
        for factor in h.contributing_factors:
            print(f"              • {factor}")
        if h.fallback_used:
            print(f"  ⚠️   Fallback used — VLM output could not be parsed")
    else:
        print("  VLM hypothesis not available")

    # ── panel 4: MLOps metrics ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  📊  MLOPS METRICS  (ML Engineer Monitoring)")
    print(SEP)
    print(f"  Total latency:   {total:,.0f} ms")
    print(f"  Video FPS:       {result.fps:.0f}")
    print(f"  Vehicles:        {len(result.vehicle_facts)}")
    print(f"  Avoidable:       {sum(1 for a in result.avoidability if a.avoidable)}")
    if h:
        print(f"  Input tokens:    {h.input_tokens:,}")
        print(f"  Output tokens:   {h.output_tokens:,}")
        cost_str = f"${h.cost_usd:.5f}"
        print(f"  LLM cost:        {cost_str}")
        print(f"  Fallback:        {'Yes ⚠️' if h.fallback_used else 'No  ✅'}")
    print(f"  Run ID:          {result.run_id[:8]}…")
    print(f"  Butterbase:      evidence trail → async write")

    # ── save frames ───────────────────────────────────────────────────────────
    if args.save_frames and result.keyframes:
        out = ROOT / "demo_frames"
        out.mkdir(exist_ok=True)
        paths = []
        for kf in result.keyframes:
            p = out / f"{kf.keyframe_type}.jpg"
            p.write_bytes(kf.image_bytes)
            paths.append(p)
        print(f"\n  📸  Keyframes saved → demo_frames/")
        for p in paths:
            print(f"      {p.name}")
        if args.open:
            for p in paths:
                subprocess.Popen(["open", str(p)])

    print(f"\n{SEP2}")
    print(f"  ✅  PIPELINE COMPLETE  —  run_id: {result.run_id[:8]}")
    print(f"  Evidence persisted to Butterbase (async background thread)")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()
