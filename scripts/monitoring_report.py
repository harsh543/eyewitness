#!/usr/bin/env python3
"""
Eyewitness — Monitoring Report
ML Engineer Model Monitoring · drift detection · cost tracking

Pulls monitoring_events and fault_analyses from Butterbase,
computes aggregate metrics, and flags drift / anomalies.

Usage:
    python scripts/monitoring_report.py           # last 50 runs
    python scripts/monitoring_report.py --limit 100
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import config

SEP  = "─" * 54
SEP2 = "═" * 54

# Thresholds from ML Engineer skill
THRESHOLDS = {
    "p95_latency_ms":   (5_000,  10_000),   # warning, critical
    "error_rate_pct":   (5.0,    15.0),
    "fallback_rate_pct":(10.0,   25.0),
    "avg_cost_usd":     (0.01,   0.05),
    "avg_confidence":   (0.5,    0.3),       # BELOW these = warning/critical
}


def _get(table: str, limit: int) -> list[dict]:
    url = f"{config.BB_API_URL}/{table}?limit={limit}&order=created_at.desc"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {config.BB_API_KEY}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())
    except Exception as exc:
        print(f"  ⚠️   Could not fetch {table}: {exc}")
        return []


def _flag(val: float, warn: float, crit: float, low: bool = False) -> str:
    if low:
        if val < crit: return "⛔ CRITICAL"
        if val < warn: return "🟡 WARNING"
        return "✅"
    else:
        if val > crit: return "⛔ CRITICAL"
        if val > warn: return "🟡 WARNING"
        return "✅"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    print(f"\n{SEP2}")
    print("  EYEWITNESS  ·  Monitoring Report")
    print(f"  Last {args.limit} events  ·  {config.BB_API_URL}")
    print(SEP2)

    events   = _get("monitoring_events",  args.limit)
    analyses = _get("fault_analyses",     args.limit)
    claims   = _get("claims",             args.limit)

    if not events and not analyses:
        print("\n  No data yet — run the pipeline on a clip first.\n")
        sys.exit(0)

    # ── latency ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  ⏱   Latency\n{SEP}")
    vlm_events = [e for e in events if e.get("stage") == "vlm" and e.get("latency_ms")]
    cv_events  = [e for e in events if e.get("stage") == "cv"  and e.get("latency_ms")]

    for label, evts in [("CV stage", cv_events), ("VLM stage", vlm_events)]:
        if not evts:
            print(f"  {label:<14} no data")
            continue
        lats  = [e["latency_ms"] for e in evts]
        p50   = sorted(lats)[len(lats) // 2]
        p95   = sorted(lats)[int(len(lats) * 0.95)]
        flag  = _flag(p95, *THRESHOLDS["p95_latency_ms"])
        print(f"  {label:<14}  p50 {p50:>7,.0f} ms   p95 {p95:>7,.0f} ms  {flag}")

    # ── cost tracking ─────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  💰  LLM Cost Tracking\n{SEP}")
    cost_events = [e for e in events if e.get("cost_usd") and e["cost_usd"] > 0]
    if cost_events:
        costs      = [e["cost_usd"] for e in cost_events]
        avg_cost   = mean(costs)
        total_cost = sum(costs)
        in_tok     = sum(e.get("input_tokens",  0) for e in cost_events)
        out_tok    = sum(e.get("output_tokens", 0) for e in cost_events)
        flag       = _flag(avg_cost, *THRESHOLDS["avg_cost_usd"])
        print(f"  Runs tracked:    {len(cost_events)}")
        print(f"  Avg cost/run:    ${avg_cost:.5f}  {flag}")
        print(f"  Total cost:      ${total_cost:.4f}")
        print(f"  Total tokens:    {in_tok + out_tok:,}  (in {in_tok:,} / out {out_tok:,})")
    else:
        print("  No cost data yet")

    # ── fallback rate (drift proxy) ───────────────────────────────────────────
    print(f"\n{SEP}\n  📉  Fallback Rate  (drift proxy)\n{SEP}")
    vlm_all      = [e for e in events if e.get("stage") == "vlm"]
    fallback_all = [e for e in vlm_all if e.get("fallback_used")]
    if vlm_all:
        rate = len(fallback_all) / len(vlm_all) * 100
        flag = _flag(rate, *THRESHOLDS["fallback_rate_pct"])
        print(f"  Fallback rate:   {rate:.1f}%  ({len(fallback_all)}/{len(vlm_all)})  {flag}")
        if rate > THRESHOLDS["fallback_rate_pct"][0]:
            print("  → Inspect prompt engineering or model version")
    else:
        print("  No VLM events yet")

    # ── confidence distribution ───────────────────────────────────────────────
    print(f"\n{SEP}\n  🎯  VLM Confidence Distribution\n{SEP}")
    conf_data = [e.get("vlm_confidence", 0) for e in vlm_all if e.get("vlm_confidence")]
    if conf_data:
        avg_conf = mean(conf_data)
        flag     = _flag(avg_conf, *THRESHOLDS["avg_confidence"], low=True)
        buckets  = {"high (>0.7)": 0, "mid (0.4–0.7)": 0, "low (<0.4)": 0}
        for c in conf_data:
            if c > 0.7:   buckets["high (>0.7)"] += 1
            elif c > 0.4: buckets["mid (0.4–0.7)"] += 1
            else:         buckets["low (<0.4)"] += 1
        print(f"  Avg confidence:  {avg_conf:.2f}  {flag}")
        for bucket, count in buckets.items():
            bar = "█" * count + "░" * (len(conf_data) - count)
            print(f"  {bucket:<18}  {bar[:20]}  {count}")
    else:
        print("  No confidence data yet")

    # ── vehicle + avoidability stats ──────────────────────────────────────────
    print(f"\n{SEP}\n  🚗  Scene Statistics\n{SEP}")
    cv_with_data = [e for e in cv_events if e.get("vehicle_count")]
    if cv_with_data:
        avg_veh   = mean(e["vehicle_count"]    for e in cv_with_data)
        avg_avoid = mean(e.get("avoidable_count", 0) for e in cv_with_data)
        print(f"  Avg vehicles/clip:  {avg_veh:.1f}")
        print(f"  Avg avoidable/clip: {avg_avoid:.1f}")

    # ── evidence trail health ─────────────────────────────────────────────────
    print(f"\n{SEP}\n  🗄️   Evidence Trail Health\n{SEP}")
    print(f"  Claims:           {len(claims)}")
    print(f"  Fault analyses:   {len(analyses)}")
    overrides = [a for a in analyses if a.get("override_reason")]
    print(f"  Human overrides:  {len(overrides)}")

    print(f"\n{SEP2}")
    print("  Report complete")
    print(f"  Dashboard: https://butterbase.ai/app/{config.BB_APP_ID}")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()
