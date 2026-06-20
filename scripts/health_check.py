#!/usr/bin/env python3
"""
Eyewitness — System Health Check
ML Engineer Deployment Workflow · Validation step

Checks: env keys · Butterbase (all 5 tables) · Anthropic (latency) · YOLO weights
Prints pass/fail for each; exits 1 on any hard failure.

Usage:
    python scripts/health_check.py
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import config

P = "✅"; F = "❌"; W = "⚠️ "

SEP = "─" * 54


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check_env() -> bool:
    section("1 · Environment")
    keys = {"ANTHROPIC_API_KEY": config.ANTHROPIC_API_KEY,
            "BUTTERBASE_API_KEY": config.BB_API_KEY}
    ok = True
    for name, val in keys.items():
        if val and not val.startswith("paste_"):
            print(f"  {P}  {name}  ({val[:12]}…)")
        else:
            print(f"  {F}  {name}  NOT SET")
            ok = False
    return ok


def check_butterbase() -> bool:
    section("2 · Butterbase  (append-only evidence trail)")
    tables = ["claims", "facts", "frames", "fault_analyses", "monitoring_events"]
    ok = True
    for table in tables:
        try:
            url = f"{config.BB_API_URL}/{table}?limit=1"
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {config.BB_API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                rows = json.loads(r.read())
            count = len(rows) if isinstance(rows, list) else "?"
            print(f"  {P}  {table:<22} ({count} rows)")
        except Exception as exc:
            print(f"  {F}  {table:<22} {exc}")
            ok = False
    return ok


def check_anthropic() -> bool:
    section("3 · Anthropic  (LLM Integration · latency SLA)")
    import anthropic
    SLA_MS = 5000
    try:
        t0     = time.perf_counter()
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp   = client.messages.create(
            model=config.CLAUDE_MODEL, max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        latency = (time.perf_counter() - t0) * 1000
        tokens  = resp.usage.input_tokens + resp.usage.output_tokens
        sla     = P if latency < SLA_MS else W
        print(f"  {P}  model      {config.CLAUDE_MODEL}")
        print(f"  {sla}  latency    {latency:.0f} ms  (SLA < {SLA_MS} ms)")
        print(f"  {P}  tokens     {tokens}")
        cost = (resp.usage.input_tokens  * config.COST_INPUT_PER_1K_USD / 1000 +
                resp.usage.output_tokens * config.COST_OUTPUT_PER_1K_USD / 1000)
        print(f"  {P}  cost       ${cost:.6f}")
        return True
    except Exception as exc:
        print(f"  {F}  {exc}")
        return False


def check_yolo() -> bool:
    section("4 · YOLO Weights")
    model_file = Path(config.YOLO_MODEL)
    if model_file.exists():
        mb = model_file.stat().st_size / 1e6
        print(f"  {P}  {config.YOLO_MODEL}  ({mb:.0f} MB, ready)")
        return True
    else:
        print(f"  {W}  {config.YOLO_MODEL} not found — auto-downloads on first run")
        return True   # not a hard failure


def check_tenacity() -> bool:
    section("5 · MLOps Stack  (retry · validation · monitoring)")
    libs = {
        "tenacity":     ("tenacity",   "retry with exponential backoff"),
        "pydantic":     ("pydantic",   "structured output validation"),
        "python-dotenv":("dotenv",     "secrets management"),
    }
    ok = True
    for lib, (import_name, desc) in libs.items():
        try:
            __import__(import_name)
            print(f"  {P}  {lib:<16} {desc}")
        except ImportError:
            print(f"  {F}  {lib:<16} NOT INSTALLED  →  pip install {lib}")
            ok = False
    return ok


def main() -> None:
    print(f"\n{'=' * 54}")
    print("  EYEWITNESS  ·  Health Check")
    print(f"  {config.BB_API_URL}")
    print(f"{'=' * 54}")

    results = [
        check_env(),
        check_butterbase(),
        check_anthropic(),
        check_yolo(),
        check_tenacity(),
    ]

    print(f"\n{'=' * 54}")
    if all(results):
        print(f"  {P}  ALL SYSTEMS GO  —  ready to demo")
    else:
        failed = sum(1 for r in results if not r)
        print(f"  {F}  {failed} check(s) failed  —  fix before demo")
    print(f"{'=' * 54}\n")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
