#!/usr/bin/env python3
"""Bridge one DeepSeek Harness pre-execute event into DirectorHarness."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from director_harness.events import DirectorEventLog
from director_harness.harness import DirectorHarness


def main() -> int:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Director bridge input must be a JSON object")
    raw_input = payload.get("tool_input")
    tool_input = raw_input if isinstance(raw_input, dict) else {}
    director = DirectorHarness(
        event_log=DirectorEventLog(os.environ.get("DIRECTOR_LOG_PATH"))
    )
    decision = director.preflight_registered_tool(
        tool_name=str(payload.get("tool_name") or "unknown"),
        tool_input=tool_input,
        session_id=str(payload.get("session_id") or ""),
        tool_use_id=str(payload.get("tool_use_id") or ""),
        backend="deepseek-harness",
    )
    print(
        json.dumps(
            {
                "action": decision.action,
                "reason": decision.reason,
                "mcp_repair_supported": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
