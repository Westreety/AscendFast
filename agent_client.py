"""Thin bridge: Python → claude CLI headless agent calls."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from verify import record_agent_call

# Set ASCENDFAST_USE_LLM_AGENT=0 to force rule-based fallback (offline / tests).
AGENT_ENABLED: bool = os.environ.get("ASCENDFAST_USE_LLM_AGENT", "1") != "0"

_PROJECT_ROOT = Path(__file__).parent


def call_agent(agent_name: str, prompt: str, *, timeout: int = 120) -> str | None:
    """Run `claude -p <prompt> --agent <agent_name>` headless; return text or None.

    Before every None return, log an agent_call StageOutcome distinguishing the
    failure kind (disabled/timeout/subprocess_error/agent_error) so "为什么没效果"
    stops being a black box. The None contract itself is unchanged; callers don't move.
    """
    if not AGENT_ENABLED:
        record_agent_call(agent_name, "disabled")
        return None
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--agent", agent_name, "--output-format", "json",
             "--permission-mode", "acceptEdits"],
            capture_output=True, text=True, timeout=timeout, cwd=_PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        record_agent_call(agent_name, "timeout", f"exceeded {timeout}s")
        return None
    except Exception as exc:  # noqa: BLE001
        record_agent_call(agent_name, "subprocess_error", f"{type(exc).__name__}: {exc}")
        return None
    #如果 result.stdout 不是合法 JSON，return None
    try:
        outer = json.loads(result.stdout)
    except json.JSONDecodeError: 
        record_agent_call(agent_name, "agent_error", "stdout was not valid CLI JSON")
        return None
    
    if outer.get("is_error") or outer.get("subtype") != "success":
        record_agent_call(agent_name, "agent_error", str(outer.get("subtype") or "is_error"))
        return None
    record_agent_call(agent_name, "ok")
    # {
    # "subtype": "success",
    # "is_error": false,
    # "result": "agent 返回内容"
    # }
    return outer.get("result")


def call_agent_json(agent_name: str, prompt: str, *, timeout: int = 120) -> dict | list | None:
    """Like call_agent but parse .result as JSON; return None on any failure."""
    prompt_with_constraint = (
        prompt + "\n\nIMPORTANT: Reply with ONLY valid JSON, no markdown fences."
    )
    raw = call_agent(agent_name, prompt_with_constraint, timeout=timeout)
    if raw is None:
        return None
    # Strip optional ```json … ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    # Find first { or [
    m = re.search(r"[\[{]", cleaned)
    if m:
        cleaned = cleaned[m.start():]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        record_agent_call(agent_name, "bad_json", "result was not parseable JSON")
        return None
