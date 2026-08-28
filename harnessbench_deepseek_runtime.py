"""Thin DeepSeek Harness + Writer/Director bridge for HarnessBench."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from writer_excute import (
    decide_execution,
    run_execute_stage,
    run_writer_harness,
    summarize_writer_result,
)
from writer_harness.actor_harness import DeepSeekHarnessActorExecutor
from writer_harness.models import InteractionMode

RESULT_LABEL = "DeepSeekHarness-compatible local result"
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class HarnessBenchRoundConfig:
    workspace: Path
    sandbox: Path
    prompt_file: Path
    session_id: str
    task_id: str
    project_root: Path
    actor_model: str
    writer_model: str
    api_timeout_sec: float = 300.0
    director_harness_enabled: bool = True


@dataclass(frozen=True)
class HarnessBenchRoundResult:
    status: str
    task_id: str
    session_id: str
    workspace: str
    session_restored: bool
    assistant_text: str
    error_events: tuple[str, ...]
    tool_calls: int
    tool_errors: int
    session_file: str
    trace_file: str
    actor_trace_file: str
    actor_backend: str
    actor_run_metadata: dict[str, Any]
    writer_required: bool
    writer_mandatory_passed: bool
    writer_state_unchanged: bool
    writer_event_validation: dict[str, Any]
    writer_usage: dict[str, int]
    writer_report_file: str
    director_enabled: bool
    director_event_validation: dict[str, Any] | None
    director_events: tuple[dict[str, Any], ...]
    director_log_file: str | None


class HarnessBenchTaskActor:
    """Own one live DeepSeek Harness runtime for every round of one task."""

    def __init__(self) -> None:
        self._executor: DeepSeekHarnessActorExecutor | None = None
        self._signature: tuple[str, ...] | None = None

    def execute(self, args: SimpleNamespace, prompt: str) -> dict[str, Any]:
        signature = (
            str(args.actor_model or ""),
            str(args.actor_base_url or ""),
            str(args.dsh_provider or ""),
            str(args.dsh_cwd or ""),
            str(args.dsh_runtime_cwd or ""),
            str(args.dsh_session_root or ""),
            str(args.dsh_session_id or ""),
            str(args.dsh_cordis or ""),
            str(args.dsh_runtime_bin or ""),
            str(args.dsh_director_stage or ""),
        )
        if self._signature is not None and signature != self._signature:
            raise RuntimeError("DeepSeek Harness task runtime configuration changed between rounds")
        if self._executor is None:
            self._signature = signature
            self._executor = DeepSeekHarnessActorExecutor(
                model=args.actor_model,
                base_url=args.actor_base_url,
                api_key=args.actor_api_key,
                provider=args.dsh_provider,
                cwd=args.dsh_cwd,
                runtime_cwd=args.dsh_runtime_cwd,
                session_root=args.dsh_session_root,
                session_id=args.dsh_session_id,
                session_id_per_execute=False,
                cordis=args.dsh_cordis,
                runtime_bin=args.dsh_runtime_bin,
                request_timeout_seconds=args.dsh_request_timeout,
                director_stage=args.dsh_director_stage,
                persistent_runtime=True,
            )
        result = self._executor.execute(prompt, InteractionMode.VANILLA)
        return {
            "ok": result.ok,
            "mode": result.mode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.return_code,
            "final_prompt": result.final_prompt,
            "actor_backend": result.actor_backend,
            "actor_run_metadata": result.actor_run_metadata,
            "tool_trace": result.tool_trace,
        }

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        self._signature = None
        if executor is not None:
            executor.close()


def _safe_segment(value: str, fallback: str) -> str:
    candidate = _SAFE_SEGMENT.sub("-", value).strip("-._")
    return candidate[:100] or fallback


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _workspace_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        snapshot[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _planning_state_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> bool:
    for relative in set(before) | set(after):
        if before.get(relative) == after.get(relative):
            continue
        if relative.startswith("in/") and relative not in before and relative in after:
            continue
        return False
    return True


def _replace_path(value: Any, source: Path, target: Path) -> Any:
    """Replace an isolated planning path in nested Writer output."""

    source_text = str(source.resolve())
    target_text = str(target.resolve())
    if isinstance(value, str):
        return value.replace(source_text, target_text)
    if isinstance(value, list):
        return [_replace_path(item, source, target) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_path(item, source, target) for item in value)
    if isinstance(value, dict):
        return {key: _replace_path(item, source, target) for key, item in value.items()}
    return value


def _validate(config: HarnessBenchRoundConfig) -> None:
    workspace = config.workspace.resolve()
    sandbox = config.sandbox.resolve()
    prompt_file = config.prompt_file.resolve()
    if not workspace.is_dir():
        raise ValueError(f"HarnessBench workspace is not a directory: {workspace}")
    if not prompt_file.is_file():
        raise ValueError(f"HarnessBench prompt file is missing: {prompt_file}")
    if workspace != sandbox and sandbox not in workspace.parents:
        raise ValueError(f"workspace must be inside sandbox: {workspace} not under {sandbox}")
    if not config.session_id.strip() or not config.task_id.strip():
        raise ValueError("HarnessBench task and session IDs must not be empty")
    if config.api_timeout_sec <= 0:
        raise ValueError("DeepSeek Harness API timeout must be positive")
    for relative in (
        "writer_excute.py",
        "dsh_configs/openai_compatible.cordis.yml",
        "scripts/dsh_director_bridge.py",
    ):
        if not (config.project_root / relative).is_file():
            raise ValueError(f"required project file is missing: {relative}")


def _register_proxy_route(
    *,
    task_id: str,
    session_id: str,
    role: str,
    upstream: str,
) -> str:
    proxy_base = os.environ.get("HARNESSBENCH_LLM_PROXY_URL", "").strip().rstrip("/")
    routes_raw = os.environ.get("HARNESSBENCH_LLM_PROXY_ROUTES", "").strip()
    if not proxy_base or not routes_raw:
        return upstream.rstrip("/")
    prefix = (
        f"/deepseek-harness/{_safe_segment(task_id, 'task')}/"
        f"{_safe_segment(session_id, 'session')}/{_safe_segment(role, 'role')}"
    )
    routes_file = Path(routes_raw).expanduser().resolve()
    routes: dict[str, Any] = {}
    if routes_file.is_file():
        try:
            loaded = json.loads(routes_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                routes = loaded
        except json.JSONDecodeError:
            pass
    routes[prefix] = {
        "upstream": upstream.rstrip("/"),
        "framework": f"deepseek-harness-{role}",
        "provider": "openai-compatible",
    }
    _write_json(routes_file, routes)
    return f"{proxy_base}{prefix}"


def _round_tag(prompt_file: Path) -> str:
    match = re.search(r"round(\d+)", prompt_file.stem)
    return f"round{match.group(1)}" if match else _safe_segment(prompt_file.stem, "round")


def _build_args(
    config: HarnessBenchRoundConfig,
    *,
    writer_base_url: str,
    actor_base_url: str,
    session_root: Path,
    session_id: str,
    session_per_execute: bool,
    director_stage: str,
    dsh_cwd: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        actor_backend="deepseek-harness",
        writer_backend="openai-compatible",
        writer_model=config.writer_model,
        writer_base_url=writer_base_url,
        writer_api_key=os.environ.get("WRITER_API_KEY"),
        actor_model=config.actor_model,
        actor_base_url=actor_base_url,
        actor_api_key=os.environ.get("ACTOR_API_KEY"),
        actor_output_format=None,
        execute_output_format=None,
        dsh_provider="actor-openai-compatible",
        dsh_cwd=str((dsh_cwd or config.workspace).resolve()),
        dsh_runtime_cwd=str(config.project_root.resolve()),
        dsh_session_root=str(session_root.resolve()),
        dsh_session_id=session_id,
        dsh_session_id_per_execute=session_per_execute,
        dsh_cordis=str((config.project_root / "dsh_configs" / "openai_compatible.cordis.yml").resolve()),
        dsh_runtime_bin=os.environ.get("DSH_RUNTIME_BIN"),
        dsh_request_timeout=config.api_timeout_sec,
        dsh_director_stage=director_stage,
    )


def _find_session_file(session_root: Path, session_id: str) -> Path | None:
    candidates = list(session_root.rglob(f"{session_id}/session.jsonl*"))
    return candidates[0] if candidates else None


def _writer_validation(summary: Mapping[str, Any], tool_trace: Mapping[str, Any]) -> dict[str, Any]:
    calls = [value for value in tool_trace.get("tool_calls", []) if isinstance(value, dict)]
    completed = [value for value in calls if value.get("status") in {"completed", "failed"}]
    writer_usage = summary.get("writer_usage")
    writer_requests = (
        int(writer_usage.get("request_count") or 0)
        if isinstance(writer_usage, Mapping)
        else 0
    )
    judgment_completed = isinstance(summary.get("judge_completeness_evaluation"), Mapping)
    plan_generated = isinstance(summary.get("final_script_report"), Mapping)
    complete = (
        plan_generated
        and judgment_completed
        and writer_requests > 0
        and len(completed) == len(calls)
    )
    return {
        "writer_mandatory_passed": plan_generated and judgment_completed and writer_requests > 0,
        "validation_mode": "writer-plan-plus-deepseek-trace",
        "writer_plan_generated": plan_generated,
        "writer_judgment_completed": judgment_completed,
        "writer_request_count": writer_requests,
        "actor_planning_session_id": (summary.get("actor_run_metadata") or {}).get("session_id"),
        "event_count": int(plan_generated) + int(judgment_completed) + len(calls) + len(completed),
        "tool_step_count": len(calls),
        "event_types": [
            "actor_plan_generated",
            "writer_judgment_completed",
            "actor_tool_call_observed",
            "actor_tool_result_observed",
        ],
        "incomplete_steps": {} if complete else {"pending": len(calls) - len(completed)},
        "open_steps": [] if complete else [str(value.get("call_id") or "") for value in calls if value not in completed],
        "events_complete": complete,
    }


def _director_validation(
    *, enabled: bool, tool_trace: Mapping[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not enabled:
        return None
    calls = [value for value in tool_trace.get("tool_calls", []) if isinstance(value, dict)]
    call_ids = {str(value.get("call_id") or "") for value in calls if value.get("call_id")}
    checked_ids = {str(value.get("tool_use_id") or "") for value in events if value.get("tool_use_id")}
    events_by_call = {
        str(value.get("tool_use_id")): value
        for value in events
        if value.get("tool_use_id")
    }
    ordered_call_ids = {
        call_id
        for call in calls
        if (call_id := str(call.get("call_id") or ""))
        and isinstance(call.get("completed_at"), (int, float))
        and isinstance(events_by_call.get(call_id, {}).get("timestamp"), (int, float))
        and float(events_by_call[call_id]["timestamp"]) * 1000
        <= float(call["completed_at"])
    }
    return {
        "enabled": True,
        "backend": "deepseek-harness",
        "hook": "tools/pre-execute",
        "event_count": len(events),
        "checked_tool_use_count": len(call_ids & checked_ids),
        "tool_call_count": len(calls),
        "all_tool_calls_checked": call_ids.issubset(checked_ids),
        "director_before_tool_completion": call_ids.issubset(ordered_call_ids),
        "ordered_tool_use_count": len(call_ids & ordered_call_ids),
        "ordering_basis": "Director JSONL timestamp precedes the matching DeepSeek Harness tool/result event",
        "event_types": sorted({str(value.get("event") or "") for value in events if value.get("event")}),
        "statuses": sorted({str(value.get("status") or "") for value in events if value.get("status")}),
        "mcp_repair_supported": False,
    }


def execute_harnessbench_round(
    config: HarnessBenchRoundConfig,
    *,
    task_actor: HarnessBenchTaskActor | None = None,
) -> HarnessBenchRoundResult:
    _validate(config)
    state_root = config.sandbox.resolve() / "deepseek-harnessbench"
    session_root = state_root / "sessions"
    session_root.mkdir(parents=True, exist_ok=True)
    director_log = state_root / "director" / "events.jsonl"
    trace_file = state_root / "rounds.jsonl"
    round_tag = _round_tag(config.prompt_file)
    prompt = config.prompt_file.read_text(encoding="utf-8")
    session_restored = _find_session_file(session_root, config.session_id) is not None

    writer_upstream = str(os.environ.get("WRITER_BASE_URL") or "").strip()
    actor_upstream = str(os.environ.get("ACTOR_BASE_URL") or "").strip()
    if not writer_upstream or not actor_upstream:
        raise ValueError("WRITER_BASE_URL and ACTOR_BASE_URL are required")
    if not os.environ.get("WRITER_API_KEY") or not os.environ.get("ACTOR_API_KEY"):
        raise ValueError("WRITER_API_KEY and ACTOR_API_KEY are required")
    writer_base_url = _register_proxy_route(
        task_id=config.task_id,
        session_id=config.session_id,
        role=f"writer-{round_tag}",
        upstream=writer_upstream,
    )
    actor_base_url = _register_proxy_route(
        task_id=config.task_id,
        session_id=config.session_id,
        role="actor",
        upstream=actor_upstream,
    )

    os.environ["DSH_SNAPSHOT"] = "1"
    os.environ["DSH_DIRECTOR_PYTHON"] = sys.executable
    os.environ["DSH_DIRECTOR_BRIDGE"] = str(
        (config.project_root / "scripts" / "dsh_director_bridge.py").resolve()
    )
    os.environ["DIRECTOR_LOG_PATH"] = str(director_log)
    os.environ["DIRECTOR_HARNESS_ENABLED"] = "false"

    planning_workspace = state_root / "writer-workspaces" / round_tag / "workspace"
    planning_workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.workspace, planning_workspace, dirs_exist_ok=True)
    planning_prompt = prompt.replace(
        str(config.workspace.resolve()),
        str(planning_workspace.resolve()),
    )
    planning_args = _build_args(
        config,
        writer_base_url=writer_base_url,
        actor_base_url=actor_base_url,
        session_root=session_root,
        session_id=f"{config.session_id}-writer-{round_tag}",
        session_per_execute=True,
        director_stage="writer",
        dsh_cwd=planning_workspace,
    )
    before = _workspace_snapshot(config.workspace)
    writer_result = run_writer_harness(
        planning_args,
        "writer_harness",
        planning_prompt,
        config.project_root,
    )
    after = _workspace_snapshot(config.workspace)
    writer_state_unchanged = _planning_state_unchanged(before, after)
    if not writer_state_unchanged:
        raise RuntimeError("Writer planning changed the HarnessBench workspace before Actor execution")
    writer_result["query"] = planning_prompt
    summary = _replace_path(
        summarize_writer_result(writer_result),
        planning_workspace,
        config.workspace,
    )
    summary["query"] = prompt
    decision = decide_execution(summary)
    if not decision.should_execute or not decision.execution_prompt:
        raise RuntimeError(f"Writer did not produce an executable final script: {decision.rationale}")

    writer_report_file = (
        state_root / "writer" / f"{_safe_segment(config.session_id, 'session')}-{round_tag}.json"
    )
    _write_json(
        writer_report_file,
        {
            "task_id": config.task_id,
            "session_id": config.session_id,
            "writer_session_id": (summary.get("actor_run_metadata") or {}).get("session_id"),
            "writer_model": config.writer_model,
            "planning_workspace": str(planning_workspace),
            "writer_usage": summary.get("writer_usage") or {},
            "final_script_report": summary.get("final_script_report"),
            "online_completeness_judgment": summary.get("online_completeness_judgment"),
            "judge_completeness_evaluation": summary.get("judge_completeness_evaluation"),
            "execution_decision": asdict(decision),
            "planning_state_unchanged": writer_state_unchanged,
        },
    )

    actor_args = copy.copy(planning_args)
    actor_args.dsh_cwd = str(config.workspace.resolve())
    actor_args.dsh_session_id = config.session_id
    actor_args.dsh_session_id_per_execute = False
    actor_args.dsh_director_stage = "actor"
    director_offset = len(_read_jsonl(director_log))
    os.environ["DIRECTOR_HARNESS_ENABLED"] = (
        "true" if config.director_harness_enabled else "false"
    )
    if task_actor is None:
        execution_result = run_execute_stage(
            actor_args,
            decision.execution_prompt,
            config.project_root,
        )
    else:
        execution_result = task_actor.execute(actor_args, decision.execution_prompt)
    tool_trace = execution_result.get("tool_trace") or {}
    director_events = _read_jsonl(director_log)[director_offset:]
    for event in director_events:
        event.setdefault("requested_tool_name", event.get("tool_name", ""))

    actor_metadata = execution_result.get("actor_run_metadata") or {}
    actor_trace_file = (
        state_root / "actor" / f"{_safe_segment(config.session_id, 'session')}-{round_tag}.json"
    )
    _write_json(
        actor_trace_file,
        {
            "task_id": config.task_id,
            "session_id": config.session_id,
            "round": round_tag,
            "actor_backend": "deepseek-harness",
            "actor_run_metadata": actor_metadata,
            "tool_trace": tool_trace,
            "assistant_text": execution_result.get("stdout", ""),
            "stderr": execution_result.get("stderr", ""),
        },
    )

    writer_validation = _writer_validation(summary, tool_trace)
    director_validation = _director_validation(
        enabled=config.director_harness_enabled,
        tool_trace=tool_trace,
        events=director_events,
    )
    error_events = tuple(
        value
        for value in (str(execution_result.get("stderr") or "").strip(),)
        if value
    )
    tool_calls = int(tool_trace.get("tool_call_count") or 0)
    tool_errors = int(tool_trace.get("tool_error_count") or 0)
    if not execution_result.get("ok"):
        status = "agent_task_failed"
    elif tool_errors:
        status = "completed_with_tool_errors"
    else:
        status = "completed"
    session_file = _find_session_file(session_root, config.session_id)
    compact_actor_metadata = {
        "backend": "deepseek-harness",
        "session_id": actor_metadata.get("session_id"),
        "finish_reason": actor_metadata.get("finish_reason"),
        "session_root": actor_metadata.get("session_root"),
        "cwd": actor_metadata.get("cwd"),
        "runtime_cwd": actor_metadata.get("runtime_cwd"),
        "cordis": actor_metadata.get("cordis"),
        "event_count": len(actor_metadata.get("events") or []),
    }
    result = HarnessBenchRoundResult(
        status=status,
        task_id=config.task_id,
        session_id=config.session_id,
        workspace=str(config.workspace.resolve()),
        session_restored=session_restored,
        assistant_text=str(execution_result.get("stdout") or "").strip(),
        error_events=error_events,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        session_file=str(session_file or ""),
        trace_file=str(trace_file),
        actor_trace_file=str(actor_trace_file),
        actor_backend="deepseek-harness",
        actor_run_metadata=compact_actor_metadata,
        writer_required=True,
        writer_mandatory_passed=writer_validation["writer_mandatory_passed"] is True,
        writer_state_unchanged=writer_state_unchanged,
        writer_event_validation=writer_validation,
        writer_usage={
            str(key): int(value or 0)
            for key, value in (summary.get("writer_usage") or {}).items()
            if isinstance(value, (int, float))
        },
        writer_report_file=str(writer_report_file),
        director_enabled=config.director_harness_enabled,
        director_event_validation=director_validation,
        director_events=tuple(director_events),
        director_log_file=str(director_log) if config.director_harness_enabled else None,
    )
    trace_payload = asdict(result)
    trace_payload.pop("assistant_text", None)
    _append_jsonl(trace_file, trace_payload)
    return result


def public_result_payload(result: HarnessBenchRoundResult) -> dict[str, Any]:
    payload = asdict(result)
    payload.pop("assistant_text", None)
    payload["result_label"] = RESULT_LABEL
    return payload
