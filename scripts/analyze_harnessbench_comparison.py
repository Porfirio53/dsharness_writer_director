#!/usr/bin/env python3
"""Build reproducible comparison metrics for the three HarnessBench runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = PROJECT_ROOT / "HarnessBench" / "tasks"

EXPERIMENTS = {
    "oh_original": {
        "label": "OpenHarness Original",
        "root": Path(
            "/home/patton/projects/harness/results/full/"
            "results_without_Writer/HarnessBench"
        ),
        "result_parts": ("results", "openharness-local", "qwen3.6-plus"),
    },
    "oh_wd": {
        "label": "OpenHarness Original+W&D",
        "root": Path(
            "/home/patton/projects/harness/results/runs/writer-director-full/"
            "20260817T040809Z-group-writer-v1/HarnessBench"
        ),
        "result_parts": ("results", "openharness-local", "qwen3.6-plus"),
    },
    "dsh_wd": {
        "label": "DeepSeek Harness+W&D",
        "root": PROJECT_ROOT
        / "results/runs/deepseek-writer-director-full/"
        "20260825T064925Z-group-writer-v1/HarnessBench",
        "result_parts": (
            "results",
            "deepseek-harness-writer-director",
            "qwen3.6-plus",
        ),
    },
}

PAIRWISE = (
    ("oh_original", "oh_wd"),
    ("oh_original", "dsh_wd"),
    ("oh_wd", "dsh_wd"),
)

SCORE_FIELDS = (
    "oracle_outcome_score",
    "oracle_quality",
    "outcome_score",
    "tool_use_appropriate",
    "consistency",
    "robustness",
    "process_score",
    "security_score",
    "combined_score",
)

RESOURCE_FIELDS = (
    "elapsed_sec",
    "adapter_result_count",
    "proxy_round_count",
    # ``request_count``/``input_tokens``/... are retained as aliases for the
    # values reported by ``usage_summary``.  For W&D runs those values are
    # session-level proxy totals (Actor + Writer), not Actor-only totals.
    "request_count",
    "reported_request_count",
    "input_tokens",
    "reported_input_tokens",
    "output_tokens",
    "reported_output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "noncache_tokens",
    "reported_noncache_tokens",
    "total_tokens",
    "reported_total_tokens",
    "actor_request_count",
    "actor_input_tokens",
    "actor_output_tokens",
    "actor_noncache_tokens",
    "actor_total_tokens",
    "tool_calls",
    "proxy_tool_calls",
    "tool_errors",
    "adapter_payloads_with_tool_errors",
    "session_restore_event_count",
    "session_restored_any",
    "api_timeout_payload_count",
    "api_timeout_event_count",
    "director_tool_call_count",
    "director_ordered_tool_use_count",
    "writer_input_tokens",
    "writer_output_tokens",
    "writer_total_tokens",
    "writer_request_count",
    "writer_event_count",
            "writer_tool_step_count",
            "writer_report_count",
            "api_timeout_payload_count",
            "api_timeout_event_count",
            "director_event_count",
    "director_checked_tool_use_count",
        "actor_event_count",
    "task_timeout_sec",
    "request_log_line_count",
    "upstream_attempts_sum",
    "pipeline_request_count",
    "pipeline_input_tokens",
    "pipeline_output_tokens",
    "pipeline_noncache_tokens",
    "pipeline_total_tokens",
)


def parse_json_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def read_writer_report(path_value: Any) -> dict[str, Any]:
    """Read the optional Writer report referenced by an adapter payload.

    OpenHarness stores the Writer model-call count in the report's
    ``model_request_count`` field, while DeepSeek stores a compact
    ``writer_usage`` object in the report/payload.  Keeping this helper
    tolerant of both schemas lets the accounting code use the most precise
    request count available instead of assuming one Writer call per adapter
    round.
    """
    if not path_value:
        return {}
    path = Path(str(path_value)).expanduser()
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_request_log(path_value: Any) -> dict[str, Any]:
    """Summarize the optional usage-proxy request log for one result.

    Older Original artifacts have been cleaned up and retain only the path in
    ``usage_summary``; in that case the summary explicitly reports that the
    request log was unavailable rather than treating it as zero requests.
    """
    path = Path(str(path_value)).expanduser() if path_value else None
    if path is None or not path.is_file():
        return {
            "available": False,
            "line_count": 0,
            "status_counts": {},
            "framework_counts": {},
            "framework_usage": {},
            "provider_counts": {},
            "model_counts": {},
            "upstream_attempts_sum": 0,
            "upstream_attempts_max": 0,
        }
    status_counts: Counter[str] = Counter()
    framework_counts: Counter[str] = Counter()
    framework_usage: dict[str, Counter[str]] = defaultdict(Counter)
    provider_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    attempts_sum = 0
    attempts_max = 0
    line_count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                line_count += 1
                if item.get("status") is not None:
                    status_counts[str(item["status"])] += 1
                if item.get("framework") is not None:
                    framework = str(item["framework"])
                    framework_counts[framework] += 1
                    framework_usage[framework]["request_count"] += 1
                    for field in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_tokens",
                        "cache_write_tokens",
                        "total_tokens",
                    ):
                        framework_usage[framework][field] += int(item.get(field) or 0)
                if item.get("provider") is not None:
                    provider_counts[str(item["provider"])] += 1
                if item.get("response_model") is not None:
                    model_counts[str(item["response_model"])] += 1
                attempts = int(item.get("upstream_attempts") or 0)
                attempts_sum += attempts
                attempts_max = max(attempts_max, attempts)
    except OSError:
        return {
            "available": False,
            "line_count": 0,
            "status_counts": {},
            "framework_counts": {},
            "framework_usage": {},
            "provider_counts": {},
            "model_counts": {},
            "upstream_attempts_sum": 0,
            "upstream_attempts_max": 0,
        }
    return {
        "available": True,
        "line_count": line_count,
        "status_counts": dict(status_counts),
        "framework_counts": dict(framework_counts),
        "framework_usage": {
            framework: dict(values)
            for framework, values in sorted(framework_usage.items())
        },
        "provider_counts": dict(provider_counts),
        "model_counts": dict(model_counts),
        "upstream_attempts_sum": attempts_sum,
        "upstream_attempts_max": attempts_max,
    }


def finite_values(values: Iterable[Any]) -> np.ndarray:
    cleaned = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            cleaned.append(number)
    return np.asarray(cleaned, dtype=float)


def describe(values: Iterable[Any]) -> dict[str, Any]:
    arr = finite_values(values)
    if not arr.size:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def bootstrap_mean_ci(
    values: Iterable[Any], *, seed: int = 20260828, samples: int = 20_000
) -> list[float] | None:
    arr = finite_values(values)
    if not arr.size:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(samples, arr.size))
    means = arr[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def pearson(left: Iterable[Any], right: Iterable[Any]) -> float | None:
    a = finite_values(left)
    b = finite_values(right)
    if a.size != b.size or a.size < 2 or np.isclose(a.std(), 0) or np.isclose(b.std(), 0):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_task_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(TASKS_ROOT.glob("*/task.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            task = yaml.safe_load(handle) or {}
        task_id = str(task.get("task_id") or path.parent.name)
        prompt_files = task.get("prompt_files")
        expected_rounds = len(prompt_files) if isinstance(prompt_files, list) else 1
        metadata[task_id] = {
            "task_id": task_id,
            "title": str(task.get("title") or ""),
            "class": str(task.get("class") or "Unspecified"),
            "difficulty": str(task.get("difficulty") or "unlabeled").lower(),
            "expected_rounds": expected_rounds,
            "timeout_sec": task.get("timeout_sec"),
            "has_hooks": bool(task.get("hooks_module")),
            "tags": list(task.get("tags") or []),
        }
    return metadata


def load_summary_states(root: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    with (root / "baseline-summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    states: dict[tuple[int, str], dict[str, Any]] = {}
    for repeat in summary.get("repeats", []):
        repeat_no = int(repeat["repeat"])
        for task in repeat.get("tasks", []):
            states[(repeat_no, task["task_id"])] = {
                "state": task.get("state"),
                "agent_statuses": task.get("agent_statuses") or [],
            }
    return states, summary


def extract_row(
    experiment: str,
    repeat: int,
    path: Path,
    task_meta: dict[str, dict[str, Any]],
    state_meta: dict[str, Any],
    *,
    archived_retry: bool = False,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    task_id = str(result.get("task_id") or path.stem)
    scoring = result.get("scoring") or {}
    rubric = scoring.get("rubric") or {}
    rubric_scores = rubric.get("scores") or {}
    usage = result.get("usage_summary") or {}
    adapter_results = result.get("adapter_results") or []

    statuses: list[str] = []
    error_events: list[str] = []
    tool_calls = 0
    tool_errors = 0
    adapter_payloads_with_tool_errors = 0
    writer_input = 0
    writer_output = 0
    writer_total = 0
    writer_requests = 0
    writer_report_count = 0
    writer_usage_sources: Counter[str] = Counter()
    writer_usage_mismatch_count = 0
    writer_events = 0
    writer_tool_steps = 0
    writer_mandatory_passed = True
    writer_state_unchanged = True
    writer_events_complete = True
    director_events = 0
    director_checked = 0
    director_tool_calls = 0
    director_ordered = 0
    director_all_checked = True
    director_before_completion = True
    director_statuses: Counter[str] = Counter()
    director_event_types: Counter[str] = Counter()
    writer_event_types: Counter[str] = Counter()
    actor_events = 0
    finish_reasons: Counter[str] = Counter()
    session_restore_events = 0
    session_restored_any = False
    api_timeout_payload_count = 0
    api_timeout_event_count = 0

    for adapter_result in adapter_results:
        payload = parse_json_maybe(adapter_result.get("stdout"))
        status = payload.get("status")
        if status:
            statuses.append(str(status))
        events = payload.get("error_events") or []
        error_events.extend(str(event) for event in events)
        timeout_events = [
            str(event)
            for event in events
            if re.search(r"time\s*out|timeout|timed\s*out", str(event), re.I)
        ]
        api_timeout_event_count += len(timeout_events)
        adapter_stderr = str(adapter_result.get("stderr") or "")
        if re.search(r"time\s*out|timeout|timed\s*out", adapter_stderr, re.I) or timeout_events:
            api_timeout_payload_count += 1
        tool_calls += int(payload.get("tool_calls") or 0)
        payload_tool_errors = int(payload.get("tool_errors") or 0)
        tool_errors += payload_tool_errors
        if payload_tool_errors > 0:
            adapter_payloads_with_tool_errors += 1

        # Writer usage is a subset of the same proxy session as Actor usage.
        # OpenHarness's public payload omits request_count/total_tokens, but
        # its report contains the exact model_request_count (2 or 4 calls in
        # this run).  DeepSeek exposes request_count directly.  Prefer the
        # payload values for token totals and use the report as a schema-aware
        # fallback/validation source.
        payload_writer_usage = payload.get("writer_usage")
        payload_writer_usage = (
            payload_writer_usage if isinstance(payload_writer_usage, dict) else {}
        )
        writer_report = read_writer_report(payload.get("writer_report_file"))
        report_writer_usage = writer_report.get("writer_usage")
        if not isinstance(report_writer_usage, dict):
            report_writer_usage = writer_report.get("usage")
        if not isinstance(report_writer_usage, dict):
            report_writer_usage = {}
        writer_has_evidence = bool(payload_writer_usage or report_writer_usage)
        if writer_has_evidence:
            writer_report_count += int(bool(writer_report))
            payload_input = int(payload_writer_usage.get("input_tokens") or 0)
            payload_output = int(payload_writer_usage.get("output_tokens") or 0)
            report_input = int(report_writer_usage.get("input_tokens") or 0)
            report_output = int(report_writer_usage.get("output_tokens") or 0)
            # If both representations are present, they should agree.  Keep
            # the payload as the primary source because it is what the
            # adapter actually returned, while recording any discrepancy.
            if payload_writer_usage and (
                payload_input != report_input or payload_output != report_output
            ) and writer_report:
                writer_usage_mismatch_count += 1
            writer_input += payload_input if payload_writer_usage else report_input
            writer_output += payload_output if payload_writer_usage else report_output

            if "request_count" in payload_writer_usage:
                request_value = int(payload_writer_usage.get("request_count") or 0)
                writer_usage_sources["payload.request_count"] += 1
            elif "model_request_count" in writer_report:
                request_value = int(writer_report.get("model_request_count") or 0)
                writer_usage_sources["report.model_request_count"] += 1
            elif "request_count" in report_writer_usage:
                request_value = int(report_writer_usage.get("request_count") or 0)
                writer_usage_sources["report.writer_usage.request_count"] += 1
            else:
                # Last-resort compatibility for an old report with usage but
                # no explicit count.  This branch is surfaced in the output.
                request_value = 1
                writer_usage_sources["fallback_one_per_payload"] += 1
            writer_requests += request_value
        else:
            writer_usage_sources["none"] += 1

        writer_validation = payload.get("writer_event_validation") or {}
        writer_events += int(writer_validation.get("event_count") or 0)
        writer_tool_steps += int(writer_validation.get("tool_step_count") or 0)
        for event_type in writer_validation.get("event_types") or []:
            writer_event_types[str(event_type)] += 1
        if payload.get("writer_required"):
            writer_mandatory_passed &= bool(payload.get("writer_mandatory_passed"))
            writer_state_unchanged &= bool(payload.get("writer_state_unchanged"))
            writer_events_complete &= bool(writer_validation.get("events_complete"))

        director_validation = payload.get("director_event_validation") or {}
        director_events += int(director_validation.get("event_count") or 0)
        director_checked += int(director_validation.get("checked_tool_use_count") or 0)
        director_tool_calls += int(director_validation.get("tool_call_count") or 0)
        director_ordered += int(director_validation.get("ordered_tool_use_count") or 0)
        if payload.get("director_enabled"):
            director_all_checked &= bool(director_validation.get("all_tool_calls_checked"))
            director_before_completion &= bool(
                director_validation.get("director_before_tool_completion")
            )
        for event in payload.get("director_events") or []:
            status_value = event.get("status")
            if status_value:
                director_statuses[str(status_value)] += 1
            event_type = event.get("event")
            if event_type:
                director_event_types[str(event_type)] += 1

        actor_metadata = payload.get("actor_run_metadata") or {}
        actor_events += int(actor_metadata.get("event_count") or 0)
        finish_reason = actor_metadata.get("finish_reason")
        if finish_reason:
            finish_reasons[str(finish_reason)] += 1
        if payload.get("session_restored"):
            session_restore_events += 1
            session_restored_any = True

    proxy_trace = parse_json_maybe((result.get("adapter_result") or {}).get("stdout"))
    proxy_tool_names: Counter[str] = Counter()
    proxy_tool_calls = 0
    proxy_rounds = proxy_trace.get("rounds") or []
    for round_data in proxy_rounds:
        for tool_call in round_data.get("tool_calls") or []:
            proxy_tool_calls += 1
            name = tool_call.get("name") or tool_call.get("tool_name") or "<unknown>"
            proxy_tool_names[str(name)] += 1

    request_log = read_request_log(usage.get("log_file"))

    # ``usage_summary`` is the session-level proxy total.  In Original it is
    # Actor-only; in both W&D runs it includes Writer calls because Writer uses
    # the same loopback proxy route.  Keep these values as the reported
    # (inclusive) accounting basis and never add Writer to them again.
    reported_request_count = int(usage.get("request_count") or 0)
    reported_input_tokens = int(usage.get("input_tokens") or 0)
    reported_output_tokens = int(usage.get("output_tokens") or 0)
    cache_read_tokens = int(usage.get("cache_read_tokens") or 0)
    cache_write_tokens = int(usage.get("cache_write_tokens") or 0)
    reported_noncache_tokens = reported_input_tokens + reported_output_tokens
    reported_total_tokens = int(
        usage.get("total_tokens")
        or (reported_noncache_tokens + cache_read_tokens + cache_write_tokens)
    )
    writer_total = writer_input + writer_output

    # Derive Actor-only usage where possible.  DeepSeek request logs label
    # Actor/Writer frameworks explicitly.  OpenHarness labels both as
    # ``openharness``; for that run we subtract the independently recorded
    # Writer subset and mark the result as an estimate.  Original has no
    # Writer stage, so its reported usage is Actor usage directly.
    request_framework_usage = request_log.get("framework_usage") or {}
    actor_log_frameworks = {
        str(name): value
        for name, value in request_framework_usage.items()
        if "actor" in str(name).casefold()
    }
    writer_log_frameworks = {
        str(name): value
        for name, value in request_framework_usage.items()
        if "writer" in str(name).casefold()
    }
    if actor_log_frameworks:
        actor_request_count = sum(
            int(value.get("request_count") or 0)
            for value in actor_log_frameworks.values()
        )
        actor_input_tokens = sum(
            int(value.get("input_tokens") or 0)
            for value in actor_log_frameworks.values()
        )
        actor_output_tokens = sum(
            int(value.get("output_tokens") or 0)
            for value in actor_log_frameworks.values()
        )
        actor_total_tokens = sum(
            int(value.get("total_tokens") or 0)
            for value in actor_log_frameworks.values()
        )
        actor_accounting_method = "request_log_framework_split"
        actor_usage_exact = True
    else:
        actor_request_count = reported_request_count - writer_requests
        actor_input_tokens = reported_input_tokens - writer_input
        actor_output_tokens = reported_output_tokens - writer_output
        actor_total_tokens = reported_total_tokens - writer_total
        if writer_requests or writer_total:
            actor_accounting_method = "reported_proxy_minus_writer_subset"
            actor_usage_exact = False
        else:
            actor_accounting_method = "reported_proxy_no_writer"
            actor_usage_exact = True
    # Defensive flags make an unexpected schema/token omission visible rather
    # than silently presenting negative Actor usage.
    actor_usage_nonnegative = all(
        value >= 0
        for value in (
            actor_request_count,
            actor_input_tokens,
            actor_output_tokens,
            actor_total_tokens,
        )
    )
    if not actor_usage_nonnegative:
        actor_accounting_method += ":invalid_negative_residual"

    # The pipeline total is the deduplicated proxy/session total.  Writer
    # fields above are a decomposition of this total, not an amount to add.
    pipeline_request_count = reported_request_count
    pipeline_input_tokens = reported_input_tokens
    pipeline_output_tokens = reported_output_tokens
    pipeline_noncache_tokens = reported_noncache_tokens
    pipeline_total_tokens = reported_total_tokens
    writer_log_request_count = sum(
        int(value.get("request_count") or 0) for value in writer_log_frameworks.values()
    )
    writer_log_input_tokens = sum(
        int(value.get("input_tokens") or 0) for value in writer_log_frameworks.values()
    )
    writer_log_output_tokens = sum(
        int(value.get("output_tokens") or 0) for value in writer_log_frameworks.values()
    )
    writer_log_total_tokens = sum(
        int(value.get("total_tokens") or 0) for value in writer_log_frameworks.values()
    )
    writer_log_matches_report = (
        not writer_log_frameworks
        or (
            writer_log_request_count == writer_requests
            and writer_log_input_tokens == writer_input
            and writer_log_output_tokens == writer_output
        )
    )
    component_usage_matches_reported = (
        actor_usage_nonnegative
        and actor_request_count + writer_requests == reported_request_count
        and actor_input_tokens + writer_input == reported_input_tokens
        and actor_output_tokens + writer_output == reported_output_tokens
        and actor_total_tokens + writer_total == reported_total_tokens
    )
    proxy_totals = proxy_trace.get("totals") or {}
    proxy_usage_matches = (
        int(proxy_totals.get("llm_rounds") or 0) == reported_request_count
        and int(proxy_totals.get("input_tokens") or 0) == reported_input_tokens
        and int(proxy_totals.get("output_tokens") or 0) == reported_output_tokens
        and int(proxy_totals.get("total_tokens") or 0) == reported_total_tokens
    )
    meta = task_meta.get(task_id, {})
    row = {
        "experiment": experiment,
        "repeat": repeat,
        "task_id": task_id,
        "title": meta.get("title", ""),
        "class": meta.get("class", "Unspecified"),
        "difficulty": meta.get("difficulty", "unlabeled"),
        "expected_rounds": meta.get("expected_rounds", len(adapter_results) or 1),
        "task_timeout_sec": meta.get("timeout_sec"),
        "has_hooks": meta.get("has_hooks", False),
        "tags": meta.get("tags", []),
        "actual_rounds": len(adapter_results),
        "adapter_result_count": len(adapter_results),
        "state": state_meta.get("state"),
        "agent_statuses": state_meta.get("agent_statuses") or statuses,
        "adapter_statuses": statuses,
        "archived_retry": archived_retry,
        "oracle_outcome_score": scoring.get("oracle_outcome_score"),
        "oracle_quality": scoring.get("oracle_quality"),
        "outcome_llm_weight": scoring.get("outcome_llm_weight"),
        "outcome_score": scoring.get("outcome_score"),
        "tool_use_appropriate": rubric_scores.get("tool_use_appropriate"),
        "consistency": rubric_scores.get("consistency"),
        "robustness": rubric_scores.get("robustness"),
        "process_score": scoring.get("process_score"),
        "security_score": scoring.get("security_score"),
        "combined_score": scoring.get("combined_score"),
        "rubric_skipped": bool(rubric.get("skipped")),
        "rubric_parse_error": bool(rubric.get("parse_error")),
        "proxy_trace_error": scoring.get("proxy_trace_error"),
        "elapsed_sec": float(result.get("elapsed_sec") or 0),
        "usage_available": bool(usage.get("available")),
        "usage_log_file": usage.get("log_file"),
        "request_log_available": request_log["available"],
        "request_log_line_count": request_log["line_count"],
        "request_status_counts": request_log["status_counts"],
        "request_framework_counts": request_log["framework_counts"],
        "request_framework_usage": request_log["framework_usage"],
        "request_provider_counts": request_log["provider_counts"],
        "request_model_counts": request_log["model_counts"],
        "upstream_attempts_sum": request_log["upstream_attempts_sum"],
        "upstream_attempts_max": request_log["upstream_attempts_max"],
        # Backward-compatible aliases: these are *reported proxy/session*
        # values, inclusive of Writer for W&D runs.
        "request_count": reported_request_count,
        "reported_request_count": reported_request_count,
        "input_tokens": reported_input_tokens,
        "reported_input_tokens": reported_input_tokens,
        "output_tokens": reported_output_tokens,
        "reported_output_tokens": reported_output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "noncache_tokens": reported_noncache_tokens,
        "reported_noncache_tokens": reported_noncache_tokens,
        "total_tokens": reported_total_tokens,
        "reported_total_tokens": reported_total_tokens,
        "actor_request_count": actor_request_count,
        "actor_input_tokens": actor_input_tokens,
        "actor_output_tokens": actor_output_tokens,
        "actor_noncache_tokens": actor_input_tokens + actor_output_tokens,
        "actor_total_tokens": actor_total_tokens,
        "actor_usage_exact": actor_usage_exact,
        "actor_usage_nonnegative": actor_usage_nonnegative,
        "actor_accounting_method": actor_accounting_method,
        "component_usage_matches_reported": component_usage_matches_reported,
        "tool_calls": tool_calls,
        "proxy_tool_calls": proxy_tool_calls,
        "proxy_round_count": len(proxy_rounds),
        "proxy_usage_matches_summary": proxy_usage_matches,
        "tool_count_matches_proxy": tool_calls == proxy_tool_calls,
        "tool_errors": tool_errors,
        "adapter_payloads_with_tool_errors": adapter_payloads_with_tool_errors,
        "error_event_count": len(error_events),
        "error_events": error_events,
        "tool_names": dict(proxy_tool_names),
        "writer_input_tokens": writer_input,
        "writer_output_tokens": writer_output,
        "writer_total_tokens": writer_total,
        "writer_request_count": writer_requests,
        "writer_report_count": writer_report_count,
        "writer_usage_sources": dict(writer_usage_sources),
        "writer_usage_mismatch_count": writer_usage_mismatch_count,
        "writer_log_request_count": writer_log_request_count,
        "writer_log_input_tokens": writer_log_input_tokens,
        "writer_log_output_tokens": writer_log_output_tokens,
        "writer_log_total_tokens": writer_log_total_tokens,
        "writer_log_matches_report": writer_log_matches_report,
        "writer_event_count": writer_events,
        "writer_tool_step_count": writer_tool_steps,
        "writer_mandatory_passed": writer_mandatory_passed,
        "writer_state_unchanged": writer_state_unchanged,
        "writer_events_complete": writer_events_complete,
        "writer_event_types": dict(writer_event_types),
        "director_event_count": director_events,
        "director_checked_tool_use_count": director_checked,
        "director_tool_call_count": director_tool_calls,
        "director_ordered_tool_use_count": director_ordered,
        "director_all_tool_calls_checked": director_all_checked,
        "director_before_tool_completion": director_before_completion,
        "director_statuses": dict(director_statuses),
        "director_event_types": dict(director_event_types),
        "actor_event_count": actor_events,
        "finish_reasons": dict(finish_reasons),
        "session_restore_event_count": session_restore_events,
        "session_restored_any": session_restored_any,
        "api_timeout_payload_count": api_timeout_payload_count,
        "api_timeout_event_count": api_timeout_event_count,
        "pipeline_request_count": pipeline_request_count,
        "pipeline_input_tokens": pipeline_input_tokens,
        "pipeline_output_tokens": pipeline_output_tokens,
        "pipeline_noncache_tokens": pipeline_noncache_tokens,
        "pipeline_total_tokens": pipeline_total_tokens,
    }
    return row


def load_experiment(
    experiment: str, task_meta: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = EXPERIMENTS[experiment]
    root: Path = config["root"]
    states, summary = load_summary_states(root)
    rows: list[dict[str, Any]] = []
    for repeat in (1, 2):
        result_dir = root / f"repeat-{repeat:02d}"
        for part in config["result_parts"]:
            result_dir /= part
        paths = sorted(result_dir.glob("*.json"))
        if len(paths) != 106:
            raise RuntimeError(f"{experiment} repeat {repeat}: expected 106 results, got {len(paths)}")
        for path in paths:
            task_id = path.stem
            rows.append(
                extract_row(
                    experiment,
                    repeat,
                    path,
                    task_meta,
                    states.get((repeat, task_id), {}),
                )
            )
    return rows, summary


def load_deepseek_retries(task_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    root: Path = EXPERIMENTS["dsh_wd"]["root"] / "retry-history"
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("repeat-*/*/*/*.json")):
        repeat = int(path.parts[-4].split("-")[-1])
        rows.append(
            extract_row(
                "dsh_wd_retry",
                repeat,
                path,
                task_meta,
                {},
                archived_retry=True,
            )
        )
    return rows


def group_task_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, dict[str, float | None]] = {}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    for task_id, task_rows in sorted(by_task.items()):
        record: dict[str, float | None] = {}
        for field in SCORE_FIELDS + RESOURCE_FIELDS:
            arr = finite_values(row.get(field) for row in task_rows)
            record[field] = float(arr.mean()) if arr.size else None
        grouped[task_id] = record
    return grouped


def aggregate_tool_names(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("tool_names") or {})
    total = sum(counts.values())
    return [
        {"name": name, "count": count, "share": count / total if total else 0.0}
        for name, count in counts.most_common()
    ]


TOOL_NAME_NORMALIZATION = {
    "read_file": "read",
    "read": "read",
    "write_file": "write",
    "write": "write",
    "edit_file": "edit",
    "edit": "edit",
    "image_generation": "image_generation",
    "image_to_text": "image_to_text",
}


def aggregate_normalized_tool_names(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for name, count in (row.get("tool_names") or {}).items():
            normalized = TOOL_NAME_NORMALIZATION.get(str(name), str(name))
            counts[normalized] += int(count)
    total = sum(counts.values())
    return [
        {"name": name, "count": count, "share": count / total if total else 0.0}
        for name, count in counts.most_common()
    ]


def aggregate_counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, dict):
            counts.update({str(key): int(count) for key, count in value.items()})
        elif isinstance(value, str):
            counts[value] += 1
        else:
            counts.update(str(item) for item in (value or []))
    return dict(sorted(counts.items()))


def aggregate_nested_counter(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    """Sum a mapping-of-mappings such as framework token usage."""
    output: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = row.get(field)
        if not isinstance(value, dict):
            continue
        for group, metrics in value.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, amount in metrics.items():
                output[str(group)][str(metric_name)] += int(amount or 0)
    return {
        group: dict(sorted(metrics.items()))
        for group, metrics in sorted(output.items())
    }


def availability_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if bool(row.get(field)))


def last_status(row: dict[str, Any]) -> str:
    """Return the terminal adapter status recorded for one result row."""
    statuses = row.get("adapter_statuses") or []
    if statuses:
        return str(statuses[-1])
    state = row.get("state")
    return str(state) if state else "unknown"


def load_config_snapshot(experiment: str) -> dict[str, Any]:
    """Keep the run configuration needed to judge comparability.

    The raw run-config files contain the complete task list and, for the
    OpenHarness writer run, a large source manifest.  The comparison artifact
    only needs the stable execution knobs and source/version locks; retaining
    those fields makes the report auditable without duplicating the manifest.
    """
    root: Path = EXPERIMENTS[experiment]["root"]
    path = root / "run-config.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    selected_keys = (
        "schema_version",
        "result_label",
        "dataset_id",
        "dataset_manifest",
        "repeats",
        "model",
        "profile",
        "api_format",
        "temperature",
        "seed",
        "max_turns",
        "api_timeout_sec",
        "actor_backend",
        "execution_mode",
        "openharness_mode",
        "writer_model",
        "writer_max_tokens",
        "source_lock_enforced",
        "director_harness_enabled",
        "director_mcp_catalog",
        "director_mcp_repair_supported",
        "public_url_mode",
        "grading",
        "created_at",
    )
    snapshot = {key: config[key] for key in selected_keys if key in config}
    for key in (
        "harnessbench_git",
        "harnessbench_source_sha256",
        "openharness_git",
        "openharness_source_sha256",
        "writer_deployment",
        "deepseek_harness_project_git",
        "deepseek_harness_source_sha256",
    ):
        if key in config:
            value = config[key]
            # The deployment object can include a detailed per-file lock.  It
            # is useful, but omit bulky status arrays from the compact snapshot.
            if key.endswith("_git") and isinstance(value, dict):
                value = {
                    item: value[item]
                    for item in ("root", "branch", "commit", "dirty")
                    if item in value
                }
            snapshot[key] = value
    snapshot["run_config_path"] = str(path)
    snapshot["result_root"] = str(root)

    # ``run-config.json`` intentionally omits some adapter-level knobs.  Keep
    # the local HarnessBench config alongside it so timeout/adapter settings
    # are auditable (and so a missing DSH CLI flag is not mistaken for zero).
    harness_local_path = root / "harness.local.json"
    if harness_local_path.is_file():
        try:
            with harness_local_path.open("r", encoding="utf-8") as handle:
                harness_local = json.load(handle)
            model_cfg = (harness_local.get("models") or {}).get(
                next(iter((harness_local.get("models") or {})), ""), {}
            )
            if isinstance(model_cfg, dict):
                snapshot["adapter_config"] = {
                    key: model_cfg.get(key)
                    for key in (
                        "adapter",
                        "command",
                        "timeout_sec",
                        "api_timeout_sec",
                        "project_root",
                        "director_harness_enabled",
                    )
                    if key in model_cfg
                }
                args = model_cfg.get("args")
                if isinstance(args, list):
                    snapshot["adapter_args"] = [str(item) for item in args]
            snapshot["harness_local_path"] = str(harness_local_path)
        except (OSError, json.JSONDecodeError):
            snapshot["adapter_config_read_error"] = True
    app_local_path = root / "repeat-01" / "app.local.json"
    if app_local_path.is_file():
        try:
            with app_local_path.open("r", encoding="utf-8") as handle:
                app_local = json.load(handle)
            if isinstance(app_local, dict):
                snapshot["app_local_config"] = {
                    key: app_local.get(key)
                    for key in ("default_timeout_sec", "data_dir", "tasks_dir")
                    if key in app_local
                }
            snapshot["app_local_path"] = str(app_local_path)
        except (OSError, json.JSONDecodeError):
            snapshot["app_local_config_read_error"] = True
    return snapshot


def load_run_state_snapshot(experiment: str) -> dict[str, Any]:
    """Return run lifecycle information without embedding every task record."""
    root: Path = EXPERIMENTS[experiment]["root"]
    path = root / "run-state.json"
    if not path.exists():
        return {"run_state_path": str(path), "available": False}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    task_runs = state.get("task_runs") or []
    nonzero = [
        {"repeat": item.get("repeat"), "task_id": item.get("task_id"), "returncode": item.get("returncode")}
        for item in task_runs
        if item.get("returncode") not in (None, 0)
    ]
    resume_history = state.get("resume_history") or []
    config_path = root / "run-config.json"
    final_config: dict[str, Any] = {}
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                final_config = json.load(handle)
        except (OSError, json.JSONDecodeError):
            final_config = {}
    final_hashes = {
        key: value
        for key, value in final_config.items()
        if key.endswith("_source_sha256") and isinstance(value, dict)
    }
    source_drift: list[dict[str, Any]] = []
    for index, item in enumerate(resume_history):
        for hash_group, historical in item.items():
            if not hash_group.endswith("_source_sha256") or not isinstance(historical, dict):
                continue
            final = final_hashes.get(hash_group, {})
            changed = sorted(
                key for key in set(historical) | set(final)
                if historical.get(key) != final.get(key)
            )
            if changed:
                source_drift.append(
                    {
                        "resume_index": index,
                        "hash_group": hash_group,
                        "changed_files": changed,
                    }
                )
    return {
        "available": True,
        "run_state_path": str(path),
        "finished_at": state.get("finished_at"),
        "interrupted_at": state.get("interrupted_at"),
        "last_finished": state.get("last_finished"),
        "task_run_record_count": len(task_runs),
        "nonzero_returncode_count": len(nonzero),
        "nonzero_returncodes": nonzero,
        "resume_count": len(resume_history),
        "source_lock_drift_vs_final_config": source_drift,
        "resume_history": resume_history,
    }


def score_thresholds(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    arr = finite_values(row.get(field) for row in rows)
    return {
        "eq_1": int(np.sum(np.isclose(arr, 1.0))),
        "ge_0_9": int(np.sum(arr >= 0.9)),
        "ge_0_8": int(np.sum(arr >= 0.8)),
        "lt_0_5": int(np.sum(arr < 0.5)),
        "eq_0": int(np.sum(np.isclose(arr, 0.0))),
    }


def aggregate_experiment(
    experiment: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    task_means = group_task_means(rows)
    metrics = {field: describe(row.get(field) for row in rows) for field in SCORE_FIELDS + RESOURCE_FIELDS}
    ci = {
        field: bootstrap_mean_ci(record.get(field) for record in task_means.values())
        for field in ("outcome_score", "process_score", "combined_score")
    }
    # Correlations are descriptive only (the benchmark is not a causal study).
    correlation_resources = (
        "tool_calls",
        "proxy_tool_calls",
        "tool_errors",
        "request_count",
        "pipeline_noncache_tokens",
        "elapsed_sec",
    )
    score_correlations: dict[str, dict[str, float | None]] = {}
    for score_field in ("outcome_score", "process_score", "combined_score"):
        score_correlations[score_field] = {}
        for resource_field in correlation_resources:
            score_correlations[score_field][resource_field] = pearson(
                (record.get(score_field) for record in task_means.values()),
                (record.get(resource_field) for record in task_means.values()),
            )
    per_repeat: dict[str, Any] = {}
    for repeat in (1, 2):
        repeat_rows = [row for row in rows if row["repeat"] == repeat]
        per_repeat[str(repeat)] = {
            field: describe(row.get(field) for row in repeat_rows)
            for field in (
                "outcome_score",
                "process_score",
                "combined_score",
                "elapsed_sec",
                "request_count",
                "reported_request_count",
                "actor_request_count",
                "noncache_tokens",
                "reported_noncache_tokens",
                "actor_noncache_tokens",
                "total_tokens",
                "reported_total_tokens",
                "actor_total_tokens",
                "writer_total_tokens",
                "writer_request_count",
                "api_timeout_payload_count",
                "api_timeout_event_count",
                "tool_calls",
                "proxy_tool_calls",
                "tool_errors",
                "pipeline_request_count",
                "pipeline_noncache_tokens",
                "pipeline_total_tokens",
            )
        }

    stability: dict[str, Any] = {}
    by_repeat_task = {(row["repeat"], row["task_id"]): row for row in rows}
    task_ids = sorted(task_means)
    for field in (
        "outcome_score",
        "process_score",
        "combined_score",
        "tool_calls",
        "proxy_tool_calls",
        "elapsed_sec",
    ):
        first = np.asarray([by_repeat_task[(1, task_id)][field] for task_id in task_ids], dtype=float)
        second = np.asarray([by_repeat_task[(2, task_id)][field] for task_id in task_ids], dtype=float)
        delta = second - first
        stability[field] = {
            "repeat_2_minus_1_mean": float(delta.mean()),
            "mean_abs_delta": float(np.abs(delta).mean()),
            "median_abs_delta": float(np.median(np.abs(delta))),
            "pearson_r": pearson(first, second),
            "exact_match_count": int(np.sum(np.isclose(delta, 0.0))),
            "abs_delta_ge_0_1_count": int(np.sum(np.abs(delta) >= 0.1))
            if field in SCORE_FIELDS
            else None,
        }

    adapter_statuses = aggregate_counter(rows, "adapter_statuses")
    agent_statuses = aggregate_counter(rows, "agent_statuses")
    director_statuses = aggregate_counter(rows, "director_statuses")
    result_rows_with_tool_errors = sum(1 for row in rows if row["tool_errors"] > 0)
    unique_tasks_with_tool_errors = len(
        {row["task_id"] for row in rows if row["tool_errors"] > 0}
    )
    adapter_rounds_with_tool_errors = sum(
        row.get("adapter_payloads_with_tool_errors", 0) for row in rows
    )
    task_rounds_with_errors = sum(1 for row in rows if row["error_event_count"] > 0)
    tool_mismatch_rows = [
        {"repeat": row["repeat"], "task_id": row["task_id"], "adapter": row["tool_calls"], "proxy": row["proxy_tool_calls"]}
        for row in rows
        if not row["tool_count_matches_proxy"]
    ]

    # ``noncache_tokens`` is the reported/session total (kept as a backwards
    # compatible alias).  Explicit fields below make the decomposition and
    # the deduplicated pipeline total auditable.
    noncache_total = metrics["reported_noncache_tokens"].get("sum", 0.0)
    reported_input_total = metrics["reported_input_tokens"].get("sum", 0.0)
    reported_output_total = metrics["reported_output_tokens"].get("sum", 0.0)
    reported_total = metrics["reported_total_tokens"].get("sum", 0.0)
    reported_request_total = metrics["reported_request_count"].get("sum", 0.0)
    actor_input_total = metrics["actor_input_tokens"].get("sum", 0.0)
    actor_output_total = metrics["actor_output_tokens"].get("sum", 0.0)
    actor_noncache_total = metrics["actor_noncache_tokens"].get("sum", 0.0)
    actor_total = metrics["actor_total_tokens"].get("sum", 0.0)
    actor_request_total = metrics["actor_request_count"].get("sum", 0.0)
    writer_total = metrics["writer_total_tokens"].get("sum", 0.0)
    tool_total = metrics["tool_calls"].get("sum", 0.0)
    proxy_tool_total = metrics["proxy_tool_calls"].get("sum", 0.0)
    request_total = metrics["request_count"].get("sum", 0.0)
    writer_request_total = metrics["writer_request_count"].get("sum", 0.0)
    pipeline_request_total = metrics["pipeline_request_count"].get("sum", 0.0)
    pipeline_noncache_total = metrics["pipeline_noncache_tokens"].get("sum", 0.0)
    pipeline_total = metrics["pipeline_total_tokens"].get("sum", 0.0)

    return {
        "label": EXPERIMENTS[experiment]["label"],
        "config": load_config_snapshot(experiment),
        "run_state": load_run_state_snapshot(experiment),
        "result_rows": len(rows),
        "task_count": len(task_means),
        "state_counts": summary.get("state_counts"),
        "metrics": metrics,
        "score_mean_ci_95_task_bootstrap": ci,
        "task_mean_score_resource_pearson_r": score_correlations,
        "thresholds": {
            field: score_thresholds(rows, field)
            for field in ("outcome_score", "process_score", "combined_score")
        },
        "per_repeat": per_repeat,
        "repeat_stability": stability,
        "adapter_statuses": adapter_statuses,
        "agent_statuses": agent_statuses,
        "terminal_statuses": dict(Counter(last_status(row) for row in rows)),
        # Keep the old key for consumers of schema v1; the explicit names are
        # less ambiguous (a row is one task×repeat result, not necessarily a
        # unique task).
        "tasks_with_tool_errors": result_rows_with_tool_errors,
        "result_rows_with_tool_errors": result_rows_with_tool_errors,
        "unique_tasks_with_tool_errors": unique_tasks_with_tool_errors,
        "adapter_rounds_with_tool_errors": adapter_rounds_with_tool_errors,
        "total_tool_error_events": int(metrics["tool_errors"].get("sum", 0.0)),
        "rows_with_error_events": task_rounds_with_errors,
        "tool_error_rate": metrics["tool_errors"].get("sum", 0.0) / tool_total if tool_total else 0.0,
        "proxy_tool_error_rate": metrics["tool_errors"].get("sum", 0.0) / proxy_tool_total if proxy_tool_total else 0.0,
        "usage_available_count": sum(1 for row in rows if row["usage_available"]),
        "request_log_available_count": availability_count(rows, "request_log_available"),
        "request_log_line_total": int(sum(row["request_log_line_count"] for row in rows)),
        "request_status_counts": aggregate_counter(rows, "request_status_counts"),
        "request_framework_counts": aggregate_counter(rows, "request_framework_counts"),
        "request_framework_usage": aggregate_nested_counter(rows, "request_framework_usage"),
        "request_provider_counts": aggregate_counter(rows, "request_provider_counts"),
        "request_model_counts": aggregate_counter(rows, "request_model_counts"),
        "upstream_attempts_total": int(sum(row["upstream_attempts_sum"] for row in rows)),
        "upstream_attempts_max": int(max((row["upstream_attempts_max"] for row in rows), default=0)),
        "rubric_skipped_count": sum(1 for row in rows if row["rubric_skipped"]),
        "rubric_parse_error_count": sum(1 for row in rows if row["rubric_parse_error"]),
        "proxy_trace_error_count": sum(1 for row in rows if row["proxy_trace_error"]),
        "tool_count_mismatch_count": len(tool_mismatch_rows),
        "tool_count_mismatches": tool_mismatch_rows,
        "adapter_tool_calls_total": tool_total,
        "proxy_tool_calls_total": proxy_tool_total,
        "tool_call_count_difference_proxy_minus_adapter": proxy_tool_total - tool_total,
        "proxy_usage_summary_match_count": sum(1 for row in rows if row["proxy_usage_matches_summary"]),
        "writer_report_available_count": int(sum(row["writer_report_count"] for row in rows)),
        "writer_usage_mismatch_count": int(sum(row["writer_usage_mismatch_count"] for row in rows)),
        "writer_usage_source_counts": aggregate_counter(rows, "writer_usage_sources"),
        "writer_log_matches_report_count": int(sum(1 for row in rows if row["writer_log_matches_report"])),
        "top_tools": aggregate_tool_names(rows),
        "top_tools_normalized": aggregate_normalized_tool_names(rows),
        "writer_token_share_of_noncache": writer_total / noncache_total if noncache_total else None,
        "writer_token_share_of_pipeline_noncache": writer_total / pipeline_noncache_total if pipeline_noncache_total else None,
        "nonwriter_tokens": actor_noncache_total,
        "requests_per_tool_call": request_total / tool_total if tool_total else None,
        "requests_per_proxy_tool_call": request_total / proxy_tool_total if proxy_tool_total else None,
        "writer_requests_total": writer_request_total,
        "pipeline_requests_total": pipeline_request_total,
        "pipeline_noncache_tokens_total": pipeline_noncache_total,
        "pipeline_total_tokens_total": pipeline_total,
        "reported_usage_totals": {
            "request_count": reported_request_total,
            "input_tokens": reported_input_total,
            "output_tokens": reported_output_total,
            "noncache_tokens": noncache_total,
            "total_tokens": reported_total,
        },
        "actor_usage_totals": {
            "request_count": actor_request_total,
            "input_tokens": actor_input_total,
            "output_tokens": actor_output_total,
            "noncache_tokens": actor_noncache_total,
            "total_tokens": actor_total,
        },
        "writer_usage_totals": {
            "request_count": writer_request_total,
            "input_tokens": metrics["writer_input_tokens"].get("sum", 0.0),
            "output_tokens": metrics["writer_output_tokens"].get("sum", 0.0),
            "noncache_tokens": writer_total,
            "total_tokens": writer_total,
        },
        "actor_accounting_methods": aggregate_counter(
            rows, "actor_accounting_method"
        ),
        "actor_usage_exact_count": int(sum(1 for row in rows if row["actor_usage_exact"])),
        "actor_usage_nonnegative_count": int(sum(1 for row in rows if row["actor_usage_nonnegative"])),
        "component_usage_matches_reported_count": int(
            sum(1 for row in rows if row["component_usage_matches_reported"])
        ),
        "finish_reasons": aggregate_counter(rows, "finish_reasons"),
        "writer_event_types": aggregate_counter(rows, "writer_event_types"),
        "director_event_types": aggregate_counter(rows, "director_event_types"),
        "session_restore_event_count": int(sum(row["session_restore_event_count"] for row in rows)),
        "session_restored_result_count": int(sum(row["session_restored_any"] for row in rows)),
        "api_timeout_payload_count": int(sum(row["api_timeout_payload_count"] for row in rows)),
        "api_timeout_event_count": int(sum(row["api_timeout_event_count"] for row in rows)),
        "director_statuses": director_statuses,
        "writer_gate": summary.get("writer_smoke_gate"),
        "director_gate": summary.get("director_smoke_gate"),
        "task_means": task_means,
    }


def category_tables(
    all_rows: dict[str, list[dict[str, Any]]],
    field: str,
    group_field: str,
) -> list[dict[str, Any]]:
    task_meta: dict[str, str] = {}
    values: dict[str, dict[str, float]] = {}
    for experiment, rows in all_rows.items():
        means = group_task_means(rows)
        values[experiment] = {
            task_id: float(record[field]) for task_id, record in means.items() if record[field] is not None
        }
        for row in rows:
            task_meta[row["task_id"]] = str(row[group_field])

    groups = sorted(set(task_meta.values()))
    output = []
    for group in groups:
        task_ids = sorted(task_id for task_id, value in task_meta.items() if value == group)
        record: dict[str, Any] = {"group": group, "task_count": len(task_ids)}
        for experiment in EXPERIMENTS:
            arr = finite_values(values[experiment].get(task_id) for task_id in task_ids)
            record[experiment] = float(arr.mean()) if arr.size else None
        record["oh_wd_minus_original"] = record["oh_wd"] - record["oh_original"]
        record["dsh_wd_minus_original"] = record["dsh_wd"] - record["oh_original"]
        record["dsh_wd_minus_oh_wd"] = record["dsh_wd"] - record["oh_wd"]
        output.append(record)
    return output


def pairwise_comparisons(aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for left, right in PAIRWISE:
        pair_key = f"{right}_minus_{left}"
        left_means = aggregates[left]["task_means"]
        right_means = aggregates[right]["task_means"]
        task_ids = sorted(set(left_means) & set(right_means))
        pair_metrics: dict[str, Any] = {}
        for field in (
            "outcome_score",
            "process_score",
            "combined_score",
            "elapsed_sec",
            "request_count",
            "actor_request_count",
            "noncache_tokens",
            "actor_noncache_tokens",
            "total_tokens",
            "actor_total_tokens",
            "tool_calls",
            "proxy_tool_calls",
            "tool_errors",
            "pipeline_request_count",
            "pipeline_noncache_tokens",
            "pipeline_total_tokens",
        ):
            left_arr = np.asarray([left_means[task_id][field] for task_id in task_ids], dtype=float)
            right_arr = np.asarray([right_means[task_id][field] for task_id in task_ids], dtype=float)
            delta = right_arr - left_arr
            epsilon = 1e-9
            material = 0.01 if field in ("outcome_score", "process_score", "combined_score") else epsilon
            pair_metrics[field] = {
                "left_mean": float(left_arr.mean()),
                "right_mean": float(right_arr.mean()),
                "mean_delta": float(delta.mean()),
                "relative_delta": float(delta.mean() / left_arr.mean()) if left_arr.mean() else None,
                "paired_delta_ci_95_task_bootstrap": bootstrap_mean_ci(
                    delta, seed=20260828 + len(pair_metrics)
                ),
                "win_tie_loss_any": {
                    "right_wins": int(np.sum(delta > epsilon)),
                    "ties": int(np.sum(np.abs(delta) <= epsilon)),
                    "right_losses": int(np.sum(delta < -epsilon)),
                },
                "win_tie_loss_material": {
                    "right_wins": int(np.sum(delta > material)),
                    "ties": int(np.sum(np.abs(delta) <= material)),
                    "right_losses": int(np.sum(delta < -material)),
                    "threshold": material,
                },
            }
        output[pair_key] = pair_metrics
    return output


def task_comparison_table(
    all_rows: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    means = {experiment: group_task_means(rows) for experiment, rows in all_rows.items()}
    task_meta = {row["task_id"]: row for row in all_rows["dsh_wd"]}
    rows_by_experiment_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for experiment, rows in all_rows.items():
        for item in rows:
            rows_by_experiment_task[(experiment, item["task_id"])].append(item)
    output = []
    for task_id in sorted(means["oh_original"]):
        row = {
            "task_id": task_id,
            "title": task_meta[task_id]["title"],
            "class": task_meta[task_id]["class"],
            "difficulty": task_meta[task_id]["difficulty"],
        }
        for field in (
            "outcome_score",
            "oracle_outcome_score",
            "oracle_quality",
            "tool_use_appropriate",
            "consistency",
            "robustness",
            "process_score",
            "security_score",
            "combined_score",
            "tool_calls",
            "proxy_tool_calls",
            "tool_errors",
            "actor_request_count",
            "request_count",
            "pipeline_request_count",
            "actor_input_tokens",
            "actor_output_tokens",
            "actor_noncache_tokens",
            "actor_total_tokens",
            "input_tokens",
            "reported_input_tokens",
            "output_tokens",
            "reported_output_tokens",
            "cache_read_tokens",
            "noncache_tokens",
            "reported_noncache_tokens",
            "total_tokens",
            "reported_total_tokens",
            "pipeline_noncache_tokens",
            "pipeline_total_tokens",
            "writer_total_tokens",
            "writer_request_count",
            "writer_report_count",
            "writer_event_count",
            "writer_tool_step_count",
            "director_event_count",
            "director_checked_tool_use_count",
            "actor_event_count",
            "elapsed_sec",
            "task_timeout_sec",
        ):
            for experiment in EXPERIMENTS:
                row[f"{experiment}_{field}"] = means[experiment][task_id][field]
        for experiment in EXPERIMENTS:
            task_rows = rows_by_experiment_task[(experiment, task_id)]
            row[f"{experiment}_terminal_statuses"] = dict(
                Counter(last_status(item) for item in task_rows)
            )
            row[f"{experiment}_error_row_count"] = sum(
                1 for item in task_rows if item["tool_errors"] > 0
            )
            row[f"{experiment}_proxy_mismatch_row_count"] = sum(
                1 for item in task_rows if not item["tool_count_matches_proxy"]
            )
        row["oh_wd_minus_original_outcome"] = (
            row["oh_wd_outcome_score"] - row["oh_original_outcome_score"]
        )
        row["dsh_wd_minus_original_outcome"] = (
            row["dsh_wd_outcome_score"] - row["oh_original_outcome_score"]
        )
        row["dsh_wd_minus_oh_wd_outcome"] = (
            row["dsh_wd_outcome_score"] - row["oh_wd_outcome_score"]
        )
        row["dsh_wd_minus_oh_wd_combined"] = (
            row["dsh_wd_combined_score"] - row["oh_wd_combined_score"]
        )
        output.append(row)
    return output


def retry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for row in rows:
        statuses = set(row.get("adapter_statuses") or [])
        events = " ".join(row.get("error_events") or [])
        if "already has a persisted log" in events:
            reason_counts["session_id_collision"] += 1
        elif "502" in events or "proxy upstream error" in events:
            reason_counts["upstream_502"] += 1
        elif "deepseek_harness_process_failed" in statuses:
            reason_counts["deepseek_harness_process_failed"] += 1
        else:
            reason_counts["other"] += 1
    metrics = {
        field: describe(row.get(field) for row in rows)
        for field in (
            "elapsed_sec",
            "request_count",
            "actor_request_count",
            "input_tokens",
            "actor_input_tokens",
            "output_tokens",
            "actor_output_tokens",
            "noncache_tokens",
            "actor_noncache_tokens",
            "total_tokens",
            "actor_total_tokens",
            "tool_calls",
            "proxy_tool_calls",
            "tool_errors",
            "writer_total_tokens",
            "writer_request_count",
            "pipeline_request_count",
            "pipeline_noncache_tokens",
            "pipeline_total_tokens",
        )
    }
    return {
        "archived_attempt_count": len(rows),
        "unique_task_repeat_pairs": len({(row["repeat"], row["task_id"]) for row in rows}),
        "reason_counts": dict(sorted(reason_counts.items())),
        "metrics": metrics,
        "task_ids": [f"r{row['repeat']}:{row['task_id']}" for row in rows],
    }


def build_analysis() -> dict[str, Any]:
    task_meta = load_task_metadata()
    all_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        rows, summary = load_experiment(experiment, task_meta)
        all_rows[experiment] = rows
        summaries[experiment] = summary

    aggregates = {
        experiment: aggregate_experiment(experiment, rows, summaries[experiment])
        for experiment, rows in all_rows.items()
    }
    retries = load_deepseek_retries(task_meta)
    task_table = task_comparison_table(all_rows)

    top_changes: dict[str, Any] = {}
    for field in (
        "oh_wd_minus_original_outcome",
        "dsh_wd_minus_original_outcome",
        "dsh_wd_minus_oh_wd_outcome",
        "dsh_wd_minus_oh_wd_combined",
    ):
        sorted_rows = sorted(task_table, key=lambda row: row[field])
        top_changes[field] = {
            "bottom_10": [
                {"task_id": row["task_id"], "delta": row[field]} for row in sorted_rows[:10]
            ],
            "top_10": [
                {"task_id": row["task_id"], "delta": row[field]}
                for row in reversed(sorted_rows[-10:])
            ],
        }

    return {
        "analysis_schema_version": 3,
        "usage_accounting": {
            "reported": "usage_summary / usage-proxy session total; W&D includes Actor and Writer",
            "pipeline": "deduplicated reported session total (Writer is not added a second time)",
            "actor": "DeepSeek: request-log framework split; OpenHarness W&D: reported total minus Writer report subset (estimate); Original: reported total",
            "writer": "adapter writer_usage, with OpenHarness model_request_count recovered from writer_report_file",
        },
        "task_count": len(task_meta),
        "expected_result_rows_per_experiment": len(task_meta) * 2,
        "experiments": aggregates,
        "pairwise": pairwise_comparisons(aggregates),
        "by_class": {
            field: category_tables(all_rows, field, "class")
            for field in (
                "outcome_score",
                "process_score",
                "combined_score",
                "tool_calls",
                "proxy_tool_calls",
                "noncache_tokens",
                "actor_noncache_tokens",
                "pipeline_noncache_tokens",
                "elapsed_sec",
                "request_count",
                "actor_request_count",
                "pipeline_request_count",
                "tool_errors",
            )
        },
        "by_difficulty": {
            field: category_tables(all_rows, field, "difficulty")
            for field in ("outcome_score", "process_score", "combined_score")
        },
        "deepseek_retry_history": retry_summary(retries),
        "top_changes": top_changes,
        "task_table": task_table,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("all", "summary", "pairwise", "class", "difficulty", "retries", "tasks"),
        default="all",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    analysis = build_analysis()
    if args.section == "summary":
        payload = {"experiments": analysis["experiments"]}
        for experiment in payload["experiments"].values():
            experiment.pop("task_means", None)
    elif args.section == "pairwise":
        payload = analysis["pairwise"]
    elif args.section == "class":
        payload = analysis["by_class"]
    elif args.section == "difficulty":
        payload = analysis["by_difficulty"]
    elif args.section == "retries":
        payload = analysis["deepseek_retry_history"]
    elif args.section == "tasks":
        payload = analysis["task_table"]
    else:
        payload = analysis

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
