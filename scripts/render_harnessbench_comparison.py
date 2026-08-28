#!/usr/bin/env python3
"""Render the reproducible HarnessBench three-way comparison artifacts.

The numerical source of truth is ``analyze_harnessbench_comparison``.  This
module deliberately contains presentation code only: it imports the analysis
functions, writes a machine-readable snapshot, and renders a Chinese report
whose tables are populated from that snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_ROOT / "results" / "docs"

# Make the sibling analysis module importable when this file is run directly.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_harnessbench_comparison import (  # noqa: E402
    EXPERIMENTS,
    build_analysis,
    load_experiment,
    load_task_metadata,
)


EXPERIMENT_ORDER = ("oh_original", "oh_wd", "dsh_wd")
EXPERIMENT_SHORT = {
    "oh_original": "OH Original",
    "oh_wd": "OH Original+W&D",
    "dsh_wd": "DeepSeek Harness+W&D",
}


def num(value: Any, digits: int = 3) -> str:
    """Format a finite number for Markdown without unnecessary noise."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "—"
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def integer(value: Any) -> str:
    return num(value, 0)


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def signed(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.{digits}f}"


def signed_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        value = float(value) * 100
    except (TypeError, ValueError):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = [str(item) for item in headers]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [md_escape(item) for item in row]
        if len(values) < len(headers):
            values += ["—"] * (len(headers) - len(values))
        lines.append("| " + " | ".join(values[: len(headers)]) + " |")
    return "\n".join(lines)


def metric(analysis: dict[str, Any], experiment: str, field: str, stat: str = "mean") -> Any:
    return analysis["experiments"][experiment]["metrics"].get(field, {}).get(stat)


def ci_text(ci: Any, digits: int = 3) -> str:
    if not ci or len(ci) != 2:
        return "—"
    return f"[{num(ci[0], digits)}, {num(ci[1], digits)}]"


def status_text(value: Any) -> str:
    if not isinstance(value, dict):
        return md_escape(value)
    return ", ".join(f"{key}={integer(count)}" for key, count in sorted(value.items())) or "—"


def top_tool_text(tools: list[dict[str, Any]], limit: int = 8) -> str:
    return ", ".join(
        f"{item['name']} {integer(item['count'])} ({pct(item['share'])})"
        for item in tools[:limit]
    ) or "—"


def source_path(experiment: str) -> str:
    return str(EXPERIMENTS[experiment]["root"])


def collect_raw_rows() -> list[dict[str, Any]]:
    metadata = load_task_metadata()
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        loaded, _ = load_experiment(experiment, metadata)
        rows.extend(loaded)
    return rows


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    # The task table includes JSON objects for status fields.  CSV keeps those
    # values lossless and human-readable rather than silently dropping them.
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            output: dict[str, Any] = {}
            for key in keys:
                value = record.get(key)
                output[key] = json_compact(value) if isinstance(value, (dict, list)) else value
            writer.writerow(output)


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_category_csvs(analysis: dict[str, Any]) -> None:
    for field, records in analysis["by_class"].items():
        write_csv(
            DOCS_DIR / f"harnessbench_by_class_{field}.csv",
            records,
        )
    for field, records in analysis["by_difficulty"].items():
        write_csv(
            DOCS_DIR / f"harnessbench_by_difficulty_{field}.csv",
            records,
        )


def render_report(analysis: dict[str, Any], raw_rows: list[dict[str, Any]]) -> str:
    exps = analysis["experiments"]
    pairwise = analysis["pairwise"]
    lines: list[str] = []
    add = lines.append

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    add("# HarnessBench 三组实验对比报告")
    add("")
    add(f"> 生成时间：`{generated}`（UTC）。数值由 `scripts/analyze_harnessbench_comparison.py` 直接读取三组结果目录计算；本报告不改写原始结果。")
    add("")
    add("## 1. 摘要与结论")
    add("")
    add("本报告比较同一套 HarnessBench 的 106 个任务、每题 2 轮（每组 212 条最终结果）：OpenHarness Original、OpenHarness Original+Writer&Director（以下简称 OH W&D）以及 DeepSeek Harness+Writer&Director（以下简称 DSH W&D）。三组都启用 full grading、loopback，评分模型均标记为 `qwen3.6-plus`。")
    add("")
    add(table(
        ["指标（212 条最终记录均值）", "OH Original", "OH W&D", "DSH W&D"],
        [
            ["final outcome", num(metric(analysis, "oh_original", "outcome_score")), num(metric(analysis, "oh_wd", "outcome_score")), num(metric(analysis, "dsh_wd", "outcome_score"))],
            ["process", num(metric(analysis, "oh_original", "process_score")), num(metric(analysis, "oh_wd", "process_score")), num(metric(analysis, "dsh_wd", "process_score"))],
            ["combined", num(metric(analysis, "oh_original", "combined_score")), num(metric(analysis, "oh_wd", "combined_score")), num(metric(analysis, "dsh_wd", "combined_score"))],
            ["平均耗时（秒）", num(metric(analysis, "oh_original", "elapsed_sec"), 1), num(metric(analysis, "oh_wd", "elapsed_sec"), 1), num(metric(analysis, "dsh_wd", "elapsed_sec"), 1)],
            ["去重后 pipeline/proxy requests 总数", integer(exps["oh_original"]["pipeline_requests_total"]), integer(exps["oh_wd"]["pipeline_requests_total"]), integer(exps["dsh_wd"]["pipeline_requests_total"])],
            ["Actor requests（Original/DSH 精确；OH W&D 为残差估计）", integer(exps["oh_original"]["actor_usage_totals"]["request_count"]), integer(exps["oh_wd"]["actor_usage_totals"]["request_count"]), integer(exps["dsh_wd"]["actor_usage_totals"]["request_count"])],
            ["Writer model requests", integer(exps["oh_original"]["writer_usage_totals"]["request_count"]), integer(exps["oh_wd"]["writer_usage_totals"]["request_count"]), integer(exps["dsh_wd"]["writer_usage_totals"]["request_count"])],
            ["adapter tool calls 总数", integer(exps["oh_original"]["adapter_tool_calls_total"]), integer(exps["oh_wd"]["adapter_tool_calls_total"]), integer(exps["dsh_wd"]["adapter_tool_calls_total"])],
            ["proxy-observed tool calls 总数", integer(exps["oh_original"]["proxy_tool_calls_total"]), integer(exps["oh_wd"]["proxy_tool_calls_total"]), integer(exps["dsh_wd"]["proxy_tool_calls_total"])],
            ["去重后 pipeline non-cache tokens 总数", integer(exps["oh_original"]["pipeline_noncache_tokens_total"]), integer(exps["oh_wd"]["pipeline_noncache_tokens_total"]), integer(exps["dsh_wd"]["pipeline_noncache_tokens_total"])],
            ["Writer non-cache tokens 总数", integer(exps["oh_original"]["writer_usage_totals"]["noncache_tokens"]), integer(exps["oh_wd"]["writer_usage_totals"]["noncache_tokens"]), integer(exps["dsh_wd"]["writer_usage_totals"]["noncache_tokens"])],
        ],
    ))
    add("")
    add("主要观察（仅描述本数据，不作超出实验设计的因果推断）：")
    add("")
    add(f"- OH W&D 相对 Original 的 final outcome 变化为 {signed(pairwise['oh_wd_minus_oh_original']['outcome_score']['mean_delta'])}（{signed_pct(pairwise['oh_wd_minus_oh_original']['outcome_score']['relative_delta'])}），其 paired 95% bootstrap CI 为 `{ci_text(pairwise['oh_wd_minus_oh_original']['outcome_score']['paired_delta_ci_95_task_bootstrap'])}`；combined 下降 {num(abs(pairwise['oh_wd_minus_oh_original']['combined_score']['mean_delta']))}（{pct(abs(pairwise['oh_wd_minus_oh_original']['combined_score']['relative_delta']))}），CI 不跨 0。")
    add(f"- DSH W&D 相对 Original 的 final outcome 为 {signed(pairwise['dsh_wd_minus_oh_original']['outcome_score']['mean_delta'])}，combined 为 {signed(pairwise['dsh_wd_minus_oh_original']['combined_score']['mean_delta'])}；process 下降 {signed(pairwise['dsh_wd_minus_oh_original']['process_score']['mean_delta'])}。因此 DSH W&D 在本数据上尚未超过 Original 的总体质量。")
    add(f"- DSH W&D 相对 OH W&D 的 process 提升 {signed(pairwise['dsh_wd_minus_oh_wd']['process_score']['mean_delta'])}，CI `{ci_text(pairwise['dsh_wd_minus_oh_wd']['process_score']['paired_delta_ci_95_task_bootstrap'])}`；combined 提升 {signed(pairwise['dsh_wd_minus_oh_wd']['combined_score']['mean_delta'])}，但 CI `{ci_text(pairwise['dsh_wd_minus_oh_wd']['combined_score']['paired_delta_ci_95_task_bootstrap'])}` 跨 0，不能据此宣称稳定的整体 combined 提升。")
    add(f"- DSH W&D 的 adapter tool error 从 OH W&D 的 {integer(exps['oh_wd']['metrics']['tool_errors']['sum'])} 降到 {integer(exps['dsh_wd']['metrics']['tool_errors']['sum'])}，错误率从 {pct(exps['oh_wd']['tool_error_rate'])} 降到 {pct(exps['dsh_wd']['tool_error_rate'])}；同时 adapter 调用均值由 {num(metric(analysis, 'oh_wd', 'tool_calls'))} 增至 {num(metric(analysis, 'dsh_wd', 'tool_calls'))}。这里的错误率分母是 adapter tool-call 总数，不是 request 数。")
    add(f"- W&D 的运行代价显著高于 Original：按去重后的 proxy/session 总量，OH W&D 的 pipeline non-cache tokens 为 {integer(exps['oh_wd']['pipeline_noncache_tokens_total'])}（{num(exps['oh_wd']['pipeline_noncache_tokens_total'] / exps['oh_original']['pipeline_noncache_tokens_total'], 2)}× Original），DSH W&D 为 {integer(exps['dsh_wd']['pipeline_noncache_tokens_total'])}（{num(exps['dsh_wd']['pipeline_noncache_tokens_total'] / exps['oh_original']['pipeline_noncache_tokens_total'], 2)}×）。")
    add("- 三组 security score 均为 1.000（212/212 记录），不能据此区分安全性优劣；它只说明本次 security gate 没有扣分。Process 子项（tool-use appropriate、consistency、robustness）与 security 是不同维度，不能用 security=1 推断整体过程可靠。")
    add("")
    add("## 2. 实验定义与可比性")
    add("")
    add(table(
        ["实验", "结果目录", "dataset / mode", "Actor/backend", "Writer / Director"],
        [
            [EXPERIMENT_SHORT[e], source_path(e), f"{exps[e]['config'].get('dataset_id')} / {exps[e]['config'].get('openharness_mode', '—')}", exps[e]['config'].get('actor_backend', 'OpenHarness'), "未启用" if e == "oh_original" else "Writer + Director" ]
            for e in EXPERIMENT_ORDER
        ],
    ))
    add("")
    add("### 固定配置与差异")
    add("")
    config_rows = []
    for e in EXPERIMENT_ORDER:
        c = exps[e]["config"]
        grading = c.get("grading") or {}
        config_rows.append([
            EXPERIMENT_SHORT[e],
            (c.get("adapter_config") or {}).get("adapter", "—"),
            c.get("model", "—"),
            f"temperature={c.get('temperature', '—')}; seed={c.get('seed', '—')}; max_turns={c.get('max_turns', '—')}",
            c.get("api_timeout_sec", "—"),
            (c.get("adapter_config") or {}).get("timeout_sec", "—"),
            (c.get("app_local_config") or {}).get("default_timeout_sec", "—"),
            c.get("public_url_mode", "—"),
            grading.get("mode", "—"),
            grading.get("rubric_model", "—"),
            c.get("writer_model", "—"),
            c.get("writer_max_tokens", "—"),
            c.get("director_harness_enabled", False),
            grading.get("rubric_api_key_source", "—"),
        ])
    add(table(
        ["实验", "adapter", "模型", "采样/轮数", "API timeout(s)", "adapter timeout(s)", "runner default timeout(s)", "URL mode", "grading", "rubric model", "Writer model", "Writer max tokens", "Director", "rubric key source"],
        config_rows,
    ))
    add("")
    add("配置字段若显示 `—`，表示该字段未写入该实验的 run-config，而不是运行时取值为 0；例如 DSH adapter 的具体 Actor max-turns/temperature/seed 没有在其 `run-config.json` 中声明，需以 `harness.local.json`、adapter 实现及结果埋点为准。")
    add("")
    add("三组任务 ID 均为 001–106；writer-full 两组使用的任务 manifest SHA256 为 `896dfdb552ddc18ccbf2c1e92d9d75b40732a5200d9e2359d74de79ed6e167ca`，内容一致。Original 的历史 manifest 路径已经不存在，但其 `run-config.json`、结果文件及任务列表仍可读取。")
    add("")
    add("源码锁与复现信息（完整对象保存在 `harnessbench_comparison_metrics.json`）：")
    add("")
    source_rows = []
    for e in EXPERIMENT_ORDER:
        c = exps[e]["config"]
        hgit = c.get("harnessbench_git") or {}
        ogit = c.get("openharness_git") or {}
        source_rows.append([
            EXPERIMENT_SHORT[e],
            hgit.get("commit", "—"),
            (c.get("harnessbench_source_sha256") or {}).get("src/harnessbench/usage_proxy.py", "—"),
            ogit.get("commit", "—"),
            (c.get("deepseek_harness_project_git") or {}).get("commit", "—"),
            c.get("source_lock_enforced", "—"),
        ])
    add(table(["实验", "HarnessBench commit", "run-config usage_proxy SHA256", "OpenHarness commit", "DSH project commit", "source lock"], source_rows))
    add("")
    dsh_state = exps["dsh_wd"]["run_state"]
    if dsh_state.get("source_lock_drift_vs_final_config"):
        add("**重要可比性限制：** DSH W&D 的 `run-state.json` 记录了 1 次 resume。resume 历史中的 HarnessBench/DeepSeek 源码 SHA 与最终 `run-config.json` 不同（涉及 `usage_proxy.py`、runner/adapter 及 DeepSeek runtime、writer 文件等）。因此 DSH 的 212 条最终记录跨越至少两个源码快照；下文将其作为实际运行结果报告，不把它描述成单一不可变构建。")
        add("")
    add("其他不可完全消除的差异包括：三组 grading endpoint/key source 不同；Original 有 cache-read tokens，而两组 W&D 没有；OpenHarness 与 DeepSeek 的工具命名和 trace 埋点不同。")
    add("")
    add("## 3. 结果覆盖率、状态与运行可靠性")
    add("")
    coverage_rows = []
    for e in EXPERIMENT_ORDER:
        a = exps[e]
        coverage_rows.append([
            EXPERIMENT_SHORT[e],
            f"{integer(a['result_rows'])}/{integer(analysis['expected_result_rows_per_experiment'])}",
            status_text(a.get("state_counts")),
            status_text(a.get("terminal_statuses")),
            status_text(a.get("adapter_statuses")),
            f"{integer(a['usage_available_count'])}/{integer(a['result_rows'])}",
            integer(a["result_rows_with_tool_errors"]),
            integer(a["unique_tasks_with_tool_errors"]),
            integer(a["total_tool_error_events"]),
            integer(a["proxy_trace_error_count"]),
            integer(a["api_timeout_payload_count"]),
            integer(a["api_timeout_event_count"]),
        ])
    add(table(
        ["实验", "结果覆盖", "summary state_counts", "每结果终态", "adapter status occurrences", "usage available", "有 tool error 的结果行", "涉及唯一任务", "tool error 总数", "proxy trace error", "timeout payloads", "timeout events"],
        coverage_rows,
    ))
    add("")
    add("说明：`summary state_counts` 是 runner 的结果状态；`adapter status occurrences` 按每个 adapter payload 计数（多轮任务可能一条最终结果含多个 payload）；`每结果终态` 只取该结果最后一个 payload。因此这些列不应相加后当作同一个分母。timeout 列是 adapter payload/error event 中匹配 timeout 文本的审计计数。OH W&D 的 repeat-02 `086-sql-migration-preflight-rollback` 终态为 `agent_task_failed`，日志明确记录 `API error: Request timed out.`，但仍有结构化 oracle/score 结果，故保留在 212 条最终结果中并单列。")
    add("")
    add("### API 请求日志与会话恢复")
    add("")
    add("请求日志是对实际 proxy 记录的补充审计字段；Original 的历史 `requests.jsonl` 已被清理，因此该组只能依赖结果 JSON 的 `usage_summary`，不能把日志缺失解释成没有请求。W&D 的 `usage_summary`/proxy 行数是 Actor+Writer 的会话总量；只有 DSH 日志按 framework 直接标出 Actor/Writer，OH W&D 的 Actor 数是用 Writer 报告从会话总量中扣除得到的估计。")
    add("")
    api_rows = []
    for e in EXPERIMENT_ORDER:
        a = exps[e]
        api_rows.append([
            EXPERIMENT_SHORT[e],
            f"{integer(a['request_log_available_count'])}/{integer(a['result_rows'])}",
            integer(a["request_log_line_total"]),
            status_text(a.get("request_status_counts")),
            status_text(a.get("request_framework_counts")),
            status_text(a.get("request_provider_counts")),
            status_text(a.get("request_model_counts")),
            integer(a["upstream_attempts_total"]),
            integer(a["upstream_attempts_max"]),
            integer(a["session_restore_event_count"]),
            integer(a["session_restored_result_count"]),
        ])
    add(table(["实验", "request log 可用结果", "log lines", "HTTP status", "framework", "provider", "response model", "upstream attempts", "单结果最大 attempts", "session restored payloads", "发生恢复的结果"], api_rows))
    add("")
    add("三组最终结果均报告 `usage_available=212/212`，每组有 24 个 adapter payload 标记 `session_restored=true`（对应 16 条最终结果行）；这属于 runner 的跨 round/session 行为，不应与 API 失败混同。OH W&D/DSH W&D 可读取的 proxy request log 均为 HTTP 200；DSH 的历史重试中另有 502，详见第 8 节。")
    add("")
    add(f"Writer 请求审计：OH W&D 的 236 个 Writer adapter payload 实际展开为 {integer(exps['oh_wd']['writer_usage_totals']['request_count'])} 次模型请求（由 Writer report 的 `model_request_count` 汇总），DSH W&D 为 {integer(exps['dsh_wd']['writer_usage_totals']['request_count'])} 次；因此不能把 adapter payload 数 236 当作 Writer API 请求数。")
    add("")
    add("### 每轮结果与稳定性")
    add("")
    repeat_rows = []
    for e in EXPERIMENT_ORDER:
        for repeat, stats in exps[e]["per_repeat"].items():
            repeat_rows.append([
                EXPERIMENT_SHORT[e], repeat,
                num(stats["outcome_score"]["mean"]), num(stats["process_score"]["mean"]), num(stats["combined_score"]["mean"]),
                num(stats["elapsed_sec"]["mean"], 1), num(stats["request_count"]["mean"], 2), num(stats["actor_request_count"]["mean"], 2), integer(stats["tool_calls"]["sum"]), integer(stats["tool_errors"]["sum"]),
            ])
    add(table(["实验", "repeat", "outcome", "process", "combined", "平均耗时(s)", "平均 reported/proxy request", "平均 Actor request", "tool calls", "tool errors"], repeat_rows))
    add("")
    stability_rows = []
    for e in EXPERIMENT_ORDER:
        s = exps[e]["repeat_stability"]
        stability_rows.append([
            EXPERIMENT_SHORT[e],
            signed(s["outcome_score"]["repeat_2_minus_1_mean"]), num(s["outcome_score"]["mean_abs_delta"]), num(s["outcome_score"]["pearson_r"]), integer(s["outcome_score"]["abs_delta_ge_0_1_count"]),
            signed(s["combined_score"]["repeat_2_minus_1_mean"]), num(s["combined_score"]["mean_abs_delta"]), num(s["combined_score"]["pearson_r"]), integer(s["combined_score"]["abs_delta_ge_0_1_count"]),
        ])
    add(table(["实验", "outcome R2−R1 均值", "outcome |Δ|均值", "outcome Pearson r", "outcome |Δ|≥0.1", "combined R2−R1 均值", "combined |Δ|均值", "combined Pearson r", "combined |Δ|≥0.1"], stability_rows))
    add("")
    add("## 4. 评分指标")
    add("")
    add("所有评分位于 0–1。`oracle_outcome_score` 是任务 oracle 的确定性结果分；`outcome_score` 是最终 outcome（对有 quality LLM 权重的图像任务会融合 quality）；`combined_score` 按结果中的公式为 outcome_effective × process × security。不要把 `baseline-summary.json` 的 `outcome_mean` 与这里的 final outcome 直接混称：前者是 summary 中的 oracle/outcome 字段口径，图像任务融合后会产生差异。")
    add("")
    score_rows = []
    score_fields = [
        ("oracle_outcome_score", "oracle outcome"),
        ("oracle_quality", "oracle quality（仅加权任务）"),
        ("outcome_score", "final outcome"),
        ("tool_use_appropriate", "tool_use_appropriate"),
        ("consistency", "consistency"),
        ("robustness", "robustness"),
        ("process_score", "process"),
        ("security_score", "security"),
        ("combined_score", "combined"),
    ]
    for field, label in score_fields:
        row = [label]
        for e in EXPERIMENT_ORDER:
            s = exps[e]["metrics"][field]
            row.extend([num(s.get("mean")), num(s.get("median")), num(s.get("std")), num(s.get("p90"))])
        score_rows.append(row)
    headers = ["指标"] + [f"{EXPERIMENT_SHORT[e]} mean/median/std/P90" for e in EXPERIMENT_ORDER]
    # Four subcolumns are clearer when expanded as separate columns.
    headers = ["指标"] + [item for e in EXPERIMENT_ORDER for item in (f"{EXPERIMENT_SHORT[e]} mean", f"{EXPERIMENT_SHORT[e]} median", f"{EXPERIMENT_SHORT[e]} std", f"{EXPERIMENT_SHORT[e]} P90")]
    add(table(headers, score_rows))
    add("")
    add("### 分数阈值计数（212 条记录）")
    add("")
    threshold_rows = []
    for field, label in (("outcome_score", "outcome"), ("process_score", "process"), ("combined_score", "combined")):
        for e in EXPERIMENT_ORDER:
            t = exps[e]["thresholds"][field]
            threshold_rows.append([EXPERIMENT_SHORT[e], label, t["eq_1"], t["ge_0_9"], t["ge_0_8"], t["lt_0_5"], t["eq_0"]])
    add(table(["实验", "指标", "=1", "≥0.9", "≥0.8", "<0.5", "=0"], threshold_rows))
    add("")
    add("各任务两轮均值的 score mean 95% bootstrap CI（按 106 个任务重采样）如下；这是描述性不确定性区间，不是跨不同模型/构建的因果显著性证明：")
    add("")
    ci_rows = []
    for e in EXPERIMENT_ORDER:
        ci = exps[e]["score_mean_ci_95_task_bootstrap"]
        ci_rows.append([EXPERIMENT_SHORT[e], ci_text(ci["outcome_score"]), ci_text(ci["process_score"]), ci_text(ci["combined_score"])])
    add(table(["实验", "outcome CI", "process CI", "combined CI"], ci_rows))
    add("")
    add("## 5. 工具调用、错误与行为轨迹")
    add("")
    tool_rows = []
    for e in EXPERIMENT_ORDER:
        a = exps[e]
        tool_rows.append([
            EXPERIMENT_SHORT[e],
            integer(a["adapter_tool_calls_total"]), num(metric(analysis, e, "tool_calls"), 2), num(metric(analysis, e, "tool_calls", "median"), 2), num(metric(analysis, e, "tool_calls", "p90"), 2), num(metric(analysis, e, "tool_calls", "p95"), 2), integer(metric(analysis, e, "tool_calls", "max")),
            integer(a["proxy_tool_calls_total"]), num(metric(analysis, e, "proxy_tool_calls"), 2),
            integer(a["total_tool_error_events"]), pct(a["tool_error_rate"]), integer(a["result_rows_with_tool_errors"]), integer(a["unique_tasks_with_tool_errors"]), integer(a["tool_count_mismatch_count"]),
        ])
    add(table(["实验", "adapter calls 总数", "adapter 均值", "中位数", "P90", "P95", "最大", "proxy calls 总数", "proxy 均值", "tool errors", "error rate", "错误结果行", "错误唯一任务", "adapter/proxy 不一致行"], tool_rows))
    add("")
    add("`adapter tool calls` 是 adapter payload 上报值；`proxy-observed tool calls` 是 usage proxy 的 response trace 逐调用计数。OH Original 与 OH W&D 两者完全一致；DSH W&D 为 adapter 3,117、proxy 3,425，相差 308（50/212 条结果行存在差异）。这是埋点口径差异，不能直接解释为 308 次执行错误；两者均保留供审计。")
    add("")
    add("### 工具名称分布")
    add("")
    add("原始名称（各 backend 命名不同）前 8 名：")
    add("")
    add(table(["实验", "top tools（名称:次数/占比）"], [[EXPERIMENT_SHORT[e], top_tool_text(exps[e]["top_tools"])] for e in EXPERIMENT_ORDER]))
    add("")
    add("为便于跨 backend 阅读，`read_file→read`、`write_file→write`、`edit_file→edit` 做了展示层归一化；原始名称仍在 JSON/JSONL：")
    add("")
    add(table(["实验", "归一化 top tools"], [[EXPERIMENT_SHORT[e], top_tool_text(exps[e]["top_tools_normalized"])] for e in EXPERIMENT_ORDER]))
    add("")
    add("### Writer / Director / Actor 事件")
    add("")
    wd_rows = []
    for e in EXPERIMENT_ORDER:
        a = exps[e]
        wg = a.get("writer_gate") or {}
        dg = a.get("director_gate") or {}
        wd_rows.append([
            EXPERIMENT_SHORT[e], integer(metric(analysis, e, "writer_total_tokens", "sum")), num(metric(analysis, e, "writer_total_tokens"), 1), integer(metric(analysis, e, "writer_request_count", "sum")), integer(metric(analysis, e, "writer_report_count", "sum")), integer(metric(analysis, e, "writer_event_count", "sum")), integer(metric(analysis, e, "writer_tool_step_count", "sum")),
            integer(metric(analysis, e, "director_event_count", "sum")), integer(metric(analysis, e, "director_checked_tool_use_count", "sum")), status_text(a.get("director_statuses")), integer(metric(analysis, e, "actor_event_count", "sum")),
            f"{wg.get('mandatory_passed', '—')}/{wg.get('expected_rounds', '—')}", f"{dg.get('all_tool_calls_checked', '—')}/{dg.get('expected_rounds', '—')}",
        ])
    add(table(["实验", "Writer tokens 总", "Writer tokens/结果均值", "Writer model requests", "Writer adapter payloads", "Writer events", "Writer tool steps", "Director events", "Director checked tools", "Director statuses", "Actor events", "Writer gate", "Director gate"], wd_rows))
    add("")
    add("OH W&D 的 Director event statuses 为 passed=855、failed=86、skipped=1,699；其 event_count 包含 tool_result/重复检查语义。DSH W&D 的 Director hook 只记录 tool_check，3,117 次均为 passed，并额外报告 ordered_tool_use_count=3,117。两组 smoke gate 都是 236/236 通过，说明 gate 覆盖通过，不等于每个 event status 都是 passed。")
    add("")
    add("Writer event 类型：OH W&D 每个 adapter round 均包含 `action_allowed`、`global_plan_created`、`step_action_proposed`、`step_check_completed`、`postcondition_checked`；DSH W&D 为 `actor_plan_generated`、`writer_judgment_completed`、`actor_tool_call_observed`、`actor_tool_result_observed`。这反映实现协议差异。DSH actor `finish_reason=completed` 为 236/236 adapter payload。Writer adapter payload 是阶段/round 记录，不等于模型请求数。")
    add("")
    add("## 6. 请求数、token 与耗时资源")
    add("")
    add("口径说明：Original 的 `usage_summary` 是 Actor-only；两组 W&D 的 `usage_summary`/proxy 日志是同一会话中的 Actor+Writer 去重总量。Writer usage 是该总量的子集，不能再次相加。`pipeline_*` 现在定义为去重后的会话总量；`actor_*` 为 Actor 分解（DSH 由 framework 日志精确拆分，OH W&D 由总量减 Writer 子集得到估计）。所有 total/mean 等均按 212 条最终结果记录；耗时总和不是并发运行的 wall-clock。")
    add("")
    resource_fields = [
        ("elapsed_sec", "elapsed(s)", 1),
        ("request_count", "reported/proxy requests（W&D 含 Writer）", 2),
        ("actor_request_count", "Actor requests（OH W&D 为估计）", 2),
        ("pipeline_request_count", "pipeline requests（去重）", 2),
        ("input_tokens", "reported input tokens（W&D 含 Writer）", 0),
        ("actor_input_tokens", "Actor input tokens", 0),
        ("output_tokens", "reported output tokens（W&D 含 Writer）", 0),
        ("actor_output_tokens", "Actor output tokens", 0),
        ("cache_read_tokens", "cache-read tokens", 0),
        ("noncache_tokens", "reported non-cache tokens", 0),
        ("actor_noncache_tokens", "Actor non-cache tokens", 0),
        ("total_tokens", "reported total tokens", 0),
        ("actor_total_tokens", "Actor total tokens", 0),
        ("writer_total_tokens", "Writer tokens", 0),
        ("pipeline_noncache_tokens", "pipeline non-cache tokens（去重）", 0),
        ("pipeline_total_tokens", "pipeline total tokens（去重）", 0),
        ("tool_calls", "adapter tool calls", 2),
        ("proxy_tool_calls", "proxy tool calls", 2),
        ("tool_errors", "tool errors", 2),
    ]
    resource_rows = []
    for field, label, digits in resource_fields:
        row = [label]
        for e in EXPERIMENT_ORDER:
            s = exps[e]["metrics"][field]
            row.extend([num(s.get("sum"), digits), num(s.get("mean"), digits), num(s.get("median"), digits), num(s.get("p90"), digits), num(s.get("p95"), digits), num(s.get("max"), digits)])
        resource_rows.append(row)
    resource_headers = ["资源指标"] + [item for e in EXPERIMENT_ORDER for item in (f"{EXPERIMENT_SHORT[e]} sum", f"{EXPERIMENT_SHORT[e]} mean", f"{EXPERIMENT_SHORT[e]} median", f"{EXPERIMENT_SHORT[e]} P90", f"{EXPERIMENT_SHORT[e]} P95", f"{EXPERIMENT_SHORT[e]} max")]
    add(table(resource_headers, resource_rows))
    add("")
    add("缓存口径尤其重要：Original 有 `cache_read_tokens=8,341,248`，而 OH W&D、DSH W&D 均为 0。因此三组的 `reported total_tokens` 不能直接当成相同计费口径；报告同时列 reported/session、Actor 分解、Writer 子集与去重 pipeline non-cache。")
    add("")
    pipeline_rows = []
    for e in EXPERIMENT_ORDER:
        a = exps[e]
        pipeline_rows.append([
            EXPERIMENT_SHORT[e], integer(a["pipeline_requests_total"]), integer(a["actor_usage_totals"]["request_count"]), integer(a["writer_usage_totals"]["request_count"]), integer(a["actor_usage_totals"]["noncache_tokens"]), integer(a["writer_usage_totals"]["noncache_tokens"]), integer(a["pipeline_noncache_tokens_total"]), integer(a["pipeline_total_tokens_total"]), pct(a["writer_token_share_of_pipeline_noncache"]), num(a["requests_per_tool_call"], 2), num(a["requests_per_proxy_tool_call"], 2), ", ".join(f"{k}={v}" for k, v in a.get("actor_accounting_methods", {}).items()),
        ])
    add(table(["实验", "pipeline/proxy requests 总（去重）", "Actor requests", "Writer model requests", "Actor non-cache", "Writer non-cache", "pipeline non-cache 总（去重）", "pipeline total 总（去重）", "Writer / pipeline non-cache", "reported requests / adapter call", "reported requests / proxy call", "Actor accounting"], pipeline_rows))
    add("")
    add("### 按任务类别")
    add("")
    add("下表为每类任务的两轮均值；完整 class CSV 见附件。")
    class_outcome = {r["group"]: r for r in analysis["by_class"]["outcome_score"]}
    class_combined = {r["group"]: r for r in analysis["by_class"]["combined_score"]}
    class_tool = {r["group"]: r for r in analysis["by_class"]["tool_calls"]}
    class_pipe = {r["group"]: r for r in analysis["by_class"]["pipeline_noncache_tokens"]}
    class_elapsed = {r["group"]: r for r in analysis["by_class"]["elapsed_sec"]}
    class_rows = []
    for group in sorted(class_outcome):
        o, c, t, p, el = class_outcome[group], class_combined[group], class_tool[group], class_pipe[group], class_elapsed[group]
        class_rows.append([group, o["task_count"], num(o["oh_original"]), num(o["oh_wd"]), num(o["dsh_wd"]), signed(o["oh_wd_minus_original"]), signed(o["dsh_wd_minus_oh_wd"]), num(c["oh_original"]), num(c["oh_wd"]), num(c["dsh_wd"]), num(t["oh_original"], 1), num(t["oh_wd"], 1), num(t["dsh_wd"], 1), integer(p["dsh_wd"]), num(el["dsh_wd"], 1)])
    add(table(["类别", "题数", "outcome OH-O", "outcome OH-WD", "outcome DSH", "OH-WD−O", "DSH−OH-WD", "combined O", "combined OH-WD", "combined DSH", "calls O", "calls OH-WD", "calls DSH", "DSH pipeline tokens", "DSH elapsed(s)"], class_rows))
    add("")
    add("类别观察：DSH W&D 在 Long-running Autonomy、Office、SRE 的 outcome 高于 OH W&D；在 Software Engineering、Data/BI、Vertical 等类别存在明显回落或 mixed pattern。由于每类样本量仅 7–22 个，类别差异应视为诊断线索而非稳定排名。")
    add("")
    add("### 按难度")
    add("")
    difficulty_rows = []
    d_out = {r["group"]: r for r in analysis["by_difficulty"]["outcome_score"]}
    d_proc = {r["group"]: r for r in analysis["by_difficulty"]["process_score"]}
    d_comb = {r["group"]: r for r in analysis["by_difficulty"]["combined_score"]}
    for group in sorted(d_out):
        o, p, c = d_out[group], d_proc[group], d_comb[group]
        difficulty_rows.append([group, o["task_count"], num(o["oh_original"]), num(o["oh_wd"]), num(o["dsh_wd"]), signed(o["dsh_wd_minus_oh_wd"]), num(p["oh_original"]), num(p["oh_wd"]), num(p["dsh_wd"]), num(c["oh_original"]), num(c["oh_wd"]), num(c["dsh_wd"])])
    add(table(["difficulty", "题数", "outcome O", "outcome OH-WD", "outcome DSH", "DSH−OH-WD", "process O", "process OH-WD", "process DSH", "combined O", "combined OH-WD", "combined DSH"], difficulty_rows))
    add("")
    add("`unlabeled` 是 task.yaml 中没有 difficulty 字段的 24 个任务，不应误读为 easy/hard。")
    add("")
    add("## 7. 成对任务比较与单任务差异")
    add("")
    add("成对比较使用同一 task ID 的两轮均值；`win/tie/loss` 是 right 相对 left 的 106 题计数，material 比较把 score 差异绝对值 ≤0.01 视为 tie。")
    add("")
    pair_rows = []
    pair_labels = {
        "oh_wd_minus_oh_original": "OH W&D − OH Original",
        "dsh_wd_minus_oh_original": "DSH W&D − OH Original",
        "dsh_wd_minus_oh_wd": "DSH W&D − OH W&D",
    }
    for key, label in pair_labels.items():
        v = pairwise[key]
        for field, field_label in (("outcome_score", "outcome"), ("process_score", "process"), ("combined_score", "combined")):
            d = v[field]
            w = d["win_tie_loss_any"]
            wm = d["win_tie_loss_material"]
            pair_rows.append([label, field_label, signed(d["mean_delta"]), signed_pct(d["relative_delta"]), ci_text(d["paired_delta_ci_95_task_bootstrap"]), f"{w['right_wins']}/{w['ties']}/{w['right_losses']}", f"{wm['right_wins']}/{wm['ties']}/{wm['right_losses']}"])
    add(table(["比较", "指标", "均值 Δ", "相对变化", "paired 95% CI", "任意 Δ W/T/L", "material Δ W/T/L"], pair_rows))
    add("")
    task_map = {r["task_id"]: r for r in analysis["task_table"]}
    def change_rows(change_key: str, count: int = 10) -> list[list[Any]]:
        changes = analysis["top_changes"][change_key]
        output: list[list[Any]] = []
        for side in ("top_10", "bottom_10"):
            for item in changes[side][:count]:
                task = task_map[item["task_id"]]
                output.append(["正向" if side == "top_10" else "负向", item["task_id"], task["title"], task["class"], task["difficulty"], signed(item["delta"])])
        return output
    add("### DSH W&D 相对 OH W&D 的 outcome 变化（前/后 10）")
    add("")
    add(table(["方向", "task", "title", "class", "difficulty", "DSH−OH-WD outcome"], change_rows("dsh_wd_minus_oh_wd_outcome")))
    add("")
    add("### DSH W&D 相对 OH W&D 的 combined 变化（前/后 10）")
    add("")
    add(table(["方向", "task", "title", "class", "difficulty", "DSH−OH-WD combined"], change_rows("dsh_wd_minus_oh_wd_combined")))
    add("")
    add("完整 106 题两轮均值及资源列在 `harnessbench_task_comparison.csv`；完整 636 条 task×repeat 原始派生行（含 error_events、tool_names、Writer/Director 字段）在 `harnessbench_result_rows.jsonl`。")
    add("")
    add("## 8. 重试、归档与异常")
    add("")
    retry = analysis["deepseek_retry_history"]
    add(f"DSH W&D 的 retry-history 共归档 {integer(retry['archived_attempt_count'])} 次尝试，涉及 {integer(retry['unique_task_repeat_pairs'])} 个 task×repeat 对；这些归档尝试**不纳入**上文 212 条最终评分均值，但代表实际运行成本和可靠性。")
    add("")
    retry_rows = []
    for reason, count in sorted(retry["reason_counts"].items()):
        retry_rows.append([reason, count, pct(count / retry["archived_attempt_count"] if retry["archived_attempt_count"] else 0)])
    add(table(["归档原因", "次数", "占比"], retry_rows))
    add("")
    rm = retry["metrics"]
    add(table(["归档资源", "总量", "均值/次", "中位数", "P90", "最大"], [
        [label, integer(rm[field]["sum"]), num(rm[field]["mean"], digits), num(rm[field]["median"], digits), num(rm[field]["p90"], digits), num(rm[field]["max"], digits)]
        for field, label, digits in (("elapsed_sec", "elapsed(s)", 1), ("request_count", "reported/proxy requests（含 Writer）", 2), ("actor_request_count", "Actor requests", 2), ("noncache_tokens", "reported non-cache tokens（含 Writer）", 0), ("actor_noncache_tokens", "Actor non-cache tokens", 0), ("tool_calls", "adapter tool calls", 1), ("writer_request_count", "Writer model requests", 1), ("writer_total_tokens", "Writer tokens", 0))
    ]))
    add("")
    add(f"归档尝试额外耗时合计约 {num(rm['elapsed_sec']['sum'], 1)} 秒、reported/proxy non-cache tokens {integer(rm['noncache_tokens']['sum'])}（其中 Writer {integer(rm['writer_total_tokens']['sum'])}）；原因分类为 session ID collision={retry['reason_counts'].get('session_id_collision', 0)}、DeepSeek process failed={retry['reason_counts'].get('deepseek_harness_process_failed', 0)}、upstream 502={retry['reason_counts'].get('upstream_502', 0)}。最终日志显示 repeat-02 的 `106-release-approval-gate-plan` 重试以 rc=0 完成，最终覆盖为 212/212。")
    add("")
    add("## 9. 方法、限制与客观结论")
    add("")
    add("统计方法：每组先读取 212 条最终 JSON；任务级比较先对两轮取均值，再在 106 个共同 task 上做 paired delta；CI 为固定随机种子、20,000 次任务级 bootstrap 的 percentile 95% 区间。均值/中位数/P90/P95/max 同时报告，避免单一均值掩盖长尾。")
    add("")
    add("限制：")
    add("")
    add("- 只有 2 个 repeat，稳定性估计有限；bootstrap CI 不是独立重复实验的替代品。")
    add("- 三组并非只改变一个变量：backend、Writer/Director 实现、grading endpoint/key source、源码 commit、缓存行为均有差异；因此不能把差异归因于某一个组件。")
    add("- DSH resume 期间源码锁发生漂移；初始 run-config 的 usage_proxy SHA 与 resume history 的 SHA 不同。")
    add("- Original 的 cache-read tokens 为非零，W&D 为零；成本比较应优先使用去重 pipeline non-cache 与分解后的 Actor/Writer 列，而不是直接比较 reported total。")
    add("- OH W&D 的 proxy framework 字段没有区分 Actor 与 Writer，Actor token/request 是从会话总量扣除 Writer report 的可审计估计；DSH 的 framework 拆分则可直接核验。")
    add("- DSH adapter/proxy 工具调用计数存在 308 次总量差异；工具名称也有 `read_file`/`read` 等协议差异。")
    add("- full grading 的图像任务（008、013）使用 quality LLM 融合 outcome；`oracle_outcome_score`、`outcome_score`、`combined_score` 必须分开解读。")
    add("")
    add("综合结论：在这次固定数据和配置下，OH Original 的 final outcome 与 combined 均值最高、process 几乎满分且资源开销最低。OH W&D 的结构化 Writer/Director gate 全部通过，但伴随显著 process/combined 回落和约两倍耗时；其去重 pipeline non-cache 为 Original 的约 2.62 倍。DSH W&D 相比 OH W&D 明显改善 process、工具错误率和 token 效率：去重 pipeline non-cache 低约 22.2%，而平均耗时仅增加约 4.4%；然而 final outcome 仍略低，combined 的整体提升 CI 跨 0。因而更准确的表述是：**DSH backend 在 W&D 管线中的执行可靠性/资源效率优于 OH W&D，质量指标呈 mixed、尚不足以证明整体优于 OH W&D，更没有超过 Original。**")
    add("")
    add("## 10. 数据附件与复现")
    add("")
    add("- `harnessbench_comparison_metrics.json`：完整分析对象（配置、run-state、评分/资源分布、pairwise CI、类别/难度、重试、task 表）。")
    add("- `harnessbench_task_comparison.csv`：106 题两轮均值的宽表，含所有主要评分、工具、请求、token、Writer/Director/Actor 和终态字段。")
    add("- `harnessbench_result_rows.jsonl`：636 条 task×repeat 派生行，含工具名、错误事件和埋点差异。")
    add("- `harnessbench_by_class_*.csv`、`harnessbench_by_difficulty_*.csv`：分组明细。")
    add("")
    add("复现命令：")
    add("")
    add("```bash")
    add("cd /home/patton/projects/dsharness_wd")
    add("python3 -m py_compile scripts/analyze_harnessbench_comparison.py scripts/render_harnessbench_comparison.py")
    add("python3 scripts/render_harnessbench_comparison.py")
    add("python3 scripts/analyze_harnessbench_comparison.py --section all --compact > /tmp/harnessbench_analysis.json")
    add("```")
    add("")
    add("原始数据目录见第 2 节；报告生成脚本不会修改这些目录。")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    global DOCS_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DOCS_DIR)
    args = parser.parse_args()
    DOCS_DIR = args.output_dir.resolve()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    analysis = build_analysis()
    raw_rows = collect_raw_rows()
    # The JSON snapshot intentionally excludes the duplicated raw rows; they
    # are available losslessly in the JSONL attachment.
    (DOCS_DIR / "harnessbench_comparison_metrics.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    write_csv(DOCS_DIR / "harnessbench_task_comparison.csv", analysis["task_table"])
    write_raw_jsonl(DOCS_DIR / "harnessbench_result_rows.jsonl", raw_rows)
    write_category_csvs(analysis)
    (DOCS_DIR / "harnessbench_openharness_deepseek_comparison.md").write_text(
        render_report(analysis, raw_rows), encoding="utf-8"
    )
    print(f"wrote artifacts to {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
