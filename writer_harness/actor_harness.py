"""封装演员 Harness 执行后端。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .models import ExecutionResult, InteractionMode, WriterHarnessReport


def extract_report_metadata_from_stdout(stdout: str) -> tuple[dict | None, str | None, str | None, str | None]:
    """从演员 Harness 的标准输出中提取结构化执行剧本。

    参数说明：
    - stdout: 演员 Harness 的原始标准输出文本，可能是纯文本，也可能夹带 JSON。

    返回值依次为：
    - script_report: 解析出的结构化剧本对象；
    - source: 剧本来源标记，描述结果来自何处；
    - origin: 剧本对象的产生主体；
    - transport: 剧本在链路中的传递路径。
    """

    text = stdout.strip()
    if not text:
        return None, None, None, None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None, None, None, None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, None, None, None
    if not isinstance(payload, dict):
        return None, None, None, None
    source = payload.get("script_report_source")
    origin = payload.get("script_report_object_origin")
    transport = payload.get("script_report_transport_path")
    if isinstance(payload.get("script_report"), dict):
        return payload["script_report"], source or "actor_harness_output", origin or "actor_harness", transport or "actor_harness_output"
    if {"task_profile", "difficulty_profile", "execution_plan"}.issubset(payload.keys()):
        return payload, source or "actor_harness_output", origin or "actor_harness", transport or "actor_harness_output"
    return None, None, None, None


def extract_tool_trace_from_dsh_events(events: list[Any]) -> dict[str, Any]:
    tool_events: list[dict[str, Any]] = []
    tool_sequence: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}

    for sequence_index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event_type == "tool/call":
            call_id = str(data.get("callId") or "")
            tool_name = str(data.get("name") or "unknown")
            raw_arguments = data.get("arguments")
            tool_input: Any = raw_arguments
            if isinstance(raw_arguments, str):
                try:
                    tool_input = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    pass
            call = {
                "sequence_index": sequence_index,
                "call_id": call_id or None,
                "tool_name": tool_name,
                "turn": data.get("turn"),
                "step": data.get("step"),
                "raw_arguments": raw_arguments,
                "tool_input": tool_input,
                "status": "running",
                "result": None,
                "error": None,
                "meta": None,
            }
            tool_calls.append(call)
            if call_id:
                calls_by_id[call_id] = call
            tool_events.append(
                {
                    "type": "tool_started",
                    "tool_name": tool_name,
                    "tool_use_id": call_id or None,
                    "tool_input": tool_input,
                    "output": None,
                    "is_error": None,
                }
            )
            tool_sequence.append({"name": tool_name, "type": "start", "call_id": call_id or None, "sequence_index": sequence_index})
        elif event_type == "tool/result":
            message = data.get("message") if isinstance(data.get("message"), dict) else {}
            source = message.get("source") if isinstance(message.get("source"), dict) else {}
            call_id = str(source.get("callId") or data.get("callId") or "")
            error = data.get("error") if isinstance(data.get("error"), dict) else None
            result = message or data.get("message")
            call = calls_by_id.get(call_id)
            tool_name = call["tool_name"] if call else str(data.get("name") or "unknown")
            if call:
                call["status"] = "failed" if error else "completed"
                call["result"] = result
                call["error"] = error
                call["meta"] = data.get("meta")
            tool_events.append(
                {
                    "type": "tool_completed",
                    "tool_name": tool_name,
                    "tool_use_id": call_id or None,
                    "tool_input": None,
                    "output": result,
                    "is_error": bool(error),
                    "error": error,
                    "meta": data.get("meta"),
                }
            )
            tool_sequence.append({"name": tool_name, "type": "complete", "call_id": call_id or None, "sequence_index": sequence_index})

    return {
        "schema_version": 1,
        "backend": "deepseek-harness",
        "trace_protocol": "dsh-session-events-v1",
        "tool_events": tool_events,
        "tool_sequence": tool_sequence,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "tool_error_count": sum(call["status"] == "failed" for call in tool_calls),
        "assistant_text": "",
        "used_ask_user_question": any(call["tool_name"] == "ask_user_question" for call in tool_calls),
        "director_events": [],
    }


def extract_dsh_assistant_text(events: list[Any]) -> str:
    for event in reversed(events):
        if not isinstance(event, dict) or event.get("type") != "assistant/message":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        message = data.get("message") if isinstance(data.get("message"), dict) else data
        content = message.get("content") if isinstance(message.get("content"), list) else []
        text_parts = [str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        if text_parts:
            return "".join(text_parts)
    return ""


class ActorHarnessExecutor:
    """演员 Harness 执行器抽象。

    支持真实 OpenHarness CLI 与 DeepSeek Harness Python SDK，保持
    orchestrator 与在线执行脚本不依赖具体执行后端。
    """

    def execute(self, prompt: str, mode: InteractionMode, script_report: WriterHarnessReport | None = None) -> ExecutionResult:
        """执行演员 Harness。

        参数说明：
        - prompt: 最终要交给演员 Harness 的文本输入。
        - mode: 当前交互模式，用于标记是直连、固定脚本还是编剧生成剧本。
        - script_report: 编剧 Harness 已产出的结构化执行剧本；若提供，
          可随结果一并透传，方便在线执行脚本或上层 orchestrator 继续消费。
        """

        raise NotImplementedError


class OpenHarnessCLIActorExecutor(ActorHarnessExecutor):
    """通过 `oh -p` 接入 OpenHarness，把 OpenHarness 作为真实演员 Harness 执行系统。"""

    actor_backend = "openharness"

    def __init__(
        self,
        oh_bin: str = "oh",
        dry_run: bool = True,
        output_format: str | None = None,
        openharness_src: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_format: str | None = None,
    ):
        """初始化基于 OpenHarness CLI 的演员执行器。

        参数说明：
        - oh_bin: OpenHarness CLI 可执行文件名或绝对路径。
        - dry_run: 是否只做 dry-run，不真正进入工具执行。
        - output_format: 演员 Harness 输出格式，如 text / json / stream-json。
        - openharness_src: OpenHarness 源码 src 路径；当本地未正式安装时，
          用于通过 PYTHONPATH 注入源码。
        - model/base_url/api_key/api_format: 透传给 OpenHarness 的模型配置。
        """

        self.oh_bin = oh_bin
        self.dry_run = dry_run
        self.output_format = output_format
        self.openharness_src = openharness_src
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_format = api_format

    def execute(self, prompt: str, mode: InteractionMode, script_report: WriterHarnessReport | None = None) -> ExecutionResult:
        """调用 OpenHarness CLI 执行真实演员 Harness 任务。"""

        if shutil.which(self.oh_bin) is None:
            return ExecutionResult(
                ok=False,
                mode=mode.value,
                final_prompt=prompt,
                stdout="",
                stderr=f"未找到 OpenHarness CLI：{self.oh_bin}。请先安装 OpenHarness 并确认 CLI 可用。",
                return_code=127,
                script_report=script_report,
                script_report_source=("actor_harness_passthrough_script_report" if script_report is not None else None),
                script_report_object_origin=("writer_harness" if script_report is not None else None),
                script_report_transport_path=("writer_harness_to_actor_harness_passthrough" if script_report is not None else None),
                actor_backend="openharness",
            )
        command = [self.oh_bin]
        # #region debug-point A:long-prompt-transport
        exec("try:\n import json as _j, urllib.request as _u, time as _t\n _p='.dbg/script-output-missing.env'; _c=open(_p, encoding='utf-8').read(); _url=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SERVER_URL=')), 'http://127.0.0.1:7777/event'); _sid=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SESSION_ID=')), 'script-output-missing'); _u.urlopen(_u.Request(_url, data=_j.dumps({'sessionId': _sid, 'runId': 'post-fix', 'hypothesisId': 'A', 'location': 'actor_harness.py:126', 'msg': '[DEBUG] actor prompt transport selected', 'data': {'promptLength': len(prompt), 'usesLiteralPromptArgument': True}, 'ts': int(_t.time() * 1000)}).encode(), headers={'Content-Type': 'application/json'}), timeout=1).read()\nexcept: pass")
        # #endregion
        if self.dry_run:
            command.append("--dry-run")
        if self.model:
            command.extend(["--model", self.model])
        if self.base_url:
            command.extend(["--base-url", self.base_url])
        if self.api_key:
            command.extend(["--api-key", self.api_key])
        if self.api_format:
            command.extend(["--api-format", self.api_format])
        command.extend(["-p", prompt])
        if self.output_format:
            command.extend(["--output-format", self.output_format])
        env = os.environ.copy()
        if self.openharness_src:
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = self.openharness_src if not existing_pythonpath else f"{self.openharness_src}{os.pathsep}{existing_pythonpath}"
        if env.get("DIRECTOR_HARNESS_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
            project_root = Path(__file__).resolve().parents[1]
            existing_pythonpath = env.get("PYTHONPATH")
            python_paths = [str(project_root)]
            if existing_pythonpath:
                python_paths.append(existing_pythonpath)
            env["PYTHONPATH"] = os.pathsep.join(python_paths)
        # #region debug-point B:openharness-invocation
        exec("try:\n import json as _j, urllib.request as _u, time as _t\n _p='.dbg/multiturn-permissions.env'; _c=open(_p, encoding='utf-8').read(); _url=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SERVER_URL=')), 'http://127.0.0.1:7777/event'); _sid=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SESSION_ID=')), 'multiturn-permissions'); _u.urlopen(_u.Request(_url, data=_j.dumps({'sessionId': _sid, 'runId': 'pre-fix', 'hypothesisId': 'B', 'location': 'actor_harness.py:150', 'msg': '[DEBUG] OpenHarness subprocess prepared', 'data': {'cwd': os.getcwd(), 'dryRun': self.dry_run, 'commandHead': command[:2], 'hasOpenHarnessSrc': bool(self.openharness_src)}, 'ts': int(_t.time() * 1000)}).encode(), headers={'Content-Type': 'application/json'}), timeout=1).read()\nexcept: pass")
        # #endregion
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        actor_report, report_source, report_origin, report_transport = extract_report_metadata_from_stdout(completed.stdout)
        return ExecutionResult(
            ok=completed.returncode == 0,
            mode=mode.value,
            final_prompt=prompt,
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
            script_report=script_report,
            script_report_source=report_source,
            script_report_object_origin=report_origin,
            script_report_transport_path=report_transport,
            final_script_report=actor_report,
            actor_backend="openharness",
        )


class DeepSeekHarnessActorExecutor(ActorHarnessExecutor):
    """通过 DeepSeek Harness Python SDK 执行演员任务。"""

    actor_backend = "deepseek-harness"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.project_root = Path(__file__).resolve().parents[1]
        self.provider = "actor-openai-compatible"
        self.runtime_cwd = str(self.project_root)
        self.session_root = str(self.project_root / ".dsh-sessions")
        self.cordis = str(self.project_root / "dsh_configs" / "openai_compatible.cordis.yml")

    def execute(self, prompt: str, mode: InteractionMode, script_report: WriterHarnessReport | None = None) -> ExecutionResult:
        try:
            harness_class = self._load_harness_class()
            kwargs = self._build_harness_kwargs()
            with harness_class(**kwargs) as harness:
                run = harness.run(prompt)
        except Exception as exc:
            return ExecutionResult(
                ok=False,
                mode=mode.value,
                final_prompt=prompt,
                stdout="",
                stderr=f"DeepSeek Harness 执行失败：{exc}",
                return_code=1,
                script_report=script_report,
                script_report_source=("actor_harness_passthrough_script_report" if script_report is not None else None),
                script_report_object_origin=("writer_harness" if script_report is not None else None),
                script_report_transport_path=("writer_harness_to_actor_harness_passthrough" if script_report is not None else None),
                actor_backend="deepseek-harness",
            )
        events = list(getattr(run, "events", []) or [])
        # #region debug-point A:dsh-run-result
        exec("try:\n import json as _j, urllib.request as _u, time as _t\n _p='.dbg/dsh-empty-output.env'; _c=open(_p, encoding='utf-8').read(); _url=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SERVER_URL=')), 'http://127.0.0.1:7777/event'); _sid=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SESSION_ID=')), 'dsh-empty-output'); _events=list(getattr(run, 'events', []) or []); _notifications=list(getattr(run, 'notifications', []) or []); _u.urlopen(_u.Request(_url, data=_j.dumps({'sessionId': _sid, 'runId': 'pre-fix', 'hypothesisId': 'A', 'location': 'actor_harness.py:dsh-run-result', 'msg': '[DEBUG] DSH run result received', 'data': {'hasFinalResponse': bool(getattr(run, 'final_response', None)), 'finalResponseLength': len(str(getattr(run, 'final_response', '') or '')), 'eventCount': len(_events), 'eventTypes': [e.get('type') for e in _events if isinstance(e, dict)], 'notificationCount': len(_notifications), 'notificationMethods': [str(getattr(n, 'method', '')) for n in _notifications], 'sessionIdPresent': bool(getattr(run, 'session_id', None)), 'finishReason': getattr(run, 'finish_reason', None)}, 'ts': int(_t.time() * 1000)}).encode(), headers={'Content-Type': 'application/json'}), timeout=1).read()\nexcept: pass")
        # #endregion
        # #region debug-point C:dsh-error-evidence
        exec("try:\n import json as _j, urllib.request as _u, time as _t\n _p='.dbg/dsh-empty-output.env'; _c=open(_p, encoding='utf-8').read(); _url=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SERVER_URL=')), 'http://127.0.0.1:7777/event'); _sid=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SESSION_ID=')), 'dsh-empty-output'); _events=list(getattr(run, 'events', []) or []); _notifications=list(getattr(run, 'notifications', []) or []); _turn_end=next((e.get('data') for e in reversed(_events) if isinstance(e, dict) and e.get('type') == 'turn/end'), {}); _notes=[{'method': str(getattr(n, 'method', '')), 'eventType': getattr(n, 'payload', {}).get('event', {}).get('type') if isinstance(getattr(n, 'payload', {}), dict) else None, 'status': getattr(n, 'payload', {}).get('status') if isinstance(getattr(n, 'payload', {}), dict) else None} for n in _notifications]; _u.urlopen(_u.Request(_url, data=_j.dumps({'sessionId': _sid, 'runId': 'pre-fix', 'hypothesisId': 'C', 'location': 'actor_harness.py:dsh-error-evidence', 'msg': '[DEBUG] DSH error evidence collected', 'data': {'turnEnd': _turn_end, 'notifications': _notes}, 'ts': int(_t.time() * 1000)}).encode(), headers={'Content-Type': 'application/json'}), timeout=1).read()\nexcept: pass")
        # #endregion
        tool_trace = extract_tool_trace_from_dsh_events(events)
        stdout = str(getattr(run, "final_response", "") or extract_dsh_assistant_text(events))
        # #region debug-point B:dsh-output-resolution
        exec("try:\n import json as _j, urllib.request as _u, time as _t\n _p='.dbg/dsh-empty-output.env'; _c=open(_p, encoding='utf-8').read(); _url=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SERVER_URL=')), 'http://127.0.0.1:7777/event'); _sid=next((x.split('=',1)[1] for x in _c.splitlines() if x.startswith('DEBUG_SESSION_ID=')), 'dsh-empty-output'); _u.urlopen(_u.Request(_url, data=_j.dumps({'sessionId': _sid, 'runId': 'pre-fix', 'hypothesisId': 'B', 'location': 'actor_harness.py:dsh-output-resolution', 'msg': '[DEBUG] DSH output resolution completed', 'data': {'resolvedStdoutLength': len(stdout), 'usedEventFallback': not bool(getattr(run, 'final_response', None)) and bool(stdout), 'toolCallCount': tool_trace.get('tool_call_count'), 'sessionRoot': getattr(run, 'session_root', None)}, 'ts': int(_t.time() * 1000)}).encode(), headers={'Content-Type':'application/json'}), timeout=1).read()\nexcept: pass")
        # #endregion
        actor_report, report_source, report_origin, report_transport = extract_report_metadata_from_stdout(stdout)
        notifications = [
            {"method": str(item.method), "payload": item.payload}
            for item in (getattr(run, "notifications", []) or [])
        ]
        finish_reason = getattr(run, "finish_reason", None)
        tool_trace["session_id"] = getattr(run, "session_id", None)
        tool_trace["finish_reason"] = finish_reason
        turn_error = next(
            (
                event.get("data", {}).get("reason", {}).get("error")
                for event in reversed(events)
                if isinstance(event, dict)
                and event.get("type") == "turn/end"
                and isinstance(event.get("data"), dict)
                and isinstance(event["data"].get("reason"), dict)
                and isinstance(event["data"]["reason"].get("error"), dict)
            ),
            None,
        )
        stderr = ""
        if finish_reason == "error":
            message = turn_error.get("message") if isinstance(turn_error, dict) else "DeepSeek Harness 回合以 error 结束，未返回最终输出。"
            stderr = f"DeepSeek Harness 执行失败：{message}"
        return ExecutionResult(
            ok=finish_reason != "error",
            mode=mode.value,
            final_prompt=prompt,
            stdout=stdout,
            stderr=stderr,
            return_code=1 if finish_reason == "error" else 0,
            script_report=script_report,
            script_report_source=report_source,
            script_report_object_origin=report_origin,
            script_report_transport_path=report_transport,
            final_script_report=actor_report,
            actor_backend="deepseek-harness",
            actor_run_metadata={
                "backend": "deepseek-harness",
                "session_id": getattr(run, "session_id", None),
                "finish_reason": finish_reason,
                "session_root": getattr(run, "session_root", None),
                "events": events,
                "notifications": notifications,
            },
            tool_trace=tool_trace,
        )

    def _load_harness_class(self) -> Any:
        try:
            from deepseek_harness import DeepSeekHarness
        except ImportError as exc:
            raise RuntimeError(
                "未找到 deepseek_harness SDK。请在当前 Python 环境安装 deepseek-harness-sdk。"
            ) from exc
        return DeepSeekHarness

    def _build_harness_kwargs(self) -> dict[str, Any]:
        Path(self.session_root).mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        optional_values = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "runtime_cwd": self.runtime_cwd,
            "session_root": self.session_root,
            "cordis": self.cordis,
            "env": {
                "DEEPSEEK_API_KEY": self.api_key or "",
                "DEEPSEEK_BASE_URL": self.base_url or "",
                "DEEPSEEK_MODEL": self.model or "",
            },
        }
        kwargs.update({key: value for key, value in optional_values.items() if value is not None})
        return kwargs
