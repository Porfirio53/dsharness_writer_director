from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from harnessbench.adapters.base import BaseAdapter
from harnessbench.models import AdapterRunContext, AdapterRunResult


class DeepSeekWriterDirectorAdapter(BaseAdapter):
    """Task-scoped adapter that keeps one DeepSeek runtime across all rounds."""

    name = "deepseek_writer_director"

    def __init__(self) -> None:
        self._task_actor = None

    def run(self, ctx: AdapterRunContext) -> AdapterRunResult:
        os.environ.update({key: str(value) for key, value in ctx.env.items()})
        os.environ.update(
            {
                "HARNESSBENCH_TASK_ID": ctx.task.task_id,
                "HARNESSBENCH_WORKSPACE": str(ctx.workspace),
                "HARNESSBENCH_SANDBOX": str(ctx.sandbox),
                "HARNESSBENCH_SESSION_ID": ctx.session_id,
                "HARNESSBENCH_PROMPT_FILE": str(ctx.prompt_file),
                "HARNESSBENCH_MODEL_ID": ctx.model_id,
            }
        )
        project_root = Path(str(ctx.model_config.get("project_root") or "")).resolve()
        if not (project_root / "harnessbench_deepseek_runtime.py").is_file():
            raise ValueError(f"invalid DeepSeek Writer/Director project root: {project_root}")
        project_text = str(project_root)
        if project_text not in sys.path:
            sys.path.insert(0, project_text)

        from harnessbench_deepseek_runtime import (
            HarnessBenchRoundConfig,
            HarnessBenchTaskActor,
            execute_harnessbench_round,
            public_result_payload,
        )

        if self._task_actor is None:
            self._task_actor = HarnessBenchTaskActor()
        config = HarnessBenchRoundConfig(
            workspace=ctx.workspace,
            sandbox=ctx.sandbox,
            prompt_file=ctx.prompt_file,
            session_id=ctx.session_id,
            task_id=ctx.task.task_id,
            project_root=project_root,
            actor_model=str(ctx.model_config.get("model") or ctx.model_id),
            writer_model=str(
                ctx.model_config.get("writer_model")
                or ctx.model_config.get("model")
                or ctx.model_id
            ),
            api_timeout_sec=float(ctx.model_config.get("api_timeout_sec") or 300.0),
            director_harness_enabled=bool(
                ctx.model_config.get("director_harness_enabled", True)
            ),
        )
        try:
            result = execute_harnessbench_round(config, task_actor=self._task_actor)
        except Exception as exc:
            payload = {
                "status": "deepseek_harness_process_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "task_id": ctx.task.task_id,
                "session_id": ctx.session_id,
                "result_label": "DeepSeekHarness-compatible local result",
            }
            return AdapterRunResult(
                ok=False,
                command=["in-process", self.name],
                stdout=json.dumps(payload, ensure_ascii=False),
                metadata={"returncode": 20},
            )

        payload = public_result_payload(result)
        ok = result.status not in {"agent_task_failed"}
        return AdapterRunResult(
            ok=ok,
            command=["in-process", self.name],
            stdout=json.dumps(payload, ensure_ascii=False),
            stderr="\n".join(result.error_events),
            metadata={"returncode": 0 if ok else 1},
        )

    def close(self) -> None:
        task_actor = self._task_actor
        self._task_actor = None
        if task_actor is not None:
            task_actor.close()
