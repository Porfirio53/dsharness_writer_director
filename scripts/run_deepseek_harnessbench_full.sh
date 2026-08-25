#!/usr/bin/env bash
set -euo pipefail

check_only=0
if [[ "${1:-}" == "--check-only" ]]; then
  check_only=1
  shift
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${1:-$(cd "${script_dir}/.." && pwd)}"
env_file="${2:-${project_root}/.env}"
default_run_id="$(date -u +%Y%m%dT%H%M%SZ)-group-writer-v1"
result_root="${3:-${project_root}/results/runs/deepseek-writer-director-full/${default_run_id}}"
python_bin="${project_root}/.venv/bin/python"
harnessbench_root="${project_root}/HarnessBench"
task_manifest="${project_root}/results/config/harnessbench-writer-full-v1.tasks.json"
output_dir="${result_root}/HarnessBench"

if [[ ! -x "${python_bin}" ]]; then
  echo "Project virtualenv Python not found: ${python_bin}" >&2
  exit 20
fi

cd "${project_root}"
"${python_bin}" scripts/run_deepseek_harnessbench.py preflight \
  --harnessbench-root "${harnessbench_root}" \
  --task-manifest "${task_manifest}" \
  --env-file "${env_file}" \
  --grading full \
  --public-url-mode loopback \
  --writer-workspace-root "${project_root}" \
  --director-harness-enabled

if (( check_only )); then
  exit 0
fi

resume_args=()
if [[ -f "${output_dir}/run-config.json" ]]; then
  resume_args+=(--resume)
fi

"${python_bin}" scripts/run_deepseek_harnessbench.py run \
  --harnessbench-root "${harnessbench_root}" \
  --task-manifest "${task_manifest}" \
  --env-file "${env_file}" \
  --grading full \
  --public-url-mode loopback \
  --api-timeout-sec 300 \
  --repeats 2 \
  --writer-workspace-root "${project_root}" \
  --director-harness-enabled \
  --output-dir "${output_dir}" \
  "${resume_args[@]}"
