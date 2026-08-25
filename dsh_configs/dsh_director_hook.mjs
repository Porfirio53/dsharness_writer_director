import { spawnSync } from 'node:child_process'

export const name = 'writer-director-pre-execute'
export const inject = ['tools']

function enabled() {
  return ['1', 'true', 'yes', 'on'].includes(
    String(process.env.DIRECTOR_HARNESS_ENABLED ?? '').trim().toLowerCase(),
  ) && process.env.DSH_DIRECTOR_STAGE === 'actor'
}

export function apply(ctx, config = {}) {
  ctx.on('tools/pre-execute', async (exec, next) => {
    if (!enabled()) return next()

    const pythonBin = config.pythonBin || process.env.DSH_DIRECTOR_PYTHON || 'python3'
    const bridgePath = config.bridgePath || process.env.DSH_DIRECTOR_BRIDGE
    if (!bridgePath) {
      return { kind: 'deny', reason: 'Director bridge path is not configured' }
    }
    const completed = spawnSync(pythonBin, [bridgePath], {
      input: JSON.stringify({
        tool_name: exec.name,
        tool_input: exec.arguments ?? {},
        tool_use_id: String(exec.callId ?? ''),
        session_id: process.env.DSH_DIRECTOR_SESSION_ID ?? '',
      }),
      encoding: 'utf8',
      env: process.env,
      timeout: Number(config.timeoutMs ?? 10000),
    })
    if (completed.error || completed.status !== 0) {
      const detail = completed.error?.message || completed.stderr?.trim() || `exit ${completed.status}`
      return { kind: 'deny', reason: `Director preflight failed: ${detail}` }
    }
    let decision
    try {
      decision = JSON.parse(completed.stdout)
    } catch {
      return { kind: 'deny', reason: 'Director preflight returned invalid JSON' }
    }
    if (decision.action === 'deny') {
      return { kind: 'deny', reason: decision.reason || 'Director denied the tool call' }
    }
    return next()
  }, { prepend: true })
}
