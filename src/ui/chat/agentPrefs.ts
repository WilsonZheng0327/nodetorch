// Per-user agent preferences (browser-local). The tool executor lives in the
// browser, so the approval gate is enforced here too: useAgentChat consults
// requiresApproval() before running a gated tool call, and AgentSettings
// exposes the toggle.

/** Tools that control training — the ones gated behind user approval. */
export const APPROVAL_TOOLS = new Set(['start_training', 'stop_training']);

const ASK_KEY = 'agent-ask-before-training';

/** Whether the assistant must ask before starting/stopping training. Default: yes. */
export function askBeforeTraining(): boolean {
  try {
    return localStorage.getItem(ASK_KEY) !== '0';
  } catch {
    return true;
  }
}

export function setAskBeforeTraining(ask: boolean): void {
  try {
    localStorage.setItem(ASK_KEY, ask ? '1' : '0');
  } catch {
    /* storage unavailable — the default (ask) applies */
  }
}

/** True if this tool call must wait for the user's Allow/Deny. */
export function requiresApproval(tool: string): boolean {
  return APPROVAL_TOOLS.has(tool) && askBeforeTraining();
}
