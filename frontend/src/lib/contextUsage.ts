import type { AgentContextInfo, KimiContextInfo } from "./api";

type Usage = { percent: number; used?: number; limit?: number; estimated: boolean; compacted: boolean };
const valid = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;
const percent = (ratio: number) => Math.round(Math.min(1, Math.max(0, ratio)) * 100);

export function contextUsage(context?: AgentContextInfo | null, native?: KimiContextInfo | null): Usage | null {
  if (valid(context?.estimated_tokens) && valid(context?.budget) && context.budget > 0) {
    return { percent: percent(context.estimated_tokens / context.budget), used: context.estimated_tokens,
      limit: context.budget, estimated: true, compacted: Boolean(context.compacted) };
  }
  if (valid(native?.context_tokens) && valid(native?.max_context_tokens) && native.max_context_tokens > 0) {
    return { percent: percent(native.context_tokens / native.max_context_tokens), used: native.context_tokens,
      limit: native.max_context_tokens, estimated: false, compacted: false };
  }
  if (valid(native?.context_usage)) {
    return { percent: percent(native.context_usage), estimated: false, compacted: false };
  }
  return null;
}
