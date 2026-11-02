/** Display and admission helpers for this workspace's free-model policy. */
export function isFreeResearchModel(model: { free?: boolean; provider?: string; model?: string; id?: string }): boolean {
  return model.free === true;
}

export function availabilityReason(reason: string | undefined, zh: boolean): string {
  const reasons: Record<string, [string, string]> = {
    web_interactions_pending: ["Not available in Web yet.", "暂未开放 Web 使用。"],
    adapter_pending: ["Not available in Web yet.", "暂未开放 Web 使用。"],
    model_protocol_incompatible: ["Choose a compatible model.", "请选择兼容模型。"],
    runtime_missing: ["Runtime unavailable.", "运行环境未配置。"],
    credential_unavailable: ["Add an API key.", "请配置 API Key。"],
    claude_controls_incompatible: ["Choose a compatible model.", "请选择兼容模型。"],
  };
  return reasons[reason || ""]?.[zh ? 1 : 0]
    || (zh ? "当前不可用。" : "Currently unavailable.");
}

export function sessionMatches(query: string, ...fields: Array<string | undefined>): boolean {
  const words = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const text = fields.filter(Boolean).join(" ").toLocaleLowerCase();
  return words.every(word => text.includes(word));
}


export type ExecutionChoice = { harness: string; model_id: string; ready: boolean; reasons: string[] };
export function modelAvailability(choices: ExecutionChoice[], harness: string, modelId: string, zh: boolean) {
  const choice = choices.find(c => c.harness === harness && c.model_id === modelId);
  if (choice?.ready) return { ready: true, label: "" };
  const reasons = choice?.reasons || [];
  const label = reasons.includes("credential_unavailable") ? (zh ? "未配置 Key" : "No API key")
    : reasons.includes("free_model_required") ? (zh ? "付费模型" : "Paid model")
    : reasons.some(r => ["model_protocol_incompatible", "claude_controls_incompatible", "chat_controls_incompatible"].includes(r)) ? (zh ? "不兼容" : "Incompatible")
    : (zh ? "不可用" : "Unavailable");
  return { ready: false, label };
}

export function preferredModel(choices: ExecutionChoice[], harness: string, current = "", fallback = "") {
  const ready = choices.filter(c => c.harness === harness && c.ready);
  return ready.find(c => c.model_id === current)?.model_id || ready.find(c => c.model_id === fallback)?.model_id || ready[0]?.model_id || "";
}

/** Labels never carry routing or pricing metadata; IDs still identify the route. */
export function modelName(model: { label?: string; model?: string; id?: string }): string {
  const label = (model.label || '').split(/\s+[·|]\s+/)[0].trim();
  const name = label || (model.model || model.id || '').split('/').pop() || '';
  return name.replace(/^[^:]+:\s+/, '').replace(/\s*\((?:free|免费)\)\s*$/i, '').replace(/:free$/i, '').trim();
}

export function visibleExecutionModels<T extends { id: string }>(models: T[], choices: ExecutionChoice[], harness: string, online: boolean): T[] {
  return models.filter(model => {
    const pair = choices.find(c => c.harness === harness && c.model_id === model.id);
    // Local menus reflect configured credentials; online menus show usable pairs only.
    return Boolean(pair && (online ? pair.ready : !pair.reasons.includes('credential_unavailable')));
  });
}

export function connectionNotice(message: string, zh: boolean): string {
  return /failed to fetch|fetch failed|network(?:error| request| error)?|connection (?:lost|closed|reset|interrupted)|econn(?:reset|refused)|load failed/i.test(message)
    ? (zh ? '连接已中断，请重试。' : 'Connection lost. Please try again.') : message;
}
