import { modelAvailability, modelName, visibleExecutionModels, type ExecutionChoice } from "../lib/researchUi";

type Model = { id: string; label: string; provider: string; model: string };
export function ExecutionModelOptions({ models, combinations, harness, zh, online, selected = "" }: {
  models: Model[]; combinations: ExecutionChoice[]; harness: string; zh: boolean; online: boolean; selected?: string;
}) {
  const visible = visibleExecutionModels(models, combinations, harness, online);
  return <>
    {!visible.some(model => model.id === selected) && <option value="" disabled>{visible.length ? (zh ? "选择模型" : "Select a model") : (zh ? "暂无可用模型" : "No models available")}</option>}
    {visible.map(model => <option key={model.id} value={model.id} disabled={!modelAvailability(combinations, harness, model.id, zh).ready}>{modelName(model)}</option>)}
  </>;
}

export function executionModelItems(models: Model[], combinations: ExecutionChoice[], harness: string, zh: boolean, online: boolean) {
  return visibleExecutionModels(models, combinations, harness, online).map(model => {
    const availability = modelAvailability(combinations, harness, model.id, zh);
    return { value: model.id, label: modelName(model), disabled: !availability.ready, hint: availability.label };
  });
}
