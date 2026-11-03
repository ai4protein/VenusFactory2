import { useEffect, useState } from "react";
import { getAgentExecutionOptions, type AgentExecutionOptions, type AgentExecutionSelection } from "../lib/api";
import { preferredModel, modelAvailability } from "../lib/researchUi";
import { executionModelItems } from "./ExecutionModelOptions";
import { ComposerSelect } from "./ComposerSelect";

type Props = {
  value: AgentExecutionSelection | null;
  onChange: (value: AgentExecutionSelection | null, userInitiated?: boolean) => void;
  onReadyChange: (ready: boolean) => void;
  locked: boolean; busy: boolean; lang: string; online: boolean;
};

export function AgentExecutionPicker({ value, onChange, onReadyChange, locked, busy, lang, online }: Props) {
  const [options, setOptions] = useState<AgentExecutionOptions | null>(null);
  const [failed, setFailed] = useState(false);
  const zh = lang === "zh";
  useEffect(() => {
    let active = true;
    setFailed(false);
    getAgentExecutionOptions().then(data => { if (active) setOptions(data); })
      .catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    if (!locked && !busy && options) {
      const harness = value?.harness || options.default.harness;
      const model_id = preferredModel(options.combinations, harness, value?.model_id, options.shared_default_model);
      if (!value || value.model_id !== model_id) onChange({ harness, model_id }, false);
    }
  }, [options, value, locked, busy, onChange]);
  const harness = value?.harness || "kimi-code";
  const current = options?.harnesses.find(h => h.id === harness);
  const ready = Boolean((locked && value?.harness === "langgraph") || current?.ready && (value ? modelAvailability(options?.combinations || [], harness, value.model_id, zh).ready : locked));
  useEffect(() => { onReadyChange(ready); }, [ready, onReadyChange]);
  return <>
    <ComposerSelect label="Harness" ariaLabel="Agent harness" value={harness}
      disabled={locked || busy || !options} disabledHint={locked ? (zh ? "在新对话中更换" : "Start a new chat to change") : undefined}
      placeholder={failed ? (zh ? "加载失败" : "Could not load") : (zh ? "加载中…" : "Loading…")}
      items={[...(locked && value?.harness === "langgraph" ? [{ value: "langgraph", label: "LangGraph" }] : []),
        ...(options?.harnesses || []).map(h => ({ value: h.id, label: h.label, disabled: !h.ready, hint: zh ? "未配置" : "Not configured" }))]}
      onChange={next => onChange({ harness: next, model_id: preferredModel(options?.combinations || [], next, value?.model_id, options?.shared_default_model) })}/>
    {value && <ComposerSelect label={zh ? "模型" : "Model"} ariaLabel="Agent LLM" value={value.model_id} helpText={zh ? "根据API反馈速度不一可能需要耐心等待" : "API response times vary. Please allow some time for a response."}
      disabled={locked || busy || !options} disabledHint={locked ? (zh ? "在新对话中更换" : "Start a new chat to change") : undefined}
      placeholder={zh ? "暂无可用模型" : "No models available"}
      items={executionModelItems(options?.models || [], options?.combinations || [], harness, zh, online)}
      onChange={model_id => onChange({ ...value, model_id })}/>}
  </>;
}
