import { RunSpinner } from "./RunSpinner";
import type { ChatMode } from "../lib/useModelRegistry";

export function ResearchWorkspaceHeader({ mode, onModeChange, busy, researching, waiting, lang, runningModes = {} }: {
  mode: ChatMode | "researcher"; onModeChange: (mode: ChatMode | "researcher") => void;
  runningModes?: Partial<Record<ChatMode | "researcher", boolean>>;
  busy: boolean; researching: boolean; waiting: boolean; lang: string;
}) {
  const zh = lang === "zh";
  return <header className="research-workspace-header">
    <div className="workspace-mode-switch" role="group" aria-label={zh ? "工作模式" : "Workspace mode"}>
      <button type="button" aria-pressed={mode === "science_agent"} disabled={busy} onClick={() => onModeChange("science_agent")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 3V6a2 2 0 0 1 2-2Z"/><path d="M7 9h10M7 13h6"/></svg>
        <span>Agent</span>{runningModes.science_agent && <RunSpinner lang={lang}/>}
      </button>
      <button type="button" aria-pressed={mode === "science_expert"} disabled={busy} onClick={() => onModeChange("science_expert")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6M10 3v7L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-9V3M7 16h10"/></svg>
        <span>Scientist</span>{runningModes.science_expert && <RunSpinner lang={lang}/>}
      </button>
      <button type="button" aria-pressed={mode === "researcher"} disabled={busy} onClick={() => onModeChange("researcher")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6M10 7v6"/></svg>
        <span>Researcher</span>{runningModes.researcher && <RunSpinner lang={lang}/>}
      </button>
    </div>
    {(busy || researching || waiting) && <span className={`workspace-state${busy || researching ? " is-running" : ""}`} role="status">
      <i aria-hidden="true"/>{researching ? (zh ? "研究进行中" : "Research running") : busy ? (zh ? "执行中" : "Working") : waiting ? (zh ? "等待你的操作" : "Needs your input") : (zh ? "待命" : "Idle")}
    </span>}
  </header>;
}
