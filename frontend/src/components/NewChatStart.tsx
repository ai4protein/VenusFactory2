import { CHAT_MODE_LABELS, type WorkspaceChatMode } from "../lib/unifiedChatHistory";
import { ChatModeIcon } from "./ChatModeIcon";
import "./NewChatStart.css";

const modes: WorkspaceChatMode[] = ["science_agent", "science_expert", "researcher"];
const copy = {
  science_agent: { title: ["What would you like to explore?", "今天想探索什么？"], hint: ["Ask questions, explore data and use tools.", "自由问答、分析数据、调用工具。"] },
  science_expert: { title: ["What would you like to develop?", "你想研发什么？"], hint: ["PI-led research: plan, experiment, critique and improve.", "PI 主导科研协作，在实验与评议中迭代改进。"] },
  researcher: { title: ["Which protein would you like to study?", "你想研究哪个蛋白？"], hint: ["Explore the evidence and build an in-depth report.", "检索信息、预测分析，生成深入的研究报告。"] },
};

export function NewChatStart({ mode, lang, onChange }: {
  mode: WorkspaceChatMode; lang: string; onChange: (mode: WorkspaceChatMode) => void;
}) {
  const language = lang === "zh" ? 1 : 0;
  return <section className="new-chat-start" aria-label={language ? "开始新对话" : "Start a new chat"}>
    <h1>{copy[mode].title[language]}</h1>
    <div className="new-chat-mode-options" role="radiogroup" aria-label={language ? "对话模式" : "Chat mode"}>
      {modes.map((value, index) => <button key={value} type="button" role="radio" aria-checked={mode === value}
        aria-label={CHAT_MODE_LABELS[value]} tabIndex={mode === value ? 0 : -1}
        onClick={() => { if (value !== mode) onChange(value); }}
        onKeyDown={event => {
          if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? 0 : event.key === "End" ? modes.length - 1 : (index + (["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1) + modes.length) % modes.length;
          const buttons = event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('button[role="radio"]');
          buttons?.[next]?.focus(); onChange(modes[next]);
        }}>
        <ChatModeIcon mode={value}/><span>{CHAT_MODE_LABELS[value]}</span>
      </button>)}
    </div>
    <p>{copy[mode].hint[language]}</p>
  </section>;
}
