import { useLang } from "../lib/i18n";

type Props = {
  onDecide: (action: "satisfied" | "modify_plan" | "continue") => void;
  disabled?: boolean;
};

const STRINGS = {
  en: {
    satisfied: "Satisfied",
    satisfiedHint: "Mark task as complete",
    modify: "Modify & Re-execute",
    modifyHint: "Edit the plan and run again",
    continue: "Continue Analysis",
    continueHint: "Provide new instructions"
  },
  zh: {
    satisfied: "已满意",
    satisfiedHint: "标记任务为完成",
    modify: "修改并重新执行",
    modifyHint: "编辑计划并再次执行",
    continue: "继续分析",
    continueHint: "提供新的指令"
  }
};

export function IterationDecision({ onDecide, disabled }: Props) {
  const t = useLang().t(STRINGS);
  return (
    <div className="iteration-decision">
      <div className="iteration-decision-options">
        <button
          className="iteration-btn iteration-btn-satisfied"
          onClick={() => onDecide("satisfied")}
          disabled={disabled}
        >
          <span className="iteration-btn-icon">&#10003;</span>
          <span className="iteration-btn-label">{t.satisfied}</span>
          <span className="iteration-btn-hint">{t.satisfiedHint}</span>
        </button>
        <button
          className="iteration-btn iteration-btn-modify"
          onClick={() => onDecide("modify_plan")}
          disabled={disabled}
        >
          <span className="iteration-btn-icon">&#8635;</span>
          <span className="iteration-btn-label">{t.modify}</span>
          <span className="iteration-btn-hint">{t.modifyHint}</span>
        </button>
        <button
          className="iteration-btn iteration-btn-continue"
          onClick={() => onDecide("continue")}
          disabled={disabled}
        >
          <span className="iteration-btn-icon">&#43;</span>
          <span className="iteration-btn-label">{t.continue}</span>
          <span className="iteration-btn-hint">{t.continueHint}</span>
        </button>
      </div>
    </div>
  );
}
