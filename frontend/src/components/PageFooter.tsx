import { useLang } from "../lib/i18n";

const STRINGS = {
  en: {
    notice: "AI-generated content. Please verify critical details before use.",
    builtBy: "Built by",
    and: "and"
  },
  zh: {
    notice: "AI 生成内容，重要信息请自行核实后再使用。",
    builtBy: "构建者",
    and: "与"
  }
};

export function PageFooter() {
  const t = useLang().t(STRINGS);
  return (
    <footer className="chat-footer-note">
      <span>{t.notice}</span>
      <span>
        {t.builtBy}{" "}
        <a href="https://tyang816.github.io/" target="_blank" rel="noreferrer">
          Yang Tan
        </a>
        {" · "}
        <a href="mailto:tanyang.august@sjtu.edu.cn">tanyang.august@sjtu.edu.cn</a>
      </span>
      <span>
        {t.and}{" "}
        <a href="mailto:zlr_zmm@163.com">Lingrong Zhang</a>
        {" · "}
        <a href="mailto:zlr_zmm@163.com">zlr_zmm@163.com</a>
      </span>
    </footer>
  );
}
