import { useLang } from "../lib/i18n";

type Variant = "pill" | "inline" | "compact";

interface Props {
  variant?: Variant;
  className?: string;
}

export function LanguageSwitch({ variant = "pill", className }: Props) {
  const { lang, setLang } = useLang();
  const cls = `vf2-lang vf2-lang-${variant}${className ? ` ${className}` : ""}`;
  return (
    <div className={cls} role="group" aria-label="language switch">
      <button
        type="button"
        className={`vf2-lang-btn ${lang === "en" ? "active" : ""}`}
        onClick={() => setLang("en")}
        aria-pressed={lang === "en"}
        title="English"
      >
        EN
      </button>
      <button
        type="button"
        className={`vf2-lang-btn ${lang === "zh" ? "active" : ""}`}
        onClick={() => setLang("zh")}
        aria-pressed={lang === "zh"}
        title="中文"
      >
        中
      </button>
    </div>
  );
}
