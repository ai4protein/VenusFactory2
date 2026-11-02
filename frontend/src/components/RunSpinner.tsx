import "./RunSpinner.css";
export function RunSpinner({ lang = "en" }: { lang?: string }) {
  return <span className="run-spinner" role="img" aria-label={lang === "zh" ? "运行中" : "Running"} title={lang === "zh" ? "运行中" : "Running"}/>;
}
