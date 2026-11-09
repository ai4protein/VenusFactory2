import type { ChatHistoryItem } from "../lib/api";
import { renderMarkdown } from "../lib/markdown";
import "./ModelResultCard.css";

export function ModelResultCard({ item, lang }: { item: ChatHistoryItem; lang: string }) {
  const result = item.model_result;
  if (!result) return null;
  const zh = lang === "zh";
  const labels = { report: zh ? "查看报告" : "View report", csv: "CSV", json: "JSON" };
  const openFile = (path: string) => {
    // The existing Files panel owns session authentication, preview and download.
    window.dispatchEvent(new CustomEvent("venus:preview-file", { detail: { key: "agent/" + path } }));
  };
  return <article className="model-result-card" aria-label={zh ? "预测结果" : "Prediction result"}>
    <header>{zh ? "预测结果" : "Prediction result"}</header>
    <div className="model-result-values" dangerouslySetInnerHTML={{ __html: renderMarkdown(item.content) }}/>
    <details>
      <summary>{zh ? "适用范围" : "Applicability"}</summary>
      <div className="model-result-scope" dangerouslySetInnerHTML={{ __html: renderMarkdown(result.scope_markdown) }}/>
    </details>
    <footer>
      {(["report", "csv", "json"] as const).map(kind => {
        const path = result.files[kind];
        return path ? <button key={kind} type="button" onClick={() => openFile(path)}>{labels[kind]}</button> : null;
      })}
      {result.file_error && <span role="status">{zh ? "结果已保留，文件暂时无法下载。" : "Results are retained; files are currently unavailable."}</span>}
    </footer>
  </article>;
}
