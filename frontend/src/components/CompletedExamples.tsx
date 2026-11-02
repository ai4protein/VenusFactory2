import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import DOMPurify from "dompurify";
import { marked } from "marked";
import "./CompletedExamples.css";

type Example = { id: string; title: string; detail: string };
type ExampleDocument = {
  title: string; summary: string; objective: string; report: string;
  plan: { title: string; status: string }[];
  files: { name: string; content: string }[];
  sources: { id: string; title: string; url: string }[];
};
const examples: Record<"scientist" | "researcher", Example[]> = {
  scientist: [
    { id: "peptide-composition", title: "Peptide composition tool", detail: "Python tool · 19 checks · source code" },
    { id: "petase-retrieval", title: "PETase data retrieval", detail: "Public data · validation · project files" },
  ],
  researcher: [
    { id: "abl1-report", title: "ABL1 · Mouse", detail: "8 chapters · function, structure and variants" },
    { id: "hemoglobin-report", title: "Hemoglobin α · Human", detail: "8 topics · evidence and model predictions" },
  ],
};
const safeLink = (url: string) => /^https?:\/\//i.test(url);

export function CompletedExamples({ mode, lang }: { mode: "scientist" | "researcher"; lang: string }) {
  const [selected, setSelected] = useState<Example | null>(null);
  const language = lang === "zh" ? 1 : 0;
  return <section className="completed-examples" aria-label={language ? "结果样例" : "Completed examples"}>
    <h2>{language ? "结果样例" : "Completed examples"}</h2>
    <div className="completed-example-cards">{examples[mode].map(example => <button type="button" key={example.id} onClick={() => setSelected(example)}>
      <span className="example-book" aria-hidden="true">▤</span><span><strong>{example.title}</strong><small>{example.detail}</small></span><span aria-hidden="true">↗</span>
    </button>)}</div>
    {selected && createPortal(<ExampleReader key={selected.id} example={selected} lang={lang} onClose={() => setSelected(null)}/>, document.body)}
  </section>;
}

function ExampleReader({ example, lang, onClose }: { example: Example; lang: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [document, setDocument] = useState<ExampleDocument | null>(null), [error, setError] = useState(false);
  const [file, setFile] = useState(""), [retry, setRetry] = useState(0);
  const language = lang === "zh" ? 1 : 0;
  useEffect(() => { const element = dialog.current; element?.showModal(); return () => element?.close(); }, []);
  useEffect(() => {
    const controller = new AbortController(); setError(false);
    fetch(`/examples/${example.id}.json`, { signal: controller.signal }).then(async response => {
      if (!response.ok || !response.headers.get("content-type")?.includes("json")) throw new Error("Example unavailable");
      return response.json() as Promise<ExampleDocument>;
    }).then(setDocument).catch(error => { if (error.name !== "AbortError") setError(true); });
    return () => controller.abort();
  }, [example.id, retry]);
  const selectedFile = document?.files.find(item => item.name === file);
  const markdown = selectedFile ? /\.md$/i.test(file) : true;
  const content = selectedFile?.content || document?.report || "";
  return <dialog ref={dialog} className="example-reader" aria-labelledby="example-reader-title" onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="example-reader-shell">
      <header><div><small>{language ? "结果样例" : "Completed example"}</small><h2 id="example-reader-title">{example.title}</h2></div>
        <a href={`/examples/${example.id}.zip`} download>{language ? "下载 ZIP" : "Download ZIP"} ↓</a>
        <button type="button" onClick={onClose} aria-label={language ? "关闭样例" : "Close example"}>×</button>
      </header>
      {error ? <div className="example-reader-loading"><p>{language ? "样例暂时未能加载。" : "The example could not be loaded."}</p><button type="button" onClick={() => setRetry(n => n + 1)}>{language ? "重试" : "Retry"}</button></div> : !document ? <p className="example-reader-loading" role="status">{language ? "正在打开样例…" : "Opening example…"}</p> : <div className="example-reader-layout">
        <nav aria-label={language ? "样例文件" : "Example files"}>
          <button type="button" aria-current={!file ? "page" : undefined} onClick={() => setFile("")}>{language ? "报告与结果" : "Report & results"}</button>
          {document.files.map(item => <button type="button" key={item.name} aria-current={file === item.name ? "page" : undefined} onClick={() => setFile(item.name)}>{item.name}</button>)}
        </nav>
        <main key={file} tabIndex={0}>
          {!file && <><p className="example-summary">{document.summary}</p>{!!document.plan.length && <details className="example-plan"><summary>{language ? "研究计划" : "Research plan"}</summary><ol>{document.plan.map((step, i) => <li key={i} data-status={step.status}><span>{step.status === "completed" ? "✓" : "◐"}</span>{step.title}</li>)}</ol></details>}</>}
          {markdown ? <article className="example-markdown" dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(marked.parse(content) as string, { FORBID_TAGS: ["img", "iframe", "style", "form", "input", "button"] }) }}/> : <pre className="example-source"><code>{content}</code></pre>}
          {!file && !!document.sources.length && <section className="example-sources"><h3>{language ? "参考来源" : "Sources"}</h3><ol>{document.sources.filter(source => safeLink(source.url)).map(source => <li key={source.id}><a href={source.url} target="_blank" rel="noreferrer">{source.id} · {source.title}</a></li>)}</ol></section>}
        </main>
      </div>}
    </div>
  </dialog>;
}
