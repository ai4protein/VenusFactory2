import { useEffect, useState } from "react";
import { MolstarViewer } from "./MolstarViewer";
import { fileUrlFor } from "../lib/richContent";
import { renderMarkdown } from "../lib/markdown";

/**
 * Modal that previews a session file inline. Picks renderer by extension:
 *   - pdb/cif/mmcif/ent → MolstarViewer (3D)
 *   - png/jpg/svg/gif/webp/bmp/tiff → <img>
 *   - html/htm → sandboxed <iframe>
 *   - csv/tsv → parsed table
 *   - fasta family → monospace text (chunked + colorized residues optional)
 *   - md → rendered markdown (with hljs / table scroll wrapper)
 *   - txt/json/yaml/log + unknown → fenced code block (no highlight for txt)
 *
 * Backend serves all of these through `/api/files/inline?path=...` which
 * enforces containment + extension allowlist — we trust whatever it returns.
 */
export interface PreviewableFile {
  name: string;
  abs: string;
  size: number;
  ext: string;
}

const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "tiff", "tif"]);
// Anything Molstar can render — PDB/CIF for biomolecules, MOL/SDF/MOL2 for
// small molecules / ligands.
const STRUCT_EXTS = new Set(["pdb", "cif", "mmcif", "ent", "mol", "sdf", "mol2"]);
const HTML_EXTS = new Set(["html", "htm"]);
const CSV_EXTS = new Set(["csv", "tsv"]);
const FASTA_EXTS = new Set(["fasta", "fa", "faa", "fna", "ffn", "frn"]);
const MD_EXTS = new Set(["md", "markdown"]);
const TEXT_EXTS = new Set([
  "txt", "json", "yaml", "yml", "log", "py", "sh", "js", "ts", "tsx", "jsx",
  "css", "html", "xml", "sql", "r",
]);

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function parseDelim(text: string, sep: string): { headers: string[]; rows: string[][] } | null {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length < 1) return null;
  const headers = lines[0].split(sep);
  const rows = lines.slice(1).map((l) => l.split(sep));
  return { headers, rows };
}

function CsvView({ url, sep }: { url: string; sep: string }) {
  const [data, setData] = useState<{ headers: string[]; rows: string[][] } | null>(null);
  const [err, setErr] = useState<string>("");
  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((t) => { if (!cancelled) setData(parseDelim(t.slice(0, 1_000_000), sep)); })
      .catch((e) => { if (!cancelled) setErr(String(e)); });
    return () => { cancelled = true; };
  }, [url, sep]);
  if (err) return <div className="fpm-error">{err}</div>;
  if (!data) return <div className="fpm-loading">Loading…</div>;
  const rows = data.rows.slice(0, 200);
  return (
    <div className="fpm-csv">
      <table>
        <thead>
          <tr>{data.headers.map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => <td key={j}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {data.rows.length > rows.length && (
        <div className="fpm-csv-truncated">
          Showing first {rows.length} of {data.rows.length} rows.
        </div>
      )}
    </div>
  );
}

function TextView({ url, lang }: { url: string; lang?: string }) {
  const [text, setText] = useState<string>("");
  const [err, setErr] = useState<string>("");
  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((t) => { if (!cancelled) setText(t.slice(0, 2_000_000)); })
      .catch((e) => { if (!cancelled) setErr(String(e)); });
    return () => { cancelled = true; };
  }, [url]);
  if (err) return <div className="fpm-error">{err}</div>;
  if (text === "") return <div className="fpm-loading">Loading…</div>;
  // Render via markdown so code fences get hljs highlighting for free.
  const fenced = "```" + (lang || "") + "\n" + text + "\n```";
  const html = renderMarkdown(fenced);
  return <div className="fpm-text chat-msg-body" dangerouslySetInnerHTML={{ __html: html }} />;
}

function MarkdownView({ url }: { url: string }) {
  const [text, setText] = useState<string>("");
  const [err, setErr] = useState<string>("");
  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((t) => { if (!cancelled) setText(t); })
      .catch((e) => { if (!cancelled) setErr(String(e)); });
    return () => { cancelled = true; };
  }, [url]);
  if (err) return <div className="fpm-error">{err}</div>;
  if (text === "") return <div className="fpm-loading">Loading…</div>;
  return <div className="fpm-md chat-msg-body" dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }} />;
}

export function FilePreviewModal({
  file,
  onClose,
}: {
  file: PreviewableFile | null;
  onClose: () => void;
}) {
  // Close on Escape.
  useEffect(() => {
    if (!file) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [file, onClose]);

  if (!file) return null;
  const ext = (file.ext || "").toLowerCase();
  const url = fileUrlFor(file.abs);

  let body: React.ReactNode;
  if (STRUCT_EXTS.has(ext)) {
    body = (
      <div className="fpm-molstar-wrap">
        <MolstarViewer filePath={file.abs} label={file.name} />
      </div>
    );
  } else if (IMAGE_EXTS.has(ext)) {
    body = (
      <div className="fpm-image-wrap">
        <img src={url} alt={file.name} />
      </div>
    );
  } else if (HTML_EXTS.has(ext)) {
    body = (
      <iframe
        className="fpm-iframe"
        src={url}
        title={file.name}
        sandbox=""
        referrerPolicy="no-referrer"
      />
    );
  } else if (CSV_EXTS.has(ext)) {
    body = <CsvView url={url} sep={ext === "tsv" ? "\t" : ","} />;
  } else if (MD_EXTS.has(ext)) {
    body = <MarkdownView url={url} />;
  } else if (FASTA_EXTS.has(ext)) {
    body = <TextView url={url} lang="" />;
  } else if (TEXT_EXTS.has(ext)) {
    body = <TextView url={url} lang={ext === "py" ? "python" : ext} />;
  } else {
    body = (
      <div className="fpm-unknown">
        <p>No inline preview for <code>.{ext || "(no extension)"}</code>.</p>
        <a href={url} target="_blank" rel="noopener noreferrer">Open in new tab ↗</a>
      </div>
    );
  }

  return (
    <div className="fpm-backdrop" onClick={onClose}>
      <div className="fpm-modal" onClick={(e) => e.stopPropagation()}>
        <header className="fpm-head">
          <span className="fpm-name" title={file.abs}>{file.name}</span>
          <span className="fpm-meta">
            <span>{fmtSize(file.size)}</span>
            <span>·</span>
            <span>.{ext}</span>
          </span>
          <a
            className="fpm-open"
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            title="Open in new tab"
          >↗</a>
          <button
            type="button"
            className="fpm-close"
            onClick={onClose}
            aria-label="Close"
          >✕</button>
        </header>
        <div className="fpm-body">{body}</div>
      </div>
    </div>
  );
}
