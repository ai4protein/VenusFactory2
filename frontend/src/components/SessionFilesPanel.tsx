import { useEffect, useState } from "react";
import { fileUrlFor } from "../lib/richContent";
import { FilePreviewModal, type PreviewableFile } from "./FilePreviewModal";

/**
 * Lists every file in the current chat session's working directory.
 * Polls every `pollMs` when `liveRefresh` is true (i.e. a run is in flight)
 * so newly-generated artifacts surface without a manual refresh.
 *
 * Files are linked through `/api/files/inline?path=...` for inline preview
 * (image/html/csv/fasta/pdb). The backend enforces containment and the
 * extension allowlist there, so this panel can't be used as an exfil
 * channel even if the session dir contains stray files.
 */
interface SessionFile {
  name: string;
  rel: string;
  abs: string;
  size: number;
  mtime: number;          // unix seconds (server)
  ext: string;
}

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function fmtTime(secs: number): string {
  try {
    const d = new Date(secs * 1000);
    return d.toLocaleTimeString();
  } catch {
    return "";
  }
}

const EXT_ICONS: Record<string, string> = {
  // Biomolecule + small-molecule structures (all Molstar-renderable)
  pdb: "🧪", cif: "🧪", mmcif: "🧪", ent: "🧪",
  sdf: "⚛️", mol: "⚛️", mol2: "⚛️",
  png: "🖼️", jpg: "🖼️", jpeg: "🖼️", svg: "🖼️", gif: "🖼️", webp: "🖼️",
  fasta: "🧬", fa: "🧬", faa: "🧬", fna: "🧬", ffn: "🧬", frn: "🧬",
  csv: "📊", tsv: "📊", json: "📋", yaml: "📋", yml: "📋",
  html: "🌐", htm: "🌐",
  py: "🐍", sh: "💻", js: "📜", ts: "📜",
  txt: "📄", md: "📝", log: "📃",
};

function iconFor(ext: string): string {
  return EXT_ICONS[ext.toLowerCase()] || "📁";
}

interface Props {
  sessionId: string;
  authHeaders?: Record<string, string>;
  liveRefresh?: boolean;
  pollMs?: number;
}

export function SessionFilesPanel({
  sessionId,
  authHeaders,
  liveRefresh = false,
  pollMs = 5000,
}: Props) {
  const [files, setFiles] = useState<SessionFile[]>([]);
  const [sessionDir, setSessionDir] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<PreviewableFile | null>(null);

  const fetchFiles = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const r = await fetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/files`, {
        headers: authHeaders || {},
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setFiles(j.files || []);
      setSessionDir(j.session_dir || "");
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setFiles([]);
    setSessionDir("");
    setError("");
    if (sessionId) fetchFiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  useEffect(() => {
    if (!liveRefresh || !sessionId) return;
    const id = window.setInterval(fetchFiles, pollMs);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveRefresh, sessionId, pollMs]);

  if (!sessionId) {
    return (
      <div className="session-files-empty">
        Start a chat to see its working directory.
      </div>
    );
  }

  return (
    <div className="session-files">
      <div className="session-files-head">
        <span className="session-files-title">Session files</span>
        <span className="session-files-count">
          {loading ? "…" : `${files.length}`}
        </span>
        <button
          type="button"
          className="session-files-refresh"
          onClick={fetchFiles}
          title="Refresh"
          disabled={loading}
        >
          ↻
        </button>
      </div>
      {/* Don't expose the absolute filesystem path — that's user-machine
          structure they shouldn't have to see (and online-mode users would
          leak our deployment layout). The session's own scope is enough
          context; files render with their relative names anyway. */}
      {error && <div className="session-files-error">{error}</div>}
      {!error && files.length === 0 && !loading && (
        <div className="session-files-empty">No files yet.</div>
      )}
      <div className="session-files-list">
        {files.map((f) => (
          <button
            key={f.abs}
            type="button"
            className="session-files-item"
            onClick={() => setPreview({
              name: f.name,
              abs: f.abs,
              size: f.size,
              ext: f.ext,
            })}
            title={`${f.rel}\n${fmtSize(f.size)} · ${fmtTime(f.mtime)}\nclick to preview`}
          >
            <span className="session-files-icon">{iconFor(f.ext)}</span>
            <span className="session-files-name">{f.name}</span>
            <span className="session-files-meta">{fmtSize(f.size)}</span>
          </button>
        ))}
      </div>
      <FilePreviewModal file={preview} onClose={() => setPreview(null)} />
    </div>
  );
}
