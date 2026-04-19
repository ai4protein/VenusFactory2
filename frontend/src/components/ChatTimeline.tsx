import { useEffect, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import type { ChatHistoryItem } from "../lib/api";
import { MolstarViewer } from "./MolstarViewer";

const DEFAULT_AVATAR = "https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png";
const USER_AVATAR =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#ffdff0"/>
          <stop offset="100%" stop-color="#ffeec8"/>
        </linearGradient>
      </defs>
      <circle cx="48" cy="48" r="46" fill="url(#bg)"/>
      <circle cx="33" cy="40" r="5" fill="#8b6f5a"/>
      <circle cx="63" cy="40" r="5" fill="#8b6f5a"/>
      <path d="M30 57c5 8 31 8 36 0" fill="none" stroke="#8b6f5a" stroke-width="5" stroke-linecap="round"/>
      <circle cx="24" cy="52" r="6" fill="#ffc7da" opacity="0.85"/>
      <circle cx="72" cy="52" r="6" fill="#ffc7da" opacity="0.85"/>
    </svg>`
  );

const AVATAR_BY_ROLE: Record<string, string> = {
  principal_investigator: "/img/agent_role/principal_investigator.png",
  computational_biologist: "/img/agent_role/computational_biologist.png",
  machine_learning_specialist: "/img/agent_role/machine_learning_specialist.png",
  scientific_critic: "/img/agent_role/scientific_critic.png"
};

const ROLE_ALIAS_TO_CANONICAL: Record<string, string> = {
  pi: "principal_investigator",
  principalinvestigator: "principal_investigator",
  principal_investigator: "principal_investigator",
  cb: "computational_biologist",
  computationalbiologist: "computational_biologist",
  computational_biologist: "computational_biologist",
  mls: "machine_learning_specialist",
  machinelearningspecialist: "machine_learning_specialist",
  machine_learning_specialist: "machine_learning_specialist",
  sc: "scientific_critic",
  scientificcritic: "scientific_critic",
  scientific_critic: "scientific_critic"
};

function normalizeRoleId(roleId?: string, role?: string): string {
  const raw = (roleId || role || "").trim();
  if (!raw || raw === "assistant" || raw === "user") return "";
  const normalized = raw.toLowerCase().replace(/[\s-]+/g, "_");
  const compact = normalized.replace(/_/g, "");
  return ROLE_ALIAS_TO_CANONICAL[normalized] || ROLE_ALIAS_TO_CANONICAL[compact] || normalized;
}

function roleDisplayName(roleId: string, isUser: boolean) {
  if (isUser) return "User";
  if (!roleId) return "Assistant";
  return roleId.split("_").join(" ").replace(/\b\w/g, (m: string) => m.toUpperCase());
}

function roleAvatar(roleId: string, isUser: boolean) {
  if (isUser) return USER_AVATAR;
  return AVATAR_BY_ROLE[roleId] || DEFAULT_AVATAR;
}

function renderMarkdown(text: string) {
  const html = marked.parse(text || "", { async: false }) as string;
  return DOMPurify.sanitize(html);
}

function fallbackCopy(text: string): boolean {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.cssText = "position:fixed;left:-9999px";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    return true;
  } catch {
    return false;
  } finally {
    document.body.removeChild(ta);
  }
}

function statusMsgClass(content: string): string {
  if (!content) return "";
  if (content.startsWith("🤔") || content.startsWith("💭")) return "msg-status-thinking";
  if (content.startsWith("🔍")) return "msg-status-searching";
  if (content.startsWith("📋")) return "msg-status-planning";
  if (content.startsWith("⏳")) return "msg-status-executing";
  if (content.startsWith("📝") || content.startsWith("✍")) return "msg-status-writing";
  if (content.startsWith("✅")) return "msg-status-success";
  if (content.startsWith("❌")) return "msg-status-failed";
  return "";
}

const STRUCTURE_EXT_RE = /(?:[\w.\/~-]+\/)*[\w.-]+\.(?:pdb|cif|mmcif|ent)\b/gi;

function extractStructurePaths(text: string): string[] {
  const matches = text.match(STRUCTURE_EXT_RE) || [];
  return [...new Set(matches)].filter(
    (p) => p.includes("/") && !p.startsWith("http")
  );
}

/* ── Rich content extraction ── */

type RichAttachment =
  | { type: "download"; filename: string; url: string }
  | { type: "image"; filename: string; url: string }
  | { type: "csv_table"; filename: string; headers: string[]; rows: string[][] };

const RE_DOWNLOAD = /📎\s*\*\*Cloud Download:\*\*\s*\[([^\]]+)\]\(([^)]+)\)/g;
const RE_IMAGE = /🖼️\s*\*\*Generated Image:\*\*\s*\[([^\]]+)\]\(([^)]+)\)/g;
const RE_FILE_PREVIEW = /\*\*File Preview\s*\(([^)]+)\):\*\*\s*```[^\n]*\n([\s\S]*?)```/g;

function parseCSVLike(text: string): { headers: string[]; rows: string[][] } | null {
  const lines = text.trim().split("\n").filter((l) => l.trim());
  if (lines.length < 2) return null;
  const sep = lines[0].includes("\t") ? "\t" : ",";
  const headers = lines[0].split(sep).map((h) => h.trim());
  if (headers.length < 2) return null;
  const rows = lines.slice(1).map((line) => line.split(sep).map((c) => c.trim()));
  return { headers, rows };
}

function extractRichAttachments(text: string): { cleanText: string; attachments: RichAttachment[] } {
  const attachments: RichAttachment[] = [];
  let cleanText = text;

  for (const m of text.matchAll(RE_DOWNLOAD)) {
    attachments.push({ type: "download", filename: m[1], url: m[2] });
    cleanText = cleanText.replace(m[0], "");
  }

  for (const m of text.matchAll(RE_IMAGE)) {
    attachments.push({ type: "image", filename: m[1], url: m[2] });
    cleanText = cleanText.replace(m[0], "");
  }

  for (const m of text.matchAll(RE_FILE_PREVIEW)) {
    const filename = m[1];
    const content = m[2];
    if (filename.match(/\.csv$/i)) {
      const parsed = parseCSVLike(content);
      if (parsed) {
        attachments.push({ type: "csv_table", filename, ...parsed });
        cleanText = cleanText.replace(m[0], "");
        continue;
      }
    }
    // non-CSV previews stay as code blocks in markdown
  }

  return { cleanText, attachments };
}

const FILE_ICONS: Record<string, string> = {
  csv: "table",
  tsv: "table",
  fasta: "dna",
  fa: "dna",
  pdb: "cube",
  cif: "cube",
  html: "code",
  json: "braces",
  "tar.gz": "archive",
  png: "image",
  jpg: "image",
  svg: "image",
};

function fileIcon(filename: string): string {
  const ext = filename.includes(".tar.gz")
    ? "tar.gz"
    : filename.split(".").pop()?.toLowerCase() || "";
  return FILE_ICONS[ext] || "file";
}

function DownloadCard({ filename, url }: { filename: string; url: string }) {
  const icon = fileIcon(filename);
  return (
    <a className="rich-download-card" href={url} target="_blank" rel="noopener noreferrer" download={filename}>
      <span className={`rich-file-icon icon-${icon}`} />
      <span className="rich-download-name" title={filename}>{filename}</span>
      <span className="rich-download-action">Download</span>
    </a>
  );
}

function ImageEmbed({ filename, url }: { filename: string; url: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rich-image-wrap">
      <img
        className={`rich-image${expanded ? " expanded" : ""}`}
        src={url}
        alt={filename}
        onClick={() => setExpanded(!expanded)}
        loading="lazy"
      />
      <span className="rich-image-caption">{filename}</span>
    </div>
  );
}

function CsvPreviewTable({ filename, headers, rows }: { filename: string; headers: string[]; rows: string[][] }) {
  const MAX_ROWS = 20;
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? rows : rows.slice(0, MAX_ROWS);
  return (
    <div className="rich-csv-wrap">
      <div className="rich-csv-header">
        <span className="rich-csv-title">{filename}</span>
        <span className="rich-csv-meta">{rows.length} rows</span>
      </div>
      <div className="rich-csv-scroll">
        <table className="rich-csv-table">
          <thead>
            <tr>
              {headers.map((h, i) => (
                <th key={i}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={ci}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > MAX_ROWS && !showAll && (
        <button className="rich-csv-more" onClick={() => setShowAll(true)}>
          Show all {rows.length} rows
        </button>
      )}
    </div>
  );
}

function RichAttachmentList({ attachments }: { attachments: RichAttachment[] }) {
  if (attachments.length === 0) return null;
  return (
    <div className="rich-attachments">
      {attachments.map((att, i) => {
        if (att.type === "download") return <DownloadCard key={i} filename={att.filename} url={att.url} />;
        if (att.type === "image") return <ImageEmbed key={i} filename={att.filename} url={att.url} />;
        if (att.type === "csv_table") return <CsvPreviewTable key={i} filename={att.filename} headers={att.headers} rows={att.rows} />;
        return null;
      })}
    </div>
  );
}

function ChatMessageBody({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.querySelectorAll("pre").forEach((pre) => {
      if (pre.querySelector(".code-copy-btn")) return;
      const btn = document.createElement("button");
      btn.className = "code-copy-btn";
      btn.textContent = "Copy";
      btn.onclick = () => {
        const text = pre.querySelector("code")?.textContent || pre.textContent || "";
        const onSuccess = () => {
          btn.textContent = "Copied!";
          setTimeout(() => (btn.textContent = "Copy"), 1500);
        };
        if (navigator.clipboard?.writeText) {
          navigator.clipboard.writeText(text).then(onSuccess).catch(() => {
            fallbackCopy(text) && onSuccess();
          });
        } else {
          fallbackCopy(text) && onSuccess();
        }
      };
      pre.appendChild(btn);
    });
  }, [html]);

  return (
    <div
      ref={ref}
      className="chat-msg-body"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

const SUGGESTED_PROMPTS = [
  "Predict the structure of sequence MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
  "Search UniProt for human insulin receptor protein",
  "Analyze zero-shot mutation predictions for ESM2 on my uploaded FASTA",
  "What protein engineering strategies can improve enzyme thermostability?",
];

interface ChatTimelineProps {
  items: ChatHistoryItem[];
  streamingIndex?: number;
  onSuggestedPrompt?: (text: string) => void;
}

export function ChatTimeline({ items, streamingIndex = -1, onSuggestedPrompt }: ChatTimelineProps) {
  return (
    <div className="chat-timeline">
      {items.length === 0 && (
        <div className="chat-empty-state">
          <img
            className="chat-empty-logo"
            src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png"
            alt="VenusFactory"
          />
          <h3 className="chat-empty-title">How can I help with your protein research?</h3>
          <p className="chat-empty-subtitle">
            Ask about structure prediction, mutation analysis, database search, or protein engineering.
          </p>
          <div className="chat-suggested-prompts">
            {SUGGESTED_PROMPTS.map((prompt, i) => (
              <button
                key={i}
                className="chat-suggested-btn"
                onClick={() => onSuggestedPrompt?.(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      )}
      {items.map((item, idx) => {
        const isUser = item.role === "user";
        const roleId = normalizeRoleId(item.role_id, item.role);
        const roleLabel = roleDisplayName(roleId, isUser);
        const avatar = roleAvatar(roleId, isUser);
        const isStreaming = idx === streamingIndex;
        const rawContent = item.content || "";
        const { cleanText, attachments } = isUser || isStreaming
          ? { cleanText: rawContent, attachments: [] as RichAttachment[] }
          : extractRichAttachments(rawContent);
        const structurePaths = isUser ? [] : extractStructurePaths(rawContent);
        const statusCls = isUser ? "" : statusMsgClass(rawContent);
        return (
          <div key={idx} className={`chat-msg ${isUser ? "user" : "assistant"} with-avatar${isStreaming ? " streaming" : ""}${statusCls ? ` ${statusCls}` : ""}`}>
            <img
              className="chat-msg-avatar"
              src={avatar}
              alt={roleLabel}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).src = DEFAULT_AVATAR;
              }}
            />
            <div className="chat-msg-content">
              <div className="chat-msg-role">{roleLabel}</div>
              <ChatMessageBody html={renderMarkdown(cleanText)} />
              <RichAttachmentList attachments={attachments} />
              {structurePaths.length > 0 && (
                <div className="chat-msg-structures">
                  {structurePaths.map((sp) => (
                    <MolstarViewer
                      key={sp}
                      filePath={sp}
                      label={sp.split("/").pop() || sp}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
