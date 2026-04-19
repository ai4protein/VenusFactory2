import { useEffect, useRef, useState, useCallback } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import type { ChatHistoryItem } from "../lib/api";
import { submitFeedback } from "../lib/api";
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
  return DOMPurify.sanitize(html, {
    ADD_TAGS: ["img"],
    ADD_ATTR: ["src", "alt", "style", "width", "height", "loading"],
  });
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
const IMAGE_EXT_RE = /\.(?:png|jpe?g|gif|webp|bmp|svg|tiff?)$/i;

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
    const isImg = IMAGE_EXT_RE.test(m[1]) || IMAGE_EXT_RE.test(m[2]);
    attachments.push({ type: isImg ? "image" : "download", filename: m[1], url: m[2] });
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

function FeedbackButtons({
  sessionId,
  messageIndex,
}: {
  sessionId: string;
  messageIndex: number;
}) {
  const [rating, setRating] = useState<"like" | "dislike" | null>(null);
  const [sending, setSending] = useState(false);

  const handleRate = useCallback(
    async (value: "like" | "dislike") => {
      if (sending || rating === value) return;
      setSending(true);
      try {
        await submitFeedback(sessionId, messageIndex, value);
        setRating(value);
      } catch {
        // silently ignore
      } finally {
        setSending(false);
      }
    },
    [sessionId, messageIndex, rating, sending]
  );

  return (
    <>
      <button
        className={`feedback-btn feedback-like${rating === "like" ? " active" : ""}`}
        onClick={() => handleRate("like")}
        disabled={sending}
        title="Helpful"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 10v12" /><path d="M15 5.88L14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.88Z" />
        </svg>
      </button>
      <button
        className={`feedback-btn feedback-dislike${rating === "dislike" ? " active" : ""}`}
        onClick={() => handleRate("dislike")}
        disabled={sending}
        title="Not helpful"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M17 14V2" /><path d="M9 18.12L10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22h0a3.13 3.13 0 0 1-3-3.88Z" />
        </svg>
      </button>
    </>
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

function formatTime(ts: number): string {
  const d = new Date(ts);
  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

function formatDuration(ms: number): string | null {
  if (ms < 1500) return null;
  const secs = Math.round(ms / 1000);
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

function QuoteButton({ content, onQuote }: { content: string; onQuote: (t: string) => void }) {
  return (
    <button
      className="feedback-btn quote-btn"
      title="Quote reply"
      onClick={() => onQuote(content)}
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 21c3-3 6-6 6-12H3V3h6" /><path d="M15 21c3-3 6-6 6-12h-6V3h6" />
      </svg>
    </button>
  );
}

const SUGGESTED_PROMPTS = [
  "Search PubMed for recent PETase engineering studies, download PDB 5XJH, and use ProteinMPNN to redesign the active-site region for improved thermostability",
  "Download AlphaFold structure for human EGFR (P00533), predict beneficial mutations with ESM2 and ProtSSN, and identify top 10 stabilizing candidates",
  "Query BRENDA for lipase EC 3.1.1.3 kinetics across thermophilic organisms, find the most thermostable homolog on UniProt, and predict its solubility",
  "Predict the function of protein P04637, find its interaction partners on STRING, and check tissue expression from Human Protein Atlas",
];

interface ChatTimelineProps {
  items: ChatHistoryItem[];
  streamingIndex?: number;
  onSuggestedPrompt?: (text: string) => void;
  sessionId?: string;
  searchQuery?: string;
  onQuoteReply?: (text: string) => void;
}

export function ChatTimeline({ items, streamingIndex = -1, onSuggestedPrompt, sessionId, searchQuery, onQuoteReply }: ChatTimelineProps) {
  const msgTimesRef = useRef<Map<number, number>>(new Map());
  const prevSessionRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (prevSessionRef.current !== sessionId) {
      msgTimesRef.current.clear();
      prevSessionRef.current = sessionId;
    }
    items.forEach((_, idx) => {
      if (!msgTimesRef.current.has(idx)) {
        msgTimesRef.current.set(idx, Date.now());
      }
    });
  }, [items.length, sessionId]);

  const getDuration = useCallback(
    (idx: number): string | null => {
      const item = items[idx];
      if (item.role === "user") return null;
      const nextIdx = idx + 1;
      if (nextIdx < items.length && items[nextIdx].role !== "user") return null;
      let userIdx = idx - 1;
      while (userIdx >= 0 && items[userIdx].role !== "user") userIdx--;
      if (userIdx < 0) return null;
      const userTime = msgTimesRef.current.get(userIdx);
      const assistantTime = msgTimesRef.current.get(idx);
      if (!userTime || !assistantTime) return null;
      return formatDuration(assistantTime - userTime);
    },
    [items]
  );

  const queryLower = (searchQuery || "").toLowerCase();

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
        const msgTime = msgTimesRef.current.get(idx);
        const duration = getDuration(idx);
        const dimmed = queryLower && !rawContent.toLowerCase().includes(queryLower);
        return (
          <div
            key={idx}
            className={`chat-msg ${isUser ? "user" : "assistant"} with-avatar${isStreaming ? " streaming" : ""}${statusCls ? ` ${statusCls}` : ""}${dimmed ? " search-dimmed" : ""}`}
          >
            <img
              className="chat-msg-avatar"
              src={avatar}
              alt={roleLabel}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).src = DEFAULT_AVATAR;
              }}
            />
            <div className="chat-msg-content">
              <div className="chat-msg-role">
                {roleLabel}
                {msgTime && <span className="chat-msg-time">{formatTime(msgTime)}</span>}
                {duration && <span className="chat-msg-duration">{duration}</span>}
              </div>
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
              {!isUser && !isStreaming && (sessionId || onQuoteReply) && (
                <div className="chat-feedback-buttons">
                  {sessionId && <FeedbackButtons sessionId={sessionId} messageIndex={idx} />}
                  {onQuoteReply && (
                    <QuoteButton content={rawContent} onQuote={onQuoteReply} />
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
