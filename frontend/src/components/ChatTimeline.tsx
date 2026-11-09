import { Fragment, useEffect, useRef, useState, useCallback } from "react";
import { ModelResultCard } from "./ModelResultCard";
import type { ChatHistoryItem, SecurityEvent } from "../lib/api";
import { submitFeedback } from "../lib/api";
import { renderMarkdown } from "../lib/markdown";
import { ToolExecutionList, isResearchNoiseTool, type ToolExecution } from "./ToolExecutionCard";
import { StructureFileButton } from "./StructureFileButton";
import { isStructureFile, openStructure, structurePathFromLink } from "../lib/structurePreview";
import { useLang } from "../lib/i18n";
import { LOGO_VENUS_URL } from "../lib/brandAssets";
import { buildConversationProcesses, isProcessItem, isResearchStatusItem, toolActivityLabel, toolState, type ConversationProcess } from "../lib/conversationProcess";

const TIMELINE_STRINGS = {
  en: {
    helpful: "Helpful",
    notHelpful: "Not helpful",
    quote: "Quote reply",
    copy: "Copy",
    copied: "Copied!",
    downloadCard: "Download",
    thinking: "Thinking",
    thought: "Thought",
    thoughtFor: (s: number) => (s < 60 ? `Thought for ${s}s` : `Thought for ${Math.floor(s / 60)}m ${s % 60}s`),
    researchProcess: (n: number) => `Search process (${n})`,
    compaction: "Context compressed",
  },
  zh: {
    helpful: "有帮助",
    notHelpful: "没帮助",
    quote: "引用回复",
    copy: "复制",
    copied: "已复制！",
    downloadCard: "下载",
    thinking: "正在思考中",
    thought: "已思考",
    thoughtFor: (s: number) => (s < 60 ? `思考了 ${s}s` : `思考了 ${Math.floor(s / 60)}分 ${s % 60}秒`),
    researchProcess: (n: number) => `检索过程 (${n})`,
    compaction: "上下文已压缩",
  }
};

/** PI/SC long-form reports that start with a markdown H1. */
function isHighlightReport(item: ChatHistoryItem): boolean {
  if (item.role !== "assistant" || item.kind === "status" || item.kind === "thinking") return false;
  const c = (item.content || "").trim();
  return c.startsWith("# ") && c.length >= 160;
}


const DEFAULT_AVATAR = LOGO_VENUS_URL;
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
  // Generic assistant label — avoid hardcoding "Agent" so Science Expert turns
  // are not mislabeled when role_id is absent.
  if (!roleId) return "Assistant";
  return roleId.split("_").join(" ").replace(/\b\w/g, (m: string) => m.toUpperCase());
}

function roleAvatar(roleId: string, isUser: boolean) {
  if (isUser) return USER_AVATAR;
  return AVATAR_BY_ROLE[roleId] || DEFAULT_AVATAR;
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

const STRUCTURE_EXT_RE = /(?:[\w.\/~-]+\/)*[\w.-]+\.(?:pdb|cif|mmcif|ent|mol|sdf|mol2|bcif)\b/gi;

function extractStructurePaths(text: string): string[] {
  const matches = text.match(STRUCTURE_EXT_RE) || [];
  // Dedupe by exact string first, then drop any path whose basename
  // is already covered by a longer / more-qualified path in the set.
  // Without this collapse, a message that mentions `P00734.pdb` in a
  // markdown link AND `/path/to/P00734.pdb` as a URL would render the
  // same structure TWICE in two MolstarViewers.
  const unique = [...new Set(matches)].filter((p) => !p.startsWith("http"));
  return unique.filter((p) => {
    return !unique.some((other) => other !== p && other.endsWith("/" + p));
  });
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
  const t = useLang().t(TIMELINE_STRINGS);
  if (isStructureFile(filename)) return <StructureFileButton filePath={structurePathFromLink(url)} label={filename}/>;
  return (
    <a className="rich-download-card" href={url} target="_blank" rel="noopener noreferrer" download={filename}>
      <span className={`rich-file-icon icon-${icon}`} />
      <span className="rich-download-name" title={filename}>{filename}</span>
      <span className="rich-download-action">{t.downloadCard}</span>
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
  const t = useLang().t(TIMELINE_STRINGS);

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
        title={t.helpful}
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 10v12" /><path d="M15 5.88L14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.88Z" />
        </svg>
      </button>
      <button
        className={`feedback-btn feedback-dislike${rating === "dislike" ? " active" : ""}`}
        onClick={() => handleRate("dislike")}
        disabled={sending}
        title={t.notHelpful}
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M17 14V2" /><path d="M9 18.12L10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22h0a3.13 3.13 0 0 1-3-3.88Z" />
        </svg>
      </button>
    </>
  );
}

function ChatMessageBody({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const t = useLang().t(TIMELINE_STRINGS);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.querySelectorAll("pre").forEach((pre) => {
      if (pre.querySelector(".code-copy-btn")) return;
      const btn = document.createElement("button");
      btn.className = "code-copy-btn";
      btn.textContent = t.copy;
      btn.onclick = () => {
        const text = pre.querySelector("code")?.textContent || pre.textContent || "";
        const onSuccess = () => {
          btn.textContent = t.copied;
          setTimeout(() => (btn.textContent = t.copy), 1500);
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
  }, [html, t]);

  return (
    <div
      ref={ref}
      className="chat-msg-body"
      onClick={e => { const a = (e.target as HTMLElement).closest("a"); const href = a?.getAttribute("href"); if (href && isStructureFile(href) && !e.ctrlKey && !e.metaKey && !e.shiftKey) { e.preventDefault(); openStructure({ path: structurePathFromLink(href), name: a?.textContent || undefined }); } }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

/**
 * Banner listing kimi tool calls the backend security policy refused.
 * Default closed; click summary to expand.
 */
function SecurityEventsBanner({ events }: { events: SecurityEvent[] }) {
  if (events.length === 0) return null;
  const last5 = events.slice(-5).reverse();
  return (
    <details className="chat-security-banner">
      <summary>
        <span className="chat-security-icon">🛡️</span>
        <span className="chat-security-label">
          {events.length === 1
            ? "1 tool call blocked by security policy"
            : `${events.length} tool calls blocked by security policy`}
        </span>
      </summary>
      <div className="chat-security-body">
        {last5.map((e, i) => (
          <div key={i} className="chat-security-item">
            <div className="chat-security-item-head">
              <code>{e.tool}</code>
              <span className="chat-security-item-ts">
                {(() => {
                  try { return new Date(e.ts).toLocaleTimeString(); }
                  catch { return e.ts; }
                })()}
              </span>
            </div>
            <div className="chat-security-item-reason">{e.reason}</div>
            {e.input_preview && (
              <pre className="chat-security-item-input">{e.input_preview}</pre>
            )}
          </div>
        ))}
        {events.length > 5 && (
          <div className="chat-security-more">…{events.length - 5} more older</div>
        )}
      </div>
    </details>
  );
}

/** A quiet progress label; reasoning remains expandable when available. */
function ConversationProcessView({ process }: { process: ConversationProcess }) {
  const { lang } = useLang();
  const zh = lang === "zh";
  const [open, setOpen] = useState(process.live);
  useEffect(() => setOpen(process.live), [process.live]);
  const runningTools = process.tools.filter(tool => toolState(tool, process.live) === "running");
  const stopped = process.tools.filter(tool => toolState(tool, process.live) === "stopped").length;
  const count = process.tools.length;
  const latest = runningTools[runningTools.length - 1];
  const label = process.live
    ? latest ? toolActivityLabel(latest, zh) : zh ? "正在思考中" : "Thinking"
    : count ? (zh ? `${count} 个执行步骤` : `${count} ${count === 1 ? "step" : "steps"}`) : (zh ? "已思考" : "Thought");
  const status = stopped ? (zh ? `${stopped} 项已停止` : `${stopped} stopped`) : "";
  const className = `conversation-process${process.live ? " is-active" : ""}`;
  const messages = process.messages.filter(({ item }) => item.content.trim());
  if (!count && !messages.length && !process.live) return null;
  if (!count && !messages.length) return <div className={className} role="status"><span className="process-label">{label}</span></div>;
  return <details className={className} open={open} onToggle={e => setOpen(e.currentTarget.open)}>
    <summary>
      <span className="process-label" aria-live="polite">{label}</span>
      {status && <span className="process-step-count">{status}</span>}
      <svg className="process-chevron" width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m6 4 4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
    </summary>
    <div className="process-body">
      {messages.map(({ item, index }) => item.kind === "thinking" || item.phase === "thinking"
        ? <details className="process-note" key={index}><summary>{zh ? "思考" : "Thinking"}</summary><ChatMessageBody html={renderMarkdown(item.content)}/></details>
        : <div className="process-status-note" key={index}><ChatMessageBody html={renderMarkdown(item.content)}/></div>)}
      <ToolExecutionList tools={process.tools} recentCount={process.tools.length} variant="process" live={process.live}/>
    </div>
  </details>;
}

function QuoteButton({ content, onQuote }: { content: string; onQuote: (text: string) => void }) {
  const t = useLang().t(TIMELINE_STRINGS);
  return (
    <button
      className="feedback-btn quote-btn"
      title={t.quote}
      onClick={() => onQuote(content)}
    >
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 21c3-3 6-6 6-12H3V3h6" /><path d="M15 21c3-3 6-6 6-12h-6V3h6" />
      </svg>
    </button>
  );
}

const SUGGESTED_PROMPTS = {
  "en": [
    {
      "title": "Explore a protein",
      "desc": "Look up function, domains and related structures.",
      "text": "Find the function, domains and available structures of mouse ABL1 (P00520), with links to the sources."
    },
    {
      "title": "Analyze my data",
      "desc": "Inspect an uploaded table and visualize the results.",
      "text": "Inspect the data I upload, check missing values and units, and suggest useful plots for the measurements."
    },
    {
      "title": "Find a prediction tool",
      "desc": "Choose an available model for your experiment.",
      "text": "Which available tools can predict protein solubility? Compare their inputs and suitable use cases, and help me choose one."
    },
    {
      "title": "Search the literature",
      "desc": "Gather evidence for a protein engineering question.",
      "text": "Find studies on improving PETase stability and activity. Summarize the experimental evidence and link the papers."
    }
  ]
};

interface ChatTimelineProps {
  items: ChatHistoryItem[];
  streamingIndex?: number;
  running?: boolean;
  onSuggestedPrompt?: (text: string) => void;
  sessionId?: string;
  onQuoteReply?: (text: string) => void;
  /** Tool executions grouped into the process preceding each turn's answer. */
  toolExecutions?: Array<Record<string, unknown>>;
  /** Retry the last user message; shown as an icon on the latest assistant
   * message when the run is not in flight. */
  onRetry?: () => void;
  /** When true the retry button is hidden (run still in flight or quota out). */
  retryDisabled?: boolean;
  /** Backend-side security policy denials (kimi tool calls we refused).
   * Rendered as a collapsible warning banner under the timeline. */
  securityEvents?: SecurityEvent[];
}

export function ChatTimeline({ items, streamingIndex = -1, running = false, onSuggestedPrompt, sessionId, onQuoteReply, toolExecutions = [], onRetry, retryDisabled = false, securityEvents = [] }: ChatTimelineProps) {
  const { lang } = useLang();
  const t = useLang().t(TIMELINE_STRINGS);
  const processes = buildConversationProcesses(items, (toolExecutions as ToolExecution[]).filter(tool => !isResearchNoiseTool(tool)), running);
  const processAt = new Map(processes.map(process => [process.anchor, process]));
  const lastAssistantIdx = items.reduce((last, item, idx) => item.role === "assistant" && item.kind !== "model_result" && !isProcessItem(item) && item.content.trim() ? idx : last, -1);
  const renderProcess = (index: number) => {
    const process = processAt.get(index);
    return process ? <ConversationProcessView key={`${sessionId}/${process.key}`} process={process}/> : null;
  };
  const renderMessage = (item: ChatHistoryItem, idx: number) => {
    if (isProcessItem(item)) return null;
    if (item.kind === "model_result") return <ModelResultCard item={item} lang={lang}/>;
    if (item.kind === "compaction" || item.phase === "compaction") {
      return <details className="chat-compaction"><summary>{t.compaction}</summary><div className="chat-compaction-body"><ChatMessageBody html={renderMarkdown((item.content || "").replace(/^🗜️\s*/, "").trim())}/></div></details>;
    }
    const isUser = item.role === "user";
    const rawContent = item.content || "";
    if (!rawContent.trim()) return null;
    const roleId = normalizeRoleId(item.role_id, item.role);
    const roleLabel = roleDisplayName(roleId, isUser);
    const avatar = roleAvatar(roleId, isUser);
    const isStreaming = item.role === "assistant" && idx === streamingIndex;
    const { cleanText, attachments } = isUser || isStreaming
      ? { cleanText: rawContent, attachments: [] as RichAttachment[] }
      : extractRichAttachments(rawContent);
    const structurePaths = isUser || isStreaming ? [] : extractStructurePaths(cleanText);
    const statusCls = isUser ? "" : statusMsgClass(rawContent);
    const isStatus = !isUser && isResearchStatusItem(item);
    const isReport = !isUser && isHighlightReport(item);
    return (
          <div
            className={`chat-msg ${isUser ? "user" : "assistant"} with-avatar${isStreaming ? " streaming" : ""}${statusCls ? ` ${statusCls}` : ""}${isStatus ? " msg-status-muted" : ""}${isReport ? " msg-report-highlight" : ""}`}
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
              <ChatMessageBody html={renderMarkdown(cleanText)} />
              <RichAttachmentList attachments={attachments} />
              {structurePaths.length > 0 && (
                <div className="chat-msg-structures">
                  {structurePaths.map((sp) => (
                    <StructureFileButton
                      key={sp}
                      filePath={sp}
                      label={sp.split("/").pop() || sp}
                    />
                  ))}
                </div>
              )}
              {!isUser && !isStreaming && !isStatus && (sessionId || onQuoteReply || onRetry) && (
                <div className="chat-feedback-buttons">
                  {sessionId && <FeedbackButtons sessionId={sessionId} messageIndex={idx} />}
                  {onQuoteReply && (
                    <QuoteButton content={rawContent} onQuote={onQuoteReply} />
                  )}
                  {onRetry && idx === lastAssistantIdx && (
                    <button
                      type="button"
                      className="chat-msg-retry-btn"
                      onClick={onRetry}
                      disabled={retryDisabled}
                      title={lang === "zh" ? "重新生成" : "Retry this turn"}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 12a9 9 0 1 0 3-6.7" />
                        <path d="M3 4v5h5" />
                      </svg>
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
    );
  };
  return <div className="chat-timeline">
      {items.length === 0 && !running && (
        <div className="chat-empty-state">
          <img
            className="chat-empty-logo"
            src={LOGO_VENUS_URL}
            alt="VenusFactory"
          />
          <h3 className="chat-empty-title">{lang === "zh" ? "从一个研究问题开始" : "Start with a research question"}</h3>
          <div className="chat-suggested-prompts">
            {SUGGESTED_PROMPTS.en.map((prompt, i) => (
              <button
                key={i}
                className="chat-suggested-btn"
                onClick={() => onSuggestedPrompt?.(prompt.text)}
              >
                <span className="chat-suggested-title">{prompt.title}</span>
                <span className="chat-suggested-desc">{prompt.desc}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    {items.map((item, index) => <Fragment key={index}>{renderProcess(index)}{renderMessage(item, index)}</Fragment>)}
    {renderProcess(items.length)}
    {securityEvents.length > 0 && <SecurityEventsBanner events={securityEvents}/>}
  </div>;
}
