import type { ChatHistoryItem } from "./api";
import type { ToolExecution } from "../components/ToolExecutionCard";

export type ToolState = "running" | "complete" | "error" | "stopped";

export function toolState(tool: ToolExecution, live = true): ToolState {
  const status = String(tool.status || "").toLowerCase();
  if (tool.is_error || tool.error || ["error", "failed", "failure", "denied", "blocked"].includes(status)) return "error";
  if (["cancelled", "canceled", "interrupted", "aborted", "stopped"].includes(status)) return "stopped";
  if (["ok", "success", "completed", "complete", "done"].includes(status)) return "complete";
  if (["running", "pending", "queued"].includes(status)) return live ? "running" : "stopped";
  const output = tool.output ?? tool.outputs;
  const hasOutput = output != null && (typeof output !== "string" || output.trim().length > 0);
  if (tool.finished_at || hasOutput) return "complete";
  return live ? "running" : "stopped";
}

export function isResearchStatusItem(item: ChatHistoryItem): boolean {
  if (item.role !== "assistant") return false;
  if (["thinking", "checkpoint", "compaction", "model_result"].includes(item.kind || "") || item.phase === "compaction") return false;
  if (item.phase === "sub_report_checkpoint") return false;
  if (item.kind === "status" || item.phase?.startsWith("research_")) return true;
  const content = (item.content || "").trim();
  if (content.startsWith("📄") || /sub-report .+ complete|小报告完成|decide whether to continue|请.*决定是否继续/i.test(content)) return false;
  return content.startsWith("🔎") || (content.startsWith("🔍") && content.length < 220)
    || (/^\*\*(第\s*\d+|Section\s+\d+)/i.test(content) && content.length < 180)
    || (/^\*\*Query\s+\w+/i.test(content) && content.length < 500);
}

export function isProcessItem(item: ChatHistoryItem): boolean {
  return item.role === "assistant" && (item.kind === "thinking" || item.phase === "thinking" || isResearchStatusItem(item));
}

export type ConversationProcess = {
  key: string;
  anchor: number;
  messages: Array<{ item: ChatHistoryItem; index: number }>;
  tools: ToolExecution[];
  live: boolean;
};

/** A single process per user turn. Tool updates replace their earlier event;
 * missing turn IDs stay visible in the latest turn, including before a reply. */
export function buildConversationProcesses(items: ChatHistoryItem[], tools: ToolExecution[], running: boolean): ConversationProcess[] {
  const groups: ConversationProcess[] = [];
  const byTurn = new Map<string, ConversationProcess>();
  let current: ConversationProcess | undefined;
  items.forEach((item, index) => {
    if (!current || item.role === "user") {
      if (current?.anchor === items.length) current.anchor = index;
      current = { key: `turn-${index}`, anchor: items.length, messages: [], tools: [], live: false };
      groups.push(current);
    }
    if (item.turn_id) byTurn.set(String(item.turn_id), current);
    if (item.role === "assistant" && current.anchor === items.length) current.anchor = index;
    if (isProcessItem(item)) current.messages.push({ item, index });
  });
  if (!current) {
    current = { key: "turn-pending", anchor: items.length, messages: [], tools: [], live: false };
    groups.push(current);
  }
  current.live = running;
  const merged: ToolExecution[] = [];
  const byCall = new Map<string, number>();
  for (const tool of tools) {
    const id = String(tool.tool_call_id || "");
    const previous = id ? byCall.get(id) : undefined;
    if (previous != null) {
      const old = merged[previous];
      merged[previous] = { ...old, ...tool, turn_id: tool.turn_id || old.turn_id };
    } else {
      if (id) byCall.set(id, merged.length);
      merged.push(tool);
    }
  }
  for (const tool of merged) (byTurn.get(String(tool.turn_id || "")) || current).tools.push(tool);
  return groups.filter(group => group.messages.length || group.tools.length || group.live);
}

const labels: Record<string, [string, string]> = {
  ask_user: ["Ask a question", "询问信息"],
  request_confirmation: ["Request confirmation", "请求确认"],
  search_generated_capabilities: ["Search model library", "检索模型库"],
  describe_generated_capability: ["Inspect model", "查看模型"],
  run_generated_capability: ["Run model", "运行模型"],
  search_tools: ["Find tools", "查找工具"],
  search_tool: ["Find tools", "查找工具"],
  read_file: ["Read file", "读取文件"],
  write_file: ["Write file", "写入文件"],
  execute_python: ["Run Python", "运行 Python"],
  run_python: ["Run Python", "运行 Python"],
  search_pubmed: ["Search literature", "检索文献"],
  web_search: ["Search the web", "检索网页"],
};

export function toolActivityLabel(tool: ToolExecution, zh: boolean): string {
  const raw = String(tool.tool_name || tool.name || "tool");
  const name = raw.split(/__|[.:/]/).pop() || raw;
  if (labels[name]) return labels[name][zh ? 1 : 0];
  const text = name.replace(/[_-]+/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}
