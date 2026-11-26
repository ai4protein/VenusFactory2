import test from "node:test";
import assert from "node:assert/strict";
import { checkpointForSession, checkpointPlaceholder } from "../src/pages/chat/conversationCheckpoint.ts";
import { chatMessages } from "../src/pages/chat/chatMessages.ts";
import { conversationHistory } from "../src/pages/chat/conversationHistory.ts";

test("a restored question takes precedence over a stale approval and the selected mode", () => {
  const gate = checkpointForSession({ engine: "kimi-code", waiting_for: "kimi_approval", kimi_pending_question: {} }, "science_expert");
  assert.equal(gate, "ask");
  assert.equal(checkpointPlaceholder("approve", true, chatMessages.en), chatMessages.en.composerWaitingAskUser);
});

test("an older expert snapshot restores its checkpoint from status", () => {
  assert.equal(checkpointForSession({ status: "waiting_for_sub_report_review" }, "science_agent"), "subReport");
  assert.equal(checkpointForSession({ status: "waiting_for_clarification", waiting_for: "another_action" }, "science_expert"), null);
  assert.equal(checkpointForSession(null, "science_agent"), null);
});

test("same-ID chats and reports retain separate selections, titles and activity", () => {
  const input = {
    sessions: [{ session_id: "shared", chat_mode: "science_agent", title: "Chat", status: "completed" }],
    reports: [{ id: "shared", title: "Report", created: 42, status: "running" }], activity: {},
    activeId: "shared", activeTitle: "Draft title", agentRunId: "", scientistRunId: "", online: true, untitled: "New Chat",
  };
  const chat = conversationHistory({ ...input, activeMode: "science_agent" });
  const report = conversationHistory({ ...input, activeMode: "researcher" });
  assert.notEqual(chat.activeKey, report.activeKey);
  assert.deepEqual(report.rows.map(row => [row.title, row.running]), [["Chat", false], ["Report", true]]);
  assert.equal(chat.rows[0].title, "Draft title");
});
