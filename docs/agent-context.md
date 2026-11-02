# Agent conversation context

Agent automatically compresses earlier turns into working memory before the next request. Kimi Code, DeepSeek Harness, Codex and Claude Code share this host-side policy. Existing legacy Kimi sessions also use the configured trigger for their native compaction operation.

The original conversation log is retained. A private, content-addressed archive in the session workspace retains the full older message groups, including complete tool call/result pairs. The summary carries goals, constraints, protein identifiers, results with provenance, file references and pending work. Recent turns remain verbatim when they fit; oversized turns are summarized earlier. The latest user request is never silently truncated.

Configure under local Settings → Environment variables → Usage limits → Agent context, or in the server `.env`:

| Setting | Default | Meaning |
|---|---:|---|
| `WEBUI_V2_AGENT_CONTEXT_TOKENS` | 32000 | Conversation text budget |
| `WEBUI_V2_AGENT_RECENT_TURNS` | 8 | Maximum prior turns retained verbatim |
| `WEBUI_V2_AGENT_COMPACT_PERCENT` | 80 | Token-budget compression trigger |
| `WEBUI_V2_AGENT_SUMMARY_TOKENS` | 2000 | Working-memory summary budget |
| `WEBUI_V2_AGENT_MAX_TURNS` | 0 | User turns per conversation; 0 is unlimited |

Restart the relevant service after editing environment settings. Online users cannot edit these settings. The turn limit is separate from daily conversation allowance and from per-turn model/tool-call budgets. Retrying the last turn does not consume another user turn; new accepted turns do, including interrupted turns.

Text accounting uses conservative UTF-8 byte estimates, not billing token counts. The conversation budget is clamped to the selected model capacity with room for output, native instructions and tool definitions. Native harnesses continue to manage their own in-turn tool context. This setting controls host-supplied conversation memory, not an exact tokenizer-independent cap on the entire wire request. Large single inputs should be attached as files.

Compaction uses one request to the same admitted free model and relay, counts against the existing model-call budget, and never switches to a paid provider. Internal summary text is not streamed into the user's answer. Cancellation stops compaction. A failed summary leaves the previous working history intact so a retry can recover. Summaries are lossy; exact older details can be checked in the retained archive.
