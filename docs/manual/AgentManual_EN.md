# Agent: Your AI Assistant for Protein Engineering

**Agent** is the conversational entry point of VenusFactory2. Describe what you want in plain English (or Chinese), and Agent plans a multi-step workflow, runs the right tools, reviews intermediate results with you, and assembles a final report.

---

## 1. When to Use the Agent

| Scenario | Why Agent fits |
| :--- | :--- |
| You want a multi-step analysis without wiring tools manually | Agent plans the pipeline for you |
| You need follow-up questions on the same protein | The conversation keeps full context |
| You want to inspect / edit the plan before execution | You can reorder, edit, or remove steps (Science Expert) |
| Long Expert runs with optional manual gates | Plan is still confirmed in Plan Editor; steps and sub-reports auto-advance by default (use step-by-step plan confirm only if you need per-step pauses) |

For a *single* prediction with no orchestration, **Quick Tools** is faster.
For full control over model + dataset selection, use **Advanced Tools**.

---

## 2. Interface Layout

The chat workspace has three columns:

| Column | What's there |
| :--- | :--- |
| **Left — Sessions** | New chat, session list, delete, copy session ID. Each session keeps its own history and uploaded files. |
| **Center — Conversation** | Message timeline, **Science Agent / Science Expert** mode switch, model picker (**Local Expert only**; Online Expert uses a fixed backend model with no selector), file attach (browser upload or Workspace picker), Regenerate / Export / Stop / Send. Expert checkpoint widgets (plan, clarification, iteration, optional step/sub-report) appear inline when needed. |
| **Right — Execution Status** | Live status, last 12 tool runs, tail of the conversation log. Useful for watching long runs. |

In **online** mode, a quota pill shows your remaining daily chats; once exhausted, Send / Regenerate / file upload are disabled.

---

## 3. Two Chat Modes

Use the segmented control above the composer to pick a mode. The choice is saved in your browser; switching mid-session may change behavior — start a new session if results look inconsistent.

### 3.1 Science Agent (kimi-code)

- **Engine:** local **kimi-code** daemon (not the LangGraph PI/CB/MLS/SC graph).
- **Behavior:** streaming tool use (VenusFactory MCP, shell where allowed), collapsible **Thinking** blocks, inline tool-run cards in the timeline.
- **Context (aligned with [Kimi Code sessions](https://www.kimi.com/code/docs/en/kimi-code-cli/guides/sessions.html)):**
  - Auto-compresses when context usage ≥ ~90%; timeline shows compaction summaries.
  - Read-only context usage pill only — no Plan / Compact / Fork / Clear composer buttons (agent manages context itself).
  - Stop aborts the in-flight kimi prompt when possible.
- **Model:** fixed to kimi-code; configure providers via Settings / `.env` (see WebUI wiki). Online mode may use `bwrap` sandboxing.
- **When to use:** open-ended tasks where you want a single agent to call tools fluidly without the Expert planning pipeline.

### 3.2 Science Expert (LangGraph)

- **Engine:** LangGraph **`graph`** pipeline with four roles — **PI** (plan / research) → **CB** (concrete step plan) → **MLS** (execution checks) → **SC** (review).
- **PI first:** every new run starts with PI analysis / clarification (including an explicit Skip Research option). No silent keyword jump straight to CB.
- **Plan gate:** you still review and confirm the plan in **Plan Editor** (edit, reorder, delete steps).
- **Flow defaults (fewer stops):**
  - **After plan confirm:** prefer **Confirm & Auto-execute** (recommended; backend default). Steps run without a pause after each tool step. Choose **Confirm Plan** only if you want per-step checkpoints.
  - **Sub-reports:** when literature sections do run, the pipeline auto-advances by default instead of stopping on every section; you can still rewrite or skip from the checkpoint if one appears.
  - **Final SC report:** paper-level manuscript (~5000–8000 words), not a short 800–1500 word brief.
- **Model:** **Local** — pick any graph-engine LLM in the model selector (built-in or custom OpenAI-style). **Online** — no model selector; the platform uses a fixed backend model.

Expert runs still move through the phases below; not every phase appears on every task.

```
You ─▶ Clarification ─▶ Plan ─▶ Execution ─▶ Sub-report ─▶ Iteration ─▶ Final report
```

#### Clarification

Agent may ask a few short questions before planning — multiple-choice or text. Pick the option that matches your intent (or use the "Other" field). Submit to continue.

> **Tip:** There are no `/skip-research` / `/loop` slash commands. Use **Skip research** in the clarification form, or send an execution-style request (tool run without literature intent) — the pipeline may skip research automatically.

#### Plan

Agent proposes an ordered list of steps. Each step has a tool name and a short task description. In the **Plan Editor** you can:

- Edit any step's description
- Reorder with ↑ / ↓
- Remove a step (min 1 required)
- **Confirm & Auto-execute** (default path) or **Confirm Plan** for step-by-step gates

#### Execution

Steps run sequentially. With auto-execute, the right panel streams tool logs without a step checkpoint. If you confirmed without auto-execute, each step may show **continue / abort**.

#### Sub-report (literature only)

When research sections run, short sub-reports may appear; by default the run continues without waiting on every section. If a checkpoint is shown, you can **Continue Research**, **Comment & Rewrite**, or **Skip to Report**.

#### Iteration

After the main pipeline finishes you get a final decision:

- **Satisfied** — finalize the report
- **Modify & Re-execute** — go back to planning with edits
- **Continue Analysis** — start a follow-up with new instructions

The final report is exported as a structured bundle — both **HTML** and **PDF** are downloadable via the Export button.

### 3.3 Example Prompts

The Agent is built around natural-language, goal-driven requests. Some patterns that the planner handles well:

| Workflow | Example prompt | Tools the planner typically uses |
| :--- | :--- | :--- |
| **Function / structural localization** | *"What is the likelihood of this protein being nuclear?"* — *"Where is the DNA binding site on this zinc finger protein?"* | Protein Function (Localization, Sorting Signal); Functional Residue (Binding Site) |
| **Rational design & optimization** | *"Find the best 5 mutations to increase both activity and stability."* — *"This sequence has an Instability Index > 40; recommend 3 stabilizing mutations."* | Directed Evolution (Activity, Stability); Physical & Chemical Properties |
| **Sequence & structure retrieval** | *"My experiment uses UniProt ID P05798; retrieve its sequence and structure."* — *"Use InterPro to query the domain structure of this sequence."* | InterPro / UniProt download tools |
| **Integrated analysis** | *"Analyze the solubility and thermostability of this sequence, and give experimental protocol recommendations based on the results."* | Multiple prediction tools chained through the Analysis module |
| **Complex task decomposition** | *"Analyze the stability of this sequence (P60002.fasta), and find the top 5 mutations that will increase stability."* | Planner splits into: Function Prediction (Stability) → Directed Evolution → Analysis |
| **Conditional logic** | *"If the Instability Index is greater than 40, recommend 5 stabilizing mutations."* | Physical & Chemical Properties first; Mutation Prediction only fires if the threshold is met |
| **Report synthesis** | *"Summarize the top 3 results from the mutation scan and explain why they are predicted to be stable."* | Analysis module synthesizing raw data into natural-language |

> **Tip:** the more concrete your goal, the better the plan. *"Find mutations"* is vague; *"find 5 mutations that improve thermostability without hurting activity"* is actionable.

---

## 4. Attaching Data

The paperclip / Workspace icons next to the input bar give you two paths:

| Path | Use when |
| :--- | :--- |
| **Upload** | A fresh file from your machine (FASTA, PDB, CSV, TXT). Auto-categorised by extension. |
| **Workspace** | Files you previously uploaded or that tools produced earlier; pick from a searchable list. |

Workspace is **local-mode only**. In online mode the picker is disabled.

---

## 5. Model Picker

**Science Expert (Local):** above the input bar, switch the LLM that drives the LangGraph pipeline:

- **Built-in:** Gemini 2.5 Pro, GPT-4o, Claude 3.7, DeepSeek-R1
- **Custom OpenAI-style:** Add your own endpoint (display name, model name, API key, base URL). Saved in your browser; **local mode only**.

**Science Expert (Online):** no model selector — the platform pins a fixed backend model (client cannot change it).

**Science Agent:** uses kimi-code only; the selector shows a fixed Agent pill instead of graph models.

If you switch models mid-conversation (Local Expert), you'll see a notice — model behavior may differ.

---

## 6. Best Practices

- **State the goal, not the buttons.** "Find 5 stabilizing mutations for this enzyme" beats "run mutation tool and sort."
- **Reuse context.** Within a session, refer back to "the sequence" rather than re-attaching.
- **Trust but review.** The Plan Editor lets you fix bad plans before you spend tokens / compute.
- **Save what matters.** Agent does not retain server-side history beyond the session — download the final report or push files to Workspace before closing.
- **Watch the right panel.** When a step looks stuck, the live tool log usually shows whether it's actually downloading data, calling a model, or hung.

---

## 7. Online vs Local Mode

| Capability | Local | Online |
| :--- | :---: | :---: |
| Built-in model providers | ✓ | ✓ |
| Custom OpenAI-style models | ✓ | — |
| Workspace upload / replace / delete | ✓ | — (view-only) |
| Daily chat quota | unlimited | enforced (see quota pill) |

A small badge in the sidebar always shows the current runtime mode.
