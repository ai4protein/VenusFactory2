# Agent: Your AI Assistant for Protein Engineering

**Agent** is the conversational entry point of VenusFactory2. Describe what you want in plain English (or Chinese), and Agent plans a multi-step workflow, runs the right tools, reviews intermediate results with you, and assembles a final report.

---

## 1. When to Use the Agent

| Scenario | Why Agent fits |
| :--- | :--- |
| You want a multi-step analysis without wiring tools manually | Agent plans the pipeline for you |
| You need follow-up questions on the same protein | The conversation keeps full context |
| You want to inspect / edit the plan before execution | You can reorder, edit, or remove steps |
| You want to checkpoint long runs | After each step / sub-report you can continue, rewrite, or abort |

For a *single* prediction with no orchestration, **Quick Tools** is faster.
For full control over model + dataset selection, use **Advanced Tools**.

---

## 2. Interface Layout

The chat workspace has three columns:

| Column | What's there |
| :--- | :--- |
| **Left — Sessions** | New chat, session list, delete, copy session ID. Each session keeps its own history and uploaded files. |
| **Center — Conversation** | Message timeline, model picker, file attach (browser upload or Workspace picker), Regenerate / Export / Stop / Send. Inline checkpoint widgets appear here when Agent needs your input. |
| **Right — Execution Status** | Live status, last 12 tool runs, tail of the conversation log. Useful for watching long runs. |

In **online** mode, a quota pill shows your remaining daily chats; once exhausted, Send / Regenerate / file upload are disabled.

---

## 3. The Conversation Flow

A typical run cycles through five phases. Agent surfaces each one as a dedicated checkpoint card, so you stay in control without micromanaging.

```
You ─▶ Clarification ─▶ Plan ─▶ Execution ─▶ Sub-report ─▶ Iteration ─▶ Final report
```

### 3.1 Clarification

Agent may ask a few short questions before planning — multiple-choice or text. Pick the option that matches your intent (or use the "Other" field). Submit to continue.

> **Tip:** You can use `/loop` `/skip-research` style hints in your initial message to bypass clarification on simple tasks.

### 3.2 Plan

Agent proposes an ordered list of steps. Each step has a tool name and a short task description. In the **Plan Editor** you can:

- Edit any step's description
- Reorder with ↑ / ↓
- Remove a step (min 1 required)
- Toggle **auto-execute** before clicking Confirm

### 3.3 Execution

Steps run sequentially. For each step you may see:

- **Step checkpoint:** continue / abort the pipeline
- A streaming log of the tool call in the right panel

### 3.4 Sub-report Checkpoint

After each analytical step, Agent generates a short sub-report and waits for one of three decisions:

- **Continue Research** — accept and move on
- **Comment & Rewrite** — leave a comment; Agent will redo the sub-report with your feedback
- **Skip to Report** — drop this branch and head to the final summary

### 3.5 Iteration

After the main pipeline finishes you get a final decision:

- **Satisfied** — finalize the report
- **Modify & Re-execute** — go back to planning with edits
- **Continue Analysis** — start a follow-up with new instructions

The final report is exported as a structured bundle — both **HTML** and **PDF** are downloadable via the Export button.

---

## 3.6 Example Prompts

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

Above the input bar you can switch the LLM that drives the agent:

- **Built-in:** Gemini 2.5 Pro, GPT-4o, Claude 3.7, DeepSeek-R1
- **Custom OpenAI-style:** Add your own endpoint (display name, model name, API key, base URL). Saved in your browser; **local mode only**.

If you switch models mid-conversation, you'll see a notice — model behavior may differ.

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
