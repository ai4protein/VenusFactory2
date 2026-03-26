## Agent: Your AI Assistant for Protein Engineering

**Agent** is an AI-powered conversation interface that lets you analyze proteins using natural language. Instead of clicking through multiple tools and tabs, simply describe what you want to do in plain English (or Chinese), and Agent will automatically select the right models, run the analyses, and synthesize the results for you.

---

## 1. What Can Agent Do For You?

Agent is designed to help you complete protein engineering tasks quickly without needing to learn complex tool configurations:

- **Ask questions in plain language**: "What is the solubility of this protein?" or "Find mutations to improve stability"
- **Upload your data easily**: Paste sequences, upload FASTA/PDB files, or provide UniProt IDs
- **Get comprehensive answers**: Agent automatically combines predictions from multiple models and explains the results in simple terms
- **Work conversationally**: Ask follow-up questions without re-uploading data - Agent remembers your conversation context
- **Save time**: Instead of running multiple separate analyses, Agent does it all in one conversation

---

## 2. Interface Overview

The Agent interface is a clean chat environment designed for fluid conversational interaction.

| Element | Description |
| :--- | :--- |
| **New Chat** Button | Starts a new conversation thread, clearing the previous context. |
| **Delete Chat** Button | Permanently removes the currently selected conversation record. |
| **Conversations** List | Displays a history of past chat sessions, allowing the user to resume an interrupted workflow. |
| **Chat Input Bar** | The primary area for inputting requests, questions, or commands. |
| **Paperclip/Attachment Icon** | Allows the user to upload **data files** (FASTA, PDB) directly into the chat context for AI analysis. |

---

## 3. Usage Guide: Conversational Workflows

The Agent is skilled at understanding and executing **goal-driven** tasks common in protein engineering, planned intelligently by the internal Agent architecture.

### 3.1 Core Workflow Capabilities

| Workflow Category | Example Tasks Handled by the Agent | Corresponding Core Models |
| :--- | :--- | :--- |
| **Function/Structural Localization** | "What is the likelihood of this protein being nuclear?" "Where is the **DNA binding site** on this zinc finger protein?" | **Protein Function** (Localization, Sorting Signal); **Functional Residue ** (Binding Site) |
| **Rational Design & Optimization** | "Find the best 5 mutations to increase both **activity** and **stability**." "This sequence has an **Instability Index** > 40; recommend 3 stabilizing mutations." | **Directed Evolution** (Activity, Stability); **Physical & Chemical Properties** |
| **Sequence & Structure Retrieval** | "My experiment uses UniProt ID P05798; retrieve its sequence and structure." "Use InterPro to query the **domain structure** of this sequence." | **InterPro Query**; **UniProt Query** |
| **Integrated Analysis** | "Analyze the solubility and thermostability of this sequence, and give experimental protocol recommendations based on the results." | Automatically integrates multiple prediction models and runs the **Analysis module**. |

### 3.2 Advanced Workflow Commands

| Command Type | Example Request | Agent Action (Planned and Executed) |
| :--- | :--- | :--- |
| **Complex Task Decomposition** | "Analyze the stability of this sequence (P60002.fasta), and find the top 5 mutations that will **increase stability**." | **Planner** breaks the request down: 1. Run Function Prediction (Stability); 2. Run Directed Evolution (Workers execute); 3. Run the Analysis module for result sorting and summary. |
| **Conditional Logic** | "If the **Instability Index** is greater than 40, recommend 5 stabilizing mutations." | Runs **Physicochemical Analysis** and only proceeds to **Mutation Prediction** if the condition is met. |
| **Report Synthesis** | "Summarize the top 3 results from the mutation scan and explain why they are predicted to be stable." | Executes the **Analysis module** to synthesize and interpret raw data into a concise, natural-language response. |

---

## 4. Best Practices and Important Note

- **Be Goal-Oriented**: State your ultimate experimental goal directly (e.g., "I need a variant with better solubility") rather than listing buttons to press.
- **Leverage Memory**: Within the same chat, you do not need to re-upload the sequence; simply refer to it in subsequent requests.
- **Trust the Planner**: Submit your high-level objective, and allow the internal **Planner** to automatically orchestrate the required prediction and analysis steps for you.
- **File Context**: Always upload the necessary **PDB file** before requesting structure-based predictions.
- **Important Note**: **Agent does not retain permanent user data or conversation memory.** When you close your browser or start a new chat, the previous dialogue and context will be cleared. **Please be sure to save** any important conversation records and prediction results externally.