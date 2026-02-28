"""Load role prompts from src/agent/prompts/*.md and expose ChatPromptTemplates.
Multi-agent roles and avatars: img/agent_role/{role_id}.png.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_PROMPTS_DIR = Path(__file__).parent / "prompts"
# Project root: src/agent/prompts.py -> src/agent -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_AVATAR_DIR = _PROJECT_ROOT / "img" / "agent_role"

# Multi-agent role IDs and avatar filenames (under img/agent_role/)
# No separate "chat_system": the agent in charge decides which role's prompt to use.
ROLE_AVATAR_FILES: Dict[str, str] = {
    "principal_investigator": "principal_investigator.png",
    "machine_learning_specialist": "machine_learning_specialist.png",
    "computational_biologist": "computational_biologist.png",
    "scientific_critic": "scientific_critic.png",
}

ROLE_DISPLAY_NAMES: Dict[str, str] = {
    "principal_investigator": "Principal Investigator",
    "machine_learning_specialist": "Machine Learning Specialist",
    "computational_biologist": "Computational Biologist",
    "scientific_critic": "Scientific Critic",
}


def _load_md(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def get_role_avatar_path(role_id: str, fallback_to_first: bool = True) -> Optional[str]:
    """Return absolute path to role avatar under img/agent_role. Use for multi-agent UI."""
    filename = ROLE_AVATAR_FILES.get(role_id)
    if not filename and fallback_to_first and ROLE_AVATAR_FILES:
        filename = next(iter(ROLE_AVATAR_FILES.values()))
    if not filename:
        return None
    path = _AGENT_AVATAR_DIR / filename
    return str(path) if path.exists() else None


# --- Principal Investigator (Planner) ---
PLANNER_SYSTEM_PROMPT_TEMPLATE = _load_md("principal_investigator")

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# --- PI report: same principal_investigator prompt; invoke with input = search results + "Output report" → StrOutputParser ---
# (Report mode is defined in principal_investigator.md: "WHEN TO OUTPUT A RESEARCH REPORT")

# --- PI Research Phase: PI runs search tools itself (multi-turn), then outputs report; used by create_pi_research_agent ---
PI_RESEARCH_SYSTEM_TEMPLATE = _load_md("principal_investigator_research")
PI_RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PI_RESEARCH_SYSTEM_TEMPLATE),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# --- PI 5-section flow: plan sections → sub-report → draft → Suggest steps (for CB/MLS) ---
PI_SECTIONS_TEMPLATE = _load_md("principal_investigator_sections")
PI_SECTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PI_SECTIONS_TEMPLATE),
    ("human", "Output only the JSON array of sections (max 5). No other text."),
])
PI_SUB_REPORT_TEMPLATE = _load_md("principal_investigator_sub_report")
PI_SUB_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PI_SUB_REPORT_TEMPLATE),
    ("human", "Section: {section_name} (focus: {focus}).\n\nSearch results:\n{search_results}\n\nWrite a long sub-report (3–5 paragraphs) that reads and analyzes each reference [1], [2], [3] with at least 1–2 sentences each, then synthesizes. Use two sections at the same level: ## Sub-report (prose), then ## References (one reference per line, format [n] [Title](URL)). Output only the sub-report:"),
])
PI_FINAL_REPORT_TEMPLATE = _load_md("principal_investigator_final_report")
PI_FINAL_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PI_FINAL_REPORT_TEMPLATE),
    ("human", "References and sub-reports by section:\n{sub_reports}\n\nUser question: {input}\n\nOutput only the research draft with ## Abstract, ## Introduction, ## Related Work, and ## References. In ## References use one line per reference (line break after each [n] [Title](URL)); do not put multiple references on the same line. Use the same language as the user."),
])
# --- PI Suggest steps: draft + user input → Tools + Steps for CB/MLS (separate message) ---
PI_SUGGEST_STEPS_TEMPLATE = _load_md("principal_investigator_suggest_steps")
PI_SUGGEST_STEPS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PI_SUGGEST_STEPS_TEMPLATE),
    ("human", "Draft:\n{draft_report}\n\nUser question: {input}\n\nOutput only the ## Suggest steps section (Tools + Steps) for CB/MLS:"),
])

# --- CB Pipeline Planner: same computational_biologist prompt (Mode A); invoke with pi_report, pi_suggest_steps, ... ---
CB_PLANNER_SYSTEM = _load_md("computational_biologist")
CB_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CB_PLANNER_SYSTEM),
    ("human", "Turn the PI report and Suggest steps above into a concrete pipeline. Output only the JSON array."),
])

# --- Machine Learning Specialist ---
ML_WORKER_SYSTEM_TEMPLATE = _load_md("machine_learning_specialist")

ML_WORKER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ML_WORKER_SYSTEM_TEMPLATE),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# --- Computational Biologist ---
CB_WORKER_SYSTEM_TEMPLATE = _load_md("computational_biologist")

CB_WORKER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CB_WORKER_SYSTEM_TEMPLATE),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Backward compatibility: default worker is CB (tool selection + pipeline execution)
WORKER_PROMPT = CB_WORKER_PROMPT

# Four roles: PI, MLS, CB, SC (alias for use in chat_agent)
PI_PROMPT = PLANNER_PROMPT
MLS_PROMPT = ML_WORKER_PROMPT
CB_PROMPT = CB_WORKER_PROMPT

# --- Scientific Critic (SC): one prompt file for synthesis and direct chat ---
SC_SYSTEM = _load_md("scientific_critic")
FINALIZER_PROMPT = ChatPromptTemplate.from_template(SC_SYSTEM)
SC_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SC_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# PI answer with citations: same PI prompt, used with StrOutputParser when plan is empty (no tools).
# Invoke with input = message that includes "Literature results" block + "answer with citations"; no JSON.
SC_PROMPT = FINALIZER_PROMPT


def get_multi_agent_roles() -> List[Dict[str, Any]]:
    """Return config for each multi-agent role: role_id, name, system_prompt, avatar_path, prompt.
    Avatars are loaded from img/agent_role/{role_id}.png when present.
    """
    return [
        {
            "role_id": "principal_investigator",
            "name": ROLE_DISPLAY_NAMES["principal_investigator"],
            "system_prompt": PLANNER_SYSTEM_PROMPT_TEMPLATE,
            "avatar_path": get_role_avatar_path("principal_investigator"),
            "prompt": PLANNER_PROMPT,
        },
        {
            "role_id": "machine_learning_specialist",
            "name": ROLE_DISPLAY_NAMES["machine_learning_specialist"],
            "system_prompt": ML_WORKER_SYSTEM_TEMPLATE,
            "avatar_path": get_role_avatar_path("machine_learning_specialist"),
            "prompt": ML_WORKER_PROMPT,
        },
        {
            "role_id": "computational_biologist",
            "name": ROLE_DISPLAY_NAMES["computational_biologist"],
            "system_prompt": CB_WORKER_SYSTEM_TEMPLATE,
            "avatar_path": get_role_avatar_path("computational_biologist"),
            "prompt": CB_WORKER_PROMPT,
        },
        {
            "role_id": "scientific_critic",
            "name": ROLE_DISPLAY_NAMES["scientific_critic"],
            "system_prompt": SC_SYSTEM,
            "avatar_path": get_role_avatar_path("scientific_critic"),
            "prompt": SC_CHAT_PROMPT,
        },
    ]
