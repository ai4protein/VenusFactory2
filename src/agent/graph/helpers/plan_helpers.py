"""Planner-side helpers: parsing, JSON repair, normalization, skill enforcement."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from agent.chat_agent_utils import _parse_cb_plan
from logger import get_logger

_logger = get_logger("agent.graph")


def _parse_sections(raw: str) -> list[dict[str, Any]]:
    try:
        raw = raw.strip()
        if "```" in raw:
            start = raw.find("```")
            if "json" in raw[: start + 10]:
                start = raw.find("```") + 7
            else:
                start = raw.find("```") + 3
            end = raw.find("```", start)
            raw = raw[start:end if end > 0 else None].strip()
        i = raw.find("[")
        if i >= 0:
            depth = 0
            for j in range(i, len(raw)):
                if raw[j] == "[":
                    depth += 1
                elif raw[j] == "]":
                    depth -= 1
                    if depth == 0:
                        raw = raw[i : j + 1]
                        break
        sections_list = json.loads(raw)
        if not isinstance(sections_list, list):
            return []
        parsed = []
        for s in sections_list[:5]:
            sec_name = (s.get("section_name") or "Section").strip() or "Section"
            focus = (s.get("focus") or "both").strip() or "both"
            queries = s.get("search_queries") or s.get("search_query") or []
            if isinstance(queries, str):
                queries = [queries]
            elif not isinstance(queries, list):
                queries = [""]
            parsed.append({"section_name": sec_name, "search_queries": queries, "focus": focus})
        return parsed
    except Exception:
        return []


def _parse_clarification_questions(raw: str) -> list[dict[str, Any]]:
    try:
        raw = raw.strip()
        if "```" in raw:
            start = raw.find("```")
            if "json" in raw[: start + 10]:
                start = raw.find("```") + 7
            else:
                start = raw.find("```") + 3
            end = raw.find("```", start)
            raw = raw[start:end if end > 0 else None].strip()
        i = raw.find("[")
        if i >= 0:
            depth = 0
            for j in range(i, len(raw)):
                if raw[j] == "[":
                    depth += 1
                elif raw[j] == "]":
                    depth -= 1
                    if depth == 0:
                        raw = raw[i : j + 1]
                        break
        questions = json.loads(raw)
        if not isinstance(questions, list):
            return []
        parsed = []
        for q in questions[:4]:
            if not isinstance(q, dict):
                continue
            question = (q.get("question") or "").strip()
            options = q.get("options") or []
            if isinstance(options, str):
                options = [options]
            if not question or not options:
                continue
            parsed.append({
                "question": question,
                "options": [str(o) for o in options],
                "allow_multiple": bool(q.get("allow_multiple", False)),
            })
        return parsed if len(parsed) >= 2 else []
    except Exception:
        return []


def _canonicalize_tool_name(raw_name: Any, available_tools_list: str) -> str:
    """Normalize tool names from LLM output against known available tools."""
    name = str(raw_name or "").strip()
    if not name:
        return ""
    if not available_tools_list:
        return name
    candidates = [x.strip() for x in str(available_tools_list).split(",") if x.strip()]
    if not candidates:
        return name
    lower_map = {c.lower(): c for c in candidates}
    direct = lower_map.get(name.lower())
    if direct:
        return direct
    compact = re.sub(r"[\s\-_]+", "", name.lower())
    for c in candidates:
        if re.sub(r"[\s\-_]+", "", c.lower()) == compact:
            return c
    return name


async def _repair_plan_json_with_llm(llm: Any, raw_content: str) -> list[dict[str, Any]]:
    """Ask model to rewrite planner output into strict JSON array format."""
    if not llm or not raw_content:
        return []
    prompt = (
        "Convert the following pipeline draft into strict JSON array only.\n"
        "Each item must be an object with keys: step, task_description, tool_name, tool_input.\n"
        "Return JSON array only. No markdown fences, no explanation.\n\n"
        f"Draft:\n{str(raw_content)[:10000]}"
    )
    try:
        fixed = await llm.ainvoke([HumanMessage(content=prompt)])
        fixed_text = getattr(fixed, "content", None) or str(fixed) or ""
        parsed = _parse_cb_plan(fixed_text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


async def _retry_plan_for_model_compat(
    llm: Any,
    user_text: str,
    pi_report: str,
    pi_suggest_steps: str,
    protein_context_summary: str,
    available_tools_list: str,
) -> list[dict[str, Any]]:
    """Fallback planner retry for models that frequently return [] despite executable intent."""
    if llm is None:
        return []
    prompt = (
        "You are Computational Biologist.\n"
        "Generate an executable pipeline as JSON array ONLY.\n"
        "No markdown, no explanation.\n"
        "Each item must contain: step(int), task_description(str), tool_name(str), tool_input(object).\n"
        "tool_name must be chosen exactly from AVAILABLE_TOOLS.\n"
        "If user request is actionable, DO NOT return []. Return at least one executable step.\n\n"
        f"USER_REQUEST:\n{user_text}\n\n"
        f"PI_REPORT:\n{pi_report}\n\n"
        f"SUGGEST_STEPS:\n{pi_suggest_steps}\n\n"
        f"PROTEIN_CONTEXT:\n{protein_context_summary}\n\n"
        f"AVAILABLE_TOOLS:\n{available_tools_list}\n"
    )
    try:
        msg = await llm.ainvoke([HumanMessage(content=prompt)])
        content = getattr(msg, "content", None) or str(msg) or ""
        _logger.info("CB planner compat retry output (first 1200 chars): %s", content[:1200])
        parsed = _parse_cb_plan(content)
        _logger.info("CB planner compat retry parsed steps count: %d", len(parsed) if isinstance(parsed, list) else 0)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        _logger.warning("CB planner compat retry failed: %s", e)
        return []


def _format_clarification_answers(questions: list[dict], answers: list[dict]) -> str:
    parts = []
    for i, ans in enumerate(answers):
        if i >= len(questions):
            break
        q = questions[i]
        selected = ans.get("selected_options", [])
        custom = (ans.get("custom_text") or "").strip()
        option_texts = []
        for idx in selected:
            if isinstance(idx, int) and 0 <= idx < len(q.get("options", [])):
                opt = q["options"][idx]
                if opt.lower() not in ("other", "其他"):
                    option_texts.append(opt)
        if custom:
            option_texts.append(custom)
        if option_texts:
            parts.append(f"Q: {q['question']}\nA: {', '.join(option_texts)}")
    return "\n\n".join(parts)


def _looks_like_execution_request(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    exec_keywords = [
        "download", "predict", "identify", "analyze", "analyse", "run", "execute",
        "optimize", "mutation", "stability", "stabilizing", "pipeline", "workflow",
        "design", "dock", "structure", "alphafold", "esm", "protssn", "uniprot",
        "生成", "下载", "预测", "分析", "执行", "运行", "突变", "稳定性", "结构", "流程",
    ]
    return any(k in raw for k in exec_keywords)


def _normalize_step_number(raw_step: Any, fallback: int) -> int:
    try:
        if isinstance(raw_step, str):
            match = re.search(r"\d+", raw_step)
            if match:
                return int(match.group(0))
        if raw_step is not None:
            return int(raw_step)
    except Exception:
        pass
    return fallback


def _extract_skill_ids_from_metadata(skills_metadata: str) -> list[str]:
    if not isinstance(skills_metadata, str) or not skills_metadata.strip():
        return []
    ids = re.findall(r"skill_id:\s*`([^`]+)`", skills_metadata)
    seen = set()
    out: list[str] = []
    for s in ids:
        sid = str(s).strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def _pick_skill_for_code_step(task_desc: str, available_skill_ids: list[str]) -> str | None:
    if not available_skill_ids:
        return None
    lower = (task_desc or "").lower()
    preferred: list[str] = []
    if any(k in lower for k in ("plot", "figure", "visual", "chart", "draw")):
        preferred = ["matplotlib", "seaborn", "biopython"]
    elif any(k in lower for k in ("fasta", "sequence", "pdb", "structure", "mutation")):
        preferred = ["biopython", "matplotlib", "seaborn"]
    else:
        preferred = ["biopython", "matplotlib", "seaborn"]
    for sid in preferred:
        if sid in available_skill_ids:
            return sid
    return available_skill_ids[0]


def _enforce_skill_first_plan(
    normalized_plan: list[dict[str, Any]],
    available_tools_list: str,
    skills_metadata: str,
    ui_lang: str,
) -> list[dict[str, Any]]:
    if not normalized_plan:
        return normalized_plan
    available_tools = {t.strip() for t in str(available_tools_list or "").split(",") if t.strip()}
    if "read_skill" not in available_tools:
        return normalized_plan

    skill_ids = _extract_skill_ids_from_metadata(skills_metadata)
    if not skill_ids:
        return normalized_plan

    code_tools = {"python_repl", "agent_generated_code"}
    existing_steps = [p.get("step") for p in normalized_plan if isinstance(p.get("step"), int)]
    next_aux_step = (max(existing_steps) + 1) if existing_steps else 1
    enforced: list[dict[str, Any]] = []

    for p in normalized_plan:
        tname = str(p.get("tool_name") or "").strip()
        if tname in code_tools:
            prev_is_read_skill = bool(enforced) and str(enforced[-1].get("tool_name") or "").strip() == "read_skill"
            if not prev_is_read_skill:
                chosen_skill = _pick_skill_for_code_step(str(p.get("task_description") or ""), skill_ids)
                if chosen_skill:
                    enforced.append(
                        {
                            "step": next_aux_step,
                            "task_description": (
                                f"加载技能 `{chosen_skill}`，为后续代码执行提供规范接口与参数参考。"
                                if ui_lang == "zh"
                                else f"Load skill `{chosen_skill}` to provide API and parameter guidance for the next code step."
                            ),
                            "tool_name": "read_skill",
                            "tool_input": {"skill_id": chosen_skill},
                        }
                    )
                    next_aux_step += 1
        enforced.append(p)
    return enforced
