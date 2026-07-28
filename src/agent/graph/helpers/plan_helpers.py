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
        # Cap sections + queries so Expert research cannot flood the timeline
        # (3 sections × 3 queries × 6 sources was drowning sub-reports).
        for s in sections_list[:3]:
            sec_name = (s.get("section_name") or "Section").strip() or "Section"
            focus = (s.get("focus") or "both").strip() or "both"
            queries = s.get("search_queries") or s.get("search_query") or []
            if isinstance(queries, str):
                queries = [queries]
            elif not isinstance(queries, list):
                queries = [""]
            cleaned = [str(q).strip() for q in queries if str(q).strip()][:2]
            if not cleaned:
                cleaned = [sec_name]
            parsed.append({
                "section_name": sec_name,
                "search_queries": cleaned,
                "focus": focus,
            })
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
    """Fallback planner retry for models that frequently return [] despite executable intent.

    Robust to reasoning-style models (DeepSeek-V4-Pro, GLM-4.6): if ``content``
    comes back empty but ``additional_kwargs.reasoning_content`` is populated,
    parse the reasoning text (the model often embeds the JSON there). If still
    empty, do one more attempt with an explicit "do NOT reason, emit JSON
    directly" instruction.
    """
    if llm is None:
        return []
    prompt = (
        "You are Computational Biologist.\n"
        "Generate an executable pipeline as JSON array ONLY.\n"
        "No markdown, no explanation, no chain-of-thought.\n"
        "Each item must contain: step(int), task_description(str), tool_name(str), tool_input(object).\n"
        "tool_name must be chosen exactly from AVAILABLE_TOOLS.\n"
        "If user request is actionable, DO NOT return []. Return at least one executable step.\n\n"
        f"USER_REQUEST:\n{user_text}\n\n"
        f"PI_REPORT:\n{pi_report}\n\n"
        f"SUGGEST_STEPS:\n{pi_suggest_steps}\n\n"
        f"PROTEIN_CONTEXT:\n{protein_context_summary}\n\n"
        f"AVAILABLE_TOOLS:\n{available_tools_list}\n"
    )

    def _extract_content(msg: Any) -> str:
        content = getattr(msg, "content", None) or ""
        if str(content).strip():
            return str(content)
        extra = getattr(msg, "additional_kwargs", None) or {}
        if isinstance(extra, dict):
            rc = str(extra.get("reasoning_content") or "")
            if rc.strip():
                return rc
        return str(msg) or ""

    try:
        msg = await llm.ainvoke([HumanMessage(content=prompt)])
        content = _extract_content(msg)
        _logger.info("CB planner compat retry output (first 1200 chars): %s", content[:1200])
        parsed = _parse_cb_plan(content)
        _logger.info("CB planner compat retry parsed steps count: %d", len(parsed) if isinstance(parsed, list) else 0)
        if isinstance(parsed, list) and parsed:
            return parsed

        # Second attempt: be even more explicit about no-reasoning.
        prompt2 = (
            "Output ONLY a JSON array. No markdown, no comments, no <think> tags, "
            "no reasoning prefix. Start your response with `[` and end with `]`.\n\n"
            + prompt
        )
        msg2 = await llm.ainvoke([HumanMessage(content=prompt2)])
        content2 = _extract_content(msg2)
        _logger.info("CB planner compat retry #2 output (first 1200 chars): %s", content2[:1200])
        parsed2 = _parse_cb_plan(content2)
        _logger.info("CB planner compat retry #2 parsed steps count: %d", len(parsed2) if isinstance(parsed2, list) else 0)
        return parsed2 if isinstance(parsed2, list) else []
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


def _looks_like_research_request(text: str) -> bool:
    """True when the user clearly wants literature / survey-style research."""
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    research_keywords = [
        "literature", "review", "survey", "paper", "papers", "pubmed", "preprint",
        "cite", "citation", "references", "related work",
        "调研", "文献", "综述", "论文", "查阅", "检索文献", "相关工作", "背景调研",
    ]
    return any(k in raw for k in research_keywords)


def _looks_like_execution_request(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    exec_keywords = [
        "download", "predict", "identify", "analyze", "analyse", "run", "execute",
        "optimize", "mutation", "stability", "stabilizing", "pipeline", "workflow",
        "design", "dock", "structure", "alphafold", "esm", "protssn", "uniprot",
        "generate", "compute", "score", "fold", "sequence", "fasta", "pdb",
        "string", "interaction partner", "protein atlas", "hpa", "tissue expression",
        "function of protein", "zero-shot", "directed evolution",
        # Obvious in-platform / common tool-library names (research keywords still win).
        "venusrem", "venusplm", "venusfactory", "foldseek", "esmfold", "colabfold",
        "interpro", "blast", "proteinmpnn", "rfdiffusion", "diffdock", "rosetta",
        "生成", "下载", "预测", "分析", "执行", "运行", "突变", "稳定性", "结构", "流程",
        "计算", "打分", "折叠", "序列", "工具", "相互作用", "组织表达", "蛋白功能",
    ]
    return any(k in raw for k in exec_keywords)


def _should_skip_research_phase(text: str) -> bool:
    """Execution-shaped asks without research intent skip PI clarification+search."""
    return _looks_like_execution_request(text) and not _looks_like_research_request(text)


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


# Protein-engineering domain rules for auto-inserting read_skill before code steps.
# Order matters: first matching keyword group wins. Domain/workflow rules MUST
# precede generic plot/visual rules so "STRING network visualization" does not
# steal nature_figure over string_database.
_DOMAIN_SKILL_RULES: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("润色", "polish", "academic english", "manuscript polish"),
        ["nature_polishing", "nature_writing"],
    ),
    (
        ("写论文", "写摘要", "写引言", "manuscript", "abstract", "introduction draft"),
        ["nature_writing", "nature_polishing"],
    ),
    (
        ("zero-shot", "zero shot", "突变预测", "beneficial mutation", "mutation prediction", "定向进化", "protssn", "venusplm"),
        ["zero_shot_mutation_workflow", "biopython"],
    ),
    (
        ("proteinmpnn", "inverse fold", "sequence design", "固定骨架", "de novo"),
        ["proteinmpnn_design_workflow", "biopython"],
    ),
    (("uniprot", "swiss-prot", "uniprotkb"), ["uniprot_database", "biopython"]),
    (("string", "ppi", "interaction partner", "蛋白互作", "interaction network"), ["string_database"]),
    (("rcsb", "pdb id", "experimental structure", "crystal structure"), ["rcsb_database", "protein_structure_pipeline"]),
    (("alphafold", "plddt", "pae"), ["alphafold_database", "protein_structure_pipeline"]),
    (("foldseek", "structural similar", "结构相似"), ["foldseek_structural_similarity", "rcsb_database"]),
    (("chembl", "compound", "ic50", "smiles"), ["chembl_database", "rdkit"]),
    (("rdkit", "分子", "ligand", "小分子"), ["rdkit", "chembl_database"]),
    (("kegg", "pathway", "通路"), ["kegg_database"]),
    (("brenda", "ec number", "enzyme kinetic", "km ", "kcat"), ["brenda_database"]),
    (("interpro", "domain", "结构域", "pfam"), ["interpro_domain_annotation", "uniprot_database"]),
    (("fda", "openfda", "adverse", "510k"), ["fda"]),
    (("msa", "clustal", "multiple sequence", "多序列"), ["clustalo_msa", "biopython"]),
    (("mmseqs", "blast homolog", "序列同源", "similarity search"), ["protein_sequence_similarity_search", "biopython"]),
    (("pymol", "render structure", "结构渲染"), ["pymol"]),
    (
        ("solubility", "optimal temperature", "binding site", "activity site", "物化性质", "功能预测"),
        ["protein_property_prediction", "biopython"],
    ),
    (
        ("esmfold", "structure prediction", "结构预测", "structure pipeline", "protein_structure_pipeline"),
        ["protein_structure_pipeline", "alphafold_database", "biopython"],
    ),
    (
        ("hpa", "tissue expression", "subcellular", "蛋白图谱", "表达谱", "human protein atlas"),
        ["hpa_expression_context"],
    ),
    (
        ("finetune", "fine-tune", "fine tune", "训练模型", "train_protein", "custom model", "微调"),
        ["venus_finetune_workflow"],
    ),
    (
        ("hypothesis", "定向进化轮次", "下一步实验", "falsif", "工程假设"),
        ["protein_engineering_hypothesis"],
    ),
    (
        ("maxit", "pdb2cif", "cif2pdb", "chain sequence", "file prep", "apo check", "文件预处理"),
        ["structure_file_prep", "biopython"],
    ),
    (("pubmed", "文献", "literature"), ["pubmed", "openalex", "biorxiv"]),
    (("arxiv",), ["arxiv", "biorxiv"]),
    (("biorxiv", "medrxiv"), ["biorxiv", "pubmed"]),
    # Visualization last — only when no domain keyword matched above
    (
        ("配图", "作图", "论文图", "可视化", "出图", "科研绘图", "figure", "plot", "visual", "chart", "draw", "heatmap", "热图"),
        ["nature_figure", "matplotlib", "seaborn"],
    ),
    (("fasta", "genbank", "seqio", "biopython"), ["structure_file_prep", "biopython"]),
    # Narrow file-prep fallback (avoid bare "sequence"/"structure"/"mutation")
    (("pdb file", "mmcif", "结构文件"), ["structure_file_prep", "biopython"]),
]


def _pick_skill_for_code_step(task_desc: str, available_skill_ids: list[str]) -> str | None:
    """Pick a domain skill for a code step; return None if no confident match (do not blind-fallback)."""
    if not available_skill_ids:
        return None
    available = set(available_skill_ids)
    lower = (task_desc or "").lower()
    for keywords, preferred in _DOMAIN_SKILL_RULES:
        if any(k in lower for k in keywords):
            for sid in preferred:
                if sid in available:
                    return sid
    return None


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
