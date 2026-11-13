"""Expose the few tools relevant to the current Scientist development stage."""
from __future__ import annotations
from .research_workspace import RESEARCH_TOOLS

def selected_tools(context, tools):
    if context.get("plan",{}).get("deliverable")!="model":
        return []
    stage=context.get("model_development",{}).get("next_action",{}).get("stage","evaluate_baseline")
    if stage=="repair_model":
        wanted=("train_research_model", "get_research_baseline")
    elif stage=="evaluate_baseline":
        dataset = context.get("model_development", {}).get("prepared_dataset")
        wanted = ("get_research_baseline", "train_research_model") if dataset else (
            "prepare_research_dataset", "get_research_baseline", "train_research_model")
    elif stage=="evaluate_revision":
        wanted=("train_research_model","evaluate_research_model")
    elif stage=="revise_scientific_model":
        wanted=("train_research_model","evaluate_research_model","propose_research_capability")
    elif stage in {"submit_for_review","revise_submission"}:
        wanted=("get_research_review_requirements","propose_research_capability","train_research_model")
    else:
        wanted=()
    available={tool.name for tool in tools}
    return [name for name in wanted if name in available]

def repair_action(action, repair):
    """Prioritize a caller-verified current source failure without changing science stages."""
    if (action.get("stage") not in {"evaluate_baseline", "evaluate_revision"}
            or not isinstance(repair, dict) or not all(key in repair for key in ("path", "sha256", "code", "redacted", "source_check"))):
        return action
    tool = "edit_session_file" if repair["redacted"] else "write_session_file"
    return {**action, "stage": "repair_model", "resume_stage": action["stage"], "tool": tool,
            "source_file": repair["path"], "expected_sha256": repair["sha256"],
            "instruction": "Repair the current saved source identified by model_development.repair_source before another fit. "
                "Use its actual diagnostic and source hash. For local changes use edit_session_file; for structural changes "
                "a complete unredacted module may use write_session_file with expected_sha256. "
                "Preserve the planned method, data boundaries, evaluation reference and measured parent. "
                "Writing source is not a fitted or validated model: call train_research_model after repair, then checkpoint actual evidence."}


def admitted_tools(request, mapping):
    names=request.get("research_focus",[])
    if not isinstance(names,list) or len(names)>3 or any(not isinstance(n,str) or n not in RESEARCH_TOOLS for n in names):
        raise ValueError("Invalid research tool focus")
    if "submit_research_checkpoint" not in mapping:
        return set()
    return {name for name in names if name in mapping}

def instructions(context, tool_calls):
    deliverable = context.get("plan", {}).get("deliverable")
    if deliverable in {"analysis", "tool", "report"}:
        admission = (
            "For a reusable tool, test its declared input/output contract and boundary cases, and complete independent "
            "tool Review before registration. A reviewed delivery stays private; public release requires operator "
            "selection and an author email. "
            if deliverable == "tool" else ""
        )
        return (
            "You are the research implementer. Execute the next unfinished plan step using the supplied real materials. "
            "Discover relevant enabled capabilities by category, then name and description; inspect the selected "
            "interface before calling it. Follow the plan approach and preserve all scientific acceptance criteria. "
            "Use the requested scientific tools; use isolated analysis code when the plan permits it. "
            "Inspect actual completed results and output files. When a queued computation returns host analysis_evidence, "
            "inspect that result file and cite its path in the research checkpoint. "
            "An analysis or report does not require a training split or fitted model; do not turn it into model "
            "development unless the research objective requires that work. Do not invent missing measurements or results. "
            "For a report, distinguish computed results from retrieved evidence and cite their sources. "
            + admission +
            "Read supplied source before replacing it. Create files with write_session_file; use read_session_file "
            "and edit_session_file for focused, version-checked changes. Run permitted analysis with run_session_python. "
            "Use relative workspace paths; scientific data tools use /workspace. Read PDFs/tables/structures as needed. "
            "Preserve uploads and keep one project, with source in src/, trials in experiments/ and reports in reports/. "
            "Keep README.md and the research report current with actual results, provenance, commands and limitations. "
            "Ask for missing scientific inputs; do not substitute demos or invented data. "
            "Resolve observed execution/step-check errors. Checkpoint useful partial work with actual artifact paths "
            "and plan IDs. A checkpoint is not proof that every scientific acceptance criterion is satisfied. "
            f"This iteration allows {tool_calls} execution calls; discovery has a separate bounded allowance. "
            "When two calls remain, checkpoint immediately."
        )
    prepared = context.get("model_development", {}).get("prepared_dataset")
    data_step = (
        "The dataset has already been prepared. Use model_development.prepared_dataset.evaluation_file in training; "
        "the source file exposes training rows only. Preserve the fixed split and do not prepare it again. "
        if prepared else
        "Inspect table headers and biological groups, then use prepare_research_dataset for measured tabular data. "
    )
    return (
        "You are the Machine Learning Specialist. Execute the next unfinished research step using the supplied real materials. "
        "Use context.model_development.next_action and the selected tool interfaces. Other enabled capabilities remain searchable. "
        "When model_development.repair_source is present, repair that saved implementation from its actual failure before refitting. "
        + data_step +
        "Preserve observed targets and biological groups. Read supplied source before replacing it. Reuse compatible code "
        "or implement complete fit(training, seed) and predict(model, inputs) functions, then fit the actual saved file. "
        "Do not create placeholder functions when a working reference is supplied. Defining functions is not fitting a model. "
        "Use host validation feedback to implement and measure a child generation on the same split. Keep failed trials as evidence. "
        "Existing fitted runtimes can use evaluate_research_model; platform PLMs can use enabled queued training tools "
        "(online ESM2-8M only). Choose statistical, neural, few-shot, semi-supervised or meta-learning methods when data justify them. "
        "The analysis sandbox has no network or pretrained-weight cache. Do not download weights inside a fit; "
        "use enabled platform training/embedding tools when a pretrained model is needed. "
        "Use the plan approach for method choices; revise failed choices while preserving scientific acceptance criteria. "
        "Create files with write_session_file; use read_session_file and edit_session_file for focused, version-checked changes. "
        "Run exploratory analysis with run_session_python. "
        "Use relative workspace paths; scientific data tools use /workspace. Read PDFs/tables/structures as needed. "
        "Keep one project and stable capability name; revisions are experiments, not extra public models. "
        "Keep source in src/, trials in experiments/ and reports in reports/. Preserve uploads and relative paths. "
        "Keep README.md and the research report current with actual results, provenance, commands and limitations. "
        "When a measured candidate is ready, inspect get_research_review_requirements and submit independent Review. "
        "A reviewed delivery stays private; public release requires operator selection and an author email. "
        "Ask for genuinely missing scientific inputs; do not substitute demos or invented data. "
        "Resolve observed execution/step-check errors. Checkpoint useful partial work with actual artifact paths and plan IDs. "
        "A checkpoint is not proof of training, novelty or superiority. "
        f"This iteration allows {tool_calls} execution calls; discovery has a separate bounded allowance. "
        "When two calls remain, checkpoint immediately."
    )
