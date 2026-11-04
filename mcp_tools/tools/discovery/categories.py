"""Small discovery facets shared across deployments; policy decides visibility."""
from mcp_tools.scheduler.catalog import classify, tool_family

CATEGORIES = {
    "databases": ("Biology databases", "Find protein, gene, target, pathway and compound records or remote similarity searches."),
    "literature": ("Web and literature", "Find papers, public datasets and web evidence."),
    "compute": ("Computational tools", "Use pretrained models, protein design, docking and sequence or structure calculations."),
    "factory_models": ("Factory-trained models", "Apply task-specific models trained on platform datasets or user checkpoints."),
    "rsi": ("RSI models and methods", "Find reviewed Scientist models and reusable research methods."),
    "training": ("Model development", "Prepare data, train and evaluate models within the available resources."),
    "files": ("Session files and code", "Read or write session files and run isolated analysis code."),
    "guides": ("Manual and skills", "Find platform tutorials or a scientific workflow, then read the selected guide."),
}
ROUTERS = {
    "guides": {"query_platform_manual", "list_skills"},
    "compute": {"search_research_tools"},
    "factory_models": {"search_research_tools", "list_trained_models"},
    "rsi": {"search_research_tools", "search_generated_capabilities", "list_evolved_models"},
}

ALIASES = {
    "database": "databases", "数据库": "databases",
    "web": "literature", "search": "literature", "文献": "literature",
    "tools": "compute", "computation": "compute", "计算工具": "compute",
    "factory": "factory_models", "平台模型": "factory_models",
    "rsi_tools": "rsi", "rsi_models": "rsi", "scientist": "rsi",
    "模型训练": "training", "workspace": "files", "文件": "files",
    "skills": "guides", "skill": "guides", "manual": "guides", "手册": "guides",
}

def category_id(value):
    if not isinstance(value, str):
        raise ValueError("Choose a category ID from the category list.")
    value = value.strip().casefold()
    return ALIASES.get(value, value)

def tool_categories(name, tool):
    meta = getattr(tool, "metadata", None) or {}
    explicit = meta.get("discovery_category")
    if explicit in CATEGORIES:
        return {explicit}
    if name == "search_research_tools":
        return {"compute", "factory_models", "rsi"}
    if name in {"query_platform_manual", "list_skills", "read_skill"}:
        return {"guides"}
    if name in {"train_protein_model", "generate_training_config"}:
        return {"training"}
    family = tool_family(name, tool)
    if family == "rsi":
        return {"rsi"}
    if family == "factory":
        return {"factory_models"}
    if family == "pretrained":
        return {"compute"}
    kind = classify(name, meta)["category"]
    categories = {"web_search": "literature", "protein_database": "databases",
             "remote_analysis": "databases", "platform_help": "guides",
             "model_training": "training", "workspace": "files",
             "sandbox_execution": "files"}
    return {categories.get(kind, "compute")}
