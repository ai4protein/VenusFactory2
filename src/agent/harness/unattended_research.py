"""Host-selected unattended execution; never grants scientific acceptance or tool access."""
POLICY = "unattended_research.v1"
GUIDANCE = (
    "This research run is unattended. All actions within the admitted goal, enabled tools and resource "
    "budget are pre-authorized; implement, test and revise without asking for routine confirmation. "
    "request_confirmation records an automatic policy decision, not a human response. "
    "Choose and record reasonable methodological assumptions. Inspect supplied materials and use enabled "
    "public information tools to resolve unknowns. If indispensable scientific inputs remain unavailable, "
    "save partial evidence and a checkpoint listing the exact missing inputs; do not wait for a human, "
    "invent observations or change the research objective. Tool permissions, sandbox isolation, free-model "
    "routes, budgets and independent SC/Review acceptance remain enforced."
)


def enabled(value):
    if value is None:
        return False
    if value != POLICY:
        raise ValueError("Unknown research execution policy")
    return True
