"""Library-first persistent Scientist goals using the existing numerical lifecycle."""

from agent.harness.native_expert_team import NATIVE_EXPERT_HARNESSES
from agent.harness.scientist_policy import validate_policy
from agent.science_goal import GoalSpec

from .generated_step_library import GeneratedStepLibrary
from .paper_step import validate_definition
from .paper_step_goal import CRITERIA, PaperStepGoal, create_step_goal
from .paper_step_sessions import read
from .paper_step_transfer import CRITERIA as TRANSFER_CRITERIA
from .paper_step_transfer import TransferredStepGoal, create_transfer_goal
from .research_tool_contract import validate_contract


def create_research_tool_goal(
    root, definition, contract, policy, profile, objective, library, *, entry_id=None, **limits
):
    validate_definition(definition)
    validate_contract(contract, definition)
    validate_policy(policy)
    if profile.mode != "science_expert" or profile.harness not in NATIVE_EXPERT_HARNESSES:
        raise ValueError("Research tool goals require a native four-role Scientist")
    library = GeneratedStepLibrary(library)
    library.goal(root)
    matches = library.find(definition)
    # Simulated records exercise engineering paths but are not donors for a
    # scientific campaign. Imported code retains the original entry identity.
    matches = [m for m in matches if m["origin_record_kind"] == "recorded_model_session"]
    ids = sorted(m["entry_id"] for m in matches)
    if entry_id is not None and entry_id not in ids:
        raise ValueError("Requested tool was not independently verified as compatible")
    selected = entry_id or (ids[0] if ids else None)
    context = {
        "contract": contract,
        "policy": policy,
        "discovery": {
            "library": str(library.root),
            "matches": ids,
            "action": "reuse" if selected else "create",
            "selected_entry": selected,
        },
    }
    spec = GoalSpec(objective, TRANSFER_CRITERIA if selected else CRITERIA, **limits)
    if selected:
        return create_transfer_goal(root, definition, profile, spec, library, selected, tool_context=context)
    return create_step_goal(root, definition, profile, spec, tool_context=context)


def research_tool_driver(store, credential, runtimes):
    protocol = read(store.root / "protocol.json")
    if "scientist_tool_context" not in protocol:
        raise ValueError("This is not a typed research-tool goal")
    driver = TransferredStepGoal if "import" in protocol else PaperStepGoal
    return driver(store, credential, runtimes)
