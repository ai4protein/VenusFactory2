"""Owner-scoped source capabilities with revalidated historical computation evidence."""

import hashlib

from agent.harness import Profile
from agent.harness.contracts import file_hash
from agent.science_goal import GoalStore, validate_assessment

from .paper_capability_library import PaperCapabilityLibrary
from .paper_step import PaperStepLifecycle, digest, validate_definition
from agent.rsi.campaigns.paper_step_goal import CRITERIA
from .paper_step_sessions import read, verify_phase
from .storage import atomic_json, file_lock, safe_child


def session_proof(state, folder):
    proof = verify_phase(folder)
    protocol = read(folder.parents[1] / "protocol.json") if folder.parent.name == "episodes" else None
    if protocol and "scientist_tool_context" in protocol:
        from .scientist_tool_context import verify_context

        verify_context(folder, protocol["scientist_tool_context"], read(folder.parents[1] / "step_definition.json"))
    base = Profile.from_dict(state["definition"]["profile"])
    actual = Profile.from_dict(read(folder / "runtime/profile.lock.json")["profile"])
    before, after = base.public(), actual.public()
    before.pop("budget")
    after.pop("budget")
    if before != after:
        raise ValueError("Historical mode, harness or model changed")
    for name in ["temperature", "thinking", "reasoning_effort", "output_tokens", "request_seconds"]:
        if getattr(base.budget, name) != getattr(actual.budget, name):
            raise ValueError("Historical inference settings changed")
    for name in ["model_calls", "tool_calls", "wall_seconds"]:
        if getattr(actual.budget, name) > getattr(base.budget, name):
            raise ValueError("Historical phase exceeds admitted limits")
    return proof


def validate_completed_step(folder):
    """Verify archives as data; never import or execute their implementation.

    The existing numerical lifecycle and session validators still require the
    exact compatible current numerical/tool implementation. Unrelated current
    repository files need not equal every historical snapshot file.
    """
    store = GoalStore(folder)
    state = store.read()
    folder = store.root
    protocol = read(folder / "protocol.json")
    definition = read(folder / "step_definition.json")
    admission = {"kind": "generated_paper_step_goal.v1", "definition_sha256": digest(definition)}
    if (
        state["status"] != "completed"
        or state["inflight"] is not None
        or tuple(state["definition"]["goal"]["criteria"]) != CRITERIA
    ):
        raise ValueError("Only completed original source goals may publish a new step")
    if protocol["admission"] != admission or state["definition"].get("admission") != {
        **admission,
        "protocol_sha256": digest(protocol),
    }:
        raise ValueError("Historical paper step admission changed")
    for relative, sha in protocol["implementation"].items():
        if file_hash(safe_child(folder / "implementation", relative)) != sha:
            raise ValueError("Historical implementation archive changed")
    read(folder / "host_private/admission.json")
    lc = PaperStepLifecycle(folder / "host_private", definition)
    if "scientist_tool_context" in protocol:
        from .research_tool_lifecycle import ResearchToolLifecycle

        lc = ResearchToolLifecycle(folder / "host_private", definition, protocol["scientist_tool_context"]["contract"])
    progress = state["progress"]
    identity = progress["capability_id"]
    packet = lc.capability(identity)
    development = safe_child(folder, progress["development_folder"])
    proof = session_proof(state, development)
    receipt = read(development / "host_acceptance.json")
    if receipt != {
        "candidate_id": packet["verification"]["candidate_id"],
        "assessment": packet["verification"],
        "source_provenance": proof,
        "capability_id": identity,
    }:
        raise ValueError("Origin source acceptance changed")
    if (
        proof["source_sha256"] != hashlib.sha256(packet["source_code"].encode()).hexdigest()
        or proof.get("source_via_model_tool_call_verified") is not True
    ):
        raise ValueError("Origin source lacks its recorded tool submission")
    checks = {
        key: {
            "passed": True,
            "evidence": [
                {
                    "path": str((development / "host_acceptance.json").relative_to(folder)),
                    "sha256": file_hash(development / "host_acceptance.json"),
                }
            ],
        }
        for key in CRITERIA[:2]
    }
    reuse = progress.get("reuse", {})
    if set(reuse) != {c["id"] for c in definition["cases"]["reuse"]} or progress.get("hold"):
        raise ValueError("Origin fresh reuse is incomplete or held")
    sessions = {proof["session_id"]}
    evidence = []
    for case_id, relative in reuse.items():
        episode = safe_child(folder, relative)
        reused = session_proof(state, episode)
        actual = lc.verify_reuse(identity, case_id)
        if (
            not actual["passed"]
            or read(episode / "host_acceptance.json") != {"verification": actual, "source_provenance": reused}
            or reused["session_id"] in sessions
        ):
            raise ValueError("Origin reuse proof changed or session is not fresh")
        sessions.add(reused["session_id"])
        evidence.append(
            {
                "path": str((episode / "host_acceptance.json").relative_to(folder)),
                "sha256": file_hash(episode / "host_acceptance.json"),
            }
        )
    checks[CRITERIA[2]] = {"passed": True, "evidence": evidence}
    assessment = {"criteria": checks}
    if assessment != state["assessment"] or not validate_assessment(store, state, assessment):
        raise ValueError("Historical completion assessment changed")
    return {
        "packet": packet,
        "definition": definition,
        "development_proof": proof,
        "sessions": sorted(sessions),
        "origin_record_kind": "simulated_model_records"
        if proof["provider"] == "test_fixture"
        else "recorded_model_session",
        "autonomous_scientific_success": False,
    }


def compatible(packet, definition):
    validate_definition(definition)
    return (
        packet["interface"] == definition["interface"]
        and packet["scope"]["covered_steps"] == definition["scope"]["covered_steps"]
        and set(packet["scope"]["excluded_steps"]) <= set(definition["scope"]["excluded_steps"])
    )


def validate_transfer(origin, definition):
    if not compatible(origin["packet"], definition):
        raise ValueError("Imported interface or scope is incompatible")
    if any(definition["tolerance"][k] > origin["definition"]["tolerance"][k] for k in ["atol", "rtol"]):
        raise ValueError("Transfer may not loosen numerical acceptance tolerance")
    old = {digest(c["inputs"]) for group in origin["definition"]["cases"].values() for c in group}
    new = {digest(c["inputs"]) for group in definition["cases"].values() for c in group}
    if old & new:
        raise ValueError("Recipient inputs overlap exact donor inputs")


class GeneratedStepLibrary(PaperCapabilityLibrary):
    """Separate namespace from fixed workflow recipes. Imported reuse is deduplicated."""

    def publish(self, folder):
        folder = self.goal(folder)
        state = GoalStore(folder).read()
        if state["definition"].get("admission", {}).get("kind") == "generated_paper_step_transfer.v1":
            from .paper_step_transfer import validate_completed_transfer

            return validate_completed_transfer(folder, self)
        origin = validate_completed_step(folder)
        payload = {
            "schema": 1,
            "kind": "generated_step_capability",
            "origin_goal": str(folder.relative_to(self.owner)),
            "interface": origin["packet"]["interface"],
            "scope": origin["packet"]["scope"],
            "packet_sha256": digest(origin["packet"]),
            "proof_sha256": digest(origin),
            "goal_sha256": file_hash(folder / "goal.json"),
            "protocol_sha256": file_hash(folder / "protocol.json"),
            "origin_record_kind": origin["origin_record_kind"],
        }
        identity = digest(payload)
        path = safe_child(self.root, "generated_entries/" + identity + ".json")
        with file_lock(self.root / ".generated_library.lock"):
            for existing_path in sorted((self.root / "generated_entries").glob("*.json")):
                existing_value = read(safe_child(self.root, "generated_entries/" + existing_path.name))
                if digest(existing_value) != existing_path.stem:
                    raise ValueError("Generated capability entry identity changed")
                if any(existing_value.get(k) != payload[k] for k in ["interface", "scope", "origin_record_kind"]):
                    continue
                existing = self.resolve(existing_path.stem)
                if existing["packet"]["source_code"] != origin["packet"]["source_code"]:
                    continue
                if existing_path.stem != identity:
                    supporting = safe_child(
                        self.root, "supporting_origins/" + existing_path.stem + "/" + identity + ".json"
                    )
                    if supporting.exists():
                        if read(supporting) != payload:
                            raise ValueError("Supporting origin evidence changed")
                    else:
                        atomic_json(supporting, payload)
                        supporting.chmod(0o444)
                return existing_path.stem
            if path.exists():
                if read(path) != payload:
                    raise ValueError("Generated capability entry changed")
            else:
                atomic_json(path, payload)
                path.chmod(0o444)
        return identity

    def resolve(self, identity):
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid generated capability identity")
        value = read(safe_child(self.root, "generated_entries/" + identity + ".json"))
        if digest(value) != identity or value.get("schema") != 1 or value.get("kind") != "generated_step_capability":
            raise ValueError("Generated capability entry identity changed")
        folder = self.goal(safe_child(self.owner, value["origin_goal"]))
        if (
            file_hash(folder / "goal.json") != value["goal_sha256"]
            or file_hash(folder / "protocol.json") != value["protocol_sha256"]
        ):
            raise ValueError("Generated capability origin changed")
        origin = validate_completed_step(folder)
        if (
            digest(origin) != value["proof_sha256"]
            or digest(origin["packet"]) != value["packet_sha256"]
            or origin["packet"]["interface"] != value["interface"]
            or origin["packet"]["scope"] != value["scope"]
            or origin["origin_record_kind"] != value["origin_record_kind"]
        ):
            raise ValueError("Generated capability proof changed")
        return {**origin, "entry_id": identity, "origin_goal": str(folder)}

    def find(self, definition):
        validate_definition(definition)
        matches = []
        for path in sorted((self.root / "generated_entries").glob("*.json")):
            value = read(safe_child(self.root, "generated_entries/" + path.name))
            if digest(value) != path.stem:
                raise ValueError("Generated capability entry identity changed")
            if compatible(value, definition):
                item = self.resolve(path.stem)
                try:
                    validate_transfer(item, definition)
                except ValueError:
                    continue
                matches.append(item)
        return matches
