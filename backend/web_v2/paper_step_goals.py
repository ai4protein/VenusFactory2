"""Product adapters for generated numerical steps and owner-scoped source reuse."""

import os
import re
from dataclasses import dataclass

from agent.rsi.generated_step_library import GeneratedStepLibrary
from agent.rsi.paper_step import digest
from agent.rsi.paper_step_catalog import catalog, task
from agent.rsi.paper_step_goal import CRITERIA as DEVELOPMENT_CRITERIA
from agent.rsi.paper_step_sessions import read
from agent.rsi.paper_step_transfer import CRITERIA as TRANSFER_CRITERIA
from agent.rsi.storage import safe_child, sha256_file


@dataclass(frozen=True)
class PaperStepDriver:
    root: object
    transfer: bool = False

    @property
    def id(self):
        return "paper_step_transfer.v1" if self.transfer else "paper_step.v1"

    @property
    def title(self):
        return "Reuse verified computation code" if self.transfer else "Develop and verify computation code"

    @property
    def criteria(self):
        return TRANSFER_CRITERIA if self.transfer else DEVELOPMENT_CRITERIA

    @property
    def cases(self):
        return tuple(catalog(self.root))

    def case_details(self, profile):
        result = {}
        for identity, item in catalog(self.root).items():
            definition = item["definition"]
            minimum = 1 + len(definition["cases"]["reuse"])
            result[identity] = {
                "label": item["label"],
                "scope": "; ".join(definition["scope"]["covered_steps"]),
                "excluded_steps": definition["scope"]["excluded_steps"],
                "paper": definition["paper"],
                "interface": definition["interface"],
                "minimum_cycles": minimum,
                "suggested_budget": {
                    "max_cycles": min(8, minimum + 2),
                    "model_calls": 192,
                    "tool_calls": 96,
                    "wall_seconds": 7200,
                },
            }
        return result

    def prepare_command(self, service, folder, spec):
        params = spec["parameters"]
        required = {"case", "entry_id"} if self.transfer else {"case"}
        if set(params) != required:
            raise ValueError("Select a computation task and, for reuse, a compatible code entry")
        item = task(service.root, params["case"])
        if self.transfer and not re.fullmatch(r"[0-9a-f]{64}", params["entry_id"]):
            raise ValueError("Invalid generated code entry")
        if spec["budget"]["max_cycles"] < 1 + len(item["definition"]["cases"]["reuse"]):
            raise ValueError("Allow an acceptance episode and every fresh-input reuse episode")
        command = [
            str(service.python),
            str(service.root / "scripts/rsi/run_web_paper_step.py"),
            "create",
            "--output",
            str(folder),
            "--profile",
            str(service.prepared_profile(folder)),
            "--objective",
            spec["objective"],
            "--case",
            params["case"],
            "--definition-sha256",
            item["definition_sha256"],
        ]
        if self.transfer:
            command += ["--entry-id", params["entry_id"]]
        for key, value in spec["budget"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return command

    def run_command(self, service, folder, action):
        command = [
            str(service.python),
            "-u",
            str(service.root / "scripts/rsi/run_web_paper_step.py"),
            action,
            "--output",
            str(folder),
        ]
        profile = service.frozen_profile(folder)
        bridge = service.frozen_credential_path(folder)
        if not os.environ.get(profile.model.credential_env) and bridge:
            command += ["--credential-toml", str(bridge)]
        return command


def drivers(root):
    return {d.id: d for d in [PaperStepDriver(root), PaperStepDriver(root, True)]}


def available_steps(service, session_id, case):
    definition = task(service.root, case)["definition"]
    library = GeneratedStepLibrary(service.session_folder(session_id) / ".generated_steps")
    matches = library.find(definition)
    return [
        {
            "entry_id": item["entry_id"],
            "paper": item["packet"]["paper"],
            "interface": item["packet"]["interface"],
            "scope": item["packet"]["scope"],
            "origin_record_kind": item["origin_record_kind"],
        }
        for item in matches[:50]
    ]


def public_step(folder, state):
    kind = state["definition"].get("admission", {}).get("kind")
    if kind not in {"generated_paper_step_goal.v1", "generated_paper_step_transfer.v1"}:
        return None
    try:
        definition = read(safe_child(folder, "step_definition.json"))
        checks = (state.get("assessment") or {}).get("criteria", {})
        progress = state["progress"]
        imported = kind == "generated_paper_step_transfer.v1"
        accepted = (
            checks.get("recipient_acceptance_passed" if imported else "accepted_step_registered", {}).get("passed")
            is True
        )
        result = {
            "paper": definition["paper"],
            "interface": definition["interface"],
            "scope": definition["scope"],
            "imported": imported,
            "registered": False,
            "reused": checks.get("fresh_inputs_reused", {}).get("passed") is True,
            "completed_inputs": len(progress.get("reuse", {})),
            "required_inputs": len(definition["cases"]["reuse"]),
            "published": False,
        }
        identity = progress.get("capability_id")
        if identity and re.fullmatch(r"[0-9a-f]{64}", identity):
            record = read(safe_child(folder, "host_private/registered/" + identity + ".json"))
            if digest(record["payload"]) != record["sha256"] or record["sha256"] != identity:
                raise ValueError("Registration identity changed")
            result.update(capability_id=identity, source_sha256=record["payload"]["source_sha256"], registered=accepted)
        publication = folder / "step_publication.json"
        if state["status"] == "completed" and publication.exists():
            value = read(publication)
            if value["goal_sha256"] != sha256_file(folder / "goal.json") or value["capability_id"] != identity:
                raise ValueError("Publication does not match completed goal")
            entry = value["entry_id"]
            if not re.fullmatch(r"[0-9a-f]{64}", entry):
                raise ValueError("Invalid published entry")
            library_record = read(safe_child(folder.parent, ".generated_steps/generated_entries/" + entry + ".json"))
            if digest(library_record) != entry:
                raise ValueError("Published library record changed")
            result.update(published=True, entry_id=entry)
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return {"evidence_available": False}
