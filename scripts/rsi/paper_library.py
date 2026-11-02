"""Host-only publication and import of verified paper capabilities across goals."""

from run_reproduction_goal import read, safe_file, verify_run

from agent.harness.contracts import file_hash
from agent.harness.host import write_json
from agent.rsi.paper_campaign import admitted_campaign
from agent.rsi.paper_capability_library import PaperCapabilityLibrary
from agent.rsi.storage import safe_child
from agent.rsi.versions import VersionStore
from agent.science_goal import GoalStore


def source_recipe(folder, protocol, paper, recipe):
    origin = recipe.get("library_origin")
    if origin is None:
        return folder, recipe
    declaration = protocol.get("capability_library")
    if (
        not declaration
        or set(origin) != {"entry_id"}
        or declaration["imports"].get(paper["case"]) != origin["entry_id"]
    ):
        raise ValueError("Imported capability is not bound to this goal")
    library = PaperCapabilityLibrary(declaration["path"])
    library.goal(folder)
    verified = library.resolve(origin["entry_id"], paper, validate_completed_recipe)
    if (
        verified["origin_goal"] == folder
        or {k: v for k, v in recipe.items() if k != "library_origin"} != verified["recipe"]
    ):
        raise ValueError("Imported recipe changed or references itself")
    return verified["origin_goal"], verified["recipe"]


def validate_completed_recipe(folder, case, *, replay_artifacts=True):
    state, protocol = GoalStore(folder).read(), read(folder / "protocol.json")
    if state["status"] != "completed" or state["inflight"] is not None:
        raise ValueError("Only completed paper goals can publish capabilities")
    papers = admitted_campaign(protocol, state)
    paper = next((p for p in papers if p["case"] == case), None)
    if paper is None:
        raise ValueError("Paper was not admitted to origin goal")
    # Verify historical snapshots, without claiming the current evaluator is
    # the historical implementation or executing archived code.
    for name, sha in protocol["implementation"].items():
        if file_hash(safe_child(folder / "implementation", name)) != sha:
            raise ValueError("Historical paper implementation changed")
    progress = state["progress"]["papers"][case]
    version = progress["recipe_version"]
    VersionStore(folder / "recipes").verify(version)
    recipe = read(folder / "recipes/versions" / version / "recipe.json")
    if (
        recipe.get("paper") != paper
        or recipe["plugin"] != paper["allocation"]["plugin"]
        or recipe["parameters"] != paper["allocation"]["parameters"]
    ):
        raise ValueError("Historical paper recipe allocation changed")
    if not replay_artifacts and "library_origin" in recipe:
        raise ValueError("Library entries must reference an original recipe")
    origin_goal, original = source_recipe(folder, protocol, paper, recipe)
    initial = safe_file(origin_goal, original["source_run"] + "/report.json").parent
    if verify_run(initial, paper["allocation"], replay_artifacts=replay_artifacts) != original["proof"]:
        raise ValueError("Historical reproduction proof changed")
    receipt = read(safe_file(folder, progress["reuse_receipt"]))
    fresh = safe_file(folder, receipt["source_run"] + "/report.json").parent
    proof = verify_run(fresh, paper["allocation"], registered_recipe=recipe, replay_artifacts=replay_artifacts)
    if (
        receipt["recipe_version"] != version
        or proof != receipt["proof"]
        or fresh.resolve() == initial.resolve()
        or set(proof["sessions"]) & set(original["proof"]["sessions"])
    ):
        raise ValueError("Historical fresh-session reuse was not verified")
    return {"recipe": recipe, "recipe_version": version}


def bind_completed_recipe(folder, case):
    """Check historical receipt hashes during admission, without numerical replay.

    Full native replay is mandatory in the detached worker before any model
    call, completion assessment or publication. Large historical predictions
    must not run inside the web service's bounded preparation request.
    """
    return validate_completed_recipe(folder, case, replay_artifacts=False)


def import_recipes(store, library, matches):
    progress = {"papers": {}}
    for case, item in matches.items():
        source = store.root / "imported_recipes" / case
        source.mkdir(parents=True, exist_ok=False)
        value = {**item["recipe"], "library_origin": {"entry_id": item["entry_id"]}}
        write_json(source / "recipe.json", value)
        version = VersionStore(store.root / "recipes").snapshot(
            source, metadata={"kind": value["kind"], "imported": True}
        )
        progress["papers"][case] = {"recipe_version": version["version_id"]}
    if matches:
        store.mutate(lambda state: state.update(progress=progress))


def publish_campaign(store):
    protocol = read(store.root / "protocol.json")
    declaration = protocol.get("capability_library")
    if not declaration:
        return None
    state = store.read()
    papers = admitted_campaign(protocol, state)
    library = PaperCapabilityLibrary(declaration["path"])
    entries = {paper["case"]: library.publish(store.root, paper["case"], validate_completed_recipe) for paper in papers}
    value = {
        "schema": 1,
        "entries": entries,
        "goal_sha256": file_hash(store.root / "goal.json"),
        "new_model_claim": False,
    }
    write_json(store.root / "capability_publication.json", value)
    return value
