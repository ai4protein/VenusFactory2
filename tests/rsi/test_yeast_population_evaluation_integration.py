"""Terminal synthetic campaign; real control fits and core evaluation replay."""

from pathlib import Path

from test_yeast_population_controls import example

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi import yeast_population_evaluation as evaluation
from agent.rsi.campaigns.scientist_model_campaign import ALL_ALLOCATIONS, ModelCampaignStore
from agent.rsi.engine.storage import atomic_json, sha256_file


def test_terminal_campaign_with_no_model_completes_real_six_control_evaluation(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot"
    repository = pilot / "repository"
    repository.mkdir(parents=True)
    worker = repository / "worker.py"
    worker.write_text("# synthetic executor, no model requests\n")
    snapshot = {"worker.py": sha256_file(worker)}
    monkeypatch.setattr(evaluation.core, "source_identity", lambda: snapshot)
    scope, _, _ = example()
    rows = [
        {
            "id": f"r{i}",
            "protein": scope["protein"],
            "group": f"population_{i // 10}",
            "values": [float(i), 0.5, 1.0, 2.0],
            "target": 2.5 * i - 1,
        }
        for i in range(60)
    ]
    bindings = {}
    for phase, values in [("training", rows[:40]), ("validation", rows[40:50]), ("future", rows[50:])]:
        file = pilot / (phase + ".json")
        atomic_json(file, values)
        bindings[phase] = {"path": str(file), "sha256": sha256_file(file)}
    reference = pilot / "reference.txt"
    reference.write_text("Synthetic population-expression fixture, not biological model evidence")
    source_ref = {"path": str(reference), "sha256": sha256_file(reference)}
    groups = pilot / "group_evidence.json"
    atomic_json(
        groups,
        {
            "schema": "context_groups.v1",
            "level": "genetic_population",
            "study_doi": "test:expression",
            "sources": [source_ref],
            "observations": {
                row["id"]: {"group": row["group"], "phase": phase}
                for phase, values in [("training", rows[:40]), ("validation", rows[40:50]), ("future", rows[50:])]
                for row in values
            },
        },
    )
    scope["grouping"]["evidence_sha256"] = sha256_file(groups)
    task = {
        "schema": 1,
        "id": "synthetic_expression",
        "stage": "expression",
        "plugin_id": "generated_context.v1",
        "scope": scope,
        "paper": {"title": "Synthetic fixture", "doi": "test:expression"},
        "references": [source_ref, {"path": str(groups), "sha256": sha256_file(groups)}],
        "data": bindings,
        "minimum_skill": 0.0,
    }
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test_fixture", "model", "https://example.org/v1"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2),
    ).public()
    allocation = {
        "id": "task",
        "task": task,
        "profile": profile,
        "generations": 1,
        "goal_budget": {"max_cycles": 4, "model_calls": 12, "tool_calls": 8, "wall_seconds": 60},
    }
    store = ModelCampaignStore.create(
        repository / "research_runs/synthetic",
        counting="program_entry.v1",
        execution=ALL_ALLOCATIONS,
        objective="Synthetic evaluation fixture",
        target_models=1,
        library_root=repository / "library",
        repository=repository,
        implementation=snapshot,
        allocations=[allocation],
        resources={"model_calls": 12, "tool_calls": 8, "wall_seconds": 60},
    )
    study = pilot / "evaluation_study"
    evaluation.core.admit(store.root, study)
    atomic_json(pilot / "snapshot.json", {"files": snapshot})
    atomic_json(pilot / "plan.json", {"allocations": [allocation]})
    atomic_json(
        pilot / "protocol.json",
        {
            "final_controls": {
                "methods": list(evaluation.METHODS),
                "implementation": "src/agent/rsi/engine/yeast_population_controls.py",
                "sha256": sha256_file(Path(evaluation.yeast_population_controls.__file__)),
            },
            "allocation_count": 1,
            "selection": {"selected_tasks": [task["id"]]},
        },
    )
    atomic_json(
        pilot / "preparation.json",
        {
            "plan_sha256": sha256_file(pilot / "plan.json"),
            "protocol_sha256": sha256_file(pilot / "protocol.json"),
            "study": str(study),
            "campaign": str(store.root),
            "allocation_count": 1,
        },
    )

    def finish(state):
        state["status"] = "needs_tasks"
        state["allocations"]["task"].update(
            status="budget_exhausted",
            attempts=[
                {
                    "status": "budget_exhausted",
                    "usage_known": False,
                    "charged": {"model_calls": 12, "tool_calls": 8, "wall_seconds": 60},
                }
            ],
        )

    store.mutate(finish)
    output = tmp_path / "scores"
    result = evaluation.evaluate(pilot, output)
    assert result["status"] == "completed"
    assert result["allocated"] == 1 and result["completed_controls"] == 6
    assert result["scored_models"] == result["publication_and_reuse_successes"] == 0
    assert result["external_model_calls"] == result["new_autonomous_models"] == 0
    assert result["rows"][0]["final_metrics"] is None
    assert evaluation.core.read(output / "core_audit/audit.json")["status"] == "passed"
