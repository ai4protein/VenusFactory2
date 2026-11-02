"""Trusted reproduction plugins with validated, shell-free invocation recipes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReproductionPlugin:
    id: str
    stages: tuple[str, ...]
    entry: str
    environment: str
    scope: str
    resource: str = "gpu"


PLUGINS = {
    "rfantibody.v1": ReproductionPlugin(
        "rfantibody.v1",
        ("backbone", "sequence", "refinement", "quality_audit"),
        "scripts/rsi/run_rfantibody_case.py",
        "controller",
        "complete computational chain at declared scale; binding and full-paper statistics excluded",
    ),
    "turbopetase.v1": ReproductionPlugin(
        "turbopetase.v1",
        ("checkpoint_validation", "author_inference", "indexing_audit", "measurement_audit"),
        "scripts/rsi/run_turbopetase_case.py",
        "turbopetase",
        "released-checkpoint inference and data audit; original training and wet-lab replication excluded",
    ),
}


PLUGINS["rfantibody.v2"] = ReproductionPlugin(
    "rfantibody.v2",
    PLUGINS["rfantibody.v1"].stages,
    PLUGINS["rfantibody.v1"].entry,
    "controller",
    "deterministic sampling in all three computational stages; experimental binding excluded",
)


PLUGINS["rfantibody.targets.v1"] = ReproductionPlugin(
    "rfantibody.targets.v1",
    PLUGINS["rfantibody.v1"].stages,
    "scripts/rsi/run_rfantibody_targets.py",
    "controller",
    "author-parameter HA or RSV site-3 nanobody computational chain; binding and publication-scale statistics excluded",
)


PLUGINS["alde.onehot_dnn.v1"] = ReproductionPlugin(
    "alde.onehot_dnn.v1",
    ("source_validation", "author_trajectories_or_reuse", "independent_trajectory_audit"),
    "scripts/rsi/run_alde_case.py",
    "controller",
    "author one-hot/DNN four-round GB1/TrpB campaigns; new computation and verified existing evidence remain distinct",
    "cpu",
)


PLUGINS["mcba.multisubstrate.v1"] = ReproductionPlugin(
    "mcba.multisubstrate.v1",
    ("source_validation", "all_notebook_tasks_or_verified_reuse", "independent_prediction_and_metric_audit"),
    "scripts/rsi/run_mcba_case.py",
    "controller",
    "all 25 released processed-input notebook tasks, with numerical/reference discrepancies retained; upstream zero-shot generation and wet-lab replication excluded",
    "cpu",
)


PLUGINS["phl7.opes_analysis.v1"] = ReproductionPlugin(
    "phl7.opes_analysis.v1",
    ("archive_admission", "all_24_opes_replays", "numerical_reference_comparison"),
    "scripts/rsi/run_phl7_case.py",
    "controller",
    "all 24 published OPES-output reweighting analyses; new MD, full-paper replication and efficacy excluded",
    "cpu",
)


PLUGINS["htfunclib.rf.v1"] = ReproductionPlugin(
    "htfunclib.rf.v1",
    (
        "source_and_split_admission",
        "functionality_10000_rounds",
        "colour_10000_rounds",
        "native_prediction_and_metric_audit",
    ),
    "scripts/rsi/run_htfunclib_case.py",
    "htfunclib",
    "GFP released processed-input two-stage training; author filter discrepancy preserved; upstream Rosetta, raw reads, novel models and full-paper replication excluded",
    "cpu",
)


PLUGINS["htfunclib.epinnet.v1"] = ReproductionPlugin(
    "htfunclib.epinnet.v1",
    (
        "score_table_admission",
        "author_epinnet_training",
        "mutation_ranking",
        "top40_library",
        "published_score_reanalysis",
        "native_weight_audit",
    ),
    "scripts/rsi/run_htfunclib_epinnet.py",
    "htfunclib",
    "GFP author EpiNNet from published Rosetta scores to the fixed library; sequence overlap and reference mismatches retained; new Rosetta jobs, autonomous model design and full-paper replication excluded",
    "cpu",
)


PLUGINS["phl7.hrex_validation.v1"] = ReproductionPlugin(
    "phl7.hrex_validation.v1",
    ("preparation", "energy_validation", "dynamics", "native_audit"),
    "scripts/rsi/run_phl7_hrex_case.py",
    "controller",
    "New WT topologies, fixed-frame energies and eight-replica four-ps HREX; no convergence, long-production or full-paper claim",
    "cpu",
)


PLUGINS["catapro.full_cv.v1"] = ReproductionPlugin(
    "catapro.full_cv.v1",
    ("raw_input_and_encoder_admission", "all_three_original_fold_cohorts", "independent_full_row_numpy_replay"),
    "scripts/rsi/run_catapro_plugin.py",
    "controller",
    "All 95,507 released enzyme-substrate rows and 30 author CV checkpoints; raw features regenerated for reproduce, existing computation explicitly identified for verify_existing. Original training, new model invention, unbiased holdout and wet-lab replication excluded.",
    "gpu",
)


PLUGINS["catapro.training.v1"] = ReproductionPlugin(
    "catapro.training.v1",
    ("verified_full_row_features", "author_five_stage_ten_fold_training", "independent_all_row_checkpoint_replay"),
    "scripts/rsi/run_catapro_training_plugin.py",
    "catapro_training",
    "All 50 original-fold fits from verified reconstructed features, using unchanged author training and explicit reconstruction decisions. Reuse performs a new read-only audit. No original grid search, exact stochastic equivalence, novel model, unbiased holdout or wet-lab claim.",
    "gpu",
)


PLUGINS["rfantibody.benchmark.v1"] = ReproductionPlugin(
    "rfantibody.benchmark.v1",
    ("backbone", "sequence", "refinement_1p0", "refinement_0p1", "refinement_0p0", "independent_condition_audits"),
    "scripts/rsi/run_rfantibody_benchmark.py",
    "controller",
    "One registered antigen: 200 backbones, eight sequences each, and paired RF2 proportions 1, 0.1, 0; reconstructed inputs and public code with independent structure replay.",
)


PLUGINS["nucb.raw_analysis.v1"] = ReproductionPlugin(
    "nucb.raw_analysis.v1",
    (
        "pinned_source_and_raw_data",
        "four_generation_reconstruction",
        "landscape",
        "library_and_plate_analysis",
        "native_replay",
    ),
    "scripts/rsi/run_nucb_case.py",
    "nuclease",
    "All four raw-count generations, 16 processed/fiducial tables, 55,760-sequence landscape, 44 library hit rates and 28 initial-rate fits; model training, sequence design, diversity analysis and wet-lab reproduction excluded.",
    "cpu",
)


PLUGINS["clean.annotation.v1"] = ReproductionPlugin(
    "clean.annotation.v1",
    ("source_and_encoder_admission", "all_sequence_embeddings", "both_released_heads", "independent_numpy_replay"),
    "scripts/rsi/run_clean_annotation.py",
    "clean",
    "All 578 released test sequences with fresh ESM-1b embeddings and both CLEAN heads (1,156 predictions); full-union label errors retained. Original contrastive training, p-value inference, novel model design and wet-lab mining excluded.",
    "cpu",
)


PLUGINS["clean.training.v1"] = ReproductionPlugin(
    "clean.training.v1",
    (
        "all_training_and_evaluation_embeddings",
        "complete_triplet_training",
        "complete_supconH_training",
        "all_cohort_evaluation",
        "independent_numerical_replay",
    ),
    "scripts/rsi/run_clean_training_plugin.py",
    "clean",
    "Fresh ESM-1b means for 22,525 records; all 7,757 released split10 sequences, 14,190 augmentations, 2,000 triplet and 1,500 SupConH epochs, and 1,156 independently audited predictions. Every reuse recomputes all stages. Publication five-fold CV, novel model invention, RSI efficacy and wet-lab validation excluded.",
    "cpu",
)


def antibody_allocation(plugin_id, parameters):
    if plugin_id == "rfantibody.benchmark.v1":
        return 200, 8
    return parameters.get("backbones", 2), parameters.get("sequences", 2)


def maximum_case_job_seconds(plugin_id):
    if plugin_id not in PLUGINS:
        raise ValueError("Unknown reproduction plugin")
    if plugin_id == "clean.training.v1":
        return 259200
    return 604800 if plugin_id == "rfantibody.benchmark.v1" else 86400


def resource_for(plugin_id, parameters):
    if plugin_id in {"catapro.full_cv.v1", "catapro.training.v1"}:
        return "cpu" if parameters.get("operation") == "verify_existing" else "gpu"
    if plugin_id == "phl7.hrex_validation.v1":
        return "gpu" if parameters.get("backend", "cpu") == "cuda" else "cpu"
    plugin = PLUGINS[plugin_id]
    return "cpu" if plugin_id == "turbopetase.v1" and parameters.get("device") == "cpu" else plugin.resource


def list_case_plugins():
    catalog = []
    for plugin in PLUGINS.values():
        if plugin.id == "rfantibody.benchmark.v1":
            properties = {"target": {"type": "string", "enum": ["ha", "rsv_site3", "tcdb", "il7r", "rbd"]}}
            required = ["target"]
        elif plugin.id.startswith("rfantibody"):
            properties = {
                "backbones": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 2},
                "sequences": {"type": "integer", "minimum": 1, "maximum": 20, "default": 2},
            }
            required = []
            if plugin.id == "rfantibody.targets.v1":
                properties["target"] = {"type": "string", "enum": ["ha", "rsv_site3"]}
                required = ["target"]
        elif plugin.id == "catapro.training.v1":
            properties = {
                "operation": {"type": "string", "enum": ["reproduce", "verify_existing"]},
                "campaign": {"type": "string", "enum": ["catapro_training_20261017"]},
            }
            required = ["operation"]
        elif plugin.id == "catapro.full_cv.v1":
            properties = {
                "operation": {"type": "string", "enum": ["reproduce", "verify_existing"]},
                "campaign": {"type": "string", "enum": ["catapro_full_20261017"]},
            }
            required = ["operation"]
        elif plugin.id == "phl7.hrex_validation.v1":
            properties = {"backend": {"type": "string", "enum": ["cpu", "cuda"], "default": "cpu"}}
            required = []
        elif plugin.id in {
            "phl7.opes_analysis.v1",
            "htfunclib.rf.v1",
            "htfunclib.epinnet.v1",
            "nucb.raw_analysis.v1",
            "clean.annotation.v1",
            "clean.training.v1",
        }:
            properties = {}
            required = []
        elif plugin.id == "mcba.multisubstrate.v1":
            properties = {
                "operation": {"type": "string", "enum": ["reproduce", "verify_existing"]},
                "campaign": {"type": "string", "enum": ["author_multisubstrate_20261012"]},
            }
            required = ["operation"]
        elif plugin.id == "alde.onehot_dnn.v1":
            properties = {
                "operation": {"type": "string", "enum": ["reproduce", "verify_existing"]},
                "seeds": {"type": "integer", "minimum": 1, "maximum": 70, "default": 5},
                "acquisitions": {
                    "type": "array",
                    "items": {"enum": ["GREEDY", "UCB", "TS"]},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "campaign": {
                    "type": "string",
                    "enum": ["author_ts70_20261011", "author_three_acquisition_pilot_20261011"],
                },
            }
            required = ["operation"]
        else:
            properties = {
                "models": {
                    "type": "array",
                    "items": {"enum": ["PF01083", "PTM311", "pretrained-model"]},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "device": {"type": "string", "enum": ["cpu", "cuda:0"], "default": "cuda:0"},
            }
            required = []
        catalog.append(
            {
                "id": plugin.id,
                "stages": list(plugin.stages),
                "scope": plugin.scope,
                "resource": plugin.resource,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        )
    return catalog


def invocation(plugin_id, parameters, *, root, output, controller_python):
    if plugin_id not in PLUGINS:
        raise ValueError("unknown reproduction plugin")
    root = root.resolve()
    output = output.resolve()
    output.relative_to((root / "research_runs/rsi").resolve())
    if output == (root / "research_runs/rsi").resolve():
        raise ValueError("a dedicated run directory is required")
    plugin = PLUGINS[plugin_id]
    python = (
        controller_python
        if plugin.environment == "controller"
        else root / "research_runs/rsi/envs" / plugin.environment / "bin/python"
    )
    if plugin_id == "catapro.training.v1":
        python = root / "research_runs/rsi/20261017_catapro_runtime_setup_v1/venv/bin/python"
    argv = [str(python), "-u", str(root / plugin.entry), "--output", str(output)]
    if plugin_id == "rfantibody.benchmark.v1":
        if set(parameters) != {"target"} or parameters["target"] not in {"ha", "rsv_site3", "tcdb", "il7r", "rbd"}:
            raise ValueError("Benchmark requires one registered target and its complete fixed allocation")
        argv += ["--target", parameters["target"]]
    elif plugin_id in {"rfantibody.v1", "rfantibody.v2", "rfantibody.targets.v1"}:
        target_plugin = plugin_id == "rfantibody.targets.v1"
        if set(parameters) - ({"backbones", "sequences", "target"} if target_plugin else {"backbones", "sequences"}):
            raise ValueError("unknown case parameter")
        if target_plugin:
            target = parameters.get("target")
            if target not in {"ha", "rsv_site3"}:
                raise ValueError("explicit registered antigen target required")
            argv += ["--target", target]
        if plugin_id == "rfantibody.v2":
            argv.append("--deterministic-sequences")
        for name, default, limit in [("backbones", 2, 1000), ("sequences", 2, 20)]:
            n = parameters.get(name, default)
            if type(n) is not int or not 1 <= n <= limit:
                raise ValueError("invalid design count")
            argv += ["--" + name, str(n)]
    elif plugin_id == "catapro.training.v1":
        if parameters == {"operation": "reproduce"}:
            argv += ["--operation", "reproduce"]
        elif parameters == {"operation": "verify_existing", "campaign": "catapro_training_20261017"}:
            argv += ["--operation", "verify_existing", "--campaign", parameters["campaign"]]
        else:
            raise ValueError("CataPro training requires all 50 fits or one registered terminal campaign")
    elif plugin_id == "catapro.full_cv.v1":
        if parameters == {"operation": "reproduce"}:
            argv += ["--operation", "reproduce"]
        elif parameters == {"operation": "verify_existing", "campaign": "catapro_full_20261017"}:
            argv += ["--operation", "verify_existing", "--campaign", parameters["campaign"]]
        else:
            raise ValueError("CataPro requires all three complete cohorts or a registered complete campaign")
    elif plugin_id == "clean.training.v1":
        if parameters != {}:
            raise ValueError(
                "CLEAN training requires 22525 fresh embeddings and both complete schedules; no overrides or old-output shortcut"
            )
    elif plugin_id == "clean.annotation.v1":
        if parameters != {}:
            raise ValueError(
                "CLEAN requires all 578 sequences and both released heads; no subset or cached-output shortcut"
            )
    elif plugin_id == "nucb.raw_analysis.v1":
        if parameters != {}:
            raise ValueError(
                "NucB requires all four generations and the complete pinned analysis; no subset or existing-output shortcut"
            )
    elif plugin_id == "htfunclib.epinnet.v1":
        if parameters != {}:
            raise ValueError("EpiNNet requires the complete pinned author allocation without overrides")
    elif plugin_id == "htfunclib.rf.v1":
        if parameters != {}:
            raise ValueError("GFP replay requires the complete pinned two-stage protocol without overrides")
    elif plugin_id == "phl7.hrex_validation.v1":
        if set(parameters) - {"backend"} or parameters.get("backend", "cpu") not in {"cpu", "cuda"}:
            raise ValueError("HREX validation accepts only its declared compute backend")
        argv += ["--backend", parameters.get("backend", "cpu")]
    elif plugin_id == "phl7.opes_analysis.v1":
        if parameters != {}:
            raise ValueError("PHL7 analysis requires all 24 allocations without source or subset overrides")
    elif plugin_id == "mcba.multisubstrate.v1":
        if parameters == {"operation": "reproduce"}:
            argv += ["--operation", "reproduce"]
        elif parameters == {"operation": "verify_existing", "campaign": "author_multisubstrate_20261012"}:
            argv += ["--operation", "verify_existing", "--campaign", parameters["campaign"]]
        else:
            raise ValueError(
                "Select full notebook reproduction or a registered existing campaign; subsets and arbitrary sources are not admitted"
            )
    elif plugin_id == "alde.onehot_dnn.v1":
        operation = parameters.get("operation")
        if operation == "verify_existing":
            if set(parameters) != {"operation", "campaign"} or parameters["campaign"] not in {
                "author_ts70_20261011",
                "author_three_acquisition_pilot_20261011",
            }:
                raise ValueError("Select a registered existing campaign without new-computation parameters")
            argv += ["--operation", operation, "--campaign", parameters["campaign"]]
        elif operation == "reproduce":
            if set(parameters) - {"operation", "seeds", "acquisitions"}:
                raise ValueError("Unknown ALDE reproduction parameter")
            seeds = parameters.get("seeds", 5)
            acquisitions = parameters.get("acquisitions", ["TS"])
            if type(seeds) is not int or not 1 <= seeds <= 70:
                raise ValueError("ALDE seed budget must be 1..70")
            if (
                not isinstance(acquisitions, list)
                or not acquisitions
                or any(type(a) is not str for a in acquisitions)
                or len(set(acquisitions)) != len(acquisitions)
                or not set(acquisitions) <= {"GREEDY", "UCB", "TS"}
            ):
                raise ValueError("Invalid acquisition allocation")
            argv += ["--operation", operation, "--seeds", str(seeds), "--acquisitions", *acquisitions]
        else:
            raise ValueError("Explicit ALDE operation required")
    elif plugin_id == "turbopetase.v1":
        if set(parameters) - {"models", "device"}:
            raise ValueError("unknown case parameter")
        models = parameters.get("models", ["PF01083", "PTM311", "pretrained-model"])
        if (
            not isinstance(models, list)
            or not models
            or len(models) != len(set(models))
            or not set(models) <= {"PF01083", "PTM311", "pretrained-model"}
        ):
            raise ValueError("invalid checkpoint selection")
        device = parameters.get("device", "cuda:0")
        if device not in {"cpu", "cuda:0"}:
            raise ValueError("unsupported device")
        argv += ["--models", *models, "--device", device]
    else:
        raise ValueError("Plugin invocation is not implemented")
    return argv
