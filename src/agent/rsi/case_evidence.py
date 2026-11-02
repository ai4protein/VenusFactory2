"""Canonical domain facts and exact, source-bound review acceptance."""

from __future__ import annotations

import math

from .protein_model import digest


def canonical_claims(plugin_id, report):
    if report.get("status") != "completed":
        raise ValueError("only completed domain reports support completion claims")
    common = {"job_status": "completed", "full_paper_reproduced": False, "wet_lab_validation": False}
    if plugin_id == "catapro.training.v1":
        from .catapro_training_evidence import canonical_training

        return {**common, **canonical_training(report)}
    if plugin_id == "catapro.full_cv.v1":
        from .catapro_evidence import canonical_catapro

        return {**common, **canonical_catapro(report)}
    if plugin_id == "clean.training.v1":
        from .clean_training_plugin import canonical_training

        return {**common, **canonical_training(report)}
    if plugin_id == "clean.annotation.v1":
        from .clean_annotation import canonical_clean

        return {**common, **canonical_clean(report)}
    if plugin_id == "nucb.raw_analysis.v1":
        from .nucb_reproduction import canonical_nucb

        return {**common, **canonical_nucb(report)}
    if plugin_id == "htfunclib.epinnet.v1":
        from .htfunclib_epinnet import canonical_epinnet

        return {**common, **canonical_epinnet(report)}
    if plugin_id == "htfunclib.rf.v1":
        from .htfunclib_case import canonical_htfunclib

        return {**common, **canonical_htfunclib(report)}
    if plugin_id == "phl7.hrex_validation.v1":
        from .phl7_hrex import canonical_hrex

        return {**common, **canonical_hrex(report)}
    if plugin_id == "phl7.opes_analysis.v1":
        from .phl7_case import canonical_phl7

        return {**common, **canonical_phl7(report)}
    if plugin_id == "mcba.multisubstrate.v1":
        from .mcba_evidence import canonical_mcba

        return {**common, **canonical_mcba(report)}
    if plugin_id == "alde.onehot_dnn.v1":
        if report.get("execution_origin") not in {"new_computation", "verified_existing"}:
            raise ValueError("ALDE computation provenance missing")
        seeds = report.get("seeds")
        proteins = report.get("proteins")
        acquisitions = report.get("acquisitions")
        if (
            type(seeds) is not int
            or not 1 <= seeds <= 70
            or proteins != ["GB1", "TrpB"]
            or not isinstance(acquisitions, list)
            or not acquisitions
            or len(set(acquisitions)) != len(acquisitions)
            or not set(acquisitions) <= {"TS", "UCB", "GREEDY"}
        ):
            raise ValueError("Invalid ALDE allocation scope")
        expected = seeds * len(proteins) * len(acquisitions)
        if report.get("allocations") != expected or report.get("completed_and_verified") != expected:
            raise ValueError("Incomplete independently verified ALDE trajectories")
        if report.get("query_budget") != {"initial": 96, "additional_rounds": 4, "batch_size": 96, "total": 480}:
            raise ValueError("Incomplete temporal query budget")
        summaries = {}
        for row in report.get("summaries", []):
            key = row["protein"] + "/" + row["condition"]
            if key in summaries or row["n_seeds"] != seeds:
                raise ValueError("Duplicate or unbalanced trajectory summary")
            if len(row["mean"]) != 5 or row.get("seed_indices") != list(range(seeds)):
                raise ValueError("Incomplete round or seed coverage")
            mean = row["mean"][-1]
            sem = row["sem"][-1] if row.get("sem") is not None else None
            if (
                not math.isfinite(mean)
                or not 0 <= mean <= 1
                or (sem is not None and (not math.isfinite(sem) or sem < 0))
            ):
                raise ValueError("Invalid measured-fitness summary")
            summaries[key] = {"seeds": seeds, "mean": round(mean, 8), "sem": round(sem, 8) if sem is not None else None}
        expected_keys = {
            p + "/" + a for p in proteins for a in [*acquisitions, "author_random", "unique_random_control"]
        }
        if set(summaries) != expected_keys:
            raise ValueError("Acquisition or control summary missing")
        return {
            **common,
            "computational_scope": "alde_onehot_dnn_four_rounds",
            "execution_origin": report["execution_origin"],
            "source_campaign": report["source_campaign"],
            "source_commit": report["source_commit"],
            "completed_trajectories": expected,
            "queries_per_trajectory": 480,
            "additional_rounds": 4,
            "endpoint": "best measured normalized fitness after 480 query events",
            "endpoint_summaries": summaries,
            "repeat_computation_is_independent_efficacy_evidence": False,
            "new_algorithm_or_agent_superiority_established": False,
        }
    if plugin_id == "rfantibody.benchmark.v1":
        from .rfantibody_benchmark import canonical_benchmark

        return {**common, **canonical_benchmark(report)}
    if plugin_id == "rfantibody.targets.v1":
        from .rfantibody_evidence import validate_target_report

        return {**common, **validate_target_report(report)}
    if plugin_id.startswith("rfantibody"):
        quality = report.get("quality_audit", {})
        if (
            report.get("computational_chain_completed") is not True
            or quality.get("sequence_preservation_verified") is not True
        ):
            raise ValueError("antibody chain or sequence evidence incomplete")
        return {
            **common,
            "computational_scope": "antibody_backbone_sequence_structure",
            "designs": quality["designs"],
            "sequence_preservation_verified": True,
            "structural_filter_passes": quality["additional_interface_passes"],
            "experimental_binding_verified": False,
        }
    if plugin_id != "turbopetase.v1":
        raise ValueError("unsupported domain evidence schema")
    comparisons = report.get("comparisons", [])
    originals = [row for row in comparisons if row["indexing"] == "author_notebook"]
    if not originals or any(row.get("independent_author_function_max_abs_error", 1) > 1e-5 for row in originals):
        raise ValueError("author inference replay not verified")
    overlaps = {}
    for row in originals:
        corrected = next(
            (
                x
                for x in comparisons
                if x["checkpoint"] == row["checkpoint"] and x["indexing"] == "residue_index_corrected"
            ),
            None,
        )
        if corrected is None or row["mask_token_offset"] != 0 or corrected["mask_token_offset"] != 1:
            raise ValueError("paired indexing evidence missing")
        overlaps[row["checkpoint"]] = len(
            {x["position"] for x in row["top10"]} & {x["position"] for x in corrected["top10"]}
        )
    return {
        **common,
        "computational_scope": "released_checkpoint_inference",
        "checkpoints": sorted(row["checkpoint"] for row in originals),
        "source_experimental_table_rows": report["measurement_rows"],
        "experimental_rows_generated_by_inference": False,
        "indexing_issue": "CLS_token_offset",
        "top10_position_overlap": overlaps,
        "duplicate_measurements_separate_from_indexing": True,
        "duplicate_mutations": sorted(report.get("duplicate_mutations", {})),
    }


def validate_review(claims, expected):
    if not isinstance(claims, dict) or set(claims) != set(expected):
        raise ValueError("review must cover exactly the declared evidence fields")
    # JSON digests distinguish false from 0 and prevent adding unsupported prose.
    if digest(claims) != digest(expected):
        different = [key for key in expected if digest(claims[key]) != digest(expected[key])]
        raise ValueError("claims disagree with controller evidence: " + ", ".join(different))
    return {
        "accepted": True,
        "claims": claims,
        "claims_sha256": digest(claims),
        "scope": "validated structured facts; free-form model commentary is not validated evidence",
    }


def render_evidence_report(job_id, claims):
    lines = ["# Verified computational evidence", "", "Job: `" + job_id + "`", ""]
    if claims["computational_scope"] == "released_checkpoint_inference":
        lines += [
            "- Replayed released checkpoints: " + ", ".join(claims["checkpoints"]) + ".",
            "- The author experimental table contains "
            + str(claims["source_experimental_table_rows"])
            + " rows; these measurements were not generated by inference.",
            "- The residue-indexing audit concerns the CLS token offset. Top-ten position overlaps: "
            + str(claims["top10_position_overlap"])
            + ".",
            "- Duplicate mutation measurements ("
            + ", ".join(claims["duplicate_mutations"])
            + ") are a separate source-table issue.",
        ]
    elif claims["computational_scope"] == "rfantibody_paired_full_scale_target":
        lines += [
            "- Generated 200 backbones and 1,600 sequence designs; assessed every sequence under RF2 hotspot proportions 1, 0.1 and 0.",
            "- Independently replayed 4,800 structure assessments with the sequence and backbone denominators kept separate.",
            "- Paired condition summaries: " + str(claims["paired_hotspot_conditions"]),
            "- Input reconstruction and public implementation choices are specified in the fixed target catalog.",
        ]
    elif claims["computational_scope"] == "catapro_verified_features_to_fifty_author_training_fits":
        lines += [
            "- Computation origin: " + claims["execution_origin"] + ".",
            "- Verified all five ten-fold stages, including separate activity-cohort kcat and Km pretraining.",
            "- Starting features were reconstructed earlier from all raw sequences and SMILES with pinned encoders.",
            "- Independent full-row prediction replay and author output parity passed for 50 newly trained checkpoints.",
            "- Metrics: " + str(claims["metrics"]),
            "- Author validation folds also select checkpoints; these scores are not an unbiased new holdout.",
            "- Reconstruction decisions: " + str(claims["reconstruction_decisions"]),
            "- Original grid search, exact stochastic equivalence, autonomous model invention and biological replication are not established.",
        ]
    elif claims["computational_scope"] == "catapro_raw_sequences_and_smiles_to_all_released_cv_predictions":
        lines += [
            "- Computation origin: " + claims["execution_origin"] + ".",
            "- Replayed all 95,507 released rows across kcat, Km and kcat/Km, using their 30 original-fold checkpoints.",
            "- Every saved prediction was independently recomputed from the frozen features and checkpoint tensors.",
            "- Metrics: " + str(claims["metrics"]),
            "- The author validation folds also selected checkpoints. This is not an unbiased new holdout.",
            "- Original training, autonomous model invention and biological replication have not been established.",
        ]
    elif claims["computational_scope"] == "clean_full_split10_sequence_embeddings_author_training_and_evaluation":
        lines += [
            "- Computed all 22,525 sequence representations and trained both complete author schedules.",
            "- Training: 7,757 split10 sequences, 14,190 augmentations, 2,000 triplet and 1,500 SupConH epochs.",
            "- Independently replayed all 1,156 label predictions in the pinned CLEAN numerical environment.",
            "- Full-union metrics: " + str(claims["metrics"]),
            "- Reuse requires fresh execution of the complete pipeline.",
            "- Publication five-fold CV, autonomous model invention, RSI efficacy and biochemical validation remain unestablished.",
        ]
    elif claims["computational_scope"] == "clean_full_released_cohorts_from_sequences_to_ec_annotation":
        lines += [
            "- Regenerated ESM-1b means for all 578 sequences; replayed both released heads for 1,156 predictions.",
            "- Independent NumPy replay starts at the bound sequence embeddings and validates all EC centers, projections, distances, labels and full-union metrics.",
            "- Out-of-cohort predicted ECs count as false positives in micro metrics and exact match.",
            "- Full-union metrics: " + str(claims["metrics"]),
            "- Original contrastive training, autonomous model invention, new biological validation and RSI efficacy are not established.",
        ]
    elif claims["computational_scope"] == "nucb_raw_counts_to_landscape_and_library_activity_analysis":
        lines += [
            "- Reconstructed all four generations from raw counts: 16 tables and 55,760 unique landscape sequences.",
            "- Recomputed 44 library hit-rate rows and 28 initial-rate fits from 8,428 released timepoints.",
            "- All 17 tables passed native reference comparison; analysis outputs and OLS fits were replayed.",
            "- Limits: " + str(claims["limits"]),
        ]
    elif claims["computational_scope"] == "htfunclib_epinnet_from_published_rosetta_scores_to_library":
        lines += [
            "- Re-trained the author EpiNNet, ranked 91 one-hot features and reconstructed the fixed top-40 sequence library.",
            "- Test evidence: " + str(claims["test_metrics"]),
            "- Published reference agreement and discrepancies: " + str(claims["reference_comparison"]),
            "- Sequence overlap between training and test: " + str(claims["overlap"]),
            "- Rosetta scores are published inputs. New atomistic jobs and an agent-generated model are not claimed.",
        ]
    elif claims["computational_scope"] == "htfunclib_gfp_processed_input_two_stage_training":
        lines += [
            "- Re-trained both published GFP classifiers with 10,000 boosting iterations each.",
            "- Data: 181,433 unique sequences; published random splits and all 74 features were verified.",
            "- Author Boolean-to-string filter issue is preserved: the second stage uses all released rows.",
            "- Test metrics: " + str(claims["test_metrics"]),
            "- Published binary reference comparison: " + str(claims["binary_reference_comparison"]),
            "- This replays processed inputs; upstream Rosetta and raw sequencing were not regenerated.",
            "- The model is the author classifier, not an agent-generated new model or cross-protein generalization result.",
        ]
    elif claims["computational_scope"] == "phl7_wt_topologies_energy_checks_and_four_ps_hrex":
        lines += [
            "- Generated eight WT topologies and verified 24 fixed-frame energy identities.",
            "- Executed eight replicas for four ps each with " + claims["compute_backend"] + ".",
            "- Native audit counted " + str(claims["accepted_swaps"]) + " accepted swaps in 67 pair attempts.",
            "- Short integration evidence only; production stability and convergence remain unestablished.",
            "- Limitations: " + str(claims["limits"]),
        ]
    elif claims["computational_scope"] == "phl7_published_opes_outputs_to_free_energy_grids":
        lines += [
            "- Recomputed all 24 free-energy analyses from published OPES outputs.",
            "- Numerical reference agreement: " + str(claims["reference_agreement_count"]) + "/24.",
            "- Discrepancy allocations: " + str(claims["analysis_discrepancies"]) + ".",
            "- New molecular dynamics were not performed; these are reanalyses of author-generated data.",
            "- Published protocol discrepancies: " + str(claims["published_method_discrepancies"]),
        ]
    elif claims["computational_scope"] == "mcba_released_notebooks_processed_inputs":
        lines += [
            "- Evidence origin: "
            + claims["execution_origin"]
            + "; source campaign: "
            + claims["source_campaign"]
            + ".",
            "- Completed all 25 released notebook tasks: nine product rankings and sixteen 40-architecture comparisons, on one protein scaffold.",
            "- Independent prediction agreement: " + str(claims["independent_prediction_replay_passed"]) + ".",
            "- Overall declared computational agreement: " + str(claims["computational_agreement_passed"]) + ".",
            "- Numerical discrepancy tasks: " + str(claims["numerical_discrepancy_tasks"]) + ".",
            "- Test-metric discrepancy tasks: " + str(claims["metric_discrepancy_tasks"]) + ".",
            "- Historical reference outcomes: " + str(claims["historical_reference_counts"]) + ".",
            "- Historical discrepancy tasks: " + str(claims["historical_discrepancy_tasks"]) + ".",
            "- The starting point is released processed assays and zero-shot tables. Upstream zero-shot features were not regenerated. Repeated computation is not independent efficacy evidence.",
        ]
    elif claims["computational_scope"] == "alde_onehot_dnn_four_rounds":
        lines += [
            "- Evidence origin: "
            + claims["execution_origin"]
            + "; source campaign: "
            + claims["source_campaign"]
            + ".",
            "- Independently verified " + str(claims["completed_trajectories"]) + " complete 480-query trajectories.",
            "- Endpoint: " + claims["endpoint"] + ".",
            "- Endpoint means and SEM by protein/condition: " + str(claims["endpoint_summaries"]) + ".",
            "- Repeated source seeds are not independent efficacy replicates; no new algorithm or agent superiority is established.",
        ]
    else:
        lines += [
            "- Completed backbone, sequence and structure stages for " + str(claims["designs"]) + " designs.",
            "- Sequence preservation was verified; "
            + str(claims["structural_filter_passes"])
            + " designs pass the declared descriptive structural filter.",
            "- Experimental binding has not been verified.",
        ]
    if "target_antigen" in claims:
        lines += [
            "- Antigen: "
            + claims["target_antigen"]
            + "; declared scale: "
            + str(claims["backbones"])
            + " backbones with "
            + str(claims["sequences_per_backbone"])
            + " sequences each.",
            "- Repeated deterministic computation is not an independent biological efficacy replicate.",
        ]
    lines += [
        "- This is not a full-paper replication or wet-lab validation.",
        "",
        "This report is rendered from controller-validated structured facts, not unverified model prose.",
        "",
    ]
    return "\n".join(lines)
