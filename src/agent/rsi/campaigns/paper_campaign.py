"""Admitted paper scopes for persistent, evidence-checked capability accumulation.

This catalog declares executable scopes, not successful experimental outcomes.
Adding a paper never grants arbitrary shell execution or relaxes native validators.
"""

from copy import deepcopy

from agent.rsi.engine.reproduction_goal_contract import allocation

PAPERS = {
    "clean": {
        "title": "Enzyme function prediction using contrastive learning",
        "doi": "10.1126/science.adf2465",
        "domain": "enzyme functional annotation and mining support",
    },
    "nucb": {
        "title": "Engineering of highly active and diverse nuclease enzymes by combining machine learning and ultra-high-throughput screening",
        "doi": "10.1101/2024.03.21.585615",
        "domain": "nuclease engineering and activity landscape analysis",
    },
    "phl7": {
        "title": "Computational engineering of the polyester hydrolase PHL7 for efficient poly(ethylene terephthalate) degradation in biocatalytic recycling processes",
        "doi": "10.1038/s41467-026-70868-4",
        "domain": "PET hydrolase engineering",
    },
    "htfunclib": {
        "title": "Designed active-site library reveals thousands of functional GFP variants",
        "doi": "10.1038/s41467-023-38099-z",
        "domain": "protein library design",
    },
    "alde": {
        "title": "Active learning-assisted directed evolution",
        "doi": "10.1038/s41467-025-55987-8",
        "domain": "active learning and directed evolution",
    },
    "mcba": {
        "title": "Accelerated enzyme engineering by machine-learning guided cell-free expression",
        "doi": "10.1038/s41467-024-55399-0",
        "domain": "multisubstrate enzyme engineering",
    },
    "turbopetase": {
        "title": "Computational redesign of a hydrolase for nearly complete PET depolymerization at industrially relevant high-solids loading",
        "doi": "10.1038/s41467-024-45662-9",
        "domain": "PET hydrolase engineering",
    },
    "rfantibody": {
        "title": "Atomically accurate de novo design of antibodies with RFdiffusion",
        "doi": "10.1038/s41586-025-09721-5",
        "domain": "VHH design",
    },
    "catapro": {
        "title": "Robust enzyme discovery and engineering with deep learning using CataPro",
        "doi": "10.1038/s41467-025-58038-4",
        "domain": "enzyme discovery and kinetic prediction",
    },
}

# Computational steps are explicit, auditable units, not arbitrary percentages of a paper.
SCOPES = {
    "clean_full_training": (
        "clean",
        [
            "regenerate all 22525 training, augmentation and evaluation ESM-1b means",
            "train both author split10 schedules: 2000 triplet and 1500 SupConH epochs",
            "evaluate all 578 cohort records with both trained heads and independently replay 1156 predictions",
        ],
        [
            "publication five-fold cross-validation and p-value inference",
            "new enzyme assays, autonomous model invention and RSI efficacy",
        ],
    ),
    "clean_all_cohorts": (
        "clean",
        [
            "regenerate ESM-1b means for all 578 released new, Price and halogenase sequences",
            "apply both released CLEAN projection heads and EC centers: 1,156 predictions",
            "independently replay network, distances and labels; count extra predicted ECs in full-union metrics",
        ],
        [
            "original contrastive training, cross-validation and p-value inference",
            "new enzyme mining, biochemical activity assays and autonomous new model invention",
        ],
    ),
    "nucb_raw_analysis": (
        "nucb",
        [
            "rebuild all four generations and 16 tables from raw counts",
            "reconstruct the 55,760-sequence landscape",
            "replay 44 library hit rates and 28 initial-rate fits; verify numerical artifacts",
        ],
        [
            "original CNN/VAE training and sequence design",
            "diversity analysis including known G1 discrepancies",
            "unseeded WT-denominator visualization",
            "new sequencing, enzyme expression and activity assays",
        ],
    ),
    "phl7_all_opes": (
        "phl7",
        ["reweight 24 published OPES outputs", "compare numerical reference outputs"],
        ["new production molecular dynamics", "enzyme expression and PET degradation assays"],
    ),
    "phl7_wt_hrex_cpu": (
        "phl7",
        ["WT topology and energy checks", "8 replicas of 4 ps HREX validation"],
        ["production-length sampling and convergence", "mutant and wet-lab validation"],
    ),
    "phl7_wt_hrex_cuda": (
        "phl7",
        ["WT topology and energy checks", "8 replicas of 4 ps HREX validation"],
        ["production-length sampling and convergence", "mutant and wet-lab validation"],
    ),
    "gfp_epinnet": (
        "htfunclib",
        ["train EpiNNet on released Rosetta scores", "independently replay predictions and reconstruct library"],
        ["new Rosetta calculations", "experimental fluorescence screening"],
    ),
    "alde_partial_three_acquisition": (
        "alde",
        [
            "GB1 and TrpB: GREEDY/UCB/TS, 5 seeds each, 30 four-round trajectories",
            "validate allocations and reference discrepancies",
        ],
        ["remaining 65 seeds per strategy and landscape", "ParPgb wet-lab directed evolution"],
    ),
    "alde_full_three_acquisition": (
        "alde",
        [
            "GB1 and TrpB: GREEDY/UCB/TS, 70 seeds each, 420 four-round trajectories",
            "validate allocations and reference discrepancies",
        ],
        ["ParPgb wet-lab directed evolution", "other model families in the publication"],
    ),
    "mcba_all_tasks": (
        "mcba",
        ["recompute all 25 released notebook tasks", "retain numerical reference discrepancies"],
        ["cell-free enzyme expression", "substrate conversion assays"],
    ),
    "turbopetase_all_checkpoints": (
        "turbopetase",
        ["replay three published checkpoints", "check reference data and predictions"],
        ["original model training", "PET depolymerization experiments"],
    ),
    "turbopetase_cpu_checkpoints": (
        "turbopetase",
        ["replay three published checkpoints on CPU", "check reference data and predictions"],
        ["original model training", "PET depolymerization experiments"],
    ),
    "vhh_ha_64x4": (
        "rfantibody",
        ["HA target: 64 backbones and 256 sequence records", "refine and validate generated structures"],
        ["remaining paper targets and full published sampling", "experimental binding and structural determination"],
    ),
    "vhh_rsv_site3_64x4": (
        "rfantibody",
        ["RSV site 3: 64 backbones and 256 sequence records", "refine and validate generated structures"],
        ["remaining paper targets and full published sampling", "experimental binding and structural determination"],
    ),
    "catapro_full_training": (
        "catapro",
        [
            "50 model fits under admitted full training allocation",
            "independently validate training outputs and predictions",
        ],
        ["wet-lab enzyme discovery and mutant activity assays"],
    ),
    "catapro_full_cohorts": (
        "catapro",
        ["rebuild sequence and substrate features", "replay 30 published checkpoints over 95,507 rows"],
        ["original model training", "wet-lab enzyme discovery and mutant activity assays"],
    ),
}

SUITES = {
    "cpu_papers": ("phl7_all_opes", "gfp_epinnet"),
    "enzyme_learning": ("alde_partial_three_acquisition", "mcba_all_tasks", "catapro_full_training"),
    "seven_papers": (
        "phl7_all_opes",
        "gfp_epinnet",
        "alde_full_three_acquisition",
        "mcba_all_tasks",
        "turbopetase_all_checkpoints",
        "vhh_ha_64x4",
        "catapro_full_training",
    ),
}

SUITES["nuclease"] = ("nucb_raw_analysis",)
SUITES["eight_papers"] = (*SUITES["seven_papers"], "nucb_raw_analysis")
SUITES["enzyme_annotation"] = ("clean_all_cohorts",)
SUITES["enzyme_annotation_training"] = ("clean_full_training",)


def paper_case(case):
    if case not in SCOPES:
        raise ValueError("Unknown admitted paper scope")
    paper, covered, omitted = SCOPES[case]
    return deepcopy(
        {
            "case": case,
            "paper_id": paper,
            **PAPERS[paper],
            "allocation": allocation(case),
            "reproduction_level": "partial_computational",
            "covered_steps": covered,
            "excluded_steps": omitted,
            "full_paper_reproduced": False,
        }
    )


def campaign_cases(suite=None, cases=None):
    if cases is not None:
        selected = list(cases)
    elif suite in SUITES:
        selected = list(SUITES[suite])
    else:
        raise ValueError("Choose an admitted paper suite or explicit case list")
    if not 1 <= len(selected) <= 16 or len(set(selected)) != len(selected):
        raise ValueError("A campaign needs 1..16 distinct admitted scopes")
    return [paper_case(case) for case in selected]


def validate_campaign(protocol):
    papers = protocol.get("papers", [])
    if protocol.get("kind") != "paper_campaign.v1" or not papers:
        raise ValueError("Invalid paper campaign protocol")
    if papers != campaign_cases(cases=[p["case"] for p in papers]):
        raise ValueError("Paper scope or allocation changed; explicit migration is required")
    return papers


def admitted_campaign(protocol, state):
    papers = validate_campaign(protocol)
    admission = {"kind": "paper_campaign.v1", "papers": papers}
    if "review_continuation" in protocol:
        admission["review_continuation"] = protocol["review_continuation"]
    if "capability_library" in protocol:
        admission["capability_library"] = protocol["capability_library"]
    if state["definition"].get("admission") != admission:
        raise ValueError("Campaign differs from the original admitted paper list or scope")
    return papers
