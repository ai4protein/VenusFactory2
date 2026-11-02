"""Declared reproduction allocations; scope is retained in registered recipes."""

from copy import deepcopy

CASES = {
    "clean_full_training": {"plugin": "clean.training.v1", "parameters": {}, "job_timeout": 259200},
    "clean_all_cohorts": {"plugin": "clean.annotation.v1", "parameters": {}, "job_timeout": 14400},
    "nucb_raw_analysis": {"plugin": "nucb.raw_analysis.v1", "parameters": {}, "job_timeout": 14400},
    "alde_partial_three_acquisition": {
        "plugin": "alde.onehot_dnn.v1",
        "parameters": {"operation": "reproduce", "seeds": 5, "acquisitions": ["GREEDY", "UCB", "TS"]},
        "job_timeout": 21600,
    },
    "turbopetase_cpu_checkpoints": {
        "plugin": "turbopetase.v1",
        "parameters": {"models": ["PF01083", "PTM311", "pretrained-model"], "device": "cpu"},
        "job_timeout": 7200,
    },
    "catapro_full_training": {
        "plugin": "catapro.training.v1",
        "parameters": {"operation": "reproduce"},
        "job_timeout": 216900,
    },
    "catapro_full_cohorts": {
        "plugin": "catapro.full_cv.v1",
        "parameters": {"operation": "reproduce"},
        "job_timeout": 86400,
    },
    "phl7_wt_hrex_cuda": {"plugin": "phl7.hrex_validation.v1", "parameters": {"backend": "cuda"}, "job_timeout": 1800},
    "phl7_wt_hrex_cpu": {"plugin": "phl7.hrex_validation.v1", "parameters": {"backend": "cpu"}, "job_timeout": 1800},
    "turbopetase_all_checkpoints": {
        "plugin": "turbopetase.v1",
        "parameters": {"models": ["PF01083", "PTM311", "pretrained-model"], "device": "cuda:0"},
        "job_timeout": 3600,
    },
    "vhh_ha_64x4": {
        "plugin": "rfantibody.targets.v1",
        "parameters": {"target": "ha", "backbones": 64, "sequences": 4},
        "job_timeout": 36000,
    },
    "vhh_rsv_site3_64x4": {
        "plugin": "rfantibody.targets.v1",
        "parameters": {"target": "rsv_site3", "backbones": 64, "sequences": 4},
        "job_timeout": 36000,
    },
    "phl7_all_opes": {"plugin": "phl7.opes_analysis.v1", "parameters": {}, "job_timeout": 1800},
    "gfp_epinnet": {"plugin": "htfunclib.epinnet.v1", "parameters": {}, "job_timeout": 3600},
    "mcba_all_tasks": {
        "plugin": "mcba.multisubstrate.v1",
        "parameters": {"operation": "reproduce"},
        "job_timeout": 43200,
    },
    "alde_full_three_acquisition": {
        "plugin": "alde.onehot_dnn.v1",
        "parameters": {"operation": "reproduce", "seeds": 70, "acquisitions": ["GREEDY", "UCB", "TS"]},
        "job_timeout": 86400,
    },
}
CRITERIA = ("verified_reproduction", "registered_recipe", "fresh_session_reuse")


def allocation(case):
    if case not in CASES:
        raise ValueError("Unknown reproduction goal case")
    return deepcopy(CASES[case])
