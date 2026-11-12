"""Shared plan verification vocabulary; host evidence determines completion."""
from types import MappingProxyType

VERIFICATION_KINDS = ("artifact", "execution", "data", "model", "revision", "review", "delivery")
VERIFICATION_DESCRIPTIONS = MappingProxyType({
    "artifact": "Checkpoint the actual result files with their unchanged content hashes.",
    "execution": (
        "Successful controlled Python execution or a trusted queued computation record, "
        "with unchanged output files cited by the checkpoint. Writing code or claiming success is not execution."
    ),
    "data": (
        "Host-validated training-data partitions and their checkpointed evaluation file, or a host-validated "
        "measured model experiment. Reading FASTA or computing physicochemical properties is not training-data "
        "partition validation; use execution for a completed computation."
    ),
    "model": "A host-validated measured model generation with unchanged checkpointed experiment/runtime files.",
    "revision": (
        "A measured candidate or ablation linked to its parent baseline, with unchanged checkpointed evidence. "
        "A revision need not outperform the baseline to be measured."
    ),
    "review": "Independent Review has approved the candidate; a generated assertion of approval is not evidence.",
    "delivery": "Record the reviewed deliverable package, not just a proposed file list.",
})
VERIFICATION_DESCRIPTION = "Choose the evidence required for this step. " + " ".join(
    f"{kind}: {VERIFICATION_DESCRIPTIONS[kind]}" for kind in VERIFICATION_KINDS
)
