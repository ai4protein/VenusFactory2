"""Choose a host-configured adapter without importing research or HTTP modules."""
from .access import require_local_execution


def adapter_for(backend, cluster, bin_directory, runner, *, legacy_pbs=False):
    require_local_execution()
    if backend == "pbs":
        from .pbs import PBSAdapter
        return PBSAdapter(cluster, bin_directory, runner, legacy_identity=legacy_pbs)
    if backend == "slurm" and not legacy_pbs:
        from .slurm import SlurmAdapter
        return SlurmAdapter(cluster, bin_directory, runner)
    raise ValueError("Unsupported scheduler backend or legacy mode")
