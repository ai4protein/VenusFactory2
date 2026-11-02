"""Progress and result metadata for generator-based calculation adapters."""
def result_patch(**values):
    # Existing HTTP consumers read value/visible; no UI framework is involved.
    return values

def ignore_progress(*args, **kwargs):
    pass
