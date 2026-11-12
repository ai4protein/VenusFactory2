"""Fit scalar target normalization on the training split, then reuse its state."""
from mcp_tools.models.factory.target_transform import fit_target_transform, transform_targets


def normalize_dataset(train_dataset, val_dataset, test_dataset, method="min_max",
                      label_column_name="label", *, return_transform=False, **kwargs):
    """Return three transformed copies; optionally append JSON-only fitted state.

    Holdouts never enter fitting. Existing three-tuple callers remain compatible.
    Empty/nonfinite training labels and undefined standard/min_max scales fail
    explicitly instead of producing invalid targets during optimization.
    """
    state = fit_target_transform(
        [row[label_column_name] for row in train_dataset], method,
        label_column=label_column_name, **kwargs,
    )
    result = []
    for dataset in (train_dataset, val_dataset, test_dataset):
        rows = [dict(row) for row in dataset]
        values = transform_targets([row[label_column_name] for row in rows], state)
        result.append([dict(row, **{label_column_name: float(value)}) for row, value in zip(rows, values)])
    return (*result, state) if return_transform else tuple(result)


def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset, label_column_name="label"):
    return normalize_dataset(train_dataset, val_dataset, test_dataset, "min_max", label_column_name)


def standard_normalize_dataset(train_dataset, val_dataset, test_dataset, label_column_name="label"):
    return normalize_dataset(train_dataset, val_dataset, test_dataset, "standard", label_column_name)


def robust_normalize_dataset(train_dataset, val_dataset, test_dataset, label_column_name="label"):
    return normalize_dataset(train_dataset, val_dataset, test_dataset, "robust", label_column_name)


def log_normalize_dataset(train_dataset, val_dataset, test_dataset, offset=1.0, label_column_name="label"):
    return normalize_dataset(train_dataset, val_dataset, test_dataset, "log", label_column_name, offset=offset)


def quantile_normalize_dataset(train_dataset, val_dataset, test_dataset, n_quantiles=1000, label_column_name="label"):
    return normalize_dataset(train_dataset, val_dataset, test_dataset, "quantile", label_column_name, n_quantiles=n_quantiles)
