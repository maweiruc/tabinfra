"""Native, explicitly validated classification fine-tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import get_backend
from .core import check_schema, validate_X, validate_y


def finetune(*, backend, task, X, y, validation_data, output_dir, model_kwargs=None, train_kwargs=None):
    """Run native fine-tuning and return a model-checkpoint artifact directory.

    Validation data is required and used for native checkpoint selection. The
    returned artifact needs a fresh Runtime fit to establish training context.
    """
    implementation = get_backend(backend, task)
    if task not in implementation.FINETUNE_TASKS:
        raise NotImplementedError(f"{backend} does not implement {task} fine-tuning in this version")
    n_rows, n_features = validate_X(X)
    y = validate_y(y, n_rows)
    if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
        raise ValueError("validation_data must be an explicit (X_val, y_val) pair")
    X_val, y_val = validation_data
    columns = X.columns if isinstance(X, pd.DataFrame) else None
    n_val = check_schema(X_val, n_features, columns)
    y_val = validate_y(y_val, n_val)
    if len(np.unique(y)) < 2:
        raise ValueError("Classification fine-tuning requires at least two training classes")
    if not np.isin(y_val, y).all():
        raise ValueError("Validation labels must be present in the training classes")
    return implementation.finetune(
        task=task,
        X=X,
        y=y,
        validation_data=(X_val, y_val),
        model_kwargs=dict(model_kwargs or {}),
        train_kwargs=dict(train_kwargs or {}),
        output_dir=output_dir,
    )
