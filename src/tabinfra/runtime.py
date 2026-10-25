"""Unified classification/regression without replacing native preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .backends import get_backend, model_info
from .core import check_backend_version, check_schema, read_manifest, validate_X, validate_y, validate_probabilities
from .generation import Generator

__all__ = ["Predictor", "Generator"]


class Predictor:
    def __init__(self, backend, task="classification", model_kwargs=None):
        get_backend(backend, task)
        self.backend = backend
        self.task = task
        self.model_kwargs = dict(model_kwargs or {})
        self._estimator = None
        self._fitted = False

    def fit(self, X, y):
        self._fitted = False
        n_rows, n_features = validate_X(X)
        y = validate_y(y, n_rows)
        if self._estimator is None:
            self._estimator = get_backend(self.backend, self.task).build(self.task, self.model_kwargs)
        self._estimator.fit(X, y)
        if self.task == "classification":
            if not hasattr(self._estimator, "classes_"):
                raise ValueError("Backend did not expose reliable classes_ mapping")
            classes = np.asarray(self._estimator.classes_)
            if classes.ndim != 1 or not pd.Index(classes).is_unique:
                raise ValueError("Invalid backend classes_ mapping")
            if set(classes) != set(y):
                raise ValueError("Backend classes_ do not match original training labels")
            self.classes_ = classes.copy()
        self.n_features_in_ = n_features
        self._columns = X.columns.copy() if isinstance(X, pd.DataFrame) else None
        self.model_info_ = model_info(self.backend, self._estimator)
        self._fitted = True
        return self

    def _check_query(self, X):
        if not self._fitted:
            raise NotFittedError("Call fit(X_context, y_context) before prediction")
        return check_schema(X, self.n_features_in_, self._columns)

    def predict(self, X):
        n_rows = self._check_query(X)
        result = np.asarray(self._estimator.predict(X))
        if result.shape != (n_rows,):
            raise ValueError("Backend predict must return one value per query row")
        if self.task == "classification" and not np.isin(result, self.classes_).all():
            raise ValueError("Backend returned labels outside classes_")
        if self.task == "regression" and not np.isfinite(result).all():
            raise ValueError("Backend returned nonfinite regression predictions")
        return result

    def predict_proba(self, X):
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only available for classification")
        n_rows = self._check_query(X)
        return validate_probabilities(self._estimator.predict_proba(X), self.classes_, n_rows)

    @classmethod
    def from_artifact(cls, artifact_dir, device=None):
        manifest = read_manifest(artifact_dir)
        check_backend_version(manifest)
        obj = cls(manifest["backend"], manifest["task"], manifest["inference_kwargs"])
        backend = get_backend(obj.backend, obj.task)
        obj._estimator = backend.load_artifact(manifest, artifact_dir, device=device)
        obj.manifest_ = manifest
        return obj
