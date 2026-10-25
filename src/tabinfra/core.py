"""Small data and model-artifact contracts shared by Runtime and Adapt."""

from __future__ import annotations

import json
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


class ArtifactCompatibilityWarning(UserWarning):
    """The installed backend has not been verified for this artifact version."""


def check_backend_version(manifest):
    """Warn on any version difference without altering checkpoint/configuration.

    Format-v1 artifacts without provenance remain loadable with an explicit
    warning. Exact version agreement is the only locally verified combination;
    it is not a guarantee of compatibility for arbitrary checkpoints.
    """
    backend = manifest["backend"]
    saved_version = manifest.get("backend_version")
    try:
        installed_version = metadata.version(backend)
    except metadata.PackageNotFoundError as exc:
        raise ImportError(f"Artifact backend {backend!r} is not installed") from exc
    if saved_version is None:
        warnings.warn(
            f"Artifact has no backend_version; compatibility with {backend}=={installed_version} "
            "cannot be verified. Checkpoint and inference configuration are unchanged.",
            ArtifactCompatibilityWarning, stacklevel=3,
        )
    elif saved_version != installed_version:
        warnings.warn(
            f"Artifact was saved with {backend}=={saved_version}, but installed version is "
            f"{installed_version}; checkpoint compatibility is unverified. Use "
            f"{backend}=={saved_version} to reproduce the recorded environment. "
            "Checkpoint and inference configuration are unchanged.",
            ArtifactCompatibilityWarning, stacklevel=3,
        )


def validate_X(X):
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError("X must be a two-dimensional NumPy array or pandas DataFrame")
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError("X must have nonempty rows and features")
    if isinstance(X, pd.DataFrame) and not X.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    return X.shape


def validate_y(y, n_rows):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n_rows:
        raise ValueError("y must be a one-dimensional single target aligned with X")
    if pd.isna(y).any():
        raise ValueError("y must not contain missing values")
    return y


def check_schema(X, n_features, columns):
    n_rows, width = validate_X(X)
    if width != n_features:
        raise ValueError(f"Expected {n_features} features, got {width}")
    if columns is not None:
        if not isinstance(X, pd.DataFrame) or not X.columns.equals(columns):
            raise ValueError("Prediction DataFrame columns and order must match fit")
    return n_rows


def validate_probabilities(probabilities, classes, n_rows):
    classes = np.asarray(classes)
    if classes.ndim != 1 or not len(classes) or not pd.Index(classes).is_unique:
        raise ValueError("classes_ must contain unique labels in probability-column order")
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (n_rows, len(classes)):
        raise ValueError("Probability shape does not match query rows and classes_")
    if not np.isfinite(p).all() or (p < -1e-7).any() or (p > 1 + 1e-6).any():
        raise ValueError("Probabilities must be finite and nonnegative, at most one")
    p = np.maximum(p, 0)
    sums = p.sum(axis=1, keepdims=True)
    if (sums <= 0).any() or not np.allclose(sums, 1, atol=1e-5, rtol=1e-5):
        raise ValueError("Probability rows must sum to one (apart from rounding error)")
    return p / sums


def checkpoint_path(manifest, artifact_dir):
    relative = manifest.get("checkpoint")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("checkpoint must be a relative path inside the artifact directory")
    directory = Path(artifact_dir).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("checkpoint must remain inside the artifact directory")
    if not path.is_file():
        raise FileNotFoundError(f"Artifact checkpoint does not exist: {path}")
    return path


def read_manifest(artifact_dir):
    with (Path(artifact_dir) / "manifest.json").open() as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != 1:
        raise ValueError("Unsupported artifact format_version")
    if manifest.get("artifact_kind") != "model_checkpoint":
        raise ValueError("Artifact must be a model_checkpoint")
    if manifest.get("requires_context_fit") is not True:
        raise ValueError("Model checkpoint artifacts require a fresh context fit")
    from .backends import get_backend

    get_backend(manifest.get("backend"), manifest.get("task"))
    if manifest.get("task") not in {"classification", "regression"}:
        raise ValueError("Invalid artifact task")
    if not isinstance(manifest.get("inference_kwargs"), dict):
        raise ValueError("Artifact inference_kwargs must be an object")
    if "backend_version" in manifest and (
        not isinstance(manifest["backend_version"], str) or not manifest["backend_version"].strip()
    ):
        raise ValueError("Artifact backend_version must be a nonempty string when provided")
    checkpoint_path(manifest, artifact_dir)
    return manifest


def write_manifest(output_dir, manifest):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path
