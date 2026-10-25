"""TabICL 2.2 native estimators and its classification fine-tuning adapter.

Heavy imports remain inside calls. Training and checkpoint selection belong to
TabICL; TabInfra records the effective configuration and audits native snapshots.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from importlib import metadata
from pathlib import Path
import shutil

IMPLEMENTED = True
TASKS = {"classification", "regression"}
FINETUNE_TASKS = {"classification"}
AUDITED_FINETUNE_VERSION = "2.2.0"
KNOWN_CLASSIFIER_CHECKPOINTS = {
    "tabicl-classifier-v1-20250208.ckpt",
    "tabicl-classifier-v1.1-20250506.ckpt",
    "tabicl-classifier-v2-20260212.ckpt",
}


def build(task, model_kwargs):
    if task not in TASKS:
        raise ValueError(f"TabICL does not support task {task!r}")
    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError as exc:
        raise ImportError("Install TabICL with pip install -e '.[tabicl]'") from exc
    cls = TabICLClassifier if task == "classification" else TabICLRegressor
    return cls(**dict(model_kwargs))


def seed_kwargs(model_kwargs, random_state):
    kwargs = dict(model_kwargs)
    if "random_state" in kwargs and kwargs["random_state"] != random_state:
        raise ValueError("Generator random_state conflicts with TabICL random_state")
    kwargs["random_state"] = random_state
    return kwargs


def class_limit(model_kwargs):
    """Known default head limit; None means no verified overall limit.

    TabICL enables hierarchical/mixed-radix inference above its 10-class head,
    except with KV caching or support_many_classes=False. Custom checkpoints may
    have different heads, so do not assume their limit.
    """
    if model_kwargs.get("model_path") is not None:
        return None
    if model_kwargs.get("checkpoint_version", "tabicl-classifier-v2-20260212.ckpt") not in KNOWN_CLASSIFIER_CHECKPOINTS:
        return None
    if not model_kwargs.get("support_many_classes", True) or model_kwargs.get("kv_cache", False):
        return 10
    return None


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state):
    """Content identity independent of the torch checkpoint container format."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_artifact(manifest, artifact_dir, device=None):
    from ..core import checkpoint_path

    if manifest["backend"] != "tabicl" or manifest["task"] != "classification":
        raise ValueError("This version only loads TabICL classification artifacts")
    path = checkpoint_path(manifest, artifact_dir)
    expected = manifest.get("checkpoint_sha256")
    if expected is not None and _file_sha256(path) != expected:
        raise ValueError("Artifact checkpoint SHA-256 does not match manifest")
    kwargs = dict(manifest["inference_kwargs"])
    kwargs.update(model_path=str(path), allow_auto_download=False)
    if device is not None:
        kwargs["device"] = device
    return build("classification", kwargs)


def _parameter_names(cls):
    return set(inspect.signature(cls).parameters)


def _finetune_kwargs(model_kwargs, train_kwargs, classifier_cls, trainer_cls):
    """Map explicit inference options, rejecting duplicate/overwritten options."""
    model = dict(model_kwargs)
    train = dict(train_kwargs)
    classifier_names = _parameter_names(classifier_cls)
    trainer_names = _parameter_names(trainer_cls)
    unknown = set(model) - classifier_names
    if unknown:
        raise TypeError(f"Unknown TabICL model_kwargs: {sorted(unknown)}")
    unknown = set(train) - trainer_names
    if unknown:
        raise TypeError(f"Unknown TabICL train_kwargs: {sorted(unknown)}")
    # Native verbose governs training logs; inner inference verbosity has its
    # own extra_classifier_kwargs channel.
    shared = trainer_names & classifier_names - {"verbose"}
    conflict = set(model) & set(train) - {"verbose"}
    if conflict:
        raise ValueError(f"Conflicting model_kwargs and train_kwargs: {sorted(conflict)}")
    if "n_estimators" in model:
        count = model.pop("n_estimators")
        if "n_estimators_inference" in train and train["n_estimators_inference"] != count:
            raise ValueError("n_estimators conflicts with n_estimators_inference")
        train["n_estimators_inference"] = count
    extra = dict(train.pop("extra_classifier_kwargs", None) or {})
    unknown = set(extra) - classifier_names
    if unknown:
        raise TypeError(f"Unknown extra_classifier_kwargs: {sorted(unknown)}")
    # Native _create_inner_estimator unconditionally overwrites these values.
    controlled = shared | {"n_estimators"}
    conflict = set(extra) & (controlled | set(model))
    if conflict:
        raise ValueError(f"Conflicting or overwritten extra_classifier_kwargs: {sorted(conflict)}")
    for key, value in model.items():
        if key in shared:
            train[key] = value
        else:
            extra[key] = value
    if extra:
        train["extra_classifier_kwargs"] = extra
    epochs = train.get("epochs", 30)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if train.get("save_interval", 1) != 1:
        raise ValueError("TabInfra update verification requires save_interval=1")
    train["save_interval"] = 1
    # Native TabICL enables W&B for every non-None value, including {}.
    if train.get("wandb_kwargs") is not None:
        raise ValueError("This local-only adapter does not enable W&B logging")
    return train


def _json_value(value):
    """Reject configurations that cannot faithfully survive a JSON manifest."""
    import torch

    if isinstance(value, (Path, torch.device)):
        return str(value)
    raise TypeError(f"Artifact configuration requires JSON values, got {type(value).__name__}")


def _audit_snapshots(trainer, directory):
    import torch

    base = torch.load(trainer.model_path_, map_location="cpu", weights_only=True)["state_dict"]
    parameter_names = {name for name, _ in trainer.model_.named_parameters()}
    frozen = {name for name, parameter in trainer.model_.named_parameters() if not parameter.requires_grad}
    expected_frozen = {
        name for name in parameter_names
        if any(
            getattr(trainer, flag) and name.startswith(prefix + ".")
            for flag, prefix in (("freeze_col", "col_embedder"), ("freeze_row", "row_interactor"),
                                 ("freeze_icl", "icl_predictor"))
        )
    }
    if frozen != expected_frozen:
        raise RuntimeError("Native requires_grad flags do not match requested frozen modules")
    trainable = parameter_names - frozen
    if not trainable:
        raise RuntimeError("Fine-tuning has no trainable parameters")
    updated = set()
    frozen_changed = set()
    snapshots = sorted(directory.glob("epoch*.ckpt"), key=lambda p: int(p.stem[5:]))
    if not snapshots:
        raise RuntimeError("No native epoch checkpoints: real training updates are unverified")
    epoch_reports = []
    updated_checkpoint = None
    updated_state_hash = None
    for path in snapshots:
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        state = snapshot["state_dict"]
        if set(base) != set(state):
            raise RuntimeError("Base and training checkpoint state keys differ")
        changed = {name for name in parameter_names if not torch.equal(base[name], state[name])}
        if any(not torch.isfinite(state[name]).all() for name in parameter_names):
            raise RuntimeError(f"Nonfinite training parameters in {path.name}")
        updated |= changed & trainable
        frozen_changed |= changed & frozen
        if changed & trainable:
            # Keep the most recent changed epoch, independently of native
            # best-by-validation selection (which may restore the base model).
            updated_checkpoint = path
            updated_state_hash = state_sha256(state)
        epoch_reports.append({"checkpoint": path.name, "changed_trainable_parameters": len(changed & trainable)})
        del snapshot, state
    selected = torch.load(directory / "best.ckpt", map_location="cpu", weights_only=True)
    selected_hash = state_sha256(selected["state_dict"])
    memory_hash = state_sha256(trainer.model_.state_dict())
    report = {
        "parameter_update_verified": bool(updated),
        "frozen_parameters_unchanged": not frozen_changed,
        "requested_freeze_flags_verified": True,
        "updated_trainable_parameters": sorted(updated),
        "changed_frozen_parameters": sorted(frozen_changed),
        "frozen_parameter_count": len(frozen),
        "trainable_parameter_count": len(trainable),
        "epoch_checkpoints": epoch_reports,
        "base_state_sha256": state_sha256(base),
        "base_checkpoint_sha256": _file_sha256(trainer.model_path_),
        "selected_checkpoint_matches_base": selected_hash == state_sha256(base),
        "selected_checkpoint_matches_in_memory": selected_hash == memory_hash,
        "selected_state_sha256": selected_hash,
        "checkpoint_sha256": _file_sha256(directory / "best.ckpt"),
        "best_validation_metric": float(selected["best_metric"]) if math.isfinite(selected["best_metric"]) else None,
        "base_checkpoint_path": str(Path(trainer.model_path_).resolve()),
        "updated_checkpoint": updated_checkpoint.name if updated_checkpoint else None,
        "updated_state_sha256": updated_state_hash,
        "updated_checkpoint_sha256": _file_sha256(updated_checkpoint) if updated_checkpoint else None,
    }
    (directory / "update_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    if not updated or frozen_changed or selected_hash != memory_hash:
        raise RuntimeError("Native training update/freeze/checkpoint audit failed; see update_checks.json")
    return report


def _export_updated_checkpoint(directory, manifest, audit):
    """Export the audited changed epoch as a separately portable artifact."""
    from ..core import write_manifest

    destination = directory / "updated_checkpoint"
    destination.mkdir()
    name = audit["updated_checkpoint"]
    shutil.copyfile(directory / name, destination / name)
    if _file_sha256(destination / name) != audit["updated_checkpoint_sha256"]:
        raise RuntimeError("Exported updated checkpoint differs from the audited epoch")
    updated_manifest = {
        **manifest,
        "checkpoint": name,
        "checkpoint_selection": "last_changed_epoch",
        "selection_includes_initial_state": False,
        "selected_checkpoint_matches_base": False,
        "checkpoint_sha256": audit["updated_checkpoint_sha256"],
        "state_sha256": audit["updated_state_sha256"],
    }
    # The full local audit belongs to the training directory; this independent
    # inference artifact needs only its checkpoint, configuration, and hashes.
    updated_manifest.pop("update_checks", None)
    write_manifest(destination, updated_manifest)
    return destination.name


def finetune(*, task, X, y, validation_data, model_kwargs, train_kwargs, output_dir):
    if task not in FINETUNE_TASKS:
        raise NotImplementedError("TabICL Adapt currently supports classification only")
    try:
        from tabicl import FinetunedTabICLClassifier, TabICLClassifier
    except ImportError as exc:
        raise ImportError("Install training dependencies with pip install -e '.[adapt]'") from exc
    version = metadata.version("tabicl")
    if version != AUDITED_FINETUNE_VERSION:
        raise RuntimeError(f"Native checkpoint audit requires tabicl=={AUDITED_FINETUNE_VERSION}; found {version}")
    kwargs = _finetune_kwargs(model_kwargs, train_kwargs, TabICLClassifier, FinetunedTabICLClassifier)
    # Fail before training if required inference configuration is not portable.
    json.dumps(kwargs, default=_json_value)
    directory = Path(output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Choose an empty output_dir to avoid stale checkpoint evidence: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    trainer = FinetunedTabICLClassifier(**kwargs)
    X_val, y_val = validation_data
    trainer.fit(X, y, X_val=X_val, y_val=y_val, output_dir=directory)
    audit = _audit_snapshots(trainer, directory)
    # This native estimator holds the selected best state, not the last epoch.
    inference = trainer._final_estimator_.get_params(deep=False)
    inference.pop("model_path", None)
    inference["allow_auto_download"] = False
    inference = json.loads(json.dumps(inference, default=_json_value))
    manifest = {
        "format_version": 1,
        "artifact_kind": "model_checkpoint",
        "backend": "tabicl",
        "task": "classification",
        "backend_version": version,
        "base_checkpoint": Path(trainer.model_path_).name,
        "base_checkpoint_sha256": audit["base_checkpoint_sha256"],
        "base_state_sha256": audit["base_state_sha256"],
        "base_checkpoint_version": trainer.checkpoint_version,
        "checkpoint": "best.ckpt",
        "checkpoint_selection": "best_validation" if audit["best_validation_metric"] is not None else "final_no_valid_metric",
        "selection_includes_initial_state": audit["best_validation_metric"] is not None,
        "parameter_update_verified": audit["parameter_update_verified"],
        "selected_checkpoint_matches_base": audit["selected_checkpoint_matches_base"],
        "requires_context_fit": True,
        "inference_kwargs": inference,
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "state_sha256": audit["selected_state_sha256"],
        "training_kwargs": json.loads(json.dumps(kwargs, default=_json_value)),
        "update_checks": "update_checks.json",
    }
    from ..core import write_manifest

    manifest["updated_artifact"] = _export_updated_checkpoint(directory, manifest, audit)
    write_manifest(directory, manifest)
    return directory
