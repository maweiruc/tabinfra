"""Fine-tune native TabICL, export a checkpoint, and verify Runtime reload."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from tabinfra.adapt import finetune
from tabinfra.runtime import Predictor
from _common import add_common_arguments, error_report, model_kwargs, print_report, read_config, write_report


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reload(artifact_dir, X_context, y_context, X_test, device):
    """Compare a fresh Runtime and native estimator using this exact artifact."""
    from tabicl import TabICLClassifier
    from tabinfra.backends.tabicl import state_sha256
    from tabinfra.core import read_manifest

    manifest = read_manifest(artifact_dir)
    checkpoint = artifact_dir / manifest["checkpoint"]
    before_digest = checkpoint_digest(checkpoint)
    native_kwargs = dict(manifest["inference_kwargs"])
    native_kwargs.update(model_path=str(checkpoint), allow_auto_download=False)
    if device is not None:
        native_kwargs["device"] = device
    native = TabICLClassifier(**native_kwargs).fit(X_context, y_context)
    native_state_digest = state_sha256(native.model_.state_dict())
    native_classes = native.classes_.copy()
    native_proba = native.predict_proba(X_test)
    native_labels = native.predict(X_test)
    del native
    gc.collect()

    predictor = Predictor.from_artifact(artifact_dir, device=device).fit(X_context, y_context)
    # Inspect both native models to catch an accidental reload of base weights.
    loaded_state_digest = state_sha256(predictor._estimator.model_.state_dict())
    if native_state_digest != manifest["state_sha256"] or loaded_state_digest != manifest["state_sha256"]:
        raise AssertionError("loaded model weights do not match this artifact's checkpoint state")
    probabilities = predictor.predict_proba(X_test)
    np.testing.assert_array_equal(predictor.classes_, native_classes)
    np.testing.assert_allclose(probabilities, native_proba, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(predictor.predict(X_test), native_labels)
    after_digest = checkpoint_digest(checkpoint)
    if before_digest != after_digest or before_digest != manifest["checkpoint_sha256"]:
        raise AssertionError("checkpoint changed or disagrees with the artifact manifest")
    return predictor, {
        "native_wrapper_probability_match": True, "classes_match": True,
        "labels_match": True, "checkpoint_sha256": after_digest,
        "native_state_sha256": native_state_digest, "runtime_state_sha256": loaded_state_digest,
        "checkpoint_matches_base": loaded_state_digest == manifest["base_state_sha256"],
        "max_probability_absolute_difference": float(np.max(np.abs(probabilities - native_proba))),
        "rtol": 1e-5, "atol": 1e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, backend=False)
    parser.add_argument("--output", type=Path, default=Path("runs/finetune"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(f"--output must be a new or empty directory; preserving existing results at {args.output}")
    config = read_config(args.config)
    kwargs = model_kwargs(config, "tabicl", "classification", args)
    training = {
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "freeze_col": True, "freeze_row": True, "freeze_icl": False,
        "n_estimators_finetune": 1, "n_estimators_validation": 1,
        "n_estimators_inference": kwargs.get("n_estimators", 1),
        "eval_metric": "log_loss", "save_interval": 1,
    }
    if kwargs.get("device") == "cpu":
        training["amp"] = False
    report = {"backend": "tabicl", "task": "classification", "model_kwargs": kwargs, "train_kwargs": training}
    start = time.perf_counter()
    adapted = updated = None
    try:
        X, encoded = make_classification(
            n_samples=160, n_features=6, n_informative=4, n_redundant=0,
            class_sep=0.8, random_state=args.seed,
        )
        y = np.where(encoded == 0, "negative", "positive")
        X_context, X_other, y_context, y_other = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=args.seed,
        )
        X_validation, X_test, y_validation, y_test = train_test_split(
            X_other, y_other, test_size=0.5, stratify=y_other, random_state=args.seed,
        )
        artifact_dir = finetune(
            backend="tabicl", task="classification", X=X_context, y=y_context,
            validation_data=(X_validation, y_validation), model_kwargs=kwargs,
            train_kwargs=training, output_dir=args.output,
        )
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        checks = json.loads((artifact_dir / "update_checks.json").read_text(encoding="utf-8"))
        report.update(
            artifact_dir=artifact_dir, manifest=manifest, training_update_checks=checks,
            parameter_update_verified=checks["parameter_update_verified"],
            selected_checkpoint_matches_base=checks["selected_checkpoint_matches_base"],
        )
        if not checks["parameter_update_verified"] or not checks["frozen_parameters_unchanged"]:
            raise AssertionError("native training did not verify an update with frozen parameters unchanged")

        # Default deployment still uses best.ckpt, even when validation selects
        # the base. A separate artifact proves changed weights can be deployed.
        adapted, reload_checks = verify_reload(artifact_dir, X_context, y_context, X_test, args.device)
        if reload_checks["checkpoint_matches_base"] != checks["selected_checkpoint_matches_base"]:
            raise AssertionError("selected checkpoint base identity disagrees with the training audit")
        probabilities = adapted.predict_proba(X_test)
        labels = adapted.predict(X_test)
        validation_proba = adapted.predict_proba(X_validation)
        validation_labels = adapted.classes_[np.argmax(validation_proba, axis=1)]
        report.update(
            model=adapted.model_info_, reload=reload_checks,
            test={
                "rows": len(X_test), "accuracy": float(accuracy_score(y_test, labels)),
                "log_loss": float(log_loss(y_test, probabilities, labels=adapted.classes_)),
            },
            validation={
                "rows": len(X_validation), "accuracy": float(accuracy_score(y_validation, validation_labels)),
                "log_loss": float(log_loss(y_validation, validation_proba, labels=adapted.classes_)),
            },
            split={"context_rows": len(X_context), "validation_rows": len(X_validation), "test_rows": len(X_test)},
        )
        del adapted
        adapted = None
        gc.collect()
        updated_dir = artifact_dir / manifest["updated_artifact"]
        updated, updated_checks = verify_reload(updated_dir, X_context, y_context, X_test, args.device)
        if updated_checks["checkpoint_matches_base"]:
            raise AssertionError("updated artifact loaded the base weights instead of changed training weights")
        if updated_checks["runtime_state_sha256"] != checks["updated_state_sha256"]:
            raise AssertionError("updated artifact differs from the audited training checkpoint")
        report.update(status="passed", updated_artifact_dir=updated_dir, updated_checkpoint_reload=updated_checks)
    except Exception as exc:
        report.update(error_report(exc))
    finally:
        del adapted, updated
        gc.collect()
        report["elapsed_seconds"] = time.perf_counter() - start
    write_report(args.output / "report.json", report)
    console_report = dict(report)
    if "training_update_checks" in console_report:
        concise_checks = dict(console_report["training_update_checks"])
        concise_checks["updated_trainable_parameter_count"] = len(concise_checks.pop("updated_trainable_parameters"))
        console_report["training_update_checks"] = concise_checks
    print_report(console_report)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
