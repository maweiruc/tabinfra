"""Run real native classifiers and regressors through TabInfra Runtime."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from tabinfra.runtime import Predictor
from _common import (
    RESERVED, add_common_arguments, error_report, model_kwargs, print_report,
    read_config, selected_backends, write_report,
)


def run_task(backend: str, task: str, kwargs: dict, seed: int) -> dict:
    start = time.perf_counter()
    predictor = None
    result = {"task": task, "backend": backend, "model_kwargs": kwargs}
    try:
        if task == "classification":
            X, encoded = make_classification(
                n_samples=160, n_features=6, n_informative=4, n_redundant=0, random_state=seed,
            )
            y = np.where(encoded == 0, "class_negative", "class_positive")
            split = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        else:
            X, y = make_regression(n_samples=160, n_features=6, n_informative=4, noise=5.0, random_state=seed)
            split = train_test_split(X, y, test_size=0.25, random_state=seed)
        X_train, X_test, y_train, y_test = split
        predictor = Predictor(backend=backend, task=task, model_kwargs=kwargs).fit(X_train, y_train)
        predictions = predictor.predict(X_test)
        if task == "classification":
            probabilities = predictor.predict_proba(X_test)
            result.update(
                accuracy=float(accuracy_score(y_test, predictions)),
                log_loss=float(log_loss(y_test, probabilities, labels=predictor.classes_)),
                classes=predictor.classes_,
            )
        else:
            result["rmse"] = float(np.sqrt(mean_squared_error(y_test, predictions)))
        result.update(status="passed", train_rows=len(X_train), test_rows=len(X_test), model=predictor.model_info_)
    except Exception as exc:
        result.update(error_report(exc))
    finally:
        del predictor
        gc.collect()
        result["elapsed_seconds"] = time.perf_counter() - start
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    args = parser.parse_args()
    config = read_config(args.config)
    report = {"seed": args.seed, "results": []}
    for backend in selected_backends(args.backend):
        for task in ("classification", "regression"):
            print(f"Running {backend} {task}...", flush=True)
            result = run_task(backend, task, model_kwargs(config, backend, task, args), args.seed)
            report["results"].append(result)
            print_report(result)
            # Persist each result so a later backend failure does not discard evidence.
            write_report(args.output / "report.json", report)
    if args.backend == "all":
        report["reserved"] = [RESERVED]
        print_report(RESERVED)
    write_report(args.output / "report.json", report)
    return int(any(item["status"] != "passed" for item in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
