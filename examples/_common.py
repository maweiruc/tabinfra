"""Small shared helpers for the runnable examples (not a product CLI)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


IMPLEMENTED_BACKENDS = ("tabicl", "tabpfn")
RESERVED = {
    "backend": "somtab",
    "status": "reserved",
    "reason": "Interface reserved; SOMTab source and weights are not integrated.",
}


def add_common_arguments(parser: argparse.ArgumentParser, *, backend: bool = True) -> None:
    if backend:
        parser.add_argument(
            "--backend", choices=("somtab", "tabicl", "tabpfn", "all"), default="all",
            help="all runs TabICL and TabPFN; SOMTab is reserved and cannot execute",
        )
    parser.add_argument("--device", default=None, help="override the native device in config, e.g. cpu or cuda")
    parser.add_argument("--config", type=Path, help="JSON mapping backend -> task -> native constructor parameters")
    parser.add_argument("--seed", type=int, default=0, help="data and conditional-model random seed (default: 0)")


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object mapping backend to task to parameters")
    for name, tasks in config.items():
        if name not in (*IMPLEMENTED_BACKENDS, "somtab") or not isinstance(tasks, dict):
            raise ValueError(f"invalid backend configuration: {name!r}")
        for task, kwargs in tasks.items():
            if task not in ("classification", "regression") or not isinstance(kwargs, dict):
                raise ValueError(f"invalid task configuration: {name}.{task}")
    return config


def model_kwargs(config: dict[str, Any], backend: str, task: str, args: argparse.Namespace) -> dict[str, Any]:
    # CLI device is an explicit override; an omitted CLI device preserves config/native defaults.
    kwargs = {"n_estimators": 1, "random_state": args.seed}
    kwargs.update(config.get(backend, {}).get(task, {}))
    if args.device is not None:
        kwargs["device"] = args.device
    return kwargs


def selected_backends(value: str) -> tuple[str, ...]:
    return IMPLEMENTED_BACKENDS if value == "all" else (value,)


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def error_report(exc: Exception) -> dict[str, Any]:
    result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    if exc.__cause__ is not None:
        result["cause"] = f"{type(exc.__cause__).__name__}: {exc.__cause__}"
    return result


def print_report(report: dict[str, Any]) -> None:
    print(json.dumps(json_value(report), indent=2, ensure_ascii=False, allow_nan=False), flush=True)
