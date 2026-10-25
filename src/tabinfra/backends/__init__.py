"""Explicit, lazily imported backend map; no native packages imported here."""

from importlib import import_module, metadata

IMPLEMENTED_BACKENDS = ("tabicl", "tabpfn")
RESERVED_BACKENDS = ("somtab",)


def get_backend(name, task=None):
    if name not in (*IMPLEMENTED_BACKENDS, *RESERVED_BACKENDS):
        raise ValueError(f"Unknown backend {name!r}; choose tabicl, tabpfn or somtab (reserved)")
    module = import_module(f".{name}", __name__)
    if not module.IMPLEMENTED:
        raise NotImplementedError("SOMTab is reserved: source is not integrated in this version")
    if task is not None and task not in module.TASKS:
        raise ValueError(f"Backend {name} does not support task {task!r}")
    return module


def model_info(name, estimator):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "not installed"
    params = estimator.get_params(deep=False)
    checkpoint = params.get("model_path")
    if checkpoint is None or str(checkpoint) == "auto":
        checkpoint = params.get("checkpoint_version", checkpoint)
    return {"backend": name, "backend_version": version, "checkpoint": str(checkpoint)}
