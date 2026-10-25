"""Native TabPFN estimators, with explicit Day-1 v2 checkpoint defaults."""

from pathlib import Path

IMPLEMENTED = True
TASKS = {"classification", "regression"}
FINETUNE_TASKS = set()
DEFAULT_CHECKPOINTS = {
    "classification": "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
    "regression": "tabpfn-v2-regressor.ckpt",
}


def build(task, model_kwargs):
    if task not in TASKS:
        raise ValueError(f"Unsupported TabPFN task: {task}")
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.model_loading import prepend_cache_path
    except ImportError as exc:
        raise ImportError('TabPFN requires pip install "tabinfra[tabpfn]"') from exc
    kwargs = dict(model_kwargs)
    kwargs.setdefault("model_path", prepend_cache_path(DEFAULT_CHECKPOINTS[task]))
    cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    return cls(**kwargs)


def seed_kwargs(model_kwargs, random_state):
    kwargs = dict(model_kwargs)
    if "random_state" in kwargs and kwargs["random_state"] != random_state:
        raise ValueError("Generator random_state conflicts with model_kwargs.random_state")
    kwargs["random_state"] = random_state
    return kwargs


def class_limit(model_kwargs):
    path = model_kwargs.get("model_path", DEFAULT_CHECKPOINTS["classification"])
    if isinstance(path, (str, Path)) and Path(path).name.startswith("tabpfn-v2-classifier"):
        return 10
    # Arbitrary user checkpoints need their native runtime checks; unknown is not unlimited.
    return None


def load_artifact(manifest, artifact_dir, device=None):
    raise NotImplementedError("TabPFN model artifact export/loading is outside Day-1 scope")
