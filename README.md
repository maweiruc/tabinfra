# TabInfra

**Early Alpha · `0.1.0a1` · Git tag `v0.1.0-alpha.1`**

TabInfra provides a small, unified Python interface for Tabular Foundation Models. **Runtime** handles classification, regression, and table generation. **Adapt** calls native training implementations and exports model artifacts that Runtime can load. This release is intended for local experiments and research; the API and artifact format may change.

| Capability | TabICL | TabPFN | SOMTab |
| --- | --- | --- | --- |
| Classification and regression | Supported | Supported | Reserved |
| `binned_ar` table generation | Supported | Supported | Reserved |
| Native classification fine-tuning, module freezing, checkpoint export and reload | Supported | Not implemented | Reserved |

SOMTab source and weights are not integrated. Selecting it explicitly raises `NotImplementedError`. In the examples, `--backend all` runs TabICL and TabPFN and reports SOMTab separately as `reserved`.

## Installation

Create a separate Python 3.11+ environment from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[tabicl,tabpfn,adapt]"
```

The base installation, `python -m pip install -e .`, provides the shared interfaces only. Install `.[tabicl]`, `.[tabpfn]`, or `.[adapt]` to select individual capabilities. Backends are imported on demand; `import tabinfra` does not import native models or download weights. This version is available as source on GitHub and has not been published to PyPI.

Backend dependencies are pinned to **TabICL 2.2.0 and TabPFN 9.2.0**. Validation used macOS 26.4 / arm64, Python 3.12.14, and PyTorch 2.14.1. All real-model checks ran on **CPU**; CUDA, MPS, other operating systems, and other Python versions have not been validated.

TabICL defaults to `tabicl-classifier-v2-20260212.ckpt` and `tabicl-regressor-v2-20260212.ckpt`. TabPFN explicitly uses the **v2** checkpoints `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` and `tabpfn-v2-regressor.ckpt`, independently of changes to upstream defaults. The default TabPFN v2 classifier supports up to 10 classes. Capabilities and licenses of custom checkpoints must be checked separately.

The first real-model run downloads weights through the native backend and may require network access or acceptance of model terms. Missing dependencies, weights, or supported devices produce explicit errors. TabInfra does not distribute model weights or call paid remote prediction APIs. To keep caches inside the project:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export TABPFN_MODEL_CACHE_DIR="$PWD/.cache/tabpfn"
export SKB_DATA_DIRECTORY="$PWD/.cache/skrub"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

Set `HF_HUB_OFFLINE=1` only after the weights are fully cached; leave offline mode disabled for the initial download.

## Classification and regression

These examples are ready to run. Replace `"tabicl"` with `"tabpfn"` to use the other backend.

```python
from sklearn.datasets import make_classification, make_regression
from tabinfra.runtime import Predictor

kwargs = {"n_estimators": 1, "random_state": 0, "device": "cpu"}
X, y = make_classification(n_samples=96, n_features=4, n_redundant=0, random_state=0)
classifier = Predictor("tabicl", "classification", kwargs).fit(X[:72], y[:72])
labels = classifier.predict(X[72:])
probabilities = classifier.predict_proba(X[72:])
classes = classifier.classes_  # Probability column j corresponds to classes[j].

Xr, yr = make_regression(n_samples=96, n_features=4, random_state=0)
regressor = Predictor("tabicl", "regression", kwargs).fit(Xr[:72], yr[:72])
values = regressor.predict(Xr[72:])
```

Inputs can be two-dimensional NumPy arrays or pandas DataFrames, with a one-dimensional, single-target label array. Classification preserves the original labels. Call `fit` before prediction and keep the feature count, DataFrame column names, and column order consistent. Runtime `fit` establishes context and native preprocessing without updating model weights. The shared layer does not impose one-hot encoding, normalization, or missing-value imputation. Regression does not provide `predict_proba`.

## Table generation

```python
import numpy as np
import pandas as pd
from tabinfra.runtime import Generator

rng = np.random.default_rng(0)
train_df = pd.DataFrame({
    "group": pd.Categorical(rng.choice(["a", "b"], 64)),
    "value": rng.normal(size=64),
})
generator = Generator(
    "tabicl", method="binned_ar", n_bins=4, random_state=0,
    model_kwargs={"n_estimators": 1, "random_state": 0, "device": "cpu"},
).fit(train_df)
synthetic = generator.generate(n_samples=32, random_state=1)
```

**`binned_ar` is a generation baseline based on conditional classification over bins, not a native continuous conditional density model.** It generates columns in a fixed order: the first column is sampled from its empirical distribution, subsequent categorical columns use conditional class probabilities, and numeric columns use quantile-bin probabilities. Floating-point values are sampled uniformly within a bin; integers are sampled from the observed training integers in that bin according to their empirical frequencies. Probability columns follow the backend's `classes_` mapping. The current target and future columns are excluded from the conditioning features.

**Category encoding can affect generated results.** Conditioning categories are assigned integer IDs in order of first appearance in the training table and passed to the backend as ordinary numeric features. The shared layer does not pass native categorical metadata. The artificial order and distances introduced by this mapping may change conditional probabilities. Using the same mapping during training and sampling does not guarantee invariance to permutations of category IDs.

Input must be a DataFrame with at least 16 rows, unique string column names, and no missing or infinite values. Supported types include integers, floats, booleans, strings, and pandas categories. Integers are treated as numeric by default; specify integer categories with `categorical_features`. Use `column_order` to set a complete generation order. Constants, repeated quantile boundaries, a single effective bin, and constant conditioning prefixes have explicit fallback behavior. Output preserves supported dtypes and the original column order.

The constructor's `random_state` controls conditional models; `generate(random_state=...)` controls sampling. Seeds in the constructor and `model_kwargs` must agree. With an explicit `random_state=None`, each `fit` resolves one model seed and stores it in `model_random_state_`. Determinism of GPU operations depends on the native implementation.

Conditional models are rebuilt for each column, with only one model retained at a time; `fit` keeps a copy of the training table. Generator persistence, arbitrary conditional generation, missing-value imputation, and precise tail modeling are not implemented. **There is no synthetic-data privacy guarantee.** Row-overlap measurements or deduplication do not replace a privacy assessment.

## Native fine-tuning and reload

```python
from sklearn.datasets import make_classification
from tabinfra.adapt import finetune
from tabinfra.runtime import Predictor

X, y = make_classification(n_samples=128, n_features=4, n_redundant=0, random_state=0)
artifact_dir = finetune(
    backend="tabicl", task="classification", X=X[:96], y=y[:96],
    validation_data=(X[96:112], y[96:112]),
    model_kwargs={"device": "cpu", "random_state": 0},
    train_kwargs={
        "epochs": 2, "learning_rate": 1e-5,
        "freeze_col": True, "freeze_row": True, "freeze_icl": False,
        "n_estimators_finetune": 1, "n_estimators_validation": 1,
        "n_estimators_inference": 1, "amp": False,
    },
    output_dir="runs/readme_finetune",  # Must be a new or empty directory.
)
adapted = Predictor.from_artifact(artifact_dir, device="cpu").fit(X[:96], y[:96])
labels = adapted.predict(X[112:])
```

Adapt calls the official `FinetunedTabICLClassifier`. It currently supports TabICL classification only, with independent freezing of the column, row, and ICL modules. Training and inference parameters are separate; unknown or conflicting parameters raise errors. Validation data selects the checkpoint; test data must not participate in selection. This adapter rejects W&B configuration, including an empty dictionary.

Artifacts contain native checkpoints, `manifest.json`, and a training audit. The default selection remains **`best.ckpt`**, which may still contain the initial weights. Reports therefore distinguish `parameter_update_verified` from `selected_checkpoint_matches_base`. A separate checkpoint and manifest under `updated_checkpoint/` export the last epoch whose weights actually changed, verifying that the updated model can be deployed. This artifact does not replace the default best model or imply better validation performance.

Load the updated artifact with `Predictor.from_artifact(artifact_dir / "updated_checkpoint", device="cpu")`. Both artifact types store model weights and require a new context `fit` after loading; artifact directories can be moved as a whole. Loading validates relative paths and the file SHA256, disables automatic downloads, and forces the artifact checkpoint to prevent fallback to base weights. Load artifacts only from trusted sources: a hash detects inconsistent content but does not authenticate its origin.

`ArtifactCompatibilityWarning` is emitted when backend versions differ or an older manifest does not record a version. Compatibility has been validated only with the pinned versions; use the dependencies recorded in an artifact to reproduce it. Small datasets may have only one training step per epoch, and native warmup starts with a zero learning rate, so the example uses two epochs. For a one-epoch update, explicitly set `use_lr_scheduler=False` and still verify that weights actually changed.

## Complete examples

```bash
# Run classification and regression.
python examples/predict.py --backend all --device cpu --output runs/predict
# Generate a table with mixed column types.
python examples/generate.py --backend all --device cpu --output runs/generate
# Train, export, and verify reload of both best and updated checkpoints.
python examples/finetune.py --device cpu --epochs 2 --output runs/finetune
```

The examples use small, programmatically generated datasets and write `report.json` or per-backend JSON reports. Generation also writes CSV files; fine-tuning retains checkpoints and audit records. The fine-tuning example compares native and Runtime weight hashes, classes, labels, and probabilities (`rtol=1e-5, atol=1e-6`); it does not require validation metrics to improve. Use a new output directory when repeating fine-tuning.

Scripts accept `--config local_config.json`; see the [configuration template](examples/config.example.json). Its structure is `backend → task → native constructor parameters`. An explicit `--device` overrides the configuration; omitting it preserves the configured or native default. Generation reads classification settings, and fine-tuning reads TabICL classification settings. When changing `--seed`, also update any explicitly configured `random_state`.

`all` runs both backends sequentially, attempts the second even if the first fails, and returns a nonzero exit code on failure. SOMTab's reserved status does not count as a pass. A report marked `passed` means the real execution path completed, not that generation quality or production suitability has been established. Users can run the three examples above to check their own environment.

## Limitations and next steps

This version has been validated only on small datasets on CPU. GPU support, compatibility across versions, large-table performance, and production stability have not been established. Generation depends on column order, binning, and category encoding. Fine-tuning is limited to TabICL classification; there is no general-purpose training framework. Next priorities are GPU validation and quality evaluation on independent data, followed by SOMTab, generator persistence, and additional native adaptation paths as needed. These capabilities are not part of this release.

## License and third-party models

TabInfra's own wrapper code is licensed under [Apache-2.0](LICENSE). This license does not cover third-party Python packages or model weights. Neither the repository nor its distributions bundle third-party model implementations, weights, or training checkpoints; backend dependencies and weights are obtained separately.

| Component | License scope |
| --- | --- |
| TabICL 2.2.0 package / current v2 weights | The main package uses [BSD-3-Clause](https://github.com/soda-inria/tabicl/blob/v2.2.0/LICENSE); the unused forecast submodule has separate terms. The [current model card](https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/README.md) declares BSD-3-Clause for the weights. |
| TabPFN 9.2.0 Python package | [Apache-2.0](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/LICENSE), with upstream [notices](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/NOTICE) and [component licenses](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/THIRD-PARTY-NOTICES.md). |
| TabPFN v2 classification / regression weights | Prior Labs License 1.1 ([classification](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt), [regression](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt)), with additional attribution and downstream model naming requirements; it is not plain Apache-2.0. |
| SOMTab | Not integrated; its license has not been reviewed. |

**TabPFN backend: Built with PriorLabs-TabPFN.** The TabPFN v2 license has no blanket noncommercial restriction, but commercial use remains subject to its full terms. When Section 10 applies to distribution or making available the work, derivative works, or products and services containing the relevant code or weights, it requires a license copy and prominent attribution; this summary and its links do not replace those obligations. Using its code, weights, or outputs (including synthetic data) to create or improve an AI model made available to others also requires that model's name to begin with `TabPFN` under that section.

Third-party dependencies retain their own licenses. Recheck applicable terms before redistributing dependencies or weights, offering a model service, replacing checkpoints, or upgrading backends. Package and weight versions have separate licenses; these weight notes apply only to the listed v2 defaults and do not establish unrestricted commercial-use rights for other versions.
