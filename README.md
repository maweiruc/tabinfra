# TabInfra

**Early Alpha · `0.1.0a1`**

TabInfra is an infrastructure toolkit for Tabular Foundation Models.

The current release includes **Runtime** for prediction (classification and regression) and table generation, and **Adapt** for native fine-tuning, checkpoint export, and loading exported checkpoints in Runtime.

| Capability | TabICL | TabPFN | SOMTab |
| --- | --- | --- | --- |
| Classification and regression | Supported | Supported | Reserved |
| `binned_ar` table generation | Supported | Supported | Reserved |
| Native classification fine-tuning, module freezing, checkpoint export and reload | Supported | Not implemented | Reserved |

SOMTab is a reserved interface; no source or weights are integrated. Selecting it raises `NotImplementedError`.

## Installation

Requires Python 3.11+. From a clone of this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[tabicl,tabpfn,adapt]"
```

Install `.[tabicl]`, `.[tabpfn]`, or `.[adapt]` for individual capabilities; `pip install -e .` installs only the shared interfaces. Native models load on demand. This version is available on GitHub, not PyPI.

Backends are pinned to **TabICL 2.2.0 and TabPFN 9.2.0**. Defaults use TabICL v2 checkpoints dated 2026-02-12 and TabPFN v2 checkpoints. The default TabPFN classifier supports up to 10 classes.

First use downloads weights through the native backend and may require accepting model terms. Capabilities and licenses of custom checkpoints must be checked separately.

## Classification and regression

Replace `"tabicl"` with `"tabpfn"` to use the other backend.

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

Use two-dimensional arrays or DataFrames and a one-dimensional target. Keep feature names and order consistent. Runtime `fit` establishes context without updating model weights; `predict_proba` is available for classification only.

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

**`binned_ar` is a generation baseline using conditional classification over bins, not a native continuous conditional density model.** It samples the first column empirically, then generates subsequent columns from previously sampled values.

**Category encoding can affect results:** conditioning categories receive integer IDs in order of first appearance and are passed as numeric features. Invariance to category-ID permutations is not guaranteed. **No synthetic-data privacy guarantee is provided.**

Input requires a DataFrame with at least 16 rows, unique string column names, and no missing or infinite values. Supported types are integers, floats, booleans, strings, and pandas categories. Use `categorical_features` for integer categories and `column_order` to set generation order. Output preserves supported dtypes and the original column order.

The constructor's `random_state` controls model fitting; `generate(random_state=...)` controls sampling. Constructor and `model_kwargs` seeds must agree. Generator persistence, arbitrary conditional generation, and missing-value imputation are not supported.

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

Adapt uses the native TabICL classifier trainer and supports independent freezing of column, row, and ICL modules. Checkpoints are selected using validation data, not test data. Use a new output directory for each run.

The default **`best.ckpt`** may retain the initial weights. A separate artifact under `updated_checkpoint/` exports the last epoch with verified weight changes; this does not imply better validation performance.

Load it with `Predictor.from_artifact(artifact_dir / "updated_checkpoint", device="cpu")`. Both artifact types require context `fit` after loading. Runtime validates checkpoint paths and hashes, loads the exported weights, and warns on backend-version mismatches. Only load artifacts from trusted sources.

The example uses two epochs because native warmup may leave weights unchanged after a single training step.

## Complete examples

```bash
# Run classification and regression.
python examples/predict.py --backend all --device cpu --output runs/predict
# Generate a table with mixed column types.
python examples/generate.py --backend all --device cpu --output runs/generate
# Train, export, and verify reload of both best and updated checkpoints.
python examples/finetune.py --device cpu --epochs 2 --output runs/finetune
```

Examples use generated datasets and save reports, synthetic CSVs, or checkpoints. `--backend all` runs TabICL and TabPFN, reports SOMTab as `reserved`, and exits nonzero on failure. Scripts accept `--config local_config.json`; see the [configuration template](examples/config.example.json). Keep `--seed` consistent with any configured `random_state`.

## Limitations and next steps

Validation used small datasets on **CPU**, macOS 26.4 / arm64, Python 3.12.14, and PyTorch 2.14.1. CUDA, MPS, other environments, large-table performance, and production stability are unverified. This Early Alpha is intended for experiments and research; APIs and artifact formats may change.

Next priorities are GPU validation and independent quality evaluation, followed by additional backend and adaptation support.

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
