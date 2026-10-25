"""Small, sequential quantile-binned autoregressive table generator.

The classifiers model bin/category probabilities, not a native continuous
distribution. Floating targets are uniform inside bins; integer targets retain
the empirical within-bin distribution. No estimator is kept after generation.
Categorical prefixes use first-occurrence integer IDs as numeric features, with
no explicit native categorical metadata. Category-ID permutation invariance is
not guaranteed; arbitrary ID order can affect the conditional predictions.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .backends import get_backend, model_info
from .core import validate_probabilities


class Generator:
    """Generate complete tables using a fixed-order ``binned_ar`` baseline.

    ``random_state`` seeds conditional estimators; the separate seed passed to
    :meth:`generate` controls empirical, categorical, and within-bin sampling.
    ``fit`` retains a private copy of the training table, not trained estimators.
    A model seed of ``None`` is resolved once per fit and recorded in
    ``model_random_state_`` so repeated sampling still resets model randomness.
    """

    def __init__(
        self,
        backend="tabicl",
        method="binned_ar",
        n_bins=8,
        categorical_features=None,
        column_order=None,
        random_state=0,
        model_kwargs=None,
    ):
        # Reserved backends must fail even for one-column/constant tables.
        module = get_backend(backend, task="classification")
        if method != "binned_ar":
            raise ValueError("Only method='binned_ar' is supported")
        if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise ValueError("random_state must be an integer or None")
        self.backend = backend
        self.method = method
        self.n_bins = int(n_bins)
        self.categorical_features = categorical_features
        self.column_order = column_order
        self.random_state = random_state
        self.model_kwargs = dict(model_kwargs or {})
        self._model_kwargs = module.seed_kwargs(self.model_kwargs, random_state)
        self._is_fitted = False

    def fit(self, train_df):
        """Validate and copy a complete training table (at least 16 rows)."""
        module = get_backend(self.backend, task="classification")
        self._is_fitted = False
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("Generator.fit requires a pandas DataFrame")
        if train_df.shape[0] < 16 or train_df.shape[1] < 1:
            raise ValueError("Training table requires at least 16 rows and one column")
        if not train_df.columns.is_unique or not all(isinstance(c, str) for c in train_df.columns):
            raise ValueError("Training column names must be unique strings")
        if train_df.isna().any().any():
            raise ValueError("Missing values are not supported by binned_ar")
        columns = train_df.columns.tolist()
        explicit = self._column_list(self.categorical_features, "categorical_features", [])
        unknown = set(explicit) - set(columns)
        if unknown:
            raise ValueError(f"Unknown categorical_features: {sorted(unknown)}")
        order = self._column_list(self.column_order, "column_order", columns)
        if len(order) != len(columns) or set(order) != set(columns):
            raise ValueError("column_order must be a complete permutation of the input columns")
        self.model_random_state_ = (
            int(np.random.SeedSequence().generate_state(1)[0])
            if self.random_state is None else int(self.random_state)
        )
        # Conflict was checked at construction; replace only native seed keys.
        self._model_kwargs.update(module.seed_kwargs({}, self.model_random_state_))
        limit = module.class_limit(self._model_kwargs)
        metadata = {}
        for column in columns:
            series = train_df[column]
            dtype = series.dtype
            is_category = isinstance(dtype, pd.CategoricalDtype)
            is_bool = pd.api.types.is_bool_dtype(dtype)
            is_integer = pd.api.types.is_integer_dtype(dtype)
            is_float = pd.api.types.is_float_dtype(dtype)
            is_text = (
                pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
            ) and all(isinstance(value, str) for value in series)
            if is_category:
                valid_category = all(
                    isinstance(value, (str, bool, Integral, float, np.floating))
                    and not (isinstance(value, (float, np.floating)) and not np.isfinite(value))
                    for value in series.cat.categories
                )
                if not valid_category:
                    raise TypeError(f"Column {column!r}: unsupported category values")
            elif not (is_bool or is_integer or is_float or is_text):
                raise TypeError(f"Column {column!r}: unsupported dtype {dtype}; use numeric or scalar categories")
            if (is_integer or is_float) and not np.isfinite(series.to_numpy(dtype=float)).all():
                raise ValueError(f"Column {column!r}: infinite values are not supported")
            categorical = is_category or is_bool or is_text or column in explicit
            values = series.to_numpy(copy=True)
            unique = pd.unique(series)
            meta = {"dtype": dtype, "values": values, "constant": len(unique) == 1}
            if categorical:
                categories = list(unique)
                if limit is not None and len(categories) > limit:
                    raise ValueError(
                        f"Column {column!r} has {len(categories)} classes; backend {self.backend!r} "
                        f"supports at most {limit} for this configuration"
                    )
                encoding = {value: index for index, value in enumerate(categories)}
                meta.update(
                    kind="categorical", categories=categories, encoding=encoding,
                    labels=np.asarray([encoding[value] for value in values], dtype=np.int64),
                )
            else:
                count = min(self.n_bins, len(unique), max(2, len(series) // 4))
                if limit is not None:
                    count = min(count, limit)
                edges = np.unique(np.quantile(series.to_numpy(dtype=float), np.linspace(0, 1, count + 1)))
                if not np.isfinite(edges).all():
                    raise ValueError(f"Column {column!r}: quantile boundaries are not finite")
                # [left, right), with the last interval closed on both sides.
                labels = np.searchsorted(edges[1:-1], series.to_numpy(dtype=float), side="right")
                bins = np.unique(labels)
                pools = {int(label): values[labels == label] for label in bins}
                intervals = {
                    int(label): (edges[label], edges[min(label + 1, len(edges) - 1)])
                    for label in bins
                }
                meta.update(
                    kind="integer" if is_integer else "float", edges=edges,
                    labels=labels, pools=pools, intervals=intervals,
                    n_effective_bins=len(bins),
                )
            metadata[column] = meta
        self._train_df = train_df.copy(deep=True)
        self.columns_ = columns
        self.column_order_ = order
        self.metadata_ = metadata
        self.effective_bins_ = {
            column: meta["n_effective_bins"]
            for column, meta in metadata.items() if meta["kind"] != "categorical"
        }
        self.model_info_ = None
        self.degeneracies_ = {}
        self._is_fitted = True
        return self

    @staticmethod
    def _column_list(value, name, default):
        if value is None:
            return list(default)
        if isinstance(value, str):
            raise ValueError(f"{name} must be a sequence of column names")
        try:
            result = list(value)
        except TypeError as error:
            raise ValueError(f"{name} must be a sequence of column names") from error
        if not all(isinstance(column, str) for column in result) or len(set(result)) != len(result):
            raise ValueError(f"{name} must contain unique string column names")
        return result

    def _encode_prefix(self, table, prefix):
        encoded = {}
        for column in prefix:
            meta = self.metadata_[column]
            values = table[column]
            if meta["kind"] == "categorical":
                encoded[column] = np.asarray([meta["encoding"][value] for value in values], dtype=np.int64)
            else:
                encoded[column] = values.to_numpy()
        return pd.DataFrame(encoded, columns=prefix)

    def generate(self, n_samples=128, random_state=None):
        """Return a DataFrame in the original column order and supported dtypes."""
        module = get_backend(self.backend, task="classification")
        if not self._is_fitted:
            raise NotFittedError("Generator must be fitted before generate")
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        rng = np.random.default_rng(random_state)
        generated = pd.DataFrame(index=pd.RangeIndex(n_samples))
        paths = {}
        self.degeneracies_ = paths
        self.model_info_ = None
        for index, column in enumerate(self.column_order_):
            meta = self.metadata_[column]
            prefix = self.column_order_[:index]
            if meta["constant"]:
                result = np.repeat(meta["values"][:1], n_samples)
                paths[column] = "constant"
            elif not prefix:
                result = rng.choice(meta["values"], size=n_samples)
                paths[column] = "empirical_first"
            elif meta["kind"] != "categorical" and meta["n_effective_bins"] == 1:
                result = rng.choice(meta["values"], size=n_samples)
                paths[column] = "empirical_single_bin"
            elif all(self.metadata_[name]["constant"] for name in prefix):
                result = rng.choice(meta["values"], size=n_samples)
                paths[column] = "empirical_constant_prefix"
            else:
                train_prefix = self._encode_prefix(self._train_df, prefix)
                query_prefix = self._encode_prefix(generated, prefix)
                estimator = None
                try:
                    estimator = module.build("classification", dict(self._model_kwargs))
                    estimator.fit(train_prefix, meta["labels"].copy())
                    classes = np.asarray(estimator.classes_)
                    expected = set(np.unique(meta["labels"]).tolist())
                    if classes.ndim != 1 or len(classes) != len(expected) or set(classes.tolist()) != expected:
                        raise ValueError("classifier classes_ do not match the target category/bin mapping")
                    probabilities = validate_probabilities(estimator.predict_proba(query_prefix), classes, n_samples)
                    cumulative = np.cumsum(probabilities, axis=1)
                    cumulative[:, -1] = 1.0
                    chosen = classes[(rng.random(n_samples)[:, None] >= cumulative).sum(axis=1)]
                    self.model_info_ = model_info(self.backend, estimator)
                except Exception as error:
                    # Retain the native exception as the cause, with target context.
                    raise RuntimeError(
                        f"Backend {self.backend!r}, generated column {column!r}: {error}"
                    ) from error
                finally:
                    # Release the previous model before constructing the next one.
                    del estimator
                if meta["kind"] == "categorical":
                    result = [meta["categories"][int(label)] for label in chosen]
                else:
                    result = np.empty(n_samples, dtype=meta["values"].dtype)
                    for label in np.unique(chosen):
                        label = int(label)
                        mask = chosen == label
                        if label not in meta["pools"]:
                            raise ValueError(f"Column {column!r}: unknown generated bin {label}")
                        if meta["kind"] == "integer":
                            result[mask] = rng.choice(meta["pools"][label], size=mask.sum())
                        else:
                            left, right = meta["intervals"][label]
                            result[mask] = rng.uniform(left, right, size=mask.sum())
                paths[column] = "conditional"
            generated[column] = pd.Series(result, index=generated.index, dtype=meta["dtype"])
        return generated[self.columns_]
