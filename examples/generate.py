"""Generate a mixed table with real TabICL/TabPFN conditional classifiers."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np
import pandas as pd

from tabinfra.runtime import Generator
from _common import (
    RESERVED, add_common_arguments, error_report, model_kwargs, print_report,
    read_config, selected_backends, write_report,
)


def training_table(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_codes = rng.choice(3, size=128, p=[0.25, 0.5, 0.25])
    signal = np.array([-2.0, 0.0, 2.0])[group_codes] + rng.normal(0, 0.7, len(group_codes))
    count = np.clip(np.rint(4 + 0.8 * signal + rng.normal(0, 0.8, len(signal))), 0, 12).astype(np.int64)
    response = 0.6 * signal + 0.5 * count + rng.normal(0, 0.4, len(signal))
    return pd.DataFrame({
        "signal": signal,
        "group": pd.Categorical(np.array(["low", "middle", "high"])[group_codes]),
        "count": count,
        "response": response,
        "constant": np.ones(len(signal), dtype=bool),
    })


def table_statistics(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include="number")
    categorical = frame.select_dtypes(exclude="number")
    return {
        "rows": len(frame),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "numeric": {
            column: {"mean": float(numeric[column].mean()), "std": float(numeric[column].std(ddof=1))}
            for column in numeric.columns
        },
        "category_frequencies": {
            column: {str(value): float(frequency) for value, frequency in categorical[column].value_counts(normalize=True).items()}
            for column in categorical.columns
        },
        "numeric_correlations": numeric.corr().to_dict(),
    }


def row_overlap(train: pd.DataFrame, generated: pd.DataFrame) -> float:
    real_rows = set(train.itertuples(index=False, name=None))
    return float(np.mean([row in real_rows for row in generated.itertuples(index=False, name=None)]))


def run_backend(backend: str, kwargs: dict, train: pd.DataFrame, args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    generator = None
    report = {
        "backend": backend, "method": "binned_ar", "model_kwargs": kwargs,
        "model_seed": args.seed, "sampling_seed": args.sample_seed, "requested_bins": args.n_bins,
        "training": table_statistics(train),
    }
    try:
        generator = Generator(
            backend=backend, method="binned_ar", n_bins=args.n_bins,
            random_state=args.seed, model_kwargs=kwargs,
        ).fit(train)
        generated = generator.generate(n_samples=args.n_samples, random_state=args.sample_seed)
        csv_path = args.output / f"{backend}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(csv_path, index=False)
        report.update(
            status="passed", csv=str(csv_path), model=generator.model_info_,
            model_seed=generator.model_random_state_,
            column_order=generator.column_order_, output_columns=list(generated.columns),
            effective_bins=generator.effective_bins_, generation_paths=generator.degeneracies_,
            generated=table_statistics(generated), training_row_overlap_ratio=row_overlap(train, generated),
        )
    except Exception as exc:
        report.update(error_report(exc))
    finally:
        del generator
        gc.collect()
        report["elapsed_seconds"] = time.perf_counter() - start
    write_report(args.output / f"{backend}.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("runs/generate"))
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=1)
    args = parser.parse_args()
    config = read_config(args.config)
    train = training_table(args.seed)
    summary = {"results": []}
    for backend in selected_backends(args.backend):
        print(f"Running {backend} binned_ar generation...", flush=True)
        report = run_backend(backend, model_kwargs(config, backend, "classification", args), train, args)
        summary["results"].append(report)
        print_report(report)
        write_report(args.output / "report.json", summary)
    if args.backend == "all":
        summary["reserved"] = [RESERVED]
        print_report(RESERVED)
    write_report(args.output / "report.json", summary)
    return int(any(item["status"] != "passed" for item in summary["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
