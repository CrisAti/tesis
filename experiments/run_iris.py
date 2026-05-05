from __future__ import annotations

import argparse
import json
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from hybrid_transfer_metric import (
    CLMConfig,
    HybridMetricConfig,
    HybridTransferMetric,
    JMDSConfig,
)


SCENARIOS = {
    "original": {
        "natural": np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        "stress": np.array([0.20, 0.40, 0.40]),
    },
    "binary": {
        "natural": np.array([1.0 / 3.0, 2.0 / 3.0]),
        "stress": np.array([0.50, 0.50]),
    },
}

EXPECTED_READINGS = {
    ("original", "natural"): (
        "Expected: local factor L=1 because Iris is already balanced; T is "
        "mainly limited by the Versicolor/Virginica overlap in G."
    ),
    ("original", "stress"): (
        "Expected: L and T drop because the target reduces Setosa, a compact "
        "class whose samples are expensive to reassign."
    ),
    ("binary", "natural"): (
        "Expected: G and T improve versus three-class Iris because Setosa vs "
        "Otra is structurally cleaner."
    ),
    ("binary", "stress"): (
        "Expected: G remains high, but L drops because samples from Otra must "
        "be assigned into Setosa to reach a 50/50 split."
    ),
}


def _load_variant(variant: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    iris = load_iris()
    X = iris.data
    y = iris.target
    if variant == "original":
        return X, y, [str(name) for name in iris.target_names]
    if variant == "binary":
        y_binary = np.where(y == 0, 0, 1)
        return X, y_binary, ["setosa", "otra"]
    raise ValueError(f"Unsupported variant: {variant}")


def _format_vector(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _run_one(variant: str, scenario: str, as_json: bool) -> dict:
    X, y, label_names = _load_variant(variant)
    metric = HybridTransferMetric(
        HybridMetricConfig(
            clm=CLMConfig(n_permutations=128, random_state=42),
            jmds=JMDSConfig(n_neighbors=10, cv_folds=5, random_state=42),
            standardize=True,
        )
    )
    result = metric.evaluate(X, y, target_proportions=SCENARIOS[variant][scenario])

    assignment_df = pd.DataFrame(
        result.assignment_matrix,
        index=[f"from_{name}" for name in label_names],
        columns=[f"to_{name}" for name in label_names],
    )

    payload = {
        "variant": variant,
        "scenario": scenario,
        "labels": label_names,
        "transfer_score": result.transfer_score,
        "global_score": result.global_score,
        "local_factor": result.local_factor,
        "local_cost": result.local_cost,
        "moved_fraction": result.moved_fraction,
        "observed_proportions": result.observed_proportions.tolist(),
        "target_proportions": result.target_proportions.tolist(),
        "assignment_matrix": result.assignment_matrix.tolist(),
        "clm_adjusted": {
            name: metric_result.adjusted
            for name, metric_result in result.clm.metrics.items()
        },
        "reading": EXPECTED_READINGS[(variant, scenario)],
    }

    if as_json:
        print(json.dumps(payload, indent=2))
        return payload

    print(f"\nVariant: {variant} | Scenario: {scenario}")
    print("-" * 72)
    print(f"T transfer score : {result.transfer_score:.4f}")
    print(f"G global score   : {result.global_score:.4f}")
    print(f"L local factor   : {result.local_factor:.4f}")
    print(f"Local cost C*    : {result.local_cost:.4f}")
    print(f"Moved fraction   : {result.moved_fraction:.4f}")
    print(f"Observed pi      : {_format_vector(result.observed_proportions)}")
    print(f"Target q         : {_format_vector(result.target_proportions)}")
    print("\nAssignment mass matrix (rows source label, columns target label):")
    print(assignment_df.round(4).to_string())
    print(f"\n{EXPECTED_READINGS[(variant, scenario)]}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Iris experiments for the hybrid CLM-JMDS metric."
    )
    parser.add_argument(
        "--variant",
        choices=["original", "binary"],
        required=True,
        help="Use the original three-class Iris labels or Setosa/Otra binary labels.",
    )
    parser.add_argument(
        "--scenario",
        choices=["natural", "stress", "all"],
        required=True,
        help="Target size constraint to evaluate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text tables.",
    )
    args = parser.parse_args()

    scenarios = ["natural", "stress"] if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        _run_one(args.variant, scenario, as_json=args.json)


if __name__ == "__main__":
    main()
