#!/usr/bin/env python3
"""
Run the KCondor CRT pipeline on a dataset.

Usage
-----
    python run_condor.py --dataset adults
    python run_condor.py --dataset propublica --B 100 --K 3 --sample-size 500
    python run_condor.py --dataset law --output results/law_results.pkl
    python run_condor.py --csv my_data.csv --y-col target --z-cols gender race --z-groups "gender" "race" "gender,race"

Supported built-in datasets: adults, propublica, law, edu.
You can also supply a custom CSV via --csv.
"""

import argparse
import os
import sys
import pickle
import time
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve paths – works wherever the repo is cloned
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from utils import load_dataset, provide_x_z, convert_scores_to_ranks
from crt_cgan import ConditionalGAN, crt_calibration_efficient
from methods import Kcondor_v2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_z_groups(z_group_strings: list[str]) -> list[list[str]]:
    """Parse z-group arguments like "gender,race" into [['gender','race']]."""
    return [g.split(",") for g in z_group_strings]


def build_results_table(results_all: dict, features_Z: list) -> pd.DataFrame:
    """Convert results dict → tidy DataFrame with feature_group, method, p_value."""
    rows = []
    for feat_idx, method_results in results_all.items():
        feat_name = "+".join(features_Z[feat_idx])
        for method_name, p_values in method_results.items():
            rows.append({
                "feature_group": feat_name,
                "method": method_name,
                "p_value": p_values[0],
            })
    return pd.DataFrame(rows)


def print_results(df: pd.DataFrame) -> None:
    """Pretty-print the results table."""
    pivot = df.pivot(index="method", columns="feature_group", values="p_value")
    print("\n" + "=" * 60)
    print("  P-values  (rows = method, cols = protected feature group)")
    print("=" * 60)
    print(pivot.to_string(float_format="{:.4f}".format))
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_name: str | None = None,
    csv_path: str | None = None,
    y_col: str | None = None,
    z_cols: list[str] | None = None,
    z_groups: list[list[str]] | None = None,
    K: int = 5,
    B: int = 200,
    sample_size: int = 1000,
    output_path: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Run the full KCondor CRT pipeline and return the results dict.

    Parameters
    ----------
    dataset_name : str, optional
        Name of a built-in dataset ('adults', 'propublica', 'law', 'edu').
    csv_path : str, optional
        Path to a custom CSV file (alternative to dataset_name).
    y_col : str, optional
        Target column name (required when csv_path is used).
    z_cols : list[str], optional
        All protected attribute columns (required when csv_path is used).
    z_groups : list[list[str]], optional
        Feature groups to test.  Each element is a list of column names that
        form one group (e.g. [['gender'], ['race'], ['gender','race']]).
        Required when csv_path is used.
    K : int
        Number of cross-validation folds for cGAN training.
    B : int
        Number of CRT bootstrap iterations.
    sample_size : int
        Balanced sample size per class.
    output_path : str, optional
        Where to save the results pickle.  Defaults to
        ``results/condor_<dataset>_B<B>_K<K>_n<sample_size>.pkl``.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``results_all`` mapping feature-group index → {method_name: [p_value]}.
    """
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if csv_path is not None:
        if y_col is None or z_groups is None:
            raise ValueError(
                "When using --csv you must also provide --y-col and --z-groups."
            )
        df = pd.read_csv(csv_path)
        # Basic cleanup
        df = df.replace("?", np.nan).dropna().drop_duplicates()
        features_Z = z_groups
        y_name = y_col
        print(f"Custom CSV '{csv_path}': {df.shape[0]} rows, {df.shape[1]} cols")
    elif dataset_name is not None:
        df, y_name, features_Z = load_dataset(dataset_name)
    else:
        raise ValueError("Provide either --dataset or --csv.")

    print(f"Target column : {y_name}")
    print(f"Feature groups: {features_Z}")
    print(f"Parameters    : K={K}, B={B}, sample_size={sample_size}")

    # ------------------------------------------------------------------
    # 2. Use the true label as ranking score
    # ------------------------------------------------------------------
    preds_o = df[y_name].values

    # ------------------------------------------------------------------
    # 3. CRT loop over feature groups
    # ------------------------------------------------------------------
    results_all: dict[int, dict] = {}
    scoring_function = Kcondor_v2

    t_start = time.time()

    for i, f_p in enumerate(features_Z):
        print(f"\n{'=' * 60}")
        print(f"Feature group {i + 1}/{len(features_Z)}: {f_p}")
        print("=" * 60)

        # Prepare balanced X, Z splits
        X_np, Z_np, y, X_np_all, Z_np_all, y_all = provide_x_z(
            df, y_name=y_name, f_p=f_p,
            sample_size_per_class=sample_size, fz=features_Z,
        )

        # Train K cGAN generators (K-fold)
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        kf_splits = list(kf.split(X_np))

        print(f"Training {K} cGAN generators …")
        trained_generators = []
        for train_idx, _ in tqdm(kf_splits, desc="cGAN folds"):
            generator = ConditionalGAN(
                x_dim=X_np_all.shape[1], z_dim=Z_np_all.shape[1],
            )
            generator.fit(X_np_all[train_idx], Z_np_all[train_idx])
            trained_generators.append(generator)

        # CRT calibration
        print(f"Running CRT with B={B} …")
        p_val = crt_calibration_efficient(
            X_np, Z_np, preds_o,
            scoring_function=scoring_function,
            kf_splits=kf_splits,
            trained_generators=trained_generators,
            B=B,
        )

        results_all[i] = {"KCondor": [p_val]}
        print(f"  KCondor p-value = {p_val:.4f}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    if output_path is None:
        os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
        tag = dataset_name or os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(
            SCRIPT_DIR, "results",
            f"condor_{tag}_B{B}_K{K}_n{sample_size}.pkl",
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results_all, f)
    print(f"Results saved → {output_path}")

    # Also save a human-readable JSON summary
    json_path = output_path.replace(".pkl", ".json")
    summary = {
        "dataset": dataset_name or csv_path,
        "K": K, "B": B, "sample_size": sample_size,
        "elapsed_seconds": round(elapsed, 1),
        "results": {},
    }
    for feat_idx, methods_results in results_all.items():
        feat_name = "+".join(features_Z[feat_idx])
        summary["results"][feat_name] = {
            m: round(pv[0], 6) for m, pv in methods_results.items()
        }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {json_path}")

    # Pretty-print
    df_res = build_results_table(results_all, features_Z)
    print_results(df_res)

    return results_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the KCondor CRT pipeline on a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--dataset", choices=["adults", "propublica", "law", "edu"],
        help="Built-in dataset name.",
    )
    grp.add_argument(
        "--csv", dest="csv_path",
        help="Path to a custom CSV file.",
    )

    parser.add_argument("--y-col", help="Target column (required with --csv).")
    parser.add_argument(
        "--z-cols", nargs="+",
        help="All protected attribute column names (required with --csv).",
    )
    parser.add_argument(
        "--z-groups", nargs="+",
        help=(
            "Feature groups to test, comma-separated within each group. "
            'E.g. --z-groups "gender" "race" "gender,race"'
        ),
    )

    parser.add_argument("--K", type=int, default=5, help="CV folds for cGAN (default: 5).")
    parser.add_argument("--B", type=int, default=200, help="CRT bootstrap iterations (default: 200).")
    parser.add_argument("--sample-size", type=int, default=1000, help="Balanced sample size per class (default: 1000).")
    parser.add_argument("--output", dest="output_path", help="Output pickle path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()

    # Parse z-groups if provided
    z_groups = parse_z_groups(args.z_groups) if args.z_groups else None

    run_pipeline(
        dataset_name=args.dataset,
        csv_path=args.csv_path,
        y_col=args.y_col,
        z_groups=z_groups,
        K=args.K,
        B=args.B,
        sample_size=args.sample_size,
        output_path=args.output_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
