#!/usr/bin/env python3
"""Recompute the GPU-only context ladder and objective-mismatch results.

The script implements the policy contract used in the paper:

* five leave-one-session-out folds;
* L1: one global GPU;
* L2: one GPU per workload family, selected by exhaustive enumeration of 2^6 policies;
* L3: one GPU per observed workload-size/shape cell;
* L1/L2 training objective:
  lexicographic minimum of CVaR10, P95, mean log regret, maximum regret,
  and a deterministic policy label;
* held-out-session oracle used only as the regret reference;
* EDP recomputed as runtime * energy from the normalized session values.

Run from the repository root:

    python scripts/compute_context_ladder.py

Optional explicit paths:

    python scripts/compute_context_ladder.py \
        --inputs data/snapshots/t1-analysis-20260730/extracted/deep_research_energy/inputs \
        --output-dir results/context-ladder
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


GPU_PLATFORMS = ("3090", "5060ti")
WORKLOADS = (
    "GEMM",
    "STRIDED_GEMM",
    "AXPY",
    "STREAM",
    "REDUCTION",
    "CONV2D",
)
OBJECTIVES = ("runtime", "energy", "edp")
EXPECTED_SESSIONS = (1, 2, 3, 4, 5)
EXPECTED_CELLS = 51
EXPECTED_GPU_ROWS = EXPECTED_CELLS * len(GPU_PLATFORMS) * len(EXPECTED_SESSIONS)


def require_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename}: missing required columns: {missing}")


def normalize_platform(series: pd.Series) -> pd.Series:
    return series.astype(str).replace({"intel": "INTEL", "amd": "AMD"})


def read_standard_sessions(path: Path, workload: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "platform",
        "session_number",
        "problem_size",
        "runtime_s",
        "energy_j",
    }
    require_columns(frame, required, path.name)
    return pd.DataFrame(
        {
            "workload": workload,
            "size": frame["problem_size"].astype(int),
            "platform": normalize_platform(frame["platform"]),
            "session": frame["session_number"].astype(int),
            "runtime": frame["runtime_s"].astype(float),
            "energy": frame["energy_j"].astype(float),
        }
    )


def load_gpu_sessions(inputs: Path) -> pd.DataFrame:
    parts = [
        read_standard_sessions(inputs / "gemm_all_sessions.csv", "GEMM"),
        read_standard_sessions(
            inputs / "strided_all_sessions.csv", "STRIDED_GEMM"
        ),
        read_standard_sessions(inputs / "stream_all_sessions.csv", "STREAM"),
        read_standard_sessions(
            inputs / "reduction_all_sessions.csv", "REDUCTION"
        ),
    ]

    conv_path = inputs / "conv_sessions.csv"
    conv = pd.read_csv(conv_path)
    require_columns(
        conv,
        {
            "platform",
            "session_number",
            "problem_size",
            "runtime_per_op_s",
            "total_energy_per_op_j",
        },
        conv_path.name,
    )
    parts.append(
        pd.DataFrame(
            {
                "workload": "CONV2D",
                "size": conv["problem_size"].astype(int),
                "platform": normalize_platform(conv["platform"]),
                "session": conv["session_number"].astype(int),
                "runtime": conv["runtime_per_op_s"].astype(float),
                "energy": conv["total_energy_per_op_j"].astype(float),
            }
        )
    )

    axpy_path = inputs / "axpy_sessions.csv"
    axpy = pd.read_csv(axpy_path)
    require_columns(
        axpy,
        {
            "platform",
            "session_number",
            "problem_size",
            "median_time_e2e_op_s",
            "median_device_energy_op_j",
        },
        axpy_path.name,
    )
    parts.append(
        pd.DataFrame(
            {
                "workload": "AXPY",
                "size": axpy["problem_size"].astype(int),
                "platform": normalize_platform(axpy["platform"]),
                "session": axpy["session_number"].astype(int),
                "runtime": axpy["median_time_e2e_op_s"].astype(float),
                "energy": axpy["median_device_energy_op_j"].astype(float),
            }
        )
    )

    sessions = pd.concat(parts, ignore_index=True)
    sessions = sessions[sessions["platform"].isin(GPU_PLATFORMS)].copy()

    # The paper defines EDP from the normalized session-level runtime and energy.
    # Recomputing it here is required to reproduce the frozen EDP L2 values
    # (CVaR10 35.41%, maximum 52.59%).
    sessions["edp"] = sessions["runtime"] * sessions["energy"]

    sessions = sessions.sort_values(
        ["workload", "size", "platform", "session"]
    ).reset_index(drop=True)

    key_columns = ["workload", "size", "platform", "session"]
    if sessions.duplicated(key_columns).any():
        duplicate = sessions.loc[
            sessions.duplicated(key_columns, keep=False), key_columns
        ]
        raise ValueError(
            "Duplicate GPU session rows found:\n"
            + duplicate.to_string(index=False)
        )

    if len(sessions) != EXPECTED_GPU_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_GPU_ROWS} GPU session rows, got {len(sessions)}"
        )

    cells = sessions[["workload", "size"]].drop_duplicates()
    if len(cells) != EXPECTED_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_CELLS} workload-size/shape cells, got {len(cells)}"
        )

    platforms = tuple(sorted(sessions["platform"].unique()))
    if platforms != tuple(sorted(GPU_PLATFORMS)):
        raise ValueError(
            f"Expected GPU platforms {GPU_PLATFORMS}, got {platforms}"
        )

    observed_sessions = tuple(sorted(sessions["session"].unique()))
    if observed_sessions != EXPECTED_SESSIONS:
        raise ValueError(
            f"Expected sessions {EXPECTED_SESSIONS}, got {observed_sessions}"
        )

    counts = sessions.groupby(["workload", "size", "platform"])[
        "session"
    ].nunique()
    if not counts.eq(len(EXPECTED_SESSIONS)).all():
        raise ValueError("Not every GPU cell-platform pair has five sessions")

    return sessions


def empirical_cvar10(regret_pct: np.ndarray) -> float:
    values = np.asarray(regret_pct, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute CVaR10 for an empty array")
    tail_count = max(1, math.ceil(values.size * 0.10))
    return float(np.sort(values)[-tail_count:].mean())


def policy_score(regret_pct: np.ndarray, label: str) -> tuple:
    values = np.asarray(regret_pct, dtype=float)
    return (
        empirical_cvar10(values),
        float(np.quantile(values, 0.95)),
        float(np.log1p(values / 100.0).mean()),
        float(values.max()),
        label,
    )


def pivot_metric(frame: pd.DataFrame, objective: str) -> pd.DataFrame:
    pivot = frame.pivot(
        index=["workload", "size", "session"],
        columns="platform",
        values=objective,
    )
    return pivot.loc[:, list(GPU_PLATFORMS)]


def training_regrets(
    train: pd.DataFrame,
    objective: str,
    chooser: Callable[[str, int], str],
) -> np.ndarray:
    pivot = pivot_metric(train, objective)
    oracle = pivot.min(axis=1).to_numpy(dtype=float)
    selected = np.array(
        [
            float(pivot.loc[index, chooser(str(index[0]), int(index[1]))])
            for index in pivot.index
        ],
        dtype=float,
    )
    return (selected / oracle - 1.0) * 100.0


def select_l1(train: pd.DataFrame, objective: str) -> tuple[str, tuple]:
    candidates: list[tuple[tuple, str]] = []
    for platform in GPU_PLATFORMS:
        regrets = training_regrets(
            train,
            objective,
            lambda _workload, _size, selected=platform: selected,
        )
        candidates.append((policy_score(regrets, platform), platform))
    candidates.sort(key=lambda item: item[0])
    score, platform = candidates[0]
    return platform, score


def l2_label(mapping: dict[str, str]) -> str:
    return "|".join(f"{workload}={mapping[workload]}" for workload in WORKLOADS)


def select_l2(
    train: pd.DataFrame, objective: str
) -> tuple[dict[str, str], tuple]:
    candidates: list[tuple[tuple, dict[str, str]]] = []
    for assignment in itertools.product(
        GPU_PLATFORMS, repeat=len(WORKLOADS)
    ):
        mapping = dict(zip(WORKLOADS, assignment, strict=True))
        label = l2_label(mapping)
        regrets = training_regrets(
            train,
            objective,
            lambda workload, _size, current=mapping: current[workload],
        )
        candidates.append((policy_score(regrets, label), mapping))
    candidates.sort(key=lambda item: item[0])
    score, mapping = candidates[0]
    return mapping, score


def select_l3(
    train: pd.DataFrame, objective: str
) -> dict[tuple[str, int], str]:
    medians = (
        train.groupby(["workload", "size", "platform"])[objective]
        .median()
        .unstack("platform")
        .loc[:, list(GPU_PLATFORMS)]
    )
    mapping: dict[tuple[str, int], str] = {}
    for index, row in medians.iterrows():
        workload, size = str(index[0]), int(index[1])
        mapping[(workload, size)] = min(
            GPU_PLATFORMS,
            key=lambda platform: (float(row[platform]), platform),
        )
    if len(mapping) != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} L3 choices, got {len(mapping)}")
    return mapping


def evaluate_policy(
    test: pd.DataFrame,
    objective: str,
    chooser: Callable[[str, int], str],
    holdout_session: int,
    level: str,
) -> list[dict[str, object]]:
    pivot = pivot_metric(test, objective)
    oracle = pivot.min(axis=1)
    rows: list[dict[str, object]] = []
    for index, row in pivot.iterrows():
        workload = str(index[0])
        size = int(index[1])
        selected = chooser(workload, size)
        selected_value = float(row[selected])
        oracle_value = float(oracle.loc[index])
        rows.append(
            {
                "objective": objective,
                "context_level": level,
                "holdout_session": holdout_session,
                "workload": workload,
                "size": size,
                "selected_platform": selected,
                "selected_value": selected_value,
                "oracle_value": oracle_value,
                "regret_pct": (selected_value / oracle_value - 1.0) * 100.0,
            }
        )
    return rows


def summarize_evaluations(
    evaluations: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in evaluations.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        regrets = group["regret_pct"].to_numpy(dtype=float)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "n_evaluations": int(regrets.size),
                "coverage_within_5_pct": float((regrets <= 5.0).mean() * 100.0),
                "median_regret_pct": float(np.median(regrets)),
                "p95_regret_pct": float(np.quantile(regrets, 0.95)),
                "cvar10_regret_pct": empirical_cvar10(regrets),
                "max_regret_pct": float(regrets.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_context_ladder(
    sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []

    for objective in OBJECTIVES:
        for holdout_session in EXPECTED_SESSIONS:
            train = sessions[sessions["session"] != holdout_session]
            test = sessions[sessions["session"] == holdout_session]

            l1_platform, l1_score = select_l1(train, objective)
            l2_mapping, l2_score = select_l2(train, objective)
            l3_mapping = select_l3(train, objective)

            policy_rows.append(
                {
                    "objective": objective,
                    "holdout_session": holdout_session,
                    "context_level": "L1",
                    "policy": l1_platform,
                    "train_cvar10_regret_pct": l1_score[0],
                    "train_p95_regret_pct": l1_score[1],
                    "train_mean_log_regret": l1_score[2],
                    "train_max_regret_pct": l1_score[3],
                }
            )
            policy_rows.append(
                {
                    "objective": objective,
                    "holdout_session": holdout_session,
                    "context_level": "L2",
                    "policy": l2_label(l2_mapping),
                    "train_cvar10_regret_pct": l2_score[0],
                    "train_p95_regret_pct": l2_score[1],
                    "train_mean_log_regret": l2_score[2],
                    "train_max_regret_pct": l2_score[3],
                }
            )

            evaluation_rows.extend(
                evaluate_policy(
                    test,
                    objective,
                    lambda _workload, _size, platform=l1_platform: platform,
                    holdout_session,
                    "L1",
                )
            )
            evaluation_rows.extend(
                evaluate_policy(
                    test,
                    objective,
                    lambda workload, _size, mapping=l2_mapping: mapping[workload],
                    holdout_session,
                    "L2",
                )
            )
            evaluation_rows.extend(
                evaluate_policy(
                    test,
                    objective,
                    lambda workload, size, mapping=l3_mapping: mapping[
                        (workload, size)
                    ],
                    holdout_session,
                    "L3",
                )
            )

    evaluations = pd.DataFrame(evaluation_rows).sort_values(
        [
            "objective",
            "context_level",
            "holdout_session",
            "workload",
            "size",
        ]
    )
    policies = pd.DataFrame(policy_rows).sort_values(
        ["objective", "context_level", "holdout_session"]
    )
    summary = summarize_evaluations(
        evaluations, ["objective", "context_level"]
    ).sort_values(["objective", "context_level"])

    expected_evaluations = len(OBJECTIVES) * 3 * EXPECTED_CELLS * len(
        EXPECTED_SESSIONS
    )
    if len(evaluations) != expected_evaluations:
        raise ValueError(
            f"Expected {expected_evaluations} context-ladder evaluations, "
            f"got {len(evaluations)}"
        )
    return summary.reset_index(drop=True), evaluations.reset_index(drop=True), policies.reset_index(drop=True)


def compute_objective_mismatch(
    sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for holdout_session in EXPECTED_SESSIONS:
        train = sessions[sessions["session"] != holdout_session]
        test = sessions[sessions["session"] == holdout_session]
        source_mappings = {
            source: select_l3(train, source) for source in OBJECTIVES
        }

        for source_objective in OBJECTIVES:
            mapping = source_mappings[source_objective]
            for evaluation_objective in OBJECTIVES:
                pivot = pivot_metric(test, evaluation_objective)
                oracle = pivot.min(axis=1)
                for index, row in pivot.iterrows():
                    workload = str(index[0])
                    size = int(index[1])
                    selected = mapping[(workload, size)]
                    selected_value = float(row[selected])
                    oracle_value = float(oracle.loc[index])
                    rows.append(
                        {
                            "selection_objective": source_objective,
                            "evaluation_objective": evaluation_objective,
                            "holdout_session": holdout_session,
                            "workload": workload,
                            "size": size,
                            "selected_platform": selected,
                            "selected_value": selected_value,
                            "oracle_value": oracle_value,
                            "regret_pct": (
                                selected_value / oracle_value - 1.0
                            )
                            * 100.0,
                        }
                    )

    evaluations = pd.DataFrame(rows).sort_values(
        [
            "selection_objective",
            "evaluation_objective",
            "holdout_session",
            "workload",
            "size",
        ]
    )
    summary = summarize_evaluations(
        evaluations,
        ["selection_objective", "evaluation_objective"],
    ).sort_values(["selection_objective", "evaluation_objective"])

    expected_evaluations = (
        len(OBJECTIVES)
        * len(OBJECTIVES)
        * EXPECTED_CELLS
        * len(EXPECTED_SESSIONS)
    )
    if len(evaluations) != expected_evaluations:
        raise ValueError(
            f"Expected {expected_evaluations} objective-mismatch evaluations, "
            f"got {len(evaluations)}"
        )
    return summary.reset_index(drop=True), evaluations.reset_index(drop=True)


EXPECTED_CONTEXT = {
    ("runtime", "L1"): (84.31, 0.00, 117.77, 109.19, 124.31),
    ("runtime", "L2"): (84.31, 0.00, 117.77, 109.19, 124.31),
    ("runtime", "L3"): (100.00, 0.00, 0.00, 0.00, 0.00),
    ("energy", "L1"): (73.73, 0.00, 136.55, 146.52, 191.31),
    ("energy", "L2"): (82.35, 0.00, 137.07, 130.21, 185.14),
    ("energy", "L3"): (99.61, 0.00, 0.00, 0.24, 6.20),
    ("edp", "L1"): (72.55, 0.00, 1080.32, 1114.59, 1443.13),
    ("edp", "L2"): (67.45, 0.00, 31.92, 35.41, 52.59),
    ("edp", "L3"): (100.00, 0.00, 0.00, 0.06, 1.53),
}


def verify_frozen_values(
    context_summary: pd.DataFrame,
    mismatch_summary: pd.DataFrame,
) -> None:
    columns = (
        "coverage_within_5_pct",
        "median_regret_pct",
        "p95_regret_pct",
        "cvar10_regret_pct",
        "max_regret_pct",
    )
    indexed = context_summary.set_index(["objective", "context_level"])
    failures: list[str] = []

    for key, expected in EXPECTED_CONTEXT.items():
        actual = tuple(round(float(indexed.loc[key, column]), 2) for column in columns)
        if actual != expected:
            failures.append(f"context {key}: expected {expected}, got {actual}")

    mismatch = mismatch_summary.set_index(
        ["selection_objective", "evaluation_objective"]
    )
    paper_checks = {
        ("edp", "energy", "coverage_within_5_pct"): 56.86,
        ("edp", "energy", "cvar10_regret_pct"): 100.58,
        ("edp", "runtime", "coverage_within_5_pct"): 88.24,
    }
    for (selection, evaluation, column), expected in paper_checks.items():
        actual = round(float(mismatch.loc[(selection, evaluation), column]), 2)
        if actual != expected:
            failures.append(
                f"mismatch {(selection, evaluation, column)}: "
                f"expected {expected}, got {actual}"
            )

    if failures:
        raise RuntimeError(
            "Frozen paper-value verification failed:\n- "
            + "\n- ".join(failures)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--inputs",
        type=Path,
        default=repo_root
        / "data"
        / "snapshots"
        / "t1-analysis-20260730"
        / "extracted"
        / "deep_research_energy"
        / "inputs",
        help="Directory containing the frozen session CSV inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "context-ladder",
        help="Directory receiving generated CSV tables",
    )
    parser.add_argument(
        "--skip-frozen-check",
        action="store_true",
        help="Generate results without checking the frozen paper values",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = args.inputs.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not inputs.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {inputs}")

    sessions = load_gpu_sessions(inputs)
    context_summary, context_evaluations, policies = compute_context_ladder(
        sessions
    )
    mismatch_summary, mismatch_evaluations = compute_objective_mismatch(
        sessions
    )

    if not args.skip_frozen_check:
        verify_frozen_values(context_summary, mismatch_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    context_summary.to_csv(
        output_dir / "context_ladder_summary.csv", index=False
    )
    context_evaluations.to_csv(
        output_dir / "context_ladder_evaluations.csv", index=False
    )
    policies.to_csv(
        output_dir / "context_ladder_fold_policies.csv", index=False
    )
    mismatch_summary.to_csv(
        output_dir / "objective_mismatch_summary.csv", index=False
    )
    mismatch_evaluations.to_csv(
        output_dir / "objective_mismatch_evaluations.csv", index=False
    )

    printable = context_summary.copy()
    for column in [
        "coverage_within_5_pct",
        "median_regret_pct",
        "p95_regret_pct",
        "cvar10_regret_pct",
        "max_regret_pct",
    ]:
        printable[column] = printable[column].map(lambda value: f"{value:.2f}")

    print(printable.to_string(index=False))
    print()
    print("PASS: context-ladder and objective-mismatch values reproduced")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
