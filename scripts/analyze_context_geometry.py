#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKLOAD_FILES = {
    "AXPY": "axpy_sessions.csv",
    "GEMM": "gemm_all_sessions.csv",
    "STRIDED_GEMM": "strided_all_sessions.csv",
    "REDUCTION": "reduction_all_sessions.csv",
    "STREAM": "stream_all_sessions.csv",
    "CONV2D": "conv_sessions.csv",
}
METRICS = ("runtime", "energy", "edp")
GPU_PLATFORMS = {"3090", "5060ti"}
SELECTION_RULES = ("mean_log_regret", "tail_aligned_cvar10")
ZERO_TOLERANCE_PCT = 1.0e-9


def normalize_platform(value: object) -> str:
    text = str(value).strip()
    lower = text.lower()
    if lower == "amd":
        return "AMD"
    if lower == "intel":
        return "INTEL"
    if lower in GPU_PLATFORMS:
        return lower
    return text


def _configuration_from_threads(threads: pd.Series) -> pd.Series:
    values = pd.to_numeric(threads, errors="coerce").fillna(-1).astype(int)
    return pd.Series(
        np.where(values < 0, "gpu_resident", values.astype(str) + "T"),
        index=threads.index,
    )


def load_workload(workload: str, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)

    if workload == "AXPY":
        out = pd.DataFrame(
            {
                "workload": workload,
                "size": raw["problem_size"],
                "platform": raw["platform"].map(normalize_platform),
                "configuration": _configuration_from_threads(raw["threads"]),
                "threads": pd.to_numeric(raw["threads"], errors="coerce")
                .fillna(-1)
                .astype(int),
                "session": raw["session_number"].astype(int),
                "runtime": raw["median_time_e2e_op_s"].astype(float),
                "energy": raw["median_total_energy_op_j"].astype(float),
                "power": raw["median_avg_power_w"].astype(float),
            }
        )
    elif workload == "CONV2D":
        out = pd.DataFrame(
            {
                "workload": workload,
                "size": raw["problem_size"],
                "platform": raw["platform"].map(normalize_platform),
                "configuration": raw["configuration"].astype(str),
                "threads": pd.to_numeric(raw["num_threads"], errors="coerce")
                .fillna(-1)
                .astype(int),
                "session": raw["session_number"].astype(int),
                "runtime": raw["runtime_per_op_s"].astype(float),
                "energy": raw["total_energy_per_op_j"].astype(float),
                "power": raw["avg_power_w"].astype(float),
            }
        )
    else:
        out = pd.DataFrame(
            {
                "workload": workload,
                "size": raw["problem_size"],
                "platform": raw["platform"].map(normalize_platform),
                "configuration": raw["configuration"].astype(str),
                "threads": pd.to_numeric(raw["num_threads"], errors="coerce")
                .fillna(-1)
                .astype(int),
                "session": raw["session_number"].astype(int),
                "runtime": raw["runtime_s"].astype(float),
                "energy": raw["energy_j"].astype(float),
                "power": raw["power_w"].astype(float),
            }
        )

    out["size"] = pd.to_numeric(out["size"], errors="raise").astype(int)
    out["edp"] = out["runtime"] * out["energy"]
    keys = [
        "workload",
        "size",
        "platform",
        "configuration",
        "threads",
        "session",
    ]
    return out.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)


def load_sessions(inputs_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for workload, filename in WORKLOAD_FILES.items():
        path = inputs_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
        frames.append(load_workload(workload, path))

    data = pd.concat(frames, ignore_index=True)
    expected = 918 * 5
    if len(data) != expected:
        raise ValueError(
            f"expected {expected} configuration-session rows, found {len(data)}"
        )
    counts = data.groupby("session").size().to_dict()
    expected_counts = {1: 918, 2: 918, 3: 918, 4: 918, 5: 918}
    if counts != expected_counts:
        raise ValueError(f"unexpected rows per session: {counts}")
    return data


def cvar_top_array(values: np.ndarray, fraction: float = 0.10) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    count = max(1, int(math.ceil(clean.size * fraction)))
    return float(np.sort(clean)[-count:].mean())


def cvar_top(values: pd.Series, fraction: float = 0.10) -> float:
    return cvar_top_array(values.dropna().to_numpy(dtype=float), fraction)


def summarize_regret(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        regrets = group["regret_pct"].astype(float)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_session_cell_evaluations": int(len(group)),
                "n_cell_clusters": int(
                    group[["workload", "size"]].drop_duplicates().shape[0]
                ),
                "holdouts_per_cell": int(group["holdout_session"].nunique()),
                "exact_oracle_count": int((regrets.abs() <= ZERO_TOLERANCE_PCT).sum()),
                "exact_oracle_fraction": float(
                    (regrets.abs() <= ZERO_TOLERANCE_PCT).mean()
                ),
                "median_regret_pct": float(regrets.median()),
                "p95_regret_pct": float(regrets.quantile(0.95)),
                "cvar10_regret_pct": cvar_top(regrets),
                "max_regret_pct": float(regrets.max()),
                "coverage_within_5pct": float((regrets <= 5.0).mean()),
                "coverage_within_10pct": float((regrets <= 10.0).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def training_medians(training: pd.DataFrame) -> pd.DataFrame:
    keys = ["workload", "size", "platform", "configuration", "threads"]
    return (
        training.groupby(keys, as_index=False)[list(METRICS)]
        .median()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def best_configuration_per_cell_platform(
    train_med: pd.DataFrame, metric: str
) -> pd.DataFrame:
    keys = ["workload", "size", "platform"]
    indices = train_med.groupby(keys, sort=True)[metric].idxmin()
    return train_med.loc[indices].copy().reset_index(drop=True)


def add_training_regret(
    platform_envelopes: pd.DataFrame, metric: str
) -> pd.DataFrame:
    oracle = (
        platform_envelopes.groupby(["workload", "size"], as_index=False)[metric]
        .min()
        .rename(columns={metric: "training_oracle_value"})
    )
    scored = platform_envelopes.merge(
        oracle, on=["workload", "size"], how="left", validate="many_to_one"
    )
    scored["training_regret_pct"] = 100.0 * (
        scored[metric] / scored["training_oracle_value"] - 1.0
    )
    scored["training_log_ratio"] = np.log(
        scored[metric] / scored["training_oracle_value"]
    )
    return scored


def candidate_score(selected: pd.DataFrame) -> dict[str, float]:
    regrets = selected["training_regret_pct"].to_numpy(dtype=float)
    logs = selected["training_log_ratio"].to_numpy(dtype=float)
    return {
        "training_cvar10_regret_pct": cvar_top_array(regrets),
        "training_p95_regret_pct": float(np.quantile(regrets, 0.95)),
        "training_mean_log_regret": float(np.mean(logs)),
        "training_max_regret_pct": float(np.max(regrets)),
        "training_coverage_within_5pct": float(np.mean(regrets <= 5.0)),
    }


def score_key(score: dict[str, float], rule: str, label: str) -> tuple[object, ...]:
    if rule == "mean_log_regret":
        return (
            score["training_mean_log_regret"],
            score["training_cvar10_regret_pct"],
            score["training_p95_regret_pct"],
            score["training_max_regret_pct"],
            label,
        )
    if rule == "tail_aligned_cvar10":
        return (
            score["training_cvar10_regret_pct"],
            score["training_p95_regret_pct"],
            score["training_mean_log_regret"],
            score["training_max_regret_pct"],
            label,
        )
    raise ValueError(f"unknown selection rule: {rule}")


def complete_platforms_by_workload(scored: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for workload, group in scored.groupby("workload", sort=True):
        n_cells = group["size"].nunique()
        complete = (
            group.groupby("platform")["size"].nunique().loc[lambda x: x == n_cells]
        )
        platforms = sorted(complete.index.astype(str).tolist())
        if not platforms:
            raise ValueError(f"no complete platform for workload {workload}")
        result[str(workload)] = platforms
    return result


def rows_for_mapping(
    scored: pd.DataFrame, mapping: dict[str, str]
) -> pd.DataFrame:
    parts = [
        scored[
            (scored["workload"] == workload)
            & (scored["platform"] == platform)
        ]
        for workload, platform in sorted(mapping.items())
    ]
    selected = pd.concat(parts, ignore_index=True)
    expected = scored[["workload", "size"]].drop_duplicates().shape[0]
    if len(selected) != expected:
        raise ValueError(
            f"policy selected {len(selected)} training cells; expected {expected}: {mapping}"
        )
    return selected


def choose_l1_policy(
    scored: pd.DataFrame, rule: str
) -> tuple[dict[str, str], pd.DataFrame, dict[str, float], int]:
    by_workload = complete_platforms_by_workload(scored)
    common = sorted(set.intersection(*(set(v) for v in by_workload.values())))
    if not common:
        raise ValueError("no platform covers all workloads for L1")

    best: tuple[tuple[object, ...], dict[str, str], pd.DataFrame, dict[str, float]] | None = None
    for platform in common:
        mapping = {workload: platform for workload in by_workload}
        selected = rows_for_mapping(scored, mapping)
        score = candidate_score(selected)
        key = score_key(score, rule, platform)
        candidate = (key, mapping, selected, score)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3], len(common)


def choose_l2_policy(
    scored: pd.DataFrame, rule: str
) -> tuple[dict[str, str], pd.DataFrame, dict[str, float], int]:
    by_workload = complete_platforms_by_workload(scored)
    workloads = sorted(by_workload)
    platform_lists = [by_workload[w] for w in workloads]
    candidate_count = math.prod(len(values) for values in platform_lists)

    best: tuple[tuple[object, ...], dict[str, str], pd.DataFrame, dict[str, float]] | None = None
    for assignment in itertools.product(*platform_lists):
        mapping = dict(zip(workloads, assignment))
        label = "|".join(f"{w}:{mapping[w]}" for w in workloads)
        selected = rows_for_mapping(scored, mapping)
        score = candidate_score(selected)
        key = score_key(score, rule, label)
        candidate = (key, mapping, selected, score)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3], candidate_count


def evaluate_selected(
    test: pd.DataFrame,
    selections: pd.DataFrame,
    metric: str,
    level: str,
    scope: str,
    holdout: int,
    selection_rule: str,
) -> pd.DataFrame:
    config_keys = ["workload", "size", "platform", "configuration", "threads"]
    selected = selections[config_keys].drop_duplicates().copy()
    selected_test = test.merge(selected, on=config_keys, how="inner")
    if len(selected_test) != 51:
        raise ValueError(
            f"{scope}/{metric}/{level}/{selection_rule}/session{holdout}: "
            f"expected 51 selected cells, found {len(selected_test)}"
        )
    oracle = (
        test.groupby(["workload", "size"], as_index=False)[metric]
        .min()
        .rename(columns={metric: "oracle_value"})
    )
    result = selected_test.merge(oracle, on=["workload", "size"], how="left")
    result["selected_value"] = result[metric]
    result["regret_pct"] = 100.0 * (
        result["selected_value"] / result["oracle_value"] - 1.0
    )
    result.insert(0, "scope", scope)
    result.insert(1, "holdout_session", holdout)
    result.insert(2, "metric", metric)
    result.insert(3, "selection_rule", selection_rule)
    result.insert(4, "context_level", level)
    return result[
        [
            "scope",
            "holdout_session",
            "metric",
            "selection_rule",
            "context_level",
            "workload",
            "size",
            "platform",
            "configuration",
            "threads",
            "selected_value",
            "oracle_value",
            "regret_pct",
        ]
    ]


def policy_choice_rows(
    scope: str,
    holdout: int,
    metric: str,
    rule: str,
    level: str,
    mapping: dict[str, str],
) -> list[dict[str, object]]:
    if level == "L1_global_platform":
        platform = next(iter(mapping.values()))
        return [
            {
                "scope": scope,
                "holdout_session": holdout,
                "metric": metric,
                "selection_rule": rule,
                "context_level": level,
                "workload": "ALL",
                "selected_platform": platform,
            }
        ]
    return [
        {
            "scope": scope,
            "holdout_session": holdout,
            "metric": metric,
            "selection_rule": rule,
            "context_level": level,
            "workload": workload,
            "selected_platform": platform,
        }
        for workload, platform in sorted(mapping.items())
    ]


def training_objective_row(
    scope: str,
    holdout: int,
    metric: str,
    rule: str,
    level: str,
    score: dict[str, float],
    candidate_count: int,
    mapping: dict[str, str],
) -> dict[str, object]:
    return {
        "scope": scope,
        "holdout_session": holdout,
        "metric": metric,
        "selection_rule": rule,
        "context_level": level,
        "candidate_count": candidate_count,
        "selected_policy_json": json.dumps(mapping, sort_keys=True),
        **score,
    }


def context_ladder(
    data: pd.DataFrame, scope: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoped = data if scope == "all_platform" else data[data["platform"].isin(GPU_PLATFORMS)]
    result_rows: list[pd.DataFrame] = []
    choice_rows: list[dict[str, object]] = []
    objective_rows: list[dict[str, object]] = []
    oracle_rows: list[pd.DataFrame] = []

    l3_label = (
        "L3_cell_configuration" if scope == "all_platform" else "L3_cell_platform"
    )

    for holdout in range(1, 6):
        training = scoped[scoped["session"] != holdout]
        test = scoped[scoped["session"] == holdout]
        train_med = training_medians(training)

        for metric in METRICS:
            envelopes = best_configuration_per_cell_platform(train_med, metric)
            scored = add_training_regret(envelopes, metric)

            for rule in SELECTION_RULES:
                l1_map, l1_selected, l1_score, l1_candidates = choose_l1_policy(
                    scored, rule
                )
                result_rows.append(
                    evaluate_selected(
                        test,
                        l1_selected,
                        metric,
                        "L1_global_platform",
                        scope,
                        holdout,
                        rule,
                    )
                )
                choice_rows.extend(
                    policy_choice_rows(
                        scope,
                        holdout,
                        metric,
                        rule,
                        "L1_global_platform",
                        l1_map,
                    )
                )
                objective_rows.append(
                    training_objective_row(
                        scope,
                        holdout,
                        metric,
                        rule,
                        "L1_global_platform",
                        l1_score,
                        l1_candidates,
                        l1_map,
                    )
                )

                l2_map, l2_selected, l2_score, l2_candidates = choose_l2_policy(
                    scored, rule
                )
                result_rows.append(
                    evaluate_selected(
                        test,
                        l2_selected,
                        metric,
                        "L2_workload_platform",
                        scope,
                        holdout,
                        rule,
                    )
                )
                choice_rows.extend(
                    policy_choice_rows(
                        scope,
                        holdout,
                        metric,
                        rule,
                        "L2_workload_platform",
                        l2_map,
                    )
                )
                objective_rows.append(
                    training_objective_row(
                        scope,
                        holdout,
                        metric,
                        rule,
                        "L2_workload_platform",
                        l2_score,
                        l2_candidates,
                        l2_map,
                    )
                )

            cell_indices = train_med.groupby(["workload", "size"], sort=True)[metric].idxmin()
            cell_selected = train_med.loc[cell_indices]
            result_rows.append(
                evaluate_selected(
                    test,
                    cell_selected,
                    metric,
                    l3_label,
                    scope,
                    holdout,
                    "cell_metric_minimum",
                )
            )

            oracle_indices = test.groupby(["workload", "size"], sort=True)[metric].idxmin()
            oracle = test.loc[oracle_indices].copy()
            oracle["oracle_value"] = oracle[metric]
            oracle["selected_value"] = oracle[metric]
            oracle["regret_pct"] = 0.0
            oracle.insert(0, "scope", scope)
            oracle.insert(1, "holdout_session", holdout)
            oracle.insert(2, "metric", metric)
            oracle.insert(3, "reference_type", "unattainable_heldout_session_oracle")
            oracle_rows.append(
                oracle[
                    [
                        "scope",
                        "holdout_session",
                        "metric",
                        "reference_type",
                        "workload",
                        "size",
                        "platform",
                        "configuration",
                        "threads",
                        "selected_value",
                        "oracle_value",
                        "regret_pct",
                    ]
                ]
            )

    return (
        pd.concat(result_rows, ignore_index=True),
        pd.DataFrame(choice_rows),
        pd.DataFrame(objective_rows),
        pd.concat(oracle_rows, ignore_index=True),
    )


def objective_mismatch(data: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = data if scope == "all_platform" else data[data["platform"].isin(GPU_PLATFORMS)]
    controls = [
        ("edp", "energy", "EDP_selected_for_energy"),
        ("runtime", "energy", "runtime_selected_for_energy"),
        ("edp", "runtime", "EDP_selected_for_runtime"),
        ("energy", "runtime", "energy_selected_for_runtime"),
    ]
    rows: list[pd.DataFrame] = []
    for holdout in range(1, 6):
        training = scoped[scoped["session"] != holdout]
        test = scoped[scoped["session"] == holdout]
        train_med = training_medians(training)
        for selection_metric, evaluation_metric, control in controls:
            indices = train_med.groupby(["workload", "size"], sort=True)[selection_metric].idxmin()
            selections = train_med.loc[indices]
            evaluated = evaluate_selected(
                test,
                selections,
                evaluation_metric,
                control,
                scope,
                holdout,
                "cell_objective_selection",
            )
            evaluated = evaluated.rename(columns={"context_level": "control"})
            evaluated.insert(4, "selection_metric", selection_metric)
            evaluated.insert(5, "evaluation_metric", evaluation_metric)
            rows.append(evaluated)
    return pd.concat(rows, ignore_index=True)


def speed_power_map(
    configurations_path: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = pd.read_csv(configurations_path)
    gpu = configs[configs["platform"].isin(GPU_PLATFORMS)].copy()
    pivot = gpu.pivot_table(
        index=["workload", "size"],
        columns="platform",
        values=["runtime", "energy", "edp", "power"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{platform}" for metric, platform in pivot.columns]
    cells = pivot.reset_index()
    cells["s_runtime_5060_over_3090"] = cells["runtime_5060ti"] / cells["runtime_3090"]
    cells["r_power_3090_over_5060"] = cells["power_3090"] / cells["power_5060ti"]
    cells["energy_ratio_3090_over_5060"] = cells["energy_3090"] / cells["energy_5060ti"]
    cells["edp_ratio_3090_over_5060"] = cells["edp_3090"] / cells["edp_5060ti"]
    s = cells["s_runtime_5060_over_3090"]
    r = cells["r_power_3090_over_5060"]
    lower = np.minimum(s, s**2)
    upper = np.maximum(s, s**2)
    cells["boundary_region"] = np.where(
        r <= lower,
        "below_both_boundaries",
        np.where(r >= upper, "above_both_boundaries", "between_boundaries"),
    )
    cells["runtime_winner"] = np.where(s >= 1.0, "3090", "5060ti")
    cells["energy_winner"] = np.where(
        cells["energy_ratio_3090_over_5060"] <= 1.0, "3090", "5060ti"
    )
    cells["edp_winner"] = np.where(
        cells["edp_ratio_3090_over_5060"] <= 1.0, "3090", "5060ti"
    )
    cells["large_vector_reduction"] = (
        cells["workload"].isin(["AXPY", "STREAM", "REDUCTION"])
        & (cells["size"] >= 16_000_000)
    )

    summary_rows: list[dict[str, object]] = []
    for region, group in cells.groupby("boundary_region", sort=True):
        summary_rows.append(
            {"group": "all_cells", "region": region, "n_cells": int(len(group))}
        )
    large = cells[cells["large_vector_reduction"]]
    for region, group in large.groupby("boundary_region", sort=True):
        summary_rows.append(
            {
                "group": "large_axpy_stream_reduction",
                "region": region,
                "n_cells": int(len(group)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    x = np.logspace(
        np.log10(max(0.3, cells["s_runtime_5060_over_3090"].min() * 0.85)),
        np.log10(cells["s_runtime_5060_over_3090"].max() * 1.15),
        400,
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for workload, group in cells.groupby("workload", sort=True):
        ax.scatter(
            group["s_runtime_5060_over_3090"],
            group["r_power_3090_over_5060"],
            label=workload,
            alpha=0.72,
            s=42,
        )
    large_group = cells[cells["large_vector_reduction"]]
    ax.scatter(
        large_group["s_runtime_5060_over_3090"],
        large_group["r_power_3090_over_5060"],
        facecolors="none",
        edgecolors="black",
        linewidths=1.3,
        s=105,
        label="Large AXPY/STREAM/REDUCTION",
    )
    ax.plot(x, x, linestyle="--", label=r"Energy boundary $r=s$")
    ax.plot(x, x**2, linestyle=":", label=r"EDP boundary $r=s^2$")
    ax.axvline(1.0, linewidth=1.0, linestyle="-.", label=r"Equal runtime $s=1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Runtime ratio $s=t_{5060}/t_{3090}$")
    ax.set_ylabel(r"Power ratio $r=P_{3090}/P_{5060}$")
    ax.set_title("Speed–Power Decision Map")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "speed_power_decision_map.png", dpi=220)
    plt.close(fig)
    return cells, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--configurations", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    reference = (
        repo
        / "data/snapshots/t1-analysis-20260730"
        / "extracted/deep_research_energy"
    )
    inputs_dir = (args.inputs_dir or (reference / "inputs")).expanduser().resolve()
    configurations = (
        args.configurations or (reference / "outputs/all_918_configurations.csv")
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or repo
        / "results/extended-analysis"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    data = load_sessions(inputs_dir)
    data.to_csv(output_dir / "normalized_4590_session_rows.csv", index=False)

    ladders: list[pd.DataFrame] = []
    choices: list[pd.DataFrame] = []
    objectives: list[pd.DataFrame] = []
    oracles: list[pd.DataFrame] = []
    for scope in ("gpu_only", "all_platform"):
        ladder, choice, objective, oracle = context_ladder(data, scope)
        ladders.append(ladder)
        choices.append(choice)
        objectives.append(objective)
        oracles.append(oracle)

    ladder = pd.concat(ladders, ignore_index=True)
    ladder.to_csv(output_dir / "context_ladder_fold_results.csv", index=False)
    ladder_summary = summarize_regret(
        ladder,
        ["scope", "metric", "selection_rule", "context_level"],
    )
    ladder_summary.to_csv(output_dir / "context_ladder_summary.csv", index=False)

    policy_choices = pd.concat(choices, ignore_index=True)
    policy_choices.to_csv(output_dir / "context_ladder_policy_choices.csv", index=False)

    training_objectives = pd.concat(objectives, ignore_index=True)
    training_objectives.to_csv(
        output_dir / "context_ladder_training_objectives.csv", index=False
    )

    oracle_reference = pd.concat(oracles, ignore_index=True)
    oracle_reference.to_csv(
        output_dir / "context_ladder_oracle_reference.csv", index=False
    )

    mismatch = pd.concat(
        [objective_mismatch(data, "gpu_only"), objective_mismatch(data, "all_platform")],
        ignore_index=True,
    )
    mismatch.to_csv(output_dir / "objective_mismatch_fold_results.csv", index=False)
    mismatch_summary = summarize_regret(
        mismatch,
        ["scope", "control", "selection_metric", "evaluation_metric"],
    )
    mismatch_summary.to_csv(
        output_dir / "objective_mismatch_summary.csv", index=False
    )

    speed_cells, speed_summary = speed_power_map(configurations, output_dir)
    speed_cells.to_csv(output_dir / "speed_power_cells.csv", index=False)
    speed_summary.to_csv(output_dir / "speed_power_summary.csv", index=False)

    manifest = {
        "analysis_version": "context-geometry-v2-tail-aligned",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs_dir": str(inputs_dir),
        "configurations": str(configurations),
        "normalized_session_rows": int(len(data)),
        "configuration_count": int(
            data.drop_duplicates(
                ["workload", "size", "platform", "configuration", "threads"]
            ).shape[0]
        ),
        "context_selection_rules": list(SELECTION_RULES),
        "tail_aligned_lexicographic_objective": [
            "training_cvar10_regret_pct",
            "training_p95_regret_pct",
            "training_mean_log_regret",
            "training_max_regret_pct",
        ],
        "evaluation_structure": {
            "session_cell_evaluations_per_scope_metric_level": 255,
            "cell_clusters": 51,
            "holdouts_per_cell": 5,
            "independence_note": (
                "The 255 values are 51 cells times five held-out sessions; "
                "evaluations within a cell are not independent."
            ),
        },
        "oracle_reference": (
            "L4 is written separately as an unattainable held-out-session oracle "
            "and is not included among deployable ladder levels."
        ),
        "context_ladder_rows": int(len(ladder)),
        "objective_mismatch_rows": int(len(mismatch)),
        "speed_power_cells": int(len(speed_cells)),
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "EXTENDED_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"PASS: extended analysis v2 written to {output_dir}")
    print("Normalized session rows: 4590")
    print("Context rules: mean_log_regret + tail_aligned_cvar10")
    print("L4 oracle: separate reference file")
    print("Objective mismatch controls: 4")
    print("Speed-power cells: 51")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
