#!/usr/bin/env python3
"""Reproduce the two-panel Chapter 5 CPU configuration figure.

Inputs:
  max_thread_policy_regret.csv
  near_free_energy_savings.csv
  bootstrap_intel_reduction_near_free.csv

Outputs:
  results/chapter5/chapter5_cpu_configuration_summary.pdf
  results/chapter5/chapter5_cpu_configuration_summary.png

Run from the repository root:
  python scripts/reproduce_chapter5_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED = {
    "runtime": (7.4701, 100.1792, 295.6953),
    "energy": (38.5172, 101.0781, 458.1979),
    "edp": (63.0559, 286.7213, 2108.7344),
}


def find_input_dir(repo_root: Path, explicit: Path | None) -> Path:
    required = {
        "max_thread_policy_regret.csv",
        "near_free_energy_savings.csv",
        "bootstrap_intel_reduction_near_free.csv",
    }

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            repo_root / "src" / "placement_analysis" / "outputs",
            repo_root
            / "data"
            / "snapshots"
            / "t1-analysis-20260730"
            / "extracted"
            / "deep_research_energy"
            / "outputs",
        ]
    )

    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate.resolve()

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the three required CSV files. Checked:\n" + checked
    )


def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    regrets = pd.read_csv(input_dir / "max_thread_policy_regret.csv")
    near = pd.read_csv(input_dir / "near_free_energy_savings.csv")
    bootstrap = pd.read_csv(
        input_dir / "bootstrap_intel_reduction_near_free.csv"
    )

    required_regret = {"metric", "regret_pct"}
    required_near = {
        "workload",
        "size",
        "platform",
        "threshold_pct",
        "chosen_cfg",
        "runtime_opt_cfg",
        "runtime_penalty_pct",
        "energy_saving_pct",
    }
    required_bootstrap = {
        "size",
        "runtime_penalty_point_pct",
        "runtime_penalty_ci95_low",
        "runtime_penalty_ci95_high",
        "energy_saving_point_pct",
        "energy_saving_ci95_low",
        "energy_saving_ci95_high",
    }

    for name, frame, required in [
        ("max_thread_policy_regret.csv", regrets, required_regret),
        ("near_free_energy_savings.csv", near, required_near),
        (
            "bootstrap_intel_reduction_near_free.csv",
            bootstrap,
            required_bootstrap,
        ),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

    selected = near[
        np.isclose(near["threshold_pct"], 1.0)
        & (near["energy_saving_pct"] > 0.0)
    ].copy()

    selected = selected.sort_values(
        ["platform", "workload", "size"]
    ).reset_index(drop=True)

    if len(selected) != 11:
        raise RuntimeError(
            f"Expected 11 energy-saving alternatives at 1%, got {len(selected)}"
        )
    if int((selected["platform"] == "INTEL").sum()) != 10:
        raise RuntimeError("Expected 10 Intel alternatives")
    if int((selected["platform"] == "AMD").sum()) != 1:
        raise RuntimeError("Expected one AMD alternative")
    if selected["workload"].nunique() != 5:
        raise RuntimeError("Expected alternatives from five workloads")

    highlighted = selected[
        (selected["platform"] == "INTEL")
        & (selected["workload"] == "REDUCTION")
        & (selected["size"] == 256_000_000)
    ]
    if len(highlighted) != 1:
        raise RuntimeError("Expected exactly one Intel REDUCTION 256M row")

    boot = bootstrap[bootstrap["size"] == 256_000_000]
    if len(boot) != 1:
        raise RuntimeError(
            "Expected exactly one bootstrap row for Intel REDUCTION 256M"
        )

    return regrets, selected, boot.iloc[0]


def summarize_regrets(regrets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("runtime", "energy", "edp"):
        values = regrets.loc[
            regrets["metric"].str.lower() == metric, "regret_pct"
        ].to_numpy(float)

        if len(values) != 102:
            raise RuntimeError(
                f"Expected 102 regret values for {metric}, got {len(values)}"
            )

        row = {
            "metric": metric,
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }
        rows.append(row)

        expected = EXPECTED[metric]
        actual = (row["median"], row["p95"], row["maximum"])
        if not np.allclose(actual, expected, rtol=0.0, atol=5e-4):
            raise RuntimeError(
                f"Unexpected {metric} summary: {actual}; expected {expected}"
            )

    return pd.DataFrame(rows)


def asymmetric_error(
    point: float,
    low: float,
    high: float,
) -> np.ndarray:
    return np.array([[point - low], [high - point]], dtype=float)


def create_figure(
    summary: pd.DataFrame,
    near: pd.DataFrame,
    bootstrap: pd.Series,
    output_dir: Path,
) -> None:
    metric_labels = {
        "runtime": "Runtime",
        "energy": "Energy",
        "edp": "EDP",
    }
    metric_order = ["runtime", "energy", "edp"]
    y_positions = np.arange(len(metric_order))

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.7),
        gridspec_kw={"width_ratios": [0.9, 1.15]},
    )

    # Panel (a): summary of maximum-thread regret.
    for column, label, marker in [
        ("median", "Median", "o"),
        ("p95", "P95", "s"),
        ("maximum", "Maximum", "^"),
    ]:
        values = [
            float(
                summary.loc[summary["metric"] == metric, column].iloc[0]
            )
            for metric in metric_order
        ]
        ax_left.scatter(
            values,
            y_positions,
            marker=marker,
            s=58,
            label=label,
        )

    ax_left.set_xscale("log")
    ax_left.set_yticks(
        y_positions,
        [metric_labels[metric] for metric in metric_order],
    )
    ax_left.invert_yaxis()
    ax_left.set_xlabel(
        "Regret of maximum-thread policy (%) — logarithmic scale"
    )
    ax_left.set_title("(a) Maximum-thread policy")
    ax_left.grid(axis="x", alpha=0.25)
    ax_left.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
    )

    # Panel (b): exactly 11 energy-saving alternatives.
    highlight_mask = (
        (near["platform"] == "INTEL")
        & (near["workload"] == "REDUCTION")
        & (near["size"] == 256_000_000)
    )

    intel = near[(near["platform"] == "INTEL") & ~highlight_mask]
    amd = near[near["platform"] == "AMD"]
    highlight = near[highlight_mask].iloc[0]

    ax_right.scatter(
        intel["runtime_penalty_pct"],
        intel["energy_saving_pct"],
        marker="o",
        s=55,
        label="Intel alternatives",
    )
    ax_right.scatter(
        amd["runtime_penalty_pct"],
        amd["energy_saving_pct"],
        marker="s",
        s=62,
        label="AMD alternatives",
    )

    point_x = float(bootstrap["runtime_penalty_point_pct"])
    point_y = float(bootstrap["energy_saving_point_pct"])
    xerr = asymmetric_error(
        point_x,
        float(bootstrap["runtime_penalty_ci95_low"]),
        float(bootstrap["runtime_penalty_ci95_high"]),
    )
    yerr = asymmetric_error(
        point_y,
        float(bootstrap["energy_saving_ci95_low"]),
        float(bootstrap["energy_saving_ci95_high"]),
    )

    # Verify that the highlighted CSV point matches the bootstrap point.
    if not np.isclose(
        float(highlight["runtime_penalty_pct"]),
        point_x,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError("Highlighted runtime point does not match bootstrap")
    if not np.isclose(
        float(highlight["energy_saving_pct"]),
        point_y,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError("Highlighted energy point does not match bootstrap")

    ax_right.errorbar(
        point_x,
        point_y,
        xerr=xerr,
        yerr=yerr,
        fmt="*",
        markersize=16,
        capsize=4,
        linewidth=1.2,
        label="Intel REDUCTION 256M",
    )

    ax_right.annotate(
        "Intel REDUCTION 256M\n4T instead of 8T",
        xy=(point_x, point_y),
        xytext=(0.43, 34.0),
        arrowprops={"arrowstyle": "-"},
        fontsize=9,
    )
    ax_right.text(
        0.02,
        0.97,
        "11 measured alternatives across 5 workloads",
        transform=ax_right.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    ax_right.set_xlim(-0.03, 1.02)
    ax_right.set_ylim(0.0, 41.5)
    ax_right.set_xlabel(
        "Runtime penalty vs. runtime-optimal thread count (%)"
    )
    ax_right.set_ylabel("Energy saving (%)")
    ax_right.set_title("(b) Energy-saving alternatives within 1% runtime")
    ax_right.grid(alpha=0.25)
    ax_right.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.90,
        bottom=0.26,
        wspace=0.28,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "chapter5_cpu_configuration_summary.pdf"
    png = output_dir / "chapter5_cpu_configuration_summary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("PASS: Chapter 5 figure reproduced")
    print("Near-free alternatives: 11")
    print("Intel alternatives: 10")
    print("AMD alternatives: 1")
    print("Workloads represented: 5")
    print(f"PDF: {pdf}")
    print(f"PNG: {png}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Reproduce the two-panel Chapter 5 figure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the three required analysis CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "chapter5",
        help="Output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = find_input_dir(repo_root, args.input_dir)
    output_dir = args.output_dir.expanduser().resolve()

    print(f"Using input directory: {input_dir}")
    regrets, near, bootstrap = load_data(input_dir)
    summary = summarize_regrets(regrets)

    print(summary.to_string(index=False))
    print()
    print(
        near[
            [
                "workload",
                "size",
                "platform",
                "chosen_cfg",
                "runtime_opt_cfg",
                "runtime_penalty_pct",
                "energy_saving_pct",
            ]
        ].to_string(index=False)
    )
    print()

    create_figure(summary, near, bootstrap, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
