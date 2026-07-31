#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


CPU_PLATFORMS = ["INTEL", "AMD"]
METRICS = ["runtime", "energy", "edp"]

METRIC_LABELS = {
    "runtime": "Runtime",
    "energy": "Energy",
    "edp": "EDP",
}

TARGET_RUNTIME_CI = (-0.014, 0.443)
TARGET_ENERGY_CI = (26.31, 35.59)


def find_input_directory(repo: Path) -> Path:
    candidates = []

    for directory in repo.glob("results/runs/*/generated/outputs"):
        if "FAILED" in str(directory):
            continue

        required = [
            directory / "all_918_configurations.csv",
            directory / "max_thread_policy_regret.csv",
        ]

        if all(path.exists() for path in required):
            candidates.append(directory)

    if candidates:
        return max(
            candidates,
            key=lambda path: path.stat().st_mtime,
        )

    frozen = (
        repo
        / "data"
        / "snapshots"
        / "t1-analysis-20260730"
        / "extracted"
        / "deep_research_energy"
        / "outputs"
    )

    required = [
        frozen / "all_918_configurations.csv",
        frozen / "max_thread_policy_regret.csv",
    ]

    if all(path.exists() for path in required):
        return frozen

    raise FileNotFoundError(
        "Keine Analyseausgabe mit all_918_configurations.csv und "
        "max_thread_policy_regret.csv gefunden."
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    filename: str,
) -> None:
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{filename}: fehlende Spalten: "
            + ", ".join(sorted(missing))
        )


def build_max_thread_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frame,
        {
            "workload",
            "size",
            "platform",
            "metric",
            "max_threads",
            "regret_pct",
        },
        "max_thread_policy_regret.csv",
    )

    data = frame.copy()
    data["platform"] = data["platform"].astype(str).str.upper()
    data["metric"] = data["metric"].astype(str).str.lower()

    cpu = data[
        data["platform"].isin(CPU_PLATFORMS)
        & data["metric"].isin(METRICS)
    ].copy()

    if len(cpu) != 306:
        raise ValueError(
            f"Erwartet wurden 306 Max-Thread-Zeilen, "
            f"gefunden wurden {len(cpu)}."
        )

    rows = []

    for metric in METRICS:
        values = cpu.loc[
            cpu["metric"] == metric,
            "regret_pct",
        ].astype(float)

        if len(values) != 102:
            raise ValueError(
                f"{metric}: erwartet 102 Werte, gefunden {len(values)}."
            )

        rows.append(
            {
                "metric": metric,
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "maximum": float(values.max()),
            }
        )

    summary = pd.DataFrame(rows)

    expected_medians = {
        "runtime": 7.47,
        "energy": 38.52,
        "edp": 63.06,
    }

    for metric, expected in expected_medians.items():
        actual = float(
            summary.loc[
                summary["metric"] == metric,
                "median",
            ].iloc[0]
        )

        if not np.isclose(actual, expected, atol=0.06):
            raise ValueError(
                f"{metric}-Median: {actual:.4f}% statt "
                f"ungefähr {expected:.2f}%."
            )

    energy_max = float(
        summary.loc[
            summary["metric"] == "energy",
            "maximum",
        ].iloc[0]
    )
    edp_max = float(
        summary.loc[
            summary["metric"] == "edp",
            "maximum",
        ].iloc[0]
    )

    if not np.isclose(energy_max, 458.20, atol=0.15):
        raise ValueError(
            f"Unerwartetes Energie-Maximum: {energy_max:.4f}%."
        )

    if not np.isclose(edp_max, 2108.73, atol=0.25):
        raise ValueError(
            f"Unerwartetes EDP-Maximum: {edp_max:.4f}%."
        )

    return summary


def build_near_free_alternatives(
    configurations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    require_columns(
        configurations,
        {
            "workload",
            "size",
            "platform",
            "threads",
            "runtime",
            "energy",
        },
        "all_918_configurations.csv",
    )

    data = configurations.copy()
    data["platform"] = data["platform"].astype(str).str.upper()
    data["size"] = data["size"].astype(int)
    data["threads"] = data["threads"].astype(int)

    cpu = data[
        data["platform"].isin(CPU_PLATFORMS)
    ].copy()

    cell_count = cpu.groupby(
        ["workload", "size", "platform"]
    ).ngroups

    if cell_count != 102:
        raise ValueError(
            f"Erwartet wurden 102 CPU-Plattform-Zellen, "
            f"gefunden wurden {cell_count}."
        )

    alternatives = []

    for (_, _, _), group in cpu.groupby(
        ["workload", "size", "platform"],
        sort=True,
    ):
        group = group.copy().sort_values("threads")

        runtime_optimum = group.loc[
            group["runtime"].idxmin()
        ]

        runtime_min = float(runtime_optimum["runtime"])
        reference_energy = float(runtime_optimum["energy"])
        reference_threads = int(runtime_optimum["threads"])

        group["runtime_penalty_pct"] = (
            group["runtime"] / runtime_min - 1.0
        ) * 100.0

        group["energy_saving_pct"] = (
            1.0 - group["energy"] / reference_energy
        ) * 100.0

        group["runtime_opt_threads"] = reference_threads

        eligible = group[
            (group["threads"] != reference_threads)
            & (group["runtime_penalty_pct"] <= 1.0 + 1e-9)
            & (group["energy_saving_pct"] > 0.0)
        ].copy()

        alternatives.append(eligible)

    near_free = pd.concat(
        alternatives,
        ignore_index=True,
    )

    if len(near_free) != 12:
        raise ValueError(
            f"Erwartet wurden 12 Near-free-Alternativen, "
            f"gefunden wurden {len(near_free)}."
        )

    unique_coordinates = near_free[
        ["runtime_penalty_pct", "energy_saving_pct"]
    ].drop_duplicates()

    if len(unique_coordinates) != 12:
        raise ValueError(
            "Die zwölf Near-free-Alternativen besitzen nicht "
            "zwölf eindeutige Koordinaten."
        )

    target = near_free[
        (near_free["platform"] == "INTEL")
        & (near_free["workload"] == "REDUCTION")
        & (near_free["size"] == 256_000_000)
        & (near_free["threads"] == 4)
        & (near_free["runtime_opt_threads"] == 8)
    ]

    if len(target) != 1:
        raise ValueError(
            "Intel REDUCTION 256M, 4T statt 8T, wurde nicht "
            "eindeutig gefunden."
        )

    target_row = target.iloc[0]

    target_runtime = float(
        target_row["runtime_penalty_pct"]
    )
    target_energy = float(
        target_row["energy_saving_pct"]
    )

    if not np.isclose(target_runtime, 0.2442, atol=0.01):
        raise ValueError(
            f"Unerwarteter Laufzeitaufschlag: "
            f"{target_runtime:.4f}%."
        )

    if not np.isclose(target_energy, 28.29, atol=0.08):
        raise ValueError(
            f"Unerwartete Energieersparnis: "
            f"{target_energy:.4f}%."
        )

    return near_free, target_row


def format_percent(value: float) -> str:
    if value >= 1000:
        return f"{value:.0f}%"

    if value >= 100:
        return f"{value:.1f}%"

    if value >= 10:
        return f"{value:.1f}%"

    return f"{value:.2f}%"


def plot_max_thread_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    """Dot plot of max-thread regret.

    Numeric values are reported in the running text and in the
    corresponding table, so no in-figure labels are drawn. This keeps
    the panel readable across three orders of magnitude.
    """
    y_positions = np.arange(len(METRICS))
    statistics = [
        ("median", "Median", "o", "black"),
        ("p95", "P95", "s", "white"),
        ("maximum", "Maximum", "^", "black"),
    ]

    all_values = []
    for column, label, marker, face in statistics:
        values = [
            float(
                summary.loc[
                    summary["metric"] == metric,
                    column,
                ].iloc[0]
            )
            for metric in METRICS
        ]
        all_values.extend(values)
        ax.scatter(
            values,
            y_positions,
            marker=marker,
            s=62,
            facecolors=face,
            edgecolors="black",
            linewidths=1.1,
            label=label,
            zorder=3,
        )

    positive = np.array(
        [value for value in all_values if value > 0],
        dtype=float,
    )

    ax.set_xscale("log")
    ax.set_xlim(
        max(float(positive.min()) * 0.55, 0.1),
        float(positive.max()) * 1.60,
    )
    ax.set_yticks(
        y_positions,
        [METRIC_LABELS[metric] for metric in METRICS],
    )
    ax.set_ylim(
        len(METRICS) - 0.55,
        -0.55,
    )

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(
        "Regret of maximum-thread policy (%) — logarithmic scale"
    )
    ax.set_title(
        "(a) Maximum-thread policy",
        loc="left",
    )
    ax.grid(
        axis="x",
        which="major",
        alpha=0.24,
    )
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handletextpad=0.4,
    )

def plot_near_free_panel(
    ax: plt.Axes,
    near_free: pd.DataFrame,
    target: pd.Series,
) -> None:
    marker_by_platform = {
        "INTEL": "o",
        "AMD": "s",
    }

    label_by_platform = {
        "INTEL": "Intel alternatives",
        "AMD": "AMD alternatives",
    }

    # Filled, clearly visible markers for all twelve alternatives.
    for platform in CPU_PLATFORMS:
        group = near_free[
            near_free["platform"] == platform
        ]

        ax.scatter(
            group["runtime_penalty_pct"],
            group["energy_saving_pct"],
            marker=marker_by_platform[platform],
            s=62,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.7,
            label=label_by_platform[platform],
            zorder=2,
        )

    target_x = float(target["runtime_penalty_pct"])
    target_y = float(target["energy_saving_pct"])

    x_error = np.array(
        [
            [target_x - TARGET_RUNTIME_CI[0]],
            [TARGET_RUNTIME_CI[1] - target_x],
        ]
    )

    y_error = np.array(
        [
            [target_y - TARGET_ENERGY_CI[0]],
            [TARGET_ENERGY_CI[1] - target_y],
        ]
    )

    ax.errorbar(
        target_x,
        target_y,
        xerr=x_error,
        yerr=y_error,
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1.2,
        zorder=4,
    )

    ax.scatter(
        [target_x],
        [target_y],
        marker="*",
        s=190,
        facecolors="white",
        edgecolors="black",
        linewidths=1.2,
        label="Intel REDUCTION 256M",
        zorder=5,
    )

    ax.annotate(
        "Intel REDUCTION 256M\n4T instead of 8T",
        xy=(target_x, target_y),
        xytext=(0.43, 35.2),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.8,
        },
        fontsize=8,
        ha="left",
        va="center",
    )

    ax.text(
        0.02,
        0.97,
        "12 measured alternatives across 5 workloads",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    maximum_saving = max(
        float(near_free["energy_saving_pct"].max()),
        TARGET_ENERGY_CI[1],
    )

    ax.set_xlim(
        min(-0.05, TARGET_RUNTIME_CI[0] - 0.02),
        1.02,
    )
    ax.set_ylim(
        0,
        maximum_saving * 1.18,
    )

    ax.set_xlabel(
        "Runtime penalty vs. runtime-optimal thread count (%)"
    )
    ax.set_ylabel("Energy saving (%)")

    ax.set_title(
        "(b) Energy-saving alternatives within 1% runtime",
        loc="left",
    )

    ax.grid(
        which="major",
        alpha=0.20,
    )
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    input_directory = find_input_directory(repo)

    max_thread_path = (
        input_directory / "max_thread_policy_regret.csv"
    )
    configurations_path = (
        input_directory / "all_918_configurations.csv"
    )

    max_thread = pd.read_csv(max_thread_path)
    configurations = pd.read_csv(configurations_path)

    summary = build_max_thread_summary(max_thread)
    near_free, target = build_near_free_alternatives(
        configurations
    )

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 9

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.15),
        gridspec_kw={
            "width_ratios": [1.05, 1.35],
            "wspace": 0.27,
        },
    )

    plot_max_thread_panel(
        ax_left,
        summary,
    )
    plot_near_free_panel(
        ax_right,
        near_free,
        target,
    )

    # Reserve space for both legends below the panels.
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.89,
        bottom=0.27,
        wspace=0.28,
    )

    output_directory = repo / "paper" / "figures"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        output_directory
        / "chapter5_cpu_configuration_summary.pdf"
    )
    png_path = (
        output_directory
        / "chapter5_cpu_configuration_summary.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("PASS")
    print(f"Input directory: {input_directory}")
    print()
    print("Maximum-thread policy summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(
        "Near-free alternatives: "
        f"{len(near_free)} rows, "
        f"{len(near_free[['runtime_penalty_pct', 'energy_saving_pct']].drop_duplicates())} "
        "unique coordinates."
    )
    print()
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"FEHLER: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
