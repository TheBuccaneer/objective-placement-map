#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd


WORKLOADS = [
    "GEMM",
    "STRIDED_GEMM",
    "AXPY",
    "STREAM",
    "REDUCTION",
    "CONV2D",
]

PLATFORM_STYLE = {
    "INTEL":   ("tab:blue",   "Intel"),
    "AMD":     ("tab:orange", "AMD"),
    "3090":    ("tab:green",  "3090"),
    "5060ti":  ("tab:red",    "5060 Ti"),
}

LARGE_SIZES = {
    16_000_000,
    32_000_000,
    64_000_000,
    128_000_000,
    256_000_000,
}


def latest_successful_output(repo: Path) -> Path:
    candidates = []

    for path in repo.glob("results/runs/*/generated/outputs"):
        if "FAILED" in str(path):
            continue
        if (
            (path / "canonical_51_cells.csv").exists()
            and (path / "near_joint_optimum_sensitivity.csv").exists()
        ):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Keine erfolgreiche Analyse mit canonical_51_cells.csv "
            "und near_joint_optimum_sensitivity.csv gefunden."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def compact_size(workload: str, value: int) -> str:
    if workload == "CONV2D":
        return f"S{value}"

    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"

    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}K"

    return str(value)


def normalize_platform(value: str) -> str:
    value = str(value).strip()

    aliases = {
        "Intel": "INTEL",
        "intel": "INTEL",
        "AMD": "AMD",
        "amd": "AMD",
        "RTX 3090": "3090",
        "RTX3090": "3090",
        "3090": "3090",
        "RTX 5060 Ti": "5060ti",
        "RTX5060Ti": "5060ti",
        "5060TI": "5060ti",
        "5060ti": "5060ti",
    }

    normalized = aliases.get(value, value)

    if normalized not in PLATFORM_STYLE:
        raise ValueError(f"Unbekannte Plattformbezeichnung: {value!r}")

    return normalized


def draw_tile(ax, x, y, platform, text):
    color, _ = PLATFORM_STYLE[platform]

    ax.add_patch(
        Rectangle(
            (x, y),
            1,
            1,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
        )
    )

    ax.text(
        x + 0.5,
        y + 0.5,
        text,
        ha="center",
        va="center",
        fontsize=6.5,
        color="white",
        fontweight="bold",
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    source_dir = latest_successful_output(repo)

    cells_path = source_dir / "canonical_51_cells.csv"
    near_path = source_dir / "near_joint_optimum_sensitivity.csv"

    cells = pd.read_csv(cells_path)
    near = pd.read_csv(near_path)

    required_cells = {
        "workload",
        "size",
        "runtime_winner",
        "energy_winner",
        "conflict",
    }
    missing = required_cells.difference(cells.columns)
    if missing:
        raise ValueError(
            "canonical_51_cells.csv fehlen Spalten: "
            + ", ".join(sorted(missing))
        )

    required_near = {"workload", "size", "threshold", "exists"}
    missing = required_near.difference(near.columns)
    if missing:
        raise ValueError(
            "near_joint_optimum_sensitivity.csv fehlen Spalten: "
            + ", ".join(sorted(missing))
        )

    cells["size"] = cells["size"].astype(int)
    cells["runtime_winner"] = cells["runtime_winner"].map(
        normalize_platform
    )
    cells["energy_winner"] = cells["energy_winner"].map(
        normalize_platform
    )

    # Ensure boolean interpretation even when CSV stores strings.
    if cells["conflict"].dtype != bool:
        cells["conflict"] = (
            cells["conflict"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

    if cells["conflict"].isna().any():
        raise ValueError("Konfliktspalte enthält nicht interpretierbare Werte.")

    if len(cells) != 51:
        raise ValueError(f"Erwartet: 51 Zellen; gefunden: {len(cells)}.")

    if int(cells["conflict"].sum()) != 24:
        raise ValueError(
            "Erwartet: 24 Konflikte; gefunden: "
            f"{int(cells['conflict'].sum())}."
        )

    conflict_keys = cells.loc[
        cells["conflict"], ["workload", "size"]
    ].copy()

    near_conflicts = near.merge(
        conflict_keys,
        on=["workload", "size"],
        how="inner",
    )

    unresolved = {0: 24}

    for threshold in (10, 20):
        subset = near_conflicts[
            near_conflicts["threshold"] == threshold
        ]

        if len(subset) != 24:
            raise ValueError(
                f"Für {threshold}% wurden {len(subset)} statt "
                "24 Konfliktzellen gefunden."
            )

        exists = (
            subset["exists"]
            if subset["exists"].dtype == bool
            else subset["exists"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

        unresolved[threshold] = int((~exists).sum())

    expected = {0: 24, 10: 22, 20: 19}
    if unresolved != expected:
        raise ValueError(
            f"Near-optimality-Werte weichen ab: {unresolved}; "
            f"erwartet: {expected}."
        )

    fig = plt.figure(figsize=(11.6, 7.2))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[4.8, 1.25],
        wspace=0.18,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    # Two rows per workload: runtime and energy.
    block_height = 2.6
    max_columns = 9

    workload_conflicts = (
        cells.groupby("workload")["conflict"]
        .agg(["sum", "count"])
        .reindex(WORKLOADS)
    )

    for workload_index, workload in enumerate(WORKLOADS):
        group = (
            cells[cells["workload"] == workload]
            .sort_values("size")
            .reset_index(drop=True)
        )

        base_y = (len(WORKLOADS) - 1 - workload_index) * block_height
        runtime_y = base_y + 1.05
        energy_y = base_y + 0.05

        ax.text(
            -0.25,
            base_y + 1.55,
            workload.replace("_", "\n"),
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

        ax.text(
            -0.25,
            runtime_y + 0.5,
            "Runtime",
            ha="right",
            va="center",
            fontsize=7,
        )
        ax.text(
            -0.25,
            energy_y + 0.5,
            "Energy",
            ha="right",
            va="center",
            fontsize=7,
        )

        for column, row in group.iterrows():
            runtime = row["runtime_winner"]
            energy = row["energy_winner"]

            draw_tile(
                ax,
                column,
                runtime_y,
                runtime,
                PLATFORM_STYLE[runtime][1],
            )
            draw_tile(
                ax,
                column,
                energy_y,
                energy,
                PLATFORM_STYLE[energy][1],
            )

            ax.text(
                column + 0.5,
                base_y - 0.08,
                compact_size(workload, int(row["size"])),
                ha="center",
                va="top",
                fontsize=6.2,
            )

            if bool(row["conflict"]):
                ax.add_patch(
                    Rectangle(
                        (column - 0.035, energy_y - 0.035),
                        1.07,
                        2.07,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )

            if (
                workload in {"AXPY", "STREAM", "REDUCTION"}
                and int(row["size"]) in LARGE_SIZES
            ):
                ax.add_patch(
                    Rectangle(
                        (column + 0.08, energy_y + 0.08),
                        0.84,
                        1.84,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.2,
                        linestyle="--",
                    )
                )

        # Empty Conv2D slots remain blank.
        for column in range(len(group), max_columns):
            ax.add_patch(
                Rectangle(
                    (column, runtime_y),
                    1,
                    1,
                    facecolor="none",
                    edgecolor="0.88",
                    linewidth=0.5,
                )
            )
            ax.add_patch(
                Rectangle(
                    (column, energy_y),
                    1,
                    1,
                    facecolor="none",
                    edgecolor="0.88",
                    linewidth=0.5,
                )
            )

        conflicts = int(workload_conflicts.loc[workload, "sum"])
        total = int(workload_conflicts.loc[workload, "count"])

        ax.text(
            max_columns + 0.18,
            base_y + 1.05,
            f"{conflicts}/{total}",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlim(-2.35, max_columns + 1.0)
    ax.set_ylim(-0.35, len(WORKLOADS) * block_height)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        max_columns + 0.18,
        len(WORKLOADS) * block_height - 0.45,
        "Conflicts",
        ha="left",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    legend_handles = [
        Patch(
            facecolor=PLATFORM_STYLE[key][0],
            label=PLATFORM_STYLE[key][1],
        )
        for key in ["INTEL", "AMD", "3090", "5060ti"]
    ]

    legend_handles.extend(
        [
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
                label="Runtime-energy conflict",
            ),
            Patch(
                facecolor="white",
                edgecolor="0.3",
                linestyle="--",
                label="15 cells from Sec. 3",
            ),
        ]
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.48, 1.025),
        ncol=3,
        fontsize=7,
        frameon=False,
    )

    thresholds = [0, 10, 20]
    values = [unresolved[x] for x in thresholds]

    bars = ax_bar.bar(
        range(len(thresholds)),
        values,
        width=0.62,
    )

    ax_bar.set_xticks(
        range(len(thresholds)),
        [f"{x}%" for x in thresholds],
    )
    ax_bar.set_ylim(0, 26)
    ax_bar.set_ylabel("Unresolved conflicts")
    ax_bar.set_xlabel("Near-optimality tolerance")
    ax_bar.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.45,
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax_bar.text(
        0.5,
        0.97,
        "24 initial conflicts",
        transform=ax_bar.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    fig.tight_layout()

    output_dir = repo / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = (
        output_dir
        / "placement_map_and_conflict_hardness.pdf"
    )
    png_path = (
        output_dir
        / "placement_map_and_conflict_hardness.png"
    )

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Input cells: {cells_path}")
    print(f"Input near:  {near_path}")
    print(f"Conflicts:   {int(cells['conflict'].sum())}/51")
    print(f"Unresolved:  {unresolved}")
    print(f"PDF:         {pdf_path}")
    print(f"PNG:         {png_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        raise SystemExit(1)
