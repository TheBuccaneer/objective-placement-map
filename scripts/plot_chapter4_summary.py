#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Audited values from the frozen analysis.
WORKLOADS = [
    "GEMM",
    "STRIDED_GEMM",
    "AXPY",
    "STREAM",
    "REDUCTION",
    "Conv2D",
]

CONFLICTS = np.array([2, 2, 5, 6, 6, 3])
TOTALS = np.array([9, 9, 9, 9, 9, 6])

TOLERANCE_LABELS = [
    "Exact",
    r"$\delta=10\%$",
    r"$\delta=20\%$",
]
UNRESOLVED = np.array([24, 22, 19])


def validate_values() -> None:
    if int(CONFLICTS.sum()) != 24:
        raise ValueError(
            f"Expected 24 conflicts, found {int(CONFLICTS.sum())}."
        )

    if int(TOTALS.sum()) != 51:
        raise ValueError(
            f"Expected 51 cells, found {int(TOTALS.sum())}."
        )

    if UNRESOLVED.tolist() != [24, 22, 19]:
        raise ValueError(
            f"Unexpected near-optimality values: {UNRESOLVED.tolist()}."
        )

    if np.any(CONFLICTS > TOTALS):
        raise ValueError("A conflict count exceeds its family total.")


def main() -> None:
    validate_values()

    repo = Path(__file__).resolve().parents[1]
    output_dir = repo / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid Type-3 fonts in the vector PDF.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 9

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.9),
        gridspec_kw={
            "width_ratios": [1.65, 1.0],
            "wspace": 0.20,
        },
    )

    # --------------------------------------------------------------
    # Panel (a): conflict distribution by workload family
    # --------------------------------------------------------------
    y_positions = np.arange(len(WORKLOADS))
    same_winner = TOTALS - CONFLICTS

    ax_left.barh(
        y_positions,
        CONFLICTS,
        label="Different winners",
    )

    ax_left.barh(
        y_positions,
        same_winner,
        left=CONFLICTS,
        label="Same winner",
        alpha=0.28,
    )

    ax_left.set_yticks(y_positions, WORKLOADS)
    ax_left.invert_yaxis()

    ax_left.set_xlim(0, 9.7)
    ax_left.set_xticks([0, 2, 4, 6, 8])

    ax_left.set_xlabel("Workload-instance cells")
    ax_left.set_title(
        "(a) Conflicts by workload family",
        loc="left",
    )

    for row, (conflict_count, total_count) in enumerate(
        zip(CONFLICTS, TOTALS)
    ):
        ax_left.text(
            total_count + 0.14,
            row,
            f"{conflict_count}/{total_count}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax_left.grid(
        axis="x",
        alpha=0.20,
    )
    ax_left.set_axisbelow(True)

    # Place the legend below panel (a), so it does not cover Conv2D.
    ax_left.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        fontsize=8,
        columnspacing=1.4,
        handlelength=2.0,
    )

    # --------------------------------------------------------------
    # Panel (b): conflict hardness
    # --------------------------------------------------------------
    x_positions = np.arange(len(TOLERANCE_LABELS))

    ax_right.plot(
        x_positions,
        UNRESOLVED,
        marker="o",
        linewidth=1.8,
        markersize=7,
    )

    ax_right.set_xticks(
        x_positions,
        TOLERANCE_LABELS,
    )

    # Focus on the observed range. All values are printed explicitly.
    ax_right.set_ylim(17, 25.5)
    ax_right.set_yticks([18, 20, 22, 24])

    ax_right.set_ylabel("Unresolved conflicts")
    ax_right.set_title(
        "(b) Conflict hardness",
        loc="left",
    )

    ax_right.grid(
        axis="y",
        alpha=0.20,
    )
    ax_right.set_axisbelow(True)

    for x_position, value in zip(x_positions, UNRESOLVED):
        ax_right.text(
            x_position,
            value + 0.28,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax_right.text(
        0.5,
        0.045,
        "19/24 remain unresolved\nat 20% tolerance",
        transform=ax_right.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    # Reserve space for the legend below panel (a).
    fig.tight_layout(rect=[0, 0.10, 1, 1])

    pdf_path = output_dir / "chapter4_conflict_summary.pdf"
    png_path = output_dir / "chapter4_conflict_summary.png"

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
    print(f"Total cells:       {int(TOTALS.sum())}")
    print(f"Total conflicts:   {int(CONFLICTS.sum())}")
    print(f"Unresolved values: {UNRESOLVED.tolist()}")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
