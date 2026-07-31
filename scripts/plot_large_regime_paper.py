#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "workload",
    "speedup_3090_vs_5060",
    "saving_5060_vs_3090",
}


def find_latest_input(repo: Path) -> Path:
    candidates = [
        path
        for path in repo.glob(
            "results/runs/*/generated/**/large_15_regime.csv"
        )
        if "FAILED" not in str(path)
    ]

    if not candidates:
        raise FileNotFoundError(
            "Keine large_15_regime.csv unter results/runs gefunden."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    source = find_latest_input(repo)

    data = pd.read_csv(source)

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "In der Eingabedatei fehlen Spalten: "
            + ", ".join(sorted(missing))
        )

    expected_workloads = ["AXPY", "STREAM", "REDUCTION"]
    data = data[data["workload"].isin(expected_workloads)].copy()

    if len(data) != 15:
        raise ValueError(
            f"Erwartet wurden 15 Zellen, gefunden wurden {len(data)}."
        )

    output_dir = repo / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))

    for workload in expected_workloads:
        group = data[data["workload"] == workload]

        ax.scatter(
            group["speedup_3090_vs_5060"],
            100.0 * group["saving_5060_vs_3090"],
            label=workload,
            s=38,
        )

    median_speedup = data["speedup_3090_vs_5060"].median()
    median_saving = 100.0 * data["saving_5060_vs_3090"].median()

    annotation = (
        f"15/15 runtime-energy conflicts\n"
        f"Median speedup: {median_speedup:.2f}×\n"
        f"Median energy saving: {median_saving:.2f}%"
    )

    ax.text(
        0.02,
        0.03,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.9,
        },
    )

    ax.set_xlabel("RTX 3090 speedup over RTX 5060 Ti")
    ax.set_ylabel("RTX 5060 Ti board-energy saving (%)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    pdf_path = output_dir / "large_regime_speedup_energy.pdf"
    png_path = output_dir / "large_regime_speedup_energy.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Input: {source}")
    print(f"PDF:   {pdf_path}")
    print(f"PNG:   {png_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        raise SystemExit(1)
