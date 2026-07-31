#!/usr/bin/env python3
"""Reproduce the large-regime runtime–energy figure from pinned sessions.

The script:
1. loads the five GPU session medians for the 15 large AXPY, STREAM,
   and REDUCTION cells;
2. computes the canonical point estimates;
3. independently resamples the five sessions of each GPU 5,000 times;
4. writes cell-specific percentile bootstrap intervals;
5. verifies the frozen headline values and neutral-boundary claims; and
6. generates PDF and PNG figures with error bars, neutral boundaries,
   and a zoomed inset.

Run from the repository root:

    .venv/bin/python scripts/reproduce_large_regime_figure.py

Default inputs:
    data/snapshots/t1-analysis-20260730/extracted/
    deep_research_energy/inputs

Default outputs:
    results/large-regime/large_15_cell_bootstrap_intervals.csv
    results/large-regime/large_regime_speedup_energy.pdf
    results/large-regime/large_regime_speedup_energy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


SEED = 20260730
BOOTSTRAPS = 5_000

WORKLOADS = ("AXPY", "STREAM", "REDUCTION")
PLATFORMS = ("3090", "5060ti")
LARGE_SIZES = (
    16_000_000,
    32_000_000,
    64_000_000,
    128_000_000,
    256_000_000,
)

EXPECTED_MEDIAN_SPEEDUP = 2.1217706
EXPECTED_MEDIAN_SAVING_PCT = 42.4566


def normalize_platform(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if lowered == "3090":
        return "3090"
    if lowered in {"5060ti", "5060 ti", "rtx 5060 ti"}:
        return "5060ti"
    return text.upper()


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    filename: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename}: missing columns: {missing}")


def load_sessions(inputs: Path) -> pd.DataFrame:
    """Load normalized GPU session medians for the three workload families."""

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
    parts = [
        pd.DataFrame(
            {
                "workload": "AXPY",
                "size": axpy["problem_size"].astype(int),
                "platform": axpy["platform"].map(normalize_platform),
                "session": axpy["session_number"].astype(int),
                "runtime": axpy["median_time_e2e_op_s"].astype(float),
                "energy": axpy["median_device_energy_op_j"].astype(float),
            }
        )
    ]

    for filename, workload in (
        ("stream_all_sessions.csv", "STREAM"),
        ("reduction_all_sessions.csv", "REDUCTION"),
    ):
        path = inputs / filename
        frame = pd.read_csv(path)
        require_columns(
            frame,
            {
                "platform",
                "session_number",
                "problem_size",
                "runtime_s",
                "energy_j",
            },
            path.name,
        )
        parts.append(
            pd.DataFrame(
                {
                    "workload": workload,
                    "size": frame["problem_size"].astype(int),
                    "platform": frame["platform"].map(normalize_platform),
                    "session": frame["session_number"].astype(int),
                    "runtime": frame["runtime_s"].astype(float),
                    "energy": frame["energy_j"].astype(float),
                }
            )
        )

    sessions = pd.concat(parts, ignore_index=True)
    sessions = sessions[
        sessions["workload"].isin(WORKLOADS)
        & sessions["size"].isin(LARGE_SIZES)
        & sessions["platform"].isin(PLATFORMS)
    ].copy()

    sessions = sessions.sort_values(
        ["workload", "size", "platform", "session"]
    ).reset_index(drop=True)

    key = ["workload", "size", "platform", "session"]
    if sessions.duplicated(key).any():
        duplicates = sessions.loc[
            sessions.duplicated(key, keep=False),
            key,
        ]
        raise ValueError(
            "Duplicate session rows:\n" + duplicates.to_string(index=False)
        )

    expected_rows = len(WORKLOADS) * len(LARGE_SIZES) * len(PLATFORMS) * 5
    if len(sessions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} GPU session rows, got {len(sessions)}"
        )

    counts = sessions.groupby(["workload", "size", "platform"])["session"].nunique()
    if not counts.eq(5).all():
        raise ValueError("Every cell/platform combination must contain five sessions")

    if not np.isfinite(sessions[["runtime", "energy"]].to_numpy()).all():
        raise ValueError("Non-finite runtime or energy values found")
    if (sessions[["runtime", "energy"]] <= 0).any().any():
        raise ValueError("Runtime and energy values must be positive")

    return sessions


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def compute_cell_intervals(
    sessions: pd.DataFrame,
    *,
    seed: int,
    bootstraps: int,
) -> pd.DataFrame:
    """Compute independent per-platform session bootstraps per cell."""

    if bootstraps < 100:
        raise ValueError("--bootstraps must be at least 100")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for workload in WORKLOADS:
        for size in LARGE_SIZES:
            cell = sessions[
                (sessions["workload"] == workload)
                & (sessions["size"] == size)
            ]

            values: dict[str, np.ndarray] = {}
            for platform in PLATFORMS:
                group = cell[cell["platform"] == platform].sort_values("session")
                if len(group) != 5:
                    raise ValueError(
                        f"Expected five sessions for {(workload, size, platform)}"
                    )
                values[platform] = group[["runtime", "energy"]].to_numpy(float)

            runtime_3090 = float(np.median(values["3090"][:, 0]))
            runtime_5060 = float(np.median(values["5060ti"][:, 0]))
            energy_3090 = float(np.median(values["3090"][:, 1]))
            energy_5060 = float(np.median(values["5060ti"][:, 1]))

            speedup_point = runtime_5060 / runtime_3090
            saving_point = (1.0 - energy_5060 / energy_3090) * 100.0

            speedup_boot = np.empty(bootstraps, dtype=float)
            saving_boot = np.empty(bootstraps, dtype=float)

            for replicate in range(bootstraps):
                medians: dict[str, np.ndarray] = {}
                for platform in PLATFORMS:
                    # Independent session resampling for each GPU, matching
                    # the GPU-platform bootstrap contract in the analysis.
                    indices = rng.integers(0, 5, size=5)
                    medians[platform] = np.median(
                        values[platform][indices],
                        axis=0,
                    )

                speedup_boot[replicate] = (
                    medians["5060ti"][0] / medians["3090"][0]
                )
                saving_boot[replicate] = (
                    1.0
                    - medians["5060ti"][1] / medians["3090"][1]
                ) * 100.0

            speedup_low, speedup_high = percentile_interval(speedup_boot)
            saving_low, saving_high = percentile_interval(saving_boot)

            rows.append(
                {
                    "workload": workload,
                    "size": size,
                    "sessions_per_platform": 5,
                    "bootstrap_replicates": bootstraps,
                    "seed": seed,
                    "speedup_3090_vs_5060_point": speedup_point,
                    "speedup_ci95_low": speedup_low,
                    "speedup_ci95_high": speedup_high,
                    "energy_saving_5060_vs_3090_point_pct": saving_point,
                    "energy_saving_ci95_low_pct": saving_low,
                    "energy_saving_ci95_high_pct": saving_high,
                    "speedup_ci_crosses_neutral_1": bool(
                        speedup_low <= 1.0 <= speedup_high
                    ),
                    "energy_ci_crosses_neutral_0": bool(
                        saving_low <= 0.0 <= saving_high
                    ),
                }
            )

    result = pd.DataFrame(rows)
    if len(result) != 15:
        raise RuntimeError(f"Expected 15 cells, got {len(result)}")
    return result


def verify_frozen_claims(intervals: pd.DataFrame) -> None:
    median_speedup = float(
        intervals["speedup_3090_vs_5060_point"].median()
    )
    median_saving = float(
        intervals["energy_saving_5060_vs_3090_point_pct"].median()
    )

    failures: list[str] = []

    if not np.isclose(
        median_speedup,
        EXPECTED_MEDIAN_SPEEDUP,
        rtol=0.0,
        atol=5e-7,
    ):
        failures.append(
            f"median speedup: expected {EXPECTED_MEDIAN_SPEEDUP}, "
            f"got {median_speedup}"
        )

    if not np.isclose(
        median_saving,
        EXPECTED_MEDIAN_SAVING_PCT,
        rtol=0.0,
        atol=5e-5,
    ):
        failures.append(
            f"median energy saving: expected {EXPECTED_MEDIAN_SAVING_PCT}, "
            f"got {median_saving}"
        )

    if (intervals["speedup_3090_vs_5060_point"] <= 1.0).any():
        failures.append("At least one point estimate does not favor the RTX 3090")
    if (
        intervals["energy_saving_5060_vs_3090_point_pct"] <= 0.0
    ).any():
        failures.append("At least one point estimate does not favor the RTX 5060 Ti")

    if intervals["speedup_ci_crosses_neutral_1"].any():
        failures.append("At least one speedup CI crosses S=1")
    if intervals["energy_ci_crosses_neutral_0"].any():
        failures.append("At least one energy-saving CI crosses G=0")

    if failures:
        raise RuntimeError(
            "Frozen large-regime verification failed:\n- "
            + "\n- ".join(failures)
        )


def asymmetric_errors(
    point: pd.Series,
    low: pd.Series,
    high: pd.Series,
) -> np.ndarray:
    return np.vstack(
        [
            point.to_numpy(float) - low.to_numpy(float),
            high.to_numpy(float) - point.to_numpy(float),
        ]
    )


def plot_figure(
    intervals: pd.DataFrame,
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Create the caption-consistent main plot and zoomed inset."""

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    workload_colors = {
        workload: cycle[index % len(cycle)]
        for index, workload in enumerate(WORKLOADS)
    }

    for workload in WORKLOADS:
        group = intervals[intervals["workload"] == workload].sort_values("size")
        x = group["speedup_3090_vs_5060_point"]
        y = group["energy_saving_5060_vs_3090_point_pct"]
        xerr = asymmetric_errors(
            x,
            group["speedup_ci95_low"],
            group["speedup_ci95_high"],
        )
        yerr = asymmetric_errors(
            y,
            group["energy_saving_ci95_low_pct"],
            group["energy_saving_ci95_high_pct"],
        )

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            capsize=2,
            markersize=5,
            linewidth=0.8,
            label=workload,
            color=workload_colors[workload],
        )

    # Neutral boundaries must be visible in the main axes.
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xlim(0.95, 2.17)
    ax.set_ylim(-2.0, 53.5)
    ax.set_xlabel("RTX 3090 speedup over RTX 5060 Ti")
    ax.set_ylabel("RTX 5060 Ti board-energy saving (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    median_speedup = intervals["speedup_3090_vs_5060_point"].median()
    median_saving = intervals[
        "energy_saving_5060_vs_3090_point_pct"
    ].median()
    ax.text(
        0.98,
        0.04,
        "15/15 runtime-energy conflicts\n"
        f"Median speedup: {median_speedup:.2f}×\n"
        f"Median energy saving: {median_saving:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )

    # Zoomed view of the observed points and their small whiskers.
    inset = inset_axes(
        ax,
        width="52%",
        height="50%",
        loc="lower left",
        borderpad=3.0,
    )

    for workload in WORKLOADS:
        group = intervals[intervals["workload"] == workload].sort_values("size")
        x = group["speedup_3090_vs_5060_point"]
        y = group["energy_saving_5060_vs_3090_point_pct"]
        inset.errorbar(
            x,
            y,
            xerr=asymmetric_errors(
                x,
                group["speedup_ci95_low"],
                group["speedup_ci95_high"],
            ),
            yerr=asymmetric_errors(
                y,
                group["energy_saving_ci95_low_pct"],
                group["energy_saving_ci95_high_pct"],
            ),
            fmt="o",
            capsize=2,
            markersize=4,
            linewidth=0.8,
            color=workload_colors[workload],
        )

    x_low = float(intervals["speedup_ci95_low"].min())
    x_high = float(intervals["speedup_ci95_high"].max())
    y_low = float(intervals["energy_saving_ci95_low_pct"].min())
    y_high = float(intervals["energy_saving_ci95_high_pct"].max())

    x_pad = max(0.003, 0.06 * (x_high - x_low))
    y_pad = max(0.5, 0.06 * (y_high - y_low))
    inset.set_xlim(x_low - x_pad, x_high + x_pad)
    inset.set_ylim(y_low - y_pad, y_high + y_pad)
    inset.grid(alpha=0.25)
    inset.tick_params(labelsize=7)
    inset.set_title("Observed cells", fontsize=8)

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.98)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Reproduce the large-regime bootstrap figure."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=(
            repo_root
            / "data"
            / "snapshots"
            / "t1-analysis-20260730"
            / "extracted"
            / "deep_research_energy"
            / "inputs"
        ),
        help="Directory containing the frozen session CSV inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "large-regime",
        help="Directory receiving the CSV, PDF, and PNG outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Bootstrap seed (default: {SEED})",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=BOOTSTRAPS,
        help=f"Number of bootstrap replicates (default: {BOOTSTRAPS})",
    )
    parser.add_argument(
        "--skip-frozen-check",
        action="store_true",
        help="Generate outputs without checking the frozen paper claims",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = args.inputs.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not inputs.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {inputs}")

    sessions = load_sessions(inputs)
    intervals = compute_cell_intervals(
        sessions,
        seed=args.seed,
        bootstraps=args.bootstraps,
    )

    if not args.skip_frozen_check:
        verify_frozen_claims(intervals)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "large_15_cell_bootstrap_intervals.csv"
    pdf_path = output_dir / "large_regime_speedup_energy.pdf"
    png_path = output_dir / "large_regime_speedup_energy.png"

    intervals.to_csv(csv_path, index=False)
    plot_figure(intervals, pdf_path=pdf_path, png_path=png_path)

    median_speedup = intervals["speedup_3090_vs_5060_point"].median()
    median_saving = intervals[
        "energy_saving_5060_vs_3090_point_pct"
    ].median()

    print(intervals.to_string(index=False))
    print()
    print(f"Median speedup: {median_speedup:.8f}x")
    print(f"Median energy saving: {median_saving:.8f}%")
    print("Speedup intervals crossing S=1: 0/15")
    print("Energy intervals crossing G=0: 0/15")
    print("PASS: large-regime figure and cell-specific intervals reproduced")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
