import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/camera-ready"


def test_uniform_conflict_analysis():
    # Reproduce the camera-ready analysis
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/camera_ready_uniform_conflicts.py"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    cells = pd.read_csv(OUT / "uniform_conflicts_51_cells.csv")
    details = pd.read_csv(OUT / "uniform_conflict_details_25_cells.csv")
    all_loso = pd.read_csv(OUT / "uniform_loso_all_by_cell.csv")
    gpu_loso = pd.read_csv(OUT / "uniform_loso_gpu_by_cell.csv")

    # Uniform classification
    assert len(cells) == 51
    assert cells.exact_conflict.sum() == 25
    assert cells.unresolved_10pct.sum() == 22
    assert cells.unresolved_20pct.sum() == 19

    # Every unresolved cell must also be an exact conflict
    assert not (
        cells.unresolved_10pct & ~cells.exact_conflict
    ).any()

    # Exactly one difference from the legacy classification
    changed = cells[
        cells.exact_conflict != cells.legacy_conflict
    ]

    assert len(changed) == 1
    assert changed.iloc[0].workload == "STRIDED_GEMM"
    assert changed.iloc[0]["size"] == 512

    # Conflict detail table
    assert len(details) == 25
    assert not details.isna().any().any()

    # Leave-one-session-out robustness
    assert len(all_loso) == 25
    assert (all_loso.holdout_conflict_folds == 5).sum() == 22

    assert len(gpu_loso) == 24
    assert (gpu_loso.holdout_conflict_folds == 5).sum() == 23

    # Known boundary case
    boundary = gpu_loso[
        (gpu_loso.workload == "STRIDED_GEMM")
        & (gpu_loso["size"] == 512)
    ]

    assert len(boundary) == 1
    assert boundary.iloc[0].holdout_conflict_folds == 4
