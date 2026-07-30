from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_input_materialization_configuration() -> None:
    config = yaml.safe_load(
        (ROOT / "config/input_materialization.yaml").read_text(encoding="utf-8")
    )["input_materialization"]
    assert config["schema_version"] == "analysis-input-materialization-v1"
    assert config["expected_input_count"] == 28
    assert config["require_exact_reference_match"] is True


def test_pinned_source_snapshot_maps_all_inputs() -> None:
    snapshots = sorted((ROOT / "data/source-snapshots").glob("energy-*"))
    assert len(snapshots) >= 1
    with (snapshots[-1] / "SOURCE_INPUT_MAP.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 28
    assert len({row["canonical_name"] for row in rows}) == 28
    assert all(row["status"] in {"exact_unique", "exact_multiple"} for row in rows)


def test_makefile_and_runner_expose_full_reproduction() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_analysis.py").read_text(encoding="utf-8")
    assert "build-inputs:" in makefile
    assert "reproduce:" in makefile
    assert "verify-input-builds:" in makefile
    assert "--inputs-from-source-snapshot" in runner
    assert '"analysis_inputs": input_materialization' in runner
