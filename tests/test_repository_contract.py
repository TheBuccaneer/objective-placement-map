from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_active_analysis_exists() -> None:
    assert (ROOT / "src/placement_analysis/analysis.py").is_file()


def test_snapshot_inputs_exist() -> None:
    package = (
        ROOT
        / "data/snapshots/t1-analysis-20260730/extracted/deep_research_energy"
    )
    inputs = sorted((package / "inputs").glob("*.csv"))
    assert len(inputs) >= 20
    assert (package / "inputs/axpy_sessions.csv").is_file()


def test_analysis_contract() -> None:
    config = yaml.safe_load((ROOT / "config/analysis.yaml").read_text())
    assert config["analysis"]["expected_csv_tables"] == 37
    assert config["analysis"]["expected_figures"] == 5
    assert config["analysis"]["exact_reference_comparison"] is True
