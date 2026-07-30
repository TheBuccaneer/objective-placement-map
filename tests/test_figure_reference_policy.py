from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_analysis", ROOT / "scripts/run_analysis.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_png(path: Path, size: tuple[int, int], value: int) -> None:
    image = Image.new("RGBA", size, (value, value, value, 255))
    image.save(path)


def test_png_bytes_may_differ_when_contract_matches(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    actual = tmp_path / "actual"
    reference.mkdir()
    actual.mkdir()

    write_png(reference / "figure.png", (20, 10), 0)
    write_png(actual / "figure.png", (20, 10), 255)

    result = MODULE.compare_figure_contract(reference, actual)

    assert result["contract_match"] is True
    assert result["byte_identical_count"] == 0
    assert result["byte_changed"] == ["figure.png"]


def test_dimension_change_fails_contract(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    actual = tmp_path / "actual"
    reference.mkdir()
    actual.mkdir()

    write_png(reference / "figure.png", (20, 10), 0)
    write_png(actual / "figure.png", (21, 10), 0)

    result = MODULE.compare_figure_contract(reference, actual)

    assert result["contract_match"] is False
    assert "figure.png" in result["structural_changes"]
