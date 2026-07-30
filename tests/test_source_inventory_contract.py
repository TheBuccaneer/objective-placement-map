from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "inventory_source_inputs", ROOT / "scripts/inventory_source_inputs.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_inventory_configuration() -> None:
    config = yaml.safe_load(
        (ROOT / "config/source_inventory.yaml").read_text(encoding="utf-8")
    )["source_inventory"]
    assert config["schema_version"] == "source-inventory-v1"
    assert config["subtree"] == "new"
    assert config["expected_input_count"] == 28
    assert config["require_exact_configured_commit"] is True
    assert config["require_clean_source_repository"] is True


def test_inventory_script_is_active() -> None:
    assert (ROOT / "scripts/inventory_source_inputs.py").is_file()


def test_choose_match_prefers_same_basename_then_shorter_path(tmp_path: Path) -> None:
    subtree = tmp_path / "new"
    short = subtree / "audit" / "target.csv"
    long = subtree / "deep" / "nested" / "renamed.csv"
    short.parent.mkdir(parents=True)
    long.parent.mkdir(parents=True)
    short.write_text("same", encoding="utf-8")
    long.write_text("same", encoding="utf-8")

    selected = MODULE.choose_match([long, short], subtree, "target.csv")
    assert selected == short
