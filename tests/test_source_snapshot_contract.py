from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CREATE = load_module(
    "create_source_snapshot", ROOT / "scripts/create_source_snapshot.py"
)
VERIFY = load_module(
    "verify_source_snapshot", ROOT / "scripts/verify_source_snapshot.py"
)


def test_source_snapshot_configuration() -> None:
    config = yaml.safe_load(
        (ROOT / "config/source_snapshot.yaml").read_text(encoding="utf-8")
    )["source_snapshot"]
    assert config["schema_version"] == "source-snapshot-v1"
    assert config["expected_input_count"] == 28
    assert config["require_exact_configured_commit"] is True
    assert config["require_clean_source_repository"] is True
    assert config["tracked_in_git"] is True


def test_snapshot_scripts_are_active() -> None:
    assert (ROOT / "scripts/create_source_snapshot.py").is_file()
    assert (ROOT / "scripts/verify_source_snapshot.py").is_file()


def test_manifest_parser_accepts_standard_format(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text("a" * 64 + "  files/new/example.csv\n", encoding="utf-8")
    assert VERIFY.parse_manifest(manifest) == {
        "files/new/example.csv": "a" * 64
    }


def test_safe_source_path_rejects_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x", encoding="utf-8")
    try:
        CREATE.safe_source_path(repo, "../outside.csv")
    except RuntimeError:
        pass
    else:
        raise AssertionError("path traversal was not rejected")
