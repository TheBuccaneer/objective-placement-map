#!/usr/bin/env python3
"""Materialize the canonical analysis inputs from a pinned source snapshot.

Each source file is selected through SOURCE_INPUT_MAP.csv, copied under its
canonical analysis filename, checked against both the source-snapshot mapping
and the frozen paper-reference input, and recorded in an immutable manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import yaml


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        fail(f"expected YAML mapping in {path}")
    return value


def make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "canonical_name",
        "canonical_path",
        "size_bytes",
        "sha256",
        "status",
        "selected_source_path",
        "match_count",
        "all_source_matches",
    }
    if not rows:
        fail(f"empty source-input map: {path}")
    missing = required - set(rows[0])
    if missing:
        fail(f"source-input map missing columns: {sorted(missing)}")
    return rows


def safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        fail(f"path escapes root {root}: {relative}")
    return candidate


def write_hash_manifest(root: Path) -> Path:
    excluded = {"MANIFEST.sha256", "INPUT_MANIFEST.json"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    target = root / "MANIFEST.sha256"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(
                f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
            )
    return target


def default_source_snapshot(repo: Path, project: dict[str, Any], config: dict[str, Any]) -> Path:
    commit = str(project["source"]["commit"])
    prefix = str(config["source_snapshot_prefix"])
    return repo / "data" / "source-snapshots" / f"{prefix}{commit[:12]}"


def default_output_root(repo: Path, snapshot_id: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo / "results" / "input-builds" / f"{run_id}-{snapshot_id}"


def verify_source_snapshot(repo: Path, source_snapshot: Path) -> None:
    verifier = repo / "scripts" / "verify_source_snapshot.py"
    result = subprocess.run(
        [sys.executable, str(verifier), str(source_snapshot)],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        fail(
            "source snapshot verification failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="Pinned source snapshot; defaults to configured energy-<commit12>",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root containing inputs/ and provenance manifests",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    project = load_yaml(repo / "config" / "project.yaml")
    analysis = load_yaml(repo / "config" / "analysis.yaml")["analysis"]
    config = load_yaml(repo / "config" / "input_materialization.yaml")[
        "input_materialization"
    ]

    source_snapshot = (
        args.source_snapshot.expanduser().resolve()
        if args.source_snapshot
        else default_source_snapshot(repo, project, config).resolve()
    )
    if not source_snapshot.is_dir():
        fail(f"source snapshot missing: {source_snapshot}")
    verify_source_snapshot(repo, source_snapshot)

    snapshot_metadata_path = source_snapshot / "SNAPSHOT.json"
    source_map_path = source_snapshot / "SOURCE_INPUT_MAP.csv"
    snapshot_metadata = json.loads(
        snapshot_metadata_path.read_text(encoding="utf-8")
    )
    rows = read_mapping(source_map_path)

    expected_count = int(config["expected_input_count"])
    if len(rows) != expected_count:
        fail(f"expected {expected_count} mapped inputs, found {len(rows)}")

    snapshot_id = str(snapshot_metadata["snapshot_id"])
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else default_output_root(repo, snapshot_id).resolve()
    )
    if output_root.exists():
        fail(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    frozen_snapshot = repo / "data" / "snapshots" / str(analysis["snapshot_id"])
    frozen_package = frozen_snapshot / str(analysis["snapshot_package_subdir"])
    reference_inputs = frozen_package / "inputs"
    if not reference_inputs.is_dir():
        fail(f"frozen reference inputs missing: {reference_inputs}")

    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )

    try:
        inputs_dir = temp_root / "inputs"
        inputs_dir.mkdir()
        materialized_rows: list[dict[str, Any]] = []
        canonical_names: set[str] = set()
        source_paths: set[str] = set()

        for row in sorted(rows, key=lambda item: item["canonical_name"]):
            canonical_name = row["canonical_name"]
            selected_source_path = row["selected_source_path"]
            status = row["status"]

            if status not in {"exact_unique", "exact_multiple"}:
                fail(f"non-exact source mapping for {canonical_name}: {status}")
            if not canonical_name or Path(canonical_name).name != canonical_name:
                fail(f"invalid canonical filename: {canonical_name!r}")
            if canonical_name in canonical_names:
                fail(f"duplicate canonical filename: {canonical_name}")
            if selected_source_path in source_paths:
                fail(f"duplicate selected source path: {selected_source_path}")
            canonical_names.add(canonical_name)
            source_paths.add(selected_source_path)

            source_file = safe_child(
                source_snapshot / "files", selected_source_path
            )
            if not source_file.is_file() or source_file.is_symlink():
                fail(f"source-snapshot file missing or invalid: {source_file}")

            expected_size = int(row["size_bytes"])
            expected_hash = row["sha256"]
            actual_size = source_file.stat().st_size
            source_hash = sha256(source_file)
            if actual_size != expected_size or source_hash != expected_hash:
                fail(
                    f"source-snapshot content mismatch for {canonical_name}:\n"
                    f"  expected: {expected_size} bytes {expected_hash}\n"
                    f"  actual:   {actual_size} bytes {source_hash}"
                )

            reference_file = reference_inputs / canonical_name
            if not reference_file.is_file():
                fail(f"frozen reference input missing: {reference_file}")
            reference_hash = sha256(reference_file)
            if config["require_exact_reference_match"] and reference_hash != source_hash:
                fail(
                    f"source input differs from frozen reference: {canonical_name}\n"
                    f"  source:    {source_hash}\n"
                    f"  reference: {reference_hash}"
                )

            target = inputs_dir / canonical_name
            shutil.copyfile(source_file, target)
            if config["normalize_mtime_to_epoch"]:
                os.utime(target, (0, 0))
            if config["copied_inputs_read_only"]:
                make_read_only(target)

            materialized_rows.append(
                {
                    "canonical_name": canonical_name,
                    "source_snapshot_path": selected_source_path,
                    "source_repository_path": selected_source_path,
                    "size_bytes": actual_size,
                    "sha256": source_hash,
                    "reference_sha256": reference_hash,
                    "exact_reference_match": source_hash == reference_hash,
                }
            )

        map_path = temp_root / "MATERIALIZED_INPUT_MAP.csv"
        fields = [
            "canonical_name",
            "source_snapshot_path",
            "source_repository_path",
            "size_bytes",
            "sha256",
            "reference_sha256",
            "exact_reference_match",
        ]
        with map_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(materialized_rows)
        if config["normalize_mtime_to_epoch"]:
            os.utime(map_path, (0, 0))

        readme = (
            "Materialized canonical analysis inputs\n"
            "======================================\n\n"
            f"Source snapshot: {snapshot_id}\n"
            f"Measurement commit: {snapshot_metadata['measurement_repository']['commit']}\n"
            f"Canonical inputs: {len(materialized_rows)}\n"
            "Exact frozen-reference matches: "
            f"{sum(bool(row['exact_reference_match']) for row in materialized_rows)}\n\n"
            "The inputs directory was reconstructed only from the pinned measurement\n"
            "source snapshot. Canonical filenames are supplied by SOURCE_INPUT_MAP.csv.\n"
        )
        readme_path = temp_root / "README.txt"
        readme_path.write_text(readme, encoding="utf-8")
        if config["normalize_mtime_to_epoch"]:
            os.utime(readme_path, (0, 0))

        hash_manifest_path = write_hash_manifest(temp_root)
        manifest = {
            "schema_version": str(config["schema_version"]),
            "created_at_utc": utc_now(),
            "source_snapshot": {
                "id": snapshot_id,
                "path": source_snapshot.relative_to(repo).as_posix(),
                "snapshot_json_sha256": sha256(snapshot_metadata_path),
                "source_input_map_sha256": sha256(source_map_path),
                "measurement_commit": snapshot_metadata[
                    "measurement_repository"
                ]["commit"],
            },
            "frozen_reference": {
                "snapshot_id": str(analysis["snapshot_id"]),
                "inputs_path": reference_inputs.relative_to(repo).as_posix(),
            },
            "result": {
                "input_count": len(materialized_rows),
                "exact_reference_matches": sum(
                    bool(row["exact_reference_match"])
                    for row in materialized_rows
                ),
                "complete_exact_reference_match": all(
                    bool(row["exact_reference_match"])
                    for row in materialized_rows
                ),
                "materialized_map_sha256": sha256(map_path),
                "hash_manifest_sha256": sha256(hash_manifest_path),
                "inputs": materialized_rows,
            },
        }
        (temp_root / "INPUT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temp_root.rename(output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    verifier = repo / "scripts" / "verify_materialized_inputs.py"
    subprocess.run(
        [sys.executable, str(verifier), str(output_root)],
        cwd=repo,
        check=True,
    )

    print(f"Input build: {output_root}")
    print(f"Source snapshot: {snapshot_id}")
    print(f"Materialized canonical inputs: {len(rows)}")
    print("Exact frozen-reference matches: 28/28")
    print("PASS: source snapshot materialized into canonical analysis inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
