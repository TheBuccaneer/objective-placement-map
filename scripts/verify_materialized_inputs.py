#!/usr/bin/env python3
"""Verify a canonical input build produced from a source snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import NoReturn


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_hash_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError:
            fail(f"invalid hash-manifest line {line_number}: {raw!r}")
        if relative in entries:
            fail(f"duplicate hash-manifest path: {relative}")
        entries[relative] = digest
    if not entries:
        fail("hash manifest is empty")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_build", type=Path)
    args = parser.parse_args()

    root = args.input_build.expanduser().resolve()
    if not root.is_dir():
        fail(f"input build missing: {root}")

    required = [
        root / "INPUT_MANIFEST.json",
        root / "MANIFEST.sha256",
        root / "MATERIALIZED_INPUT_MAP.csv",
        root / "README.txt",
        root / "inputs",
    ]
    for path in required:
        if not path.exists():
            fail(f"required input-build path missing: {path}")

    manifest = json.loads(
        (root / "INPUT_MANIFEST.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != "analysis-input-materialization-v1":
        fail("unsupported input-materialization schema")

    entries = parse_hash_manifest(root / "MANIFEST.sha256")
    exempt = {"INPUT_MANIFEST.json", "MANIFEST.sha256"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in exempt
    }
    if set(entries) != actual:
        fail(
            "input-build file-set mismatch; "
            f"missing={sorted(set(entries)-actual)}, "
            f"unexpected={sorted(actual-set(entries))}"
        )
    for relative, expected in sorted(entries.items()):
        actual_hash = sha256(root / relative)
        if actual_hash != expected:
            fail(
                f"hash mismatch for {relative}: expected {expected}, got {actual_hash}"
            )

    with (root / "MATERIALIZED_INPUT_MAP.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    result = manifest.get("result", {})
    expected_count = int(result.get("input_count", -1))
    if expected_count != 28 or len(rows) != 28:
        fail(
            f"expected 28 materialized inputs; manifest={expected_count}, map={len(rows)}"
        )
    if result.get("complete_exact_reference_match") is not True:
        fail("input build does not exactly match the frozen reference")

    input_files = sorted((root / "inputs").glob("*.csv"))
    if len(input_files) != 28:
        fail(f"expected 28 input CSVs, found {len(input_files)}")

    mapped_names = {row["canonical_name"] for row in rows}
    actual_names = {path.name for path in input_files}
    if mapped_names != actual_names:
        fail(
            "canonical input-name mismatch; "
            f"missing={sorted(mapped_names-actual_names)}, "
            f"unexpected={sorted(actual_names-mapped_names)}"
        )

    for row in rows:
        path = root / "inputs" / row["canonical_name"]
        actual_hash = sha256(path)
        if actual_hash != row["sha256"]:
            fail(f"mapped input hash mismatch: {row['canonical_name']}")
        if row["exact_reference_match"].lower() != "true":
            fail(f"input is not an exact reference match: {row['canonical_name']}")

    print(f"Input-build verification OK: {len(input_files)}/28 files")
    print(f"Source snapshot: {manifest['source_snapshot']['id']}")
    print(
        "Measurement commit: "
        f"{manifest['source_snapshot']['measurement_commit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
