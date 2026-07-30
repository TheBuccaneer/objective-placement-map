#!/usr/bin/env python3
"""Verify a source snapshot and its direct mapping to canonical inputs."""

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


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError:
            fail(f"invalid manifest line {number}: {raw!r}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            fail(f"invalid SHA-256 at manifest line {number}")
        if relative in entries:
            fail(f"duplicate manifest path: {relative}")
        entries[relative] = digest
    if not entries:
        fail("manifest is empty")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()

    root = args.snapshot.expanduser().resolve()
    if not root.is_dir():
        fail(f"snapshot directory missing: {root}")

    manifest_path = root / "MANIFEST.sha256"
    snapshot_path = root / "SNAPSHOT.json"
    map_path = root / "SOURCE_INPUT_MAP.csv"
    for required in [manifest_path, snapshot_path, map_path, root / "README.txt"]:
        if not required.is_file():
            fail(f"required snapshot file missing: {required}")

    entries = parse_manifest(manifest_path)
    exempt = {"MANIFEST.sha256", "SNAPSHOT.json"}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in exempt
    }
    listed_files = set(entries)
    missing = sorted(listed_files - actual_files)
    unexpected = sorted(actual_files - listed_files)
    if missing or unexpected:
        fail(f"manifest file-set mismatch; missing={missing}, unexpected={unexpected}")

    for relative, expected in sorted(entries.items()):
        path = root / relative
        actual = sha256(path)
        if actual != expected:
            fail(
                f"hash mismatch for {relative}:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != "source-snapshot-v1":
        fail("unsupported source snapshot schema")
    if snapshot.get("snapshot_id") != root.name:
        fail("snapshot id does not match directory name")
    if snapshot["contents"]["manifest_sha256"] != sha256(manifest_path):
        fail("manifest hash does not match SNAPSHOT.json")

    with map_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = int(snapshot["contents"]["file_count"])
    if len(rows) != expected_count:
        fail(f"map row count {len(rows)} does not equal snapshot count {expected_count}")

    snapshot_files = snapshot["contents"]["files"]
    by_name = {item["canonical_name"]: item for item in snapshot_files}
    if len(by_name) != expected_count:
        fail("duplicate canonical names in SNAPSHOT.json")

    total_bytes = 0
    for row in rows:
        name = row["canonical_name"]
        if name not in by_name:
            fail(f"canonical input missing from SNAPSHOT.json: {name}")
        item = by_name[name]
        if item["source_path"] != row["selected_source_path"]:
            fail(f"source path mismatch for {name}")
        if item["sha256"] != row["sha256"]:
            fail(f"source hash mismatch for {name}")
        if int(item["size_bytes"]) != int(row["size_bytes"]):
            fail(f"source size mismatch for {name}")
        copied = root / item["snapshot_path"]
        if not copied.is_file():
            fail(f"copied source file missing for {name}: {copied}")
        if sha256(copied) != row["sha256"]:
            fail(f"copied source content mismatch for {name}")
        total_bytes += copied.stat().st_size

    if total_bytes != int(snapshot["contents"]["total_bytes"]):
        fail("snapshot total-byte count is inconsistent")

    commit = snapshot["measurement_repository"]["commit"]
    print(f"Source snapshot verification OK: {len(rows)}/{len(rows)} files")
    print(f"Snapshot: {root.name}")
    print(f"Measurement commit: {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
