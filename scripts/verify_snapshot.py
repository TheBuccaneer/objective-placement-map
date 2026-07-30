#!/usr/bin/env python3
"""Verify every file in an imported snapshot against its SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=Path)
    args = parser.parse_args()

    root = args.snapshot_dir.expanduser().resolve()
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        parser.error(f"manifest missing: {manifest}")

    checked = 0
    failures: list[str] = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            expected, rel = raw.split("  ", 1)
        except ValueError:
            failures.append(f"line {line_number}: malformed manifest entry")
            continue
        path = root / rel
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {rel}: expected {expected}, got {actual}")
        checked += 1

    if failures:
        print("Snapshot verification FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Snapshot verification OK: {checked}/{checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
