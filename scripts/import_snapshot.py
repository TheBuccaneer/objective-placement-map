#!/usr/bin/env python3
"""Import an analysis ZIP as an immutable, hashed snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "--version"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("snapshot_dir", type=Path)
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()

    if not archive.is_file():
        parser.error(f"archive does not exist: {archive}")
    if snapshot_dir.exists():
        parser.error(f"snapshot directory already exists: {snapshot_dir}")

    snapshot_dir.mkdir(parents=True)
    copied_archive = snapshot_dir / "SOURCE_ARCHIVE.zip"
    shutil.copy2(archive, copied_archive)

    extracted = snapshot_dir / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(copied_archive) as zf:
        bad_member = next(
            (
                name
                for name in zf.namelist()
                if Path(name).is_absolute() or ".." in Path(name).parts
            ),
            None,
        )
        if bad_member:
            raise RuntimeError(f"unsafe ZIP member: {bad_member}")
        for member in zf.infolist():
            member_path = Path(member.filename)
            if "__pycache__" in member_path.parts or member_path.suffix in {".pyc", ".pyo"}:
                continue
            zf.extract(member, extracted)

    manifest_lines: list[str] = []
    for file_path in sorted(p for p in snapshot_dir.rglob("*") if p.is_file()):
        if file_path.name in {"MANIFEST.sha256", "SNAPSHOT.json"}:
            continue
        rel = file_path.relative_to(snapshot_dir)
        manifest_lines.append(f"{sha256(file_path)}  {rel.as_posix()}")

    (snapshot_dir / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )

    metadata = {
        "snapshot_format": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_original_path": str(archive),
        "source_archive_name": archive.name,
        "source_archive_sha256": sha256(copied_archive),
        "python_version": sys.version,
        "git_version": git_version(),
        "file_count": len(manifest_lines),
    }
    (snapshot_dir / "SNAPSHOT.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    make_read_only(snapshot_dir)
    print(f"Imported immutable snapshot: {snapshot_dir}")
    print(f"Files hashed: {len(manifest_lines)}")
    print(f"Archive SHA-256: {metadata['source_archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
