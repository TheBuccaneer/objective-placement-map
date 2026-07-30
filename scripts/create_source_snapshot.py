#!/usr/bin/env python3
"""Create an immutable, Git-trackable snapshot of the exact source inputs.

The command consumes a successful source-inventory run, copies only the 28
selected files from the pinned energy/new commit while preserving their source
relative paths, writes a SHA-256 manifest, and verifies the result before
publishing it under data/source-snapshots/energy-<commit12>/.
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        fail(f"expected a YAML mapping in {path}")
    return value


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        fail(
            f"git {' '.join(args)} failed in {repo}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def source_remote(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def latest_complete_inventory(repo: Path, expected_commit: str) -> Path:
    root = repo / "results" / "source-inventory"
    candidates: list[Path] = []
    if root.is_dir():
        for path in sorted(root.iterdir()):
            manifest_path = path / "SOURCE_INVENTORY.json"
            map_path = path / "SOURCE_INPUT_MAP.csv"
            if not path.is_dir() or not manifest_path.is_file() or not map_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            measurement = manifest.get("measurement_repository", {})
            result = manifest.get("result", {})
            if (
                measurement.get("commit") == expected_commit
                and result.get("complete_direct_provenance") is True
            ):
                candidates.append(path)
    if not candidates:
        fail(
            "no complete source inventory found for configured commit; run:\n"
            "  make source-inventory SOURCE_REPO=~/projects/energy"
        )
    return candidates[-1]


def read_map(path: Path) -> list[dict[str, str]]:
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
        fail(f"source-input map is missing columns: {sorted(missing)}")
    return rows


def safe_source_path(source_repo: Path, relative: str) -> Path:
    candidate = (source_repo / relative).resolve()
    try:
        candidate.relative_to(source_repo)
    except ValueError:
        fail(f"source path escapes repository: {relative}")
    if not candidate.is_file() or candidate.is_symlink():
        fail(f"selected source file is missing or invalid: {candidate}")
    return candidate


def make_read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def write_manifest(snapshot_root: Path) -> Path:
    excluded = {"MANIFEST.sha256", "SNAPSHOT.json"}
    files = sorted(
        path
        for path in snapshot_root.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    manifest_path = snapshot_root / "MANIFEST.sha256"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(
                f"{sha256(path)}  {relative_posix(path, snapshot_root)}\n"
            )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo",
        required=True,
        type=Path,
        help="Local clean clone of github.com/TheBuccaneer/energy",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help="Explicit source-inventory directory; defaults to latest complete run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing snapshot only when its verification fails",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    project = load_yaml(repo / "config/project.yaml")
    config = load_yaml(repo / "config/source_snapshot.yaml")["source_snapshot"]

    source_repo = args.source_repo.expanduser().resolve()
    if not (source_repo / ".git").exists():
        fail(f"source repository is not a Git clone: {source_repo}")

    configured_commit = str(project["source"]["commit"])
    actual_commit = run_git(source_repo, "rev-parse", "HEAD")
    source_status = run_git(source_repo, "status", "--porcelain")

    if config["require_exact_configured_commit"] and actual_commit != configured_commit:
        fail(
            "measurement repository commit mismatch:\n"
            f"  configured: {configured_commit}\n"
            f"  actual:     {actual_commit}"
        )
    if config["require_clean_source_repository"] and source_status:
        fail("measurement repository is not clean:\n" + source_status)

    inventory = (
        args.inventory.expanduser().resolve()
        if args.inventory
        else latest_complete_inventory(repo, actual_commit)
    )
    inventory_manifest_path = inventory / "SOURCE_INVENTORY.json"
    inventory_map_path = inventory / "SOURCE_INPUT_MAP.csv"
    if not inventory_manifest_path.is_file() or not inventory_map_path.is_file():
        fail(f"invalid inventory directory: {inventory}")

    inventory_manifest = json.loads(
        inventory_manifest_path.read_text(encoding="utf-8")
    )
    if inventory_manifest["measurement_repository"]["commit"] != actual_commit:
        fail("inventory commit does not match source repository commit")
    if inventory_manifest["result"]["complete_direct_provenance"] is not True:
        fail("inventory does not provide complete direct provenance")
    if inventory_manifest["result"]["map_sha256"] != sha256(inventory_map_path):
        fail("source-input map hash differs from inventory manifest")

    rows = read_map(inventory_map_path)
    expected_count = int(config["expected_input_count"])
    if len(rows) != expected_count:
        fail(f"expected {expected_count} source inputs, found {len(rows)}")

    prefix = str(config["directory_prefix"])
    snapshot_id = f"{prefix}{actual_commit[:12]}"
    final_root = repo / "data" / "source-snapshots" / snapshot_id

    if final_root.exists():
        verifier = repo / "scripts" / "verify_source_snapshot.py"
        result = subprocess.run(
            [sys.executable, str(verifier), str(final_root)],
            cwd=repo,
            check=False,
        )
        if result.returncode == 0:
            print(f"Source snapshot already exists and verifies: {final_root}")
            return 0
        if not args.force:
            fail(
                f"existing snapshot failed verification: {final_root}\n"
                "rerun with --force only after inspecting the failure"
            )
        shutil.rmtree(final_root)

    parent = final_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=parent)
    )

    try:
        files_root = temp_root / "files"
        files_root.mkdir(parents=True)

        copied: list[dict[str, Any]] = []
        selected_paths: set[str] = set()
        canonical_names: set[str] = set()

        for row in rows:
            status = row["status"]
            selected_rel = row["selected_source_path"]
            canonical_name = row["canonical_name"]
            if status not in {"exact_unique", "exact_multiple"}:
                fail(f"input is not exactly matched: {canonical_name}: {status}")
            if not selected_rel:
                fail(f"input has no selected source path: {canonical_name}")
            if selected_rel in selected_paths:
                fail(f"duplicate selected source path: {selected_rel}")
            if canonical_name in canonical_names:
                fail(f"duplicate canonical input name: {canonical_name}")
            selected_paths.add(selected_rel)
            canonical_names.add(canonical_name)

            source_path = safe_source_path(source_repo, selected_rel)
            expected_size = int(row["size_bytes"])
            expected_hash = row["sha256"]
            actual_size = source_path.stat().st_size
            actual_hash = sha256(source_path)
            if actual_size != expected_size or actual_hash != expected_hash:
                fail(
                    f"source file changed since inventory: {selected_rel}\n"
                    f"  expected size/hash: {expected_size} {expected_hash}\n"
                    f"  actual size/hash:   {actual_size} {actual_hash}"
                )

            target = files_root / selected_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            os.utime(target, (0, 0))
            if config["copied_files_read_only"]:
                make_read_only(target)

            copied.append(
                {
                    "canonical_name": canonical_name,
                    "canonical_path": row["canonical_path"],
                    "source_path": selected_rel,
                    "snapshot_path": relative_posix(target, temp_root),
                    "size_bytes": actual_size,
                    "sha256": actual_hash,
                }
            )

        shutil.copyfile(inventory_map_path, temp_root / "SOURCE_INPUT_MAP.csv")
        os.utime(temp_root / "SOURCE_INPUT_MAP.csv", (0, 0))

        readme = (
            "Pinned measurement-source snapshot\n"
            "==================================\n\n"
            f"Source repository: {source_remote(source_repo) or '(no origin)'}\n"
            f"Source commit: {actual_commit}\n"
            f"Source subtree: {project['source']['subtree']}\n"
            f"Copied canonical inputs: {len(copied)}\n"
            f"Inventory run: {inventory.name}\n\n"
            "The files directory preserves each selected path relative to the\n"
            "measurement repository root. Files are copied byte-for-byte and\n"
            "made read-only. MANIFEST.sha256 verifies all materialized content.\n"
        )
        (temp_root / "README.txt").write_text(readme, encoding="utf-8")
        os.utime(temp_root / "README.txt", (0, 0))

        manifest_path = write_manifest(temp_root)
        snapshot = {
            "schema_version": str(config["schema_version"]),
            "created_at_utc": utc_now(),
            "snapshot_id": snapshot_id,
            "analysis_repository": {
                "commit": run_git(repo, "rev-parse", "HEAD"),
                "dirty": bool(run_git(repo, "status", "--porcelain")),
            },
            "measurement_repository": {
                "origin": source_remote(source_repo),
                "commit": actual_commit,
                "configured_commit": configured_commit,
                "commit_timestamp": run_git(
                    source_repo, "show", "-s", "--format=%cI", actual_commit
                ),
                "subtree": str(project["source"]["subtree"]),
                "clean_at_import": not bool(source_status),
            },
            "source_inventory": {
                "run_id": inventory.name,
                "inventory_manifest_sha256": sha256(inventory_manifest_path),
                "source_input_map_sha256": sha256(inventory_map_path),
            },
            "contents": {
                "file_count": len(copied),
                "total_bytes": sum(int(item["size_bytes"]) for item in copied),
                "manifest_sha256": sha256(manifest_path),
                "files": copied,
            },
        }
        (temp_root / "SNAPSHOT.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temp_root.rename(final_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    verifier = repo / "scripts" / "verify_source_snapshot.py"
    subprocess.run(
        [sys.executable, str(verifier), str(final_root)],
        cwd=repo,
        check=True,
    )

    print(f"Source snapshot created: {final_root}")
    print(f"Source commit: {actual_commit}")
    print(f"Copied files: {len(rows)}")
    print("PASS: immutable source snapshot created and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
