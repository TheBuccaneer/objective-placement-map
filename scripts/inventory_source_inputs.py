#!/usr/bin/env python3
"""Locate canonical analysis inputs inside the pinned measurement repository.

This command does not modify either repository. It compares the 28 frozen
analysis-input CSVs against regular files below energy/new using file size and
SHA-256, then writes a provenance inventory under results/source-inventory/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def fail(message: str) -> "NoReturn":
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        fail(f"expected a YAML mapping in {path}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_run_id(commit: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{commit[:12]}"


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


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


def choose_match(matches: list[Path], subtree_root: Path, canonical_name: str) -> Path:
    """Choose deterministically while preserving every alternate in the manifest."""
    if not matches:
        fail("cannot choose from an empty match set")

    def rank(path: Path) -> tuple[int, int, str]:
        rel = relative_posix(path, subtree_root)
        basename_penalty = 0 if path.name == canonical_name else 1
        return basename_penalty, len(Path(rel).parts), rel

    return sorted(matches, key=rank)[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo",
        type=Path,
        required=True,
        help="Local clone of github.com/TheBuccaneer/energy",
    )
    parser.add_argument("--run-id", help="Optional explicit inventory run id")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Return success even if some frozen inputs are not direct source files",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    project = load_yaml(repo / "config/project.yaml")
    inventory_config = load_yaml(repo / "config/source_inventory.yaml")[
        "source_inventory"
    ]

    source_repo = args.source_repo.expanduser().resolve()
    if not (source_repo / ".git").exists():
        fail(f"source repository is not a Git clone: {source_repo}")

    configured_commit = str(project["source"]["commit"])
    actual_commit = run_git(source_repo, "rev-parse", "HEAD")
    source_status = run_git(source_repo, "status", "--porcelain")

    if inventory_config["require_exact_configured_commit"]:
        if actual_commit != configured_commit:
            fail(
                "measurement repository commit mismatch:\n"
                f"  configured: {configured_commit}\n"
                f"  actual:     {actual_commit}"
            )

    if inventory_config["require_clean_source_repository"] and source_status:
        fail(
            "measurement repository working tree is not clean:\n"
            + source_status
        )

    subtree = str(inventory_config["subtree"])
    subtree_root = source_repo / subtree
    if not subtree_root.is_dir():
        fail(f"configured source subtree does not exist: {subtree_root}")

    snapshot_id = str(inventory_config["canonical_snapshot_id"])
    package_subdir = str(inventory_config["canonical_package_subdir"])
    inputs_subdir = str(inventory_config["canonical_inputs_subdir"])
    reference_inputs = (
        repo / "data" / "snapshots" / snapshot_id / package_subdir / inputs_subdir
    )
    if not reference_inputs.is_dir():
        fail(f"canonical input directory is missing: {reference_inputs}")

    canonical_files = sorted(reference_inputs.glob("*.csv"))
    expected_count = int(inventory_config["expected_input_count"])
    if len(canonical_files) != expected_count:
        fail(
            f"expected {expected_count} canonical CSV inputs, found "
            f"{len(canonical_files)}"
        )

    canonical: list[dict[str, Any]] = []
    sizes: set[int] = set()
    for path in canonical_files:
        stat = path.stat()
        item = {
            "canonical_name": path.name,
            "canonical_path": relative_posix(path, repo),
            "size_bytes": stat.st_size,
            "sha256": sha256(path),
        }
        canonical.append(item)
        sizes.add(stat.st_size)

    source_candidates: list[Path] = []
    files_seen = 0
    bytes_considered = 0
    for root, directories, filenames in os.walk(subtree_root):
        directories[:] = sorted(
            name for name in directories if name not in {".git", "__pycache__"}
        )
        for filename in sorted(filenames):
            path = Path(root) / filename
            try:
                if not path.is_file() or path.is_symlink():
                    continue
                files_seen += 1
                size = path.stat().st_size
                if inventory_config["hash_only_candidate_sizes"] and size not in sizes:
                    continue
                source_candidates.append(path)
                bytes_considered += size
            except OSError as exc:
                fail(f"cannot inspect source file {path}: {exc}")

    by_key: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path in source_candidates:
        key = (path.stat().st_size, sha256(path))
        by_key[key].append(path)

    rows: list[dict[str, Any]] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    selected_paths: set[str] = set()

    for item in canonical:
        key = (int(item["size_bytes"]), str(item["sha256"]))
        matches = sorted(
            by_key.get(key, []),
            key=lambda p: relative_posix(p, subtree_root),
        )
        if matches:
            selected = choose_match(
                matches, subtree_root, str(item["canonical_name"])
            )
            selected_rel = relative_posix(selected, source_repo)
            selected_paths.add(selected_rel)
            status = "exact_unique" if len(matches) == 1 else "exact_multiple"
            if len(matches) > 1:
                ambiguous.append(str(item["canonical_name"]))
        else:
            selected = None
            selected_rel = ""
            status = "unmatched"
            unmatched.append(str(item["canonical_name"]))

        rows.append(
            {
                "canonical_name": item["canonical_name"],
                "canonical_path": item["canonical_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "status": status,
                "selected_source_path": selected_rel,
                "match_count": len(matches),
                "all_source_matches": " | ".join(
                    relative_posix(path, source_repo) for path in matches
                ),
            }
        )

    run_id = args.run_id or safe_run_id(actual_commit)
    output_root = repo / "results" / "source-inventory" / run_id
    if output_root.exists():
        fail(f"inventory output already exists: {output_root}")
    output_root.mkdir(parents=True)

    map_path = output_root / "SOURCE_INPUT_MAP.csv"
    write_csv(
        map_path,
        rows,
        [
            "canonical_name",
            "canonical_path",
            "size_bytes",
            "sha256",
            "status",
            "selected_source_path",
            "match_count",
            "all_source_matches",
        ],
    )

    source_head_timestamp = run_git(
        source_repo, "show", "-s", "--format=%cI", actual_commit
    )
    manifest = {
        "schema_version": str(inventory_config["schema_version"]),
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "analysis_repository": {
            "path": str(repo),
            "commit": run_git(repo, "rev-parse", "HEAD"),
            "dirty": bool(run_git(repo, "status", "--porcelain")),
        },
        "measurement_repository": {
            "path": str(source_repo),
            "remote": source_remote(source_repo),
            "commit": actual_commit,
            "configured_commit": configured_commit,
            "commit_timestamp": source_head_timestamp,
            "clean": not bool(source_status),
            "subtree": subtree,
        },
        "canonical_reference": {
            "snapshot_id": snapshot_id,
            "inputs_directory": relative_posix(reference_inputs, repo),
            "input_count": len(canonical_files),
        },
        "scan": {
            "regular_files_seen": files_seen,
            "candidate_files_hashed": len(source_candidates),
            "candidate_bytes_hashed": bytes_considered,
            "selection_policy": str(inventory_config["selection_policy"]),
        },
        "result": {
            "matched_inputs": len(canonical_files) - len(unmatched),
            "unmatched_inputs": unmatched,
            "inputs_with_multiple_exact_matches": ambiguous,
            "selected_unique_source_paths": len(selected_paths),
            "complete_direct_provenance": not unmatched,
            "map_sha256": sha256(map_path),
        },
    }
    (output_root / "SOURCE_INVENTORY.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (output_root / "README.txt").write_text(
        "Source-input inventory\n"
        "======================\n\n"
        f"Measurement commit: {actual_commit}\n"
        f"Canonical inputs: {len(canonical_files)}\n"
        f"Exact matches: {len(canonical_files) - len(unmatched)}\n"
        f"Unmatched: {len(unmatched)}\n"
        f"Multiple-match inputs: {len(ambiguous)}\n\n"
        "This inventory does not copy or modify source data. It records direct\n"
        "byte-level provenance between frozen analysis inputs and files under\n"
        "the pinned measurement-repository subtree.\n",
        encoding="utf-8",
    )

    print(f"Source commit: {actual_commit}")
    print(f"Source repository clean: {not bool(source_status)}")
    print(f"Regular files scanned: {files_seen}")
    print(f"Candidate files hashed: {len(source_candidates)}")
    print(
        "Canonical inputs with exact source match: "
        f"{len(canonical_files) - len(unmatched)}/{len(canonical_files)}"
    )
    print(f"Inputs with multiple exact matches: {len(ambiguous)}")
    print(f"Inventory directory: {output_root}")

    if unmatched:
        print("Unmatched canonical inputs:", file=sys.stderr)
        for name in unmatched:
            print(f"  - {name}", file=sys.stderr)
        if not args.allow_unmatched:
            return 2

    print("PASS: source-input provenance inventory completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
