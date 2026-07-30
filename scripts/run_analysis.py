#!/usr/bin/env python3
"""Run the active analysis in an isolated, provenance-recorded directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(command: list[str], cwd: Path, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def git_info(repo: Path) -> dict[str, Any]:
    commit = run_text(["git", "rev-parse", "HEAD"], repo)
    short = run_text(["git", "rev-parse", "--short=12", "HEAD"], repo)
    status = run_text(["git", "status", "--porcelain"], repo)
    try:
        remote = run_text(["git", "remote", "get-url", "origin"], repo)
    except RuntimeError:
        remote = None
    return {
        "commit": commit,
        "short_commit": short,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "origin": remote,
    }


def make_writable(root: Path) -> None:
    for path in [root, *sorted(root.rglob("*"))]:
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
        else:
            path.chmod(mode | stat.S_IWUSR)


def file_map(root: Path, pattern: str) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.glob(pattern))
        if path.is_file()
    }


def compare_directory(reference: Path, actual: Path, suffix: str) -> dict[str, Any]:
    ref = {
        p.relative_to(reference).as_posix(): sha256(p)
        for p in sorted(reference.rglob(f"*{suffix}"))
        if p.is_file()
    }
    got = {
        p.relative_to(actual).as_posix(): sha256(p)
        for p in sorted(actual.rglob(f"*{suffix}"))
        if p.is_file()
    }
    missing = sorted(set(ref) - set(got))
    unexpected = sorted(set(got) - set(ref))
    changed = sorted(name for name in set(ref) & set(got) if ref[name] != got[name])
    return {
        "reference_count": len(ref),
        "actual_count": len(got),
        "exact_match": not missing and not unexpected and not changed,
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="Explicit run identifier")
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Require byte-identical CSV and PNG outputs versus the frozen snapshot",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep copied source package and inputs inside the run directory",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    project = load_yaml(repo / "config/project.yaml")
    config = load_yaml(repo / "config/analysis.yaml")
    analysis_cfg = config["analysis"]
    execution_cfg = config["execution"]

    snapshot_id = str(analysis_cfg["snapshot_id"])
    snapshot_root = repo / "data/snapshots" / snapshot_id
    package = snapshot_root / str(analysis_cfg["snapshot_package_subdir"])
    active_script = repo / str(analysis_cfg["active_script"])
    if not package.is_dir():
        raise FileNotFoundError(f"snapshot package missing: {package}")
    if not active_script.is_file():
        raise FileNotFoundError(f"active analysis missing: {active_script}")

    git = git_info(repo)
    default_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + git["short_commit"]
    run_id = args.run_id or default_run_id
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in run_id):
        raise ValueError("run id contains unsupported characters")

    runs_root = repo / "results/runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final_dir = runs_root / run_id
    if final_dir.exists():
        raise FileExistsError(f"run directory already exists: {final_dir}")

    started = utc_now()
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs_root))
    workspace = temp_dir / "workspace"
    log_dir = temp_dir / "logs"
    log_dir.mkdir(parents=True)

    try:
        shutil.copytree(package, workspace)
        make_writable(workspace)
        shutil.copy2(active_script, workspace / "analysis.py")
        (workspace / "outputs").mkdir(exist_ok=True)
        (workspace / "figures").mkdir(exist_ok=True)
        for path in (workspace / "outputs").glob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        for path in (workspace / "figures").glob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

        env = os.environ.copy()
        env.update(
            {
                "PYTHONHASHSEED": str(execution_cfg["python_hash_seed"]),
                "MPLBACKEND": str(execution_cfg["matplotlib_backend"]),
                "TZ": str(execution_cfg["timezone"]),
                "LC_ALL": str(execution_cfg["locale"]),
                "LANG": str(execution_cfg["locale"]),
            }
        )
        command = [sys.executable, "analysis.py"]
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (log_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (log_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"analysis failed with exit code {result.returncode}; see {log_dir}"
            )

        csv_count = len(list((workspace / "outputs").glob("*.csv")))
        fig_count = len(list((workspace / "figures").glob("*.png")))
        expected_csv = int(analysis_cfg["expected_csv_tables"])
        expected_fig = int(analysis_cfg["expected_figures"])
        if csv_count != expected_csv:
            raise RuntimeError(f"expected {expected_csv} CSVs, generated {csv_count}")
        if fig_count != expected_fig:
            raise RuntimeError(f"expected {expected_fig} figures, generated {fig_count}")

        comparison: dict[str, Any] = {"enabled": args.compare_reference}
        if args.compare_reference:
            csv_cmp = compare_directory(package / "outputs", workspace / "outputs", ".csv")
            png_cmp = compare_directory(package / "figures", workspace / "figures", ".png")
            comparison.update({"csv": csv_cmp, "figures": png_cmp})
            comparison["exact_match"] = csv_cmp["exact_match"] and png_cmp["exact_match"]
            if not comparison["exact_match"]:
                raise RuntimeError("generated results differ from frozen reference")

        generated_dir = temp_dir / "generated"
        generated_dir.mkdir()
        shutil.move(str(workspace / "outputs"), generated_dir / "outputs")
        shutil.move(str(workspace / "figures"), generated_dir / "figures")
        for filename in [
            "ANALYSIS_COMPLETION_REPORT.md",
            "PAPER_READY_ANALYSIS_SUMMARY_DE.md",
            "SHA256SUMS_COMPLETE.txt",
        ]:
            source = workspace / filename
            if source.exists():
                shutil.copy2(source, generated_dir / filename)

        snapshot_metadata: dict[str, Any] = {}
        metadata_path = snapshot_root / "SNAPSHOT.json"
        if metadata_path.is_file():
            snapshot_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        finished = utc_now()
        manifest = {
            "schema_version": str(analysis_cfg["run_schema_version"]),
            "run_id": run_id,
            "status": "PASS",
            "started_at_utc": started,
            "finished_at_utc": finished,
            "command": command,
            "repository": git,
            "measurement_source": project.get("source", {}),
            "snapshot": {
                "id": snapshot_id,
                "snapshot_json_sha256": sha256(metadata_path) if metadata_path.is_file() else None,
                "source_archive_sha256": sha256(snapshot_root / "SOURCE_ARCHIVE.zip"),
                "metadata": snapshot_metadata,
            },
            "analysis": {
                "active_script": str(active_script.relative_to(repo)),
                "active_script_sha256": sha256(active_script),
                "stdout_log": "logs/stdout.log",
                "stderr_log": "logs/stderr.log",
            },
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "packages": package_versions(
                    ["numpy", "pandas", "matplotlib", "scipy", "PyYAML"]
                ),
                "variables": {
                    "PYTHONHASHSEED": env["PYTHONHASHSEED"],
                    "MPLBACKEND": env["MPLBACKEND"],
                    "TZ": env["TZ"],
                    "LC_ALL": env["LC_ALL"],
                },
            },
            "generated": {
                "csv_count": csv_count,
                "figure_count": fig_count,
                "csv_sha256": file_map(generated_dir / "outputs", "*.csv"),
                "figure_sha256": file_map(generated_dir / "figures", "*.png"),
            },
            "reference_comparison": comparison,
        }
        (temp_dir / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if args.keep_workspace:
            shutil.move(str(workspace), temp_dir / "source_workspace")
        else:
            shutil.rmtree(workspace)

        temp_dir.rename(final_dir)
        print(f"PASS: analysis run {run_id}")
        print(f"Run directory: {final_dir}")
        print(f"Generated: {csv_count} CSV tables and {fig_count} figures")
        if args.compare_reference:
            print("Reference comparison: byte-identical")
        return 0
    except Exception:
        failed = runs_root / f"{run_id}-FAILED"
        if failed.exists():
            shutil.rmtree(failed)
        temp_dir.rename(failed)
        print(f"FAILED run retained at: {failed}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
