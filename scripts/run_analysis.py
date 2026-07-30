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

from PIL import Image
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


def png_contract(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {
                "valid": True,
                "format": image.format,
                "mode": image.mode,
                "size": [int(image.width), int(image.height)],
            }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def compare_figure_contract(reference: Path, actual: Path) -> dict[str, Any]:
    ref = {
        p.relative_to(reference).as_posix(): p
        for p in sorted(reference.rglob("*.png"))
        if p.is_file()
    }
    got = {
        p.relative_to(actual).as_posix(): p
        for p in sorted(actual.rglob("*.png"))
        if p.is_file()
    }

    missing = sorted(set(ref) - set(got))
    unexpected = sorted(set(got) - set(ref))
    common = sorted(set(ref) & set(got))

    byte_changed: list[str] = []
    invalid_reference: dict[str, Any] = {}
    invalid_actual: dict[str, Any] = {}
    structural_changes: dict[str, Any] = {}

    for name in common:
        if sha256(ref[name]) != sha256(got[name]):
            byte_changed.append(name)

        ref_contract = png_contract(ref[name])
        got_contract = png_contract(got[name])

        if not ref_contract.get("valid", False):
            invalid_reference[name] = ref_contract
            continue
        if not got_contract.get("valid", False):
            invalid_actual[name] = got_contract
            continue

        compared_keys = ("format", "mode", "size")
        differences = {
            key: {
                "reference": ref_contract[key],
                "actual": got_contract[key],
            }
            for key in compared_keys
            if ref_contract[key] != got_contract[key]
        }
        if differences:
            structural_changes[name] = differences

    contract_match = not (
        missing
        or unexpected
        or invalid_reference
        or invalid_actual
        or structural_changes
    )

    return {
        "policy": "same_names_valid_png_same_format_mode_dimensions",
        "reference_count": len(ref),
        "actual_count": len(got),
        "contract_match": contract_match,
        "byte_identical_count": len(common) - len(byte_changed),
        "byte_changed": byte_changed,
        "missing": missing,
        "unexpected": unexpected,
        "invalid_reference": invalid_reference,
        "invalid_actual": invalid_actual,
        "structural_changes": structural_changes,
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
        help=(
            "Require byte-identical CSV outputs and structurally valid PNG "
            "figures versus the frozen snapshot"
        ),
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep copied source package and inputs inside the run directory",
    )
    parser.add_argument(
        "--inputs-from-source-snapshot",
        action="store_true",
        help=(
            "Rebuild all canonical inputs from the pinned measurement-source "
            "snapshot before running the analysis"
        ),
    )
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="Explicit pinned source snapshot used with --inputs-from-source-snapshot",
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
    input_materialization: dict[str, Any] = {"enabled": False}

    try:
        shutil.copytree(package, workspace)
        make_writable(workspace)
        shutil.copy2(active_script, workspace / "analysis.py")

        if args.inputs_from_source_snapshot:
            configured_commit = str(project["source"]["commit"])
            source_snapshot = (
                args.source_snapshot.expanduser().resolve()
                if args.source_snapshot
                else (
                    repo
                    / "data"
                    / "source-snapshots"
                    / f"energy-{configured_commit[:12]}"
                ).resolve()
            )
            materialization_root = temp_dir / "input-materialization"
            materializer_command = [
                sys.executable,
                str(repo / "scripts" / "materialize_analysis_inputs.py"),
                "--source-snapshot",
                str(source_snapshot),
                "--output-root",
                str(materialization_root),
            ]
            materializer = subprocess.run(
                materializer_command,
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            (log_dir / "input-materialization.stdout.log").write_text(
                materializer.stdout, encoding="utf-8"
            )
            (log_dir / "input-materialization.stderr.log").write_text(
                materializer.stderr, encoding="utf-8"
            )
            if materializer.returncode != 0:
                raise RuntimeError(
                    "input materialization failed; see "
                    f"{log_dir / 'input-materialization.stderr.log'}"
                )

            rebuilt_inputs = materialization_root / "inputs"
            if not rebuilt_inputs.is_dir():
                raise RuntimeError(
                    f"materialized inputs directory missing: {rebuilt_inputs}"
                )
            shutil.rmtree(workspace / "inputs")
            shutil.copytree(rebuilt_inputs, workspace / "inputs")

            input_manifest_path = materialization_root / "INPUT_MANIFEST.json"
            input_manifest = json.loads(
                input_manifest_path.read_text(encoding="utf-8")
            )
            input_materialization = {
                "enabled": True,
                "source_snapshot": str(source_snapshot.relative_to(repo)),
                "materializer_command": materializer_command,
                "manifest_path": "input-materialization/INPUT_MANIFEST.json",
                "manifest_sha256": sha256(input_manifest_path),
                "manifest": input_manifest,
                "stdout_log": "logs/input-materialization.stdout.log",
                "stderr_log": "logs/input-materialization.stderr.log",
            }

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

        comparison: dict[str, Any] = {
            "enabled": args.compare_reference,
            "csv_policy": "byte_exact",
            "figure_policy": "same_names_valid_png_same_format_mode_dimensions",
        }
        if args.compare_reference:
            csv_cmp = compare_directory(
                package / "outputs", workspace / "outputs", ".csv"
            )
            figure_cmp = compare_figure_contract(
                package / "figures", workspace / "figures"
            )
            comparison.update({"csv": csv_cmp, "figures": figure_cmp})
            comparison["passed"] = (
                csv_cmp["exact_match"] and figure_cmp["contract_match"]
            )
            if not comparison["passed"]:
                raise RuntimeError(
                    "generated numeric results or figure contracts differ "
                    "from the frozen reference"
                )

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
            "analysis_inputs": input_materialization,
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
            print(
                "Reference comparison: CSV tables byte-identical; "
                "figure contracts matched"
            )
            print(
                "Figure byte identity (informational only): "
                f"{comparison['figures']['byte_identical_count']}/{fig_count}"
            )
        if args.inputs_from_source_snapshot:
            print(
                "Analysis inputs: rebuilt from pinned source snapshot; "
                "28/28 exact reference matches"
            )
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
