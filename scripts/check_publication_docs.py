#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "data/snapshots/t1-analysis-20260730"
    / "extracted/deep_research_energy"
)

REQUIRED = [
    ROOT / "README.md",
    ROOT / "REPRODUCING.md",
    ROOT / "LICENSE",
    ROOT / "docs/DATA_DICTIONARY.md",
    ROOT / "docs/LICENSING.md",
    ROOT / "docs/decisions/0005-publication-documentation.md",
]


def fail(message: str) -> None:
    raise SystemExit(f"Documentation check failed: {message}")


def main() -> int:
    for path in REQUIRED:
        if not path.is_file():
            fail(f"missing {path.relative_to(ROOT)}")
        if path.stat().st_size == 0:
            fail(f"empty {path.relative_to(ROOT)}")

    project_yaml = (ROOT / "config/project.yaml").read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*commit:\s*([0-9a-f]{40})\s*$", project_yaml)
    if not match:
        fail("no pinned 40-character source commit in config/project.yaml")

    source_commit = match.group(1)
    reproduce = (ROOT / "REPRODUCING.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dictionary = (ROOT / "docs/DATA_DICTIONARY.md").read_text(encoding="utf-8")

    for document, name in [
        (reproduce, "REPRODUCING.md"),
        (readme, "README.md"),
        (dictionary, "docs/DATA_DICTIONARY.md"),
    ]:
        if source_commit not in document:
            fail(f"{name} does not contain the pinned source commit")

    for token in ["make reproduce", "37 CSV", "5 figure", "28"]:
        if token.lower() not in reproduce.lower():
            fail(f"REPRODUCING.md does not mention {token!r}")

    inputs = sorted((REFERENCE / "inputs").glob("*.csv"))
    outputs = sorted((REFERENCE / "outputs").glob("*.csv"))
    figures = sorted((REFERENCE / "figures").glob("*.png"))

    if len(inputs) != 28:
        fail(f"expected 28 canonical inputs, found {len(inputs)}")
    if len(outputs) != 37:
        fail(f"expected 37 reference outputs, found {len(outputs)}")
    if len(figures) != 5:
        fail(f"expected five reference figures, found {len(figures)}")

    for path in inputs + outputs + figures:
        if f"`{path.name}`" not in dictionary:
            fail(f"data dictionary does not list {path.name}")

    placeholders = [
        "REPLACE_WITH",
        "TODO",
        "TBD",
        "<INSERT",
    ]
    for path in REQUIRED:
        text = path.read_text(encoding="utf-8")
        for placeholder in placeholders:
            if placeholder in text:
                fail(
                    f"{path.relative_to(ROOT)} contains placeholder "
                    f"{placeholder!r}"
                )

    print("Publication documentation check OK")
    print(f"Pinned source commit: {source_commit}")
    print(f"Canonical inputs documented: {len(inputs)}")
    print(f"Generated CSV outputs documented: {len(outputs)}")
    print(f"Generated figures documented: {len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
