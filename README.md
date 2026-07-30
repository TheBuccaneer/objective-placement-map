# Objective-Dependent CPU/GPU Placement

Reproducible analysis repository for the energy/runtime placement study.

## Repository rules

1. Files under `data/snapshots/` are immutable input snapshots.
2. Mutable or newly collected files belong in `data/raw/` and are not committed directly.
3. Every analysis run writes to a new directory under `results/runs/`.
4. Paper claims must reference a frozen run manifest and exact output file.
5. Sessions, not individual repetitions, are the primary statistical units.

## Initial setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py data/snapshots/t1-analysis-20260730
```

The initial snapshot is an archived, independently reproduced analysis state.
It is preserved unchanged; active analysis code will be migrated into `src/` in the next step.

## Reproducing the canonical analysis

```bash
source .venv/bin/activate
make verify
make test
make analysis-check
```

Every run is isolated under `results/runs/<run-id>/` and contains:

- `RUN_MANIFEST.json`,
- stdout and stderr logs,
- 37 regenerated CSV tables,
- five regenerated figures.

The check target requires byte-identical CSV tables. Figures must have the same names, be valid PNG files, and match the reference format, color mode, and dimensions. PNG byte hashes are recorded but are informational because renderer metadata and font rasterization can vary across environments.
Generated runs are intentionally excluded from Git; publish a selected run as a
separate release artifact and cite its manifest.


## Tracing frozen inputs to the measurement repository

The configured measurement commit can be checked against a local clean clone:

```bash
make source-inventory SOURCE_REPO=~/projects/energy
```

The command scans only the configured `new` subtree and maps each of the 28
frozen analysis inputs to exact source files using file size and SHA-256. It
writes a local `SOURCE_INVENTORY.json` and `SOURCE_INPUT_MAP.csv` under
`results/source-inventory/<run-id>/`. Unmatched inputs fail the command and must
later be reconstructed through explicit transformation code.
