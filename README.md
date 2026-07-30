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

The check target requires byte-identical outputs versus the frozen reference.
Generated runs are intentionally excluded from Git; publish a selected run as a
separate release artifact and cite its manifest.
