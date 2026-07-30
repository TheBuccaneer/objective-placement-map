# ADR 0001: Separate immutable inputs, active code, and generated outputs

## Decision

- Preserve imported analysis packages unchanged under `data/snapshots/`.
- Develop active analysis code under `src/placement_analysis/`.
- Write each execution to a timestamped directory under `results/runs/`.
- Store file hashes and provenance metadata beside every snapshot and run.

## Reason

This prevents generated results, source data, and historical analysis versions from being mixed.
