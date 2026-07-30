# ADR 0002: Execute active analysis in isolated run directories

## Decision

The active script is versioned at `src/placement_analysis/analysis.py`.
Every execution copies the frozen input package into a temporary workspace,
replaces only `analysis.py` with the active version, and writes the generated
artifacts to `results/runs/<run-id>/`.

Each successful run records:

- repository and source-measurement commits,
- snapshot and active-script hashes,
- Python and package versions,
- deterministic environment variables,
- hashes of all 37 CSV tables and five figures,
- byte-exact CSV and structural figure-validation status against the frozen reference.

## Rationale

The immutable source snapshot, active analysis code, and generated outputs must
never be mixed. Isolated runs make every paper claim traceable to one manifest.
