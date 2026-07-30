# ADR 0006: Rebuild canonical inputs from the pinned measurement snapshot

## Context

The active analysis was already reproducible from 28 frozen canonical input
CSVs. A separate provenance audit then established that all 28 inputs are exact,
unique byte-level matches to files under the pinned `energy/new` commit. The
remaining gap was that the normal analysis runner still copied its inputs from
the historical analysis package rather than rebuilding them from the pinned
measurement-source snapshot.

## Decision

The repository provides two new operations:

- `make build-inputs` materializes all 28 canonical inputs from
  `data/source-snapshots/energy-<commit12>/` and verifies that every resulting
  CSV is byte-identical to its frozen paper-reference input.
- `make reproduce` performs the full analysis with those newly materialized
  inputs and retains the complete input-materialization manifest inside the
  analysis run.

The source-relative path, canonical filename, size, SHA-256 hash, frozen
reference hash, measurement commit, and source-snapshot identity are recorded
for every input.

## Consequences

The end-to-end chain no longer depends on the historical analysis package for
its active inputs. The historical package remains only the immutable numeric
reference used to detect regressions. A successful `make reproduce` therefore
proves both input provenance and exact regeneration of all 37 numerical result
tables.
