# ADR 0005: Materialize the exact measurement-source inputs

## Context

The source inventory proves that all 28 frozen analysis inputs are byte-identical
to unique files below the pinned `energy/new` commit. Depending on a mutable
working clone alone would nevertheless make later reproduction fragile: paths
could move, branches could advance, and local files could be edited.

## Decision

A source snapshot is created under
`data/source-snapshots/energy-<commit12>/`. It contains only the 28 files that
feed the analysis, preserving their paths relative to the measurement repository
root. Every file is copied byte-for-byte, made read-only, and covered by
`MANIFEST.sha256`.

The snapshot also records:

- the complete measurement commit and origin,
- the source-inventory run and map hash,
- the analysis-repository commit used to create it,
- canonical input names and source paths,
- file sizes and SHA-256 values.

The snapshot is committed to Git because it is small, finite, and constitutes
the exact evidentiary input to the paper analysis. The larger measurement
repository remains the authoritative full campaign archive.

## Consequences

The paper analysis no longer depends on the current state of a separate clone.
Any later source import can be verified independently, while the full path back
to `energy/new` remains explicit.
