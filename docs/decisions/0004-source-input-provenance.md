# ADR 0004: Discover source provenance by exact content identity

## Context

The frozen analysis package contains 28 canonical CSV inputs. Historical audit
pipelines used workload-specific filenames and directories, so relying only on
basenames would be fragile and could silently select an obsolete copy.

## Decision

The source-inventory command requires the measurement repository to:

- be at the exact commit recorded in `config/project.yaml`,
- have a clean working tree,
- contain the configured `new` subtree.

Every canonical input is matched to files below `new` by exact file size and
SHA-256. All matches are recorded. When multiple identical copies exist, the
selected path is deterministic: matching basename first, then the shortest
relative path, then lexical order.

This step only inventories provenance. It does not copy, normalize, or modify
measurement data.

## Consequences

Direct repository outputs are distinguished from analysis-derived inputs. Any
unmatched canonical input causes the command to fail unless the diagnostic
`--allow-unmatched` option is supplied. Derived inputs must later receive an
explicit transformation script rather than an invented source path.
