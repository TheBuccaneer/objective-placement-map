# ADR 0003: Validate numeric outputs exactly and figures structurally

## Context

A clean analysis run regenerated all 37 CSV tables byte-for-byte, while all five
PNG hashes differed from the frozen reference. PNG bytes can vary across
Matplotlib, Pillow, FreeType, font, and metadata environments even when the
underlying numerical results and plot construction are unchanged.

## Decision

`make analysis-check` applies two different policies:

1. Every canonical CSV table must be byte-identical to the frozen reference.
2. Every canonical figure must:
   - have the expected filename,
   - be a valid PNG,
   - match the reference image format, color mode, and pixel dimensions.

Figure SHA-256 values are still recorded in every run manifest for provenance,
but PNG byte identity is informational rather than a pass/fail criterion.

## Consequences

Scientific claims remain protected by exact comparison of all numerical tables.
The pipeline still fails for missing, corrupt, renamed, or structurally changed
figures, while avoiding false failures caused solely by renderer-level binary
variation.
