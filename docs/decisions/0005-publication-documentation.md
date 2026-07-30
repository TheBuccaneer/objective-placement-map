# ADR 0005: Freeze the analysis and document the reproducible interface

## Status

Accepted.

## Context

The active pipeline already reproduces all 37 numerical result tables exactly
from 28 inputs materialized from a pinned measurement snapshot. Refactoring the
large analysis script would introduce unnecessary risk before paper writing.

## Decision

The current analysis implementation is frozen. Publication preparation focuses
on:

- one complete reproduction guide;
- an explicit input/output data dictionary;
- a clear licensing boundary;
- a concise repository README;
- automated documentation checks.

`CITATION.cff`, archival release tags, and Zenodo integration are deferred.

## Consequences

The repository is publication-readable without changing the validated
numerical pipeline. Future modularization may occur after the paper snapshot,
provided all frozen-output contracts continue to pass.
