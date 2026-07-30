# Input provenance

**Snapshot completion date:** 2026-07-30  
**Repository:** `https://github.com/TheBuccaneer/energy`  
**Relevant subtree:** `new`

The package is self-contained: all computations use the CSV files stored under `inputs/`. SHA-256 hashes are recorded in `SHA256SUMS_COMPLETE.txt`.

## Inputs retained from the original deep-research package

These files contain workload-level audit, envelope, placement, pairwise, or configuration summaries:

- `axpy_config_summary.csv`
- `axpy_cross.csv`
- `axpy_envelopes.csv`
- `conv_config_summary.csv`
- `conv_envelopes.csv`
- `conv_leaders.csv`
- `conv_sessions.csv`
- `gemm_config_summary.csv`
- `gemm_envelopes.csv`
- `gemm_placement.csv`
- `gemm_sessions.csv`
- `reduction_config_summary.csv`
- `reduction_pairwise.csv`
- `reduction_placement2.csv`
- `reduction_sessions.csv`
- `stream_config_summary.csv`
- `stream_pairwise.csv`
- `stream_placement.csv`
- `stream_sessions.csv`
- `strided_config_summary.csv`
- `strided_envelopes.csv`
- `strided_placement.csv`
- `strided_sessions.csv`

The shorter `*_sessions.csv` files in this list preserve the original package provenance and selected-configuration support tables. The complete nested analysis uses the full session snapshots listed below.

## Full all-configuration session snapshots added for nested validation

| Package file | Repository source path |
|---|---|
| `axpy_sessions.csv` | `new/analyse/AXPY/all_platforms/axpy_session_summary.csv` |
| `gemm_all_sessions.csv` | `new/ALL AUDIT/GEMM/analyse/unified_session_medians.csv` |
| `strided_all_sessions.csv` | `new/ALL AUDIT/STRIDED_GEMM/analyse/unified_session_medians.csv` |
| `stream_all_sessions.csv` | `new/ALL AUDIT/STREAM/analyse/unified_session_medians.csv` |
| `reduction_all_sessions.csv` | `new/ALL AUDIT/REDUCTION/analyse/unified_session_medians.csv` |

`conv_sessions.csv` already contains the complete five-session-by-configuration Conv2D snapshot required by the combined analysis.

## Normalization performed by `analysis.py`

The script harmonizes only schema-level names and representation:

- platform labels (`amd` → `AMD`, `intel` → `INTEL`);
- CPU configuration labels from thread counts (`8` → `8T`);
- GPU thread sentinel (`-1`) to the configuration label `gpu_resident`;
- workload-specific column names to common fields for runtime, energy, EDP, power, temperature, clock, session, and thread count.

It does not impute missing measurements, alter session values, or replace workload-specific canonical placement classifications.

## Statistical unit

Every full-session snapshot contains exactly five session medians for every measured configuration. The completed combined table contains:

- 918 configurations;
- 5 session medians per configuration;
- 4590 session rows total.
