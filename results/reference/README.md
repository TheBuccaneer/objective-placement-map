# Frozen reference results

The canonical reference outputs are stored inside the immutable snapshot:

`data/snapshots/t1-analysis-20260730/extracted/deep_research_energy/outputs/`

and the five canonical figures under:

`data/snapshots/t1-analysis-20260730/extracted/deep_research_energy/figures/`

`make analysis-check` regenerates all outputs in an isolated run directory and
requires byte-identical CSV files. Figures must have the same names, be valid PNG files, and match the reference format, mode, and dimensions. Their byte hashes remain recorded as provenance but are not a scientific pass/fail criterion.
