# Frozen reference results

The canonical reference outputs are stored inside the immutable snapshot:

`data/snapshots/t1-analysis-20260730/extracted/deep_research_energy/outputs/`

and the five canonical figures under:

`data/snapshots/t1-analysis-20260730/extracted/deep_research_energy/figures/`

`make analysis-check` regenerates all outputs in an isolated run directory and
requires byte-identical CSV and PNG files.
