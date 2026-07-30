# Deep Research Package: CPU/GPU Energy–Runtime Placement

This package reproduces the T1 cross-workload placement analysis from frozen CSV snapshots of the current `energy/new` audit outputs. It contains 918 configurations, five session medians per configuration, 51 workload–size/shape cells, six workloads, four platforms, and all CPU thread sweeps.

## Run from a clean state

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MPLCONFIGDIR=/tmp/mplconfig python analysis.py
sha256sum -c SHA256SUMS_COMPLETE.txt
```

`analysis.py` deletes previously generated CSV and PNG outputs before rebuilding them. A successful run prints:

```text
PASS: regenerated 36 CSV tables and five figures
PASS: full 918-configuration x five-session nested validation completed
```

No private absolute path is required. The snapshots under `inputs/` are sufficient for the included analysis.

## What is reproduced

The pipeline regenerates all 17 filenames from the original deep-research package and adds publication-oriented robustness analyses:

- canonical 51-cell placement map;
- all 918 configurations and platform envelopes;
- 24 canonical runtime/energy conflicts;
- 15-cell large AXPY/STREAM/REDUCTION GPU regime;
- static all-platform and GPU-only policy regret;
- EDP rank alignment;
- strict and tolerance-aware Pareto analysis;
- CPU-thread trade-offs and near-free energy savings;
- fixed-selection session direction support;
- nested leave-one-session-out for all-platform, GPU-only, and CPU-thread selection;
- nested validation of all 24 canonical conflicts and the 23 symmetric GPU-only conflicts;
- session-bootstrap intervals for the headline GPU regime and policy tails;
- Intel package versus package+DRAM sensitivity where available;
- available RTX 3090 clock, temperature, power, and throttle metadata;
- a frozen paper-claim table.

## Main files

- `REPORT.md` — original adversarial research report.
- `ANALYSIS_COMPLETION_REPORT.md` — completion and robustness report.
- `analysis.py` — complete executable pipeline.
- `analysis_legacy.py` — original incomplete script retained for provenance.
- `outputs/claim_freeze_table.csv` — paper-ready claim definitions and robustness status.
- `outputs/canonical_51_cells.csv` — canonical placement map.
- `outputs/all_918_configurations.csv` — unified measured configuration table.
- `outputs/nested_loso_summary.csv` — out-of-session selection summary.
- `outputs/nested_gpu_conflict_loso_by_cell.csv` — nested validation of the symmetric GPU conflict core.
- `outputs/session_instability_diagnostics.csv` — high-session-variation configurations and boundary cases.
- `outputs/bootstrap_headline_intervals.csv` — session-bootstrap intervals for the 15-cell regime.
- `outputs/bootstrap_gpu_policy_intervals.csv` — session-bootstrap intervals for static GPU policy regret.
- `outputs/practical_pareto_summary.csv` — dominance sensitivity.
- `figures/` — five regenerated figures.

## Statistical scope

- The five session medians are the independent measurement units.
- The ten measurements within each session are technical repetitions, not an independent sample of 50.
- Counts such as `24/51` and `15/15` describe the complete measured suite; confidence intervals apply to measurement-derived ratios and selected values, not to a fictitious population of workload cells.
- Nested leave-one-session-out selects the platform/configuration using four sessions and evaluates that fixed choice on the fifth.
- The canonical 24-cell conflict map is defined by the workload-specific audit rules. The non-circular conflict-severity result is the tolerance sensitivity: 22/24 conflicts remain without a joint 10%-near-optimum and 19/24 remain without a joint 20%-near-optimum.

## Important interpretation boundaries

- The strongest energy claims are GPU-only because both GPUs use the same NVML board-energy domain.
- CPU/GPU energy comparisons refer to the documented RAPL and NVML domains; they are not full-system energy claims.
- GPU measurements are `gpu_resident`; allocation, initialization, and PCIe transfer are outside the measured window.
- The numeric configured/default RTX 3090 power limit is not present in the supplied snapshots. Available clock, temperature, power, and AXPY throttle-mask data are reproduced, but a numeric stock-power-limit statement still requires the original run metadata or logs.
- `logical_bytes_per_op` and derived nominal intensity are semantic proxies, not measured physical traffic.
