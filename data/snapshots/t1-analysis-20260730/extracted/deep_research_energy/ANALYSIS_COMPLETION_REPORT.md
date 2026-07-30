# Completion Report — T1 Energy–Runtime Placement Analysis

## Pipeline status

The rebuilt pipeline regenerates all 17 original CSV outputs, all five figures, and the additional robustness outputs from input snapshots.

## Confirmed core

- Canonical cells: **51**
- Canonical conflicts: **24/51**
- Fixed full-data selected winners with 5/5 session direction: **48/48 objective rows**
- Large-regime speedup: **2.121874x** (95% session-bootstrap CI 2.121340–2.123785)
- Large-regime RTX 5060 Ti energy saving: **42.4739%** (95% CI 42.2244–42.9774)

## Nested leave-one-session-out

- all_platform/edp: platform match 98.82%, within 2% 95.29%, within 5% 95.29%, median holdout regret 0.0000%.
- all_platform/energy: platform match 98.43%, within 2% 95.29%, within 5% 95.29%, median holdout regret 0.0000%.
- all_platform/runtime: platform match 98.04%, within 2% 95.29%, within 5% 95.69%, median holdout regret 0.0000%.
- gpu_only/edp: platform match 99.61%, within 2% 100.00%, within 5% 100.00%, median holdout regret 0.0000%.
- gpu_only/energy: platform match 99.61%, within 2% 99.61%, within 5% 99.61%, median holdout regret 0.0000%.
- gpu_only/runtime: platform match 100.00%, within 2% 100.00%, within 5% 100.00%, median holdout regret 0.0000%.
- cpu_threads/edp: platform match 100.00%, within 2% 81.96%, within 5% 86.86%, median holdout regret 0.0000%.
- cpu_threads/energy: platform match 100.00%, within 2% 88.04%, within 5% 91.96%, median holdout regret 0.0000%.
- cpu_threads/runtime: platform match 100.00%, within 2% 83.33%, within 5% 89.02%, median holdout regret 0.0000%.

Across all 24 canonical conflict cells, the four-session training folds selected different runtime and energy platforms in 5/5 folds for **24/24** cells. The held-out session itself retained the all-platform conflict in 5/5 folds for **22/24** cells and in at least 4/5 folds for **24/24** cells.
For the symmetric GPU-only core, all **23/23** canonical RTX 3090-vs-RTX 5060 Ti conflict cells retained the conflict and both objective-specific selections within 5% of the held-out GPU oracle in every fold.

The two all-platform boundary cells are REDUCTION 16M and STREAM 4M. Each contains one strongly deviating AMD session; the median-of-five canonical classification remains unchanged, but nested holdout reporting must distinguish 22/24 fully stable all-platform cells from the perfect 23/23 GPU-only core.

## Static GPU policy tails

- Always RTX 3090 energy CVaR10: **486.87%** (95% session-bootstrap CI 482.59–490.22%); within 5% of the GPU energy oracle in **29.41%** of cells.
- Always RTX 5060 Ti runtime CVaR10: **427.26%** (95% CI 426.64–427.76%); within 5% of the GPU runtime oracle in **15.69%** of cells.

## CPU-thread selection

- Maximum available threads have median energy regret **38.52%** and median EDP regret **63.06%** across the 102 CPU cells.
- Intel REDUCTION 256M: accepting **0.244%** runtime penalty saves **28.29%** energy; the 95% session-bootstrap intervals are [-0.014, 0.443]% runtime and [26.31, 35.59]% energy.
- Nested CPU-thread choices are less stable than GPU platform choices: 91.96% of energy folds, 89.02% of runtime folds, and 86.86% of EDP folds remain within 5% of the held-out thread oracle. High tail regrets are concentrated in high-session-variation AMD REDUCTION/STREAM configurations.

## Pareto sensitivity

- 0% tolerance: mean front 1.549/18, mean dominated fraction 91.39%, CPU on front in 6/51 cells.
- 2% tolerance: mean front 1.510/18, mean dominated fraction 91.61%, CPU on front in 6/51 cells.
- 5% tolerance: mean front 1.510/18, mean dominated fraction 91.61%, CPU on front in 6/51 cells.

## Intel DRAM sensitivity

- AXPY: thread optimum changed in 0/9 cells; median package-to-total increment at package optimum 2.647%.
- REDUCTION: thread optimum changed in 0/9 cells; median package-to-total increment at package optimum 1.308%.
- STREAM: thread optimum changed in 0/9 cells; median package-to-total increment at package optimum 2.320%.

## Remaining limitation

The supplied and public session snapshots do not contain the numeric configured/default RTX 3090 power-limit value for all workloads. Available AXPY throttle masks and clock/temperature summaries are reproduced, but the numeric stock-power-limit statement still requires metadata from the run environment or logs.
