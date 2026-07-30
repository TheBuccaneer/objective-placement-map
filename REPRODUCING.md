# Reproducing the objective-placement analysis

This document describes the complete path from the pinned measurement
snapshot to the generated paper tables and figures.

## Reproduced state

- measurement repository: `https://github.com/TheBuccaneer/energy`
- measurement subtree: `new`
- pinned measurement commit: `e54128a613e7d6adc46150c020b26b6f98a4c0a2`
- pinned source snapshot: `data/source-snapshots/energy-e54128a613e7/`
- canonical analysis inputs: **28 CSV files**
- generated result tables: **37 CSV files**
- generated figures: **5 PNG files**

The primary statistical unit is the independent session. The pipeline does
not treat the ten repetitions within a session as independent experiments.

## Requirements

- Linux
- Git
- Python 3.11 or newer
- GNU Make

The tested local environment used Python 3.13.7. Exact installed Python
packages are recorded in `requirements-lock.txt`.

## Clone and environment setup

```bash
git clone git@github.com:TheBuccaneer/objective-placement-map.git
cd objective-placement-map

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

## Verify immutable snapshots

```bash
make verify
make verify-source-snapshots
```

Expected result:

```text
Snapshot verification OK
Source snapshot verification OK: 28/28 files
```

## Rebuild the 28 canonical inputs

```bash
make build-inputs
make verify-input-builds
```

Expected result:

```text
Materialized canonical inputs: 28
Exact frozen-reference matches: 28/28
```

## Run the complete reproduction

```bash
make reproduce
```

The command performs the following chain:

1. verifies the pinned source snapshot;
2. materializes the 28 canonical analysis inputs;
3. runs the complete analysis;
4. creates 37 CSV result tables;
5. creates five PNG figures;
6. compares all CSVs byte-for-byte with the frozen reference;
7. validates the figure contract;
8. writes a run manifest with Git, Python, dependency, input, and output
   provenance.

Expected result:

```text
PASS: analysis run ...
Generated: 37 CSV tables and 5 figures
Reference comparison: CSV tables byte-identical; figure contracts matched
Analysis inputs: rebuilt from pinned source snapshot; 28/28 exact reference matches
```

## Inspect the latest run

```bash
RUN="$(
  find results/runs     -mindepth 1     -maxdepth 1     -type d     ! -name '*-FAILED'     | sort     | tail -n1
)"

echo "$RUN"
cat "$RUN/RUN_MANIFEST.json"
```

Each successful run contains:

```text
results/runs/<run-id>/
├── RUN_MANIFEST.json
├── generated/
│   ├── outputs/
│   └── figures/
└── logs/
    ├── stdout.log
    └── stderr.log
```

## Run tests

```bash
make test
make docs-check
```

## Recreate the source inventory from a local measurement checkout

This step is optional for ordinary reproduction because the pinned source
snapshot is already included. It verifies the direct byte-level relationship
to a clean checkout of the measurement repository.

```bash
make source-inventory SOURCE_REPO=~/projects/energy
```

The local measurement checkout must be clean and must resolve to:

```text
e54128a613e7d6adc46150c020b26b6f98a4c0a2
```

Expected result:

```text
Canonical inputs with exact source match: 28/28
Inputs with multiple exact matches: 0
PASS: source-input provenance inventory completed
```

## Interpretation of figure differences

CSV result tables are required to be byte-identical to the frozen reference.
PNG files are validated structurally rather than by binary hash because
Matplotlib, Pillow, FreeType, fonts, and embedded metadata can change PNG bytes
without changing the numerical result. The run manifest records both the
figure contract and informational PNG hashes.

## Failure handling

Failed runs are retained under:

```text
results/runs/<run-id>-FAILED/
```

Inspect their logs before deletion. After diagnosis:

```bash
make clean-failed
```

## Scientific scope

GPU measurements use the documented `gpu_resident` scope: allocations,
initialization, and host/device transfers are outside the measured time and
energy interval. GPU energy is device/board energy reported by NVML. CPU and
GPU energy domains are therefore documented separately; the symmetric
GPU-only comparison is the strongest cross-device energy basis.
