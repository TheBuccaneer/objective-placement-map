# Objective-Dependent CPU/GPU Placement Map

Reproducible analysis repository for a four-platform, six-workload
energy-runtime measurement study.

The repository evaluates when runtime-, energy-, and EDP-optimal placement
decisions diverge across:

- Intel Core i9-7900X
- AMD Threadripper 3970X
- NVIDIA RTX 3090
- NVIDIA RTX 5060 Ti

and six workload families:

- GEMM
- STRIDED_GEMM
- AXPY
- STREAM
- REDUCTION
- CONV2D

## Reproducibility status

The complete analysis path is closed and tested:

```text
energy/new@e54128a613e7d6adc46150c020b26b6f98a4c0a2
        ↓
28 pinned source files
        ↓
28 rebuilt canonical analysis inputs
        ↓
37 generated CSV result tables
        ↓
5 generated figures
        ↓
reference and provenance validation
```

A successful reproduction requires:

- **28/28** source-derived inputs to match the frozen reference;
- **37/37** generated CSVs to be byte-identical;
- all five generated figures to satisfy the reference figure contract;
- the complete nested session validation to pass.

## Quick reproduction

```bash
git clone git@github.com:TheBuccaneer/objective-placement-map.git
cd objective-placement-map

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

make verify
make verify-source-snapshots
make reproduce
```

Expected final output:

```text
PASS: analysis run ...
Generated: 37 CSV tables and 5 figures
Reference comparison: CSV tables byte-identical; figure contracts matched
Analysis inputs: rebuilt from pinned source snapshot; 28/28 exact reference matches
```

See [REPRODUCING.md](REPRODUCING.md) for the complete procedure.

## Repository structure

```text
.
├── config/                 # pinned source and analysis policy
├── data/
│   ├── source-snapshots/   # immutable upstream-derived inputs
│   └── snapshots/          # frozen reference analysis bundle
├── docs/
│   ├── decisions/          # architecture decision records
│   └── DATA_DICTIONARY.md
├── results/
│   ├── input-builds/       # locally rebuilt canonical inputs
│   ├── runs/               # isolated analysis runs
│   └── source-inventory/   # byte-level provenance inventories
├── scripts/                # reproducibility and provenance entry points
├── src/                    # active Python package
├── tests/                  # repository and pipeline contracts
├── REPRODUCING.md
└── Makefile
```

## Primary commands

```bash
make verify
make test
make docs-check
make build-inputs
make reproduce
make source-inventory SOURCE_REPO=~/projects/energy
```

## Analysis scope

The analysis covers **918 measured configurations**, five independent sessions,
and **51 canonical workload-size/shape cells**. Sessions are the primary
statistical units.

GPU results use the documented `gpu_resident` scope. Allocations,
initialization, and host/device transfers are outside the measured interval.
GPU energy is NVML device/board energy. CPU energy is based on documented RAPL
domains. Cross-domain claims must retain these boundaries.

## Documentation

- [Complete reproduction guide](REPRODUCING.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Licensing scope](docs/LICENSING.md)
- [Repository decisions](docs/decisions/)

## License

Original analysis code and repository documentation are licensed under the
[MIT License](LICENSE). Bundled measurement snapshots retain the terms and
attribution requirements of their upstream source. See
[docs/LICENSING.md](docs/LICENSING.md).
