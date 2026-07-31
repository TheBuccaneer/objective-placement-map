SHELL := /bin/bash
PYTHON := .venv/bin/python
SNAPSHOT := data/snapshots/t1-analysis-20260730
RUN_ARGS ?=
SOURCE_REPO ?= $(HOME)/projects/energy
SOURCE_INVENTORY_ARGS ?=
SOURCE_SNAPSHOT_ARGS ?=
INPUT_BUILD_ARGS ?=

.PHONY: setup verify test analysis analysis-check list-runs clean-failed source-inventory list-source-inventories source-snapshot verify-source-snapshots list-source-snapshots build-inputs verify-input-builds list-input-builds reproduce

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

verify:
	$(PYTHON) scripts/verify_snapshot.py $(SNAPSHOT)

test:
	$(PYTHON) -m pytest

analysis:
	$(PYTHON) scripts/run_analysis.py $(RUN_ARGS)

analysis-check:
	$(PYTHON) scripts/run_analysis.py --compare-reference $(RUN_ARGS)

list-runs:
	@find results/runs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort

clean-failed:
	@find results/runs -mindepth 1 -maxdepth 1 -type d -name '*-FAILED' -print -exec rm -rf {} +


source-inventory:
	$(PYTHON) scripts/inventory_source_inputs.py --source-repo "$(SOURCE_REPO)" $(SOURCE_INVENTORY_ARGS)

list-source-inventories:
	@find results/source-inventory -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort


source-snapshot:
	$(PYTHON) scripts/create_source_snapshot.py --source-repo "$(SOURCE_REPO)" $(SOURCE_SNAPSHOT_ARGS)

verify-source-snapshots:
	@set -e; found=0; \
	for snapshot in data/source-snapshots/energy-*; do \
	  if [ -d "$$snapshot" ]; then \
	    found=1; \
	    $(PYTHON) scripts/verify_source_snapshot.py "$$snapshot"; \
	  fi; \
	done; \
	if [ "$$found" -eq 0 ]; then echo "No source snapshots found" >&2; exit 1; fi

list-source-snapshots:
	@find data/source-snapshots -mindepth 1 -maxdepth 1 -type d -name 'energy-*' -printf '%f\n' | sort


build-inputs:
	$(PYTHON) scripts/materialize_analysis_inputs.py $(INPUT_BUILD_ARGS)

verify-input-builds:
	@set -e; found=0; \
	for build in results/input-builds/*; do \
	  if [ -d "$$build" ]; then \
	    found=1; \
	    $(PYTHON) scripts/verify_materialized_inputs.py "$$build"; \
	  fi; \
	done; \
	if [ "$$found" -eq 0 ]; then echo "No input builds found" >&2; exit 1; fi

list-input-builds:
	@find results/input-builds -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort

reproduce:
	$(PYTHON) scripts/run_analysis.py --compare-reference --inputs-from-source-snapshot $(RUN_ARGS)

.PHONY: docs-check
docs-check:
	$(PYTHON) scripts/check_publication_docs.py

.PHONY: extended-analysis
extended-analysis:
	$(PYTHON) scripts/analyze_context_geometry.py --repo .
