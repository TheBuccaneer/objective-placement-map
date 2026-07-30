SHELL := /bin/bash
PYTHON := .venv/bin/python
SNAPSHOT := data/snapshots/t1-analysis-20260730
RUN_ARGS ?=
SOURCE_REPO ?= $(HOME)/projects/energy
SOURCE_INVENTORY_ARGS ?=
SOURCE_SNAPSHOT_ARGS ?=

.PHONY: setup verify test analysis analysis-check list-runs clean-failed source-inventory list-source-inventories source-snapshot verify-source-snapshots list-source-snapshots

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
