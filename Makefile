SHELL := /bin/bash
PYTHON := .venv/bin/python
SNAPSHOT := data/snapshots/t1-analysis-20260730
RUN_ARGS ?=

.PHONY: setup verify test analysis analysis-check list-runs clean-failed

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
