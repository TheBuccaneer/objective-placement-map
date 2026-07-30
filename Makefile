SHELL := /bin/bash
PYTHON := .venv/bin/python
SNAPSHOT := data/snapshots/t1-analysis-20260730

.PHONY: setup verify test

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

verify:
	$(PYTHON) scripts/verify_snapshot.py $(SNAPSHOT)

test:
	$(PYTHON) -m pytest
