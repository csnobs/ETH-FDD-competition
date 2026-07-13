# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is the competition component for Week 1 (Information Theory) of the "From Data to Decisions" module in the ETHZ CAS in Data and Machine Learning program. The repo currently holds a minimal Python scaffold — actual Information Theory competition code (entropy/coding exercises, submission logic, etc.) has not been added yet. **Update this file as real logic and structure are added.**

## Commands
- Install deps: `pip install -r requirements.txt`
- Run tests: `pytest`
- Run a single test: `pytest tests/test_main.py::test_main_runs`
- Start app: `python main.py`

## Project structure
- `main.py` — entry point
- `src/` — core code (currently empty package)
- `data/` — raw datasets (do not edit)
- `tests/` — unit tests (pytest)

## Conventions
- Use type hints everywhere