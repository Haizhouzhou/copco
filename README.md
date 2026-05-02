# CopCo Eye Bench

Research-code checkpoint for CopCo / Eye Bench work on Danish natural-reading
eye-tracking analysis and benchmarking.

## Status

Phase 0 / repository initialization. This repository does not contain benchmark
results yet.

## Purpose

This project will support reproducible CopCo preprocessing, corpus audit, eye-tracking
feature analysis, Danish reading modeling, benchmark construction, and later carefully
scoped comparisons to other reading-related datasets where appropriate.

## Scientific Caution

CopCo is a Danish natural-reading eye-tracking corpus. It should not be described as a
dyslexia dataset. Dyslexia-related or reading-difficulty-related work must use careful
operational language and must not imply clinical diagnosis, clinical validation, or
medical screening unless explicitly supported by the project data.

## Planned Layout

```text
.
  AGENTS.md                  project rules for Codex and other agents
  pyproject.toml             minimal Python project metadata
  src/copco_eye_bench/       lightweight package namespace
  scripts/                   maintenance and validation scripts
  tests/                     lightweight tests
  docs/                      decisions and protocol notes
  logs/ai_runs/              AI run logs and index
  results/                   generated outputs, not committed
  data/                      local data, not committed
```

## Setup

Use Python 3.10 or newer. For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Lightweight Validation

```bash
python scripts/validate_env.py
python -m pytest tests/ -q
```

## Dyslexia-Labeled Reader Pipeline

The committed CLI scaffold implements the reproducible research program in
`plan/copco_dyslexia_research_plan.md` while keeping generated outputs ignored under
`results/`.

```bash
copco-build-features --config configs/copco_dyslexia_full.yaml
copco-run-lm-features --output-dir results/<run> --dry-run
copco-run-models --output-dir results/<run>
copco-fit-mixed-effects --output-dir results/<run>
copco-validate-run --output-dir results/<run>
```

For heavy LM scoring or CPU-heavy modeling, print the Slurm launcher command first:

```bash
copco-build-features --config configs/copco_dyslexia_full.yaml --print-slurm-command
copco-run-lm-features --output-dir results/<run> --print-slurm-command
copco-run-models --output-dir results/<run> --print-slurm-command
```

The pipeline uses the local `derived57` package when it is importable. If it is absent,
the run manifest records that missing input and falls back to the ignored local
`copco-processing/` schema when available. Scientific wording remains
"dyslexia-labeled reader"; outputs are not clinical diagnosis, screening, or medical
validation.

## Data Policy

Raw data, copied datasets, derived participant-level tables, large artifacts, model
checkpoints, and local environments must not be committed. Keep local data under
ignored paths such as `data/`, `raw/`, `external/`, `extracted/`, or `derived/`, and
document how to obtain or regenerate it.

## Reproducibility Policy

Benchmark results must be reproducible from committed code, documented external data
inputs, frozen target definitions, scoring rules, dataset splits, leakage controls, and
claim language. Phase 0 must freeze those choices before generating benchmark results.

## Citations

Citation entries will be added before public benchmark release:

- CopCo first-release paper: TBD
- CopCo dataset record: TBD
