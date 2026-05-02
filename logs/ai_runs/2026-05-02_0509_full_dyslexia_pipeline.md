# 2026-05-02 05:09 - full_dyslexia_pipeline

## Request Summary

Implement the full CopCo dyslexia-labeled reader research program from the prior
agent plan, starting from a fresh context.

## Plan

1. Verify repository state, fast-forward `main`, and confirm the research plan exists.
2. Inspect the CopCo environment and local data/package surfaces.
3. Add committed configs, package modules, and CLI entrypoints for feature building,
   LM features, modeling, mixed effects, validation, and Slurm command construction.
4. Add focused pytest coverage for ID construction, join cardinality, fallbacks,
   tokenizer span alignment, grouped CV leakage checks, and validation schemas.
5. Install declared runtime dependencies inside the `copco` conda environment.
6. Run validation, tests, and a tiny sample pipeline without logging raw data.
7. Commit and push safe code/docs/config/log changes only.

## Files Inspected

- `pyproject.toml`
- `README.md`
- `.gitignore`
- `docs/decisions.md`
- `scripts/validate_env.py`
- `tests/test_import.py`
- `src/copco_eye_bench/__init__.py`
- `plan/copco_dyslexia_research_plan.md`
- `copco-processing/README.md`
- legacy schema headers from `copco-processing/ExtractedFeatures`,
  `InterestAreaReports`, and participant/text stats
- Continuation inspection after CPU-timeout interruption: `src/copco_eye_bench/cli.py`,
  `config.py`, `features.py`, `ids.py`, `lm_features.py`, `mixed_effects.py`,
  `modeling.py`, `resources.py`, `slurm.py`, `splits.py`, `validation.py`,
  `configs/copco_dyslexia_full.yaml`, `configs/copco_dyslexia_smoke.yaml`, and tests.

## Files Modified

- `pyproject.toml`
- `README.md`
- `docs/decisions.md`
- `scripts/validate_env.py`
- `configs/copco_dyslexia_full.yaml`
- `configs/copco_dyslexia_smoke.yaml`
- `src/copco_eye_bench/cli.py`
- `src/copco_eye_bench/config.py`
- `src/copco_eye_bench/features.py`
- `src/copco_eye_bench/ids.py`
- `src/copco_eye_bench/lm_features.py`
- `src/copco_eye_bench/mixed_effects.py`
- `src/copco_eye_bench/modeling.py`
- `src/copco_eye_bench/resources.py`
- `src/copco_eye_bench/slurm.py`
- `src/copco_eye_bench/splits.py`
- `src/copco_eye_bench/validation.py`
- `tests/test_ids_features_splits_validation.py`
- `tests/test_lm_alignment.py`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-02_0509_full_dyslexia_pipeline.md`

## Commands Run

- `pwd`
- `git status --short --branch`
- `python --version`
- `git pull --ff-only origin main`
- `test -f plan/copco_dyslexia_research_plan.md`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco ...`
- lightweight schema/header inspection commands
- `git rev-list --left-right --count main...origin/main`
- `python scripts/validate_env.py`
- `python -m pytest tests/ -q`
- `python -m ruff check .`
- `copco-build-features --config configs/copco_dyslexia_smoke.yaml --output-dir results/copco_dyslexia_smoke_codex_20260502_062253`
- `copco-run-lm-features --config configs/copco_dyslexia_smoke.yaml --output-dir results/copco_dyslexia_smoke_codex_20260502_062253 --dry-run`
- `copco-run-models --config configs/copco_dyslexia_smoke.yaml --output-dir results/copco_dyslexia_smoke_codex_20260502_062253`
- `copco-fit-mixed-effects --config configs/copco_dyslexia_smoke.yaml --output-dir results/copco_dyslexia_smoke_codex_20260502_062253`
- `copco-validate-run --output-dir results/copco_dyslexia_smoke_codex_20260502_062253`
- `copco-build-features --config configs/copco_dyslexia_full.yaml --print-slurm-command`
- `copco-run-lm-features --config configs/copco_dyslexia_full.yaml --output-dir results/example --print-slurm-command`

## Validation Results

- `python scripts/validate_env.py`: passed in the `copco` conda environment.
- `python -m pytest tests/ -q`: 10 passed.
- `python -m ruff check .`: passed.
- Smoke feature build completed under ignored
  `results/copco_dyslexia_smoke_codex_20260502_062253/`.
  - `word_observations`: 3,802 rows
  - `words`: 1,901 rows
  - `sentences`: 85 rows
  - `paragraphs`: 30 rows
  - `participants`: 2 rows
  - IA cross-check now read 2 matching files out of 57 available for the sample.
- LM smoke was dry-run only; torch/transformers are not installed in the active env.
- Models skipped on the smoke run because the first two sampled participants are
  single-class typical labels.
- Mixed-effects smoke wrote reports with 0 complete hypotheses, expected for missing
  LM/regression predictors in the smoke sample.
- `copco-validate-run` passed with no schema, duplicate-key, manifest, or leakage
  errors.
- Slurm launcher print checks confirmed CPU/GPU commands activate the `copco`
  environment before preflight and run under `set -euo pipefail`.

## Final Response Summary

- Implemented and validated the scaffold, then continued after interruption to reduce
  login-node CPU risk by honoring smoke sample defaults, limiting sampled IA report
  reads, and adding CPU Slurm launcher printing for feature builds.

## Commit / Push Status

- Committed locally; final commit SHA is reported in the assistant response.
- Push pending during log update.
