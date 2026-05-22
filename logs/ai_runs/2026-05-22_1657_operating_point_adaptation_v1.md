# OperatingPointAdaptation v1

## Request Summary

Implement OperatingPointAdaptation v1 to analyze existing D3-family prediction
probabilities, fixed thresholds, legal threshold learning when non-test predictions are
available, calibration, reader-level probability aggregation, and test-oracle threshold
upper bounds. Do not run a new feature search, reproduce official leaderboard methods,
or change official SOTA claims.

## Plan

1. Inspect available prediction artifacts and existing CLI patterns.
2. Add config, module, CLI entrypoints, documentation, and tests.
3. Run the analysis on configured existing prediction sources.
4. Validate generated reports and run tests/ruff/env checks.
5. Commit and push only safe code, docs, configs, tests, small reports, and this log.

## Files Inspected

- `pyproject.toml`
- `src/copco_eye_bench/cli.py`
- `src/copco_eye_bench/d3_eyebench_own_method_score_max.py`
- `tests/test_d3_eyebench_own_method_score_max.py`
- `configs/d3_eyebench_own_method_score_max_v2.yaml`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`
- configured prediction CSV schemas under ignored `results/`

## Files Modified

- `configs/operating_point_adaptation_v1.yaml`
- `src/copco_eye_bench/operating_point_adaptation.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_operating_point_adaptation.py`
- `docs/operating_point_adaptation_v1.md`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-22_1657_operating_point_adaptation_v1.md`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`
- `analysis/operating_point_adaptation_v1/*`

## Commands Run

- `pwd`
- `git status --short --branch`
- `python --version`
- `conda run -n copco python --version`
- artifact discovery with `find`, `rg`, `wc`, and small pandas schema probes
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python -m pytest tests/test_operating_point_adaptation.py -q`
- `conda run -n copco copco-run-operating-point-adaptation --config configs/operating_point_adaptation_v1.yaml --output-dir results/operating_point_adaptation_v1_20260522_171644`
- `conda run -n copco copco-validate-operating-point-adaptation --config configs/operating_point_adaptation_v1.yaml --output-dir results/operating_point_adaptation_v1_20260522_171644`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m pytest tests/ -q`
- `sbatch ... conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m ruff check .`
- `git diff --check`
- `git status --short`

## Resource Notes

The configured prediction files total roughly 33k CSV rows. This is lightweight local
analysis, not a heavy Slurm job. No model training or feature search was run.

The full local pytest run failed after 82 tests passed with unrelated `MemoryError`
failures in plotting/submission tests. Per the task contract, the full suite was rerun
via Slurm job `3374333` on `u24-chiivm0-606` with 64 CPUs and 256G requested memory.
Preflight showed 64 CPUs visible and ample memory. `sacct` reported `COMPLETED` with
exit code `0:0`; the test suite passed.

## Validation Results

- Editable install: passed.
- Environment validation: passed.
- Focused tests: `8 passed`.
- Operating-point runner: passed, loaded 17,314 filtered prediction rows from 5 sources.
- Operating-point validator: passed with no warnings.
- Local full pytest: failed only due memory pressure in unrelated tests.
- Slurm full pytest job `3374333`: `88 passed, 4 warnings`.
- Ruff: passed.
- `git diff --check`: passed.

## Final Summary

OperatingPointAdaptation v1 now produces probability audits, threshold-source audits,
fixed-threshold metrics, legal-threshold not-computed records where inner/calibration
predictions are unavailable, diagnostic test-oracle threshold upper bounds with
information budgets, identity calibration metrics, probability-first reader aggregation
metrics, threshold curves, and final decision artifacts.

Final decision category: `supplement_supporting_result`. Official SOTA claim changed:
false.

## Commit / Push

Committed as `2b75423 feat: add probability-first operating point adaptation analysis`
and pushed to `origin/codex/d3-eyebench-own-method-score-max-v2`.
