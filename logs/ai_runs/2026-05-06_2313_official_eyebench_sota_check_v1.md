# OfficialEyeBenchSOTACheck v1

## Request Summary

Resolve the remaining gap between benchmark-relative SOTA and official EyeBench SOTA
without changing the frozen CopCo D3 full-data result or the scientific claim.

## Plan

- Create a clean official EyeBench environment or record the exact blocker.
- Check official processed CopCo data and folds.
- Add a gated OfficialEyeBenchSOTACheck CLI for baseline reproduction, D3_EyeBench_Lite
  official-fold evaluation, leakage validation, comparison tables, and claim decision.
- Keep official SOTA wording disabled unless environment, data, baseline, D3, and leakage
  gates pass.

## Files Inspected

- `eyebench/environment.yml`
- `eyebench/README.md`
- `eyebench/src/data/preprocessing/get_data.sh`
- `eyebench/src/data/preprocessing/dataset_preprocessing/copco.py`
- `eyebench/src/run/single_run/test_ml.py`
- `eyebench/src/run/multi_run/raw_to_processed_results.py`
- `src/copco_eye_bench/official_eyebench_alignment.py`
- `tests/test_official_eyebench_alignment.py`

## Files Modified

- `configs/official_eyebench_sota_check_v1.yaml`
- `src/copco_eye_bench/official_eyebench_sota_check.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_official_eyebench_sota_check.py`
- `analysis/official_eyebench_sota_check_v1/*`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`

## Environment Attempts

- Existing `eyebench` env was inspected and remained broken: missing `beartype`,
  `pymovements`, `rdata`, and `datasets`; Torch import failed.
- Exact clean env attempt:
  `mamba env create -n eyebench_official -f eyebench/environment.yml`
  failed during solve with incompatible packages and missing CUDA virtual package.
- Flexible retry:
  `CONDA_OVERRIDE_CUDA=12.4 mamba env create --channel-priority flexible -n eyebench_official -f eyebench/environment.yml`
  solved but exited nonzero during the transaction with a libmamba callback failure.
- The resulting empty `eyebench_official` prefix was removed.

## Commands Run

- `pwd`
- `git status --short && git branch --show-current && git log --oneline -4 --decorate`
- `python --version`
- `git switch -c codex/official-eyebench-sota-check-v1`
- `mamba env create -n eyebench_official -f eyebench/environment.yml`
- `CONDA_OVERRIDE_CUDA=12.4 mamba env create --channel-priority flexible -n eyebench_official -f eyebench/environment.yml`
- `conda run -n eyebench_official ... import check`
- `conda run -n eyebench ... import check`
- `mamba env remove -n eyebench_official -y`
- `conda run -n copco python -m py_compile src/copco_eye_bench/official_eyebench_sota_check.py src/copco_eye_bench/cli.py`
- `conda run -n copco python -m pytest tests/test_official_eyebench_sota_check.py -q`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco copco-run-official-eyebench-sota-check --config configs/official_eyebench_sota_check_v1.yaml --output-dir results/official_eyebench_sota_check_v1_20260506_2341`
- `conda run -n copco copco-validate-official-eyebench-sota-check --config configs/official_eyebench_sota_check_v1.yaml --output-dir results/official_eyebench_sota_check_v1_20260506_2341`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m pytest tests/test_research_exploration.py::test_research_exploration_generates_reports_and_stable_metrics tests/test_submission.py::test_claim_evidence_ledger_and_metric_consistency tests/test_submission.py::test_figure_table_references_and_required_sections tests/test_submission.py::test_no_prohibited_claims_variables_and_repro_scripts -q`
- `conda run -n copco copco-validate-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1836`
- `conda run -n eyebench_official python -c "import sys; print(sys.version)"`
- `git diff --check`

## Validation Results

- Focused SOTACheck tests: passed, `3 passed`.
- SOTACheck output validation: passed with warning `official split labels are empty`.
- CopCo env validation: passed.
- Ruff: passed.
- Full test suite: first run had 49 passed and 4 transient CSV parser failures in existing
  research/submission tests; rerunning those four tests passed.
- BenchmarkBridge v1 validation: passed.
- `eyebench_official` import check: failed because the environment does not exist after
  the failed create attempt was removed.
- `git diff --check`: passed.
- Official EyeBench environment: blocked.
- Official processed CopCo data: absent.
- Official baseline reproduction: skipped.
- D3_EyeBench_Lite official metrics: skipped.

## Output

- `results/official_eyebench_sota_check_v1_20260506_2341`
- `analysis/official_eyebench_sota_check_v1/`

## Decision

Final category: `blocked_by_environment`.

The main manuscript claim remains benchmark-relative. A supplement-only note was added
to document that official SOTA verification was attempted and blocked by environment/data
gates.

## Commit / Push Status

Commit prepared with message `feat: add official EyeBench SOTA verification`.
Push is attempted after this log update.
