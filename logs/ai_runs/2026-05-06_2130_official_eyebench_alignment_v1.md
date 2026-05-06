# OfficialEyeBenchAlignment v1 AI Run

## Request Summary

Vendor the official EyeBench repository as `eyebench/`, inspect CopCo task/fold/evaluator
structure, and add an alignment layer that separates exact official-subset evaluation,
EyeBench-fold full-feature intersection evaluation, and BenchmarkBridge full-data
EyeBench-style evaluation.

## Plan

1. Branch from `codex/benchmark-bridge-v1` because BenchmarkBridge PR #1 is open.
2. Add EyeBench as a pinned submodule and write a vendor manifest.
3. Inspect EyeBench CopCo labels, fold metadata, model/data registration, and evaluator format.
4. Try the separate `eyebench` environment without modifying `copco`.
5. Implement `official_eyebench_alignment.py`, config, CLI entrypoints, tests, and reports.
6. Run validation, commit safe files only, push, and open a draft PR.

## Files Inspected

- `eyebench/README.md`
- `eyebench/environment.yml`
- `eyebench/pyproject.toml`
- `eyebench/data/CopCo/folds_metadata/`
- `eyebench/data/CopCo/labels/`
- `eyebench/src/data/`
- `eyebench/src/data/preprocessing/`
- `eyebench/src/run/`
- `eyebench/src/models/`
- `eyebench/src/configs/`
- `src/copco_eye_bench/benchmark_bridge.py`
- `src/copco_eye_bench/cli.py`
- `tests/`

## Files Modified

- `.gitmodules`
- `eyebench` submodule pointer
- `configs/official_eyebench_alignment_v1.yaml`
- `src/copco_eye_bench/official_eyebench_alignment.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_official_eyebench_alignment.py`
- `docs/eyebench_vendor_manifest.md`
- `analysis/official_eyebench_alignment_v1/*`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-06_2130_official_eyebench_alignment_v1.md`

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `git branch --show-current`
- GitHub PR inspection for BenchmarkBridge PR #1
- `git switch -c codex/official-eyebench-alignment-v1`
- `git submodule add https://github.com/EyeBench/eyebench.git eyebench`
- EyeBench structure inspection with `find`, `sed`, and pandas schema reads
- `conda env list`
- `conda run -n eyebench python -c "... import src ..."`
- `mamba env create -f eyebench/environment.yml`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python -m pytest tests/test_official_eyebench_alignment.py -q`
- `conda run -n copco python -m ruff check src/copco_eye_bench/official_eyebench_alignment.py tests/test_official_eyebench_alignment.py src/copco_eye_bench/cli.py pyproject.toml`
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "...32 CPUs 128G..." "cd /home/haizhe/copco && conda run -n copco copco-run-official-eyebench-alignment --config configs/official_eyebench_alignment_v1.yaml --output-dir results/official_eyebench_alignment_v1_20260506_2228"`
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "...32 CPUs 128G..." "cd /home/haizhe/copco && conda run -n copco copco-run-official-eyebench-alignment --config configs/official_eyebench_alignment_v1.yaml --output-dir results/official_eyebench_alignment_v1_20260506_2232"`
- `conda run -n copco copco-validate-official-eyebench-alignment --config configs/official_eyebench_alignment_v1.yaml --output-dir results/official_eyebench_alignment_v1_20260506_2232`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco copco-validate-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1836`
- `conda run -n eyebench python -c "import sys; print(sys.version)"`
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "...32 CPUs 128G..." "cd /home/haizhe/copco && conda run -n copco python -m pytest tests/ -q"`
- `git diff --check`
- `sacct -j 2741119,2741134,2741174 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `seff 2741134; seff 2741174`

## Current Environment Finding

An `eyebench` environment already exists but is not the declared EyeBench environment:
it reports Python 3.11 and fails to import EyeBench because `beartype` is missing.
`mamba env create -f eyebench/environment.yml` aborted because the existing environment
prefix would need overwrite confirmation. Official data preprocessing is therefore
blocked until the separate EyeBench environment is repaired or recreated.

## Validation Results

- Editable install succeeded.
- Targeted OfficialEyeBenchAlignment tests passed: 2 passed.
- Full pytest passed under CPU Slurm: 50 passed, 4 warnings.
- Ruff passed for the full repository.
- Environment validation passed in the `copco` environment.
- BenchmarkBridge v1 validation still passed.
- OfficialEyeBenchAlignment validation passed for `results/official_eyebench_alignment_v1_20260506_2232`.
- `git diff --check` passed.
- `seff` is not installed on this cluster image.

## Slurm / Resource Notes

OfficialEyeBenchAlignment full-data residualization/evaluation is CPU/data-processing
work. It used the repo policy CPU launcher with the teaching 32 CPU / 128G candidate
first.

- First alignment run: job `2741119`, completed, 00:03:56 elapsed, MaxRSS 12,850,432K.
- Corrected alignment run: job `2741134`, completed, 00:03:43 elapsed, MaxRSS 11,567,744K.
- Full pytest: job `2741174`, completed, 00:01:00 elapsed, MaxRSS 2,672,336K.

The corrected run used a new output directory and did not overwrite the first completed
run.

## Result Summary

- Output directory: `results/official_eyebench_alignment_v1_20260506_2232`
- EyeBench vendor method: git submodule.
- EyeBench commit: `ce87f38a3083aeed029c255716a1a51e6ae51167`
- Official EyeBench subset: skipped because the separate EyeBench environment does not
  import and `eyebench/data/CopCo/processed` is absent.
- EyeBench-fold full-feature intersection: completed for CopCo_TYP and CopCo_RCS.
- Full-data EyeBench-style: copied and validated against BenchmarkBridge v1.
- Alignment overlap: 57 common participants, 32 common speech/text IDs, 4,782 common
  trials, 31,986 common stimulus word rows, 0 unmapped EyeBench trials, 0 unmapped
  CopCo trials.
- Residualization leakage checks: no held-out reader rows, no held-out text rows, and
  no reader-group predictors used.
- Decision category: `benchmark_relative_sota_only`.

CopCo_TYP reader-aggregated D3_FullFeature_EyeBenchFolds metrics:

- unseen_reader: AUROC 0.8123, PR-AUC 0.7772, balanced accuracy 0.7387.
- unseen_text: AUROC 0.8141, PR-AUC 0.6640, balanced accuracy 0.6976.
- unseen_reader_and_text: AUROC 0.7240, PR-AUC 0.6066, balanced accuracy 0.7110.

CopCo_TYP full-data BenchmarkBridge reference:

- unseen_reader: AUROC 0.8961, balanced accuracy 0.8158.
- unseen_text: AUROC 0.8285, balanced accuracy 0.7444.
- unseen_reader_and_text: AUROC 0.8542, balanced accuracy 0.7458.

CopCo_RCS reader-aggregated D3_FullFeature_EyeBenchFolds metrics:

- unseen_reader: RMSE 1.5295, MAE 1.2230, R2 0.0998.
- unseen_text: RMSE 1.5905, MAE 1.2834, R2 0.1014.
- unseen_reader_and_text: RMSE 1.5116, MAE 1.1690, R2 0.1130.

## Final Response Summary

OfficialEyeBenchAlignment v1 was implemented and validated. EyeBench is vendored as a
submodule at `ce87f38a3083aeed029c255716a1a51e6ae51167`; exact official mode is blocked
by the missing/incorrect EyeBench environment and absent processed CopCo data. The
EyeBench-fold full-feature intersection and BenchmarkBridge full-data internal modes
completed. Final claim category is `benchmark_relative_sota_only`.

## Commit / Push Status

Committed with message `feat: add official EyeBench alignment evaluation`. Pushed branch
`codex/official-eyebench-alignment-v1` to origin and opened draft PR
`https://github.com/Haizhouzhou/copco/pull/2` against `codex/benchmark-bridge-v1`.
The final commit hash is reported in the final response.
