# BenchmarkBridge v1 AI Run

## Request Summary

Start BenchmarkBridge v1 for the frozen CopCo DFM residual gaze-profile model. Add an
EyeBench-style internal benchmark bridge for CopCo TYP and auxiliary CopCo RCS without
changing frozen Phase 4 results, adding labels, or running broad model search.

## Plan

1. Inspect existing config, CLI, validation, residualization, Phase 4, AutoResearch,
   submission, paper, and test conventions.
2. Add `configs/benchmark_bridge_v1.yaml`.
3. Add `src/copco_eye_bench/benchmark_bridge.py` with participant-text sample building,
   deterministic benchmark splits, split-specific residualization, TYP/RCS evaluation,
   compatibility reports, comparison tables, validation, and decision reporting.
4. Wire CLI entrypoints and tests.
5. Run required validation commands and the benchmark bridge.
6. Commit and push safe files only.

## Files Inspected

- `configs/`
- `src/copco_eye_bench/cli.py`
- `src/copco_eye_bench/research_exploration.py`
- `src/copco_eye_bench/phase4_confirmatory.py`
- `tests/test_phase4_confirmatory.py`
- `tests/test_autoresearch.py`
- `analysis/submission_v1/`
- `analysis/autoresearch_v1/`
- `paper/submission_v1/`
- frozen result manifests and Phase 4 metrics

## Files Modified

- `configs/benchmark_bridge_v1.yaml`
- `src/copco_eye_bench/benchmark_bridge.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_benchmark_bridge.py`
- `docs/benchmark_bridge_v1_analysis_plan.md`
- `analysis/benchmark_bridge_v1/*`
- `analysis/submission_v1/claim_evidence_ledger.csv`
- `analysis/submission_v1/claim_evidence_ledger.md`
- `paper/submission_v1/sections/06_results.tex`
- `paper/submission_v1/sections/08_limitations.tex`
- `paper/submission_v1/supplement.tex`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`
- `paper/submission_v1/tables/benchmark_bridge_typ_comparison.tex`

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `conda run -n copco which python`
- `conda run -n copco python --version`
- `conda run -n copco python -c "import sys; print(sys.executable)"`
- repository inspection commands using `rg`, `find`, `sed`, and pandas schema reads
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m pytest tests/test_benchmark_bridge.py -q`
- `conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco copco-validate-label-release --config configs/label_release_v1_1.yaml --output-dir results/label_release_v1_1_20260506_0041`
- `conda run -n copco copco-validate-phase4-confirmatory --config configs/phase4_confirmatory_sensitivity_v1.yaml --output-dir results/phase4_confirmatory_sensitivity_v1_20260506_0715`
- `conda run -n copco copco-run-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1836`
- `conda run -n copco copco-validate-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1836`
- `conda run -n copco copco-run-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1908_accelerated`
- `conda run -n copco copco-validate-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1908_accelerated`
- `git diff --check`
- `sacct -j 2740465 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `sacct -j 2740652 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `sacct -j 2740729 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`

## Validation Results

- Editable install succeeded.
- Environment validation passed in the `copco` environment.
- Targeted BenchmarkBridge tests passed: 2 passed.
- Full pytest passed under CPU Slurm: 48 passed, 4 warnings.
- Ruff passed.
- Label Release v1.1 validation passed.
- Phase 4 validation passed.
- BenchmarkBridge output validation passed for both original and accelerated outputs.
- `git diff --check` passed.

## Slurm / Resource Notes

Full BenchmarkBridge residualization/evaluation is CPU/data-processing work. Runs used
the repo policy CPU launcher with the teaching 32 CPU / 128G candidate first.

- Original BenchmarkBridge job: `2740465`, completed, 00:48:59 elapsed,
  MaxRSS 4,884,976K, AveCPU 07:22:36.
- Accelerated BenchmarkBridge job: `2740652`, completed, 00:19:24 elapsed,
  MaxRSS 21,450,836K, AveCPU 07:32:55.
- Full pytest Slurm job: `2740729`, completed, 00:00:58 elapsed,
  MaxRSS 2,757,944K, AveCPU 00:03:03.
- `seff` was not available or returned no output.

The original run crossed 29 minutes while the serial fold loop was still active, so
`logs/current_status.md` was written as a long-running job triage report. The original
job was left running. A fold-parallel accelerated attempt wrote to the separate staging
directory `results/benchmark_bridge_v1_20260506_1908_accelerated` and validated.

## Result Summary

- Primary output: `results/benchmark_bridge_v1_20260506_1836`
- Accelerated validation output: `results/benchmark_bridge_v1_20260506_1908_accelerated`
- Participant-text samples: 242
- Split regimes completed: unseen reader, unseen text, unseen reader + text,
  text-balanced unseen reader, leave-one-speech-out, participant-grouped k-fold.
- Residualization leakage checks: no held-out reader rows, no held-out text rows, and
  no reader-group predictors used for residual fitting.
- Official EyeBench mode: not run; EyeBench package/repository/CLI was unavailable
  locally.
- Decision: `main_paper_comparison`, benchmark-relative/internal EyeBench-style.

## Final Response Summary

BenchmarkBridge v1 was implemented, validated, committed, pushed, and opened as a
draft PR. Primary validated output is `results/benchmark_bridge_v1_20260506_1836`;
accelerated validation output is `results/benchmark_bridge_v1_20260506_1908_accelerated`.

## Commit / Push Status

Committed on branch `codex/benchmark-bridge-v1` with message
`feat: add EyeBench-style benchmark bridge evaluation`. Pushed to origin and opened
draft PR `https://github.com/Haizhouzhou/copco/pull/1`. Full generated result
directories, parquet tables, full prediction CSVs, Slurm logs, caches, and ignored
artifacts were not committed.
