# D3 EyeBench Protocol Optimization Campaign v1

## Request Summary

Follow `docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md` and build/run a
bounded, leakage-controlled, official-fold D3 optimization campaign through
`sbatch`, without rerunning official baselines except sanity anchors.

## Plan

- Inspect repo state, CopCo environment, protocol, prior closure evidence, and
  official EyeBench D3 loader/evaluator code.
- Add a dedicated protocol-aligned optimization runner, config, CLI entrypoints,
  tests, and teaching-account `sbatch` script.
- Run lightweight validation locally in the CopCo environment.
- Launch the actual campaign through `sbatch`.
- Validate decision/leakage/report artifacts.
- Commit and push only validated code, configs, scripts, reports, and decision
  artifacts; never large data/results.

## Files Inspected

- `docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md`
- `src/copco_eye_bench/official_eyebench_sota_check.py`
- `src/copco_eye_bench/official_eyebench_runtime_fix.py`
- `src/copco_eye_bench/official_eyebench_baseline_evaluator_closure.py`
- `src/copco_eye_bench/benchmark_bridge.py`
- `configs/official_eyebench_runtime_fix_v1.yaml`
- `configs/official_eyebench_baseline_evaluator_closure_v1.yaml`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/local_official_logistic_baseline_report.md`
- `scripts/slurm/official_eyebench_baseline_evaluator_closure_v1/run_full_validation.sbatch`
- `.gitignore`
- `pyproject.toml`

## External Reference Checked

- EyeBench project page: `https://eyebench.github.io/`
- EyeBench CopCo TYP leaderboard: `https://eyebench.github.io/eyebench/results/copco_typ/`

## Files Modified

- `.gitignore`
- `analysis/d3_eyebench_protocol_aligned_optimization_v1/*`
- `analysis/d3_eyebench_protocol_aligned_optimization_v1_accelerated/*`
- `configs/d3_eyebench_protocol_aligned_optimization_v1.yaml`
- `configs/d3_eyebench_protocol_aligned_optimization_v1_accelerated.yaml`
- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `scripts/slurm/d3_eyebench_protocol_aligned_optimization_v1/run_optimization.sbatch`
- `scripts/slurm/d3_eyebench_protocol_aligned_optimization_v1/run_optimization_accelerated.sbatch`
- `tests/test_d3_eyebench_protocol_optimization.py`
- `logs/ai_runs/2026-05-22_0742_d3_eyebench_protocol_optimization_campaign_v1.md`
- `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `git rev-parse --show-toplevel`
- `conda run -n copco which python`
- `conda run -n copco python --version`
- `conda run -n copco python -c "import sys; print(sys.executable)"`
- `rg`/`sed`/`find` inspections listed above
- `conda run -n copco python -c "<official processed schema inspection>"`
- `conda run -n copco python -m py_compile src/copco_eye_bench/d3_eyebench_protocol_optimization.py src/copco_eye_bench/cli.py`
- `conda run -n copco python -m pytest tests/test_d3_eyebench_protocol_optimization.py -q`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python -m ruff check src/copco_eye_bench/d3_eyebench_protocol_optimization.py src/copco_eye_bench/cli.py tests/test_d3_eyebench_protocol_optimization.py`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco copco-run-d3-eyebench-protocol-optimization --config configs/d3_eyebench_protocol_aligned_optimization_v1.yaml --print-slurm-command`
- `sbatch --parsable scripts/slurm/d3_eyebench_protocol_aligned_optimization_v1/run_optimization.sbatch`
- `squeue -j 3368823 -o "%i %T %M %D %R"`
- `sacct -j 3368823 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `sbatch --parsable scripts/slurm/d3_eyebench_protocol_aligned_optimization_v1/run_optimization.sbatch`
- `squeue -j 3368826 -o "%i %T %M %D %R"`
- `sstat -j 3368826.batch --format=JobID,AveCPU,AveRSS,MaxRSS,NTasks`
- `sbatch --parsable scripts/slurm/d3_eyebench_protocol_aligned_optimization_v1/run_optimization_accelerated.sbatch`
- `squeue -j 3368826,3368833 -o "%i %T %M %D %R"`
- `sacct -j 3368826,3368833 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `conda run -n copco copco-validate-d3-eyebench-protocol-optimization --config configs/d3_eyebench_protocol_aligned_optimization_v1.yaml --output-dir results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957`
- `conda run -n copco copco-validate-d3-eyebench-protocol-optimization --config configs/d3_eyebench_protocol_aligned_optimization_v1_accelerated.yaml --output-dir results/d3_eyebench_protocol_aligned_optimization_v1_accelerated_20260522_080058`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m pytest tests/ -x -vv`
- `conda run -n copco python -m pytest tests/test_research_exploration.py::test_research_exploration_generates_reports_and_stable_metrics -q`
- `conda run -n copco python -m pytest tests/test_submission.py::test_no_prohibited_claims_variables_and_repro_scripts -q`

## Validation Results

- Config validation passed for the base and accelerated D3 optimization configs.
- `python -m py_compile` for the new module and CLI: passed.
- Targeted D3 optimization tests: 4 passed.
- `python scripts/validate_env.py`: passed.
- `python -m ruff check .`: passed.
- Base output protocol validator: passed.
- Accelerated output protocol validator: passed.
- Full `python -m pytest tests/ -q`: did not produce a clean aggregate pass on the
  login node because two late tests hit transient `MemoryError` / `std::bad_alloc`
  failures in Matplotlib/pathlib after most of the suite had run.
- Isolated rerun of
  `tests/test_research_exploration.py::test_research_exploration_generates_reports_and_stable_metrics`:
  passed.
- Isolated rerun of
  `tests/test_submission.py::test_no_prohibited_claims_variables_and_repro_scripts`:
  passed.

## Slurm / Campaign Results

- Failed first attempt: job `3368823`, exit code `1:0`, elapsed `00:00:32`.
  Failure was an implementation bug: official numeric predictors contained `inf`
  values that sklearn rejected during residualization.
- Fix applied: numeric matrices now coerce infinities to missing values before
  residualization and classifier scoring.
- Base run: job `3368826`, completed with exit code `0:0`, elapsed `01:21:28`,
  MaxRSS `2838480K`, AveCPU `03:28:51`, output
  `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957`.
- Safe acceleration: job `3368833`, completed with exit code `0:0`, elapsed
  `01:07:30`, MaxRSS `8839348K`, AveCPU `02:49:16`, output
  `results/d3_eyebench_protocol_aligned_optimization_v1_accelerated_20260522_080058`.
- `seff` was unavailable on the cluster shell.
- Both successful jobs evaluated 56 candidates and stopped by
  `no_improvement_stopping_rule`.
- Locked candidate: `d3opt_0024_2d9a9f9c46`.
- Best inner-validation balanced accuracy: `0.7703242921042901`.
- Final trial-level balanced accuracy:
  - unseen_reader: `0.663349`
  - unseen_text: `0.759744`
  - unseen_reader_and_text: `0.616654`
  - average: `0.679915506990474`
- Local Logistic anchor average balanced accuracy: `0.7195458894485539`.
- Official target balanced accuracy: `0.750`.
- Final decision category: `official_compatible_but_not_sota`.
- No official baseline rerun, test-label tuning, synthetic outputs, random
  predictions, denied predictors, or full prepared CopCo join was used.

## Commit / Push

This artifact set is staged for the final campaign commit and push. No
`results/` artifacts or files at or above 100 MB are staged. The exact pushed
commit hash is reported in the assistant final response after Git completes.
