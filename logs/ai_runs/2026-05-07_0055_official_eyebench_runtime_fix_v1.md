# OfficialEyeBenchRuntimeFix v1

## Request Summary

Resolve the remaining infrastructure gap between benchmark-relative CopCo D3 results
and an official EyeBench CopCo_TYP evaluation. Keep EyeBench data, environments,
caches, and runtime logs isolated under `eyebench/` or ignored runtime results.

## Plan

1. Create `codex/official-eyebench-runtime-fix-v1` from the SOTACheck branch.
2. Verify the EyeBench submodule and isolate runtime/cache directories.
3. Follow the revised runtime rule: use `conda run -n copco` first, repair imports
   incrementally, and create a minimal Python 3.12 prefix only after a true Python
   3.12 syntax blocker is observed.
4. Add CopCo-side runtime-fix config, CLI, reports, tests, and decision gates.
5. Run validation, commit, and push without staging generated EyeBench artifacts.

## Files Inspected

- `src/copco_eye_bench/official_eyebench_sota_check.py`
- `src/copco_eye_bench/official_eyebench_alignment.py`
- `src/copco_eye_bench/cli.py`
- `configs/official_eyebench_sota_check_v1.yaml`
- `eyebench/environment.yml`
- `eyebench/pyproject.toml`
- `eyebench/src/data/preprocessing/get_data.sh`
- `eyebench/src/configs/data.py`
- `eyebench/run_commands/CopCo_TYP.md`
- `eyebench/src/run/multi_run/raw_to_processed_results.py`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`
- `eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh`

## Files Modified

- `.gitignore`
- `configs/official_eyebench_runtime_fix_v1.yaml`
- `src/copco_eye_bench/official_eyebench_runtime_fix.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_official_eyebench_runtime_fix.py`
- `docs/eyebench_runtime_fix_v1.md`
- `analysis/official_eyebench_runtime_fix_v1/*`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-07_0055_official_eyebench_runtime_fix_v1.md`

## Environment Attempts

- Slurm allocation: `SLURM_JOB_ID=3366552`, node `u24-chiivm0-607`, `gpu:0`,
  64 CPUs, 256G requested memory.
- CopCo env first: missing imports were repaired with `beartype==0.20.2`,
  `pymovements==0.25.0`, `rdata==1.0.0`, `loguru==0.7.2`, and
  `hydra-core==1.3.2`; `pip check` passed after each repair.
- CopCo env then failed on official EyeBench Python 3.12 syntax in
  `eyebench/src/data/utils.py`, line 295.
- Minimal Python 3.12 fallback was created at
  `eyebench/.envs/eyebench_official_py312_minimal`.
- Fallback repairs installed `requests`, `spacy==3.8.5`,
  `git+https://github.com/lacclab/text-metrics.git` (which exposed the renamed
  `psycholing_metrics` API), legacy
  `git+https://github.com/lacclab/text-metrics.git@v1.1.12` for the
  `text_metrics` API expected by EyeBench, `da_core_news_sm==3.8.0`,
  `typed-argument-parser==1.10.1`, and `pandas==2.2.3`.
- `pandas==2.2.3` was pinned only after a concrete pandas 3.0.3 runtime error in
  EyeBench CopCo preprocessing (`KeyError: 'paragraphId'` after groupby/apply).
- The full EyeBench `environment.yml` was not created and was not used.

## Official Data / Runtime Outcome

- Official command succeeded under the minimal Python 3.12 runtime:
  `PYTHONPATH=$PWD:$PWD/src conda run -p $PWD/.envs/eyebench_official_py312_minimal bash src/data/preprocessing/get_data.sh CopCo`.
- Processed files present:
  `ia.feather`, `fixations.feather`, `trial_level.feather`,
  `ia_trial_level_feature_keys.csv`, and `fixation_trial_level_feature_keys.csv`.
- Counts from the CopCo-side report: 57 participants, 452 text/items, 4,782 trials,
  335,597 word rows, 397,883 fixation rows.
- Official folds validated for unseen reader, unseen text, and unseen reader+text
  over 4 folds each.
- D3_EyeBench_Lite completed on official processed data/folds.
- The official EyeBench evaluator itself was not run, but the expected
  `trial_level_test_results.csv` result schema was validated.
- The local LogisticRegression baseline diagnostic reproduced the published
  logistic values within 0.003, but it was not run through the official W&B/tmux
  command-source path. The official baseline-reproduction gate therefore remains
  closed.
- Four tracked EyeBench `data/stats/*.csv` files were modified by preprocessing
  due floating-point formatting differences; these generated submodule changes were
  restored and the EyeBench submodule is clean.

## Commands Run

- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python -m pip install beartype pymovements rdata loguru hydra-core`
- `conda run -p eyebench/.envs/eyebench_official_py312_minimal python -m pip install ...`
- `PYTHONPATH=$PWD:$PWD/src conda run -p eyebench/.envs/eyebench_official_py312_minimal bash src/data/preprocessing/get_data.sh CopCo`
- `conda run -n copco copco-run-official-eyebench-runtime-fix --config configs/official_eyebench_runtime_fix_v1.yaml --output-dir results/official_eyebench_runtime_fix_v1_20260522_0005`
- `conda run -n copco copco-validate-official-eyebench-runtime-fix --config configs/official_eyebench_runtime_fix_v1.yaml --output-dir results/official_eyebench_runtime_fix_v1_20260522_0005`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m pytest tests/test_official_eyebench_runtime_fix.py -q`
- `conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco copco-validate-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1836`
- `conda run -p eyebench/.envs/eyebench_official_py312_minimal python -c "import sys; print(sys.version)"`
- `conda run -p eyebench/.envs/eyebench_official_py312_minimal python -c "import beartype; print('beartype ok')"`
- `PYTHONPATH=$PWD/eyebench:$PWD/eyebench/src PYTHONNOUSERSITE=1 conda run -p eyebench/.envs/eyebench_official_py312_minimal python -c "import src.data.preprocessing.preprocess_data; print('eyebench preprocessing import ok')"`
- `git diff --check`

## Validation Results

- Runtime-fix validation passed for
  `results/official_eyebench_runtime_fix_v1_20260522_0005`.
- `scripts/validate_env.py` passed.
- Targeted runtime-fix tests: 6 passed.
- Full test suite: 59 passed, 5 warnings.
- Ruff: passed.
- BenchmarkBridge v1 validation: passed.
- Minimal Python 3.12 fallback import checks: passed.
- `git diff --check`: passed.
- Final claim category: `blocked_by_baseline_reproduction`.
- Official SOTA allowed: false.
- D3_EyeBench_Lite official trial-level metrics:
  unseen_reader AUROC 0.8085, balanced accuracy 0.7274;
  unseen_text AUROC 0.8319, balanced accuracy 0.7341;
  unseen_reader_and_text AUROC 0.7154, balanced accuracy 0.6342.
- Reader-aggregated secondary metrics:
  unseen_reader AUROC 0.8468, balanced accuracy 0.7530;
  unseen_text AUROC 0.8606, balanced accuracy 0.7629;
  unseen_reader_and_text AUROC 0.7792, balanced accuracy 0.6201.
- Leakage flags all false: no held-out reader rows, held-out text rows, or
  reader group in residual fitting.

## Final Response Summary

OfficialEyeBenchRuntimeFix v1 obtained official EyeBench CopCo processed data and
folds, ran D3_EyeBench_Lite, validated the exact trial-level result schema, and
kept the manuscript claim benchmark-relative because official command-source
baseline reproduction and D3 official SOTA gates did not pass.

## Commit / Push Status

- Implementation commit: `7d8eaeb` (`fix: complete isolated official EyeBench runtime check`).
- Branch pushed: `codex/official-eyebench-runtime-fix-v1`.
- Push status: succeeded.
