# OfficialEyeBenchBaselineEvaluatorClosure v1

## Request Summary

Close the remaining official EyeBench baseline command-source and evaluator/result-format
gates for CopCo_TYP without changing the frozen D3 method or full-data result.

## Plan

- Start from `codex/official-eyebench-runtime-fix-v1` and branch to
  `codex/official-eyebench-baseline-evaluator-closure-v1`.
- Use the required UZH teaching Slurm allocation for runtime work.
- Reuse official processed CopCo data, official folds, and prior D3_EyeBench_Lite
  outputs.
- Repair only missing import/runtime packages in the minimal Python 3.12 prefix.
- Attempt the official EyeBench command-source baseline and evaluator paths.
- Keep local diagnostic baseline metrics separate from official command-source evidence.
- Update reports, decision JSON, supplement wording, tests, commit, and push.

## Files Inspected

- `eyebench/run_commands/CopCo_TYP.md`
- `eyebench/sweeps/CopCo_TYP_20251104/configs/LogisticRegressionMLArgs_CopCo_TYP.yaml`
- `eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh`
- `eyebench/src/run/single_run/test_ml.py`
- `eyebench/src/run/multi_run/raw_to_processed_results.py`
- `configs/official_eyebench_runtime_fix_v1.yaml`
- `src/copco_eye_bench/official_eyebench_runtime_fix.py`
- `src/copco_eye_bench/official_eyebench_sota_check.py`
- `results/official_eyebench_runtime_fix_v1_20260522_0005/`

## Slurm Allocation

- Command: `srun --partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:0 --cpus-per-task=64 --mem=256G --time=04:00:00 --pty bash`
- Job ID: `3366756`
- Host: `u24-chiivm0-606`
- Partition: `teaching`
- CPUs per task: `64`

## Dependency Repairs

- Installed/verified in `eyebench/.envs/eyebench_official_py312_minimal`:
  - `wandb==0.23.1`
  - `lightning==2.5.1`
  - `pytorch-metric-learning==2.9.0`
  - `packaging==24.2`
  - `transformers==4.47.1`
  - `seaborn==0.13.2`
  - `matplotlib==3.10.1`
- `pip check` passed after pinning `packaging==24.2`.

## Command Evidence

- `test_ml.py` first failed on missing runtime imports.
- After package repair, `test_ml.py` reached `wandb.Api()` and failed because no W&B
  API key was configured.
- Offline W&B retry produced the same W&B API-key blocker.
- The generated `wandb agent` command from the official sweep script also failed
  online and offline because W&B sweep-agent access requires login/API key.
- Continuation classified W&B access as telemetry/orchestration, not a scientific
  blocker.
- Path A official generated bash script was attempted and rejected as a launcher-only
  path because it hardcodes `$HOME/eyebench_private`.
- Path B official sweep wrapper was retried inside the py312 prefix and reached
  `wandb.sweep`; this confirmed the remaining failure is online W&B telemetry/sweep
  creation.
- Path C used official EyeBench CopCo data/folds/classes locally:
  `DATA_CONFIGS_MAPPING["CopCo_TYP"]`, `CopCoDataModule`,
  `LogisticRegressionMLArgs`, `TrainerML`, and the EyeBench sklearn logistic
  pipeline.
- Path C produced real prediction and metric files under
  `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/`.
- Local official-derived Logistic Regression metrics:
  - unseen_reader: AUROC 0.8304, balanced accuracy 0.7541.
  - unseen_text: AUROC 0.8315, balanced accuracy 0.7665.
  - unseen_reader_and_text: AUROC 0.6910, balanced accuracy 0.6380.
- D3_EyeBench_Lite trial-level metrics:
  - unseen_reader: AUROC 0.8085, balanced accuracy 0.7274.
  - unseen_text: AUROC 0.8319, balanced accuracy 0.7341.
  - unseen_reader_and_text: AUROC 0.7154, balanced accuracy 0.6342.
- D3_EyeBench_Lite reader-aggregated metrics, secondary only:
  - unseen_reader: AUROC 0.8468, balanced accuracy 0.7530.
  - unseen_text: AUROC 0.8606, balanced accuracy 0.7629.
  - unseen_reader_and_text: AUROC 0.7792, balanced accuracy 0.6201.
- Final decision: `official_compatible_but_not_sota`.
- Official online W&B-reproduced EyeBench SOTA is not allowed.
- Official-compatible local-baseline SOTA is not supported because D3 did not beat the
  reproduced local official-derived logistic baseline on the primary trial-level
  balanced-accuracy comparison.

## Modified Files

- `.gitignore`
- `configs/official_eyebench_baseline_evaluator_closure_v1.yaml`
- `docs/eyebench_baseline_evaluator_closure_v1.md`
- `src/copco_eye_bench/official_eyebench_baseline_evaluator_closure.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_official_eyebench_baseline_evaluator_closure.py`
- `scripts/slurm/official_eyebench_baseline_evaluator_closure_v1/*.sbatch`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/*`
- `paper/submission_v1/supplement_sections/18_benchmark_bridge.tex`
- `logs/ai_runs/INDEX.md`

## Validation Results

- Slurm job `3366769`, `3366770`, and final job `3366776` used:
  `teaching`, account `mlnlp2.pilot.s3it.uzh`, QoS `normal`, `--gres=gpu:0`,
  `--cpus-per-task=64`, `--mem=256G`, `--time=04:00:00`.
- Final output directory:
  `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211`.
- Final sbatch stdout:
  `results/official_eyebench_baseline_evaluator_closure_v1_sbatch/slurm/oe_closure_full_3366776.out`.
- Final sbatch stderr:
  `results/official_eyebench_baseline_evaluator_closure_v1_sbatch/slurm/oe_closure_full_3366776.err`.
- `conda run -n copco python -m pip install -e .`: passed.
- `conda run -n copco python scripts/validate_env.py`: passed.
- `conda run -n copco python -m pytest tests/ -q`: 71 passed, 5 warnings.
- `conda run -n copco python -m ruff check .`: passed.
- `conda run -n copco copco-validate-benchmark-bridge --config configs/benchmark_bridge_v1.yaml --output-dir results/benchmark_bridge_v1_20260506_1836`: passed.
- `conda run -n copco copco-run-official-eyebench-baseline-evaluator-closure --config configs/official_eyebench_baseline_evaluator_closure_v1.yaml --output-dir results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211`: passed.
- `conda run -n copco copco-validate-official-eyebench-baseline-evaluator-closure --config configs/official_eyebench_baseline_evaluator_closure_v1.yaml --output-dir results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211`: passed.
- Direct py312 runtime checks passed, including `import src`.
- `sacct -j 3366776`: completed with exit code `0:0`, elapsed `00:02:49`.
- `seff` was unavailable on the cluster shell.

## Commit / Push

Pending at log update time.
