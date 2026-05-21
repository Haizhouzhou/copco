# Official Evaluator Closure Report

- official_evaluator_run: False
- exact_result_format_validated: True
- evaluator_without_wandb_pass: True
- evaluator command: `python src/run/multi_run/raw_to_processed_results.py`
- official code path: `/home/haizhe/copco/eyebench/src/run/multi_run/raw_to_processed_results.py`
- schema checked path: `/home/haizhe/copco/results/official_eyebench_runtime_fix_v1_20260522_0005/typ/trial_level_test_results.csv`
- blocker: official evaluator aggregates official results/raw model directories and could not run because the official W&B command-source baseline did not produce raw outputs

## Import Attempt
```text
official evaluator functions import ok
/home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal/lib/python3.12/site-packages/transformers/utils/hub.py:128: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
```
