# Local Official-Derived Baseline Inventory

## Generated Scripts
- LogisticRegressionMLArgs: `/home/haizhe/copco/eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh` exists=True
- RandomForestMLArgs: `/home/haizhe/copco/eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_CopCo_TYP_folds_0_1_2_3.sh` exists=True
- Logistic generated YAML: `/home/haizhe/copco/eyebench/sweeps/CopCo_TYP_20251104/configs/LogisticRegressionMLArgs_CopCo_TYP.yaml` exists=True
- Sweep IDs are W&B IDs only: True

## Local Execution Paths
- Path A: run generated bash script if present; expected to use tmux/W&B agent
- Path B: generate scripts via official sweep_wrapper; expected to require W&B sweep creation
- Path C: local official-derived runner imports EyeBench DATA_CONFIGS_MAPPING, LogisticRegressionMLArgs, TrainerML, and DataModuleFactory metadata, then trains the matching sklearn pipeline on official processed data and official folds

## Official Code Paths
- `run_commands/utils/sweep_wrapper.sh`: `/home/haizhe/copco/eyebench/run_commands/utils/sweep_wrapper.sh` exists=True
- `run_commands/utils/test_wrapper_creator.sh`: `/home/haizhe/copco/eyebench/run_commands/utils/test_wrapper_creator.sh` exists=True
- `run_commands/utils/model_checker.sh`: `/home/haizhe/copco/eyebench/run_commands/utils/model_checker.sh` exists=True
- `src/run/single_run/test_ml.py` trains/evaluates ML runs after querying W&B sweep metadata.
- `src/run/multi_run/raw_to_processed_results.py` computes metrics from local raw prediction files.

## Prediction Schema
- `label`
- `prediction_prob`
- `prediction`
- `binary_prediction`
- `eval_regime`
- `eval_type`
- `fold_index`
- `participant_id`
- `unique_trial_id`
- `text_id`
