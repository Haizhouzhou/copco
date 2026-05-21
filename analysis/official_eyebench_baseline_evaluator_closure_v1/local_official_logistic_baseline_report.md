# Official Baseline Command-Source Report

- Baseline selected: `LogisticRegressionMLArgs`
- Command-source file: `eyebench/src/run/single_run/test_ml.py`
- tmux used: False
- Underlying generated command used: True
- Runtime prefix: `/home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal`
- W&B API available: False
- W&B API required for project: False
- W&B failure scientific blocker: False
- Local official-derived baseline pass: True
- Baseline reproduction pass: True
- Remaining blocker: none

## Command Attempts
| baseline_selected | command_source_file | command_kind | exact_command | tmux_used | underlying_generated_command_used | runtime_prefix | log_path | returncode | timed_out | status | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegressionMLArgs | eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh | path_a_generated_bash | bash /home/haizhe/copco/eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh 0 1 | False | True | /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal | /home/haizhe/copco/eyebench/.runtime_logs/closure_path_a_generated_logistic_bash.log | 0 | False | failed | generated official launcher hardcodes an unavailable private path |
| LogisticRegressionMLArgs | eyebench/run_commands/utils/sweep_wrapper.sh | path_b_sweep_wrapper | conda run -p /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal bash run_commands/utils/sweep_wrapper.sh --data_tasks CopCo_TYP --folds 0,1,2,3 --wandb_project CopCo_TYP_local_closure_20260522_015403 | False | True | /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal | /home/haizhe/copco/eyebench/.runtime_logs/closure_path_b_sweep_wrapper.log | 1 | False | failed | telemetry_orchestration_unavailable |
| LogisticRegressionMLArgs | eyebench/src/run/single_run/test_ml.py | official_test_ml_online | conda run -p /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal python src/run/single_run/test_ml.py --data_task CopCo_TYP --wandb_project CopCo_TYP_20251104 | False | True | /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal | /home/haizhe/copco/eyebench/.runtime_logs/closure_official_test_ml_online.log | 1 | False | failed | telemetry_orchestration_unavailable |
| LogisticRegressionMLArgs | eyebench/src/run/single_run/test_ml.py | official_test_ml_offline | conda run -p /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal python src/run/single_run/test_ml.py --data_task CopCo_TYP --wandb_project CopCo_TYP_20251104 | False | True | /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal | /home/haizhe/copco/eyebench/.runtime_logs/closure_official_test_ml_offline.log | 1 | False | failed | telemetry_orchestration_unavailable |
| LogisticRegressionMLArgs | eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh | official_wandb_agent_online | conda run -p /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal python -m wandb agent EyeRead/CopCo_TYP_20251104/pn6ofv0p | False | True | /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal | /home/haizhe/copco/eyebench/.runtime_logs/closure_official_wandb_agent_online.log | 1 | False | failed | telemetry_orchestration_unavailable |
| LogisticRegressionMLArgs | eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh | official_wandb_agent_offline | conda run -p /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal python -m wandb agent EyeRead/CopCo_TYP_20251104/pn6ofv0p | False | True | /home/haizhe/copco/eyebench/.envs/eyebench_official_py312_minimal | /home/haizhe/copco/eyebench/.runtime_logs/closure_official_wandb_agent_offline.log | 1 | False | failed | telemetry_orchestration_unavailable |

## Path C Official-Derived Baseline
- Attempted: True
- Result: passed
- Prediction path: `/home/haizhe/copco/results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/local_official_derived_predictions.csv`
- Trial-level EyeBench-compatible result path: `/home/haizhe/copco/results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/trial_level_test_results.csv`

### EyeBench Classes Used
```json
{
  "data_class": "CopCo_TYP",
  "datamodule_class": "CopCoDataModule",
  "datamodule_name": "CopCoDataModule",
  "item_level_features_modes": [
    "READING_SPEED"
  ],
  "model_class": "LogisticRegressionMLArgs",
  "sklearn_pipeline": [
    [
      "scaler",
      "sklearn.preprocessing.StandardScaler"
    ],
    [
      "clf",
      "sklearn.linear_model.LogisticRegression"
    ]
  ],
  "sklearn_pipeline_params": {
    "clf__C": 2.0,
    "clf__class_weight": "balanced",
    "clf__fit_intercept": true,
    "clf__max_iter": 1000,
    "clf__penalty": "l2",
    "clf__random_state": 1,
    "clf__solver": "lbfgs",
    "scaler__with_mean": true,
    "scaler__with_std": true
  },
  "trainer_class": "TrainerML"
}
```

### Metrics
| model_name | baseline_source | split_name | metric_basis | n_predictions | usable_folds | roc_auc | balanced_accuracy | pr_auc | macro_f1 | brier_score | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegressionMLArgs | local_official_derived_eyebench_classes | unseen_reader | official_trial_level_fold_mean | 3554 | 4 | 0.8304 | 0.7541 | 0.5583 | 0.6896 | 0.1843 | complete |
| LogisticRegressionMLArgs | local_official_derived_eyebench_classes | unseen_reader_and_text | official_trial_level_fold_mean | 1228 | 4 | 0.6910 | 0.6380 | 0.5314 | 0.6094 | 0.2368 | complete |
| LogisticRegressionMLArgs | local_official_derived_eyebench_classes | unseen_text | official_trial_level_fold_mean | 3554 | 4 | 0.8315 | 0.7665 | 0.5049 | 0.6880 | 0.1927 | complete |

## Local Diagnostic Baseline
The previous local diagnostic baseline remains separate and cannot unlock the official gate.

| model_name | baseline_source | split_name | metric_basis | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | balanced_accuracy | published_roc_auc | published_balanced_accuracy | delta_roc_auc | delta_balanced_accuracy | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_reader | official_trial_level_fold_mean | 5 | 3554 | 4 | 0 | 0.8304 | 0.7541 | 0.8310 | 0.7550 | -0.0006 | -0.0009 | complete |  |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_text | official_trial_level_fold_mean | 5 | 3554 | 4 | 0 | 0.8315 | 0.7665 | 0.8330 | 0.7660 | -0.0015 | 0.0005 | complete |  |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_reader_and_text | official_trial_level_fold_mean | 5 | 1228 | 4 | 0 | 0.6910 | 0.6380 | 0.6890 | 0.6350 | 0.0020 | 0.0030 | complete |  |
