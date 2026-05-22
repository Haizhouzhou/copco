# D3 EyeBench Protocol-Aligned Optimization Report

- Decision category: `official_compatible_but_not_sota`
- Stop reason: `no_improvement_stopping_rule`
- Candidate count evaluated: 56
- Best candidate: `d3opt_0024_2d9a9f9c46`
- Selection source: `inner_validation_only`
- Official baseline rerun performed: False
- Test-label tuning: False
- Synthetic outputs used: False
- Random predictions used: False
- Full prepared CopCo join used: False

## Best Candidate
```json
{
  "aggregation_set": "central_spread",
  "candidate_id": "d3opt_0024_2d9a9f9c46",
  "classifier": "logistic_regression",
  "classifier_params": {
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 2000,
    "penalty": "l2",
    "solver": "liblinear"
  },
  "outcome_set": "duration_plus_count",
  "predictor_set": "surface_surprisal_syntax",
  "residual_alpha": 1.0,
  "seed": 20260522,
  "transform": "log1p_duration"
}
```

## Trial-Level Primary Metrics
| mode | model_name | claim_type | task | split_name | evaluation_level | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | threshold | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_eyebench_subset | D3_EyeBench_Optimized | official_compatible_candidate | CopCo_TYP | unseen_reader | official_trial_level_fold_mean | 9 | 3554 | 4 | 0 | 0.8213 | 0.5798 | 0.6633 | 0.6265 | 0.1873 | 0.6050 | complete |  |
| official_eyebench_subset | D3_EyeBench_Optimized | official_compatible_candidate | CopCo_TYP | unseen_text | official_trial_level_fold_mean | 9 | 3554 | 4 | 0 | 0.8425 | 0.5619 | 0.7597 | 0.6614 | 0.1837 | 0.4475 | complete |  |
| official_eyebench_subset | D3_EyeBench_Optimized | official_compatible_candidate | CopCo_TYP | unseen_reader_and_text | official_trial_level_fold_mean | 9 | 1228 | 4 | 0 | 0.6929 | 0.5415 | 0.6167 | 0.5900 | 0.2246 | 0.5475 | complete |  |

## Reader-Aggregated Secondary Metrics
| mode | model_name | claim_type | task | split_name | evaluation_level | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | threshold | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_eyebench_subset | D3_EyeBench_Optimized | official_compatible_candidate | CopCo_TYP | unseen_reader | reader_aggregated |  | 55 | 4 | 0 | 0.8600 | 0.7990 | 0.6340 | 0.6223 | 0.1788 | 0.6050 | complete |  |
| official_eyebench_subset | D3_EyeBench_Optimized | official_compatible_candidate | CopCo_TYP | unseen_text | reader_aggregated |  | 113 | 4 | 0 | 0.8953 | 0.7320 | 0.8151 | 0.7248 | 0.1708 | 0.4475 | complete |  |
| official_eyebench_subset | D3_EyeBench_Optimized | official_compatible_candidate | CopCo_TYP | unseen_reader_and_text | reader_aggregated |  | 39 | 4 | 0 | 0.7292 | 0.6802 | 0.6265 | 0.5938 | 0.1981 | 0.5475 | complete |  |

## Top Inner-Validation Candidates
| candidate_id | residual_alpha | predictor_set | outcome_set | aggregation_set | transform | classifier | classifier_params | seed | selection_metric | selection_score | complete_folds | evaluated_folds | mean_threshold | status | skip_reason | unseen_reader_inner_balanced_accuracy | unseen_text_inner_balanced_accuracy | unseen_reader_and_text_inner_balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d3opt_0024_2d9a9f9c46 | 1.0000 | surface_surprisal_syntax | duration_plus_count | central_spread | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7703 | 12 | 12 | 0.5333 | complete |  | 0.8255 | 0.7485 | 0.7369 |
| d3opt_0036_1d46253469 | 0.1000 | surface_surprisal | all_gaze | robust_full | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7702 | 12 | 12 | 0.5250 | complete |  | 0.8035 | 0.7731 | 0.7339 |
| d3opt_0009_29325c8105 | 10.0000 | surface_surprisal | all_gaze | robust_full | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7692 | 12 | 12 | 0.5225 | complete |  | 0.8017 | 0.7726 | 0.7335 |
| d3opt_0017_e55ad47811 | 100.0000 | surface_surprisal | duration_plus_count | central_spread | raw | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7679 | 12 | 12 | 0.5450 | complete |  | 0.8216 | 0.7499 | 0.7321 |
| d3opt_0054_be5ccbbec5 | 10.0000 | surface_surprisal_syntax | all_gaze | robust_full | raw | gradient_boosting | {'n_estimators': 180, 'learning_rate': 0.05, 'max_depth': 2} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7678 | 12 | 12 | 0.2642 | complete |  | 0.7634 | 0.7981 | 0.7418 |
| d3opt_0008_df66566550 | 10.0000 | surface_surprisal_syntax | duration_core | robust_full | raw | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7670 | 12 | 12 | 0.4950 | complete |  | 0.7984 | 0.7652 | 0.7375 |
| d3opt_0014_c5981f7cbf | 0.1000 | surface_surprisal_syntax | duration_plus_count | robust_full | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7669 | 12 | 12 | 0.5042 | complete |  | 0.7977 | 0.7645 | 0.7386 |
| d3opt_0006_20828c33ee | 100.0000 | surface_surprisal_syntax | duration_core | robust_full | raw | logistic_regression | {'C': 10.0, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7665 | 12 | 12 | 0.5008 | complete |  | 0.7797 | 0.7728 | 0.7469 |
| d3opt_0029_7543248d5b | 1.0000 | surface_surprisal_syntax | duration_plus_count | central_spread | log1p_duration | logistic_regression | {'C': 10.0, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7663 | 12 | 12 | 0.5283 | complete |  | 0.8082 | 0.7481 | 0.7425 |
| d3opt_0056_76e63ddcc9 | 0.1000 | surface | all_gaze | central_spread | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7662 | 12 | 12 | 0.5517 | complete |  | 0.8143 | 0.7501 | 0.7341 |
| d3opt_0048_fb5f9ca843 | 10.0000 | surface_surprisal_syntax | all_gaze | central_spread | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7661 | 12 | 12 | 0.5400 | complete |  | 0.8136 | 0.7486 | 0.7359 |
| d3opt_0055_f928b3668b | 0.1000 | surface | duration_core | robust_full | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7659 | 12 | 12 | 0.4867 | complete |  | 0.8026 | 0.7607 | 0.7345 |
| d3opt_0012_3fccec14be | 0.1000 | surface_surprisal_syntax | all_gaze | robust_full | log1p_duration | logistic_regression | {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7658 | 12 | 12 | 0.4742 | complete |  | 0.7721 | 0.7795 | 0.7456 |
| d3opt_0053_c603c730a3 | 0.1000 | surface_surprisal_syntax | all_gaze | robust_full | log1p_duration | random_forest | {'n_estimators': 600, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7657 | 12 | 12 | 0.2717 | complete |  | 0.7416 | 0.8185 | 0.7369 |
| d3opt_0043_d9b707329a | 1.0000 | surface_surprisal_syntax | all_gaze | central_spread | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7656 | 12 | 12 | 0.5400 | complete |  | 0.8136 | 0.7486 | 0.7345 |
| d3opt_0046_f9179e50ce | 10.0000 | surface | duration_plus_count | central_spread | raw | logistic_regression | {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7655 | 12 | 12 | 0.5625 | complete |  | 0.8155 | 0.7462 | 0.7348 |
| d3opt_0021_31a9b65c57 | 1.0000 | surface_surprisal_syntax | all_gaze | robust_full | log1p_duration | extra_trees | {'n_estimators': 800, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced'} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7653 | 12 | 12 | 0.3042 | complete |  | 0.7477 | 0.8084 | 0.7398 |
| d3opt_0015_8485500590 | 1.0000 | surface | duration_plus_count | robust_full | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7652 | 12 | 12 | 0.4975 | complete |  | 0.7935 | 0.7697 | 0.7324 |
| d3opt_0001_a08ea0296c | 1.0000 | surface_surprisal_syntax | duration_core | central_spread | log1p_duration | logistic_regression | {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7650 | 12 | 12 | 0.5125 | complete |  | 0.8090 | 0.7410 | 0.7450 |
| d3opt_0002_936d309e0b | 100.0000 | surface_surprisal | duration_core | robust_full | log1p_duration | logistic_regression | {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 2000} | 20260522 | inner_validation_fold_mean_balanced_accuracy | 0.7638 | 12 | 12 | 0.4767 | complete |  | 0.7864 | 0.7649 | 0.7400 |
