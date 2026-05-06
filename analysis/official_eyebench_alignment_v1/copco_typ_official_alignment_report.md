# CopCo TYP Official Alignment Report

Rows distinguish official-subset attempts, EyeBench-fold full-feature results, and BenchmarkBridge full-data internal results.

| mode | model_name | claim_type | split_name | n_predictions | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_eyebench_subset | D3_EyeBench_Lite | official_attempt_failed | unseen_reader | 0 |  |  |  |  |  | skipped |
| official_eyebench_subset | D3_EyeBench_Lite | official_attempt_failed | unseen_text | 0 |  |  |  |  |  | skipped |
| official_eyebench_subset | D3_EyeBench_Lite | official_attempt_failed | unseen_reader_and_text | 0 |  |  |  |  |  | skipped |
| eyebench_folds_full_feature_intersection | D3_FullFeature_EyeBenchFolds | EyeBench-fold-aligned_full-feature_non-official | unseen_reader | 55 | 0.8123 | 0.7772 | 0.7387 | 0.7353 | 0.2016 | complete |
| eyebench_folds_full_feature_intersection | D3_FullFeature_EyeBenchFolds | EyeBench-fold-aligned_full-feature_non-official | unseen_text | 113 | 0.8141 | 0.6640 | 0.6976 | 0.6869 | 0.2057 | complete |
| eyebench_folds_full_feature_intersection | D3_FullFeature_EyeBenchFolds | EyeBench-fold-aligned_full-feature_non-official | unseen_reader_and_text | 39 | 0.7240 | 0.6066 | 0.7110 | 0.6991 | 0.2117 | complete |
| full_data_eyebench_style | D3_FullData_EyeBenchStyle | internal_EyeBench-style_benchmark-relative | unseen_reader | 57 | 0.8961 | 0.8738 | 0.8158 | 0.8074 | 0.1326 | complete |
| full_data_eyebench_style | D3_FullData_EyeBenchStyle | internal_EyeBench-style_benchmark-relative | unseen_text | 242 | 0.8285 | 0.5548 | 0.7444 | 0.6935 | 0.1676 | complete |
| full_data_eyebench_style | D3_FullData_EyeBenchStyle | internal_EyeBench-style_benchmark-relative | unseen_reader_and_text | 34 | 0.8542 | 0.7740 | 0.7458 | 0.7312 | 0.1501 | complete |
