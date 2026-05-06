# CopCo RCS Official Alignment Report

RCS is auxiliary. The official EyeBench target is `RCS_score`; the BenchmarkBridge full-data reference used the frozen project comprehension score, so cross-mode scale comparisons should emphasize R2.

| mode | model_name | claim_type | split_name | target_scale | n_predictions | rmse | mae | r2 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_eyebench_subset | D3_EyeBench_Lite | official_attempt_failed | unseen_reader | EyeBench_RCS_score | 0 |  |  |  | skipped |
| official_eyebench_subset | D3_EyeBench_Lite | official_attempt_failed | unseen_text | EyeBench_RCS_score | 0 |  |  |  | skipped |
| official_eyebench_subset | D3_EyeBench_Lite | official_attempt_failed | unseen_reader_and_text | EyeBench_RCS_score | 0 |  |  |  | skipped |
| eyebench_folds_full_feature_intersection | D3_FullFeature_EyeBenchFolds | EyeBench-fold-aligned_full-feature_non-official | unseen_reader | EyeBench_RCS_score | 44 | 1.5295 | 1.2230 | 0.0998 | complete |
| eyebench_folds_full_feature_intersection | D3_FullFeature_EyeBenchFolds | EyeBench-fold-aligned_full-feature_non-official | unseen_text | EyeBench_RCS_score | 90 | 1.5905 | 1.2834 | 0.1014 | complete |
| eyebench_folds_full_feature_intersection | D3_FullFeature_EyeBenchFolds | EyeBench-fold-aligned_full-feature_non-official | unseen_reader_and_text | EyeBench_RCS_score | 30 | 1.5116 | 1.1690 | 0.1130 | complete |
| full_data_eyebench_style | D3_FullData_EyeBenchStyle | internal_EyeBench-style_benchmark-relative | unseen_reader | raw_project_scale_0_1; EyeBench-compatible 1-10 = 1 + 9 * raw | 57 | 0.1374 | 0.1065 | 0.0031 | complete |
| full_data_eyebench_style | D3_FullData_EyeBenchStyle | internal_EyeBench-style_benchmark-relative | unseen_text | raw_project_scale_0_1; EyeBench-compatible 1-10 = 1 + 9 * raw | 242 | 0.1196 | 0.0904 | 0.0116 | complete |
| full_data_eyebench_style | D3_FullData_EyeBenchStyle | internal_EyeBench-style_benchmark-relative | unseen_reader_and_text | raw_project_scale_0_1; EyeBench-compatible 1-10 = 1 + 9 * raw | 34 | 0.1524 | 0.1211 | -0.0164 | complete |
