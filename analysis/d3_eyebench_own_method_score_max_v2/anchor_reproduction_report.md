# Anchor Reproduction Report

- Candidate: `candidate_0000` exact previous `D3_EyeBench_Lite` adapter
- Reproduction status: `passed`
- Tolerance: `0.001`
- Held-out reader leakage: False
- Held-out text leakage: False
- Reader group used in residualization: False

## Expected vs Actual
| split_name | metric | expected | actual | delta | tolerance | passed |
| --- | --- | --- | --- | --- | --- | --- |
| unseen_reader | balanced_accuracy | 0.7274 | 0.7274 | 0.0000 | 0.0010 | True |
| unseen_reader | roc_auc | 0.8085 | 0.8085 | 0.0000 | 0.0010 | True |
| unseen_text | balanced_accuracy | 0.7341 | 0.7341 | -0.0000 | 0.0010 | True |
| unseen_text | roc_auc | 0.8319 | 0.8319 | -0.0000 | 0.0010 | True |
| unseen_reader_and_text | balanced_accuracy | 0.6342 | 0.6342 | -0.0000 | 0.0010 | True |
| unseen_reader_and_text | roc_auc | 0.7154 | 0.7154 | -0.0000 | 0.0010 | True |

## Candidate Metrics
| candidate_id | family | feature_recipe | model_type | threshold_method | calibration_method | split_name | evaluation_level | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | sensitivity | specificity | brier_score | ece | threshold | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | official_trial_level_fold_mean | 12 | 3554 | 4 | 0 | 0.8085 | 0.5614 | 0.7274 | 0.6767 |  |  | 0.1904 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | reader_aggregated | 12 | 55 | 4 | 0 | 0.8468 | 0.7334 | 0.7530 | 0.7418 |  |  | 0.1783 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | official_trial_level_fold_mean | 12 | 3554 | 4 | 0 | 0.8319 | 0.5434 | 0.7341 | 0.6751 |  |  | 0.1871 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | reader_aggregated | 12 | 113 | 4 | 0 | 0.8606 | 0.6722 | 0.7629 | 0.7268 |  |  | 0.1673 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | official_trial_level_fold_mean | 12 | 1228 | 4 | 0 | 0.7154 | 0.5650 | 0.6342 | 0.6223 |  |  | 0.2191 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | reader_aggregated | 12 | 39 | 4 | 0 | 0.7792 | 0.6482 | 0.6201 | 0.6201 |  |  | 0.1893 |  | 0.5000 | complete |  |
