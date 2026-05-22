# Best Candidate Report

- Best candidate: `candidate_0000`
- Decision category: `d3_method_not_improved`
- Internal simple mean BA: 0.6985462748783746
- Internal simple mean AUROC: 0.7852477071692444
- Delta vs candidate_0000 BA: 0.0
- Official SOTA claimed: False

## Trial Metrics
| candidate_id | family | feature_recipe | model_type | threshold_method | calibration_method | split_name | evaluation_level | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | sensitivity | specificity | brier_score | ece | threshold | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | official_trial_level_fold_mean | 12 | 3554 | 4 | 0 | 0.8085 | 0.5614 | 0.7274 | 0.6767 |  |  | 0.1904 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | official_trial_level_fold_mean | 12 | 3554 | 4 | 0 | 0.8319 | 0.5434 | 0.7341 | 0.6751 |  |  | 0.1871 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | official_trial_level_fold_mean | 12 | 1228 | 4 | 0 | 0.7154 | 0.5650 | 0.6342 | 0.6223 |  |  | 0.2191 |  | 0.5000 | complete |  |

## Reader-Aggregated Metrics
| candidate_id | family | feature_recipe | model_type | threshold_method | calibration_method | split_name | evaluation_level | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | sensitivity | specificity | brier_score | ece | threshold | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | reader_aggregated | 12 | 55 | 4 | 0 | 0.8468 | 0.7334 | 0.7530 | 0.7418 |  |  | 0.1783 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | reader_aggregated | 12 | 113 | 4 | 0 | 0.8606 | 0.6722 | 0.7629 | 0.7268 |  |  | 0.1673 |  | 0.5000 | complete |  |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | reader_aggregated | 12 | 39 | 4 | 0 | 0.7792 | 0.6482 | 0.6201 | 0.6201 |  |  | 0.1893 |  | 0.5000 | complete |  |

## Fold-Level Metrics
| candidate_id | family | feature_recipe | model_type | threshold_method | calibration_method | split_name | fold_id | evaluation_level | n_predictions | n_features | threshold | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | sensitivity | specificity | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | 0 | official_trial_level_fold_mean | 884 | 12 | 0.5000 | 0.8357 | 0.6427 | 0.7321 | 0.7176 | 0.1622 | 0.6368 | 0.8274 | 0.1785 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | 1 | official_trial_level_fold_mean | 906 | 12 | 0.5000 | 0.7024 | 0.4819 | 0.6230 | 0.5773 | 0.2200 | 0.6078 | 0.6382 | 0.2333 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | 2 | official_trial_level_fold_mean | 799 | 12 | 0.5000 | 0.8897 | 0.6577 | 0.8130 | 0.6828 | 0.1884 | 0.8934 | 0.7326 | 0.2596 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | 3 | official_trial_level_fold_mean | 965 | 12 | 0.5000 | 0.8063 | 0.4634 | 0.7415 | 0.7289 | 0.1911 | 0.6419 | 0.8410 | 0.2088 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | 0 | official_trial_level_fold_mean | 321 | 12 | 0.5000 | 0.7587 | 0.5004 | 0.7003 | 0.6828 | 0.1867 | 0.6173 | 0.7833 | 0.1643 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | 1 | official_trial_level_fold_mean | 269 | 12 | 0.5000 | 0.5922 | 0.5463 | 0.4857 | 0.4493 | 0.2965 | 0.5333 | 0.4381 | 0.2955 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | 2 | official_trial_level_fold_mean | 404 | 12 | 0.5000 | 0.7790 | 0.6882 | 0.7120 | 0.7127 | 0.1960 | 0.6629 | 0.7611 | 0.0810 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | 3 | official_trial_level_fold_mean | 234 | 12 | 0.5000 | 0.7317 | 0.5253 | 0.6386 | 0.6445 | 0.1971 | 0.4459 | 0.8313 | 0.0954 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | 0 | official_trial_level_fold_mean | 913 | 12 | 0.5000 | 0.8644 | 0.6466 | 0.7681 | 0.7548 | 0.1505 | 0.6716 | 0.8646 | 0.1719 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | 1 | official_trial_level_fold_mean | 861 | 12 | 0.5000 | 0.8269 | 0.5532 | 0.7401 | 0.6795 | 0.1859 | 0.7486 | 0.7316 | 0.2414 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | 2 | official_trial_level_fold_mean | 797 | 12 | 0.5000 | 0.8386 | 0.4082 | 0.7431 | 0.5862 | 0.2387 | 0.8519 | 0.6343 | 0.3160 |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | 3 | official_trial_level_fold_mean | 983 | 12 | 0.5000 | 0.7974 | 0.5655 | 0.6851 | 0.6797 | 0.1735 | 0.5699 | 0.8003 | 0.1127 |
