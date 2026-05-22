# Calibration And Threshold Report

- `candidate_0000` uses the exact previous fixed `0.5` threshold.
- New candidate thresholds are either fixed `0.5` or selected on train/inner-validation folds.
- Test labels are not used for threshold selection.

## Inner-Validation Threshold Rows
| candidate_id | family | feature_recipe | model_type | threshold_method | calibration_method | split_name | fold_id | n_features | n_inner_train | n_inner_val | threshold | threshold_selection_metric | status | skip_reason | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | sensitivity | specificity | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 0 | 47 | 883 | 298 | 0.5000 |  | complete |  | 0.8884 | 0.6598 | 0.8349 | 0.8094 | 0.1240 | 0.8133 | 0.8565 | 0.0835 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 1 | 47 | 825 | 342 | 0.5000 |  | complete |  | 0.7990 | 0.6353 | 0.6313 | 0.5072 | 0.3284 | 0.8481 | 0.4144 | 0.3601 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 2 | 47 | 928 | 326 | 0.5000 |  | complete |  | 0.8793 | 0.8787 | 0.8364 | 0.8359 | 0.1339 | 0.7516 | 0.9212 | 0.0982 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 3 | 47 | 824 | 305 | 0.5000 |  | complete |  | 0.7859 | 0.6409 | 0.7048 | 0.6761 | 0.1892 | 0.7045 | 0.7051 | 0.1630 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 0 | 47 | 653 | 528 | 0.5000 |  | complete |  | 0.8985 | 0.6838 | 0.8212 | 0.7424 | 0.1365 | 0.8438 | 0.7986 | 0.1767 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 1 | 47 | 650 | 517 | 0.5000 |  | complete |  | 0.7537 | 0.7021 | 0.7015 | 0.7064 | 0.1823 | 0.5722 | 0.8309 | 0.1008 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 2 | 47 | 692 | 562 | 0.5000 |  | complete |  | 0.8046 | 0.7256 | 0.7404 | 0.7163 | 0.1945 | 0.7457 | 0.7352 | 0.1818 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 3 | 47 | 629 | 500 | 0.5000 |  | complete |  | 0.8094 | 0.5324 | 0.7389 | 0.6090 | 0.2469 | 0.8636 | 0.6141 | 0.3076 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 0 | 47 | 883 | 298 | 0.5000 |  | complete |  | 0.8943 | 0.8320 | 0.8160 | 0.7957 | 0.1147 | 0.7821 | 0.8500 | 0.0980 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 1 | 47 | 872 | 295 | 0.5000 |  | complete |  | 0.9432 | 0.8555 | 0.8605 | 0.8391 | 0.0913 | 0.8519 | 0.8692 | 0.0936 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 2 | 47 | 939 | 315 | 0.5000 |  | complete |  | 0.9257 | 0.7540 | 0.8646 | 0.8397 | 0.1073 | 0.8830 | 0.8462 | 0.1090 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 3 | 47 | 843 | 286 | 0.5000 |  | complete |  | 0.8608 | 0.6614 | 0.7735 | 0.7322 | 0.1510 | 0.7681 | 0.7788 | 0.1508 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 0 | 47 | 883 | 298 | 0.5000 |  | complete |  | 0.8772 | 0.6407 | 0.8015 | 0.7795 | 0.1301 | 0.7600 | 0.8430 | 0.0796 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 1 | 47 | 825 | 342 | 0.5000 |  | complete |  | 0.7913 | 0.6228 | 0.6313 | 0.5072 | 0.3469 | 0.8481 | 0.4144 | 0.3711 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 2 | 47 | 928 | 326 | 0.5000 |  | complete |  | 0.8851 | 0.8794 | 0.8302 | 0.8297 | 0.1312 | 0.7453 | 0.9152 | 0.1004 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader | 3 | 47 | 824 | 305 | 0.5000 |  | complete |  | 0.7835 | 0.6232 | 0.7048 | 0.6761 | 0.1943 | 0.7045 | 0.7051 | 0.1610 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 0 | 47 | 653 | 528 | 0.5000 |  | complete |  | 0.8923 | 0.6659 | 0.8113 | 0.7318 | 0.1414 | 0.8333 | 0.7894 | 0.1743 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 1 | 47 | 650 | 517 | 0.5000 |  | complete |  | 0.6885 | 0.6567 | 0.6765 | 0.6823 | 0.2069 | 0.5222 | 0.8309 | 0.1506 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 2 | 47 | 692 | 562 | 0.5000 |  | complete |  | 0.8115 | 0.7281 | 0.7417 | 0.7179 | 0.1930 | 0.7457 | 0.7378 | 0.1821 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_reader_and_text | 3 | 47 | 629 | 500 | 0.5000 |  | complete |  | 0.8135 | 0.5437 | 0.7445 | 0.6119 | 0.2490 | 0.8750 | 0.6141 | 0.2998 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 0 | 47 | 883 | 298 | 0.5000 |  | complete |  | 0.8900 | 0.8294 | 0.8115 | 0.7890 | 0.1172 | 0.7821 | 0.8409 | 0.1071 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 1 | 47 | 872 | 295 | 0.5000 |  | complete |  | 0.9436 | 0.8577 | 0.8714 | 0.8507 | 0.0898 | 0.8642 | 0.8785 | 0.0860 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 2 | 47 | 939 | 315 | 0.5000 |  | complete |  | 0.9223 | 0.7488 | 0.8638 | 0.8421 | 0.1070 | 0.8723 | 0.8552 | 0.0952 |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | fixed_0_5 | none | unseen_text | 3 | 47 | 843 | 286 | 0.5000 |  | complete |  | 0.8613 | 0.6683 | 0.8024 | 0.7522 | 0.1552 | 0.8261 | 0.7788 | 0.1560 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 0 | 36 | 883 | 298 | 0.2900 | 0.5626 | complete |  | 0.5178 | 0.2596 | 0.5626 | 0.5294 | 0.1988 | 0.5467 | 0.5785 | 0.0850 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 1 | 36 | 825 | 342 | 0.3000 | 0.7688 | complete |  | 0.8434 | 0.6487 | 0.7688 | 0.6725 | 0.1356 | 0.8987 | 0.6388 | 0.1077 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 2 | 36 | 928 | 326 | 0.1500 | 0.8043 | complete |  | 0.8827 | 0.8990 | 0.8043 | 0.8033 | 0.1848 | 0.8571 | 0.7515 | 0.1842 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 3 | 36 | 824 | 305 | 0.3000 | 0.7053 | complete |  | 0.7184 | 0.5492 | 0.7053 | 0.6930 | 0.1738 | 0.6364 | 0.7742 | 0.0388 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader_and_text | 0 | 36 | 653 | 528 | 0.4100 | 0.6968 | complete |  | 0.7090 | 0.5026 | 0.6968 | 0.7074 | 0.1571 | 0.4792 | 0.9144 | 0.1910 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader_and_text | 1 | 36 | 650 | 517 | 0.1500 | 0.7141 | complete |  | 0.7852 | 0.6486 | 0.7141 | 0.6693 | 0.1923 | 0.8556 | 0.5727 | 0.1095 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader_and_text | 2 | 36 | 692 | 562 | 0.2600 | 0.8037 | complete |  | 0.8985 | 0.8299 | 0.8037 | 0.7783 | 0.1245 | 0.8208 | 0.7866 | 0.0854 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader_and_text | 3 | 36 | 629 | 500 | 0.2300 | 0.7323 | complete |  | 0.8047 | 0.4835 | 0.7323 | 0.6097 | 0.1435 | 0.8409 | 0.6238 | 0.1347 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_text | 0 | 36 | 883 | 298 | 0.2400 | 0.7807 | complete |  | 0.8612 | 0.7447 | 0.7807 | 0.7322 | 0.1296 | 0.8205 | 0.7409 | 0.0823 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_text | 1 | 36 | 872 | 295 | 0.3200 | 0.8124 | complete |  | 0.8821 | 0.7678 | 0.8124 | 0.7868 | 0.1197 | 0.8025 | 0.8224 | 0.0590 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_text | 2 | 36 | 939 | 315 | 0.3300 | 0.8334 | complete |  | 0.9138 | 0.8421 | 0.8334 | 0.8134 | 0.1120 | 0.8298 | 0.8371 | 0.0638 |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | d3_lite_robust | random_forest_d3 | inner_balanced_accuracy | sigmoid_cv3 | unseen_text | 3 | 36 | 843 | 286 | 0.2400 | 0.7787 | complete |  | 0.8636 | 0.7231 | 0.7787 | 0.7665 | 0.1275 | 0.6957 | 0.8618 | 0.0726 |
| candidate_0004_3f1cbc883e | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 0 | 47 | 883 | 298 | 0.2700 | 0.6761 | complete |  | 0.7012 | 0.4060 | 0.6761 | 0.6222 | 0.1767 | 0.7200 | 0.6323 | 0.0678 |
| candidate_0004_3f1cbc883e | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 1 | 47 | 825 | 342 | 0.5600 | 0.7677 | complete |  | 0.8241 | 0.6985 | 0.7677 | 0.7665 | 0.1633 | 0.6456 | 0.8897 | 0.1893 |
| candidate_0004_3f1cbc883e | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 2 | 47 | 928 | 326 | 0.2200 | 0.8589 | complete |  | 0.8813 | 0.8874 | 0.8589 | 0.8589 | 0.2088 | 0.8571 | 0.8606 | 0.2263 |
| candidate_0004_3f1cbc883e | d3_lite_plus_raw_gaze | d3_lite_raw_summary | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | 3 | 47 | 824 | 305 | 0.2200 | 0.7145 | complete |  | 0.7759 | 0.6099 | 0.7145 | 0.7048 | 0.1919 | 0.6364 | 0.7926 | 0.1158 |
| ... |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Test Threshold Summary
| candidate_id | split_name | evaluation_level | threshold | balanced_accuracy | roc_auc | brier_score | ece |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | unseen_reader | official_trial_level_fold_mean | 0.5000 | 0.7274 | 0.8085 | 0.1904 |  |
| candidate_0000 | unseen_text | official_trial_level_fold_mean | 0.5000 | 0.7341 | 0.8319 | 0.1871 |  |
| candidate_0000 | unseen_reader_and_text | official_trial_level_fold_mean | 0.5000 | 0.6342 | 0.7154 | 0.2191 |  |
| candidate_0014_3a7538097b | unseen_reader | official_trial_level_fold_mean | 0.3050 | 0.6825 | 0.7285 | 0.1491 | 0.0805 |
| candidate_0014_3a7538097b | unseen_text | official_trial_level_fold_mean | 0.3075 | 0.7765 | 0.8557 | 0.1316 | 0.1386 |
| candidate_0014_3a7538097b | unseen_reader_and_text | official_trial_level_fold_mean | 0.3450 | 0.5612 | 0.6420 | 0.1961 | 0.1206 |
| candidate_0013_936f0c9788 | unseen_reader | official_trial_level_fold_mean | 0.3025 | 0.6796 | 0.7309 | 0.1487 | 0.0847 |
| candidate_0013_936f0c9788 | unseen_text | official_trial_level_fold_mean | 0.3175 | 0.7779 | 0.8541 | 0.1323 | 0.1380 |
| candidate_0013_936f0c9788 | unseen_reader_and_text | official_trial_level_fold_mean | 0.3500 | 0.5762 | 0.6480 | 0.1952 | 0.1166 |
| candidate_0011_c51e744a96 | unseen_reader | official_trial_level_fold_mean | 0.2975 | 0.6731 | 0.7178 | 0.1503 | 0.0806 |
| candidate_0011_c51e744a96 | unseen_text | official_trial_level_fold_mean | 0.2975 | 0.7742 | 0.8566 | 0.1313 | 0.1392 |
| candidate_0011_c51e744a96 | unseen_reader_and_text | official_trial_level_fold_mean | 0.3350 | 0.5589 | 0.6302 | 0.1991 | 0.1259 |
| candidate_0009_89e1a1f5ed | unseen_reader | official_trial_level_fold_mean | 0.3775 | 0.6626 | 0.7771 | 0.1545 | 0.1076 |
| candidate_0009_89e1a1f5ed | unseen_text | official_trial_level_fold_mean | 0.3500 | 0.7973 | 0.9054 | 0.1149 | 0.1126 |
| candidate_0009_89e1a1f5ed | unseen_reader_and_text | official_trial_level_fold_mean | 0.4375 | 0.5735 | 0.6467 | 0.2079 | 0.1248 |
| candidate_0004_3f1cbc883e | unseen_reader | official_trial_level_fold_mean | 0.3175 | 0.6978 | 0.7522 | 0.1464 | 0.1172 |
| candidate_0004_3f1cbc883e | unseen_text | official_trial_level_fold_mean | 0.2925 | 0.7794 | 0.8451 | 0.1370 | 0.1411 |
| candidate_0004_3f1cbc883e | unseen_reader_and_text | official_trial_level_fold_mean | 0.3275 | 0.5460 | 0.6686 | 0.1988 | 0.1557 |
| candidate_0010_99ae4c18ef | unseen_reader | official_trial_level_fold_mean | 0.2600 | 0.6719 | 0.7640 | 0.1533 | 0.0984 |
| candidate_0010_99ae4c18ef | unseen_text | official_trial_level_fold_mean | 0.3125 | 0.7820 | 0.9056 | 0.1160 | 0.1220 |
| candidate_0010_99ae4c18ef | unseen_reader_and_text | official_trial_level_fold_mean | 0.2975 | 0.5594 | 0.6359 | 0.2130 | 0.1429 |
