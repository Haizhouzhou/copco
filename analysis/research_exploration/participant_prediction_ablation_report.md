# Participant Prediction Ablation Report

Primary prediction uses participant-level rows only. Exposure-count variables such as `n_words_read`, `n_speeches`, `n_word_rows`, and `word_observation_count` are excluded from primary feature sets.

## Top Metric Rows
| split_name | feature_group | model | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | n_predictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_participant_out | D_dfm_exposure_and_sensitivity | logistic_regression | 0.9058 | 0.8686 | 0.8816 | 0.8667 | 0.1080 | 57 |
| leave_one_participant_out | D_dfm_exposure_and_sensitivity | linear_svm | 0.9044 | 0.8671 | 0.8553 | 0.8459 | 0.1369 | 57 |
| participant_grouped_kfold | D_dfm_exposure_and_sensitivity | logistic_regression | 0.9044 | 0.8805 | 0.8816 | 0.8667 | 0.1166 | 57 |
| participant_grouped_kfold | D_dfm_exposure_and_sensitivity | linear_svm | 0.9017 | 0.8805 | 0.8553 | 0.8459 | 0.1403 | 57 |
| leave_one_participant_out | I_no_raw_speed_or_exposure_count_features | random_forest | 0.8947 | 0.8818 | 0.8421 | 0.8557 | 0.1071 | 57 |
| leave_one_participant_out | G_all_except_segmentation | random_forest | 0.8906 | 0.8765 | 0.8421 | 0.8421 | 0.1122 | 57 |
| participant_grouped_kfold | F_all_non_leakage_features | random_forest | 0.8864 | 0.8784 | 0.8158 | 0.8074 | 0.1194 | 57 |
| leave_one_participant_out | I_no_raw_speed_or_exposure_count_features | logistic_regression | 0.8864 | 0.8824 | 0.8289 | 0.8246 | 0.1140 | 57 |
| participant_grouped_kfold | G_all_except_segmentation | random_forest | 0.8837 | 0.8695 | 0.8158 | 0.8074 | 0.1210 | 57 |
| leave_one_participant_out | B_residual_gaze_aggregates | linear_svm | 0.8809 | 0.8127 | 0.7763 | 0.7799 | 0.1633 | 57 |

## Robustness Summary
| selected_feature_group | selected_model | selected_split | observed_roc_auc | permutation_p_value | bootstrap_roc_auc_low | bootstrap_roc_auc_high | leave_one_dyslexia_min_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D_dfm_exposure_and_sensitivity | logistic_regression | leave_one_participant_out | 0.9058 | 0.0099 | 0.8162 | 0.9798 | 0.8977 |

Prediction is exploratory and should not be interpreted as screening. Phase 3 evaluates which signal families deserve deeper controlled analysis.
