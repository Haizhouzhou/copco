# Confirmatory Prediction Report

Prediction is participant-level only. Exposure-count variables are excluded from every feature group; DFM exposure-only groups are retained only as explicit comparisons.

## Best Confirmatory Model
| split_name | feature_group | model | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | n_predictions | skipped_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_participant_out | D3_dfm_residual_gaze_only | logistic_regression | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 | 57 | 0 |

## Top Metric Rows
| split_name | feature_group | model | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | n_predictions | fold_validity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_participant_out | D3_dfm_residual_gaze_only | logistic_regression | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 | 57 | all_test_predictions_generated |
| leave_one_participant_out | D2_dfm_sensitivity_only | logistic_regression | 0.8892 | 0.8611 | 0.8421 | 0.8421 | 0.1130 | 57 | all_test_predictions_generated |
| leave_one_participant_out | D3_dfm_residual_gaze_only | linear_svm | 0.8864 | 0.8079 | 0.8421 | 0.8557 | 0.1315 | 57 | all_test_predictions_generated |
| leave_one_participant_out | I_all_except_segmentation | logistic_regression | 0.8767 | 0.8761 | 0.8026 | 0.7905 | 0.1365 | 57 | all_test_predictions_generated |
| leave_one_participant_out | D4_dfm_exposure_plus_sensitivity | logistic_regression | 0.8726 | 0.8561 | 0.8158 | 0.8074 | 0.1206 | 57 | all_test_predictions_generated |
| leave_one_participant_out | D2_dfm_sensitivity_only | linear_svm | 0.8657 | 0.8359 | 0.8421 | 0.8557 | 0.1318 | 57 | all_test_predictions_generated |
| leave_one_participant_out | D4_dfm_exposure_plus_sensitivity | linear_svm | 0.8601 | 0.8560 | 0.8421 | 0.8421 | 0.1344 | 57 | all_test_predictions_generated |
| leave_one_participant_out | B_residual_gaze | logistic_regression | 0.8490 | 0.7589 | 0.7500 | 0.7467 | 0.1609 | 57 | all_test_predictions_generated |
| leave_one_participant_out | J_all_except_raw_speed | logistic_regression | 0.8380 | 0.8338 | 0.7895 | 0.7856 | 0.1551 | 57 | all_test_predictions_generated |
| leave_one_participant_out | I_all_except_segmentation | linear_svm | 0.8366 | 0.8337 | 0.7763 | 0.7689 | 0.1563 | 57 | all_test_predictions_generated |
| leave_one_participant_out | A_raw_gaze | logistic_regression | 0.8338 | 0.7622 | 0.7632 | 0.7632 | 0.1608 | 57 | all_test_predictions_generated |
| leave_one_participant_out | C_sensitivity_slopes_only | logistic_regression | 0.8324 | 0.7986 | 0.7500 | 0.7361 | 0.1671 | 57 | all_test_predictions_generated |
| leave_one_participant_out | B_residual_gaze | linear_svm | 0.8241 | 0.7138 | 0.7368 | 0.7304 | 0.1712 | 57 | all_test_predictions_generated |
| leave_one_participant_out | G_all_allowed_non_exposure | logistic_regression | 0.8241 | 0.8152 | 0.7895 | 0.7738 | 0.1639 | 57 | all_test_predictions_generated |
| leave_one_participant_out | K_all_except_exposure_variables | logistic_regression | 0.8241 | 0.8152 | 0.7895 | 0.7738 | 0.1639 | 57 | all_test_predictions_generated |
| leave_one_participant_out | A_raw_gaze | linear_svm | 0.8172 | 0.7578 | 0.7500 | 0.7467 | 0.1778 | 57 | all_test_predictions_generated |
| leave_one_participant_out | J_all_except_raw_speed | linear_svm | 0.8006 | 0.7572 | 0.7632 | 0.7524 | 0.1865 | 57 | all_test_predictions_generated |
| leave_one_participant_out | C_sensitivity_slopes_only | linear_svm | 0.7770 | 0.7345 | 0.6974 | 0.6826 | 0.1965 | 57 | all_test_predictions_generated |
| leave_one_participant_out | G_all_allowed_non_exposure | linear_svm | 0.7756 | 0.7618 | 0.7632 | 0.7524 | 0.1798 | 57 | all_test_predictions_generated |
| leave_one_participant_out | K_all_except_exposure_variables | linear_svm | 0.7756 | 0.7618 | 0.7632 | 0.7524 | 0.1798 | 57 | all_test_predictions_generated |
