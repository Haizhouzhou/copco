# DFM exposure-only and sensitivity/residual gaze comparisons.

| analysis | split_name | feature_group | model | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | calibration_intercept | calibration_slope | calibration_mean_predicted | calibration_observed_rate | status | skip_reason | fold_validity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase4_confirmatory_participant_prediction | leave_one_participant_out | D1_dfm_exposure_only | logistic_regression | 3 | 57 | 57 | 0 | 0.4238 | 0.3685 | 0.4474 | 0.4389 | 0.2684 | -0.7806 | -1.5077 | 0.4908 | 0.3333 | complete |  | all_test_predictions_generated |
| phase4_confirmatory_participant_prediction | leave_one_participant_out | D2_dfm_sensitivity_only | logistic_regression | 16 | 57 | 57 | 0 | 0.8892 | 0.8611 | 0.8421 | 0.8421 | 0.1130 | -0.5379 | 0.8076 | 0.3865 | 0.3333 | complete |  | all_test_predictions_generated |
| phase4_confirmatory_participant_prediction | leave_one_participant_out | D3_dfm_residual_gaze_only | logistic_regression | 12 | 57 | 57 | 0 | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 | -0.5321 | 0.8693 | 0.3924 | 0.3333 | complete |  | all_test_predictions_generated |
| phase4_confirmatory_participant_prediction | leave_one_participant_out | D4_dfm_exposure_plus_sensitivity | logistic_regression | 19 | 57 | 57 | 0 | 0.8726 | 0.8561 | 0.8158 | 0.8074 | 0.1206 | -0.5135 | 0.7106 | 0.3825 | 0.3333 | complete |  | all_test_predictions_generated |
