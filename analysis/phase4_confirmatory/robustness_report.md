# Robustness Report

Robustness is computed for the selected confirmatory LOPO model. Label permutation shuffles participant labels only; no word-level random split is used.

## Selected Model
| split_name | feature_group | model | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | n_predictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_participant_out | D3_dfm_residual_gaze_only | logistic_regression | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 | 57 |

## Permutation And Bootstrap
| observed_roc_auc | valid_permutations | permutation_p_value | bootstrap_roc_auc_low | bootstrap_roc_auc_high | leave_one_dyslexia_min_roc_auc |
| --- | --- | --- | --- | --- | --- |
| 0.8947 | 1000 | 0.0010 | 0.7765 | 0.9841 | 0.8801 |

## Required Sensitivity Rows
| feature_group | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score |
| --- | --- | --- | --- | --- | --- |
| D1_dfm_exposure_only | 0.4238 | 0.3685 | 0.4474 | 0.4389 | 0.2684 |
| D2_dfm_sensitivity_only | 0.8892 | 0.8611 | 0.8421 | 0.8421 | 0.1130 |
| D3_dfm_residual_gaze_only | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 |
| D4_dfm_exposure_plus_sensitivity | 0.8726 | 0.8561 | 0.8158 | 0.8074 | 0.1206 |
| J_all_except_raw_speed | 0.8380 | 0.8338 | 0.7895 | 0.7856 | 0.1551 |
| K_all_except_exposure_variables | 0.8241 | 0.8152 | 0.7895 | 0.7738 | 0.1639 |
