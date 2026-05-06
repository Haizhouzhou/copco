# Feature Stability Report

Standardized logistic coefficients are recomputed inside LOPO folds for the selected confirmatory feature group.

## Selected Model
| feature_group | model | split_name | roc_auc | pr_auc | n_predictions |
| --- | --- | --- | --- | --- | --- |
| D3_dfm_residual_gaze_only | logistic_regression | leave_one_participant_out | 0.8947 | 0.8641 | 57 |

## Top Stable Positive Features
| feature | mean_coefficient | sign_stability | positive_rate | n_folds |
| --- | --- | --- | --- | --- |
| crossfit_total_fixation_residual_dfm_entropy_slope | 1.0863 | 1.0000 | 1.0000 | 57 |
| crossfit_first_pass_residual_dfm_entropy_slope | 0.9849 | 1.0000 | 1.0000 | 57 |
| crossfit_go_past_residual_dfm_surprisal_slope | 0.9177 | 1.0000 | 1.0000 | 57 |
| crossfit_fixation_count_residual_dfm_surprisal_slope | 0.6404 | 1.0000 | 1.0000 | 57 |
| crossfit_total_fixation_residual_dfm_surprisal_slope | 0.5938 | 1.0000 | 1.0000 | 57 |
| crossfit_go_past_residual_dfm_entropy_slope | 0.1970 | 1.0000 | 1.0000 | 57 |
| crossfit_skipping_residual_dfm_entropy_slope | 0.1596 | 1.0000 | 1.0000 | 57 |
| crossfit_skipping_residual_dfm_surprisal_slope | 0.1298 | 0.9825 | 0.9825 | 57 |
| crossfit_first_pass_residual_dfm_surprisal_slope | 0.0911 | 0.9474 | 0.9474 | 57 |

## Top Stable Negative Features
| feature | mean_coefficient | sign_stability | negative_rate | n_folds |
| --- | --- | --- | --- | --- |
| crossfit_ffd_residual_dfm_entropy_slope | -0.5886 | 1.0000 | 1.0000 | 57 |
| crossfit_fixation_count_residual_dfm_entropy_slope | -0.1904 | 0.9825 | 0.9825 | 57 |
| crossfit_ffd_residual_dfm_surprisal_slope | -0.1229 | 0.9474 | 0.9474 | 57 |

## Unstable Features
_No rows._

## Stability Answers
- DFM sensitivity features stable: True
- Segmentation sensitivity features stable: False
- Raw speed dominates selected model: False
