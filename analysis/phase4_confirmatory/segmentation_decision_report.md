# Segmentation Decision Report

## Decision
- Category: `secondary_result`
- Standalone segmentation main-effect framing: `drop`
- Segmentation retained as a secondary interaction and interpretability feature.
- Pronunciation-aware labels are recommended only if the boundary-opacity interaction remains meaningful in controlled models.

## Boundary Opacity Beyond DFM Surprisal
| outcome | estimate | std_error | p_value | ci_low | ci_high | survives_controls |
| --- | --- | --- | --- | --- | --- | --- |
| log_ffd | -0.0031 | 0.0022 | 0.1530 | -0.0074 | 0.0012 | False |
| log_first_pass_duration | -0.0048 | 0.0029 | 0.0945 | -0.0105 | 0.0008 | False |
| log_go_past_time | -0.0052 | 0.0039 | 0.1907 | -0.0129 | 0.0026 | False |
| log_total_fixation_duration | -0.0077 | 0.0035 | 0.0297 | -0.0146 | -0.0008 | True |
| fixation_count | -0.0151 | 0.0045 | 0.0007 | -0.0239 | -0.0063 | True |

## Prediction Context
| feature_group | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score |
| --- | --- | --- | --- | --- | --- |
| F_segmentation_sensitivity_only | 0.6440 | 0.6666 | 0.6579 | 0.6599 | 0.2188 |
| D2_dfm_sensitivity_only | 0.8892 | 0.8611 | 0.8421 | 0.8421 | 0.1130 |
