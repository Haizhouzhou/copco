# DFM Exposure Vs Sensitivity Report

The Phase 3 combined DFM group is decomposed into explicit exposure-only, sensitivity-only, residual-gaze-only, and combined comparison groups.

## Group Summary
| feature_group | n_features | could_encode_text_assignment | allowed_primary_features | sensitivity_only |
| --- | --- | --- | --- | --- |
| A_raw_gaze | 11 | False | 0 | False |
| B_residual_gaze | 22 | False | 22 | False |
| C_sensitivity_slopes_only | 45 | False | 45 | True |
| D1_dfm_exposure_only | 3 | True | 0 | False |
| D2_dfm_sensitivity_only | 16 | False | 16 | True |
| D3_dfm_residual_gaze_only | 12 | False | 12 | True |
| D4_dfm_exposure_plus_sensitivity | 19 | True | 0 | False |
| E_segmentation_exposure_only | 2 | True | 0 | False |
| F_segmentation_sensitivity_only | 21 | False | 21 | True |
| G_all_allowed_non_exposure | 80 | False | 80 | False |
| H_all_except_dfm | 70 | True | 0 | False |
| I_all_except_segmentation | 65 | True | 0 | False |
| J_all_except_raw_speed | 79 | True | 71 | False |
| K_all_except_exposure_variables | 80 | False | 80 | False |

## Interpretation Rules
- `D1_dfm_exposure_only` can encode text assignment and is not a primary publication model.
- `D2_dfm_sensitivity_only` and `D3_dfm_residual_gaze_only` are sensitivity-only DFM groups.
- `D4_dfm_exposure_plus_sensitivity` reproduces the combined Phase 3 family as a comparison group.
- Exposure-count variables remain flagged and excluded from all feature groups.
