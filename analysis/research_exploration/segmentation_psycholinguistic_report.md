# Segmentation Psycholinguistic Report

- Model rows sampled deterministically: 100000 of 335203
- Models use participant-clustered standard errors when fit succeeds.
- Speech fixed effects are included as controls.

## Boundary Terms
| outcome | model_family | term | estimate | std_error | p_value | ci_low | ci_high | n_obs | model_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skip | opacity_controlled | prev_boundary_opacity_score_z | 0.0305 | 0.0433 | 0.3333 | -0.0159 | 0.1192 | 100000 | regularized_logistic_regression |
| skip | opacity_controlled | vv_indicator_z | 0.0100 | 0.0194 | 0.5000 | -0.0355 | 0.0228 | 100000 | regularized_logistic_regression |
| skip | opacity_controlled | vocoid_run_cross_boundary_z | 0.0098 | 0.0400 | 0.6667 | -0.0661 | 0.0586 | 100000 | regularized_logistic_regression |
| log_ffd | opacity_controlled | prev_boundary_opacity_score_z | 0.0036 | 0.0076 | 0.6667 | -0.0093 | 0.0126 | 62306 | ridge_linear_regression |
| log_ffd | opacity_controlled | vv_indicator_z | -0.0028 | 0.0036 | 0.5000 | -0.0085 | 0.0030 | 62306 | ridge_linear_regression |
| log_ffd | opacity_controlled | vocoid_run_cross_boundary_z | -0.0031 | 0.0072 | 0.6667 | -0.0118 | 0.0085 | 62306 | ridge_linear_regression |
| log_first_pass_duration | opacity_controlled | prev_boundary_opacity_score_z | 0.0048 | 0.0058 | 0.5000 | -0.0062 | 0.0129 | 62306 | ridge_linear_regression |
| log_first_pass_duration | opacity_controlled | vv_indicator_z | -0.0020 | 0.0049 | 0.6667 | -0.0075 | 0.0081 | 62306 | ridge_linear_regression |
| log_first_pass_duration | opacity_controlled | vocoid_run_cross_boundary_z | -0.0054 | 0.0053 | 0.3333 | -0.0133 | 0.0009 | 62306 | ridge_linear_regression |
| log_go_past_time | opacity_controlled | prev_boundary_opacity_score_z | -0.0078 | 0.0140 | 0.6667 | -0.0298 | 0.0138 | 62306 | ridge_linear_regression |
| log_go_past_time | opacity_controlled | vv_indicator_z | 0.0037 | 0.0052 | 0.1667 | -0.0008 | 0.0160 | 62306 | ridge_linear_regression |
| log_go_past_time | opacity_controlled | vocoid_run_cross_boundary_z | 0.0027 | 0.0127 | 0.8333 | -0.0179 | 0.0171 | 62306 | ridge_linear_regression |
| log_total_fixation_duration | opacity_controlled | prev_boundary_opacity_score_z | 0.0054 | 0.0123 | 1.0000 | -0.0122 | 0.0244 | 62306 | ridge_linear_regression |
| log_total_fixation_duration | opacity_controlled | vv_indicator_z | 0.0015 | 0.0055 | 0.8333 | -0.0078 | 0.0094 | 62306 | ridge_linear_regression |
| log_total_fixation_duration | opacity_controlled | vocoid_run_cross_boundary_z | -0.0102 | 0.0140 | 0.6667 | -0.0338 | 0.0089 | 62306 | ridge_linear_regression |
| fixation_count | opacity_controlled | prev_boundary_opacity_score_z | -0.0200 | 0.0238 | 0.3333 | -0.0554 | 0.0142 | 100000 | ridge_linear_regression |
| fixation_count | opacity_controlled | vv_indicator_z | -0.0011 | 0.0127 | 1.0000 | -0.0131 | 0.0213 | 100000 | ridge_linear_regression |
| fixation_count | opacity_controlled | vocoid_run_cross_boundary_z | 0.0035 | 0.0194 | 0.8333 | -0.0300 | 0.0270 | 100000 | ridge_linear_regression |
| log_total_fixation_duration | boundary_type_controlled | prev_boundary_type_orth_V#V | -0.0087 | 0.0170 | 0.3333 | -0.0339 | 0.0218 | 62306 | ridge_linear_regression |

## Diagnostics
| outcome | status | model_type | n_obs | n_terms |
| --- | --- | --- | --- | --- |
| skip | complete | regularized_logistic_regression | 100000 | 40 |
| log_ffd | complete | ridge_linear_regression | 62306 | 40 |
| log_first_pass_duration | complete | ridge_linear_regression | 62306 | 40 |
| log_go_past_time | complete | ridge_linear_regression | 62306 | 40 |
| log_total_fixation_duration | complete | ridge_linear_regression | 62306 | 40 |
| fixation_count | complete | ridge_linear_regression | 100000 | 40 |
| log_total_fixation_duration | complete | ridge_linear_regression | 62306 | 41 |

## Summary
- Boundary-opacity significant terms: 0
- Boundary-opacity direction consistent across modeled outcomes: False
- Effect plot: `figures/segmentation_effects.png`

These results are controlled exploratory associations, not label generation. A positive duration coefficient indicates higher gaze cost; a positive skipping coefficient indicates higher skipping odds.
