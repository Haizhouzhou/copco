# Residualization Report

Residual models use stimulus/text predictors only. They do not use reader group, participant target labels, or participant identifiers as predictors.

## Residual Model Diagnostics
| outcome | status | n_obs | n_predictors | uses_reader_group |
| --- | --- | --- | --- | --- |
| log_ffd | complete | 208901 | 40 | False |
| log_first_pass_duration | complete | 208901 | 40 | False |
| log_go_past_time | complete | 208901 | 40 | False |
| log_total_fixation_duration | complete | 208901 | 40 | False |
| skip | complete | 335203 | 40 | False |
| fixation_count | complete | 335203 | 40 | False |

## Participant Profile Summary By Reader Group
| reader_group | participants | mean_trt_residual | mean_high_opacity_cost | mean_vv_cost | mean_boundary_sensitivity |
| --- | --- | --- | --- | --- | --- |
| dyslexia_labeled | 19 | 0.2630 | -0.0229 | -0.0229 | -0.0134 |
| typical_control | 38 | -0.0778 | 0.0077 | 0.0077 | -0.0030 |

## Output
- `participant_sensitivity_profiles.parquet`
- `participant_sensitivity_profile_dictionary.md`
