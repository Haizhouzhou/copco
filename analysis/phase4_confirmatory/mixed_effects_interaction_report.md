# Mixed-Effects Interaction Report

The confirmatory interaction model focuses only on the Phase 3 candidates: reader group by word length, DFM surprisal, and previous-boundary opacity. Cluster-robust fallback models are used with speech fixed effects because crossed mixed effects are not feasible in this automated run.

## Focus Interaction Coefficients
| outcome | phase4_interaction | estimate | std_error | p_value | ci_low | ci_high | effect_direction | survives_controls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log_ffd | reader_group_x_word_length | -0.0152 | 0.0045 | 0.0007 | -0.0240 | -0.0064 | negative | True |
| log_ffd | reader_group_x_dfm_surprisal | 0.0043 | 0.0045 | 0.3405 | -0.0045 | 0.0131 | positive | False |
| log_ffd | reader_group_x_previous_boundary_opacity | -0.0031 | 0.0022 | 0.1530 | -0.0074 | 0.0012 | negative | False |
| log_first_pass_duration | reader_group_x_word_length | 0.0075 | 0.0148 | 0.6127 | -0.0215 | 0.0364 | positive | False |
| log_first_pass_duration | reader_group_x_dfm_surprisal | 0.0179 | 0.0070 | 0.0110 | 0.0041 | 0.0317 | positive | True |
| log_first_pass_duration | reader_group_x_previous_boundary_opacity | -0.0048 | 0.0029 | 0.0945 | -0.0105 | 0.0008 | negative | False |
| log_go_past_time | reader_group_x_word_length | 0.0230 | 0.0187 | 0.2183 | -0.0136 | 0.0596 | positive | False |
| log_go_past_time | reader_group_x_dfm_surprisal | 0.0292 | 0.0098 | 0.0029 | 0.0100 | 0.0484 | positive | True |
| log_go_past_time | reader_group_x_previous_boundary_opacity | -0.0052 | 0.0039 | 0.1907 | -0.0129 | 0.0026 | negative | False |
| log_total_fixation_duration | reader_group_x_word_length | 0.0099 | 0.0204 | 0.6260 | -0.0300 | 0.0499 | positive | False |
| log_total_fixation_duration | reader_group_x_dfm_surprisal | 0.0415 | 0.0098 | 0.0000 | 0.0222 | 0.0607 | positive | True |
| log_total_fixation_duration | reader_group_x_previous_boundary_opacity | -0.0077 | 0.0035 | 0.0297 | -0.0146 | -0.0008 | negative | True |
| fixation_count | reader_group_x_word_length | 0.1561 | 0.0723 | 0.0308 | 0.0144 | 0.2978 | positive | True |
| fixation_count | reader_group_x_dfm_surprisal | 0.0871 | 0.0266 | 0.0011 | 0.0350 | 0.1392 | positive | True |
| fixation_count | reader_group_x_previous_boundary_opacity | -0.0151 | 0.0045 | 0.0007 | -0.0239 | -0.0063 | negative | True |

## Diagnostics
| outcome | status | model_type | n_obs | warnings | reason |
| --- | --- | --- | --- | --- | --- |
| skip | failed |  |  |  | Singular matrix |
| log_ffd | complete | cluster_robust_ols | 193725.0000 |  |  |
| log_first_pass_duration | complete | cluster_robust_ols | 193725.0000 |  |  |
| log_go_past_time | complete | cluster_robust_ols | 193725.0000 |  |  |
| log_total_fixation_duration | complete | cluster_robust_ols | 193725.0000 |  |  |
| fixation_count | complete | cluster_robust_ols | 305677.0000 |  |  |

## Framing
- Interactions with controlled p-values below 0.05 are treated as confirmatory support.
- Non-surviving interactions should be appendix or deferred framing, not main claims.
