# Reader-Group Interaction Report

- Model rows sampled deterministically: 100000 of 335203
- Reader-group terms are interpreted as group-associated differences for dyslexia-labeled participants relative to typical/control participants.
- Models use participant-clustered standard errors when fit succeeds.

## Interaction Coefficients
| outcome | term | estimate | std_error | p_value | ci_low | ci_high | n_obs | model_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skip | reader_group_x_word_length_chars_z | 0.0136 | 0.1290 | 1.0000 | -0.2244 | 0.1667 | 100000 | regularized_logistic_regression |
| skip | reader_group_x_log_corpus_frequency_z | -0.0441 | 0.0757 | 0.5000 | -0.1662 | 0.0601 | 100000 | regularized_logistic_regression |
| skip | reader_group_x_dfm_lm_word_surprisal_z | -0.0006 | 0.0687 | 1.0000 | -0.1188 | 0.0601 | 100000 | regularized_logistic_regression |
| skip | reader_group_x_dfm_lm_word_entropy_z | -0.0220 | 0.0374 | 0.6667 | -0.0616 | 0.0446 | 100000 | regularized_logistic_regression |
| skip | reader_group_x_prev_boundary_opacity_score_z | 0.0138 | 0.0609 | 0.8333 | -0.0821 | 0.1013 | 100000 | regularized_logistic_regression |
| skip | reader_group_x_vv_indicator_z | 0.0197 | 0.0525 | 0.6667 | -0.0656 | 0.1013 | 100000 | regularized_logistic_regression |
| skip | reader_group_x_sentence_length_words_z | -0.0298 | 0.0439 | 0.6667 | -0.0721 | 0.0653 | 100000 | regularized_logistic_regression |
| log_total_fixation_duration | reader_group_x_word_length_chars_z | 0.0252 | 0.0240 | 0.0000 | 0.0023 | 0.0665 | 62354 | ridge_linear_regression |
| log_total_fixation_duration | reader_group_x_log_corpus_frequency_z | 0.0091 | 0.0197 | 0.1667 | -0.0034 | 0.0537 | 62354 | ridge_linear_regression |
| log_total_fixation_duration | reader_group_x_dfm_lm_word_surprisal_z | 0.0355 | 0.0149 | 0.0000 | 0.0230 | 0.0642 | 62354 | ridge_linear_regression |
| log_total_fixation_duration | reader_group_x_dfm_lm_word_entropy_z | 0.0002 | 0.0126 | 0.5000 | -0.0303 | 0.0097 | 62354 | ridge_linear_regression |
| log_total_fixation_duration | reader_group_x_prev_boundary_opacity_score_z | -0.0192 | 0.0105 | 0.0000 | -0.0422 | -0.0097 | 62354 | ridge_linear_regression |
| log_total_fixation_duration | reader_group_x_vv_indicator_z | 0.0089 | 0.0119 | 0.5000 | -0.0082 | 0.0305 | 62354 | ridge_linear_regression |
| log_total_fixation_duration | reader_group_x_sentence_length_words_z | 0.0042 | 0.0163 | 0.6667 | -0.0219 | 0.0309 | 62354 | ridge_linear_regression |

## Diagnostics
| outcome | status | model_type | n_obs | n_terms |
| --- | --- | --- | --- | --- |
| skip | complete | regularized_logistic_regression | 100000 | 46 |
| log_total_fixation_duration | complete | ridge_linear_regression | 62354 | 46 |

## Summary
- Interaction terms with p < 0.05: 3
- Interactions remain exploratory because text assignment imbalance is documented and labels are participant-level.
