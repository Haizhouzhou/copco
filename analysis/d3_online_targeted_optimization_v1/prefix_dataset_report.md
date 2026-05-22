# Prefix Dataset Report

- Input word rows after primary-analysis filtering: 335203
- Prefix rows: 1145
- Participants: 57
- Prefix types attempted: chronological_prefix, speech_prefix, trial_or_text_prefix, word_count_prefix
- No-future/monotonic errors: 0

## Counts by Prefix

| prefix_type | prefix_value | rows | participants | mean_words | mean_texts | stable_rate |
| --- | --- | --- | --- | --- | --- | --- |
| chronological_prefix | 100 | 57 | 57 | 100.0000 | 1.0000 | 1.0000 |
| chronological_prefix | 1000 | 57 | 57 | 1000.0000 | 1.0877 | 1.0000 |
| chronological_prefix | 250 | 57 | 57 | 250.0000 | 1.0000 | 1.0000 |
| chronological_prefix | 50 | 57 | 57 | 50.0000 | 1.0000 | 1.0000 |
| chronological_prefix | 500 | 57 | 57 | 500.0000 | 1.0175 | 1.0000 |
| chronological_prefix | all | 57 | 57 | 5880.7544 | 4.2456 | 1.0000 |
| speech_prefix | 1 | 57 | 57 | 1663.0702 | 1.0000 | 1.0000 |
| speech_prefix | 2 | 55 | 55 | 3017.9091 | 2.0000 | 1.0000 |
| speech_prefix | 3 | 43 | 43 | 4354.5349 | 3.0000 | 1.0000 |
| speech_prefix | 5 | 18 | 18 | 6705.0556 | 5.0000 | 1.0000 |
| speech_prefix | all | 57 | 57 | 5880.7544 | 4.2456 | 1.0000 |
| trial_or_text_prefix | 1 | 57 | 57 | 1663.0702 | 1.0000 | 1.0000 |
| trial_or_text_prefix | 10 | 1 | 1 | 7164.0000 | 10.0000 | 1.0000 |
| trial_or_text_prefix | 2 | 55 | 55 | 3017.9091 | 2.0000 | 1.0000 |
| trial_or_text_prefix | 3 | 43 | 43 | 4354.5349 | 3.0000 | 1.0000 |
| trial_or_text_prefix | 5 | 18 | 18 | 6705.0556 | 5.0000 | 1.0000 |
| trial_or_text_prefix | all | 57 | 57 | 5880.7544 | 4.2456 | 1.0000 |
| word_count_prefix | 100 | 57 | 57 | 100.0000 | 1.0000 | 1.0000 |
| word_count_prefix | 1000 | 57 | 57 | 1000.0000 | 1.0877 | 1.0000 |
| word_count_prefix | 250 | 57 | 57 | 250.0000 | 1.0000 | 1.0000 |
| word_count_prefix | 50 | 57 | 57 | 50.0000 | 1.0000 | 1.0000 |
| word_count_prefix | 500 | 57 | 57 | 500.0000 | 1.0175 | 1.0000 |
| word_count_prefix | all | 57 | 57 | 5880.7544 | 4.2456 | 1.0000 |

## Highest Missingness

| column | missing_rate |
| --- | --- |
| dfm_sens_gd_surprisal_corr | 0.1083 |
| dfm_sens_gd_surprisal_slope | 0.1083 |
| dfm_resid_gd_surprisal_slope | 0.1083 |
| dfm_resid_gd_surprisal_corr | 0.1083 |
| dfm_resid_trt_entropy_corr | 0.1083 |
| dfm_resid_trt_entropy_slope | 0.1083 |
| dfm_sens_trt_entropy_corr | 0.1083 |
| dfm_sens_trt_entropy_slope | 0.1083 |
| dfm_sens_gd_entropy_slope | 0.1083 |
| dfm_resid_trt_surprisal_corr | 0.1083 |
| dfm_resid_trt_surprisal_slope | 0.1083 |
| dfm_sens_gd_entropy_corr | 0.1083 |
| dfm_resid_ffd_entropy_corr | 0.1083 |
| dfm_sens_trt_surprisal_corr | 0.1083 |
| dfm_sens_trt_surprisal_slope | 0.1083 |
| dfm_sens_ffd_surprisal_slope | 0.1083 |
| dfm_sens_ffd_surprisal_corr | 0.1083 |
| dfm_resid_ffd_entropy_slope | 0.1083 |
| dfm_resid_gd_entropy_slope | 0.1083 |
| dfm_resid_ffd_surprisal_slope | 0.1083 |
| dfm_sens_ffd_entropy_corr | 0.1083 |
| dfm_sens_ffd_entropy_slope | 0.1083 |
| dfm_resid_gd_entropy_corr | 0.1083 |
| dfm_resid_ffd_surprisal_corr | 0.1083 |
| dfm_sens_fixation_count_entropy_slope | 0.0402 |

## Validation Notes

- Prefix features are cumulative summaries over rows observed through the prefix.
- Participant labels are retained only for evaluation and are not used in feature construction.
- Residual gaze features use leave-participant-out stimulus references.
- Unstable slope estimates are flagged with `_unstable` columns and summarized by `uncert_unstable_slope_rate`.
