# Participant-Text Feature Dictionary

Rows are participant-by-speech sample units. Identifiers are retained for grouping, fold construction, diagnostics, and reports; they are not primary model predictors.

| feature | category | primary_predictor_allowed | notes |
| --- | --- | --- | --- |
| sample_id | identifiers | True |  |
| participant_id | identifiers | False |  |
| speech_id | identifiers | False |  |
| text_id | identifiers | False |  |
| paragraph_id | identifiers | True |  |
| passage_id | identifiers | True |  |
| reader_group_binary | targets | False |  |
| comprehension_score | targets | True |  |
| mean_ffd | raw_gaze | True |  |
| median_ffd | raw_gaze | True |  |
| mean_gd | raw_gaze | True |  |
| median_gd | raw_gaze | True |  |
| mean_trt | raw_gaze | True |  |
| median_trt | raw_gaze | True |  |
| mean_go_past_time | raw_gaze | True |  |
| mean_fixation_count | raw_gaze | True |  |
| skipping_rate | raw_gaze | True |  |
| refixation_rate | raw_gaze | True |  |
| trt_sd | raw_gaze | True |  |
| trt_q90 | raw_gaze | True |  |
| gaze_missing_rate | raw_gaze | True |  |
| mean_ffd | reading_speed | True |  |
| mean_gd | reading_speed | True |  |
| mean_trt | reading_speed | True |  |
| mean_go_past_time | reading_speed | True |  |
| skipping_rate | reading_speed | True |  |
| mean_dfm_surprisal_exposure | dfm_exposure | True |  |
| mean_dfm_entropy_exposure | dfm_exposure | True |  |
| lm_missing_rate | dfm_exposure | True |  |
| mean_word_length_exposure | exposure | True |  |
| mean_log_frequency_exposure | exposure | True |  |
| mean_sentence_length_exposure | exposure | True |  |
| mean_dfm_surprisal_exposure | exposure | True |  |
| mean_dfm_entropy_exposure | exposure | True |  |
| mean_boundary_opacity_exposure | exposure | True |  |
| vv_boundary_exposure_rate | exposure | True |  |
| lm_missing_rate | exposure | True |  |
| embedding_missing_rate | exposure | True |  |
| parser_missing_rate | exposure | True |  |
| segmentation_missing_rate | exposure | True |  |
| sample_trt_dfm_surprisal_slope | sensitivity | True |  |
| sample_trt_dfm_entropy_slope | sensitivity | True |  |
| sample_go_past_dfm_surprisal_slope | sensitivity | True |  |
| sample_go_past_dfm_entropy_slope | sensitivity | True |  |
| sample_trt_boundary_opacity_slope | sensitivity | True |  |
| sample_trt_vv_cost | sensitivity | True |  |
| sample_go_past_boundary_opacity_slope | sensitivity | True |  |
| global_ffd_residual_mean | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_median | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_sd | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_dfm_surprisal_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_dfm_entropy_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_word_length_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_boundary_opacity_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_high_opacity_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_ffd_residual_vv_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_mean | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_median | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_sd | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_dfm_surprisal_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_dfm_entropy_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_word_length_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_boundary_opacity_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_high_opacity_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_first_pass_residual_vv_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_mean | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_median | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_sd | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_dfm_surprisal_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_dfm_entropy_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_word_length_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_boundary_opacity_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_high_opacity_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_go_past_residual_vv_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_mean | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_median | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_sd | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_dfm_surprisal_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_dfm_entropy_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_word_length_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_boundary_opacity_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_high_opacity_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_total_fixation_residual_vv_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_mean | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_median | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_sd | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_dfm_surprisal_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_dfm_entropy_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_word_length_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_boundary_opacity_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_high_opacity_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_skipping_residual_vv_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_mean | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_median | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_sd | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_dfm_surprisal_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_dfm_entropy_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_word_length_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_boundary_opacity_slope | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_high_opacity_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_fixation_count_residual_vv_cost | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_residual_unstable_count | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_residual_unstable | descriptive_global_residual | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| gaze_missing_rate | quality | True |  |
| lm_missing_rate | quality | True |  |
| embedding_missing_rate | quality | True |  |
| parser_missing_rate | quality | True |  |
| segmentation_missing_rate | quality | True |  |
| sample_sensitivity_unstable | quality | True |  |
| sample_sensitivity_unstable_count | quality | True |  |
| global_residual_unstable_count | quality | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
| global_residual_unstable | quality | False | descriptive only; split-specific cross-fitted residual features are used for primary benchmark models |
