# Online Prefix Model Report

- Metric rows: 1245
- Outer-test prediction rows used: 29554
- Fixed-threshold baseline rows from OperatingPointAdaptation v1 are included for comparison when available.

## Top Reader-Aggregated Rows

| evaluation_level | split_regime | feature_group | prefix_type | prefix_value | threshold_source | n_readers | n_prefix_rows | unstable_prefix_rate | AUROC | PR-AUC | BA | macro_F1 | Brier | sensitivity | specificity | FPR | FNR | ECE | calibration_slope | calibration_intercept | source_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reader_aggregated | leave_one_participant_out | D3_dfm_residual_gaze_only | previous_artifact | autoresearch_d3_final_reader_profile | fixed_0_5 |  |  |  | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 |  |  |  |  |  |  |  | autoresearch_d3_final_reader_profile |
| reader_aggregated | participant_grouped_kfold | dfm_residual_gaze_prefix | chronological_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8934 | 0.8732 | 0.8553 | 0.8459 | 0.1210 | 0.8421 | 0.8684 | 0.1316 | 0.1579 | 0.1239 | 0.0603 | 0.3763 |  |
| reader_aggregated | participant_grouped_kfold | dfm_residual_gaze_prefix | speech_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8934 | 0.8732 | 0.8553 | 0.8459 | 0.1210 | 0.8421 | 0.8684 | 0.1316 | 0.1579 | 0.1239 | 0.0603 | 0.3763 |  |
| reader_aggregated | participant_grouped_kfold | dfm_residual_gaze_prefix | trial_or_text_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8934 | 0.8732 | 0.8553 | 0.8459 | 0.1210 | 0.8421 | 0.8684 | 0.1316 | 0.1579 | 0.1239 | 0.0603 | 0.3763 |  |
| reader_aggregated | participant_grouped_kfold | dfm_residual_gaze_prefix | word_count_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8934 | 0.8732 | 0.8553 | 0.8459 | 0.1210 | 0.8421 | 0.8684 | 0.1316 | 0.1579 | 0.1239 | 0.0603 | 0.3763 |  |
| reader_aggregated | participant_grouped_kfold | dfm_residual_plus_uncertainty_prefix | chronological_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9058 | 0.8818 | 0.8421 | 0.8286 | 0.1290 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1126 | 0.0603 | 0.3830 |  |
| reader_aggregated | participant_grouped_kfold | dfm_residual_plus_uncertainty_prefix | speech_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9058 | 0.8818 | 0.8421 | 0.8286 | 0.1290 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1126 | 0.0603 | 0.3830 |  |
| reader_aggregated | participant_grouped_kfold | dfm_residual_plus_uncertainty_prefix | trial_or_text_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9058 | 0.8818 | 0.8421 | 0.8286 | 0.1290 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1126 | 0.0603 | 0.3830 |  |
| reader_aggregated | participant_grouped_kfold | dfm_residual_plus_uncertainty_prefix | word_count_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9058 | 0.8818 | 0.8421 | 0.8286 | 0.1290 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1126 | 0.0603 | 0.3830 |  |
| reader_aggregated | participant_grouped_kfold | dfm_sensitivity_prefix | speech_prefix | 1 | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8726 | 0.8529 | 0.8421 | 0.8286 | 0.1269 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1589 | 0.0867 | 0.3747 |  |
| reader_aggregated | participant_grouped_kfold | dfm_sensitivity_prefix | trial_or_text_prefix | 1 | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8726 | 0.8529 | 0.8421 | 0.8286 | 0.1269 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1589 | 0.0867 | 0.3747 |  |
| reader_aggregated | participant_grouped_kfold | all_allowed_online | chronological_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9044 | 0.8926 | 0.8289 | 0.8115 | 0.1219 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1276 | 0.0559 | 0.3862 |  |
| reader_aggregated | participant_grouped_kfold | all_allowed_online | speech_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9044 | 0.8926 | 0.8289 | 0.8115 | 0.1219 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1276 | 0.0559 | 0.3862 |  |
| reader_aggregated | participant_grouped_kfold | all_allowed_online | trial_or_text_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9044 | 0.8926 | 0.8289 | 0.8115 | 0.1219 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1276 | 0.0559 | 0.3862 |  |
| reader_aggregated | participant_grouped_kfold | all_allowed_online | word_count_prefix | all | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.9044 | 0.8926 | 0.8289 | 0.8115 | 0.1219 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1276 | 0.0559 | 0.3862 |  |
| reader_aggregated | participant_grouped_kfold | dfm_sensitivity_prefix | chronological_prefix | 500 | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8753 | 0.7938 | 0.8158 | 0.7946 | 0.1429 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1765 | 0.0816 | 0.3716 |  |
| reader_aggregated | participant_grouped_kfold | dfm_sensitivity_prefix | word_count_prefix | 500 | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8753 | 0.7938 | 0.8158 | 0.7946 | 0.1429 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1765 | 0.0816 | 0.3716 |  |
| reader_aggregated | participant_grouped_kfold | raw_gaze_prefix | chronological_prefix | 1000 | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8684 | 0.7843 | 0.8158 | 0.7946 | 0.1496 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1269 | 0.0653 | 0.3543 |  |
| reader_aggregated | participant_grouped_kfold | raw_gaze_prefix | word_count_prefix | 1000 | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8684 | 0.7843 | 0.8158 | 0.7946 | 0.1496 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1269 | 0.0653 | 0.3543 |  |
| reader_aggregated | participant_grouped_kfold | all_allowed_online | chronological_prefix | 500 | fixed_0_5 | 57.0000 | 57.0000 | 0.0000 | 0.8975 | 0.8461 | 0.8026 | 0.7905 | 0.1437 | 0.7895 | 0.8158 | 0.1842 | 0.2105 | 0.1534 | 0.0517 | 0.3829 |  |
