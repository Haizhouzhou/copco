# Per-Prefix Performance Report

- Curve rows: 1232
- Rows use outer-test probabilities and fixed 0.5 threshold unless otherwise stated.

## Earliest Reliability Criteria

| split_regime | criterion | earliest | feature_family | accumulator | value |
| --- | --- | --- | --- | --- | --- |
| unseen_reader | AUROC >= 0.75 | chronological_prefix:100 | all_allowed_strict_online | beta_binomial_posterior | 0.7659 |
| unseen_reader | AUROC >= 0.8 | chronological_prefix:250 | all_allowed_strict_online | entropy_weighted | 0.8310 |
| unseen_reader | BA >= 0.7 | word_count_prefix:100 | all_allowed_strict_online | mean_probability | 0.7237 |
| unseen_reader | BA >= 0.75 | chronological_prefix:250 | all_allowed_strict_online | reliability_weighted_probability | 0.7500 |
| unseen_reader | Brier <= 0.18 | word_count_prefix:250 | all_allowed_strict_online | learned_meta_aggregator | 0.1716 |
| unseen_reader_and_text | AUROC >= 0.75 | word_count_prefix:250 | all_allowed_strict_online | learned_meta_aggregator | 0.7500 |
| unseen_reader_and_text | AUROC >= 0.8 | chronological_prefix:500 | all_allowed_strict_online | uncertainty_weighted_logit | 0.8056 |
| unseen_reader_and_text | BA >= 0.7 | chronological_prefix:500 | all_allowed_strict_online | reliability_weighted_probability | 0.7014 |
| unseen_reader_and_text | BA >= 0.75 | word_count_prefix:500 | all_allowed_strict_online | uncertainty_weighted_logit | 0.7639 |
| unseen_reader_and_text | Brier <= 0.18 | chronological_prefix:1000 | all_allowed_strict_online | beta_binomial_posterior | 0.1730 |

## Top Per-Prefix Rows

| split_regime | prefix_type | prefix_value | feature_family | source_feature_group | calibrator | threshold | accumulator | n_readers | n_prefix_rows | unstable_prefix_rate | AUROC | PR-AUC | BA | macro_F1 | Brier | sensitivity | specificity | FPR | FNR | ECE | calibration_slope | calibration_intercept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen_text | speech_prefix | 2 | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | learned_meta_aggregator | 8 | 8 | 0.0000 | 0.9167 | 0.8333 | 0.7500 | 0.6190 | 0.2067 | 1.0000 | 0.5000 | 0.5000 | 0.0000 | 0.3329 | 0.4602 | 0.1879 |
| unseen_text | trial_or_text_prefix | 2 | dfm_residual_plus_uncertainty_prefix | dfm_residual_plus_uncertainty_prefix | identity | fixed_0_5 | learned_meta_aggregator | 8 | 8 | 0.0000 | 0.9167 | 0.8333 | 0.7500 | 0.7949 | 0.1160 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.1221 | 0.3488 | 0.5459 |
| participant_grouped_kfold | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | logit_mean | 57 | 57 | 0.0000 | 0.9127 | 0.8630 | 0.8158 | 0.7946 | 0.1342 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1135 | 0.0655 | 0.3797 |
| text_balanced_unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | logit_mean | 57 | 57 | 0.0000 | 0.9127 | 0.8630 | 0.8158 | 0.7946 | 0.1342 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1135 | 0.0655 | 0.3797 |
| unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | logit_mean | 57 | 57 | 0.0000 | 0.9127 | 0.8630 | 0.8158 | 0.7946 | 0.1342 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1135 | 0.0655 | 0.3797 |
| participant_grouped_kfold | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9072 | 0.8688 | 0.8158 | 0.7946 | 0.1398 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1362 | 0.0859 | 0.3849 |
| text_balanced_unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9072 | 0.8688 | 0.8158 | 0.7946 | 0.1398 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1362 | 0.0859 | 0.3849 |
| unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9072 | 0.8688 | 0.8158 | 0.7946 | 0.1398 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1362 | 0.0859 | 0.3849 |
| participant_grouped_kfold | trial_or_text_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | logit_mean | 57 | 57 | 0.0000 | 0.9072 | 0.8509 | 0.7895 | 0.7738 | 0.1383 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1057 | 0.0654 | 0.3796 |
| text_balanced_unseen_reader | trial_or_text_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | logit_mean | 57 | 57 | 0.0000 | 0.9072 | 0.8509 | 0.7895 | 0.7738 | 0.1383 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1057 | 0.0654 | 0.3796 |
| unseen_reader | trial_or_text_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | logit_mean | 57 | 57 | 0.0000 | 0.9072 | 0.8509 | 0.7895 | 0.7738 | 0.1383 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1057 | 0.0654 | 0.3796 |
| participant_grouped_kfold | trial_or_text_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9058 | 0.8666 | 0.7895 | 0.7738 | 0.1440 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1140 | 0.0858 | 0.3845 |
| text_balanced_unseen_reader | trial_or_text_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9058 | 0.8666 | 0.7895 | 0.7738 | 0.1440 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1140 | 0.0858 | 0.3845 |
| unseen_reader | trial_or_text_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9058 | 0.8666 | 0.7895 | 0.7738 | 0.1440 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1140 | 0.0858 | 0.3845 |
| participant_grouped_kfold | chronological_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9030 | 0.8639 | 0.7500 | 0.7361 | 0.1502 | 0.7368 | 0.7632 | 0.2368 | 0.2632 | 0.1348 | 0.0852 | 0.3841 |
| text_balanced_unseen_reader | chronological_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9030 | 0.8639 | 0.7500 | 0.7361 | 0.1502 | 0.7368 | 0.7632 | 0.2368 | 0.2632 | 0.1348 | 0.0852 | 0.3841 |
| unseen_reader | chronological_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | entropy_weighted | 57 | 57 | 0.0000 | 0.9030 | 0.8639 | 0.7500 | 0.7361 | 0.1502 | 0.7368 | 0.7632 | 0.2368 | 0.2632 | 0.1348 | 0.0852 | 0.3841 |
| participant_grouped_kfold | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | reliability_weighted_probability | 57 | 57 | 0.0000 | 0.9017 | 0.8582 | 0.8421 | 0.8286 | 0.1250 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1366 | 0.0939 | 0.3696 |
| text_balanced_unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | reliability_weighted_probability | 57 | 57 | 0.0000 | 0.9017 | 0.8582 | 0.8421 | 0.8286 | 0.1250 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1366 | 0.0939 | 0.3696 |
| unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | reliability_weighted_probability | 57 | 57 | 0.0000 | 0.9017 | 0.8582 | 0.8421 | 0.8286 | 0.1250 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1366 | 0.0939 | 0.3696 |
| participant_grouped_kfold | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.0000 | 0.9003 | 0.8561 | 0.8289 | 0.8115 | 0.1271 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1115 | 0.1835 | 0.4137 |
| participant_grouped_kfold | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | mean_probability | 57 | 57 | 0.0000 | 0.9003 | 0.8561 | 0.8289 | 0.8115 | 0.1260 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1116 | 0.0944 | 0.3702 |
| text_balanced_unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.0000 | 0.9003 | 0.8561 | 0.8289 | 0.8115 | 0.1271 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1115 | 0.1835 | 0.4137 |
| text_balanced_unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | mean_probability | 57 | 57 | 0.0000 | 0.9003 | 0.8561 | 0.8289 | 0.8115 | 0.1260 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1116 | 0.0944 | 0.3702 |
| unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.0000 | 0.9003 | 0.8561 | 0.8289 | 0.8115 | 0.1271 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1115 | 0.1835 | 0.4137 |
| unseen_reader | speech_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | mean_probability | 57 | 57 | 0.0000 | 0.9003 | 0.8561 | 0.8289 | 0.8115 | 0.1260 | 0.8421 | 0.8158 | 0.1842 | 0.1579 | 0.1116 | 0.0944 | 0.3702 |
| participant_grouped_kfold | chronological_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | learned_meta_aggregator | 57 | 57 | 0.0000 | 0.9003 | 0.8204 | 0.7895 | 0.8081 | 0.1215 | 0.6316 | 0.9474 | 0.0526 | 0.3684 | 0.1069 | 0.1819 | 0.6248 |
| text_balanced_unseen_reader | chronological_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | learned_meta_aggregator | 57 | 57 | 0.0000 | 0.9003 | 0.8204 | 0.7895 | 0.8081 | 0.1215 | 0.6316 | 0.9474 | 0.0526 | 0.3684 | 0.1069 | 0.1819 | 0.6248 |
| unseen_reader | chronological_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | learned_meta_aggregator | 57 | 57 | 0.0000 | 0.9003 | 0.8204 | 0.7895 | 0.8081 | 0.1215 | 0.6316 | 0.9474 | 0.0526 | 0.3684 | 0.1069 | 0.1819 | 0.6248 |
| participant_grouped_kfold | chronological_prefix | all | all_allowed_strict_online | all_allowed_online | identity | fixed_0_5 | logit_mean | 57 | 57 | 0.0000 | 0.9003 | 0.8408 | 0.7632 | 0.7524 | 0.1441 | 0.7368 | 0.7895 | 0.2105 | 0.2632 | 0.1208 | 0.0651 | 0.3797 |
