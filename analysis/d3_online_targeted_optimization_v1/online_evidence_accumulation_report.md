# Online Evidence Accumulation Report

- Online probability trajectory rows: 243656
- Accumulators evaluated: beta_binomial_posterior, entropy_weighted, learned_meta_aggregator, logit_mean, mean_probability, reliability_weighted_probability, uncertainty_weighted_logit
- Learned meta-aggregator complete rows: 243656

## Top Accumulator Metrics

| split_regime | feature_group | accumulator | prefix_type | prefix_value | n_readers | n_probability_rows | probability_variance | improvement_over_simple_mean_AUROC | improvement_over_single_trial_d3_lite_AUROC | AUROC | PR-AUC | BA | macro_F1 | Brier | sensitivity | specificity | FPR | FNR | ECE | calibration_slope | calibration_intercept | simple_mean_AUROC | d3_lite_trial_AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen_text | all_allowed_online | learned_meta_aggregator | speech_prefix | 2 | 8 | 8 | 0.0197 | 0.2500 | 0.0848 | 0.9167 | 0.8333 | 0.7500 | 0.6190 | 0.2067 | 1.0000 | 0.5000 | 0.5000 | 0.0000 | 0.3329 | 0.4602 | 0.1879 | 0.6667 | 0.8319 |
| unseen_text | dfm_residual_plus_uncertainty_prefix | learned_meta_aggregator | trial_or_text_prefix | 2 | 8 | 8 | 0.0326 | 0.5000 | 0.0848 | 0.9167 | 0.8333 | 0.7500 | 0.7949 | 0.1160 | 0.5000 | 1.0000 | 0.0000 | 0.5000 | 0.1221 | 0.3488 | 0.5459 | 0.4167 | 0.8319 |
| participant_grouped_kfold | all_allowed_online | logit_mean | speech_prefix | all | 57 | 57 | 0.1628 | 0.0125 |  | 0.9127 | 0.8630 | 0.8158 | 0.7946 | 0.1342 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1135 | 0.0655 | 0.3797 | 0.9003 |  |
| text_balanced_unseen_reader | all_allowed_online | logit_mean | speech_prefix | all | 57 | 57 | 0.1628 | 0.0125 |  | 0.9127 | 0.8630 | 0.8158 | 0.7946 | 0.1342 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1135 | 0.0655 | 0.3797 | 0.9003 |  |
| unseen_reader | all_allowed_online | logit_mean | speech_prefix | all | 57 | 57 | 0.1628 | 0.0125 | 0.1042 | 0.9127 | 0.8630 | 0.8158 | 0.7946 | 0.1342 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1135 | 0.0655 | 0.3797 | 0.9003 | 0.8085 |
| participant_grouped_kfold | all_allowed_online | entropy_weighted | speech_prefix | all | 57 | 57 | 0.1623 | 0.0069 |  | 0.9072 | 0.8688 | 0.8158 | 0.7946 | 0.1398 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1362 | 0.0859 | 0.3849 | 0.9003 |  |
| text_balanced_unseen_reader | all_allowed_online | entropy_weighted | speech_prefix | all | 57 | 57 | 0.1623 | 0.0069 |  | 0.9072 | 0.8688 | 0.8158 | 0.7946 | 0.1398 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1362 | 0.0859 | 0.3849 | 0.9003 |  |
| unseen_reader | all_allowed_online | entropy_weighted | speech_prefix | all | 57 | 57 | 0.1623 | 0.0069 | 0.0987 | 0.9072 | 0.8688 | 0.8158 | 0.7946 | 0.1398 | 0.8421 | 0.7895 | 0.2105 | 0.1579 | 0.1362 | 0.0859 | 0.3849 | 0.9003 | 0.8085 |
| participant_grouped_kfold | all_allowed_online | logit_mean | trial_or_text_prefix | all | 57 | 57 | 0.1617 | 0.0111 |  | 0.9072 | 0.8509 | 0.7895 | 0.7738 | 0.1383 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1057 | 0.0654 | 0.3796 | 0.8961 |  |
| text_balanced_unseen_reader | all_allowed_online | logit_mean | trial_or_text_prefix | all | 57 | 57 | 0.1617 | 0.0111 |  | 0.9072 | 0.8509 | 0.7895 | 0.7738 | 0.1383 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1057 | 0.0654 | 0.3796 | 0.8961 |  |
| unseen_reader | all_allowed_online | logit_mean | trial_or_text_prefix | all | 57 | 57 | 0.1617 | 0.0111 | 0.0987 | 0.9072 | 0.8509 | 0.7895 | 0.7738 | 0.1383 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1057 | 0.0654 | 0.3796 | 0.8961 | 0.8085 |
| participant_grouped_kfold | all_allowed_online | entropy_weighted | trial_or_text_prefix | all | 57 | 57 | 0.1619 | 0.0097 |  | 0.9058 | 0.8666 | 0.7895 | 0.7738 | 0.1440 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1140 | 0.0858 | 0.3845 | 0.8961 |  |
| text_balanced_unseen_reader | all_allowed_online | entropy_weighted | trial_or_text_prefix | all | 57 | 57 | 0.1619 | 0.0097 |  | 0.9058 | 0.8666 | 0.7895 | 0.7738 | 0.1440 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1140 | 0.0858 | 0.3845 | 0.8961 |  |
| unseen_reader | all_allowed_online | entropy_weighted | trial_or_text_prefix | all | 57 | 57 | 0.1619 | 0.0097 | 0.0973 | 0.9058 | 0.8666 | 0.7895 | 0.7738 | 0.1440 | 0.7895 | 0.7895 | 0.2105 | 0.2105 | 0.1140 | 0.0858 | 0.3845 | 0.8961 | 0.8085 |
| participant_grouped_kfold | all_allowed_online | entropy_weighted | chronological_prefix | all | 57 | 57 | 0.1619 | 0.0166 |  | 0.9030 | 0.8639 | 0.7500 | 0.7361 | 0.1502 | 0.7368 | 0.7632 | 0.2368 | 0.2632 | 0.1348 | 0.0852 | 0.3841 | 0.8864 |  |
| text_balanced_unseen_reader | all_allowed_online | entropy_weighted | chronological_prefix | all | 57 | 57 | 0.1619 | 0.0166 |  | 0.9030 | 0.8639 | 0.7500 | 0.7361 | 0.1502 | 0.7368 | 0.7632 | 0.2368 | 0.2632 | 0.1348 | 0.0852 | 0.3841 | 0.8864 |  |
| unseen_reader | all_allowed_online | entropy_weighted | chronological_prefix | all | 57 | 57 | 0.1619 | 0.0166 | 0.0945 | 0.9030 | 0.8639 | 0.7500 | 0.7361 | 0.1502 | 0.7368 | 0.7632 | 0.2368 | 0.2632 | 0.1348 | 0.0852 | 0.3841 | 0.8864 | 0.8085 |
| participant_grouped_kfold | all_allowed_online | reliability_weighted_probability | speech_prefix | all | 57 | 57 | 0.1180 | 0.0014 |  | 0.9017 | 0.8582 | 0.8421 | 0.8286 | 0.1250 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1366 | 0.0939 | 0.3696 | 0.9003 |  |
| text_balanced_unseen_reader | all_allowed_online | reliability_weighted_probability | speech_prefix | all | 57 | 57 | 0.1180 | 0.0014 |  | 0.9017 | 0.8582 | 0.8421 | 0.8286 | 0.1250 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1366 | 0.0939 | 0.3696 | 0.9003 |  |
| unseen_reader | all_allowed_online | reliability_weighted_probability | speech_prefix | all | 57 | 57 | 0.1180 | 0.0014 | 0.0932 | 0.9017 | 0.8582 | 0.8421 | 0.8286 | 0.1250 | 0.8421 | 0.8421 | 0.1579 | 0.1579 | 0.1366 | 0.0939 | 0.3696 | 0.9003 | 0.8085 |
