# Unseen Text Failure Analysis

Unseen-text remains the hardest split for the v1 locked candidate. The rescue candidates are legal diagnostic rows selected without optimizing the final model solely for unseen text.

## Rescue Candidates

| split_regime | n_readers | n_prefix_rows | coverage | undecided_rate | mean_words_to_decision | mean_texts_to_decision | evidence_cost | earliness_score | unstable_prefix_rate | AUROC | PR-AUC | BA | macro_F1 | Brier | sensitivity | specificity | FPR | FNR | ECE | calibration_slope | calibration_intercept | candidate_id | candidate_group | feature_family | source_feature_group | calibrator | threshold_policy | accumulator | stopping_policy | prefix_type | prefix_value | threshold | calibration_source | clean_result | official_claim_allowed | selection_source | rescue_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen_text | 52 | 52 | 1.0000 | 0.0000 | 1000.0000 | 1.0000 | 0.7568 | 0.2432 | 0.0000 | 0.8639 | 0.8361 | 0.7546 | 0.7204 | 0.2331 | 0.8235 | 0.6857 | 0.3143 | 0.1765 | 0.2850 | 0.0733 | 0.2670 | unseen_text_rescue_04 | unseen_text_specialist | all_allowed_strict_online | all_allowed_online | identity | inner_cv_regime_specific | entropy_weighted | fixed_budget | word_count_prefix | 1000 | 0.6234 | identity | True | False | inner_oof | regime_specific_threshold |
| unseen_text | 52 | 52 | 1.0000 | 0.0000 | 1000.0000 | 1.0000 | 0.7568 | 0.2432 | 0.0000 | 0.8555 | 0.7928 | 0.8261 | 0.8112 | 0.1488 | 0.8235 | 0.8286 | 0.1714 | 0.1765 | 0.1104 | 0.2389 | 0.4578 | unseen_text_rescue_05 | unseen_text_specialist | all_allowed_strict_online | all_allowed_online | sigmoid | inner_cv_regime_specific | mean_probability | fixed_budget | word_count_prefix | 1000 | 0.4554 | sigmoid | True | False | inner_oof | regime_specific_calibrator |
| unseen_text | 52 | 52 | 1.0000 | 0.0000 | 1000.0000 | 1.0000 | 0.7568 | 0.2432 | 0.0000 | 0.7597 | 0.6041 | 0.6529 | 0.6233 | 0.1929 | 0.7059 | 0.6000 | 0.4000 | 0.2941 | 0.1177 | 0.1652 | 0.4071 | unseen_text_rescue_03 | unseen_text_specialist | dfm_residual_plus_uncertainty_prefix | dfm_residual_plus_uncertainty_prefix | sigmoid | inner_cv_regime_specific | logit_mean | fixed_budget | word_count_prefix | 1000 | 0.4149 | sigmoid | True | False | inner_oof | text_difficulty_residualized |
| unseen_text | 52 | 52 | 1.0000 | 0.0000 | 1000.0000 | 1.0000 | 0.7568 | 0.2432 | 0.0000 | 0.7042 | 0.5018 | 0.7134 | 0.6345 | 0.2124 | 0.9412 | 0.4857 | 0.5143 | 0.0588 | 0.1125 | 0.1331 | 0.3930 | unseen_text_rescue_00 | unseen_text_specialist | all_allowed_strict_online | all_allowed_online | sigmoid | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | word_count_prefix | 1000 | 0.2322 | sigmoid | True | False | inner_oof | text_shift_calibrated |

## Error Concentration by Held-Out Text

| terminal_text_id | wrong_rows | readers |
| --- | --- | --- |
| 7905 | 111 | 16 |
| 1323 | 51 | 6 |
| 7946 | 26 | 7 |
| 11171 | 21 | 5 |
| 1125 | 18 | 4 |
| 1165 | 16 | 3 |
| 18473 | 10 | 1 |
| 22811 | 10 | 1 |
| 7856 | 2 | 1 |

Interpretation: text-shift and exposure-feature sensitivity remain plausible error sources when held-out text performance lags reader-centered splits. The final v2 claim therefore remains reader-regime / project-specific rather than full-table official SOTA.
