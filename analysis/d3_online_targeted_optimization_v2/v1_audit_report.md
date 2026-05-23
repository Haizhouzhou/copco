# D3 Online v1 Audit Report

## Audit Findings

| audit_item | value |
| --- | --- |
| v1_output_dir | /home/haizhe/copco/results/d3_online_targeted_optimization_v1_fast5_20260523_012750 |
| v1_was_fast_mode | True |
| v1_candidate_search_truncated | True |
| v1_candidates_evaluated | 36 |
| prefix_budgets_evaluated | word_count_prefix:50, chronological_prefix:50, chronological_prefix:100, word_count_prefix:100, word_count_prefix:250, chronological_prefix:250, word_count_prefix:500, chronological_prefix:500, word_count_prefix:1000, chronological_prefix:1000, trial_or_text_prefix:1, trial_or_text_prefix:2, trial_or_text_prefix:3, speech_prefix:1, speech_prefix:2, speech_prefix:3, previous_artifact:d3_eyebench_lite_candidate_0000, previous_artifact:autoresearch_d3_final_reader_profile, previous_artifact:benchmark_bridge_d3_full_data, speech_prefix:all, trial_or_text_prefix:all, chronological_prefix:all, word_count_prefix:all |
| v1_selected_prefix_value_all | not_explicit; selected no_stop final sequence |
| v1_selected_no_stop | True |
| all_allowed_predictor_count | 201 |
| all_allowed_future_like_columns | none_detected |
| best_candidate_uses_future_beyond_prefix | no for prefix rows; yes it consumes all prefixes via no_stop |
| online_primary_score_includes_earliness | true |
| v1_selected_candidate | online_d3_0021 |
| v1_validation_primary_score | 0.7617221831076706 |
| v1_validation_no_earliness_score | 0.8463579813048288 |
| v1_locked_primary_mean_AUROC | 0.8786 |
| v1_locked_primary_mean_BA | 0.8474 |
| v1_best_truly_online | offline_like_late_accumulation_not_early_detector |

## Locked v1 Test Rows

| split_regime | n_readers | earliness_score | online_primary_score | online_no_earliness_score | AUROC | PR-AUC | BA | macro_F1 | Brier | sensitivity | specificity | FPR | FNR | ECE | calibration_slope | calibration_intercept | candidate_id | feature_family | calibrator | threshold_policy | accumulator | stopping_policy | selection_source | clean_result | official_claim_allowed | benchmark_relative_claim_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| participant_grouped_kfold | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| text_balanced_unseen_reader | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_reader | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_reader_and_text | 17 | 0.0000 | 0.7684 | 0.8538 | 0.8611 | 0.8671 | 0.8264 | 0.8235 | 0.1506 | 0.8750 | 0.7778 | 0.2222 | 0.1250 | 0.2050 | 0.1861 | 0.6391 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_text | 57 | 0.0000 | 0.5797 | 0.6442 | 0.7078 | 0.4724 | 0.6842 | 0.6298 | 0.2292 | 0.8421 | 0.5263 | 0.4737 | 0.1579 | 0.1840 | 0.0864 | 0.3697 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |

Conclusion: v1 selected a strong reader-centered late/full-sequence candidate, but it is offline-like for deployment because `no_stop` consumes the final sequence.
