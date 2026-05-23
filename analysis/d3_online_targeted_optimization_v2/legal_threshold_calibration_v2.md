# Legal Threshold and Calibration Summary v2

All clean v2 selections use `selection_source=inner_oof`; outer-test labels are used only for locked evaluation.

## Final Rows

| final_model | split_regime | candidate_id | candidate_group | calibrator | calibration_source | threshold_policy | threshold_source | threshold | legal_inner_only | fitted_calibrator_used | learned_threshold_used | clean_result | official_claim_allowed | validation_score | AUROC | BA | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_offline_all_full_evidence | participant_grouped_kfold | v2_candidate_0000 | offline_all_full_evidence | identity | identity | fixed_0_5 | inner_oof | 0.5000 | True | False | False | True | False | 0.7937 | 0.8989 | 0.8158 | 0.1186 |
| best_offline_all_full_evidence | text_balanced_unseen_reader | v2_candidate_0000 | offline_all_full_evidence | identity | identity | fixed_0_5 | inner_oof | 0.5000 | True | False | False | True | False | 0.7937 | 0.8989 | 0.8158 | 0.1186 |
| best_offline_all_full_evidence | unseen_reader | v2_candidate_0000 | offline_all_full_evidence | identity | identity | fixed_0_5 | inner_oof | 0.5000 | True | False | False | True | False | 0.7937 | 0.8989 | 0.8158 | 0.1186 |
| best_online_late_accumulation | participant_grouped_kfold | v2_candidate_0012 | online_late_accumulation | isotonic | isotonic | fixed_0_5 | inner_oof | 0.5000 | True | True | False | True | False | 0.7130 | 0.7784 | 0.6842 | 0.1784 |
| best_online_late_accumulation | text_balanced_unseen_reader | v2_candidate_0012 | online_late_accumulation | isotonic | isotonic | fixed_0_5 | inner_oof | 0.5000 | True | True | False | True | False | 0.7130 | 0.7784 | 0.6842 | 0.1784 |
| best_online_late_accumulation | unseen_reader | v2_candidate_0012 | online_late_accumulation | isotonic | isotonic | fixed_0_5 | inner_oof | 0.5000 | True | True | False | True | False | 0.7130 | 0.7784 | 0.6842 | 0.1784 |
| best_online_late_accumulation | unseen_reader_and_text | v2_candidate_0012 | online_late_accumulation | isotonic | isotonic | fixed_0_5 | inner_oof | 0.5000 | True | True | False | True | False | 0.7130 | 0.7014 | 0.5833 | 0.2581 |
| best_online_late_accumulation | unseen_text | v2_candidate_0012 | online_late_accumulation | isotonic | isotonic | fixed_0_5 | inner_oof | 0.5000 | True | True | False | True | False | 0.7130 | 0.7647 | 0.7387 | 0.1801 |
| best_online_mid_detection | participant_grouped_kfold | v2_candidate_0019 | online_mid_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.3693 | True | False | True | True | False | 0.7591 | 0.7950 | 0.7763 | 0.1596 |
| best_online_mid_detection | text_balanced_unseen_reader | v2_candidate_0019 | online_mid_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.3693 | True | False | True | True | False | 0.7591 | 0.7950 | 0.7763 | 0.1596 |
| best_online_mid_detection | unseen_reader | v2_candidate_0019 | online_mid_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.3693 | True | False | True | True | False | 0.7591 | 0.7950 | 0.7763 | 0.1596 |
| best_online_mid_detection | unseen_reader_and_text | v2_candidate_0019 | online_mid_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.3693 | True | False | True | True | False | 0.7591 | 0.7639 | 0.7014 | 0.2283 |
| best_online_mid_detection | unseen_text | v2_candidate_0019 | online_mid_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.3693 | True | False | True | True | False | 0.7591 | 0.7696 | 0.6828 | 0.1765 |
| best_online_early_detection | participant_grouped_kfold | v2_candidate_0031 | online_early_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.4322 | True | False | True | True | False | 0.7286 | 0.7770 | 0.7632 | 0.1788 |
| best_online_early_detection | text_balanced_unseen_reader | v2_candidate_0031 | online_early_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.4322 | True | False | True | True | False | 0.7286 | 0.7770 | 0.7632 | 0.1788 |
| best_online_early_detection | unseen_reader | v2_candidate_0031 | online_early_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.4322 | True | False | True | True | False | 0.7286 | 0.7770 | 0.7632 | 0.1788 |
| best_online_early_detection | unseen_reader_and_text | v2_candidate_0031 | online_early_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.4322 | True | False | True | True | False | 0.7286 | 0.8333 | 0.8194 | 0.2035 |
| best_online_early_detection | unseen_text | v2_candidate_0031 | online_early_detection | identity | identity | inner_cv_regime_specific | inner_oof | 0.4322 | True | False | True | True | False | 0.7286 | 0.6884 | 0.6447 | 0.1920 |
| best_online_stopping_detector | participant_grouped_kfold | v2_candidate_0043 | online_stopping_detector | identity | two_sided:0.25:0.75 | inner_cv_global | inner_oof | 0.5000 | True | False | True | True | False | 0.6955 | 0.5177 | 0.4958 | 0.2465 |
| best_online_stopping_detector | text_balanced_unseen_reader | v2_candidate_0043 | online_stopping_detector | identity | two_sided:0.25:0.75 | inner_cv_global | inner_oof | 0.5000 | True | False | True | True | False | 0.6955 | 0.5177 | 0.4958 | 0.2465 |
| best_online_stopping_detector | unseen_reader | v2_candidate_0043 | online_stopping_detector | identity | two_sided:0.25:0.75 | inner_cv_global | inner_oof | 0.5000 | True | False | True | True | False | 0.6955 | 0.5177 | 0.4958 | 0.2465 |
| best_online_stopping_detector | unseen_reader_and_text | v2_candidate_0043 | online_stopping_detector | identity | two_sided:0.25:0.75 | inner_cv_global | inner_oof | 0.5000 | True | False | True | True | False | 0.6955 | 0.7857 | 0.5000 | 0.2591 |
| best_online_stopping_detector | unseen_text | v2_candidate_0043 | online_stopping_detector | identity | two_sided:0.25:0.75 | inner_cv_global | inner_oof | 0.5000 | True | False | True | True | False | 0.6955 | 0.6165 | 0.5597 | 0.1953 |
| best_unseen_text_specialist | unseen_text | unseen_text_rescue_04 | unseen_text_specialist | identity | identity | inner_cv_regime_specific | inner_oof | 0.6234 | True | False | True | True | False |  | 0.8639 | 0.7546 | 0.2331 |

## Candidate Coverage

| candidate_group | calibrator | threshold_policy | candidates | non_empty_validation |
| --- | --- | --- | --- | --- |
| offline_all_full_evidence | identity | fixed_0_5 | 3 | 3 |
| offline_all_full_evidence | sigmoid | inner_cv_global | 1 | 1 |
| online_early_detection | identity | fixed_0_5 | 1 | 0 |
| online_early_detection | identity | inner_cv_global | 1 | 1 |
| online_early_detection | identity | inner_cv_prefix_specific | 1 | 0 |
| online_early_detection | identity | inner_cv_regime_specific | 1 | 1 |
| online_early_detection | isotonic | fixed_0_5 | 1 | 1 |
| online_early_detection | isotonic | inner_cv_global | 1 | 0 |
| online_early_detection | isotonic | inner_cv_prefix_specific | 1 | 0 |
| online_early_detection | isotonic | inner_cv_regime_specific | 1 | 0 |
| online_early_detection | sigmoid | fixed_0_5 | 1 | 1 |
| online_early_detection | sigmoid | inner_cv_global | 1 | 0 |
| online_early_detection | sigmoid | inner_cv_prefix_specific | 1 | 0 |
| online_early_detection | sigmoid | inner_cv_regime_specific | 1 | 0 |
| online_late_accumulation | identity | fixed_0_5 | 1 | 0 |
| online_late_accumulation | identity | inner_cv_global | 1 | 1 |
| online_late_accumulation | identity | inner_cv_prefix_specific | 1 | 0 |
| online_late_accumulation | identity | inner_cv_regime_specific | 1 | 1 |
| online_late_accumulation | isotonic | fixed_0_5 | 1 | 1 |
| online_late_accumulation | isotonic | inner_cv_global | 1 | 0 |
| online_late_accumulation | isotonic | inner_cv_prefix_specific | 1 | 0 |
| online_late_accumulation | isotonic | inner_cv_regime_specific | 1 | 0 |
| online_late_accumulation | sigmoid | fixed_0_5 | 1 | 1 |
| online_late_accumulation | sigmoid | inner_cv_global | 1 | 0 |
| online_late_accumulation | sigmoid | inner_cv_prefix_specific | 1 | 0 |
| online_late_accumulation | sigmoid | inner_cv_regime_specific | 1 | 0 |
| online_mid_detection | identity | fixed_0_5 | 1 | 0 |
| online_mid_detection | identity | inner_cv_global | 1 | 1 |
| online_mid_detection | identity | inner_cv_prefix_specific | 1 | 0 |
| online_mid_detection | identity | inner_cv_regime_specific | 1 | 1 |
| online_mid_detection | isotonic | fixed_0_5 | 1 | 1 |
| online_mid_detection | isotonic | inner_cv_global | 1 | 0 |
| online_mid_detection | isotonic | inner_cv_prefix_specific | 1 | 0 |
| online_mid_detection | isotonic | inner_cv_regime_specific | 1 | 0 |
| online_mid_detection | sigmoid | fixed_0_5 | 1 | 1 |
| online_mid_detection | sigmoid | inner_cv_global | 1 | 0 |
| online_mid_detection | sigmoid | inner_cv_prefix_specific | 1 | 0 |
| online_mid_detection | sigmoid | inner_cv_regime_specific | 1 | 0 |
| online_stopping_detector | identity | inner_cv_global | 12 | 4 |
