# Online Targeted Optimization Report

- Candidates evaluated: 36
- Selection used inner-validation / inner-OOF predictions only.
- Locked candidate: `online_d3_0021`

## Validation Ranking

| candidate_id | feature_family | calibrator | threshold_policy | accumulator | stopping_policy | selection_source | validation_primary_score | validation_no_earliness_score | validation_rows | official_claim_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | 0.7617 | 0.8464 | 777 | False |
| online_d3_0009 | dfm_residual_plus_uncertainty_prefix | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | 0.7408 | 0.8232 | 777 | False |
| online_d3_0018 | all_allowed_online | identity | inner_cv_prefix_specific | entropy_weighted | no_stop | inner_oof | 0.7272 | 0.8079 | 777 | False |
| online_d3_0012 | all_allowed_online | identity | fixed_0_5 | mean_probability | no_stop | inner_oof | 0.7252 | 0.8058 | 777 | False |
| online_d3_0022 | all_allowed_online | identity | inner_cv_prefix_specific | learned_meta_aggregator | confidence_stop | inner_oof | 0.7163 | 0.7300 | 777 | False |
| online_d3_0023 | all_allowed_online | sigmoid | two_sided_confidence | learned_meta_aggregator | cost_sensitive_stop | inner_oof | 0.7114 | 0.7181 | 777 | False |
| online_d3_0010 | dfm_residual_plus_uncertainty_prefix | identity | inner_cv_prefix_specific | learned_meta_aggregator | confidence_stop | inner_oof | 0.6994 | 0.7179 | 777 | False |
| online_d3_0015 | all_allowed_online | sigmoid | two_sided_confidence | logit_mean | no_stop | inner_oof | 0.6987 | 0.7763 | 777 | False |
| online_d3_0011 | dfm_residual_plus_uncertainty_prefix | sigmoid | two_sided_confidence | learned_meta_aggregator | cost_sensitive_stop | inner_oof | 0.6977 | 0.7160 | 777 | False |
| online_d3_0006 | dfm_residual_plus_uncertainty_prefix | identity | inner_cv_prefix_specific | entropy_weighted | no_stop | inner_oof | 0.6922 | 0.7692 | 777 | False |
| online_d3_0000 | dfm_residual_plus_uncertainty_prefix | identity | fixed_0_5 | mean_probability | no_stop | inner_oof | 0.6862 | 0.7625 | 777 | False |
| online_d3_0020 | all_allowed_online | identity | fixed_0_5 | entropy_weighted | cost_sensitive_stop | inner_oof | 0.6801 | 0.6816 | 777 | False |
| online_d3_0003 | dfm_residual_plus_uncertainty_prefix | sigmoid | two_sided_confidence | logit_mean | no_stop | inner_oof | 0.6794 | 0.7549 | 777 | False |
| online_d3_0019 | all_allowed_online | sigmoid | two_sided_confidence | entropy_weighted | confidence_stop | inner_oof | 0.6790 | 0.6804 | 777 | False |
| online_d3_0016 | all_allowed_online | identity | fixed_0_5 | logit_mean | confidence_stop | inner_oof | 0.6726 | 0.6748 | 777 | False |

## Locked Test Results

| split_regime | n_readers | earliness_score | online_primary_score | online_no_earliness_score | AUROC | PR-AUC | BA | macro_F1 | Brier | sensitivity | specificity | FPR | FNR | ECE | calibration_slope | calibration_intercept | candidate_id | feature_family | calibrator | threshold_policy | accumulator | stopping_policy | selection_source | clean_result | official_claim_allowed | benchmark_relative_claim_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| participant_grouped_kfold | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| text_balanced_unseen_reader | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_reader | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_reader_and_text | 17 | 0.0000 | 0.7684 | 0.8538 | 0.8611 | 0.8671 | 0.8264 | 0.8235 | 0.1506 | 0.8750 | 0.7778 | 0.2222 | 0.1250 | 0.2050 | 0.1861 | 0.6391 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_text | 57 | 0.0000 | 0.5797 | 0.6442 | 0.7078 | 0.4724 | 0.6842 | 0.6298 | 0.2292 | 0.8421 | 0.5263 | 0.4737 | 0.1579 | 0.1840 | 0.0864 | 0.3697 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |

## Baseline Comparison Inputs

| baseline | split_regime | evaluation_level | AUROC | BA | Brier |
| --- | --- | --- | --- | --- | --- |
| benchmark_bridge_d3_full_data | unseen_reader | trial_level |  |  | 0.1595 |
| benchmark_bridge_d3_full_data | unseen_reader_and_text | trial_level | 0.8389 | 0.7798 | 0.1873 |
| benchmark_bridge_d3_full_data | unseen_text | trial_level | 0.8695 | 0.7648 | 0.2412 |
| d3_eyebench_lite_candidate_0000 | unseen_reader | trial_level | 0.8085 | 0.7274 | 0.1904 |
| d3_eyebench_lite_candidate_0000 | unseen_reader_and_text | trial_level | 0.7154 | 0.6342 | 0.2191 |
| d3_eyebench_lite_candidate_0000 | unseen_text | trial_level | 0.8319 | 0.7341 | 0.1871 |
