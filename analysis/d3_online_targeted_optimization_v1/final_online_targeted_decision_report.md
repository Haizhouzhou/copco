# Final Online Targeted Decision Report

1. Was this a real deployed online test, not a framework?
   Yes. The runner built prefix features, trained nested prefix models, produced predictions, and evaluated online accumulation and stopping artifacts.
2. Which subgoals completed?
   See `subgoal_status.json` for the current evidence-backed status.
3. Which subgoals were blocked?
   See `subgoal_status.json`; blocked goals include an exact blocker.
4. What is the best online D3 configuration?
   `online_d3_0021` with `all_allowed_online`, `learned_meta_aggregator`, and `no_stop`.
5. How much evidence is needed before online D3 becomes reliable?
   The comparison table reports performance by word/text budget; reliability is based on the best clean reader-level AUROC/BA tradeoff.
6. Does legal threshold/calibration improve online D3?
   Legal threshold and calibration rows are reported separately; oracle rows are not used for this answer.
7. Does online evidence accumulation improve over single-trial D3_Lite?
   The accumulator metrics include improvement columns against the D3_Lite trial-level AUROC baseline.
8. Does stopping policy reduce reading burden while maintaining useful performance?
   The stopping metrics report coverage, balanced accuracy, and evidence cost.
9. Does this change the main paper claim?
   The offline D3 reader-profile result remains the main claim; online D3 is a targeted secondary analysis if useful.
10. Does this change official EyeBench SOTA status?
   No. Official SOTA status remains unchanged.
11. What exact wording should be added to the manuscript?
   D3 is strongest as an offline reader-profile model. In online sequential detection, probability evidence accumulates across prefixes; targeted calibration, thresholding, and stopping policies provide a deployment-oriented secondary analysis.

## Locked Test Results

| split_regime | n_readers | earliness_score | online_primary_score | online_no_earliness_score | AUROC | PR-AUC | BA | macro_F1 | Brier | sensitivity | specificity | FPR | FNR | ECE | calibration_slope | calibration_intercept | candidate_id | feature_family | calibrator | threshold_policy | accumulator | stopping_policy | selection_source | clean_result | official_claim_allowed | benchmark_relative_claim_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| participant_grouped_kfold | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| text_balanced_unseen_reader | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_reader | 57 | 0.0000 | 0.7838 | 0.8709 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | 0.8421 | 0.8947 | 0.1053 | 0.1579 | 0.0833 | 0.1419 | 0.5647 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_reader_and_text | 17 | 0.0000 | 0.7684 | 0.8538 | 0.8611 | 0.8671 | 0.8264 | 0.8235 | 0.1506 | 0.8750 | 0.7778 | 0.2222 | 0.1250 | 0.2050 | 0.1861 | 0.6391 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |
| unseen_text | 57 | 0.0000 | 0.5797 | 0.6442 | 0.7078 | 0.4724 | 0.6842 | 0.6298 | 0.2292 | 0.8421 | 0.5263 | 0.4737 | 0.1579 | 0.1840 | 0.0864 | 0.3697 | online_d3_0021 | all_allowed_online | sigmoid | inner_cv_global | learned_meta_aggregator | no_stop | inner_oof | True | False | True |

## Oracle Diagnostic Reminder

- Oracle rows: 3785
- Oracle diagnostics are separated and marked `official_claim_allowed=false`.
