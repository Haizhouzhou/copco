# Final Publication Decision Report

- Readiness: `ready_for_manuscript_drafting`
- Recommended title: DFM Predictability Sensitivity in Danish Natural-Reading Gaze Profiles

## Final Main Claim
Participant-level DFM predictability sensitivity and cross-fitted residualized gaze-cost profiles distinguish dyslexia-labeled and typical/control readers in Danish natural reading.

## Exact Supporting Result
`D3_dfm_residual_gaze_only` logistic regression under LOPO: ROC-AUC 0.8947, PR-AUC 0.8641, balanced accuracy 0.8421, macro F1 0.8421, Brier 0.1159.

## Required Answers
1. Final main claim: DFM predictability sensitivity and residualized gaze-cost profiles distinguish groups.
2. Supporting result: locked Phase 4 D3 LOPO metrics and robustness tests.
3. Selected model: D3 logistic regression LOPO.
4. Central feature family: cross-fitted DFM residual gaze sensitivity.
5. DFM exposure is not the explanation: D1 exposure-only is weak.
6. Result is not just text exposure: exposure-count variables are prohibited and D3 has none.
7. Result is not just raw speed: D3 has no raw speed/global-duration aggregates.
8. Boundary opacity: secondary interpretability interaction.
9. Segmentation does not play the central role: standalone main-effect framing is dropped.
10. Appendix: exposure audits, calibration, bootstrap/permutation, influence, mixed fallback diagnostics.
11. Deferred: Gemma sensitivity, pronunciation-aware segmentation, true parser syntax.
12. Dropped: random word-level prediction and clinical/screening claims.
13. Remaining before submission: prose, figure polishing, and reviewer framing.
14. Ready for manuscript drafting: `True`.
15. Recommended title listed above.
16. Contribution list below.

## Contribution List
- A prepared Danish natural-reading gaze, linguistic, LM, and label pipeline for dyslexia-labeled reader analysis.
- A cross-fitted residualized participant sensitivity-profile method.
- Evidence that DFM predictability sensitivity, not DFM exposure, drives strong participant-level prediction.
- Secondary evidence that reader-group differences involve word length, DFM surprisal, and previous-boundary opacity.

## Decision Gates
| gate | passed | value |
| --- | --- | --- |
| roc_auc_lower_bootstrap_bound_gt_0_70 | True | 0.7765 |
| permutation_p_value_lt_0_01 | True | 0.0010 |
| dfm_sensitivity_outperforms_exposure_only | True | D3=0.8947368421052632; D1=0.4238227146814405 |
| primary_has_no_exposure_count_variables | True | [] |
| one_prediction_per_participant | True | 57 |
| no_leakage_validation_errors | True | [] |
| stable_feature_interpretation | True | {'raw_speed_dominates': False} |

## Categories
| result | category |
| --- | --- |
| DFM residual gaze sensitivity | main_paper_result |
| DFM surprisal interaction | secondary_result |
| word-length interaction | secondary_result |
| boundary-opacity interaction | secondary_result |
| DFM exposure-only comparison | appendix_result |
| Gemma sensitivity | defer |
| standalone segmentation main effect | drop |
| random word-level prediction | drop |
