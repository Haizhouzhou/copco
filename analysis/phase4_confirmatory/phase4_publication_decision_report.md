# Phase 4 Publication Decision Report

## Main Decision
Main paper should focus on participant-level DFM predictability sensitivity and cross-fitted residualized gaze-cost profiles if the selected confirmatory model remains above the prespecified performance threshold and robustness tests remain supportive.

Boundary opacity should be retained only as an interaction and interpretability feature unless stronger evidence emerges.

## Required Answers
| question | answer | evidence |
| --- | --- | --- |
| Does participant-level prediction survive cross-fitted residualization? | True | D3_dfm_residual_gaze_only logistic_regression leave_one_participant_out: ROC-AUC=0.8947, PR-AUC=0.8641, balanced accuracy=0.8421 |
| Is DFM sensitivity more important than DFM exposure? | True | D1=D1_dfm_exposure_only logistic_regression leave_one_participant_out: ROC-AUC=0.4238, PR-AUC=0.3685, balanced accuracy=0.4474; D2=D2_dfm_sensitivity_only logistic_regression leave_one_participant_out: ROC-AUC=0.8892, PR-AUC=0.8611, balanced accuracy=0.8421; D3=D3_dfm_residual_gaze_only logistic_regression leave_one_participant_out: ROC-AUC=0.8947, PR-AUC=0.8641, balanced accuracy=0.8421; D4=D4_dfm_exposure_plus_sensitivity logistic_regression leave_one_participant_out: ROC-AUC=0.8726, PR-AUC=0.8561, balanced accuracy=0.8158 |
| Does performance survive removal of exposure-count variables? | True | Exposure-count variables are excluded from every Phase 4 feature group. |
| Does performance survive removal of exposure-only variables? | True | K_all_except_exposure_variables logistic_regression leave_one_participant_out: ROC-AUC=0.8241, PR-AUC=0.8152, balanced accuracy=0.7895 |
| Does performance survive removal of raw speed/global-duration variables? | True | J_all_except_raw_speed logistic_regression leave_one_participant_out: ROC-AUC=0.8380, PR-AUC=0.8338, balanced accuracy=0.7895 |
| Are word length, DFM surprisal, and boundary-opacity interactions stable? | True | 8 controlled focus interactions survive. |
| Is segmentation a main finding, secondary finding, or deferred? | secondary_result | Standalone main-effect framing is dropped; boundary opacity is retained only as interaction/interpretability. |

## Decision Categories
| result | category |
| --- | --- |
| participant-level DFM sensitivity and cross-fitted residualized gaze-cost profiles | main_paper_result |
| DFM exposure-only prediction | appendix_result |
| boundary opacity | secondary_result |
| standalone segmentation main-effect framing | drop |
| random word-level prediction | drop |
| parser-syntax claims while parser status is surface_heuristic_fallback | defer |

## Robustness And Stability
| permutation_p_value | leave_one_dyslexia_min_roc_auc | stable_dfm_features | stable_segmentation_features | raw_speed_dominates |
| --- | --- | --- | --- | --- |
| 0.0010 | 0.8801 | 12 | 0 | False |

## Still Needed Before Writing
- Confirm Gemma sensitivity only if gated model access is available and alignment safeguards pass.
- Decide whether the cluster-robust interaction fallback is sufficient for the paper or whether a slower dedicated mixed-model run is needed.
- Keep parser-syntax claims deferred while parser status remains `surface_heuristic_fallback`.

## Exclude From Main Paper
- Exposure-count and text-assignment proxy models.
- Random word-level predictive evaluations.
- Standalone segmentation main-effect framing.
- Clinical screening or diagnostic claims.
