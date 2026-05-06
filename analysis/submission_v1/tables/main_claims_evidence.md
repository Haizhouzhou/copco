# Main claims and evidence ledger summary.

| claim_id | claim_text | claim_category | metric_statistic | status |
| --- | --- | --- | --- | --- |
| C01 | DFM residual gaze profiles predict participant group. | main | ROC-AUC 0.8947; PR-AUC 0.8641 | supported |
| C02 | DFM sensitivity dominates DFM exposure. | supporting | D1 ROC-AUC 0.4238; D2 0.8892; D3 0.8947 | supported |
| C03 | Prediction survives permutation and bootstrap robustness. | supporting | Permutation p=0.000999; ROC-AUC CI [0.7765, 0.9841] | supported |
| C04 | Cross-fitted residualization avoids using held-out participants in residual fitting. | supporting | LOPO residualization described and validated. | supported |
| C05 | Exposure-count variables are absent from the primary model. | supporting | No prohibited exposure-count variables in D3. | supported |
| C06 | Raw speed does not dominate. | supporting | D3 uses DFM residual gaze slopes, not raw speed/global duration aggregates. | supported |
| C07 | DFM surprisal interactions provide explanatory support. | secondary | Several controlled DFM surprisal interactions survive. | partially_supported |
| C08 | Boundary opacity is secondary. | secondary | Boundary-opacity interaction retained only for interpretation. | partially_supported |
| C09 | Standalone segmentation main effect is not supported. | appendix | Standalone segmentation framing dropped. | appendix_only |
| C10 | Word-level classification is secondary. | appendix | Participant label is the target; participant-level prediction is primary. | appendix_only |
