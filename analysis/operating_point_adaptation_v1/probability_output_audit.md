# Probability Output Audit

Status counts: {'passed': 7}

| check                         | status | detail                                                                                                            |
| ----------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------- |
| p_pred_exists                 | passed | missing p_pred rows: 0                                                                                            |
| p_pred_in_unit_interval       | passed | outside [0,1]: 0                                                                                                  |
| auroc_pr_auc_score_source     | passed | Metrics are recomputed from normalized p_pred.                                                                    |
| y_pred_documented_threshold   | passed | Existing y_pred matches fixed 0.5 where y_pred is present; all metrics are recomputed with documented thresholds. |
| duplicate_prediction_keys     | passed | duplicate keys: 0                                                                                                 |
| missing_labels                | passed | missing labels: 0                                                                                                 |
| participant_speech_predictors | passed | Prediction tables expose participant_id/speech_id/text_id only as grouping identifiers.                           |
