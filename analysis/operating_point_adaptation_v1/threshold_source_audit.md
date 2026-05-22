# Threshold Source Audit

Status counts: {'passed': 4}

| check                                 | status | detail                                                                                                   |
| ------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------- |
| unknown_threshold_source              | passed | rows with unknown threshold source: 0                                                                    |
| ba_f1_threshold_source                | passed | metric rows missing threshold source: 0                                                                  |
| test_oracle_mixed_with_clean_metrics  | passed | Clean fixed/legal rows are separate from test_oracle_diagnostic rows.                                    |
| reader_aggregation_uses_probabilities | passed | Reader aggregation methods aggregate probabilities/logits; hard majority vote is labelled baseline only. |
