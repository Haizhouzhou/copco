# CopCo TYP BenchmarkBridge Report

Primary model: D3 DFM residual gaze-only logistic regression. Metrics are internal EyeBench-style unless the compatibility report states that official folds were used.

## Reader-Aggregated D3 Metrics
| split_name | n_predictions | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | skipped_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unseen_reader | 57 | 0.8961 | 0.8738 | 0.8158 | 0.8074 | 0.1326 | 0 |
| unseen_text | 242 | 0.8285 | 0.5548 | 0.7444 | 0.6935 | 0.1676 | 0 |
| unseen_reader_and_text | 34 | 0.8542 | 0.7740 | 0.7458 | 0.7312 | 0.1501 | 0 |
| text_balanced_unseen_reader | 57 | 0.9100 | 0.8979 | 0.8553 | 0.8459 | 0.1233 | 0 |
| leave_one_speech_out | 242 | 0.8285 | 0.5548 | 0.7444 | 0.6935 | 0.1676 | 0 |
| participant_grouped_kfold | 57 | 0.9100 | 0.8979 | 0.8553 | 0.8459 | 0.1233 | 0 |
