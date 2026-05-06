| model | unseen_reader_balanced_accuracy | unseen_text_balanced_accuracy | unseen_reader_text_balanced_accuracy | average_balanced_accuracy | unseen_reader_AUROC | unseen_text_AUROC | unseen_reader_text_AUROC | average_AUROC | evaluation_level | official_mode | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3_dfm_residual_gaze_only | 0.8158 | 0.7444 | 0.7458 | 0.7687 | 0.8961 | 0.8285 | 0.8542 | 0.8596 | reader_aggregated | False | BenchmarkBridge internal EyeBench-style split. |
| Chance | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | EyeBench reported central value | True | Analytic chance reference. |
| Reading Speed |  |  |  |  |  |  |  |  | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| Logistic Regression |  |  |  |  |  |  |  |  | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| SVM |  |  |  |  |  |  |  |  | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| Random Forest |  |  |  |  |  |  |  |  | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| AhnCNN | 0.7770 |  | 0.6560 | 0.7165 | 0.8530 |  | 0.7490 | 0.8010 | EyeBench reported central value | True | Gate central values supplied in BenchmarkBridge v1 request. |
| BEyeLSTM |  |  |  |  |  |  |  |  | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| RoBERTEye-W |  |  |  |  |  |  |  |  | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| best_reported_baseline | 0.7770 |  | 0.6560 | 0.7165 | 0.8530 |  | 0.7490 | 0.8010 | EyeBench reported central value | True | Gate central values supplied in BenchmarkBridge v1 request. |
