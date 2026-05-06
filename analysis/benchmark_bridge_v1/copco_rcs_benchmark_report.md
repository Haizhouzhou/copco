# CopCo RCS BenchmarkBridge Report

Target scale: raw_project_scale_0_1; EyeBench-compatible 1-10 = 1 + 9 * raw. RCS is auxiliary and does not affect the main TYP claim.

## Reader-Aggregated D3 Metrics
| split_name | n_predictions | rmse | mae | r2 | skipped_folds |
| --- | --- | --- | --- | --- | --- |
| unseen_reader | 57 | 0.1374 | 0.1065 | 0.0031 | 0 |
| unseen_text | 242 | 0.1196 | 0.0904 | 0.0116 | 0 |
| unseen_reader_and_text | 34 | 0.1524 | 0.1211 | -0.0164 | 0 |
