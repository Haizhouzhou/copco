# Official Baseline Reproduction Report

This is a local reproduction using official processed CopCo data and official folds.
It is not accepted as a successful official baseline reproduction unless processed data exist and the local metrics are close to the published EyeBench table.

| model_name | baseline_source | split_name | metric_basis | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | balanced_accuracy | published_roc_auc | published_balanced_accuracy | delta_roc_auc | delta_balanced_accuracy | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_reader | official_trial_level_fold_mean | 5 | 3554 | 4 | 0 | 0.8304 | 0.7541 | 0.8310 | 0.7550 | -0.0006 | -0.0009 | complete |  |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_text | official_trial_level_fold_mean | 5 | 3554 | 4 | 0 | 0.8315 | 0.7665 | 0.8330 | 0.7660 | -0.0015 | 0.0005 | complete |  |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_reader_and_text | official_trial_level_fold_mean | 5 | 1228 | 4 | 0 | 0.6910 | 0.6380 | 0.6890 | 0.6350 | 0.0020 | 0.0030 | complete |  |
