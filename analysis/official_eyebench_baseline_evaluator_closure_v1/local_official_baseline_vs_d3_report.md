# Official Baseline vs D3 Comparison Report

- Primary comparison uses official trial-level metrics.
- Reader-aggregated values are secondary and are not used for official SOTA.
- Official command-source baseline pass: True
- Local official-derived baseline pass: True
- D3 beats local official-derived baseline: False
- D3 beats strongest published baseline: False

## Local Official-Derived Logistic Baseline
| model_name | baseline_source | split_name | metric_basis | n_predictions | usable_folds | roc_auc | balanced_accuracy | pr_auc | macro_f1 | brier_score | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegressionMLArgs | local_official_derived_eyebench_classes | unseen_reader | official_trial_level_fold_mean | 3554 | 4 | 0.8304 | 0.7541 | 0.5583 | 0.6896 | 0.1843 | complete |
| LogisticRegressionMLArgs | local_official_derived_eyebench_classes | unseen_reader_and_text | official_trial_level_fold_mean | 1228 | 4 | 0.6910 | 0.6380 | 0.5314 | 0.6094 | 0.2368 | complete |
| LogisticRegressionMLArgs | local_official_derived_eyebench_classes | unseen_text | official_trial_level_fold_mean | 3554 | 4 | 0.8315 | 0.7665 | 0.5049 | 0.6880 | 0.1927 | complete |

## D3 Trial-Level Primary Metrics
| mode | model_name | claim_type | task | split_name | evaluation_level | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_eyebench_subset | D3_EyeBench_Lite | official_compatible | CopCo_TYP | unseen_reader | official_trial_level_fold_mean | 12 | 3554 | 4 | 0 | 0.8085 | 0.5614 | 0.7274 | 0.6767 | 0.1904 | complete |  |
| official_eyebench_subset | D3_EyeBench_Lite | official_compatible | CopCo_TYP | unseen_text | official_trial_level_fold_mean | 12 | 3554 | 4 | 0 | 0.8319 | 0.5434 | 0.7341 | 0.6751 | 0.1871 | complete |  |
| official_eyebench_subset | D3_EyeBench_Lite | official_compatible | CopCo_TYP | unseen_reader_and_text | official_trial_level_fold_mean | 12 | 1228 | 4 | 0 | 0.7154 | 0.5650 | 0.6342 | 0.6223 | 0.2191 | complete |  |

## D3 Reader-Aggregated Secondary Metrics
| mode | model_name | claim_type | task | split_name | evaluation_level | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_eyebench_subset | D3_EyeBench_Lite | official_compatible | CopCo_TYP | unseen_reader | reader_aggregated | 12 | 55 | 4 | 0 | 0.8468 | 0.7334 | 0.7530 | 0.7418 | 0.1783 | complete |  |
| official_eyebench_subset | D3_EyeBench_Lite | official_compatible | CopCo_TYP | unseen_text | reader_aggregated | 12 | 113 | 4 | 0 | 0.8606 | 0.6722 | 0.7629 | 0.7268 | 0.1673 | complete |  |
| official_eyebench_subset | D3_EyeBench_Lite | official_compatible | CopCo_TYP | unseen_reader_and_text | reader_aggregated | 12 | 39 | 4 | 0 | 0.7792 | 0.6482 | 0.6201 | 0.6201 | 0.1893 | complete |  |

## Published EyeBench CopCo_TYP References
| model | mode | claim_type | metric_basis | official_mode | exact_folds | exact_processed_data | unseen_reader_balanced_accuracy | unseen_text_balanced_accuracy | unseen_reader_text_balanced_accuracy | average_balanced_accuracy | unseen_reader_AUROC | unseen_text_AUROC | unseen_reader_text_AUROC | average_AUROC | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority Class / Chance | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.5030 | 0.4960 | 0.5010 | 0.5000 | 0.5030 | 0.4960 | 0.5010 | 0.5000 | Published EyeBench formatted CopCo_TYP test table central value. |
| Reading Speed | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.5770 | 0.5490 | 0.5060 | 0.5440 | 0.6070 | 0.5620 | 0.5090 | 0.5593 | Published EyeBench formatted CopCo_TYP test table central value. |
| Text-Only Roberta | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4700 | 0.5000 | 0.5040 | 0.4913 | Published EyeBench formatted CopCo_TYP test table central value. |
| Logistic Regression~\cite{meziere2023using} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.7550 | 0.7660 | 0.6350 | 0.7187 | 0.8310 | 0.8330 | 0.6890 | 0.7843 | Published EyeBench formatted CopCo_TYP test table central value. |
| SVM~\cite{hollenstein2023zuco} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.7070 | 0.7740 | 0.6470 | 0.7093 | 0.7070 | 0.7740 | 0.6470 | 0.7093 | Published EyeBench formatted CopCo_TYP test table central value. |
| Random Forest~\cite{makowski2024detection} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.6980 | 0.8150 | 0.5970 | 0.7033 | 0.8010 | 0.9150 | 0.6590 | 0.7917 | Published EyeBench formatted CopCo_TYP test table central value. |
| AhnRNN~\citep{ahn2020towards} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5010 | 0.5000 | 0.5000 | 0.5003 | Published EyeBench formatted CopCo_TYP test table central value. |
| AhnCNN~\citep{ahn2020towards} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.7770 | 0.7750 | 0.6560 | 0.7360 | 0.8530 | 0.8570 | 0.7490 | 0.8197 | Published EyeBench formatted CopCo_TYP test table central value. |
| BEyeLSTM~\citep{reich_inferring_2022} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.7190 | 0.7680 | 0.6470 | 0.7113 | 0.7940 | 0.8500 | 0.6920 | 0.7787 | Published EyeBench formatted CopCo_TYP test table central value. |
| PLM-AS~\citep{Yang2023PLMASPL} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.5520 | 0.5730 | 0.5590 | 0.5613 | 0.5760 | 0.5850 | 0.5940 | 0.5850 | Published EyeBench formatted CopCo_TYP test table central value. |
| PLM-AS-RM~\citep{haller2022eye} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.6090 | 0.7160 | 0.5460 | 0.6237 | 0.6390 | 0.8010 | 0.5500 | 0.6633 | Published EyeBench formatted CopCo_TYP test table central value. |
| RoBERTEye-W~\citep{Shubi2024Finegrained} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.7000 | 0.6850 | 0.6190 | 0.6680 | 0.7830 | 0.7670 | 0.6820 | 0.7440 | Published EyeBench formatted CopCo_TYP test table central value. |
| RoBERTEye-F~\citep{Shubi2024Finegrained} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.6060 | 0.6030 | 0.5400 | 0.5830 | 0.7190 | 0.7470 | 0.6330 | 0.6997 | Published EyeBench formatted CopCo_TYP test table central value. |
| MAG-Eye~\citep{Shubi2024Finegrained} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.4720 | 0.4970 | 0.5140 | 0.4943 | 0.4590 | 0.5470 | 0.5610 | 0.5223 | Published EyeBench formatted CopCo_TYP test table central value. |
| PostFusion-Eye~\citep{Shubi2024Finegrained} | official_eyebench_reported_baseline | official_reported_reference | published_fold_mean | True | True | True | 0.6470 | 0.6890 | 0.5700 | 0.6353 | 0.7310 | 0.7810 | 0.6550 | 0.7223 | Published EyeBench formatted CopCo_TYP test table central value. |

## Previous Local Diagnostic Baseline
| model_name | baseline_source | split_name | metric_basis | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | balanced_accuracy | published_roc_auc | published_balanced_accuracy | delta_roc_auc | delta_balanced_accuracy | status | skip_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_reader | official_trial_level_fold_mean | 5 | 3554 | 4 | 0 | 0.8304 | 0.7541 | 0.8310 | 0.7550 | -0.0006 | -0.0009 | complete |  |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_text | official_trial_level_fold_mean | 5 | 3554 | 4 | 0 | 0.8315 | 0.7665 | 0.8330 | 0.7660 | -0.0015 | 0.0005 | complete |  |
| LogisticRegressionMLArgs | official_processed_data_local_reproduction | unseen_reader_and_text | official_trial_level_fold_mean | 5 | 1228 | 4 | 0 | 0.6910 | 0.6380 | 0.6890 | 0.6350 | 0.0020 | 0.0030 | complete |  |
