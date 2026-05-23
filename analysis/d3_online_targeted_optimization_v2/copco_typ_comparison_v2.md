# CopCo TYP Comparison v2

This table compares project-specific v2 locked rows to published CopCo_TYP baseline central values. The internal simple means are not official EyeBench average columns.

| final_model | split_regime | baseline_model | D3_BA | baseline_BA | beats_BA | D3_AUROC | baseline_AUROC | beats_AUROC | official_comparable_average | internal_simple_mean_D3_BA | internal_simple_mean_D3_AUROC | internal_simple_mean_baseline_BA | internal_simple_mean_baseline_AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_offline_all_full_evidence | unseen_reader | AhnCNN | 0.8158 | 0.7770 | True | 0.8989 | 0.8530 | True | False | 0.8158 | 0.8989 | 0.7770 | 0.8530 |
| best_offline_all_full_evidence | unseen_reader | Random Forest | 0.8158 | 0.6980 | True | 0.8989 | 0.8010 | True | False | 0.8158 | 0.8989 | 0.7770 | 0.8530 |
| best_offline_all_full_evidence | unseen_reader | Logistic Regression | 0.8158 | 0.7550 | True | 0.8989 | 0.8310 | True | False | 0.8158 | 0.8989 | 0.7770 | 0.8530 |
| best_online_late_accumulation | unseen_reader | AhnCNN | 0.6842 | 0.7770 | False | 0.7784 | 0.8530 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_reader | Random Forest | 0.6842 | 0.6980 | False | 0.7784 | 0.8010 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_reader | Logistic Regression | 0.6842 | 0.7550 | False | 0.7784 | 0.8310 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_reader_and_text | AhnCNN | 0.5833 | 0.6560 | False | 0.7014 | 0.7490 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_reader_and_text | Random Forest | 0.5833 | 0.5970 | False | 0.7014 | 0.6590 | True | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_reader_and_text | Logistic Regression | 0.5833 | 0.6350 | False | 0.7014 | 0.6890 | True | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_text | AhnCNN | 0.7387 | 0.7750 | False | 0.7647 | 0.8570 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_text | Random Forest | 0.7387 | 0.8150 | False | 0.7647 | 0.9150 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_text | Logistic Regression | 0.7387 | 0.7660 | False | 0.7647 | 0.8330 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader | AhnCNN | 0.7763 | 0.7770 | False | 0.7950 | 0.8530 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader | Random Forest | 0.7763 | 0.6980 | True | 0.7950 | 0.8010 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader | Logistic Regression | 0.7763 | 0.7550 | True | 0.7950 | 0.8310 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader_and_text | AhnCNN | 0.7014 | 0.6560 | True | 0.7639 | 0.7490 | True | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader_and_text | Random Forest | 0.7014 | 0.5970 | True | 0.7639 | 0.6590 | True | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader_and_text | Logistic Regression | 0.7014 | 0.6350 | True | 0.7639 | 0.6890 | True | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_text | AhnCNN | 0.6828 | 0.7750 | False | 0.7696 | 0.8570 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_text | Random Forest | 0.6828 | 0.8150 | False | 0.7696 | 0.9150 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_text | Logistic Regression | 0.6828 | 0.7660 | False | 0.7696 | 0.8330 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_reader | AhnCNN | 0.7632 | 0.7770 | False | 0.7770 | 0.8530 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_reader | Random Forest | 0.7632 | 0.6980 | True | 0.7770 | 0.8010 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_reader | Logistic Regression | 0.7632 | 0.7550 | True | 0.7770 | 0.8310 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_reader_and_text | AhnCNN | 0.8194 | 0.6560 | True | 0.8333 | 0.7490 | True | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_reader_and_text | Random Forest | 0.8194 | 0.5970 | True | 0.8333 | 0.6590 | True | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_reader_and_text | Logistic Regression | 0.8194 | 0.6350 | True | 0.8333 | 0.6890 | True | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_text | AhnCNN | 0.6447 | 0.7750 | False | 0.6884 | 0.8570 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_text | Random Forest | 0.6447 | 0.8150 | False | 0.6884 | 0.9150 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_text | Logistic Regression | 0.6447 | 0.7660 | False | 0.6884 | 0.8330 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader | AhnCNN | 0.4958 | 0.7770 | False | 0.5177 | 0.8530 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader | Random Forest | 0.4958 | 0.6980 | False | 0.5177 | 0.8010 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader | Logistic Regression | 0.4958 | 0.7550 | False | 0.5177 | 0.8310 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader_and_text | AhnCNN | 0.5000 | 0.6560 | False | 0.7857 | 0.7490 | True | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader_and_text | Random Forest | 0.5000 | 0.5970 | False | 0.7857 | 0.6590 | True | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader_and_text | Logistic Regression | 0.5000 | 0.6350 | False | 0.7857 | 0.6890 | True | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_text | AhnCNN | 0.5597 | 0.7750 | False | 0.6165 | 0.8570 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_text | Random Forest | 0.5597 | 0.8150 | False | 0.6165 | 0.9150 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_text | Logistic Regression | 0.5597 | 0.7660 | False | 0.6165 | 0.8330 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_unseen_text_specialist | unseen_text | AhnCNN | 0.7546 | 0.7750 | False | 0.8639 | 0.8570 | True | False | 0.7546 | 0.8639 | 0.8150 | 0.9150 |
| best_unseen_text_specialist | unseen_text | Random Forest | 0.7546 | 0.8150 | False | 0.8639 | 0.9150 | False | False | 0.7546 | 0.8639 | 0.8150 | 0.9150 |
| best_unseen_text_specialist | unseen_text | Logistic Regression | 0.7546 | 0.7660 | False | 0.8639 | 0.8330 | True | False | 0.7546 | 0.8639 | 0.8150 | 0.9150 |
| best_offline_all_full_evidence | unseen_reader | best_provided_baseline | 0.8158 | 0.7770 | True | 0.8989 | 0.8530 | True | False | 0.8158 | 0.8989 | 0.7770 | 0.8530 |
| best_online_early_detection | unseen_reader | best_provided_baseline | 0.7632 | 0.7770 | False | 0.7770 | 0.8530 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_reader_and_text | best_provided_baseline | 0.8194 | 0.6560 | True | 0.8333 | 0.7490 | True | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_early_detection | unseen_text | best_provided_baseline | 0.6447 | 0.8150 | False | 0.6884 | 0.9150 | False | False | 0.7424 | 0.7662 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_reader | best_provided_baseline | 0.6842 | 0.7770 | False | 0.7784 | 0.8530 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_reader_and_text | best_provided_baseline | 0.5833 | 0.6560 | False | 0.7014 | 0.7490 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_late_accumulation | unseen_text | best_provided_baseline | 0.7387 | 0.8150 | False | 0.7647 | 0.9150 | False | False | 0.6687 | 0.7482 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader | best_provided_baseline | 0.7763 | 0.7770 | False | 0.7950 | 0.8530 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_reader_and_text | best_provided_baseline | 0.7014 | 0.6560 | True | 0.7639 | 0.7490 | True | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_mid_detection | unseen_text | best_provided_baseline | 0.6828 | 0.8150 | False | 0.7696 | 0.9150 | False | False | 0.7202 | 0.7762 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader | best_provided_baseline | 0.4958 | 0.7770 | False | 0.5177 | 0.8530 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_reader_and_text | best_provided_baseline | 0.5000 | 0.6560 | False | 0.7857 | 0.7490 | True | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_online_stopping_detector | unseen_text | best_provided_baseline | 0.5597 | 0.8150 | False | 0.6165 | 0.9150 | False | False | 0.5185 | 0.6400 | 0.7493 | 0.8390 |
| best_unseen_text_specialist | unseen_text | best_provided_baseline | 0.7546 | 0.8150 | False | 0.8639 | 0.9150 | False | False | 0.7546 | 0.8639 | 0.8150 | 0.9150 |
