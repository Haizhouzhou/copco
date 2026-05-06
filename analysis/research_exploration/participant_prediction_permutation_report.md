# Participant Prediction Permutation Report

The permutation test shuffles participant target labels and reruns the selected participant-level ablation. It is a robustness screen, not final model optimization.

| selected_feature_group | selected_model | selected_split | observed_roc_auc | permutation_count | permutation_p_value | bootstrap_roc_auc_low | bootstrap_roc_auc_high | leave_one_dyslexia_min_roc_auc | leave_one_dyslexia_max_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_dfm_exposure_and_sensitivity | logistic_regression | leave_one_participant_out | 0.9058 | 100 | 0.0099 | 0.8162 | 0.9798 | 0.8977 | 0.9459 |
