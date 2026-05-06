# EyeBench Structure Report

| area | finding |
| --- | --- |
| README.md | Documents CopCo_TYP/CopCo_RCS and three regimes. |
| environment.yml | Defines Python 3.12.10, PyTorch CUDA, Hydra, WandB, and preprocessing deps. |
| data/CopCo/folds_metadata | Subjects/items/trial_id fold files: {'subjects': ['fold_0.csv', 'fold_1.csv', 'fold_2.csv', 'fold_3.csv'], 'items': ['fold_0.csv', 'fold_1.csv', 'fold_2.csv', 'fold_3.csv'], 'trial_ids': ['fold_0_trial_ids_by_regime.csv', 'fold_1_trial_ids_by_regime.csv', 'fold_2_trial_ids_by_regime.csv', 'fold_3_trial_ids_by_regime.csv']} |
| data/CopCo/labels | {"participant_stats.csv": {"rows": 58, "columns": ["subj", "comprehension_accuracy", "number_of_speeches", "number_of_questions", "absolute_reading_time", "relative_reading_time", "words_per_minute", "age", "sex", "native_language", "vision", "score_reading_comprehension_test", "dyslexia", "pseudohomophone_score"]}, "stimuli_and_comp_results.csv": {"rows": 2506, "columns": ["Session_Name_", "Trial_Index_", "Trial_Recycled_", "condition", "text", "question", "expected_key", "QUESTION_KEY_PRESSED", "QUESTION_RT", "QUESTION_ACCURACY", "SENTENCE_RT", "speechid", "paragraphid", "counterbalance", "LAST_COUNTERBALANCE"]}, "word2char_IA_mapping.csv": {"rows": 32476, "columns": ["index", "part", "speechId", "paragraphId", "wordId", "word", "characters", "char_IA_ids", "sentenceId"]}} |
| src/run/multi_run/raw_to_processed_results.py | Computes AUROC/BA and RMSE/MAE/R2 from trial_level_test_results.csv. |
| src/data/preprocessing/create_folds.py | Creates train/val/test regimes from subject and item folds. |
| processed data | CopCo processed data present: False |

## Summary
- Official CopCo tasks: `CopCo_TYP`, `CopCo_RCS`.
- Official split names: `seen_subject_unseen_item`, `unseen_subject_seen_item`, `unseen_subject_unseen_item`.
- CopCo fold metadata include subjects, items, and trial IDs.
- Official result aggregation reads `trial_level_test_results.csv` files and emits metric CSVs.
- Official processed CopCo data are not present unless `data/CopCo/processed/` exists.
