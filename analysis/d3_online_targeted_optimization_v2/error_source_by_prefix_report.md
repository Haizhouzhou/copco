# Error Source by Prefix Report

- Error rows: 7076
- Learned meta persistent FP/FN rows: 596
- Mean probability persistent FP/FN rows: 720
- DFM sensitivity stabilization proxy: unstable prefix rate falls when `stable_enough_for_prediction` increases.

## Prefix Error Summary

| split_regime | prefix_type | prefix_value | accumulator | rows | false_positives | false_negatives | corrected_by_more_evidence | persistent_errors | unstable_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen_text | word_count_prefix | 250 | learned_meta_aggregator | 57 | 8 | 6 | 15 | 29 | 0.0000 |
| unseen_text | word_count_prefix | 50 | mean_probability | 57 | 11 | 2 | 15 | 28 | 0.0000 |
| unseen_text | chronological_prefix | 50 | mean_probability | 57 | 11 | 2 | 15 | 28 | 0.0000 |
| unseen_text | word_count_prefix | 250 | mean_probability | 57 | 11 | 2 | 14 | 27 | 0.0000 |
| unseen_text | chronological_prefix | 250 | learned_meta_aggregator | 57 | 8 | 6 | 13 | 27 | 0.0000 |
| unseen_text | word_count_prefix | 100 | mean_probability | 57 | 11 | 2 | 14 | 27 | 0.0000 |
| unseen_text | word_count_prefix | 50 | learned_meta_aggregator | 57 | 10 | 2 | 15 | 27 | 0.0000 |
| unseen_text | chronological_prefix | 250 | mean_probability | 57 | 11 | 2 | 14 | 27 | 0.0000 |
| unseen_text | chronological_prefix | 100 | mean_probability | 57 | 11 | 2 | 13 | 26 | 0.0000 |
| unseen_text | word_count_prefix | 500 | mean_probability | 56 | 10 | 2 | 14 | 26 | 0.0000 |
| unseen_text | chronological_prefix | 500 | mean_probability | 56 | 9 | 2 | 12 | 23 | 0.0000 |
| unseen_text | trial_or_text_prefix | 1 | learned_meta_aggregator | 57 | 9 | 7 | 5 | 21 | 0.0000 |
| unseen_text | chronological_prefix | 50 | learned_meta_aggregator | 57 | 1 | 7 | 12 | 20 | 0.0000 |
| unseen_text | word_count_prefix | 100 | learned_meta_aggregator | 57 | 3 | 7 | 10 | 20 | 0.0000 |
| unseen_text | word_count_prefix | 1000 | learned_meta_aggregator | 52 | 9 | 5 | 6 | 20 | 0.0000 |
| unseen_text | chronological_prefix | 100 | learned_meta_aggregator | 57 | 2 | 7 | 11 | 20 | 0.0000 |
| unseen_text | chronological_prefix | 500 | learned_meta_aggregator | 56 | 6 | 7 | 7 | 20 | 0.0000 |
| unseen_text | word_count_prefix | 500 | learned_meta_aggregator | 56 | 5 | 7 | 7 | 19 | 0.0000 |
| participant_grouped_kfold | chronological_prefix | 50 | learned_meta_aggregator | 57 | 4 | 6 | 8 | 18 | 0.0000 |
| unseen_text | speech_prefix | 1 | learned_meta_aggregator | 57 | 9 | 7 | 2 | 18 | 0.0000 |
| participant_grouped_kfold | word_count_prefix | 50 | learned_meta_aggregator | 57 | 4 | 6 | 8 | 18 | 0.0000 |
| unseen_reader | chronological_prefix | 50 | learned_meta_aggregator | 57 | 4 | 6 | 8 | 18 | 0.0000 |
| unseen_reader | chronological_prefix | 100 | mean_probability | 57 | 9 | 2 | 7 | 18 | 0.0000 |
| text_balanced_unseen_reader | chronological_prefix | 50 | learned_meta_aggregator | 57 | 4 | 6 | 8 | 18 | 0.0000 |
| text_balanced_unseen_reader | word_count_prefix | 50 | learned_meta_aggregator | 57 | 4 | 6 | 8 | 18 | 0.0000 |
| unseen_reader | word_count_prefix | 50 | learned_meta_aggregator | 57 | 4 | 6 | 8 | 18 | 0.0000 |
| text_balanced_unseen_reader | chronological_prefix | 100 | mean_probability | 57 | 9 | 2 | 7 | 18 | 0.0000 |
| participant_grouped_kfold | chronological_prefix | 100 | mean_probability | 57 | 9 | 2 | 7 | 18 | 0.0000 |
| unseen_text | chronological_prefix | 1000 | learned_meta_aggregator | 52 | 8 | 5 | 4 | 17 | 0.0000 |
| unseen_text | chronological_prefix | 1000 | mean_probability | 52 | 10 | 1 | 6 | 17 | 0.0000 |

## Text/Speech Concentrations

| split_regime | terminal_text_id | wrong_rows |
| --- | --- | --- |
| unseen_text | 7905 | 229 |
| unseen_text | 1323 | 86 |
| participant_grouped_kfold | 7905 | 68 |
| text_balanced_unseen_reader | 7905 | 68 |
| unseen_reader | 7905 | 68 |
| unseen_reader | 1125 | 62 |
| text_balanced_unseen_reader | 1125 | 62 |
| participant_grouped_kfold | 1125 | 62 |
| participant_grouped_kfold | 7946 | 61 |
| unseen_reader | 7946 | 61 |
| text_balanced_unseen_reader | 7946 | 61 |
| participant_grouped_kfold | 1165 | 53 |
| text_balanced_unseen_reader | 1165 | 53 |
| unseen_reader | 1165 | 53 |
| unseen_text | 7946 | 50 |
| unseen_reader_and_text | 7905 | 49 |
| unseen_reader_and_text | 7946 | 46 |
| unseen_text | 1165 | 45 |
| unseen_text | 1125 | 44 |
| unseen_text | 11171 | 43 |
