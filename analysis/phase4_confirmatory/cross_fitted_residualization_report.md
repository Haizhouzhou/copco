# Cross-Fitted Residualization Report

Each participant's expected gaze model is fit on all other participants and then applied to that held-out participant's word rows. Reader group, participant labels, and participant identifiers are never residualization predictors.

## Residualization Predictors
- `word_length_chars`
- `log_corpus_frequency`
- `dfm_lm_word_surprisal`
- `dfm_lm_word_entropy`
- `sentence_length_words`
- `word_position_in_sentence_norm`
- `prev_boundary_opacity_score`
- `vocoid_run_cross_boundary`
- `vv_indicator`
- `lm_missing`
- `embedding_missing`
- `parser_missing`
- `segmentation_label_missing`
- `speech_id`

## Fold Diagnostics By Outcome
| outcome | folds | complete_folds | skipped_folds | uses_reader_group | train_contains_heldout |
| --- | --- | --- | --- | --- | --- |
| fixation_count | 57 | 57 | 0 | False | False |
| log_ffd | 57 | 57 | 0 | False | False |
| log_first_pass_duration | 57 | 57 | 0 | False | False |
| log_go_past_time | 57 | 57 | 0 | False | False |
| log_total_fixation_duration | 57 | 57 | 0 | False | False |
| skip | 57 | 57 | 0 | False | False |

## Validation
- Held-out participant rows used to fit their own residual model: False
- Reader-group variables used in residualization: False
- Participant profiles produced: 57

## Output
- `participant_sensitivity_profiles_crossfit.parquet`
