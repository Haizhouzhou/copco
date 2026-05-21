# D3_EyeBench_Lite Feature Report

- Status: complete
- Feature source: official EyeBench processed CopCo data only.
- Full D3 is not claimed unless all DFM residual profile inputs are available from official data.
- participant_id and speech_id are retained only for grouping/reporting, not as predictors.
- Held-out reader rows used for residual fitting: False
- Held-out text rows used for residual fitting: False
- Reader group used in residualization: False

## Prohibited Predictors
- `RCS_score`
- `comprehension_score`
- `dyslexia`
- `dyslexia_labeled`
- `eyebench_rcs_score`
- `group_label`
- `n_speeches`
- `n_word_rows`
- `n_words_read`
- `participant_id`
- `reader_group`
- `reader_group_binary`
- `reader_group_binary_num`
- `speech_id`
- `text_id`
- `total_word_rows`
- `unique_paragraph_id`
- `unique_trial_id`
- `word_observation_count`
