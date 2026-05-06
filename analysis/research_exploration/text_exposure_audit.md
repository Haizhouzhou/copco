# Text Exposure Audit

This audit quantifies reader-group exposure in the prepared dataset. Exposure-count variables are documented as confounds and are excluded from primary predictive feature sets.

## Exposure By Reader Group
| reader_group | participants | word_rows | speeches_read | mean_words_per_participant | mean_speeches_per_participant | mean_word_length_chars | mean_log_corpus_frequency | mean_dfm_lm_word_surprisal | mean_dfm_lm_word_entropy | mean_sentence_length_words | mean_prev_boundary_opacity_score | vv_exposure_rate | embedding_missingness | lm_missingness | parser_status | mean_comprehension_score | mean_age | sex_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dyslexia_labeled | 19 | 82179 | 19 | 4325.2105 | 2.8947 | 4.7128 | 3.7630 | 3.4457 | 1.6010 | 23.3694 | 0.9021 | 0.0746 | 0.0405 | 0.0054 | surface_heuristic_fallback | 0.8050 | 37.1385 | F:12, M:7 |
| typical_control | 38 | 253024 | 32 | 6658.5263 | 4.9211 | 4.6874 | 3.7664 | 3.4333 | 1.5989 | 24.1849 | 0.9013 | 0.0750 | 0.0425 | 0.0052 | surface_heuristic_fallback | 0.8158 | 29.9377 | F:28, M:10 |

## Variables Flagged As Exposure Counts
- `n_speeches`
- `n_word_rows`
- `n_words_read`
- `total_word_rows`
- `word_observation_count`

## Interpretation
Reader groups differ in amount of text exposure and speech coverage. Later models should control text/stimulus predictors and should not treat exposure-count variables as primary predictive features.
