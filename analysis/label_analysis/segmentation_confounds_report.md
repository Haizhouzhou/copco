# Segmentation Confounds Report

Segmentation labels are deterministic stimulus descriptors. This report quantifies associations with other stimulus variables for later controlled analyses.

## Correlations With Previous-Boundary Opacity
| feature | pearson_correlation_with_prev_opacity | feature_missing_rate |
| --- | --- | --- |
| word_length_chars | -0.043417024770034035 | 0.0 |
| log_corpus_frequency | 0.08753687653815559 | 0.0 |
| dfm_lm_word_surprisal | -0.04494777768129338 | 0.005189770524604514 |
| dfm_lm_word_entropy | -0.024919639600721438 | 0.005189770524604514 |
| sentence_length_words | 0.009213986583787276 | 0.0 |
| word_index_in_sentence | 0.017982995089819428 | 0.0 |
| long_word_lix_component | 0.004132616329044346 | 0.0 |

## Reader-Group Exposure After Joining To Participant Word Rows
| reader_group | word_rows | mean_prev_opacity | vv_exposure_rate |
| --- | --- | --- | --- |
| dyslexia-labeled reader | 82179 | 0.9020749665327978 | 0.07462977159614986 |
| typical_control | 253024 | 0.9012542374348961 | 0.07500079043885165 |

## Missingness
| field | missing_rate |
| --- | --- |
| prev_boundary_opacity_score | 0.08856999937472644 |
| segmentation_word_label | 0.0 |

## Recommendation
Segmentation labels are ready for exploratory modeling with caveats. Later analyses should control for word length, word frequency, surprisal, entropy, sentence length, word position, and text assignment because orthographic boundary opacity is not randomized.
Within-sentence boundary rows analyzed: 30000.
