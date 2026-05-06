# LM Warning Audit

The DFM LM warning `non_special_token_unassigned` is treated as an audit flag. Rows are not excluded automatically because most contexts inherit this warning and validation found no alignment errors.

## Warning And Missingness By Reader Group
| reader_group | rows | warning_rows | lm_missing_rate | mean_word_length | mean_surprisal | mean_entropy | warning_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dyslexia_labeled | 82179 | 82010 | 0.0054 | 4.7128 | 3.4457 | 1.6010 | 0.9979 |
| typical_control | 253024 | 252506 | 0.0052 | 4.6874 | 3.4333 | 1.5989 | 0.9980 |

## Warning And Missingness By Speech
| speech_id | rows | warning_rows | lm_missing_rate | warning_rate |
| --- | --- | --- | --- | --- |
| 10365 | 15430 | 15430 | 0.0084 | 1.0000 |
| 10440 | 7896 | 7896 | 0.0030 | 1.0000 |
| 11171 | 9320 | 9320 | 0.0019 | 1.0000 |
| 1125 | 14697 | 14670 | 0.0061 | 0.9982 |
| 1165 | 12859 | 12804 | 0.0034 | 0.9957 |
| 12063 | 9740 | 9550 | 0.0041 | 0.9805 |
| 1317 | 14562 | 14562 | 0.0062 | 1.0000 |
| 1318 | 19574 | 19574 | 0.0042 | 1.0000 |
| 1323 | 19943 | 19943 | 0.0088 | 1.0000 |
| 17526 | 21834 | 21834 | 0.0045 | 1.0000 |
| 18473 | 8730 | 8680 | 0.0115 | 0.9943 |
| 18561 | 17392 | 17344 | 0.0018 | 0.9972 |
| 18670 | 11000 | 11000 | 0.0045 | 1.0000 |
| 202150 | 101 | 101 | 0.0099 | 1.0000 |
| 202151 | 154 | 154 | 0.0065 | 1.0000 |
| 202152 | 116 | 116 | 0.0000 | 1.0000 |
| 202201 | 300 | 300 | 0.0000 | 1.0000 |
| 202202 | 306 | 306 | 0.0000 | 1.0000 |
| 202203 | 189 | 189 | 0.0000 | 1.0000 |
| 202204 | 304 | 304 | 0.0000 | 1.0000 |
| 202205 | 286 | 286 | 0.0140 | 1.0000 |
| 202206 | 402 | 402 | 0.0000 | 1.0000 |
| 202207 | 314 | 314 | 0.0000 | 1.0000 |
| 202208 | 284 | 284 | 0.0000 | 1.0000 |
| 202209 | 264 | 264 | 0.0000 | 1.0000 |
| 22811 | 14304 | 14148 | 0.0067 | 0.9891 |
| 26670 | 19940 | 19940 | 0.0065 | 1.0000 |
| 26682 | 26728 | 26728 | 0.0068 | 1.0000 |
| 7797 | 15158 | 15158 | 0.0044 | 1.0000 |
| 7856 | 18520 | 18520 | 0.0032 | 1.0000 |
| 7905 | 37335 | 37207 | 0.0051 | 0.9966 |
| 7946 | 17221 | 17188 | 0.0034 | 0.9981 |

## Warning Examples
| speech_id | sentence_id | word_id | word | lm_alignment_status | lm_alignment_warning | lm_missing |
| --- | --- | --- | --- | --- | --- | --- |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w0 | I | warning | non_special_token_unassigned | True |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w1 | slutningen | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w2 | af | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w3 | juli | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w4 | 1852 | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w5 | spadserede | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w6 | en | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w7 | ung | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w8 | kvinde | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w9 | på | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w10 | stierne | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w11 | her | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w12 | ved | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s612 | 7905_p0_s612_w13 | Rønnebæksholm. | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s613 | 7905_p0_s613_w14 | Hun | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s613 | 7905_p0_s613_w15 | øvede | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s613 | 7905_p0_s613_w16 | sig | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s613 | 7905_p0_s613_w17 | på | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s613 | 7905_p0_s613_w18 | at | warning | non_special_token_unassigned | False |
| 7905 | 7905_p0_s613 | 7905_p0_s613_w19 | holde | warning | non_special_token_unassigned | False |

## Do Warnings Cause Missing LM Values?
| lm_alignment_warning | rows | lm_missing_rate |
| --- | --- | --- |
| non_special_token_unassigned | 334516 | 0.0052 |
|  | 687 | 0.0160 |

## Distribution Shift If LM-Missing Rows Are Excluded
| feature | all_rows_mean | lm_complete_mean | lm_missing_mean |
| --- | --- | --- | --- |
| word_length_chars | 4.6936 | 4.7064 | 2.2737 |
| log_corpus_frequency | 3.7655 | 3.7550 | 5.7478 |
| sentence_length_words | 23.9850 | 24.0061 | 20.0057 |
| prev_boundary_opacity_score | 0.9015 | 0.9015 |  |
