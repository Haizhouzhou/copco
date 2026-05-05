# Segmentation Label Distribution Report

- Total boundary rows including sentence-initial rows: 31986
- Within-sentence boundary count: 30000
- Total word count: 31986
- Total sentence count: 1986

## Boundary Type Distribution
| value | count | proportion |
| --- | --- | --- |
| C#C | 14055 | 0.4685 |
| V#C | 6412 | 0.21373333333333333 |
| C#V | 6280 | 0.20933333333333334 |
| V#V | 2406 | 0.0802 |
| unknown | 847 | 0.028233333333333333 |

## Distribution By Speech
| speech_id | orth_boundary_type | count |
| --- | --- | --- |
| 10365 | C#C | 635 |
| 10365 | C#V | 312 |
| 10365 | V#C | 317 |
| 10365 | V#V | 103 |
| 10365 | unknown | 50 |
| 10440 | C#C | 428 |
| 10440 | C#V | 197 |
| 10440 | V#C | 185 |
| 10440 | V#V | 75 |
| 10440 | unknown | 16 |
| 11171 | C#C | 465 |
| 11171 | C#V | 198 |
| 11171 | V#C | 214 |
| 11171 | V#V | 78 |
| 11171 | unknown | 26 |
| 1125 | C#C | 741 |
| 1125 | C#V | 279 |
| 1125 | V#C | 284 |
| 1125 | V#V | 139 |
| 1125 | unknown | 54 |
| 1165 | C#C | 531 |
| 1165 | C#V | 221 |
| 1165 | V#C | 257 |
| 1165 | V#V | 87 |
| 1165 | unknown | 3 |
| 12063 | C#C | 493 |
| 12063 | C#V | 169 |
| 12063 | V#C | 176 |
| 12063 | V#V | 67 |
| 12063 | unknown | 10 |
| 1317 | C#C | 650 |
| 1317 | C#V | 348 |
| 1317 | V#C | 325 |
| 1317 | V#V | 142 |
| 1317 | unknown | 41 |
| 1318 | C#C | 808 |
| 1318 | C#V | 392 |
| 1318 | V#C | 403 |
| 1318 | V#V | 143 |
| 1318 | unknown | 42 |
| 1323 | C#C | 798 |
| 1323 | C#V | 357 |
| 1323 | V#C | 381 |
| 1323 | V#V | 135 |
| 1323 | unknown | 40 |
| 17526 | C#C | 978 |
| 17526 | C#V | 532 |
| 17526 | V#C | 528 |
| 17526 | V#V | 208 |
| 17526 | unknown | 67 |
| 18473 | C#C | 341 |
| 18473 | C#V | 208 |
| 18473 | V#C | 167 |
| 18473 | V#V | 67 |
| 18473 | unknown | 52 |
| 18561 | C#C | 491 |
| 18561 | C#V | 198 |
| 18561 | V#C | 201 |
| 18561 | V#V | 84 |
| 18561 | unknown | 28 |

## Distribution By Sentence Length Bin
| sentence_length_bin | sentences | mean_opacity | vv_rate |
| --- | --- | --- | --- |
| (-0.001, 5.0] | 247 | 0.7385693215339233 | 0.052968960863697706 |
| (10.0, 20.0] | 731 | 0.9133526569208849 | 0.08286698565669863 |
| (20.0, 40.0] | 458 | 0.9104314187129232 | 0.0826790976590843 |
| (40.0, inf] | 69 | 0.8866326017839443 | 0.07333009958303949 |
| (5.0, 10.0] | 481 | 0.8682679932679933 | 0.06857901857901857 |

## Boundary Examples
| orth_boundary_type | prev_word | word | boundary_opacity_score | boundary_id |
| --- | --- | --- | --- | --- |
| C#C | studenter, | kære | 0.0 | 10365_p0_s485_b3 |
| C#C | og | kære | 0.0 | 10365_p0_s485_b6 |
| C#C | et | smukt | 0.0 | 10365_p0_s486_b4 |
| C#C | smukt | syn. | 0.0 | 10365_p0_s486_b5 |
| C#C | bygning | for | 0.0 | 10365_p0_s487_b13 |
| C#V | Hvor | er | 1.0 | 10365_p0_s486_b1 |
| C#V | er | I | 1.0 | 10365_p0_s486_b2 |
| C#V | Det | er | 1.0 | 10365_p0_s487_b1 |
| C#V | folkeskolen | i | 1.0 | 10365_p0_s487_b18 |
| C#V | rygsækken | og | 1.0 | 10365_p0_s487_b20 |
| V#C | Kære | smukke | 2.0 | 10365_p0_s485_b1 |
| V#C | mange | flotte | 2.0 | 10365_p0_s485_b10 |
| V#C | flotte | hvide | 2.0 | 10365_p0_s485_b11 |
| V#C | hvide | huer. | 2.0 | 10365_p0_s485_b12 |
| V#C | smukke | studenter, | 2.0 | 10365_p0_s485_b2 |
| V#V | forældre | og | 3.0 | 10365_p0_s485_b5 |
| V#V | I | et | 3.0 | 10365_p0_s486_b3 |
| V#V | sommerfugle | i | 3.0 | 10365_p0_s487_b22 |
| V#V | startede | et | 3.0 | 10365_p0_s487_b27 |
| V#V | tre | år | 3.0 | 10365_p0_s487_b4 |
| unknown | i | 1955 | nan | 10365_p10_s538_b4 |
| unknown | 1955 | kun | nan | 10365_p10_s538_b5 |
| unknown | I | – | nan | 10365_p11_s543_b4 |
| unknown | – | det | nan | 10365_p11_s543_b5 |
| unknown | forældre | – | nan | 10365_p12_s547_b4 |

## Top Words Involved In V#V Boundaries
| word | count |
| --- | --- |
| i | 343 |
| og | 319 |
| at | 208 |
| er | 182 |
| en | 141 |
| vi | 111 |
| ikke | 108 |
| af | 104 |
| på | 100 |
| I | 96 |
| et | 65 |
| de | 59 |
| også | 58 |
| os | 45 |
| om | 45 |
| så | 42 |
| alle | 42 |
| være | 30 |
| eller | 28 |
| andre | 27 |

## High-Opacity Boundary Examples
| speech_id | sentence_id | prev_word | word | vocoid_run_cross_boundary |
| --- | --- | --- | --- | --- |
| 10365 | 10365_p0_s485 | forældre | og | 2 |
| 10365 | 10365_p0_s486 | I | et | 2 |
| 10365 | 10365_p0_s487 | tre | år | 2 |
| 10365 | 10365_p0_s487 | sommerfugle | i | 2 |
| 10365 | 10365_p0_s487 | startede | et | 2 |
| 10365 | 10365_p0_s488 | tre | år, | 2 |
| 10365 | 10365_p10_s538 | i | USA, | 2 |
| 10365 | 10365_p10_s538 | skabe | en | 2 |
| 10365 | 10365_p10_s539 | starte | en | 2 |
| 10365 | 10365_p10_s540 | i | øvrigt | 2 |
| 10365 | 10365_p11_s541 | sige, | at | 2 |
| 10365 | 10365_p11_s541 | stå | op | 2 |
| 10365 | 10365_p11_s542 | gøre | en | 2 |
| 10365 | 10365_p11_s543 | må | I | 2 |
| 10365 | 10365_p11_s544 | de | unge | 2 |
| 10365 | 10365_p12_s546 | i | øjnene: | 2 |
| 10365 | 10365_p12_s547 | være | ærlige, | 2 |
| 10365 | 10365_p12_s547 | tingene | og | 2 |
| 10365 | 10365_p12_s550 | Mere | end | 2 |
| 10365 | 10365_p12_s551 | fingre | i | 2 |
