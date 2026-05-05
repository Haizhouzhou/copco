# Prepared Dataset Readiness Report

- Participant target labels complete: True
- Label sources/provenance documented: true
- Segmentation labels complete enough: True
- Segmentation labels confounded with other linguistic features: documented; control later
- Quality labels complete: True
- Split labels leakage-safe: True
- Text assignments balanced enough for exploratory analysis: documented with caveats
- Readiness judgement: ready_with_caveats

## Caveats To Carry Forward
- Participant provenance is operational project metadata in v1.1.
- Segmentation-opacity labels are orthographic proxies, not pronunciation-aware labels.
- Parser fields remain surface_heuristic_fallback and should not be interpreted as full syntax.
- DFM LM alignment warnings are documented; Gemma sensitivity remains pending because access was gated.
- Text exposure and label balance require confound-controlled analysis in the next phase.

## Prepared Dataset Row Counts
| table | rows |
| --- | --- |
| analysis_ready_word_level_v1_1 | 335203 |
| analysis_ready_sentence_level_v1_1 | 1986 |
| analysis_ready_participant_level_v1_1 | 57 |

## Recommended Next Analyses
- Fit controlled psycholinguistic models using participant-grouped or mixed-effects designs.
- Evaluate segmentation-opacity effects with word length, frequency, surprisal, entropy, and text controls.
- Run parser upgrade sensitivity once DaCy/spaCy environment issues are resolved.
- Keep participant-level prediction exploratory and leakage-safe.
