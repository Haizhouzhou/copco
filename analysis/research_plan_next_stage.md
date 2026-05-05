# Danish Natural-Reading Eye-Tracking Signatures of Dyslexia-Labeled Readers

## Project Title

Danish natural-reading eye-tracking signatures of dyslexia-labeled readers: integrating gaze, linguistic complexity, and language-model predictability.

## Research Objective

Estimate psycholinguistic and predictive signatures of dyslexia-labeled reader behavior in Danish natural reading, while keeping claims exploratory and non-clinical.

## Dataset Summary

CopCo is treated as a Danish natural-reading eye-tracking corpus with operational dyslexia labels. The release reports participant, text, sentence, word, and gaze counts in `feature_release_report.md`.

## Label-Provenance Summary

Labels support cautious wording such as dyslexia-labeled participants and typical/control participants. They do not by themselves support clinical diagnosis wording.

## Completed Engineering Milestones

- Stable identifiers and leakage-aware split tables.
- Full gaze/classical feature tables.
- Parser/morphosyntactic feature layer with fallback diagnostics.
- DFM causal-LM surprisal and entropy.
- Embedding feature layer and compact semantic features.
- Joined modeling tables and validation reports.

## Feature Families Now Available

- Participant-specific gaze features.
- Classical lexical, surface, readability, and position features.
- Parser or parser-fallback morphosyntactic features with diagnostics.
- DFM base causal-LM surprisal and entropy.
- Optional Gemma base causal-LM sensitivity when model access is available.
- Sentence and paragraph embeddings plus compact semantic-cohesion features.
- Joined word, sentence, paragraph, and participant-level modeling tables.

## Main Hypotheses

- Dyslexia-labeled readers may show elevated gaze cost after controlling for lexical and syntactic difficulty.
- Dyslexia-labeled readers may show different sensitivity to word length.
- Dyslexia-labeled readers may show different sensitivity to word frequency.
- Dyslexia-labeled readers may show different sensitivity to LM-derived surprisal.
- Participant-level gaze-sensitivity profiles may support exploratory classification, but not clinical diagnosis.

## Analysis Strategy

Start with psycholinguistic validation, then group-associated effects, then participant-level exploratory prediction. Feature validation and interpretable effects are primary.

## Model Strategy

Use interpretable model ladders before complex learners. Participant-level classification is primary for dyslexia-label prediction; word-level ladders are secondary because word rows inherit participant labels.

## Validation Strategy

Validate row counts, stable keys, duplicate counts, missing feature rates, LM alignment warnings, split leakage, model metric schemas, and report completeness before treating outputs as release artifacts.

## Leakage-Control Strategy

Participant-level labels require participant-grouped evaluation. Random word-level splits are not allowed.

## Statistical Modeling Plan

Use mixed-effects models where feasible, with participant and stimulus grouping. Use robust clustered or HC3 models as fallbacks when mixed models fail.

## Predictive Modeling Plan

Primary predictive unit is participant-level. Report skipped folds, class balance, uncertainty, and invalid metric conditions.

## Sensitivity Analyses

Gemma base-model LM features are a sensitivity analysis, not a blocker for the primary DFM feature release.

## Limitations

Operational labels, no independent clinical validation, parser fallback limitations, and no external validation dataset yet.

## Next Data-Collection Needs

Clarify label provenance, formal diagnostic instruments if available, age and sex completeness, comprehension scoring provenance, and external validation or replication data.

## Paper Outline

1. Dataset and label-provenance framing.
2. Feature engineering and alignment validation.
3. Psycholinguistic sanity checks.
4. Group-associated gaze and linguistic-sensitivity effects.
5. Exploratory participant-level prediction.
6. Limitations, sensitivity analyses, and next data needs.

## Immediate Next Tasks

Review label provenance, inspect mixed-effects convergence, decide paper hypotheses, and identify external validation or additional participant metadata needs.
