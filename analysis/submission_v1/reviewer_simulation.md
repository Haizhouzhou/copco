# Reviewer Simulation

## Reviewer 1: NLP/ML

- Likely score: borderline accept to accept.
- Strengths: leakage-aware LOPO validation, frozen result, permutation/bootstrap robustness,
  reproducibility capsule.
- Concerns: small participant count, potential feature-selection narrative, external
  generalization.
- Manuscript changes needed: foreground locked model selection and no random word-level
  splits.
- Appendix evidence needed: feature dictionary, split policy, permutation/bootstrap details.
- Rebuttal-ready response: the main claim is participant-level and uses frozen
  participant-grouped validation; exposure-only and count-variable checks reject the
  most direct leakage/confound explanations.

## Reviewer 2: Psycholinguistics / Eye Tracking

- Likely score: weak accept to accept.
- Strengths: residualized gaze-cost profiles, DFM predictability bridge, focused
  interaction synthesis.
- Concerns: mixed-effects fallback, interpretability of residual slopes, calibration with
  small N.
- Manuscript changes needed: explain gaze outcomes and residualization in plain language.
- Appendix evidence needed: interaction synthesis and calibration details.
- Rebuttal-ready response: word-level interactions are secondary; the primary result is a
  participant-level confirmatory prediction result with cross-fitted residualization.

## Reviewer 3: Danish / Reading / Dyslexia Domain

- Likely score: weak accept if limitations are clear.
- Strengths: Danish natural reading focus, careful label language, boundary-opacity
  interpretability.
- Concerns: operational label provenance, no clinical claim, orthographic boundary proxy,
  Gemma pending.
- Manuscript changes needed: make label provenance and Danish-only scope explicit.
- Appendix evidence needed: segmentation construction, null standalone segmentation result,
  dataset caveats.
- Rebuttal-ready response: the paper explicitly avoids clinical screening claims and treats
  boundary opacity as secondary interpretation, not the main effect.
