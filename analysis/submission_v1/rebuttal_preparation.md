# Rebuttal Preparation

The central rebuttal position is that the paper reports a frozen confirmatory package, not
an exploratory modeling search. The selected model, metrics, robustness checks, and
scientific decisions were fixed in AutoResearch v1.

## Likely Objection: Leakage Or Text Exposure

Response: the primary model is participant-level LOPO, contains no direct exposure-count
variables, and DFM exposure-only performs poorly. Cross-fitted residualization fits gaze
models only on training participants for each held-out participant.

## Likely Objection: Small Participant Count

Response: the paper frames the result as a strong internal natural-reading finding and
does not claim clinical validation. Bootstrap, permutation, and influence analyses are
reported, and external validation is listed as future work.

## Likely Objection: Segmentation Interpretation

Response: standalone segmentation main-effect framing is dropped. Boundary opacity is a
secondary interpretability feature, and pronunciation-aware labels are deferred.

## Likely Objection: Parser Or Syntax Claims

Response: parser status is surface heuristic fallback; parser-syntax claims are explicitly
deferred.
