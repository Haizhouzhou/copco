# D3 Online Detection Goal Contract v1

This contract separates the frozen offline D3 interpretation from the new
D3OnlineTargetedOptimization v1 online deployment task.

## A. Offline Task

The offline task assumes that a full reader record is available before scoring.
The model is a reader-level D3 residualized gaze-profile predictor: it summarizes
the reader's gaze behavior over all available natural-reading evidence and scores
the resulting reader profile.

Primary role: main scientific result and offline profile upper-bound. The frozen
offline D3 result is not modified by this phase.

## B. Online Task

The online task assumes that only prefix evidence is available at time `t`.
For each prefix, the detector emits a probability `p_t` using only evidence
observed through that prefix. A decision policy may then emit one of:

- `positive`
- `negative`
- `continue`

Primary role: deployment-oriented, task-adapted sequential detection. It is a
secondary analysis unless it satisfies a separate official protocol.

## C. Evaluation Levels

- `trial_or_prefix_level`: one prediction per observed prefix row.
- `reader_aggregated`: probabilities accumulated or aggregated to reader level.
- `online_stopping_decision`: first emitted positive/negative decision or final
  fallback decision for each reader.
- `offline_full_reader`: full-record reader profile result from frozen D3-family
  artifacts.

## D. Clean and Diagnostic Results

Clean results learn thresholds, calibrators, accumulators, and stopping policies
only from train, inner-validation, or calibration rows. Clean rows must not use
outer-test labels for model selection, calibration, threshold learning, or
stopping-policy tuning.

Oracle rows may optimize thresholds or stopping rules on outer-test labels only
as diagnostic upper bounds. Every oracle row must set:

- `clean_result=false`
- `official_claim_allowed=false`
- `benchmark_relative_claim_allowed=false`

## E. Success Gates

D3OnlineTargetedOptimization v1 is complete only if the validator confirms that:

- GOAL 0 contract documents exist.
- Nested prediction artifacts exist.
- Online prefix data exists and passes no-future-evidence checks.
- Legal thresholds are learned or an exact data blocker is recorded.
- Fitted calibration is attempted or an exact data-size blocker is recorded.
- Online evidence accumulation is evaluated.
- Online stopping policies are evaluated.
- Before/after comparisons are produced.
- A final online configuration is selected from validation data.
- The final decision report answers the online/offline interpretation questions.

If a gate cannot be completed, the subgoal status file must mark it `blocked`
with an exact blocker and evidence paths for completed independent work.
