# Phase 3 Research Exploration Decision Report

## Signal Categories
| question | category |
| --- | --- |
| Segmentation opacity beyond controls | not_supported |
| Reader-group interactions | promising_signal |
| Participant-level prediction after ablation | strong_signal |

## Answers
- Which feature families explain gaze behavior? Classical lexical factors, DFM LM features, and segmentation-opacity features are all evaluated in controlled models; effect strength is summarized in the coefficient tables.
- Does segmentation opacity predict gaze beyond controls? not_supported.
- Are reader-group interactions present? promising_signal; treat as exploratory.
- Does participant-level prediction survive residualization/removal of exposure variables? strong_signal.
- Does segmentation add explanatory or predictive value? Standalone segmentation main effects are not_supported; segmentation is better retained as a sensitivity and interaction covariate for now.
- Does LM surprisal add explanatory or predictive value? Participant-level DFM exposure and sensitivity features are the strongest predictive signal in this exploration; Gemma remains pending.

## Recommended Phase 4 Directions
- Core Phase 4 direction 1: participant-level DFM predictability and residualized gaze-cost profiles with strict participant-level validation.
- Core Phase 4 direction 2: reader-group sensitivity interactions for word length, DFM surprisal, and boundary opacity, with text/speech sensitivity checks.

## Drop Or Defer
- Defer pronunciation-aware segmentation until a deterministic Danish pronunciation resource is integrated.
- Drop standalone segmentation-opacity main-effect publication framing unless Phase 4 sensitivity checks overturn the current result.
- Defer Gemma sensitivity until gated model access is resolved.
- Drop random word-level predictive evaluation; it is not valid for participant-level labels.
- Defer parser-syntax claims while parser status is `surface_heuristic_fallback`.

## Key Robustness Values
| selected_feature_group | selected_model | selected_split | observed_roc_auc | permutation_p_value | bootstrap_roc_auc_low | bootstrap_roc_auc_high |
| --- | --- | --- | --- | --- | --- | --- |
| D_dfm_exposure_and_sensitivity | logistic_regression | leave_one_participant_out | 0.9058 | 0.0099 | 0.8162 | 0.9798 |
