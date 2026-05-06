# Data
The frozen prepared dataset contains 57 participants: 19 dyslexia-labeled and 38
typical/control readers. The prepared tables contain 335,203 word-level rows, 1,986
sentence-level rows, and 57 participant-level rows. Table (tab:dataset-summary)
summarizes the dataset, and Table (tab:feature-label-release) summarizes the frozen
feature and label releases.

The analysis uses Feature Release v1 and Label Release v1.1. Quality labels record LM
missingness, parser status, and label availability. The parser status is
`surface_heuristic_fallback`, so no parser-syntax claim is made. Segmentation
labels are deterministic orthographic boundary-opacity proxies, not pronunciation-aware
labels. Text assignment is not randomized; this motivates the DFM exposure-only and
text-exposure sensitivity checks. All predictive validation is participant-grouped, with
leave-one-participant-out as the primary split policy.
