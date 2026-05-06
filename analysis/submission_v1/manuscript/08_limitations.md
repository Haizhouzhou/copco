# Limitations
The strongest limitation is the participant count. The analysis contains 57
participants, including 19 dyslexia-labeled readers, so the result is a strong internal
confirmatory finding rather than an externally validated screening model. Labels are
operational research labels; they should not be interpreted as clinical diagnosis,
clinical screening, or medical validation. No independent external dataset is included,
and generalization beyond Danish natural reading remains open.

Text and speech exposure are not randomized, so the paper relies on exposure-only
comparisons, removal of exposure-count variables, and text-exposure audits. These
checks reduce the plausibility of a text-assignment explanation but do not replace an
independent balanced replication. Calibration is reported, but calibration estimates
are limited by the small participant sample and should be treated cautiously.
Participant influence remains possible despite the leave-one-dyslexia-labeled and
remove-one-participant sensitivity checks.

LM alignment warnings are recorded and should be considered when interpreting
individual token-level features. Boundary-opacity labels are deterministic
orthographic proxies, not pronunciation-aware segmentation labels. Parser status is
`surface_heuristic_fallback`, so parser-syntax claims are deferred. Gemma
sensitivity is pending gated access and is not used in the main claim. Reviewer risks
are summarized in Table (tab:reviewer-risk).
