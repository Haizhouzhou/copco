# Limitations
The main limitation is sample size: the analysis contains 57 participants, including 19
dyslexia-labeled readers. Labels are operational research labels and should not be
interpreted as clinical diagnosis, screening, or medical validation. The package does
not include an independent external dataset, so generalization beyond this Danish
natural-reading setting remains open.

Text assignment is not randomized, which is why the paper emphasizes exposure-only and
text-exposure audits. LM alignment warnings are recorded and should be considered when
interpreting individual token-level features. Boundary-opacity labels are deterministic
orthographic proxies, not pronunciation-aware segmentation labels. Parser status is
`surface_heuristic_fallback`, so parser-syntax claims are deferred. Gemma
sensitivity is pending gated access and is not used in the main claim. Reviewer risks
are summarized in Table (tab:reviewer-risk).
