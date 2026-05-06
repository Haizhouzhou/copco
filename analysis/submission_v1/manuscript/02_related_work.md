# Related Work
Natural-reading eye-tracking corpora support analyses of reading behavior under
connected text rather than isolated word presentation
[kennedy2003d,kliegl2006tracking]. CopCo provides Danish natural-reading
material that can be aligned with gaze, linguistic, and language-model features. We use
the current labels as operational research labels and do not claim clinical diagnosis,
screening, or medical validation.

Reader-level prediction from eye movements has been explored in dyslexia and reading
difficulty settings, including prior Danish natural-reading prediction work. The
present contribution is narrower and more confirmatory: it audits a frozen
participant-level result and separates text exposure from participant-level
predictability sensitivity. LM surprisal has long been connected to reading time and
processing difficulty [hale2001probabilistic,levy2008expectation,smith2013effect].
Here, DFM surprisal and entropy are used as contextual predictability features rather
than as standalone text-difficulty labels.

Danish vocalic and boundary-opacity literature motivates the secondary
orthographic-boundary analysis. The current boundary-opacity variables are deterministic
orthographic proxies, not pronunciation-aware phonological labels. They are therefore
used only as secondary interpretability features. Explainable reader-level prediction
requires participant-grouped validation, leakage controls, calibration checks, and
feature-stability analysis rather than random word-level splits.
