# D3 Algorithm Overview

D3 is the residualized DFM gaze-profile model family. It models how a reader's gaze
responds to contextual predictability from a Danish foundation language model (DFM),
after removing low-level word, text, and quality confounds. The explainable unit is a
reader profile: residual gaze sensitivity to DFM surprisal/entropy and related
predictability summaries.

D3 has four regimes:

- Full-data/offline reader profile: all available reading evidence is used to build a
  participant-level profile. This is the main scientific result and upper-bound
  reader-profile interpretation.
- D3 Lite: a reduced official-compatible trial-level model. It preserves an EyeBench
  stress-test anchor but is not equivalent to full D3.
- Online fixed-budget D3: prefix evidence is accumulated up to a fixed word/text
  budget and converted into probability `p_t`.
- Online stopping detector: sequential probabilities may stop early, but current
  coverage-risk results do not support readiness.
