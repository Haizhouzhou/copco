# Introduction
Eye movements provide a time-resolved behavioral trace of reading. Fixation durations,
refixations, skipping, go-past time, and fixation counts respond to lexical,
contextual, and reader-level factors, making natural-reading gaze data a useful setting
for studying how linguistic pressure differs across reader groups
[rayner1998eye,duchowski2017eye]. Danish natural reading is a useful case because
orthography, morphology, and boundary-related vocalic patterns create pressures that
are not identical to English-centered benchmarks.

The project starts from a constrained question. We do not ask whether an arbitrary
classifier can separate rows of eye-tracking data, and we do not treat word rows as
independent labels. The target label is participant-level, so the primary prediction
task must also be participant-level. The scientific question is which participant
profile carries the signal: text exposure, global reading speed, or sensitivity to
contextual predictability.

Language-model predictability offers a bridge between psycholinguistic reading-time
theory and modern NLP. We use DFM surprisal and entropy as contextual predictability
signals, then estimate how each participant's residual gaze costs vary with those
signals under cross-fitted residualization. The frozen result supports one main story:
participant-level DFM predictability sensitivity and cross-fitted residualized
gaze-cost profiles distinguish dyslexia-labeled and typical/control readers in Danish
natural reading.

This paper makes four contributions:
\begin{enumerate}
\item A prepared Danish natural-reading gaze, linguistic, LM, and label pipeline for dyslexia-labeled reader analysis.
\item A cross-fitted residualized participant sensitivity-profile method.
\item Evidence that DFM predictability sensitivity, not DFM exposure, drives strong participant-level prediction.
\item Secondary evidence that reader-group differences involve word length, DFM surprisal, and previous-boundary opacity.
\end{enumerate}
