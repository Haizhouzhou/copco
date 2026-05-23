# D3 Offline Algorithm

Input data are prepared CopCo word/trial/participant features with operational TYP/RCS
labels. Gaze outcomes are residualized using cross-fitted controls that exclude
`reader_group`. Participant-level features summarize residual gaze and DFM
predictability sensitivity across the full reader record.

The main classifier is logistic regression evaluated at participant level with
leave-one-participant-out / reader-level evaluation. It is offline because a full
reader record is available before prediction. The allowed interpretation is an
explainable reader-profile method, not an online screening or clinical diagnostic.
