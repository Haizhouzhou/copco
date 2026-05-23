# Prohibited Features and Leakage Policy

Prohibited predictors include `participant_id`, `speech_id`, `text_id`, random word
splits, future total word/text counts in online rows, and `reader_group` inside
residualization. Clean metrics must not use test labels for thresholds/calibration.
Oracle rows are diagnostic only. Official SOTA requires official data, folds, and
evaluator chain support.
