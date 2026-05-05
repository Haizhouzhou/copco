# Split Policy v1

## Allowed Splits
- `leave_one_participant_out`: primary participant-level evaluation split.
- `participant_grouped_kfold`: secondary participant-grouped cross-validation split.
- `sensitivity_exclude_uncertain_labels`: documented sensitivity subset.

## Prohibited Splits
Random word-level train/test splitting is not allowed because word rows from the same participant are not independent and would leak participant-level target labels.

## Fold Representation
Every split label row keeps a participant wholly in one fold role. Invalid folds are kept with `split_valid = false` and a `skip_reason`; they are not silently dropped.

## Imbalance Handling
Class counts are reported per fold. Later predictive analyses should report skipped folds, confidence intervals where feasible, and avoid interpreting exploratory classification as screening.
