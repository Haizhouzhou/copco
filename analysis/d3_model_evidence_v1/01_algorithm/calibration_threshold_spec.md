# Calibration and Threshold Specification

Clean thresholds are fixed 0.5 or learned from train/inner-validation/calibration rows.
Sigmoid/Platt and isotonic calibration are allowed only when fitted on non-test rows.
Test-label oracle thresholds are diagnostic upper bounds only and must have
`official_claim_allowed=false`.
