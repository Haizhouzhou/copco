# OperatingPointAdaptation v1 final decision

Decision category: `supplement_supporting_result`

1. Are D3_Lite low BA results mainly threshold-related?
   Fixed-threshold and oracle-threshold rows quantify this directly. Maximum D3_Lite split-oracle BA improvement over fixed 0.5: 0.0229.
2. Does legal threshold learning improve BA/Macro F1?
   Legal threshold learning was not computed from the available artifacts because no train/inner-validation/calibration prediction rows were present.
3. Does calibration improve Brier/calibration slope?
   Only identity calibration is final for the available artifacts; fitted legal calibrators require calibration prediction rows.
4. Does reader-level probability aggregation improve AUROC/PR-AUC?
   Reader probability aggregation is reported as secondary evidence in reader_probability_aggregation_metrics.csv.
5. How much improvement is possible under test-oracle threshold?
   Maximum D3_Lite split-oracle BA improvement over fixed 0.5: 0.0229.
6. How many bits of label information does the oracle threshold use?
   The oracle information budget is log2(number_of_candidate_thresholds) and is reported per threshold policy in test_oracle_information_budget.csv.
7. Does any legal result change the official SOTA status?
   No. The official SOTA claim is unchanged and remains false.
8. Does the oracle result show an upper-bound implementation potential?
   Yes, where oracle BA improves over fixed 0.5 it is diagnostic implementation potential only.
9. What wording should be used in the paper?
   Threshold and calibration analyses show that D3 is probability-first and reader-profile oriented. Test-oracle thresholds provide an upper bound but are not used for benchmark or SOTA claims.

Test-oracle threshold results are diagnostic upper bounds and cannot be used for official benchmark or SOTA claims.
