# Feature Family Ablation Report

This table is grouped by D3 feature recipe. Test values are present only for `candidate_0000` and train/inner-validation-ranked top candidates.

| family | feature_recipe | candidates | best_inner_ba | best_test_ba | best_test_auroc |
| --- | --- | --- | --- | --- | --- |
| d3_lite_anchor | d3_lite_exact | 1 | 0.6985 | 0.6985 | 0.7852 |
| d3_lite_calibration_variant | d3_lite_exact | 5 | 0.7577 |  |  |
| d3_lite_plus_full_official_extension | d3_lite_all | 4 | 0.8007 | 0.6779 | 0.7443 |
| d3_lite_plus_raw_gaze | d3_lite_raw_summary | 5 | 0.7960 | 0.6778 | 0.7764 |
| d3_lite_plus_robust_residuals | d3_lite_robust | 4 | 0.7494 |  |  |
| d3_lite_plus_text_gaze_interactions | d3_lite_interactions | 5 | 0.7559 |  |  |
