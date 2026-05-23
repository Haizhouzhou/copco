# Figure Source Manifest

| figure_source_id | possible_content | source_files | required_filter | x_candidates | y_candidates | grouping_candidates | caution_notes | no_figure_generated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dfm_ablation_source | DFM exposure versus sensitivity plot source | 03_canonical_metrics/canonical_metrics_long.csv | source_phase=AutoResearch_v1 | feature_family | AUROC,PR_AUC,balanced_accuracy | algorithm_regime | Source material only; no figure generated. | True |
| online_prefix_curve_source | Online prefix performance curve source | 03_canonical_metrics/canonical_online_prefix_results.csv | source_phase=D3OnlineTargetedOptimization_v2 | prefix_value,evidence_budget | AUROC,balanced_accuracy,Brier | split_regime,feature_family,accumulator | v1/v2 rows remain separated. | True |

No figures are generated in v1.1.
