from __future__ import annotations

import json

import pandas as pd

from copco_eye_bench.d3_online_targeted_optimization_v2 import (
    ONLINE_SELECTED_GROUPS,
    _add_evidence_cost,
    _candidate_space,
    validate_d3_online_targeted_optimization_v2,
)


def _config(tmp_path) -> dict:
    return {
        "d3_online_targeted_optimization_v2": {
            "repo_analysis_dir": str(tmp_path / "analysis"),
            "v1_output_dir": str(tmp_path / "v1_results"),
            "v1_analysis_dir": str(tmp_path / "v1_analysis"),
            "copco_typ_baseline_table": str(tmp_path / "baselines.csv"),
            "feature_families": [
                "residual_gaze_prefix",
                "dfm_sensitivity_prefix",
                "dfm_residual_gaze_prefix",
                "dfm_residual_plus_uncertainty_prefix",
                "all_allowed_strict_online",
            ],
            "accumulators": [
                "mean_probability",
                "logit_mean",
                "entropy_weighted",
                "learned_meta_aggregator",
            ],
            "calibrators": ["identity", "sigmoid", "isotonic"],
            "thresholds": [
                "fixed_0_5",
                "inner_cv_global",
                "inner_cv_prefix_specific",
                "inner_cv_regime_specific",
            ],
            "stopping_policies": [
                "confidence_stop",
                "cost_sensitive_stop",
                "target_sensitivity_stop",
                "coverage_constrained_stop",
            ],
            "evidence_budgets": {
                "early": {"word_count_prefix": [50, 100, 250], "trial_or_text_prefix": [1]},
                "mid": {
                    "word_count_prefix": [50, 100, 250, 500],
                    "trial_or_text_prefix": [1, 2, 3],
                },
                "late": {
                    "word_count_prefix": [250, 500, 1000],
                    "trial_or_text_prefix": [2, 3, 5],
                },
            },
        }
    }


def test_v2_candidate_space_separates_online_groups(tmp_path) -> None:
    space = _candidate_space(_config(tmp_path))
    counts = space.groupby("candidate_group").size().to_dict()
    assert counts["online_late_accumulation"] >= 12
    assert counts["online_mid_detection"] >= 12
    assert counts["online_early_detection"] >= 12
    assert counts["online_stopping_detector"] >= 12

    online = space[space["candidate_group"].isin(ONLINE_SELECTED_GROUPS)]
    assert not online["prefix_value"].astype(str).eq("all").any()
    assert not online["stopping_policy"].astype(str).eq("no_stop").any()


def test_v2_evidence_cost_uses_available_sequence_maximum() -> None:
    rows = pd.DataFrame(
        {
            "split_role": ["outer_test"],
            "split_regime": ["unseen_reader"],
            "fold_id": [0],
            "source_feature_group": ["residual_gaze_prefix"],
            "accumulator": ["mean_probability"],
            "participant_id": ["P1"],
            "n_words_observed": [250],
            "n_texts_observed": [1],
            "max_words_available": [1000],
            "max_texts_available": [5],
        }
    )
    scored = _add_evidence_cost(rows)
    assert round(float(scored.loc[0, "evidence_cost"]), 3) == 0.225
    assert round(float(scored.loc[0, "earliness_score"]), 3) == 0.775


def test_v2_validator_rejects_all_prefix_online_detector(tmp_path) -> None:
    config = _config(tmp_path)
    analysis = tmp_path / "analysis"
    output = tmp_path / "output"
    analysis.mkdir()
    output.mkdir()
    required_text = [
        "v1_audit_report.md",
        "per_prefix_performance_report.md",
        "error_source_by_prefix_report.md",
        "unseen_text_failure_analysis.md",
        "copco_typ_comparison_v2.md",
        "final_decision_report.md",
    ]
    for name in required_text:
        (analysis / name).write_text("ok\n", encoding="utf-8")
    pd.DataFrame({"x": [1]}).to_csv(analysis / "per_prefix_performance_curves.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(analysis / "error_source_by_prefix.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(analysis / "unseen_text_rescue_candidates.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(analysis / "strict_candidate_search_space.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(analysis / "strict_candidate_validation_ranking.csv", index=False)
    final = pd.DataFrame(
        {
            "candidate_group": ["online_late_accumulation"],
            "prefix_value": ["all"],
            "stopping_policy": ["confidence_stop"],
            "official_claim_allowed": [False],
            "selection_source": ["inner_oof"],
        }
    )
    final.to_csv(analysis / "strict_final_models.csv", index=False)
    final.to_csv(analysis / "strict_locked_test_results.csv", index=False)
    final.to_csv(analysis / "copco_typ_comparison_v2.csv", index=False)
    (analysis / "subgoal_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (analysis / "final_decision_v2.json").write_text(
        json.dumps({"official_sota_claim_changed": False}), encoding="utf-8"
    )

    report = validate_d3_online_targeted_optimization_v2(config, output, repo_root=tmp_path)
    assert report["status"] == "failed"
    assert "selected online detector uses prefix_value=all" in report["errors"]
