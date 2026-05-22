from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from copco_eye_bench.operating_point_adaptation import (
    VALID_DECISION_CATEGORIES,
    aggregate_reader_probabilities,
    best_threshold,
    fixed_threshold_metrics,
    information_bits,
    legal_threshold_analysis,
    run_operating_point_adaptation,
    test_oracle_threshold_analysis as compute_test_oracle_threshold_analysis,
    threshold_candidates,
    validate_operating_point_adaptation,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_name": ["synthetic"] * 8,
            "source_type": ["unit"] * 8,
            "model_name": ["D3_unit"] * 8,
            "candidate_id": ["candidate_0000"] * 8,
            "feature_group": ["D3"] * 8,
            "model": ["logistic"] * 8,
            "task": ["CopCo_TYP"] * 8,
            "split_regime": ["unseen_reader"] * 8,
            "fold_id": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "evaluation_level": ["trial_level"] * 8,
            "sample_id": [f"s{i}" for i in range(8)],
            "trial_id": [f"t{i}" for i in range(8)],
            "participant_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
            "speech_id": ["S1", "S2", "S1", "S2", "S1", "S2", "S1", "S2"],
            "text_id": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
            "y_true": [0, 0, 1, 1, 0, 0, 1, 1],
            "p_pred": [0.1, 0.2, 0.55, 0.6, 0.3, 0.4, 0.65, 0.9],
            "y_pred": [0, 0, 1, 1, 0, 0, 1, 1],
            "threshold": [0.5] * 8,
            "threshold_source": ["fixed_0_5"] * 8,
            "role": ["test"] * 8,
            "probability_column": ["p_pred"] * 8,
            "label_column": ["y_true"] * 8,
            "prediction_column": ["y_pred"] * 8,
            "source_path": ["memory"] * 8,
        }
    )


def test_threshold_candidate_count_and_bits() -> None:
    candidates = threshold_candidates([0.2, 0.7, 0.7])
    assert candidates.tolist() == [0.0, 0.2, 0.5, 0.7, 1.0]
    assert information_bits(len(candidates)) == math.log2(5)


def test_best_threshold_uses_candidate_scores() -> None:
    learned = best_threshold([0, 0, 1, 1], [0.2, 0.3, 0.4, 0.9], metric="BA")
    assert learned["threshold"] == 0.4
    assert learned["n_candidate_thresholds"] == 7


def test_legal_threshold_does_not_use_test_labels_when_pool_missing() -> None:
    legal_metrics, thresholds = legal_threshold_analysis(_predictions())
    assert legal_metrics.empty
    assert not thresholds.empty
    assert set(thresholds["status"]) == {"not_computed_missing_inner_validation_predictions"}


def test_legal_threshold_uses_inner_validation_pool() -> None:
    test = _predictions()
    inner = test.copy()
    inner["role"] = "inner_validation"
    inner["p_pred"] = [0.1, 0.2, 0.7, 0.8, 0.1, 0.2, 0.7, 0.8]
    combined = pd.concat([test, inner], ignore_index=True)
    legal_metrics, thresholds = legal_threshold_analysis(combined)
    assert not legal_metrics.empty
    assert (legal_metrics["threshold_source"] != "test_oracle_diagnostic").all()
    assert (thresholds["threshold_source"] != "test_oracle_diagnostic").all()


def test_oracle_threshold_is_marked_diagnostic() -> None:
    fixed = fixed_threshold_metrics(_predictions())
    oracle, thresholds, budget = compute_test_oracle_threshold_analysis(
        _predictions(), fixed, pd.DataFrame()
    )
    assert not oracle.empty
    assert (oracle["official_claim_allowed"] == False).all()  # noqa: E712
    assert (thresholds["threshold_source"] == "test_oracle_diagnostic").all()
    assert budget["information_bits"].notna().all()


def test_probability_aggregation_formulas() -> None:
    preds = _predictions()
    simple = aggregate_reader_probabilities(preds, method="simple_mean_probability")
    logit = aggregate_reader_probabilities(preds, method="logit_mean_probability")
    majority = aggregate_reader_probabilities(preds, method="majority_vote_hard_label")
    assert len(simple) == 4
    assert round(float(simple.loc[simple["participant_id"] == "P1", "p_pred"].iloc[0]), 6) == 0.15
    assert logit["p_pred"].between(0, 1).all()
    assert set(majority["aggregation_basis"]) == {"hard_label_baseline"}


def test_metric_schema_contains_threshold_and_probability_source() -> None:
    metrics = fixed_threshold_metrics(_predictions())
    assert {"threshold_source", "score_source", "AUROC", "PR-AUC", "BA", "macro_F1"} <= set(
        metrics.columns
    )
    assert set(metrics["score_source"]) == {"p_pred"}


def test_missing_prediction_blocker_and_final_decision(tmp_path: Path) -> None:
    config = {
        "run": {"name": "operating_point_adaptation_unit", "output_root": str(tmp_path)},
        "operating_point_adaptation": {
            "repo_analysis_dir": str(tmp_path / "analysis_repo"),
            "output_layout": {"analysis": "analysis/operating_point_adaptation_v1"},
            "prediction_sources": [
                {
                    "name": "missing",
                    "prediction_path": str(tmp_path / "missing.csv"),
                    "evaluation_level": "trial_level",
                }
            ],
        },
    }
    manifest = run_operating_point_adaptation(config, tmp_path / "out", repo_root=tmp_path)
    assert manifest["decision_category"] == "invalid_missing_predictions"
    decision_path = (
        tmp_path
        / "out"
        / "analysis"
        / "operating_point_adaptation_v1"
        / "final_operating_point_decision.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision_category"] in VALID_DECISION_CATEGORIES
    assert decision["official_sota_claim_changed"] is False
    report = validate_operating_point_adaptation(config, tmp_path / "out", repo_root=tmp_path)
    assert report["status"] == "passed", report
