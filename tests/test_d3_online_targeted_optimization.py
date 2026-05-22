from __future__ import annotations

import json
import math

import pandas as pd

from copco_eye_bench.d3_online_targeted_optimization import (
    _validate_split_disjointness,
    accumulate_probability_sequence,
    classification_metrics,
    classify_final_decision,
    compute_evidence_cost,
    learn_threshold_from_pool,
    make_outer_splits,
    oracle_diagnostics,
    validate_no_future_evidence,
    validate_prefix_monotonicity,
    validate_subgoal_status_payload,
)


def _prefix_frame() -> pd.DataFrame:
    rows = []
    for idx, participant in enumerate(["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]):
        label = idx % 2
        for order, words in enumerate([50, 100], start=1):
            text = "T1" if order == 1 else "T2"
            rows.append(
                {
                    "prefix_row_id": len(rows),
                    "participant_id": participant,
                    "reader_group": "dyslexia_labeled" if label else "typical_control",
                    "reader_group_binary": label,
                    "prefix_type": "word_count_prefix",
                    "prefix_value": str(words),
                    "prefix_order_index": order,
                    "n_words_observed": words,
                    "n_trials_observed": order,
                    "n_texts_observed": order,
                    "n_speeches_observed": order,
                    "n_fixations_observed": words * 2,
                    "terminal_text_id": text,
                    "observed_text_ids": text if order == 1 else "T1|T2",
                    "evidence_available_until_prefix": f"{participant}_{words}",
                    "stable_enough_for_prediction": True,
                    "raw_trt_mean": float(words),
                    "resid_trt_mean": float(words) / 10,
                    "dfm_exp_surprisal_mean": 1.0,
                    "dfm_sens_trt_surprisal_slope": 0.1,
                    "dfm_resid_trt_surprisal_slope": 0.2,
                    "uncert_inverse_sqrt_words": 1 / math.sqrt(words),
                }
            )
    return pd.DataFrame(rows)


def test_prefix_monotonicity_and_no_future_evidence() -> None:
    prefix = _prefix_frame()
    assert validate_prefix_monotonicity(prefix) == []
    assert validate_no_future_evidence(prefix) == []
    broken = prefix.copy()
    broken.loc[broken.index[-1], "n_words_observed"] = 1
    assert validate_prefix_monotonicity(broken)


def test_split_disjointness_unseen_reader_and_text() -> None:
    prefix = _prefix_frame()
    folds = make_outer_splits(prefix, "unseen_reader", n_splits=2, seed=3)
    assert folds
    preds = []
    for fold in folds:
        for role, indices in [("train_fit", fold.train_indices), ("outer_test", fold.test_indices)]:
            part = prefix.loc[indices].copy()
            part["split_regime"] = "unseen_reader"
            part["fold_id"] = fold.fold_id
            part["split_role"] = role
            preds.append(part)
    pred = pd.concat(preds, ignore_index=True)
    assert _validate_split_disjointness(pred) == []


def test_nested_prediction_role_separation_validator_catches_overlap() -> None:
    pred = pd.DataFrame(
        {
            "split_regime": ["unseen_reader", "unseen_reader"],
            "fold_id": [0, 0],
            "split_role": ["train_fit", "outer_test"],
            "participant_id": ["P1", "P1"],
            "observed_text_ids": ["T1", "T2"],
        }
    )
    assert _validate_split_disjointness(pred)


def test_threshold_learning_uses_supplied_pool_only() -> None:
    pool = pd.DataFrame({"y_true": [0, 0, 1, 1], "p_pred": [0.1, 0.2, 0.7, 0.8]})
    learned = learn_threshold_from_pool(pool, policy="balanced_accuracy_threshold")
    assert learned["status"] == "complete"
    assert 0.2 <= learned["threshold"] <= 0.7


def test_calibration_and_metric_basics() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9], 0.5)
    assert metrics["BA"] == 1.0
    assert metrics["Brier"] < 0.15


def test_online_accumulator_formulas() -> None:
    probs = [0.2, 0.8]
    mean = accumulate_probability_sequence(probs, "mean_probability")
    logit = accumulate_probability_sequence(probs, "logit_mean")
    beta = accumulate_probability_sequence(probs, "beta_binomial_posterior")
    assert mean[-1] == 0.5
    assert 0.0 <= logit[-1] <= 1.0
    assert beta[-1] == 0.5


def test_evidence_cost_calculation() -> None:
    cost = compute_evidence_cost({"n_words_observed": 50, "n_texts_observed": 1}, 100, 2)
    assert cost["combined_evidence_cost"] == 0.5
    assert cost["earliness_score"] == 0.5


def test_oracle_official_claim_false(tmp_path) -> None:
    analysis = tmp_path / "analysis" / "d3_online_targeted_optimization_v1"
    analysis.mkdir(parents=True)
    config = {"d3_online_targeted_optimization": {"repo_analysis_dir": str(analysis)}}
    online = pd.DataFrame(
        {
            "split_role": ["outer_test"] * 4,
            "split_regime": ["unseen_reader"] * 4,
            "fold_id": [0] * 4,
            "feature_group": ["f"] * 4,
            "accumulator": ["mean_probability"] * 4,
            "prefix_type": ["word_count_prefix"] * 4,
            "participant_id": ["P1", "P2", "P3", "P4"],
            "y_true": [0, 0, 1, 1],
            "p_t": [0.1, 0.2, 0.8, 0.9],
        }
    )
    metrics, _ = oracle_diagnostics(config, online, tmp_path)
    assert not metrics.empty
    assert (metrics["official_claim_allowed"] == False).all()  # noqa: E712


def test_subgoal_status_schema_and_final_categories() -> None:
    payload = {
        "subgoals": {
            goal: {"name": name, "status": "completed", "evidence_paths": ["x"], "blocker": ""}
            for goal, name in {
                "GOAL_0": "execution docs",
                "GOAL_1": "prefix datasets",
                "GOAL_2": "nested prediction artifacts",
                "GOAL_3": "D3 online prefix models",
                "GOAL_4": "legal calibration and threshold learning",
                "GOAL_5": "online evidence accumulation",
                "GOAL_6": "online stopping policies",
                "GOAL_7": "targeted online optimization loop",
                "GOAL_8": "oracle upper-bound diagnostics",
                "GOAL_9": "error trajectory analysis",
                "GOAL_10": "online/offline comparison and final decision",
                "GOAL_11": "manuscript and supplement update",
                "GOAL_12": "validator and tests",
                "GOAL_13": "commit and push",
            }.items()
        }
    }
    assert validate_subgoal_status_payload(json.loads(json.dumps(payload))) == []
    locked = pd.DataFrame({"AUROC": [0.8], "BA": [0.72]})
    assert classify_final_decision(locked, 12) == "both_offline_and_online_capable"
