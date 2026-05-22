from __future__ import annotations

from pathlib import Path

import yaml

from copco_eye_bench.d3_eyebench_own_method_score_max import (
    PROHIBITED_FEATURES,
    VALID_DECISION_CATEGORIES,
    build_candidate_specs,
    validate_d3_eyebench_own_method_score_max_config,
)


CONFIG_PATH = Path("configs/d3_eyebench_own_method_score_max_v2.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_own_method_config_passes_static_validation() -> None:
    report = validate_d3_eyebench_own_method_score_max_config(_config())
    assert report["status"] == "passed", report


def test_candidate_0000_is_exact_first_anchor() -> None:
    candidates = build_candidate_specs(_config())
    assert candidates
    first = candidates[0]
    assert first.candidate_id == "candidate_0000"
    assert first.anchor_exact is True
    assert first.feature_recipe == "d3_lite_exact"
    assert first.model_type == "official_lite_logistic"
    assert first.threshold_method == "fixed_0_5"


def test_candidate_budget_includes_anchor_and_d3_family_only() -> None:
    config = _config()
    candidates = build_candidate_specs(config)
    assert len(candidates) == config["d3_eyebench_own_method_score_max"]["budget"]["max_candidates"]
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert all(candidate.family.startswith("d3_") for candidate in candidates)
    assert all(candidate.as_dict()["preserves_previous_d3_lite_features"] for candidate in candidates)


def test_prohibited_predictors_cover_contract_identifiers() -> None:
    configured = set(_config()["d3_eyebench_own_method_score_max"]["prohibited_features"])
    assert PROHIBITED_FEATURES <= configured


def test_decision_categories_match_contract() -> None:
    assert VALID_DECISION_CATEGORIES == {
        "d3_method_improved",
        "d3_method_competitive_but_not_improved",
        "d3_method_exploratory_gain_only",
        "d3_method_not_improved",
        "blocked_by_environment",
        "blocked_by_data",
        "blocked_by_evaluator",
    }
