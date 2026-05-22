from __future__ import annotations

from pathlib import Path

import yaml

from copco_eye_bench.d3_eyebench_protocol_optimization import (
    BASE_DENYLIST,
    VALID_DECISION_CATEGORIES,
    build_candidate_specs,
    validate_d3_eyebench_protocol_optimization_config,
)


def test_protocol_optimization_config_parses() -> None:
    config = yaml.safe_load(
        Path("configs/d3_eyebench_protocol_aligned_optimization_v1.yaml").read_text()
    )
    report = validate_d3_eyebench_protocol_optimization_config(config)
    assert report["status"] == "passed", report


def test_candidate_budget_and_ids_are_deterministic() -> None:
    config = yaml.safe_load(
        Path("configs/d3_eyebench_protocol_aligned_optimization_v1.yaml").read_text()
    )
    candidates = build_candidate_specs(config)
    assert len(candidates) == config["d3_eyebench_protocol_optimization"]["budget"]["max_candidates"]
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert candidates[0].candidate_id.startswith("d3opt_0001_")


def test_required_decision_categories_match_protocol() -> None:
    assert {
        "official_sota_claim_allowed",
        "official_compatible_d3_improved_but_not_sota",
        "official_compatible_but_not_sota",
        "optimization_inconclusive",
        "blocked_by_environment",
        "blocked_by_data",
        "blocked_by_evaluator",
    } == VALID_DECISION_CATEGORIES


def test_predictor_denylist_covers_protocol_identifiers() -> None:
    config = yaml.safe_load(
        Path("configs/d3_eyebench_protocol_aligned_optimization_v1.yaml").read_text()
    )
    configured = set(config["d3_eyebench_protocol_optimization"]["prohibited_features"])
    assert BASE_DENYLIST <= configured
