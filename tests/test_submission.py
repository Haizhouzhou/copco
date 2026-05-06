from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from copco_eye_bench.submission import (
    FINAL_MODEL_GROUP,
    MANUSCRIPT_SECTIONS,
    SUBMISSION_FIGURES,
    SUBMISSION_TABLES,
    SUPPLEMENT_SECTIONS,
    build_submission_package,
    validate_submission_package,
)


FEATURES = [
    "crossfit_ffd_residual_dfm_surprisal_slope",
    "crossfit_ffd_residual_dfm_entropy_slope",
    "crossfit_total_fixation_residual_dfm_surprisal_slope",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x01\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _write_frozen_autoresearch(root: Path) -> Path:
    auto = root / "autoresearch"
    metric = {
        "analysis": "phase4_confirmatory_participant_prediction",
        "split_name": "leave_one_participant_out",
        "feature_group": FINAL_MODEL_GROUP,
        "model": "logistic_regression",
        "n_features": len(FEATURES),
        "n_predictions": 6,
        "usable_folds": 6,
        "skipped_folds": 0,
        "roc_auc": 0.89,
        "pr_auc": 0.86,
        "balanced_accuracy": 0.83,
        "macro_f1": 0.83,
        "brier_score": 0.12,
        "calibration_intercept": -0.2,
        "calibration_slope": 0.9,
        "calibration_mean_predicted": 0.5,
        "calibration_observed_rate": 0.5,
        "status": "complete",
        "skip_reason": "",
        "fold_validity": "all_test_predictions_generated",
    }
    _write_json(
        auto / "final_model" / "final_model_manifest.json",
        {
            "selected_feature_group": FINAL_MODEL_GROUP,
            "selected_model": "logistic_regression",
            "split_name": "leave_one_participant_out",
            "features": FEATURES,
            "metrics": metric,
            "prohibited_variables_present": [],
        },
    )
    _write_json(
        auto / "decision" / "final_decision.json",
        {"publication_readiness": "ready_for_manuscript_drafting"},
    )
    _write_json(auto / "manifest.json", {"status": "complete"})
    _write_json(auto / "run_summary.json", {"status": "complete"})
    _write_csv(auto / "final_model" / "final_model_metrics.csv", [metric])
    _write_text(
        auto / "final_model" / "final_model_feature_dictionary.md",
        "\n".join(f"- `{feature}`: DFM residual gaze-cost sensitivity" for feature in FEATURES),
    )
    dfm_rows = [
        {**metric, "feature_group": "D1_dfm_exposure_only", "roc_auc": 0.42, "pr_auc": 0.37},
        {**metric, "feature_group": "D2_dfm_sensitivity_only", "roc_auc": 0.88, "pr_auc": 0.84},
        metric,
        {
            **metric,
            "feature_group": "D4_dfm_exposure_plus_sensitivity",
            "roc_auc": 0.87,
            "pr_auc": 0.85,
        },
    ]
    _write_csv(auto / "stress_tests" / "dfm_exposure_vs_sensitivity.csv", dfm_rows)
    _write_csv(
        auto / "stress_tests" / "bootstrap_results.csv",
        [
            {"metric": "roc_auc", "observed": 0.89, "n_bootstrap": 1000, "ci_low": 0.74, "ci_high": 0.98},
            {"metric": "pr_auc", "observed": 0.86, "n_bootstrap": 1000, "ci_low": 0.72, "ci_high": 0.97},
        ],
    )
    _write_csv(
        auto / "stress_tests" / "permutation_results.csv",
        [{"iteration": idx, "roc_auc": 0.2 + idx * 0.01} for idx in range(6)],
    )
    _write_csv(
        auto / "stress_tests" / "calibration_summary.csv",
        [{"probability_bin": "overall", "n": 6, "mean_predicted": 0.5, "observed_rate": 0.5}],
    )
    _write_csv(
        auto / "stress_tests" / "influence_analysis.csv",
        [
            {
                "participant_id": f"P{idx}",
                "misclassified": idx % 2 == 0,
                "high_leverage_flag": idx == 1,
                "delta_roc_auc": 0.01 * idx,
            }
            for idx in range(6)
        ],
    )
    _write_csv(
        auto / "stress_tests" / "text_exposure_sensitivity.csv",
        [{"variable": "n_word_rows", "corr_with_score": 0.1, "corr_with_abs_error": 0.2}],
    )
    _write_csv(
        auto / "stress_tests" / "feature_stability.csv",
        [
            {
                "feature_group": FINAL_MODEL_GROUP,
                "feature": feature,
                "mean_coefficient": 0.5,
                "sd_coefficient": 0.1,
                "n_folds": 6,
                "positive_rate": 1.0,
                "negative_rate": 0.0,
                "sign_stability": 1.0,
                "abs_mean_coefficient": 0.5,
                "feature_family": "DFM residual sensitivity",
            }
            for feature in FEATURES
        ],
    )
    for rel, rows in {
        "dataset_summary_table.csv": [{"table": "participants", "rows": 6}],
        "feature_release_summary_table.csv": [{"release": "feature", "status": "complete"}],
        "label_release_summary_table.csv": [{"release": "label", "status": "complete"}],
        "robustness_summary_table.csv": [{"test": "permutation", "result": "p=0.001"}],
        "interaction_synthesis_table.csv": [
            {"phase4_interaction": "reader_group_x_dfm_surprisal", "survives_controls": True}
        ],
        "reviewer_risk_table.csv": [{"risk": "small participant count", "risk_level": "high"}],
        "final_claim_support_table.csv": [{"claim": "DFM residual gaze", "category": "main"}],
    }.items():
        _write_csv(auto / "tables" / rel, rows)
    for source in [
        "pipeline_overview",
        "dfm_exposure_vs_sensitivity_auc",
        "final_model_roc_curve",
        "final_model_pr_curve",
        "permutation_null_distribution",
        "bootstrap_auc_distribution",
        "feature_stability_coefficients",
        "calibration_plot",
        "interaction_effects_summary",
        "participant_error_analysis",
        "text_exposure_vs_prediction_audit",
    ]:
        _write_png(auto / "figures" / f"{source}.png")
    return auto


def _mini_config(root: Path, auto: Path) -> dict:
    return {
        "run": {"name": "submission_v1", "output_root": str(root / "results")},
        "submission": {
            "frozen_inputs": {
                "autoresearch_dir": str(auto),
                "autoresearch_analysis_dir": str(root / "analysis" / "autoresearch_v1"),
            },
            "output_layout": {
                "paper_dir": str(root / "paper" / "submission_v1"),
                "analysis_dir": str(root / "analysis" / "submission_v1"),
            },
            "no_new_core_labels": True,
            "no_new_feature_families": True,
            "selected_model": {
                "feature_group": FINAL_MODEL_GROUP,
                "model": "logistic_regression",
                "split_name": "leave_one_participant_out",
            },
            "expected_metrics": {
                "roc_auc": 0.89,
                "pr_auc": 0.86,
                "balanced_accuracy": 0.83,
                "macro_f1": 0.83,
                "brier_score": 0.12,
                "calibration_intercept": -0.2,
                "calibration_slope": 0.9,
                "n_predictions": 6,
                "skipped_folds": 0,
                "tolerance": 0.001,
            },
            "prohibited_variables": ["n_words_read", "n_speeches", "participant_id", "reader_group"],
        },
    }


def test_submission_config_parsing(tmp_path: Path) -> None:
    auto = _write_frozen_autoresearch(tmp_path)
    config = _mini_config(tmp_path, auto)
    assert config["submission"]["selected_model"]["feature_group"] == FINAL_MODEL_GROUP
    assert "n_words_read" in config["submission"]["prohibited_variables"]


def test_submission_package_build_and_validation(tmp_path: Path) -> None:
    auto = _write_frozen_autoresearch(tmp_path)
    config = _mini_config(tmp_path, auto)
    out = tmp_path / "submission"
    manifest = build_submission_package(config, out, repo_root=tmp_path)
    assert manifest["status"] == "complete"
    report = validate_submission_package(config, out, repo_root=tmp_path)
    assert report["status"] == "passed", report["errors"]


def test_claim_evidence_ledger_and_metric_consistency(tmp_path: Path) -> None:
    auto = _write_frozen_autoresearch(tmp_path)
    config = _mini_config(tmp_path, auto)
    out = tmp_path / "submission"
    build_submission_package(config, out, repo_root=tmp_path)
    ledger = pd.read_csv(tmp_path / "analysis" / "submission_v1" / "claim_evidence_ledger.csv")
    assert set(ledger["claim_id"]) == {f"C{idx:02d}" for idx in range(1, 11)}
    manuscript = (tmp_path / "paper" / "submission_v1" / "sections" / "06_results.tex").read_text()
    assert "ROC-AUC 0.8900" in manuscript
    assert "PR-AUC 0.8600" in manuscript


def test_figure_table_references_and_required_sections(tmp_path: Path) -> None:
    auto = _write_frozen_autoresearch(tmp_path)
    config = _mini_config(tmp_path, auto)
    out = tmp_path / "submission"
    build_submission_package(config, out, repo_root=tmp_path)
    paper = tmp_path / "paper" / "submission_v1"
    all_text = "\n".join((paper / "sections" / f"{section}.tex").read_text() for section in MANUSCRIPT_SECTIONS)
    for spec in SUBMISSION_TABLES.values():
        assert spec["label"] in all_text
    for spec in SUBMISSION_FIGURES.values():
        assert spec["label"] in all_text
    for section in SUPPLEMENT_SECTIONS:
        assert (paper / "supplement_sections" / f"{section}.tex").exists()


def test_no_prohibited_claims_variables_and_repro_scripts(tmp_path: Path) -> None:
    auto = _write_frozen_autoresearch(tmp_path)
    config = _mini_config(tmp_path, auto)
    out = tmp_path / "submission"
    build_submission_package(config, out, repo_root=tmp_path)
    text = "\n".join(path.read_text() for path in (tmp_path / "paper" / "submission_v1" / "sections").glob("*.tex"))
    assert "validated clinical diagnosis" not in text.lower()
    assert "standalone segmentation main effect is the main" not in text.lower()
    features = json.loads((out / "manifest.json").read_text())["selected_model"]["features"]
    assert not set(features).intersection(config["submission"]["prohibited_variables"])
    repro = tmp_path / "paper" / "submission_v1" / "reproducibility"
    assert (repro / "reproduce_submission_package.sh").exists()
    assert (repro / "reproduce_autoresearch_v1.sh").exists()
