from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from copco_eye_bench.manuscript_audit import (
    check_claim_ledger,
    check_limitations_coverage,
    check_metric_consistency,
    check_prohibited_claims,
    check_table_figure_refs,
    manuscript_audit_paths,
    run_manuscript_audit,
    validate_manuscript_audit,
)
from copco_eye_bench.submission import (
    FINAL_MODEL_GROUP,
    MANUSCRIPT_SECTIONS,
    SUBMISSION_FIGURES,
    SUBMISSION_TABLES,
    SUPPLEMENT_SECTIONS,
)


METRIC = {
    "analysis": "phase4_confirmatory_participant_prediction",
    "split_name": "leave_one_participant_out",
    "feature_group": FINAL_MODEL_GROUP,
    "model": "logistic_regression",
    "n_features": 12,
    "n_predictions": 57,
    "usable_folds": 57,
    "skipped_folds": 0,
    "roc_auc": 0.8947368421,
    "pr_auc": 0.8640879081,
    "balanced_accuracy": 0.8421052632,
    "macro_f1": 0.8421052632,
    "brier_score": 0.1159416341,
    "calibration_intercept": -0.5321293913,
    "calibration_slope": 0.8693048814,
    "calibration_mean_predicted": 0.3924,
    "calibration_observed_rate": 0.3333,
    "status": "complete",
    "skip_reason": "",
    "fold_validity": "all_test_predictions_generated",
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x01\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _references() -> str:
    keys = [
        "rayner1998eye",
        "duchowski2017eye",
        "kennedy2003d",
        "kliegl2006tracking",
        "hale2001probabilistic",
        "levy2008expectation",
        "smith2013effect",
    ]
    return "\n\n".join(
        f"@article{{{key},\n  title={{Placeholder}},\n  author={{A}},\n  year={{2000}}\n}}"
        for key in keys
    )


def _claim_rows() -> list[dict]:
    return [
        {
            "claim_id": f"C{idx:02d}",
            "claim_text": text,
            "claim_category": category,
            "evidence_file": "tables/final_model_metrics.csv",
            "evidence_table_figure": "Table",
            "metric_statistic": "ROC-AUC 0.8947",
            "sample_size": "57 participants",
            "caveat": "Operational labels; no external dataset.",
            "manuscript_section": "Results",
            "status": status,
        }
        for idx, (text, category, status) in enumerate(
            [
                ("DFM residual gaze profiles predict participant group.", "main", "supported"),
                ("DFM sensitivity dominates DFM exposure.", "supporting", "supported"),
                (
                    "Prediction survives permutation and bootstrap robustness.",
                    "supporting",
                    "supported",
                ),
                (
                    "Cross-fitted residualization avoids held-out participants.",
                    "supporting",
                    "supported",
                ),
                ("Exposure-count variables are absent.", "supporting", "supported"),
                ("Raw speed does not dominate.", "supporting", "supported"),
                (
                    "DFM surprisal interactions provide explanatory support.",
                    "secondary",
                    "partially_supported",
                ),
                ("Boundary opacity is secondary.", "secondary", "partially_supported"),
                ("Standalone segmentation main effect is not supported.", "appendix", "appendix_only"),
                ("Word-level classification is secondary.", "appendix", "appendix_only"),
            ],
            start=1,
        )
    ]


def _write_fixture(root: Path) -> tuple[dict, Path]:
    paper = root / "paper" / "submission_v1"
    analysis = root / "analysis" / "submission_v1"
    result = root / "results" / "submission_v1"
    _write_json(
        result / "manifest.json",
        {
            "selected_model": {
                "selected_feature_group": FINAL_MODEL_GROUP,
                "selected_model": "logistic_regression",
                "split_name": "leave_one_participant_out",
            },
            "final_metrics": METRIC,
        },
    )
    _write_text(
        paper / "main.tex",
        "\n".join([r"\documentclass{article}", r"\begin{document}"]
                  + [rf"\input{{sections/{name}.tex}}" for name in MANUSCRIPT_SECTIONS]
                  + [r"\end{document}"]),
    )
    _write_text(paper / "references.bib", _references())
    for name in MANUSCRIPT_SECTIONS:
        _write_text(paper / "sections" / f"{name}.tex", rf"\section{{{name}}} Draft.")
        _write_text(analysis / "manuscript" / f"{name}.md", f"# {name}\nDraft.")
    _write_text(
        paper / "supplement.tex",
        "\n".join([r"\documentclass{article}", r"\begin{document}"]
                  + [rf"\input{{supplement_sections/{name}.tex}}" for name in SUPPLEMENT_SECTIONS]
                  + [r"\end{document}"]),
    )
    for name in SUPPLEMENT_SECTIONS:
        _write_text(paper / "supplement_sections" / f"{name}.tex", rf"\section{{{name}}} Draft.")
        _write_text(analysis / "supplement" / f"{name}.md", f"# {name}\nDraft.")
    for name in SUBMISSION_FIGURES:
        _write_png(paper / "figures" / f"{name}.png")
    _write_csv(paper / "tables" / "final_model_metrics.csv", [METRIC])
    dfm_rows = [
        {**METRIC, "feature_group": "D1_dfm_exposure_only", "n_features": 3, "roc_auc": 0.4238},
        {**METRIC, "feature_group": "D2_dfm_sensitivity_only", "n_features": 16, "roc_auc": 0.8892},
        METRIC,
        {
            **METRIC,
            "feature_group": "D4_dfm_exposure_plus_sensitivity",
            "n_features": 19,
            "roc_auc": 0.8726,
        },
    ]
    _write_csv(paper / "tables" / "dfm_exposure_vs_sensitivity.csv", dfm_rows)
    for name in SUBMISSION_TABLES:
        path = paper / "tables" / f"{name}.csv"
        if not path.exists():
            _write_csv(path, [{"name": name, "value": "present"}])
    rows = _claim_rows()
    _write_csv(analysis / "claim_evidence_ledger.csv", rows)
    _write_text(analysis / "claim_evidence_ledger.md", "# Claim Ledger\n")
    config = {
        "run": {"name": "final_manuscript_audit_v1", "output_root": str(root / "results")},
        "manuscript_audit": {
            "frozen_inputs": {
                "submission_result_dir": str(result),
                "paper_dir": str(paper),
                "analysis_dir": str(analysis),
            },
            "output_layout": {
                "audit_analysis_dir": str(root / "analysis" / "final_manuscript_audit_v1"),
            },
            "expected_metrics": {
                key: METRIC[key]
                for key in [
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                    "calibration_intercept",
                    "calibration_slope",
                ]
            }
            | {"tolerance": 0.0005},
        },
    }
    return config, result


def test_final_manuscript_audit_config_parsing(tmp_path: Path) -> None:
    config, _ = _write_fixture(tmp_path)
    paths = manuscript_audit_paths(config, tmp_path / "results" / "audit", repo_root=tmp_path)
    assert paths["paper"].name == "submission_v1"
    assert paths["audit_analysis"].name == "final_manuscript_audit_v1"


def test_run_and_validate_manuscript_audit_package(tmp_path: Path) -> None:
    config, _ = _write_fixture(tmp_path)
    out = tmp_path / "results" / "audit"
    manifest = run_manuscript_audit(config, out, repo_root=tmp_path)
    assert manifest["decision"] == "ready_with_minor_manual_edits"
    report = validate_manuscript_audit(config, out, repo_root=tmp_path)
    assert report["status"] == "passed", report["errors"]


def test_metric_claim_and_prohibited_checkers(tmp_path: Path) -> None:
    config, _ = _write_fixture(tmp_path)
    out = tmp_path / "results" / "audit"
    run_manuscript_audit(config, out, repo_root=tmp_path)
    paths = manuscript_audit_paths(config, out, repo_root=tmp_path)
    assert check_metric_consistency(config, paths)["status"] == "passed"
    assert check_claim_ledger(config, paths)["status"] == "passed"
    assert check_prohibited_claims(paths)["status"] == "passed"


def test_table_figure_and_limitations_checkers(tmp_path: Path) -> None:
    config, _ = _write_fixture(tmp_path)
    out = tmp_path / "results" / "audit"
    run_manuscript_audit(config, out, repo_root=tmp_path)
    paths = manuscript_audit_paths(config, out, repo_root=tmp_path)
    assert check_table_figure_refs(paths)["status"] == "passed"
    assert check_limitations_coverage(paths)["status"] == "passed"


def test_validation_fails_when_required_report_missing(tmp_path: Path) -> None:
    config, _ = _write_fixture(tmp_path)
    out = tmp_path / "results" / "audit"
    run_manuscript_audit(config, out, repo_root=tmp_path)
    paths = manuscript_audit_paths(config, out, repo_root=tmp_path)
    (paths["audit_analysis"] / "final_readiness_report.md").unlink()
    report = validate_manuscript_audit(config, out, repo_root=tmp_path)
    assert report["status"] == "failed"
    assert any("final_readiness_report" in error for error in report["errors"])
