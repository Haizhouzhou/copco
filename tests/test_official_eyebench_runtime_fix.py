from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.benchmark_bridge import PROHIBITED_FEATURES
from copco_eye_bench.official_eyebench_runtime_fix import (
    REQUIRED_GITIGNORE_PATTERNS,
    VALID_DECISION_CATEGORIES,
    run_official_eyebench_runtime_fix,
    validate_gitignore,
    validate_official_eyebench_runtime_fix,
    validate_official_eyebench_runtime_fix_config,
    write_runtime_decision_report,
)
from copco_eye_bench.official_eyebench_sota_check import SOTA_TYP_COLUMNS
from tests.test_official_eyebench_alignment import _write_fake_eyebench
from tests.test_official_eyebench_sota_check import _write_fake_processed_copco


def _mini_runtime_config(tmp_path: Path) -> dict:
    prohibited = sorted(
        PROHIBITED_FEATURES
        | {
            "participant_id",
            "speech_id",
            "text_id",
            "unique_trial_id",
            "unique_paragraph_id",
            "dyslexia",
            "RCS_score",
            "reader_group",
            "reader_group_binary",
        }
    )
    return {
        "run": {"name": "official_eyebench_runtime_fix_v1", "output_root": str(tmp_path / "results")},
        "official_eyebench_runtime_fix": {
            "version": "v1",
            "eyebench": {
                "path": "eyebench",
                "global_processed_dir": "eyebench/data/processed",
                "processed_copco_dir": "eyebench/data/CopCo/processed",
                "folds_metadata_dir": "eyebench/data/CopCo/folds_metadata",
                "labels_dir": "eyebench/data/CopCo/labels",
                "run_preprocessing_in_cli": False,
            },
            "runtime_workspace": {
                "envs_dir": "eyebench/.envs",
                "conda_pkgs_dir": "eyebench/.conda_pkgs",
                "pip_cache_dir": "eyebench/.pip_cache",
                "cache_dir": "eyebench/.cache",
                "runtime_logs_dir": "eyebench/.runtime_logs",
            },
            "environment": {
                "primary_conda_env": "copco",
                "py312_minimal_prefix": "eyebench/.envs/eyebench_official_py312_minimal",
                "exact_prefix": "eyebench/.envs/eyebench_official_py312",
                "cpu_fallback_prefix": "eyebench/.envs/eyebench_official_cpu_runtime",
            },
            "repo_analysis_dir": str(tmp_path / "analysis" / "official_eyebench_runtime_fix_v1"),
            "deterministic_seed": 173,
            "no_new_labels": True,
            "no_feature_engineering_search": True,
            "no_broad_model_search": True,
            "forbid_random_word_level_split": True,
            "tasks": ["CopCo_TYP", "CopCo_RCS"],
            "split_regimes": ["unseen_reader", "unseen_text", "unseen_reader_and_text"],
            "prohibited_features": prohibited,
            "decision_gates": {
                "CopCo_TYP": {
                    "formatted_table": "eyebench/results/formatted_eyebench_benchmark_results/CopCo_TYP_test.csv"
                }
            },
        },
    }


def test_official_eyebench_runtime_fix_config_parses() -> None:
    import yaml

    config = yaml.safe_load(Path("configs/official_eyebench_runtime_fix_v1.yaml").read_text())
    assert validate_official_eyebench_runtime_fix_config(config)["status"] == "passed"


def test_runtime_fix_end_to_end_blocker_with_fake_data(tmp_path: Path) -> None:
    participants = [f"P{idx:02d}" for idx in range(1, 9)]
    speeches = ["S1", "S2", "S3", "S4"]
    eyebench = _write_fake_eyebench(tmp_path, participants, speeches)
    _write_fake_processed_copco(eyebench, participants, speeches)
    (eyebench / ".runtime_logs").mkdir(parents=True, exist_ok=True)
    config = _mini_runtime_config(tmp_path)
    out = tmp_path / "results" / "official_eyebench_runtime_fix_v1_test"

    manifest = run_official_eyebench_runtime_fix(config, out, repo_root=tmp_path)

    assert manifest["status"] == "complete"
    assert manifest["decision_category"] == "blocked_by_environment"
    metrics = pd.read_csv(out / "typ" / "d3_lite_trial_metrics.csv")
    reader = pd.read_csv(out / "typ" / "d3_lite_reader_aggregated_metrics.csv")
    decision = pd.read_json(out / "official_sota_decision.json", typ="series")
    assert set(SOTA_TYP_COLUMNS).issubset(metrics.columns)
    assert set(SOTA_TYP_COLUMNS).issubset(reader.columns)
    assert decision["official_sota_claim_allowed"] is False


def test_runtime_fix_validator_accepts_blocker_reports(tmp_path: Path) -> None:
    participants = [f"P{idx:02d}" for idx in range(1, 9)]
    speeches = ["S1", "S2", "S3", "S4"]
    eyebench = _write_fake_eyebench(tmp_path, participants, speeches)
    _write_fake_processed_copco(eyebench, participants, speeches)
    (tmp_path / ".gitignore").write_text("\n".join(REQUIRED_GITIGNORE_PATTERNS) + "\n", encoding="utf-8")
    config = _mini_runtime_config(tmp_path)
    out = tmp_path / "results" / "official_eyebench_runtime_fix_v1_test"

    run_official_eyebench_runtime_fix(config, out, repo_root=tmp_path)
    report = validate_official_eyebench_runtime_fix(config, out, repo_root=tmp_path)

    assert report["status"] == "passed", report
    assert report["decision_category"] in VALID_DECISION_CATEGORIES


def test_official_sota_gate_cannot_pass_without_required_gates(tmp_path: Path) -> None:
    dirs = {
        "repo_analysis": tmp_path / "analysis",
        "repo_tables": tmp_path / "analysis" / "tables",
        "result_analysis": tmp_path / "result_analysis",
        "result_tables": tmp_path / "result_tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    decision = write_runtime_decision_report(
        _mini_runtime_config(tmp_path),
        dirs,
        tmp_path / "out",
        {"official_environment_ready": False, "environment_kind": "none"},
        {"processed_data_exists": True},
        [],
        {"official_evaluator_run": True, "official_result_format_validated": True},
        pd.DataFrame([{"status": "complete"}]),
        pd.DataFrame([{"status": "complete"}]),
        {
            "heldout_reader_rows_used_for_fit": False,
            "heldout_text_rows_used_for_fit": False,
            "reader_group_used": False,
        },
    )
    assert decision["decision_category"] == "blocked_by_environment"
    assert decision["official_sota_claim_allowed"] is False


def test_runtime_fix_prohibited_predictors_and_leakage_fields() -> None:
    import yaml

    config = yaml.safe_load(Path("configs/official_eyebench_runtime_fix_v1.yaml").read_text())
    prohibited = set(config["official_eyebench_runtime_fix"]["prohibited_features"])
    assert {"participant_id", "speech_id", "text_id", "dyslexia", "RCS_score"}.issubset(prohibited)
    residual = config["official_eyebench_runtime_fix"]["residualization"]
    assert residual["reader_group_never_used"] is True
    assert residual["no_held_out_reader_rows_used_for_residual_fitting"] is True
    assert residual["no_held_out_text_rows_used_for_residual_fitting"] is True


def test_gitignore_protects_eyebench_runtime_paths() -> None:
    report = validate_gitignore(Path("."))
    assert report["status"] == "passed", report
