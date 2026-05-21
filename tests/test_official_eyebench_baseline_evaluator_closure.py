from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.benchmark_bridge import PROHIBITED_FEATURES
from copco_eye_bench.official_eyebench_baseline_evaluator_closure import (
    COMMAND_EVIDENCE_COLUMNS,
    VALID_DECISION_CATEGORIES,
    validate_gitignore,
    validate_official_eyebench_baseline_evaluator_closure,
    validate_official_eyebench_baseline_evaluator_closure_config,
    write_d3_reuse_validation_report,
    write_decision_report,
)
from copco_eye_bench.official_eyebench_sota_check import OFFICIAL_SPLITS, SOTA_TYP_COLUMNS


def _mini_config(tmp_path: Path) -> dict:
    prohibited = sorted(
        PROHIBITED_FEATURES
        | {
            "participant_id",
            "speech_id",
            "text_id",
            "unique_trial_id",
            "reader_group",
            "reader_group_binary",
            "dyslexia",
            "RCS_score",
        }
    )
    return {
        "run": {
            "name": "official_eyebench_baseline_evaluator_closure_v1",
            "output_root": str(tmp_path / "results"),
        },
        "official_eyebench_baseline_evaluator_closure": {
            "require_slurm_job": False,
            "no_new_labels": True,
            "no_feature_engineering_search": True,
            "no_broad_model_search": True,
            "forbid_random_word_level_split": True,
            "eyebench": {
                "path": "eyebench",
                "global_processed_dir": "eyebench/data/processed",
                "processed_copco_dir": "eyebench/data/CopCo/processed",
                "folds_metadata_dir": "eyebench/data/CopCo/folds_metadata",
                "labels_dir": "eyebench/data/CopCo/labels",
                "official_command_markdown": "eyebench/run_commands/CopCo_TYP.md",
                "official_logistic_sweep_config": (
                    "eyebench/sweeps/CopCo_TYP_20251104/configs/"
                    "LogisticRegressionMLArgs_CopCo_TYP.yaml"
                ),
                "official_logistic_bash_script": (
                    "eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/"
                    "LogisticRegressionMLArgs/"
                    "LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh"
                ),
                "official_random_forest_bash_script": (
                    "eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/"
                    "RandomForestMLArgs/RandomForestMLArgs_CopCo_TYP_folds_0_1_2_3.sh"
                ),
                "official_ml_test_script": "eyebench/src/run/single_run/test_ml.py",
                "official_evaluator_script": "eyebench/src/run/multi_run/raw_to_processed_results.py",
            },
            "runtime_workspace": {
                "runtime_logs_dir": "eyebench/.runtime_logs",
                "wandb_dir": "eyebench/wandb",
                "cache_dir": "eyebench/.cache",
                "pip_cache_dir": "eyebench/.pip_cache",
                "envs_dir": "eyebench/.envs",
            },
            "runtime": {
                "py312_minimal_prefix": "eyebench/.envs/eyebench_official_py312_minimal",
                "allow_import_driven_pip_repair": False,
                "import_check_modules": ["pandas"],
                "pip_repair_packages": {},
            },
            "previous_runtime_fix": {
                "d3_trial_metrics": str(tmp_path / "prev" / "trial.csv"),
                "d3_reader_metrics": str(tmp_path / "prev" / "reader.csv"),
                "d3_predictions": str(tmp_path / "prev" / "pred.csv"),
                "d3_trial_result_format": str(tmp_path / "prev" / "trial_level_test_results.csv"),
                "d3_leakage_report": str(tmp_path / "prev" / "leakage.md"),
                "local_diagnostic_baseline_metrics": str(tmp_path / "prev" / "baseline.csv"),
            },
            "repo_analysis_dir": str(tmp_path / "analysis"),
            "tasks": ["CopCo_TYP"],
            "split_regimes": list(OFFICIAL_SPLITS),
            "baseline": {"reasonable_tolerance": 0.05},
            "decision_gates": {
                "CopCo_TYP": {
                    "formatted_table": "eyebench/results/formatted_eyebench_benchmark_results/CopCo_TYP_test.csv"
                }
            },
            "prohibited_features": prohibited,
        },
    }


def _dirs(tmp_path: Path) -> dict[str, Path]:
    dirs = {
        "repo_analysis": tmp_path / "analysis",
        "repo_tables": tmp_path / "analysis" / "tables",
        "result_analysis": tmp_path / "result_analysis",
        "result_tables": tmp_path / "result_tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_fake_d3_previous(tmp_path: Path) -> None:
    prev = tmp_path / "prev"
    prev.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in OFFICIAL_SPLITS:
        rows.append(
            {
                "mode": "official_eyebench_subset",
                "model_name": "D3_EyeBench_Lite",
                "claim_type": "official_compatible",
                "task": "CopCo_TYP",
                "split_name": split,
                "evaluation_level": "official_trial_level_fold_mean",
                "n_features": 12,
                "n_predictions": 10,
                "usable_folds": 4,
                "skipped_folds": 0,
                "roc_auc": 0.7,
                "pr_auc": 0.6,
                "balanced_accuracy": 0.65,
                "macro_f1": 0.64,
                "brier_score": 0.2,
                "status": "complete",
                "skip_reason": "",
            }
        )
    pd.DataFrame(rows, columns=SOTA_TYP_COLUMNS).to_csv(prev / "trial.csv", index=False)
    reader = pd.DataFrame(rows, columns=SOTA_TYP_COLUMNS)
    reader["evaluation_level"] = "reader_aggregated"
    reader.to_csv(prev / "reader.csv", index=False)
    preds = []
    for split in OFFICIAL_SPLITS:
        for fold in range(4):
            preds.append(
                {
                    "split_name": split,
                    "fold_id": fold,
                    "participant_id": f"P{fold}",
                    "speech_id": f"S{fold}",
                    "text_id": f"T{fold}",
                    "unique_trial_id": f"{split}_{fold}",
                    "y_true": fold % 2,
                    "y_score": 0.4 + 0.1 * (fold % 2),
                }
            )
    pd.DataFrame(preds).to_csv(prev / "pred.csv", index=False)
    pd.DataFrame(
        [
            {
                "label": 0,
                "prediction_prob": 0.3,
                "binary_prediction": 0,
                "eval_regime": "unseen_subject_seen_item",
                "eval_type": "test",
                "fold_index": 0,
            }
        ]
    ).to_csv(prev / "trial_level_test_results.csv", index=False)
    (prev / "leakage.md").write_text(
        "\n".join(
            [
                "Held-out reader rows used for residual fitting: False",
                "Held-out text rows used for residual fitting: False",
                "Reader group used in residualization: False",
            ]
        ),
        encoding="utf-8",
    )


def test_closure_config_parses() -> None:
    import yaml

    config = yaml.safe_load(
        Path("configs/official_eyebench_baseline_evaluator_closure_v1.yaml").read_text()
    )
    assert validate_official_eyebench_baseline_evaluator_closure_config(config)["status"] == "passed"


def test_decision_gate_cannot_pass_without_command_source_baseline(tmp_path: Path) -> None:
    decision = write_decision_report(
        _mini_config(tmp_path),
        tmp_path / "out",
        _dirs(tmp_path),
        {"slurm_job_id": "123"},
        {"status": "passed", "fold_validation_errors": []},
        {"all_imports_ok": True, "pip_check_ok": True},
        {"baseline_reproduction_pass": False, "exact_non_installable_reason": "wandb api key"},
        {"official_evaluator_run": False, "exact_result_format_validated": True},
        {
            "status": "passed",
            "no_target_leakage": True,
            "no_participant_id_predictor": True,
            "no_speech_id_text_id_predictor": True,
            "no_exposure_count_predictor": True,
        },
        {"d3_beats_strongest_published_baseline": True},
    )
    assert decision["decision_category"] == "blocked_by_baseline_reproduction"
    assert decision["official_sota_claim_allowed"] is False


def test_wandb_api_absence_not_blocker_when_local_baseline_passes(tmp_path: Path) -> None:
    decision = write_decision_report(
        _mini_config(tmp_path),
        tmp_path / "out",
        _dirs(tmp_path),
        {"slurm_job_id": "123"},
        {"status": "passed", "fold_validation_errors": []},
        {"all_imports_ok": True, "pip_check_ok": True},
        {
            "baseline_reproduction_pass": True,
            "online_wandb_baseline_reproduced": False,
            "wandb_api_available": False,
            "wandb_online_lookup_status": "failed",
            "local_official_derived_baseline_attempted": True,
            "local_official_derived_baseline_pass": True,
            "local_official_derived_baseline_metrics_present": True,
        },
        {"official_evaluator_run": False, "exact_result_format_validated": True},
        {
            "status": "passed",
            "no_target_leakage": True,
            "no_participant_id_predictor": True,
            "no_speech_id_text_id_predictor": True,
            "no_exposure_count_predictor": True,
        },
        {
            "d3_beats_strongest_published_baseline": False,
            "d3_beats_local_official_derived_baseline": False,
        },
    )
    assert decision["wandb_api_failure_is_scientific_blocker"] is False
    assert decision["local_official_derived_baseline_pass"] is True
    assert decision["decision_category"] == "official_compatible_but_not_sota"


def test_local_official_baseline_can_support_local_baseline_sota_category(tmp_path: Path) -> None:
    decision = write_decision_report(
        _mini_config(tmp_path),
        tmp_path / "out",
        _dirs(tmp_path),
        {"slurm_job_id": "123"},
        {"status": "passed", "fold_validation_errors": []},
        {"all_imports_ok": True, "pip_check_ok": True},
        {
            "baseline_reproduction_pass": True,
            "online_wandb_baseline_reproduced": False,
            "local_official_derived_baseline_attempted": True,
            "local_official_derived_baseline_pass": True,
            "local_official_derived_baseline_metrics_present": True,
        },
        {"official_evaluator_run": False, "exact_result_format_validated": True},
        {
            "status": "passed",
            "no_target_leakage": True,
            "no_participant_id_predictor": True,
            "no_speech_id_text_id_predictor": True,
            "no_exposure_count_predictor": True,
        },
        {
            "d3_beats_strongest_published_baseline": False,
            "d3_beats_local_official_derived_baseline": True,
        },
    )
    assert decision["decision_category"] == "official_compatible_local_baseline_sota"
    assert decision["official_sota_claim_allowed"] is False
    assert decision["official_compatible_local_baseline_sota_supported"] is True


def test_decision_gate_requires_evaluator_or_exact_format(tmp_path: Path) -> None:
    decision = write_decision_report(
        _mini_config(tmp_path),
        tmp_path / "out",
        _dirs(tmp_path),
        {"slurm_job_id": "123"},
        {"status": "passed", "fold_validation_errors": []},
        {"all_imports_ok": True, "pip_check_ok": True},
        {"baseline_reproduction_pass": True},
        {"official_evaluator_run": False, "exact_result_format_validated": False},
        {
            "status": "passed",
            "no_target_leakage": True,
            "no_participant_id_predictor": True,
            "no_speech_id_text_id_predictor": True,
            "no_exposure_count_predictor": True,
        },
        {"d3_beats_strongest_published_baseline": True},
    )
    assert decision["decision_category"] == "blocked_by_evaluator"


def test_d3_reuse_validation_schema_and_no_prohibited_predictors(tmp_path: Path) -> None:
    _write_fake_d3_previous(tmp_path)
    report = write_d3_reuse_validation_report(
        _mini_config(tmp_path), tmp_path / "out", _dirs(tmp_path), tmp_path
    )
    assert report["status"] == "passed"
    assert report["no_participant_id_predictor"] is True
    assert report["no_speech_id_text_id_predictor"] is True
    assert report["no_exposure_count_predictor"] is True


def test_command_source_evidence_schema_blocks_manual_baseline() -> None:
    row = {
        column: ""
        for column in COMMAND_EVIDENCE_COLUMNS
    }
    row["command_source_file"] = "local_diagnostic.py"
    row["status"] = "complete"
    frame = pd.DataFrame([row], columns=COMMAND_EVIDENCE_COLUMNS)
    assert set(COMMAND_EVIDENCE_COLUMNS).issubset(frame.columns)
    assert "local_diagnostic" in frame["command_source_file"].iloc[0]


def test_missing_package_cannot_be_blocker_without_install_attempt_record() -> None:
    report = {
        "before": [{"module": "wandb", "ok": False, "error": "ModuleNotFoundError"}],
        "install_attempts": [],
        "after": [{"module": "wandb", "ok": False, "error": "ModuleNotFoundError"}],
    }
    missing = [row for row in report["after"] if not row["ok"]]
    assert missing
    assert report["install_attempts"] == []


def test_tmux_unavailable_with_generated_command_can_still_be_command_source() -> None:
    evidence = {
        "tmux_used": False,
        "underlying_generated_command_used": True,
        "command_source_file": "eyebench/sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/x.sh",
    }
    assert evidence["underlying_generated_command_used"] is True
    assert "eyebench/sweeps" in evidence["command_source_file"]


def test_closure_gitignore_protects_runtime_paths(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "\n".join(
            [
                "eyebench/data/",
                "eyebench/results/",
                "eyebench/logs/",
                "eyebench/wandb/",
                "eyebench/.cache/",
                "eyebench/.conda_pkgs/",
                "eyebench/.pip_cache/",
                "eyebench/.envs/",
                "eyebench/.pytest_cache/",
                "eyebench/.runtime_logs/",
                "eyebench/**/*.feather",
                "eyebench/**/*.parquet",
                "eyebench/**/*.pkl",
                "eyebench/**/*.pt",
                "eyebench/**/*.ckpt",
                "results/official_eyebench_runtime_fix_v1_*/",
                "results/official_eyebench_baseline_evaluator_closure_v1_*/",
                "results/official_eyebench_baseline_evaluator_closure_v1_sbatch/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert validate_gitignore(tmp_path)["status"] == "passed"


def test_closure_validator_rejects_invalid_sota_gate(tmp_path: Path) -> None:
    config = _mini_config(tmp_path)
    out = tmp_path / "out"
    (out / "baseline").mkdir(parents=True)
    (out / "evaluator").mkdir()
    (out / "typ").mkdir()
    (out / "preflight").mkdir()
    (out / "runtime").mkdir()
    (out / "data").mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    (out / "config_validation.json").write_text("{}", encoding="utf-8")
    for sub, name in [
        ("preflight", "preflight_report.json"),
        ("runtime", "import_repair_report.json"),
        ("data", "data_fold_revalidation_report.json"),
        ("baseline", "official_baseline_command_source_report.json"),
        ("evaluator", "official_evaluator_closure_report.json"),
        ("typ", "d3_reuse_validation_report.json"),
    ]:
        (out / sub / name).write_text("{}", encoding="utf-8")
    (out / "official_sota_decision.json").write_text(
        '{"decision_category":"official_sota_claim_allowed","official_sota_claim_allowed":true}',
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text(
        "\n".join(
            [
                "eyebench/data/",
                "eyebench/results/",
                "eyebench/logs/",
                "eyebench/wandb/",
                "eyebench/.cache/",
                "eyebench/.conda_pkgs/",
                "eyebench/.pip_cache/",
                "eyebench/.envs/",
                "eyebench/.pytest_cache/",
                "eyebench/.runtime_logs/",
                "eyebench/**/*.feather",
                "eyebench/**/*.parquet",
                "eyebench/**/*.pkl",
                "eyebench/**/*.pt",
                "eyebench/**/*.ckpt",
                "results/official_eyebench_runtime_fix_v1_*/",
                "results/official_eyebench_baseline_evaluator_closure_v1_*/",
                "results/official_eyebench_baseline_evaluator_closure_v1_sbatch/",
            ]
        ),
        encoding="utf-8",
    )
    report = validate_official_eyebench_baseline_evaluator_closure(config, out, repo_root=tmp_path)
    assert report["status"] == "failed"
    assert any("official SOTA claim allowed" in error for error in report["errors"])
    assert "official_sota_claim_allowed" in VALID_DECISION_CATEGORIES
    assert "official_compatible_local_baseline_sota" in VALID_DECISION_CATEGORIES


def test_sbatch_templates_use_required_teaching_resources() -> None:
    script_dir = Path("scripts/slurm/official_eyebench_baseline_evaluator_closure_v1")
    for name in [
        "run_local_logistic_baseline.sbatch",
        "run_local_random_forest_baseline.sbatch",
        "run_closure_validation.sbatch",
        "run_full_validation.sbatch",
    ]:
        text = (script_dir / name).read_text(encoding="utf-8")
        assert "#SBATCH --partition=teaching" in text
        assert "#SBATCH --account=mlnlp2.pilot.s3it.uzh" in text
        assert "#SBATCH --qos=normal" in text
        assert "#SBATCH --gres=gpu:0" in text
        assert "#SBATCH --cpus-per-task=64" in text
        assert "#SBATCH --mem=256G" in text
        assert "#SBATCH --time=04:00:00" in text
        assert "WANDB_DISABLED=true" in text
        assert "eyebench/.envs/eyebench_official_py312_minimal" in text or "conda run -n copco" in text
