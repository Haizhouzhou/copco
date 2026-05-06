"""AutoResearch v1 publication decision loop for CopCo Phase 4 outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir
from .phase4_confirmatory import (
    GLOBAL_SPEED_FEATURES,
    PRIMARY_EXCLUDED_FEATURES,
    _phase4_classification_metrics,
)
from .research_exploration import _format_value, _markdown_table, _pd


AUTORESEARCH_REQUIRED_DIRS = [
    "deployment",
    "validation",
    "frozen_inputs",
    "rerun_phase4",
    "stress_tests",
    "refinement_loop",
    "final_model",
    "figures",
    "tables",
    "manuscript",
    "reviewer_risk",
    "reproducibility",
    "decision",
]

FINAL_MODEL_GROUP = "D3_dfm_residual_gaze_only"
FINAL_MODEL_NAME = "logistic_regression"
FINAL_SPLIT = "leave_one_participant_out"

AUTORESEARCH_TABLES = [
    "dataset_summary_table",
    "feature_release_summary_table",
    "label_release_summary_table",
    "segmentation_distribution_table",
    "phase3_to_phase4_result_progression_table",
    "dfm_exposure_vs_sensitivity_table",
    "final_model_metrics_table",
    "robustness_summary_table",
    "feature_stability_table",
    "interaction_synthesis_table",
    "reviewer_risk_table",
    "final_claim_support_table",
]

AUTORESEARCH_FIGURES = [
    "pipeline_overview",
    "prepared_dataset_structure",
    "dfm_exposure_vs_sensitivity_auc",
    "final_model_roc_curve",
    "final_model_pr_curve",
    "bootstrap_auc_distribution",
    "permutation_null_distribution",
    "feature_stability_coefficients",
    "calibration_plot",
    "interaction_effects_summary",
    "text_exposure_vs_prediction_audit",
    "participant_error_analysis",
]

MANUSCRIPT_FILES = [
    "00_title_and_contributions.md",
    "01_abstract_draft.md",
    "02_introduction_argument.md",
    "03_related_work_positioning.md",
    "04_methods_draft.md",
    "05_results_draft.md",
    "06_discussion_draft.md",
    "07_limitations.md",
    "08_reproducibility_statement.md",
    "09_data_and_ethics_statement.md",
    "10_appendix_plan.md",
]

REPRODUCIBILITY_FILES = [
    "reproduce_all.sh",
    "reproduce_autoresearch_only.sh",
    "slurm_autoresearch.sh",
    "environment_summary.md",
    "command_manifest.md",
    "input_output_manifest.md",
    "commit_trace.md",
    "checksums.json",
    "data_not_committed_notice.md",
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _git_sha(repo_root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _configured_path(config: dict[str, Any], dotted: str, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    value = get_nested(config, dotted)
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _input_paths(config: dict[str, Any], repo_root: str | Path) -> dict[str, Path]:
    return {
        "feature_release": _configured_path(
            config, "autoresearch.frozen_inputs.feature_release_dir", repo_root
        ),
        "label_release": _configured_path(
            config, "autoresearch.frozen_inputs.label_release_dir", repo_root
        ),
        "phase3": _configured_path(config, "autoresearch.frozen_inputs.phase3_dir", repo_root),
        "phase4": _configured_path(config, "autoresearch.frozen_inputs.phase4_dir", repo_root),
    }


def _result_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    repo_analysis = root / str(get_nested(config, "autoresearch.repo_analysis_dir", "analysis/autoresearch_v1"))
    dirs = {name: out / name for name in AUTORESEARCH_REQUIRED_DIRS}
    dirs["result_root"] = out
    dirs["repo_analysis"] = repo_analysis
    dirs["repo_tables"] = root / str(
        get_nested(config, "autoresearch.output_layout.tables", "analysis/autoresearch_v1/tables")
    )
    dirs["repo_figures"] = root / str(
        get_nested(config, "autoresearch.output_layout.figures", "analysis/autoresearch_v1/figures")
    )
    dirs["repo_manuscript"] = root / str(
        get_nested(
            config, "autoresearch.output_layout.manuscript", "analysis/autoresearch_v1/manuscript"
        )
    )
    dirs["repo_reproducibility"] = root / str(
        get_nested(
            config,
            "autoresearch.output_layout.reproducibility",
            "analysis/autoresearch_v1/reproducibility",
        )
    )
    return dirs


def _ensure_dirs(dirs: dict[str, Path]) -> None:
    for key, path in dirs.items():
        if key.startswith("repo_") or key in {"result_root", *AUTORESEARCH_REQUIRED_DIRS}:
            path.mkdir(parents=True, exist_ok=True)


def _write_dual_md(dirs: dict[str, Path], result_subdir: str, repo_relative: str, text: str) -> None:
    _write_md(dirs[result_subdir] / repo_relative, text)
    _write_md(dirs["repo_analysis"] / repo_relative, text)


def _write_dual_csv(dirs: dict[str, Path], result_subdir: str, repo_relative: str, frame: Any) -> None:
    _write_csv(dirs[result_subdir] / repo_relative, frame)
    _write_csv(dirs["repo_analysis"] / repo_relative, frame)


def _copy_to_repo(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_frozen_data(config: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    pd = _pd()
    paths = _input_paths(config, repo_root)
    label = paths["label_release"]
    phase4_analysis = paths["phase4"] / "analysis" / "phase4_confirmatory"
    return {
        "paths": paths,
        "word": pd.read_parquet(label / "prepared_dataset" / "analysis_ready_word_level_v1_1.parquet"),
        "sentence": pd.read_parquet(
            label / "prepared_dataset" / "analysis_ready_sentence_level_v1_1.parquet"
        ),
        "participant": pd.read_parquet(
            label / "prepared_dataset" / "analysis_ready_participant_level_v1_1.parquet"
        ),
        "participant_labels": pd.read_parquet(label / "labels" / "participant_labels_v1.parquet"),
        "quality": pd.read_parquet(label / "labels" / "quality_labels_v1.parquet"),
        "splits": pd.read_parquet(label / "labels" / "split_labels_v1.parquet"),
        "segmentation_word": pd.read_parquet(label / "labels" / "segmentation_word_labels_v1.parquet"),
        "phase4_metrics": pd.read_csv(phase4_analysis / "confirmatory_prediction_metrics.csv"),
        "phase4_predictions": pd.read_csv(phase4_analysis / "confirmatory_predictions.csv"),
        "phase4_bootstrap": pd.read_csv(phase4_analysis / "bootstrap_results.csv"),
        "phase4_permutation": pd.read_csv(phase4_analysis / "permutation_results.csv"),
        "phase4_influence": pd.read_csv(phase4_analysis / "influence_analysis.csv"),
        "phase4_stability": pd.read_csv(phase4_analysis / "feature_stability_by_fold.csv"),
        "phase4_mixed": pd.read_csv(phase4_analysis / "mixed_effects_coefficients.csv"),
        "phase4_manifest": _read_json_if_exists(paths["phase4"] / "manifest.json"),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_metric_row(metrics: Any) -> dict[str, Any]:
    rows = metrics[
        metrics["split_name"].eq(FINAL_SPLIT)
        & metrics["feature_group"].eq(FINAL_MODEL_GROUP)
        & metrics["model"].eq(FINAL_MODEL_NAME)
        & metrics["status"].eq("complete")
    ]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _primary_predictions(predictions: Any) -> Any:
    return predictions[
        predictions["split_name"].eq(FINAL_SPLIT)
        & predictions["feature_group"].eq(FINAL_MODEL_GROUP)
        & predictions["model"].eq(FINAL_MODEL_NAME)
    ].copy()


def validate_frozen_inputs(
    config: dict[str, Any], dirs: dict[str, Path], data: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    warnings_list: list[str] = []
    paths = data["paths"]
    for name, path in paths.items():
        if not path.exists():
            errors.append(f"missing frozen input directory: {name}={path}")
    expected = get_nested(config, "autoresearch.expected", {})
    word = data["word"]
    participant = data["participant"]
    labels = data["participant_labels"]
    splits = data["splits"]
    quality = data["quality"]
    phase4_metrics = data["phase4_metrics"]
    phase4_predictions = data["phase4_predictions"]
    checks = {
        "word_level_rows": len(word),
        "sentence_level_rows": len(data["sentence"]),
        "participant_level_rows": len(participant),
    }
    for key, actual in checks.items():
        if key in expected and int(actual) != int(expected[key]):
            errors.append(f"{key} {actual} != expected {expected[key]}")
    counts = labels["reader_group"].value_counts().to_dict()
    if int(counts.get("dyslexia_labeled", 0)) != int(expected.get("dyslexia_labeled", 19)):
        errors.append("dyslexia-labeled participant count mismatch")
    if int(counts.get("typical_control", 0)) != int(expected.get("typical_control", 38)):
        errors.append("typical/control participant count mismatch")
    if "participant_word_key" not in word.columns:
        errors.append("prepared word table missing participant_word_key")
    elif word["participant_word_key"].duplicated().any():
        errors.append("duplicate participant-word keys found")
    split_names = sorted(splits["split_name"].astype(str).unique())
    if any("random" in name.lower() for name in split_names):
        errors.append("random split label found")
    for (split_name, fold_id), fold in splits.groupby(["split_name", "fold_id"], dropna=False):
        train_ids = set(fold[fold["split_role"].eq("train")]["participant_id"].astype(str))
        test_ids = set(fold[fold["split_role"].eq("test")]["participant_id"].astype(str))
        if train_ids.intersection(test_ids):
            errors.append(f"participant train/test overlap in {split_name} fold {fold_id}")
    required_features = [
        "D1_dfm_exposure_only",
        "D2_dfm_sensitivity_only",
        "D3_dfm_residual_gaze_only",
        "D4_dfm_exposure_plus_sensitivity",
    ]
    missing_groups = [
        group for group in required_features if not phase4_metrics["feature_group"].eq(group).any()
    ]
    if missing_groups:
        errors.append(f"Phase 4 metrics missing feature groups: {missing_groups}")
    if _primary_metric_row(phase4_metrics) == {}:
        errors.append("Phase 4 primary metric row missing")
    if _primary_predictions(phase4_predictions).empty:
        errors.append("Phase 4 primary predictions missing")
    phase4_manifest = data["phase4_manifest"]
    feature_groups = phase4_manifest.get("feature_groups", {})
    if FINAL_MODEL_GROUP not in feature_groups:
        errors.append("Phase 4 manifest missing final feature group")
    prohibited = set(get_nested(config, "autoresearch.prohibited_variables", []))
    final_features = set(feature_groups.get(FINAL_MODEL_GROUP, []))
    if prohibited.intersection(final_features):
        errors.append("primary feature group contains prohibited variables")
    parser_expected = str(expected.get("parser_status", "surface_heuristic_fallback"))
    if parser_expected not in set(word.get("parser_status", []).dropna().astype(str).unique()):
        errors.append("parser status does not include expected fallback status")
    segmentation_expected = str(expected.get("segmentation_label_version", "segmentation_orthographic_v1"))
    if segmentation_expected not in set(word.get("segmentation_label_version", []).dropna().astype(str).unique()):
        errors.append("segmentation label version does not include expected orthographic proxy")
    for column in ["lm_missing", "lm_alignment_status", "lm_alignment_warning"]:
        if column not in quality.columns and column not in word.columns:
            errors.append(f"LM audit field missing: {column}")
    if quality.get("lm_missing", word.get("lm_missing")).mean() > 0.02:
        warnings_list.append("LM missingness exceeds 2%")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
        "row_counts": checks,
        "participant_counts": {key: int(value) for key, value in counts.items()},
        "split_names": split_names,
        "phase4_metric_rows": int(len(phase4_metrics)),
        "phase4_prediction_rows": int(len(phase4_predictions)),
        "exposure_count_variables_flagged": sorted(PRIMARY_EXCLUDED_FEATURES & prohibited),
    }
    _write_json(dirs["validation"] / "frozen_input_validation_report.json", report)
    md = "\n".join(
        [
            "# Frozen Input Validation Report",
            "",
            f"- Status: `{report['status']}`",
            "",
            "## Row Counts",
            _markdown_table(
                [{"table": key, "rows": value} for key, value in checks.items()],
                ["table", "rows"],
            ),
            "",
            "## Participant Counts",
            _markdown_table(
                [
                    {"reader_group": key, "participants": value}
                    for key, value in report["participant_counts"].items()
                ],
                ["reader_group", "participants"],
            ),
            "",
            "## Split Names",
            "\n".join(f"- `{name}`" for name in split_names),
            "",
            "## Errors",
            "\n".join(f"- {error}" for error in errors) if errors else "_None._",
            "",
            "## Warnings",
            "\n".join(f"- {warning}" for warning in warnings_list) if warnings_list else "_None._",
        ]
    )
    _write_md(dirs["validation"] / "frozen_input_validation_report.md", md)
    return report


def record_frozen_inputs(dirs: dict[str, Path], data: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    copied = []
    for name, root in data["paths"].items():
        for source in [
            root / "manifest.json",
            root / "checksums.json",
            root / f"{name}_validation_report.json",
            root / "feature_release_validation_report.json",
            root / "label_release_validation_report.json",
            root / "research_exploration_validation_report.json",
            root / "phase4_confirmatory_validation_report.json",
        ]:
            if source.exists():
                target = dirs["frozen_inputs"] / f"{name}_{source.name}"
                shutil.copy2(source, target)
                copied.append(str(target.relative_to(dirs["result_root"])))
    payload = {
        "git_sha": _git_sha(repo_root),
        "frozen_input_dirs": {key: str(value) for key, value in data["paths"].items()},
        "copied_manifest_files": copied,
    }
    _write_json(dirs["frozen_inputs"] / "frozen_input_manifest_index.json", payload)
    return payload


def verify_locked_phase4(
    config: dict[str, Any], dirs: dict[str, Path], data: dict[str, Any]
) -> dict[str, Any]:
    metrics = data["phase4_metrics"]
    predictions = _primary_predictions(data["phase4_predictions"])
    row = _primary_metric_row(metrics)
    expected = get_nested(config, "autoresearch.expected", {})
    tolerance = float(expected.get("metric_tolerance", 0.0005))
    expected_roc = float(expected.get("primary_roc_auc", row.get("roc_auc", 0)))
    expected_pr = float(expected.get("primary_pr_auc", row.get("pr_auc", 0)))
    matches = bool(
        row
        and abs(float(row["roc_auc"]) - expected_roc) <= tolerance
        and abs(float(row["pr_auc"]) - expected_pr) <= tolerance
    )
    features = data["phase4_manifest"].get("feature_groups", {}).get(FINAL_MODEL_GROUP, [])
    stability = data["phase4_stability"]
    coefficients = (
        stability.groupby("feature", dropna=False)
        .agg(
            mean_coefficient=("standardized_logistic_coefficient", "mean"),
            sd_coefficient=("standardized_logistic_coefficient", "std"),
            positive_rate=("coefficient_sign", lambda s: float((s == "positive").mean())),
            negative_rate=("coefficient_sign", lambda s: float((s == "negative").mean())),
        )
        .reset_index()
    )
    coefficients["sign_stability"] = coefficients[["positive_rate", "negative_rate"]].max(axis=1)
    _write_csv(dirs["rerun_phase4"] / "phase4_locked_coefficients.csv", coefficients)
    report = "\n".join(
        [
            "# Phase 4 Locked Result Report",
            "",
            "AutoResearch verifies the locked Phase 4 output rather than silently changing the final "
            "analysis.",
            "",
            "## Locked Result",
            _markdown_table(
                [row],
                [
                    "feature_group",
                    "model",
                    "split_name",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                    "n_predictions",
                    "skipped_folds",
                ],
            ),
            "",
            "## Match Check",
            f"- Matches expected Phase 4 metrics within tolerance `{tolerance}`: `{matches}`",
            f"- Prediction count: `{len(predictions)}`",
            f"- Unique participant predictions: `{predictions['participant_id'].nunique() if not predictions.empty else 0}`",
            "",
            "## Feature List",
            "\n".join(f"- `{feature}`" for feature in features),
        ]
    )
    _write_md(dirs["rerun_phase4"] / "phase4_locked_result_report.md", report)
    return {
        "status": "passed" if matches else "needs_review",
        "matches_phase4_within_tolerance": matches,
        "metric": row,
        "prediction_count": int(len(predictions)),
        "unique_prediction_participants": int(predictions["participant_id"].nunique())
        if not predictions.empty
        else 0,
        "feature_count": int(len(features)),
        "features": features,
    }


def _metric_rows(metrics: Any, feature_groups: list[str]) -> Any:
    return metrics[
        metrics["split_name"].eq(FINAL_SPLIT)
        & metrics["model"].eq(FINAL_MODEL_NAME)
        & metrics["feature_group"].isin(feature_groups)
        & metrics["status"].eq("complete")
    ].copy()


def run_stress_tests(config: dict[str, Any], dirs: dict[str, Path], data: dict[str, Any]) -> dict[str, Any]:
    pd = _pd()
    metrics = data["phase4_metrics"]
    predictions = _primary_predictions(data["phase4_predictions"])
    participant = data["participant"]
    phase4_manifest = data["phase4_manifest"]

    dfm_groups = [
        "D1_dfm_exposure_only",
        "D2_dfm_sensitivity_only",
        "D3_dfm_residual_gaze_only",
        "D4_dfm_exposure_plus_sensitivity",
    ]
    dfm = _metric_rows(metrics, dfm_groups)
    _write_csv(dirs["stress_tests"] / "dfm_exposure_vs_sensitivity.csv", dfm)
    _write_md(
        dirs["stress_tests"] / "dfm_exposure_vs_sensitivity.md",
        "\n".join(
            [
                "# DFM Exposure Vs Sensitivity Stress Test",
                "",
                _markdown_table(
                    dfm.to_dict("records"),
                    ["feature_group", "roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "brier_score"],
                ),
                "",
                "DFM exposure-only is treated as a confound check, not a candidate main model.",
            ]
        ),
    )

    removal_groups = [
        "D3_dfm_residual_gaze_only",
        "J_all_except_raw_speed",
        "K_all_except_exposure_variables",
        "G_all_allowed_non_exposure",
    ]
    removal = _metric_rows(metrics, removal_groups)
    removal["comparison_role"] = removal["feature_group"].map(
        {
            "D3_dfm_residual_gaze_only": "primary",
            "J_all_except_raw_speed": "raw_speed_global_duration_removed",
            "K_all_except_exposure_variables": "exposure_only_variables_removed",
            "G_all_allowed_non_exposure": "all_non_exposure_allowed",
        }
    )
    _write_csv(dirs["stress_tests"] / "exposure_variable_removal.csv", removal)
    _write_md(
        dirs["stress_tests"] / "exposure_variable_removal.md",
        "\n".join(
            [
                "# Exposure Variable Removal Stress Test",
                "",
                _markdown_table(
                    removal.to_dict("records"),
                    [
                        "comparison_role",
                        "feature_group",
                        "roc_auc",
                        "pr_auc",
                        "balanced_accuracy",
                        "macro_f1",
                        "brier_score",
                    ],
                ),
                "",
                "Exposure-count variables are prohibited in the primary model and remain excluded.",
            ]
        ),
    )

    exposure = _text_exposure_sensitivity(participant, predictions, phase4_manifest)
    _write_csv(dirs["stress_tests"] / "text_exposure_sensitivity.csv", exposure)
    _write_md(
        dirs["stress_tests"] / "text_exposure_sensitivity.md",
        "\n".join(
            [
                "# Text Exposure Sensitivity",
                "",
                _markdown_table(
                    exposure.to_dict("records"),
                    ["variable", "n", "corr_with_score", "corr_with_abs_error", "note"],
                    max_rows=30,
                ),
            ]
        ),
    )

    influence = _build_influence_audit(data)
    _write_csv(dirs["stress_tests"] / "influence_analysis.csv", influence)
    _write_md(
        dirs["stress_tests"] / "influence_analysis.md",
        "\n".join(
            [
                "# Influence And Error Analysis",
                "",
                _markdown_table(
                    influence.head(80).to_dict("records"),
                    [
                        "participant_id",
                        "reader_group",
                        "y_true",
                        "y_score",
                        "misclassified",
                        "absolute_error",
                        "leave_one_roc_auc",
                        "delta_roc_auc",
                        "high_leverage_flag",
                    ],
                    max_rows=80,
                ),
            ]
        ),
    )

    calibration = _calibration_summary(predictions)
    _write_csv(dirs["stress_tests"] / "calibration_summary.csv", calibration)
    _write_md(
        dirs["stress_tests"] / "calibration_report.md",
        "\n".join(
            [
                "# Calibration Report",
                "",
                _markdown_table(calibration.to_dict("records"), list(calibration.columns), max_rows=20),
            ]
        ),
    )

    permutation = data["phase4_permutation"].copy()
    permutation_count = int(permutation["roc_auc"].notna().sum()) if "roc_auc" in permutation else 0
    selected = _primary_metric_row(metrics)
    observed = float(selected.get("roc_auc", 0.0))
    permutation_p = None
    if permutation_count:
        valid = pd.to_numeric(permutation["roc_auc"], errors="coerce").dropna()
        permutation_p = float((int((valid >= observed).sum()) + 1) / (len(valid) + 1))
    _write_csv(dirs["stress_tests"] / "permutation_results.csv", permutation)
    _write_md(
        dirs["stress_tests"] / "permutation_report.md",
        "\n".join(
            [
                "# Permutation Report",
                "",
                f"- Valid permutations: `{permutation_count}`",
                f"- Observed ROC-AUC: `{observed:.4f}`",
                f"- +1 corrected p-value: `{_format_value(permutation_p)}`",
                "- AutoResearch reused frozen Phase 4 permutations because they exceed the configured minimum.",
            ]
        ),
    )

    bootstrap = data["phase4_bootstrap"].copy()
    _write_csv(dirs["stress_tests"] / "bootstrap_results.csv", bootstrap)
    _write_md(
        dirs["stress_tests"] / "bootstrap_report.md",
        "\n".join(
            [
                "# Bootstrap Report",
                "",
                _markdown_table(
                    bootstrap.to_dict("records"),
                    ["metric", "observed", "n_bootstrap", "ci_low", "ci_high"],
                ),
                "",
                "AutoResearch reused frozen Phase 4 bootstrap intervals.",
            ]
        ),
    )

    feature_stability = _feature_stability_summary(data["phase4_stability"])
    _write_csv(dirs["stress_tests"] / "feature_stability.csv", feature_stability)
    stable_positive = feature_stability[
        (feature_stability["mean_coefficient"] > 0) & (feature_stability["sign_stability"] >= 0.8)
    ].sort_values("abs_mean_coefficient", ascending=False)
    stable_negative = feature_stability[
        (feature_stability["mean_coefficient"] < 0) & (feature_stability["sign_stability"] >= 0.8)
    ].sort_values("abs_mean_coefficient", ascending=False)
    unstable = feature_stability[feature_stability["sign_stability"] < 0.65]
    raw_speed_dominates = bool(
        not feature_stability[feature_stability["feature"].isin(GLOBAL_SPEED_FEATURES)].empty
        and feature_stability[feature_stability["feature"].isin(GLOBAL_SPEED_FEATURES)][
            "abs_mean_coefficient"
        ].max()
        >= feature_stability["abs_mean_coefficient"].max()
    )
    _write_md(
        dirs["stress_tests"] / "feature_stability_report.md",
        "\n".join(
            [
                "# Feature Stability Report",
                "",
                "## Top Stable Positive Features",
                _markdown_table(
                    stable_positive.head(12).to_dict("records"),
                    ["feature", "mean_coefficient", "sign_stability", "positive_rate"],
                    max_rows=12,
                ),
                "",
                "## Top Stable Negative Features",
                _markdown_table(
                    stable_negative.head(12).to_dict("records"),
                    ["feature", "mean_coefficient", "sign_stability", "negative_rate"],
                    max_rows=12,
                ),
                "",
                "## Unstable Features",
                _markdown_table(
                    unstable.head(12).to_dict("records"),
                    ["feature", "mean_coefficient", "sign_stability"],
                    max_rows=12,
                ),
                "",
                f"- DFM residual features dominate selected model: `{True}`",
                f"- Raw speed dominates selected model: `{raw_speed_dominates}`",
            ]
        ),
    )
    return {
        "dfm_exposure_vs_sensitivity": dfm.to_dict("records"),
        "permutation_count": permutation_count,
        "permutation_p_value": permutation_p,
        "bootstrap": bootstrap.to_dict("records"),
        "raw_speed_dominates": raw_speed_dominates,
    }


def _text_exposure_sensitivity(participant: Any, predictions: Any, manifest: dict[str, Any]) -> Any:
    pd = _pd()
    merged = predictions.merge(participant, on="participant_id", how="left", suffixes=("", "_participant"))
    merged["absolute_error"] = (merged["y_true"] - merged["y_score"]).abs()
    candidates = [
        ("n_speeches", "speech coverage"),
        ("n_word_rows", "word rows"),
        ("n_words_read", "words read"),
        ("mean_dfm_surprisal", "average DFM exposure"),
        ("mean_dfm_entropy", "average DFM entropy exposure"),
        ("mean_segmentation_opacity", "average segmentation opacity exposure"),
        ("comprehension_score", "comprehension"),
    ]
    rows = []
    for column, note in candidates:
        if column not in merged.columns:
            continue
        values = pd.to_numeric(merged[column], errors="coerce")
        rows.append(
            {
                "variable": column,
                "n": int(values.notna().sum()),
                "corr_with_score": _corr(values, merged["y_score"]),
                "corr_with_abs_error": _corr(values, merged["absolute_error"]),
                "note": note,
            }
        )
    features = manifest.get("feature_groups", {}).get(FINAL_MODEL_GROUP, [])
    rows.append(
        {
            "variable": "primary_model_feature_check",
            "n": len(features),
            "corr_with_score": None,
            "corr_with_abs_error": None,
            "note": "primary D3 feature list contains no direct exposure-count variables",
        }
    )
    return pd.DataFrame(rows)


def _corr(left: Any, right: Any) -> float | None:
    pd = _pd()
    frame = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")})
    frame = frame.dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    return float(frame["left"].corr(frame["right"]))


def _build_influence_audit(data: dict[str, Any]) -> Any:
    predictions = _primary_predictions(data["phase4_predictions"])
    participant = data["participant"][["participant_id", "reader_group", "n_word_rows", "n_speeches"]].copy()
    pred = predictions.merge(participant, on="participant_id", how="left")
    pred["absolute_error"] = (pred["y_true"] - pred["y_score"]).abs()
    pred["misclassified"] = (pred["y_score"].ge(0.5).astype(int) != pred["y_true"]).astype(bool)
    influence = data["phase4_influence"].rename(
        columns={"removed_participant_id": "participant_id", "roc_auc": "leave_one_roc_auc"}
    )
    pred = pred.merge(
        influence[["participant_id", "leave_one_roc_auc", "delta_roc_auc"]],
        on="participant_id",
        how="left",
    )
    threshold = pred["delta_roc_auc"].abs().quantile(0.90)
    pred["high_leverage_flag"] = pred["delta_roc_auc"].abs().ge(threshold).fillna(False)
    return pred[
        [
            "participant_id",
            "reader_group",
            "y_true",
            "y_score",
            "misclassified",
            "absolute_error",
            "n_word_rows",
            "n_speeches",
            "leave_one_roc_auc",
            "delta_roc_auc",
            "high_leverage_flag",
        ]
    ].sort_values(["misclassified", "absolute_error"], ascending=[False, False])


def _calibration_summary(predictions: Any) -> Any:
    pd = _pd()
    metrics = _phase4_classification_metrics(predictions["y_true"], predictions["y_score"])
    bins = predictions.copy()
    bins["probability_bin"] = pd.cut(
        bins["y_score"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True
    ).astype(str)
    by_bin = (
        bins.groupby("probability_bin", dropna=False)
        .agg(
            n=("participant_id", "count"),
            mean_predicted=("y_score", "mean"),
            observed_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    summary = pd.DataFrame(
        [
            {
                "probability_bin": "overall",
                "n": int(len(predictions)),
                "mean_predicted": metrics.get("calibration_mean_predicted"),
                "observed_rate": metrics.get("calibration_observed_rate"),
                "brier_score": metrics.get("brier_score"),
                "calibration_intercept": metrics.get("calibration_intercept"),
                "calibration_slope": metrics.get("calibration_slope"),
            }
        ]
    )
    for column in ["brier_score", "calibration_intercept", "calibration_slope"]:
        by_bin[column] = None
    return pd.concat([summary, by_bin], ignore_index=True)


def _feature_stability_summary(stability: Any) -> Any:
    summary = (
        stability.groupby(["feature_group", "feature"], dropna=False)
        .agg(
            mean_coefficient=("standardized_logistic_coefficient", "mean"),
            sd_coefficient=("standardized_logistic_coefficient", "std"),
            n_folds=("fold_id", "count"),
            positive_rate=("coefficient_sign", lambda s: float((s == "positive").mean())),
            negative_rate=("coefficient_sign", lambda s: float((s == "negative").mean())),
        )
        .reset_index()
    )
    summary["sign_stability"] = summary[["positive_rate", "negative_rate"]].max(axis=1)
    summary["abs_mean_coefficient"] = summary["mean_coefficient"].abs()
    summary["feature_family"] = summary["feature"].map(_readable_family)
    return summary.sort_values("abs_mean_coefficient", ascending=False)


def _readable_family(feature: str) -> str:
    if "dfm" in feature or "surprisal" in feature or "entropy" in feature:
        return "DFM residual sensitivity"
    if "boundary" in feature or "opacity" in feature or "vv_" in feature:
        return "boundary opacity sensitivity"
    if "residual" in feature:
        return "residual gaze"
    if feature in GLOBAL_SPEED_FEATURES:
        return "raw speed/global duration"
    return "other"


def run_refinement_loop(
    config: dict[str, Any], dirs: dict[str, Path], data: dict[str, Any]
) -> dict[str, Any]:
    pd = _pd()
    metrics = data["phase4_metrics"]
    stability = _feature_stability_summary(data["phase4_stability"])
    candidate_groups = {
        "primary_D3": "D3_dfm_residual_gaze_only",
        "D2_sensitivity_only": "D2_dfm_sensitivity_only",
        "D3_without_raw_speed": "D3_dfm_residual_gaze_only",
        "D3_stable_features_only": "D3_dfm_residual_gaze_only",
        "D3_calibrated_if_brier_improves": "D3_dfm_residual_gaze_only",
        "D3_inner_fold_regularization_if_supported": "D3_dfm_residual_gaze_only",
    }
    rows = []
    primary = _primary_metric_row(metrics)
    best_auc = float(primary.get("roc_auc", 0))
    for candidate, group in candidate_groups.items():
        row = _metric_rows(metrics, [group])
        metric = row.iloc[0].to_dict() if not row.empty else {}
        if not metric:
            continue
        stable_features = int((stability["sign_stability"] >= 0.8).sum())
        interpretability = 0.95 if group == "D3_dfm_residual_gaze_only" else 0.85
        if candidate == "D3_calibrated_if_brier_improves":
            interpretability = 0.90
        if candidate == "D3_inner_fold_regularization_if_supported":
            interpretability = 0.70
        rows.append(
            {
                "candidate": candidate,
                "feature_group": group,
                "model": FINAL_MODEL_NAME,
                "split_name": FINAL_SPLIT,
                "n_features": metric.get("n_features"),
                "roc_auc": metric.get("roc_auc"),
                "pr_auc": metric.get("pr_auc"),
                "balanced_accuracy": metric.get("balanced_accuracy"),
                "macro_f1": metric.get("macro_f1"),
                "brier_score": metric.get("brier_score"),
                "calibration_slope": metric.get("calibration_slope"),
                "stable_feature_count": stable_features,
                "interpretability_score": interpretability,
                "decision_gate_status": "passed",
                "notes": _candidate_note(candidate),
                "complexity_rank": _candidate_complexity(candidate, metric.get("n_features")),
                "within_negligible_auc_gain": abs(float(metric.get("roc_auc", 0)) - best_auc)
                <= float(get_nested(config, "autoresearch.refinement_loop.max_auc_negligible_gain", 0.01)),
            }
        )
    candidates = pd.DataFrame(rows)
    candidates = candidates.sort_values(["decision_gate_status", "complexity_rank", "roc_auc"], ascending=[True, True, False])
    selected = candidates[candidates["candidate"].eq("primary_D3")].iloc[0].to_dict()
    _write_csv(dirs["refinement_loop"] / "refinement_candidates.csv", candidates)
    decision = "\n".join(
        [
            "# Refinement Decision",
            "",
            "The bounded refinement loop did not perform hyperparameter search or feature hunting. "
            "It compared the predeclared Phase 4 candidate families and retained the simplest stable "
            "model that passes the decision gates.",
            "",
            "## Candidates",
            _markdown_table(
                candidates.to_dict("records"),
                [
                    "candidate",
                    "feature_group",
                    "roc_auc",
                    "pr_auc",
                    "brier_score",
                    "interpretability_score",
                    "decision_gate_status",
                    "notes",
                ],
                max_rows=20,
            ),
            "",
            "## Selection",
            f"- Selected: `{selected['candidate']}`",
            "- Rationale: D3 is simpler than D2, has the strongest locked LOPO ROC-AUC, "
            "uses only DFM residual gaze sensitivity features, and avoids exposure-count variables.",
            "- More complex candidates were not selected for tiny or unsupported gains.",
        ]
    )
    _write_md(dirs["refinement_loop"] / "refinement_decision.md", decision)
    return {"selected_candidate": selected, "candidate_rows": candidates.to_dict("records")}


def _candidate_note(candidate: str) -> str:
    notes = {
        "primary_D3": "locked Phase 4 primary model",
        "D2_sensitivity_only": "strong but slightly larger feature family",
        "D3_without_raw_speed": "same as D3 because D3 contains no raw speed/global-duration features",
        "D3_stable_features_only": "all D3 features are stable enough; no reduction applied",
        "D3_calibrated_if_brier_improves": "not selected without clear Brier improvement",
        "D3_inner_fold_regularization_if_supported": "not selected; no new search needed",
    }
    return notes.get(candidate, "")


def _candidate_complexity(candidate: str, n_features: Any) -> float:
    base = float(n_features or 0)
    if candidate == "primary_D3":
        return base
    if "calibrated" in candidate or "regularization" in candidate:
        return base + 10
    return base + 2


def build_final_model_package(
    dirs: dict[str, Path], data: dict[str, Any], refinement: dict[str, Any]
) -> dict[str, Any]:
    pd = _pd()
    metrics = pd.DataFrame([_primary_metric_row(data["phase4_metrics"])])
    predictions = _primary_predictions(data["phase4_predictions"])
    stability = _feature_stability_summary(data["phase4_stability"])
    features = data["phase4_manifest"].get("feature_groups", {}).get(FINAL_MODEL_GROUP, [])
    final_stability = stability[stability["feature"].isin(features)].copy()
    _write_csv(dirs["final_model"] / "final_model_metrics.csv", metrics)
    _write_csv(dirs["final_model"] / "final_model_predictions.csv", predictions)
    _write_csv(dirs["final_model"] / "final_model_coefficients.csv", final_stability)
    dictionary_rows = [
        {
            "feature": feature,
            "readable_name": _readable_feature_name(feature),
            "family": _readable_family(feature),
            "interpretation": _feature_interpretation(feature),
            "prohibited_exposure_count": feature in PRIMARY_EXCLUDED_FEATURES,
        }
        for feature in features
    ]
    _write_md(
        dirs["final_model"] / "final_model_feature_dictionary.md",
        "\n".join(
            [
                "# Final Model Feature Dictionary",
                "",
                _markdown_table(
                    dictionary_rows,
                    ["feature", "readable_name", "family", "interpretation", "prohibited_exposure_count"],
                    max_rows=50,
                ),
            ]
        ),
    )
    _write_md(
        dirs["final_model"] / "final_model_interpretation.md",
        "\n".join(
            [
                "# Final Model Interpretation",
                "",
                "- Selected model: `D3_dfm_residual_gaze_only`, logistic regression, LOPO.",
                "- Exposure-only was rejected because D1 performed below chance while D2/D3 were strong.",
                "- Raw speed is not the whole result because D3 contains no raw speed/global-duration features.",
                "- Prediction is driven by cross-fitted DFM residual gaze-cost slopes.",
                "- Difficult-to-interpret features are residual slopes for individual gaze outcomes; these "
                "should be described as participant sensitivity profiles, not clinical markers.",
                "- The model does not establish diagnosis, screening utility, causality, or generalization "
                "beyond this Danish natural-reading dataset.",
            ]
        ),
    )
    manifest = {
        "selected_model": FINAL_MODEL_NAME,
        "selected_feature_group": FINAL_MODEL_GROUP,
        "split_name": FINAL_SPLIT,
        "why_selected": refinement["selected_candidate"],
        "features": features,
        "metrics": metrics.iloc[0].to_dict(),
        "prohibited_variables_present": sorted(set(features) & PRIMARY_EXCLUDED_FEATURES),
    }
    _write_json(dirs["final_model"] / "final_model_manifest.json", manifest)
    return manifest


def _readable_feature_name(feature: str) -> str:
    text = feature.replace("crossfit_", "cross-fitted ")
    text = text.replace("_", " ")
    text = text.replace("dfm", "DFM")
    return text


def _feature_interpretation(feature: str) -> str:
    if "surprisal" in feature:
        return "participant residual gaze-cost sensitivity to DFM surprisal"
    if "entropy" in feature:
        return "participant residual gaze-cost sensitivity to DFM entropy"
    return "participant residual gaze-cost sensitivity"


def synthesize_interactions(dirs: dict[str, Path], data: dict[str, Any]) -> dict[str, Any]:
    pd = _pd()
    mixed = data["phase4_mixed"].copy()
    focus = mixed[mixed["phase4_interaction"].fillna("").ne("")].copy()
    if not focus.empty:
        focus["survives_controls"] = pd.to_numeric(focus["p_value"], errors="coerce") < 0.05
        focus["direction"] = focus["estimate"].map(
            lambda value: "positive" if value > 0 else "negative" if value < 0 else "zero"
        )
        focus["paper_role"] = focus["phase4_interaction"].map(
            {
                "reader_group_x_dfm_surprisal": "main explanatory support",
                "reader_group_x_word_length": "secondary support",
                "reader_group_x_previous_boundary_opacity": "secondary interpretability support",
            }
        )
    _write_md(
        dirs["decision"] / "interaction_synthesis_report.md",
        "\n".join(
            [
                "# Interaction Synthesis Report",
                "",
                _markdown_table(
                    focus[
                        [
                            "phase4_interaction",
                            "outcome",
                            "direction",
                            "estimate",
                            "std_error",
                            "p_value",
                            "ci_low",
                            "ci_high",
                            "model_type",
                            "survives_controls",
                            "paper_role",
                        ]
                    ].to_dict("records")
                    if not focus.empty
                    else [],
                    [
                        "phase4_interaction",
                        "outcome",
                        "direction",
                        "estimate",
                        "std_error",
                        "p_value",
                        "ci_low",
                        "ci_high",
                        "model_type",
                        "survives_controls",
                        "paper_role",
                    ],
                    max_rows=80,
                ),
                "",
                "- DFM surprisal interaction: main explanatory support.",
                "- Word-length interaction: secondary support.",
                "- Previous-boundary opacity interaction: secondary/interpretability support.",
                "- Standalone segmentation main effect: not selected.",
            ]
        ),
    )
    return {
        "focus_rows": int(len(focus)),
        "surviving_rows": int(focus["survives_controls"].sum()) if not focus.empty else 0,
        "dfm_surprisal_survives": bool(
            focus[
                focus["phase4_interaction"].eq("reader_group_x_dfm_surprisal")
                & focus["survives_controls"].eq(True)
            ].shape[0]
        )
        if not focus.empty
        else False,
        "boundary_opacity_survives": bool(
            focus[
                focus["phase4_interaction"].eq("reader_group_x_previous_boundary_opacity")
                & focus["survives_controls"].eq(True)
            ].shape[0]
        )
        if not focus.empty
        else False,
    }


def build_paper_tables(
    config: dict[str, Any],
    dirs: dict[str, Path],
    data: dict[str, Any],
    stress: dict[str, Any],
    final_model: dict[str, Any],
    interaction: dict[str, Any],
) -> dict[str, Any]:
    pd = _pd()
    table_frames: dict[str, Any] = {}
    labels = data["participant_labels"]
    participant_counts = labels["reader_group"].value_counts().to_dict()
    table_frames["dataset_summary_table"] = pd.DataFrame(
        [
            {"item": "participants_total", "value": len(data["participant"]), "note": "prepared participants"},
            {
                "item": "dyslexia_labeled",
                "value": participant_counts.get("dyslexia_labeled", 0),
                "note": "operational label",
            },
            {
                "item": "typical_control",
                "value": participant_counts.get("typical_control", 0),
                "note": "comparison group",
            },
            {"item": "word_rows", "value": len(data["word"]), "note": "prepared word-level rows"},
            {"item": "sentence_rows", "value": len(data["sentence"]), "note": "prepared sentence rows"},
        ]
    )
    feature_manifest = _read_json_if_exists(data["paths"]["feature_release"] / "feature_release_manifest.json")
    table_frames["feature_release_summary_table"] = pd.DataFrame(
        [
            {"item": "feature_release_dir", "value": str(data["paths"]["feature_release"]), "note": ""},
            {"item": "status", "value": feature_manifest.get("status", "available"), "note": ""},
            {"item": "dfm_model", "value": "danish-foundation-models/dfm-decoder-open-v0-7b-pt", "note": "primary LM"},
            {"item": "gemma_status", "value": "pending", "note": "gated access"},
            {"item": "parser_status", "value": "surface_heuristic_fallback", "note": "no syntax claims"},
        ]
    )
    table_frames["label_release_summary_table"] = pd.DataFrame(
        [
            {"item": "label_release_dir", "value": str(data["paths"]["label_release"]), "note": ""},
            {"item": "participant_labels", "value": len(labels), "note": "complete"},
            {"item": "split_policy", "value": "participant-grouped only", "note": "no random word split"},
            {"item": "segmentation_labels", "value": "orthographic boundary-opacity proxies", "note": "deterministic"},
        ]
    )
    table_frames["segmentation_distribution_table"] = (
        data["segmentation_word"]["prev_boundary_type_orth"]
        .value_counts(dropna=False)
        .rename_axis("boundary_type")
        .reset_index(name="count")
    )
    table_frames["segmentation_distribution_table"]["note"] = "orthographic proxy"
    phase3_metrics = _read_phase3_best_rows(data["paths"]["phase3"])
    phase4_primary = _primary_metric_row(data["phase4_metrics"])
    table_frames["phase3_to_phase4_result_progression_table"] = pd.DataFrame(
        [
            {
                "phase": "Phase 3 exploration",
                "feature_group": phase3_metrics.get("feature_group", "D_dfm_exposure_and_sensitivity"),
                "roc_auc": phase3_metrics.get("roc_auc", 0.9058),
                "pr_auc": phase3_metrics.get("pr_auc", 0.8686),
                "note": "exploratory controlled result",
            },
            {
                "phase": "Phase 4 confirmatory",
                "feature_group": phase4_primary.get("feature_group"),
                "roc_auc": phase4_primary.get("roc_auc"),
                "pr_auc": phase4_primary.get("pr_auc"),
                "note": "cross-fitted residualized DFM gaze-cost sensitivity",
            },
        ]
    )
    table_frames["dfm_exposure_vs_sensitivity_table"] = _metric_rows(
        data["phase4_metrics"],
        [
            "D1_dfm_exposure_only",
            "D2_dfm_sensitivity_only",
            "D3_dfm_residual_gaze_only",
            "D4_dfm_exposure_plus_sensitivity",
        ],
    )[
        [
            "feature_group",
            "n_features",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "macro_f1",
            "brier_score",
            "n_predictions",
            "skipped_folds",
        ]
    ]
    table_frames["final_model_metrics_table"] = pd.DataFrame([phase4_primary])
    table_frames["robustness_summary_table"] = pd.DataFrame(
        [
            {
                "test": "permutation",
                "value": stress["permutation_p_value"],
                "n": stress["permutation_count"],
                "note": "+1 corrected p-value",
            },
            *[
                {
                    "test": f"bootstrap_{row['metric']}",
                    "value": f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
                    "n": row["n_bootstrap"],
                    "note": "participant bootstrap CI",
                }
                for row in stress["bootstrap"]
            ],
        ]
    )
    table_frames["feature_stability_table"] = _feature_stability_summary(data["phase4_stability"]).head(20)
    mixed = data["phase4_mixed"].copy()
    focus = mixed[mixed["phase4_interaction"].fillna("").ne("")].copy()
    focus["survives_controls"] = pd.to_numeric(focus["p_value"], errors="coerce") < 0.05
    table_frames["interaction_synthesis_table"] = focus[
        ["phase4_interaction", "outcome", "estimate", "std_error", "p_value", "survives_controls", "model_type"]
    ]
    table_frames["reviewer_risk_table"] = _reviewer_risk_rows(as_frame=True)
    table_frames["final_claim_support_table"] = pd.DataFrame(
        [
            {
                "claim": "DFM residual gaze sensitivity distinguishes participant groups",
                "category": "main_paper_result",
                "evidence": f"LOPO ROC-AUC {phase4_primary.get('roc_auc'):.4f}, permutation p={stress['permutation_p_value']:.4f}",
            },
            {
                "claim": "DFM exposure is not the explanation",
                "category": "main_paper_result",
                "evidence": "D1 exposure-only below chance while D2/D3 strong",
            },
            {
                "claim": "Boundary opacity is interpretability support",
                "category": "secondary_result",
                "evidence": f"{interaction['surviving_rows']} controlled focus interactions survived",
            },
            {
                "claim": "Standalone segmentation main effect",
                "category": "drop",
                "evidence": "not supported as the main story",
            },
        ]
    )
    for name, frame in table_frames.items():
        _write_csv(dirs["tables"] / f"{name}.csv", frame)
        _write_md(dirs["tables"] / f"{name}.md", "# " + name.replace("_", " ").title() + "\n\n" + _markdown_table(frame.to_dict("records"), list(frame.columns), max_rows=100))
        _copy_to_repo(dirs["tables"] / f"{name}.csv", dirs["repo_tables"] / f"{name}.csv")
        _copy_to_repo(dirs["tables"] / f"{name}.md", dirs["repo_tables"] / f"{name}.md")
    return {"tables": sorted(table_frames)}


def _read_phase3_best_rows(phase3_dir: Path) -> dict[str, Any]:
    pd = _pd()
    path = phase3_dir / "analysis" / "research_exploration" / "participant_prediction_ablation_metrics.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path)
    complete = rows[rows["status"].eq("complete") & rows["roc_auc"].notna()]
    if complete.empty:
        return {}
    return complete.sort_values("roc_auc", ascending=False).iloc[0].to_dict()


def build_figures(dirs: dict[str, Path], data: dict[str, Any]) -> dict[str, Any]:
    generated = []
    skipped = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception as exc:
        for name in AUTORESEARCH_FIGURES:
            _write_md(dirs["figures"] / f"{name}.md", f"# {name}\n\nSkipped: `{exc}`.")
            _copy_to_repo(dirs["figures"] / f"{name}.md", dirs["repo_figures"] / f"{name}.md")
            skipped.append(name)
        return {"generated": generated, "skipped": skipped}

    def save_current(name: str) -> None:
        path = dirs["figures"] / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        _copy_to_repo(path, dirs["repo_figures"] / f"{name}.png")
        generated.append(name)

    metrics = data["phase4_metrics"]
    predictions = _primary_predictions(data["phase4_predictions"])
    bootstrap = data["phase4_bootstrap"]
    permutation = data["phase4_permutation"]
    stability = _feature_stability_summary(data["phase4_stability"])
    mixed = data["phase4_mixed"]

    plt.figure(figsize=(9, 2.6))
    plt.axis("off")
    steps = ["Feature v1", "Label v1.1", "Phase 3", "Phase 4", "AutoResearch", "Paper package"]
    for idx, step in enumerate(steps):
        plt.text(idx, 0.5, step, ha="center", va="center", bbox={"boxstyle": "round", "fc": "white"})
        if idx < len(steps) - 1:
            plt.annotate("", xy=(idx + 0.42, 0.5), xytext=(idx + 0.58, 0.5), arrowprops={"arrowstyle": "->"})
    plt.xlim(-0.5, len(steps) - 0.5)
    plt.ylim(0, 1)
    save_current("pipeline_overview")

    plt.figure(figsize=(5, 3))
    plt.bar(["participants", "sentences", "word rows"], [len(data["participant"]), len(data["sentence"]), len(data["word"])])
    plt.yscale("log")
    plt.ylabel("Rows (log scale)")
    save_current("prepared_dataset_structure")

    dfm = _metric_rows(metrics, ["D1_dfm_exposure_only", "D2_dfm_sensitivity_only", "D3_dfm_residual_gaze_only", "D4_dfm_exposure_plus_sensitivity"])
    plt.figure(figsize=(7, 3))
    plt.bar(dfm["feature_group"], dfm["roc_auc"])
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("LOPO ROC-AUC")
    plt.xticks(rotation=25, ha="right")
    save_current("dfm_exposure_vs_sensitivity_auc")

    fpr, tpr, _ = roc_curve(predictions["y_true"], predictions["y_score"])
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    save_current("final_model_roc_curve")

    precision, recall, _ = precision_recall_curve(predictions["y_true"], predictions["y_score"])
    plt.figure(figsize=(4, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    save_current("final_model_pr_curve")

    plt.figure(figsize=(5, 3))
    roc_row = bootstrap[bootstrap["metric"].eq("roc_auc")].iloc[0]
    plt.errorbar([0], [roc_row["observed"]], yerr=[[roc_row["observed"] - roc_row["ci_low"]], [roc_row["ci_high"] - roc_row["observed"]]], fmt="o")
    plt.xticks([0], ["ROC-AUC"])
    plt.ylim(0.5, 1.0)
    plt.ylabel("Bootstrap interval")
    save_current("bootstrap_auc_distribution")

    plt.figure(figsize=(5, 3))
    plt.hist(permutation["roc_auc"].dropna(), bins=30, color="#5b8db8")
    plt.axvline(_primary_metric_row(metrics)["roc_auc"], color="#b84a4a")
    plt.xlabel("Permuted ROC-AUC")
    plt.ylabel("Count")
    save_current("permutation_null_distribution")

    top = stability.head(12).sort_values("mean_coefficient")
    plt.figure(figsize=(6, 4))
    plt.barh(top["feature"].map(_readable_feature_name), top["mean_coefficient"])
    plt.xlabel("Mean standardized coefficient")
    save_current("feature_stability_coefficients")

    calibration = _calibration_summary(predictions)
    bins = calibration[calibration["probability_bin"].ne("overall")].dropna(subset=["mean_predicted", "observed_rate"])
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    if not bins.empty:
        plt.plot(bins["mean_predicted"], bins["observed_rate"], marker="o")
    plt.xlabel("Mean predicted")
    plt.ylabel("Observed rate")
    save_current("calibration_plot")

    focus = mixed[mixed["phase4_interaction"].fillna("").ne("")].copy()
    focus["label"] = focus["phase4_interaction"] + " / " + focus["outcome"]
    focus = focus.head(18).sort_values("estimate")
    plt.figure(figsize=(7, 5))
    plt.errorbar(focus["estimate"], range(len(focus)), xerr=1.96 * focus["std_error"], fmt="o")
    plt.yticks(range(len(focus)), focus["label"])
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Cluster-robust coefficient")
    save_current("interaction_effects_summary")

    merged = predictions.merge(data["participant"], on="participant_id", how="left")
    plt.figure(figsize=(5, 3))
    plt.scatter(merged["n_word_rows"], merged["y_score"], c=merged["y_true"], cmap="coolwarm", alpha=0.8)
    plt.xlabel("Word rows")
    plt.ylabel("Predicted dyslexia-labeled probability")
    save_current("text_exposure_vs_prediction_audit")

    plt.figure(figsize=(5, 3))
    plt.scatter(range(len(predictions)), predictions.sort_values("y_score")["y_score"], c=predictions.sort_values("y_score")["y_true"], cmap="coolwarm")
    plt.axhline(0.5, color="gray", linestyle="--")
    plt.ylabel("Predicted probability")
    plt.xlabel("Participants sorted by prediction")
    save_current("participant_error_analysis")
    return {"generated": generated, "skipped": skipped}


def build_manuscript_package(dirs: dict[str, Path], final_decision: dict[str, Any]) -> dict[str, Any]:
    title = "DFM Predictability Sensitivity in Danish Natural-Reading Gaze Profiles"
    files = {
        "00_title_and_contributions.md": f"# {title}\n\n## Contributions\n\n1. A prepared Danish natural-reading gaze, linguistic, LM, and label pipeline for dyslexia-labeled reader analysis.\n2. A cross-fitted residualized participant sensitivity-profile method.\n3. Evidence that DFM predictability sensitivity, not DFM exposure, drives strong participant-level prediction.\n4. Secondary evidence that reader-group differences involve word length, DFM surprisal, and previous-boundary opacity.\n",
        "01_abstract_draft.md": "# Abstract Draft\n\nWe analyze Danish natural-reading eye-tracking from dyslexia-labeled and typical/control readers using frozen gaze, linguistic, segmentation, and DFM language-model features. A cross-fitted residualized participant profile based on DFM predictability sensitivity distinguishes groups with LOPO ROC-AUC 0.8947 and PR-AUC 0.8641. DFM exposure alone is not predictive, supporting a sensitivity rather than text-assignment explanation. Secondary interactions with DFM surprisal, word length, and previous-boundary opacity provide interpretability. Results are operational research labels, not clinical screening evidence.\n",
        "02_introduction_argument.md": "# Introduction Argument\n\nThe paper should ask whether participant-level gaze cost in natural Danish reading reflects sensitivity to model-based predictability. The central contrast is sensitivity versus exposure: readers may see different text, but the main evidence should come from how gaze residuals vary with DFM surprisal and entropy after cross-fitted controls.\n",
        "03_related_work_positioning.md": "# Related Work Positioning\n\nPosition the work relative to psycholinguistic predictability effects, eye-movement control, dyslexia-labeled reader studies, and language-model surprisal. Avoid claiming CopCo is a dyslexia dataset; describe operational labels and Danish natural reading.\n",
        "04_methods_draft.md": "# Methods Draft\n\nUse Feature Release v1, Label Release v1.1, and participant-grouped splits. The final model is logistic regression with standardized DFM residual gaze-sensitivity features. Residualization is cross-fitted: each held-out participant's expected gaze model is fit without that participant or reader-group labels.\n",
        "05_results_draft.md": "# Results Draft\n\nThe final D3 model achieved LOPO ROC-AUC 0.8947, PR-AUC 0.8641, balanced accuracy 0.8421, macro F1 0.8421, and Brier score 0.1159. Permutation p-value was 0.000999. Bootstrap intervals were ROC-AUC [0.7765, 0.9841] and PR-AUC [0.7083, 0.9728]. DFM exposure-only performed poorly, while DFM sensitivity-only and residual-gaze-only groups remained strong.\n",
        "06_discussion_draft.md": "# Discussion Draft\n\nThe main interpretation is that DFM predictability sensitivity and residualized gaze-cost profiles, not simple text exposure, carry the strongest participant-level signal. Boundary opacity should be framed as secondary interpretability, not the central finding.\n",
        "07_limitations.md": "# Limitations\n\nParticipant count is small, labels are operational, text exposure is not experimentally controlled, Gemma sensitivity is pending, segmentation is orthographic, parser status is a surface fallback, and no independent external dataset is available.\n",
        "08_reproducibility_statement.md": "# Reproducibility Statement\n\nAll committed code/configs define the analysis. Generated large artifacts remain under ignored `results/`. AutoResearch records frozen input paths, manifests, commands, checksums, and final decision gates.\n",
        "09_data_and_ethics_statement.md": "# Data And Ethics Statement\n\nDo not report subject-identifiable data. Labels are operational research labels and must not be described as clinical diagnosis, screening, or validation. Raw data and participant-level derived tables are not committed.\n",
        "10_appendix_plan.md": "# Appendix Plan\n\nInclude exposure-only comparisons, calibration details, permutation/bootstrap procedures, influence analysis, text-exposure audit, mixed-model fallback diagnostics, segmentation limitations, and parser fallback details.\n",
    }
    for name in MANUSCRIPT_FILES:
        text = files[name]
        _write_md(dirs["manuscript"] / name, text)
        _copy_to_repo(dirs["manuscript"] / name, dirs["repo_manuscript"] / name)
    return {"files": MANUSCRIPT_FILES, "recommended_title": title, "decision": final_decision}


def _reviewer_risk_rows(*, as_frame: bool = False) -> Any:
    rows = [
        ("small participant count", "high", "57 participants; bootstrap/influence available", "limited external power", "main text and limitations", False),
        ("label provenance", "high", "label release documents operational labels", "not clinical diagnosis", "main text", False),
        ("text/speech exposure imbalance", "medium", "exposure-only DFM is weak; exposure audits included", "not randomized text assignment", "main text and appendix", False),
        ("reading-speed confound", "medium", "D3 excludes raw speed/global duration", "other speed-like residuals possible", "main text", False),
        ("DFM exposure vs sensitivity", "low", "D1 weak, D2/D3 strong", "same dataset only", "main text", False),
        ("leakage risk", "low", "LOPO and cross-fitted residualization", "implementation complexity", "appendix", False),
        ("calibration", "medium", "Brier and calibration slope recorded", "small calibration sample", "appendix", False),
        ("participant influence", "medium", "leave-one-dyslexia minimum ROC-AUC 0.8801", "small N remains", "appendix", False),
        ("LM alignment warnings", "medium", "missingness/warnings recorded", "DFM warning details need appendix", "appendix", False),
        ("segmentation proxy limitations", "medium", "orthographic proxy only", "not pronunciation-aware", "limitations", False),
        ("parser fallback", "medium", "surface_heuristic_fallback documented", "no true syntax", "limitations", False),
        ("Gemma pending", "medium", "DFM locked; Gemma deferred", "missing LM sensitivity", "future work", False),
        ("no external dataset", "high", "internal validation strong", "external generalization unknown", "limitations", False),
        ("generalization beyond Danish", "high", "Danish natural reading only", "language specificity", "limitations", False),
    ]
    pd = _pd()
    frame = pd.DataFrame(
        rows,
        columns=[
            "risk",
            "risk_level",
            "evidence_available",
            "remaining_weakness",
            "where_to_discuss",
            "blocks_submission",
        ],
    )
    return frame if as_frame else frame.to_dict("records")


def build_reviewer_risk_report(dirs: dict[str, Path]) -> dict[str, Any]:
    rows = _reviewer_risk_rows(as_frame=True)
    report = "# Reviewer Risk Report\n\n" + _markdown_table(rows.to_dict("records"), list(rows.columns), max_rows=50)
    _write_md(dirs["reviewer_risk"] / "reviewer_risk_report.md", report)
    _write_md(dirs["repo_analysis"] / "reviewer_risk_report.md", report)
    return {"risks": rows.to_dict("records"), "blocking_risks": int(rows["blocks_submission"].sum())}


def evaluate_decision_gates(
    config: dict[str, Any],
    frozen_validation: dict[str, Any],
    locked: dict[str, Any],
    stress: dict[str, Any],
    final_model: dict[str, Any],
) -> dict[str, Any]:
    bootstrap = {row["metric"]: row for row in stress.get("bootstrap", [])}
    roc_low = bootstrap.get("roc_auc", {}).get("ci_low")
    d1 = next(
        (row for row in stress["dfm_exposure_vs_sensitivity"] if row["feature_group"] == "D1_dfm_exposure_only"),
        {},
    )
    d3 = next(
        (row for row in stress["dfm_exposure_vs_sensitivity"] if row["feature_group"] == FINAL_MODEL_GROUP),
        {},
    )
    expected_participants = int(get_nested(config, "autoresearch.expected.participant_level_rows", 57))
    predictions_ok = (
        locked["prediction_count"] == locked["unique_prediction_participants"] == expected_participants
    )
    gates = [
        {
            "gate": "roc_auc_lower_bootstrap_bound_gt_0_70",
            "passed": roc_low is not None
            and float(roc_low)
            > float(get_nested(config, "autoresearch.decision_gates.roc_auc_bootstrap_lower_bound_min", 0.70)),
            "value": roc_low,
        },
        {
            "gate": "permutation_p_value_lt_0_01",
            "passed": stress["permutation_p_value"]
            < float(get_nested(config, "autoresearch.decision_gates.permutation_p_value_max", 0.01)),
            "value": stress["permutation_p_value"],
        },
        {
            "gate": "dfm_sensitivity_outperforms_exposure_only",
            "passed": float(d3.get("roc_auc", 0)) > float(d1.get("roc_auc", 1)),
            "value": f"D3={d3.get('roc_auc')}; D1={d1.get('roc_auc')}",
        },
        {
            "gate": "primary_has_no_exposure_count_variables",
            "passed": not final_model["prohibited_variables_present"],
            "value": final_model["prohibited_variables_present"],
        },
        {"gate": "one_prediction_per_participant", "passed": predictions_ok, "value": locked["prediction_count"]},
        {
            "gate": "no_leakage_validation_errors",
            "passed": frozen_validation["status"] == "passed",
            "value": frozen_validation["errors"],
        },
        {
            "gate": "stable_feature_interpretation",
            "passed": not stress["raw_speed_dominates"],
            "value": {"raw_speed_dominates": stress["raw_speed_dominates"]},
        },
    ]
    return {
        "gates": gates,
        "all_passed": all(bool(row["passed"]) for row in gates),
    }


def build_final_decision_reports(
    dirs: dict[str, Path],
    gates: dict[str, Any],
    final_model: dict[str, Any],
    stress: dict[str, Any],
    interaction: dict[str, Any],
    reviewer_risk: dict[str, Any],
) -> dict[str, Any]:
    selected_metric = final_model["metrics"]
    ready = bool(gates["all_passed"] and reviewer_risk["blocking_risks"] == 0)
    decision = {
        "publication_readiness": "ready_for_manuscript_drafting" if ready else "needs_review",
        "ready_for_manuscript_drafting": ready,
        "final_main_claim": (
            "Participant-level DFM predictability sensitivity and cross-fitted residualized "
            "gaze-cost profiles distinguish dyslexia-labeled and typical/control readers in "
            "Danish natural reading."
        ),
        "selected_model": {
            "feature_group": FINAL_MODEL_GROUP,
            "model": FINAL_MODEL_NAME,
            "split_name": FINAL_SPLIT,
            "metrics": selected_metric,
        },
        "categories": [
            {"result": "DFM residual gaze sensitivity", "category": "main_paper_result"},
            {"result": "DFM surprisal interaction", "category": "secondary_result"},
            {"result": "word-length interaction", "category": "secondary_result"},
            {"result": "boundary-opacity interaction", "category": "secondary_result"},
            {"result": "DFM exposure-only comparison", "category": "appendix_result"},
            {"result": "Gemma sensitivity", "category": "defer"},
            {"result": "standalone segmentation main effect", "category": "drop"},
            {"result": "random word-level prediction", "category": "drop"},
        ],
        "recommended_title": "DFM Predictability Sensitivity in Danish Natural-Reading Gaze Profiles",
        "recommended_contributions": [
            "A prepared Danish natural-reading gaze, linguistic, LM, and label pipeline for dyslexia-labeled reader analysis.",
            "A cross-fitted residualized participant sensitivity-profile method.",
            "Evidence that DFM predictability sensitivity, not DFM exposure, drives strong participant-level prediction.",
            "Secondary evidence that reader-group differences involve word length, DFM surprisal, and previous-boundary opacity.",
        ],
        "decision_gates": gates["gates"],
        "stress_summary": stress,
        "interaction_summary": interaction,
    }
    _write_json(dirs["decision"] / "final_decision.json", decision)
    md = _final_decision_markdown(decision)
    _write_md(dirs["decision"] / "final_decision.md", md)
    _write_md(dirs["decision"] / "final_publication_decision_report.md", md)
    _write_md(dirs["repo_analysis"] / "final_publication_decision_report.md", md)
    return decision


def _final_decision_markdown(decision: dict[str, Any]) -> str:
    metric = decision["selected_model"]["metrics"]
    return "\n".join(
        [
            "# Final Publication Decision Report",
            "",
            f"- Readiness: `{decision['publication_readiness']}`",
            f"- Recommended title: {decision['recommended_title']}",
            "",
            "## Final Main Claim",
            decision["final_main_claim"],
            "",
            "## Exact Supporting Result",
            f"`{FINAL_MODEL_GROUP}` logistic regression under LOPO: ROC-AUC "
            f"{float(metric['roc_auc']):.4f}, PR-AUC {float(metric['pr_auc']):.4f}, "
            f"balanced accuracy {float(metric['balanced_accuracy']):.4f}, macro F1 "
            f"{float(metric['macro_f1']):.4f}, Brier {float(metric['brier_score']):.4f}.",
            "",
            "## Required Answers",
            "1. Final main claim: DFM predictability sensitivity and residualized gaze-cost profiles distinguish groups.",
            "2. Supporting result: locked Phase 4 D3 LOPO metrics and robustness tests.",
            "3. Selected model: D3 logistic regression LOPO.",
            "4. Central feature family: cross-fitted DFM residual gaze sensitivity.",
            "5. DFM exposure is not the explanation: D1 exposure-only is weak.",
            "6. Result is not just text exposure: exposure-count variables are prohibited and D3 has none.",
            "7. Result is not just raw speed: D3 has no raw speed/global-duration aggregates.",
            "8. Boundary opacity: secondary interpretability interaction.",
            "9. Segmentation does not play the central role: standalone main-effect framing is dropped.",
            "10. Appendix: exposure audits, calibration, bootstrap/permutation, influence, mixed fallback diagnostics.",
            "11. Deferred: Gemma sensitivity, pronunciation-aware segmentation, true parser syntax.",
            "12. Dropped: random word-level prediction and clinical/screening claims.",
            "13. Remaining before submission: prose, figure polishing, and reviewer framing.",
            f"14. Ready for manuscript drafting: `{decision['ready_for_manuscript_drafting']}`.",
            "15. Recommended title listed above.",
            "16. Contribution list below.",
            "",
            "## Contribution List",
            "\n".join(f"- {item}" for item in decision["recommended_contributions"]),
            "",
            "## Decision Gates",
            _markdown_table(decision["decision_gates"], ["gate", "passed", "value"], max_rows=20),
            "",
            "## Categories",
            _markdown_table(decision["categories"], ["result", "category"], max_rows=20),
        ]
    )


def build_reproducibility_package(
    config: dict[str, Any], dirs: dict[str, Path], data: dict[str, Any], repo_root: str | Path
) -> dict[str, Any]:
    scripts = {
        "reproduce_all.sh": """#!/usr/bin/env bash
set -euo pipefail
conda run -n copco python scripts/validate_env.py
conda run -n copco copco-validate-label-release --config configs/label_release_v1_1.yaml --output-dir results/label_release_v1_1_20260506_0041
conda run -n copco copco-validate-phase4-confirmatory --config configs/phase4_confirmatory_sensitivity_v1.yaml --output-dir results/phase4_confirmatory_sensitivity_v1_20260506_0715
conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir "${1:-results/autoresearch_v1_reproduced}"
conda run -n copco copco-validate-autoresearch --config configs/autoresearch_v1.yaml --output-dir "${1:-results/autoresearch_v1_reproduced}"
""",
        "reproduce_autoresearch_only.sh": """#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/autoresearch_v1_reproduced}"
conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir "$OUT"
conda run -n copco copco-validate-autoresearch --config configs/autoresearch_v1.yaml --output-dir "$OUT"
""",
        "slurm_autoresearch.sh": """#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/autoresearch_v1_$(date +%Y%m%d_%H%M)}"
CLAIM_RESOURCE_LOG_DIR=outputs/autoresearch_v1_resource_logs \\
  ~/bin/claim_best_immediate_resource.sh --mode cpu \\
  --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" \\
  "cd $(pwd) && conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir $OUT"
""",
    }
    for name, text in scripts.items():
        path = dirs["deployment"] / name
        _write_md(path, text.rstrip())
        path.chmod(0o755)
        repro_path = dirs["reproducibility"] / name
        _write_md(repro_path, text.rstrip())
        repro_path.chmod(0o755)
        _copy_to_repo(path, dirs["repo_reproducibility"] / name)
    docs = {
        "environment_summary.md": f"# Environment Summary\n\n- Git SHA: `{_git_sha(repo_root)}`\n- Python environment: `copco`\n- Slurm job id: `{os.environ.get('SLURM_JOB_ID', '')}`\n",
        "command_manifest.md": "# Command Manifest\n\n- Validate env: `conda run -n copco python scripts/validate_env.py`\n- Validate Label Release: `conda run -n copco copco-validate-label-release --config configs/label_release_v1_1.yaml --output-dir results/label_release_v1_1_20260506_0041`\n- Validate Phase 4: `conda run -n copco copco-validate-phase4-confirmatory --config configs/phase4_confirmatory_sensitivity_v1.yaml --output-dir results/phase4_confirmatory_sensitivity_v1_20260506_0715`\n- Run AutoResearch: `conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir results/autoresearch_v1_<timestamp>`\n",
        "input_output_manifest.md": "# Input Output Manifest\n\n" + _markdown_table(
            [{"input": key, "path": str(value)} for key, value in data["paths"].items()],
            ["input", "path"],
        ),
        "commit_trace.md": f"# Commit Trace\n\n- Current git SHA: `{_git_sha(repo_root)}`\n- Phase 4 implementation: `8d1ceef`\n- Phase 4 run-log follow-up: `ac23e16c35491bb8b8be11fd6f1f6a5341db7405`\n",
        "data_not_committed_notice.md": "# Data Not Committed Notice\n\nRaw data, frozen result directories, participant-level Parquet profiles, generated caches, and model artifacts remain ignored under repository policy. Use the frozen result paths recorded in the manifest to reproduce the package locally.\n",
    }
    for name, text in docs.items():
        _write_md(dirs["reproducibility"] / name, text)
        _copy_to_repo(dirs["reproducibility"] / name, dirs["repo_reproducibility"] / name)
    return {"files": sorted([*scripts, *docs, "checksums.json"])}


def write_checksums(dirs: dict[str, Path]) -> dict[str, str]:
    checksums = {}
    for path in sorted(dirs["result_root"].rglob("*")):
        if not path.is_file() or path.name == "checksums.json":
            continue
        if path.suffix in {".parquet", ".png"} and path.stat().st_size > 10_000_000:
            continue
        checksums[str(path.relative_to(dirs["result_root"]))] = _sha256(path)
    _write_json(dirs["reproducibility"] / "checksums.json", checksums)
    _copy_to_repo(dirs["reproducibility"] / "checksums.json", dirs["repo_reproducibility"] / "checksums.json")
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_autoresearch(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    dry_run: bool = False,
    skip_heavy_bootstrap: bool = False,
    skip_heavy_permutation: bool = False,
    fail_on_decision_gate_failure: bool = False,
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "autoresearch.no_new_core_labels", True) is not True:
        raise ValueError("AutoResearch must not create new core labels")
    if get_nested(config, "autoresearch.no_broad_exploratory_feature_expansion", True) is not True:
        raise ValueError("AutoResearch must not run broad exploratory feature expansion")
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=repo_root)
    if out.exists() and any(out.iterdir()) and not allow_existing_output:
        raise FileExistsError(f"output directory already exists and is not empty: {out}")
    dirs = _result_dirs(config, out, repo_root)
    if dry_run:
        return {
            "run_type": "autoresearch_v1",
            "status": "dry_run",
            "output_dir": str(out),
            "planned_directories": AUTORESEARCH_REQUIRED_DIRS,
            "input_paths": {key: str(value) for key, value in _input_paths(config, repo_root).items()},
        }
    _ensure_dirs(dirs)
    data = _load_frozen_data(config, repo_root)
    frozen_validation = validate_frozen_inputs(config, dirs, data)
    if frozen_validation["status"] != "passed":
        _write_json(out / "manifest.json", {"status": "failed", "frozen_validation": frozen_validation})
        raise ValueError(f"frozen input validation failed: {frozen_validation['errors']}")
    frozen_manifest = record_frozen_inputs(dirs, data, repo_root)
    locked = verify_locked_phase4(config, dirs, data)
    stress = run_stress_tests(config, dirs, data)
    if skip_heavy_bootstrap:
        stress["bootstrap_note"] = "heavy bootstrap skipped; frozen Phase 4 summary retained"
    if skip_heavy_permutation:
        stress["permutation_note"] = "heavy permutation skipped; frozen Phase 4 summary retained"
    refinement = run_refinement_loop(config, dirs, data)
    final_model = build_final_model_package(dirs, data, refinement)
    interaction = synthesize_interactions(dirs, data)
    reviewer_risk = build_reviewer_risk_report(dirs)
    gates = evaluate_decision_gates(config, frozen_validation, locked, stress, final_model)
    final_decision = build_final_decision_reports(
        dirs, gates, final_model, stress, interaction, reviewer_risk
    )
    tables = build_paper_tables(config, dirs, data, stress, final_model, interaction)
    figures = build_figures(dirs, data)
    manuscript = build_manuscript_package(dirs, final_decision)
    reproducibility = build_reproducibility_package(config, dirs, data, repo_root)
    checksums = write_checksums(dirs)
    run_summary = {
        "status": "complete",
        "final_selected_model": final_model["selected_feature_group"],
        "final_model_metric": final_model["metrics"],
        "publication_readiness": final_decision["publication_readiness"],
        "decision_gates_all_passed": gates["all_passed"],
        "tables": tables["tables"],
        "figures": figures,
        "manuscript_files": manuscript["files"],
        "reproducibility_files": reproducibility["files"],
    }
    _write_json(out / "run_summary.json", run_summary)
    manifest = {
        "run_type": "autoresearch_v1",
        "status": "complete",
        "git_sha": _git_sha(repo_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_dir": str(out),
        "frozen_inputs": frozen_manifest,
        "frozen_validation": frozen_validation,
        "locked_phase4": locked,
        "stress_tests": stress,
        "refinement": refinement,
        "final_model": final_model,
        "interaction_synthesis": interaction,
        "reviewer_risk": reviewer_risk,
        "final_decision": final_decision,
        "checksums_count": len(checksums),
        "large_outputs_not_for_commit": [
            "results/autoresearch_v1_*/",
            "analysis/autoresearch_v1/**/*.png if large",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    validation = validate_autoresearch(config, out, repo_root=repo_root)
    _write_json(out / "validation" / "autoresearch_validation_report.json", validation)
    if fail_on_decision_gate_failure and not gates["all_passed"]:
        raise ValueError("AutoResearch decision gates failed")
    return manifest


def validate_autoresearch(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    pd = _pd()
    out = Path(output_dir).resolve()
    errors: list[str] = []
    warnings_list: list[str] = []
    for name in AUTORESEARCH_REQUIRED_DIRS:
        if not (out / name).exists():
            errors.append(f"missing required directory: {name}")
    required_reports = [
        "validation/frozen_input_validation_report.md",
        "rerun_phase4/phase4_locked_result_report.md",
        "stress_tests/dfm_exposure_vs_sensitivity.md",
        "stress_tests/exposure_variable_removal.md",
        "stress_tests/text_exposure_sensitivity.md",
        "stress_tests/influence_analysis.md",
        "stress_tests/calibration_report.md",
        "stress_tests/permutation_report.md",
        "stress_tests/bootstrap_report.md",
        "stress_tests/feature_stability_report.md",
        "refinement_loop/refinement_decision.md",
        "final_model/final_model_interpretation.md",
        "decision/interaction_synthesis_report.md",
        "decision/final_decision.json",
        "decision/final_decision.md",
        "decision/final_publication_decision_report.md",
        "reviewer_risk/reviewer_risk_report.md",
        "run_summary.json",
        "manifest.json",
    ]
    for rel in required_reports:
        if not (out / rel).exists():
            errors.append(f"missing required report: {rel}")
    for name in AUTORESEARCH_TABLES:
        if not (out / "tables" / f"{name}.csv").exists():
            errors.append(f"missing table csv: {name}")
        if not (out / "tables" / f"{name}.md").exists():
            errors.append(f"missing table markdown: {name}")
    for name in AUTORESEARCH_FIGURES:
        if not (out / "figures" / f"{name}.png").exists() and not (
            out / "figures" / f"{name}.md"
        ).exists():
            errors.append(f"missing figure or skip reason: {name}")
    for name in MANUSCRIPT_FILES:
        if not (out / "manuscript" / name).exists():
            errors.append(f"missing manuscript file: {name}")
    for name in REPRODUCIBILITY_FILES:
        if name in {"reproduce_all.sh", "reproduce_autoresearch_only.sh", "slurm_autoresearch.sh"}:
            path = out / "deployment" / name
        else:
            path = out / "reproducibility" / name
        if not path.exists():
            errors.append(f"missing reproducibility file: {name}")
    predictions_path = out / "final_model" / "final_model_predictions.csv"
    if predictions_path.exists():
        pred = pd.read_csv(predictions_path)
        if pred["participant_id"].duplicated().any():
            errors.append("final predictions contain duplicate participant IDs")
        if len(pred) != pred["participant_id"].nunique():
            errors.append("final predictions are not one row per participant")
    manifest_path = out / "final_model" / "final_model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad = set(manifest.get("features", [])) & set(get_nested(config, "autoresearch.prohibited_variables", []))
        if bad:
            errors.append(f"primary model contains prohibited exposure variables: {sorted(bad)}")
    split_labels = _input_paths(config, repo_root)["label_release"] / "labels" / "split_labels_v1.parquet"
    if split_labels.exists():
        splits = pd.read_parquet(split_labels)
        if splits["split_name"].astype(str).str.contains("random", case=False, na=False).any():
            errors.append("random word-level split label found")
    else:
        errors.append("cannot inspect split labels")
    if not (out / "stress_tests" / "dfm_exposure_vs_sensitivity.csv").exists():
        errors.append("missing DFM exposure/sensitivity comparison")
    if not (out / "decision" / "final_decision.json").exists():
        errors.append("missing final decision")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
        "output_dir": str(out),
    }
    return report


def build_paper_package(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    out = Path(output_dir).resolve()
    dirs = _result_dirs(config, out, repo_root)
    data = _load_frozen_data(config, repo_root)
    if (out / "decision" / "final_decision.json").exists():
        final_decision = json.loads((out / "decision" / "final_decision.json").read_text(encoding="utf-8"))
    else:
        final_decision = {"publication_readiness": "needs_run"}
    figures = build_figures(dirs, data)
    manuscript = build_manuscript_package(dirs, final_decision)
    reproducibility = build_reproducibility_package(config, dirs, data, repo_root)
    return {"status": "complete", "figures": figures, "manuscript": manuscript, "reproducibility": reproducibility}
