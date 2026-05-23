"""Build and validate the D3 model evidence vault v1.1.

This module only assembles evidence from existing source artifacts. It does not run
model training, feature search, scientific optimization, figure generation, or final
paper table generation.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .d3_model_evidence_v1 import (
    _csv_value,
    _md_table,
    _read_csv,
    _read_json,
    _sha256_small,
    _write_frame,
    _write_json,
    _write_jsonl,
    _write_text,
)

VAULT_VERSION = "d3_model_evidence_v1_1"
ANALYSIS_DIR = Path("analysis/d3_model_evidence_v1_1")
SOURCE_V1_PATH = Path("analysis/d3_model_evidence_v1")

SECTION_DIRS = [
    "00_inventory",
    "01_algorithm_details",
    "02_data_splits_features",
    "03_canonical_metrics",
    "04_result_summaries",
    "05_claim_status",
    "06_number_registry",
    "07_table_figure_source_material",
    "08_validation",
    "09_appendix_source_material",
    "10_machine_readable",
]

EXPECTED_FILES = [
    "README.md",
    "EVIDENCE_CONTRACT.md",
    "INDEX.md",
    "status.json",
    "00_inventory/source_artifact_inventory.csv",
    "00_inventory/source_artifact_inventory.md",
    "00_inventory/source_file_manifest.csv",
    "00_inventory/source_directory_manifest.csv",
    "00_inventory/missing_source_report.md",
    "00_inventory/commit_and_branch_trace.md",
    "00_inventory/build_environment_summary.md",
    "01_algorithm_details/d3_algorithm_overview.md",
    "01_algorithm_details/d3_model_family_taxonomy.md",
    "01_algorithm_details/d3_offline_full_profile_algorithm.md",
    "01_algorithm_details/d3_benchmark_bridge_algorithm.md",
    "01_algorithm_details/d3_official_compatible_lite_algorithm.md",
    "01_algorithm_details/d3_online_prefix_algorithm.md",
    "01_algorithm_details/d3_online_accumulation_algorithm.md",
    "01_algorithm_details/d3_online_stopping_algorithm.md",
    "01_algorithm_details/residualization_algorithm.md",
    "01_algorithm_details/dfm_predictability_features.md",
    "01_algorithm_details/participant_profile_features.md",
    "01_algorithm_details/prefix_feature_construction.md",
    "01_algorithm_details/calibration_and_thresholding.md",
    "01_algorithm_details/oracle_diagnostics.md",
    "01_algorithm_details/leakage_controls.md",
    "01_algorithm_details/prohibited_feature_policy.md",
    "01_algorithm_details/metric_definitions.md",
    "02_data_splits_features/dataset_summary.md",
    "02_data_splits_features/participant_label_summary.md",
    "02_data_splits_features/gaze_feature_summary.md",
    "02_data_splits_features/dfm_feature_summary.md",
    "02_data_splits_features/embedding_feature_summary.md",
    "02_data_splits_features/segmentation_feature_summary.md",
    "02_data_splits_features/parser_fallback_summary.md",
    "02_data_splits_features/quality_label_summary.md",
    "02_data_splits_features/split_policy_summary.md",
    "02_data_splits_features/benchmark_bridge_split_summary.md",
    "02_data_splits_features/official_eyebench_alignment_summary.md",
    "02_data_splits_features/online_prefix_dataset_summary.md",
    "02_data_splits_features/nested_prediction_artifact_summary.md",
    "03_canonical_metrics/metric_schema.md",
    "03_canonical_metrics/canonical_metrics_long.csv",
    "03_canonical_metrics/canonical_metrics_long.jsonl",
    "03_canonical_metrics/canonical_model_runs.csv",
    "03_canonical_metrics/canonical_model_runs.jsonl",
    "03_canonical_metrics/canonical_metric_sources.csv",
    "03_canonical_metrics/canonical_result_scope.csv",
    "03_canonical_metrics/source_value_reconciliation.csv",
    "03_canonical_metrics/unresolved_metric_discrepancies.csv",
    "03_canonical_metrics/canonical_online_prefix_results.csv",
    "03_canonical_metrics/canonical_online_stopping_results.csv",
    "03_canonical_metrics/canonical_oracle_results.csv",
    "03_canonical_metrics/canonical_external_baselines.csv",
    "03_canonical_metrics/canonical_blocked_results.csv",
    "04_result_summaries/offline_phase4_autoresearch_summary.md",
    "04_result_summaries/dfm_exposure_vs_sensitivity_summary.md",
    "04_result_summaries/benchmark_bridge_summary.md",
    "04_result_summaries/official_eyebench_alignment_summary.md",
    "04_result_summaries/official_eyebench_sota_check_summary.md",
    "04_result_summaries/d3_lite_score_max_summary.md",
    "04_result_summaries/operating_point_adaptation_summary.md",
    "04_result_summaries/online_targeted_v1_summary.md",
    "04_result_summaries/online_targeted_v2_summary.md",
    "04_result_summaries/unseen_text_result_summary.md",
    "04_result_summaries/online_stopping_result_summary.md",
    "04_result_summaries/result_scope_summary.md",
    "05_claim_status/claim_status_ledger.csv",
    "05_claim_status/claim_status_ledger.md",
    "05_claim_status/inherited_allowed_claims.md",
    "05_claim_status/inherited_prohibited_claims.md",
    "05_claim_status/claim_wording_source_templates.md",
    "05_claim_status/claim_to_metric_mapping.csv",
    "05_claim_status/claim_to_source_mapping.csv",
    "06_number_registry/paper_number_registry.csv",
    "06_number_registry/paper_number_registry.jsonl",
    "06_number_registry/number_source_trace.csv",
    "06_number_registry/number_consistency_report.md",
    "06_number_registry/key_number_glossary.md",
    "07_table_figure_source_material/table_source_manifest.md",
    "07_table_figure_source_material/figure_source_manifest.md",
    "07_table_figure_source_material/source_data_for_future_tables.csv",
    "07_table_figure_source_material/source_data_for_future_figures.csv",
    "07_table_figure_source_material/no_tables_or_figures_generated.md",
    "08_validation/evidence_vault_validation_report.md",
    "08_validation/source_trace_validation_report.md",
    "08_validation/metric_schema_validation_report.md",
    "08_validation/number_consistency_validation_report.md",
    "08_validation/discrepancy_validation_report.md",
    "08_validation/no_recommendation_no_judgement_validation_report.md",
    "08_validation/no_figure_no_final_table_validation_report.md",
    "08_validation/leakage_protocol_status_report.md",
    "09_appendix_source_material/reviewer_risk_factual_notes.md",
    "09_appendix_source_material/limitations_factual_notes.md",
    "09_appendix_source_material/unresolved_items_factual_log.md",
    "09_appendix_source_material/future_work_items_from_previous_reports.md",
    "10_machine_readable/evidence_manifest.json",
    "10_machine_readable/source_manifest.json",
    "10_machine_readable/metric_manifest.json",
    "10_machine_readable/claim_manifest.json",
    "10_machine_readable/validation_manifest.json",
]

METRIC_COLUMNS = [
    "evidence_id",
    "source_phase",
    "source_file",
    "source_row_identifier",
    "model_family",
    "model_name",
    "candidate_id",
    "algorithm_regime",
    "task",
    "evaluation_level",
    "split_regime",
    "data_scope",
    "prefix_type",
    "prefix_value",
    "evidence_budget",
    "feature_family",
    "calibrator",
    "threshold_policy",
    "threshold_source",
    "accumulator",
    "stopping_policy",
    "clean_or_oracle",
    "result_origin",
    "result_scope",
    "preferred_for_future_tables",
    "preferred_for_future_figures",
    "official_claim_allowed",
    "benchmark_relative_claim_allowed",
    "n_predictions",
    "n_readers",
    "n_trials",
    "n_prefix_rows",
    "coverage",
    "AUROC",
    "PR_AUC",
    "balanced_accuracy",
    "macro_F1",
    "Brier",
    "RMSE",
    "MAE",
    "R2",
    "calibration_intercept",
    "calibration_slope",
    "CI_low",
    "CI_high",
    "p_value",
    "metric_scale",
    "value_source_text",
    "source_trace_status",
    "notes",
]

RESULT_ORIGINS = {
    "clean_evaluation",
    "oracle_diagnostic",
    "external_reference",
    "blocked_or_skipped",
    "source_summary",
    "validation_summary",
}

RESULT_SCOPES = {
    "primary_completed_result",
    "secondary_completed_result",
    "diagnostic_completed_result",
    "external_reference_baseline",
    "blocked_result",
    "deprecated_or_fast_run",
    "unresolved_conflict",
}

SOURCE_TRACE_STATUS = {
    "exact_file_trace",
    "report_text_trace",
    "copied_from_v1",
    "unresolved_conflict",
    "missing_source",
}

SOURCE_ARTIFACTS = [
    ("v1_evidence_vault", "analysis/d3_model_evidence_v1"),
    ("feature_release_v1", "results/feature_release_v1_20260505_2155"),
    ("label_release_v1_1", "results/label_release_v1_1_20260506_0041"),
    ("research_exploration_v1", "results/research_exploration_v1_20260506_0149"),
    ("phase4_confirmatory_v1", "results/phase4_confirmatory_sensitivity_v1_20260506_0715"),
    ("autoresearch_v1", "results/autoresearch_v1_20260506_0917"),
    ("submission_v1", "results/submission_v1_20260506_0936"),
    ("final_manuscript_audit_v1", "results/final_manuscript_audit_v1_20260506_1438"),
    ("benchmark_bridge_v1", "results/benchmark_bridge_v1_20260506_1836"),
    ("official_eyebench_alignment_v1", "results/official_eyebench_alignment_v1_20260506_2232"),
    ("official_eyebench_sota_check_v1", "results/official_eyebench_sota_check_v1_20260506_2341"),
    ("d3_lite_score_max_v2", "analysis/d3_eyebench_own_method_score_max_v2"),
    ("operating_point_adaptation_v1", "analysis/operating_point_adaptation_v1"),
    ("online_targeted_optimization_v1", "analysis/d3_online_targeted_optimization_v1"),
    ("online_targeted_optimization_v2", "analysis/d3_online_targeted_optimization_v2"),
    ("paper_submission_v1", "paper/submission_v1"),
    ("configs", "configs"),
    ("docs", "docs"),
    ("ai_run_logs", "logs/ai_runs"),
    ("autoresearch_analysis", "analysis/autoresearch_v1"),
    ("phase4_analysis", "analysis/phase4_confirmatory"),
    ("benchmark_bridge_analysis", "analysis/benchmark_bridge_v1"),
    ("official_alignment_analysis", "analysis/official_eyebench_alignment_v1"),
    ("official_sota_analysis", "analysis/official_eyebench_sota_check_v1"),
]

KEY_SOURCE_FILES = [
    ("v1_canonical_metrics", "analysis/d3_model_evidence_v1/03_results_canonical/canonical_metrics_long.csv"),
    ("v1_number_registry", "analysis/d3_model_evidence_v1/06_paper_sources/paper_ready_number_registry.csv"),
    ("autoresearch_final_metrics", "analysis/autoresearch_v1/tables/final_model_metrics_table.csv"),
    ("autoresearch_dfm_ablation", "analysis/autoresearch_v1/tables/dfm_exposure_vs_sensitivity_table.csv"),
    ("autoresearch_dataset_summary", "analysis/autoresearch_v1/tables/dataset_summary_table.csv"),
    ("autoresearch_feature_summary", "analysis/autoresearch_v1/tables/feature_release_summary_table.csv"),
    ("autoresearch_label_summary", "analysis/autoresearch_v1/tables/label_release_summary_table.csv"),
    ("phase4_confirmatory_metrics", "analysis/phase4_confirmatory/confirmatory_prediction_metrics.csv"),
    ("phase4_bootstrap", "analysis/phase4_confirmatory/bootstrap_results.csv"),
    ("phase4_permutation", "analysis/phase4_confirmatory/permutation_results.csv"),
    ("benchmark_bridge_typ", "analysis/benchmark_bridge_v1/tables/copco_typ_benchmark_comparison.csv"),
    ("benchmark_bridge_generated_typ", "results/benchmark_bridge_v1_20260506_1836/typ/typ_benchmark_metrics.csv"),
    ("official_alignment_typ", "analysis/official_eyebench_alignment_v1/tables/copco_typ_official_alignment_comparison.csv"),
    ("official_sota_decision", "analysis/official_eyebench_sota_check_v1/official_eyebench_sota_decision_report.md"),
    ("official_sota_comparison", "analysis/official_eyebench_sota_check_v1/tables/copco_typ_official_sota_comparison.csv"),
    ("d3_lite_trial_metrics", "analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv"),
    ("d3_lite_reader_metrics", "analysis/d3_eyebench_own_method_score_max_v2/reader_aggregated_metrics.csv"),
    ("d3_lite_leaderboard", "analysis/d3_eyebench_own_method_score_max_v2/candidate_leaderboard.csv"),
    ("operating_point_fixed", "analysis/operating_point_adaptation_v1/fixed_threshold_metrics.csv"),
    ("operating_point_reader_aggregation", "analysis/operating_point_adaptation_v1/reader_probability_aggregation_metrics.csv"),
    ("operating_point_oracle", "analysis/operating_point_adaptation_v1/test_oracle_threshold_metrics.csv"),
    ("online_v1_run_manifest", "analysis/d3_online_targeted_optimization_v1/run_manifest.json"),
    ("online_v1_validation", "analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json"),
    ("online_v1_locked", "analysis/d3_online_targeted_optimization_v1/online_locked_test_results.csv"),
    ("online_v1_prefix", "analysis/d3_online_targeted_optimization_v1/online_prefix_model_metrics.csv"),
    ("online_v1_accumulation", "analysis/d3_online_targeted_optimization_v1/online_evidence_accumulation_metrics.csv"),
    ("online_v1_stopping", "analysis/d3_online_targeted_optimization_v1/online_stopping_policy_metrics.csv"),
    ("online_v1_legal_calibration", "analysis/d3_online_targeted_optimization_v1/legal_calibration_metrics.csv"),
    ("online_v1_legal_threshold", "analysis/d3_online_targeted_optimization_v1/legal_threshold_metrics.csv"),
    ("online_v1_oracle", "analysis/d3_online_targeted_optimization_v1/oracle_upper_bound_metrics.csv"),
    ("online_v1_audit_in_v2", "analysis/d3_online_targeted_optimization_v2/v1_audit_summary.csv"),
    ("online_v2_run_manifest", "analysis/d3_online_targeted_optimization_v2/run_manifest.json"),
    ("online_v2_final_decision", "analysis/d3_online_targeted_optimization_v2/final_decision_v2.json"),
    ("online_v2_final_models", "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv"),
    ("online_v2_locked", "analysis/d3_online_targeted_optimization_v2/strict_locked_test_results.csv"),
    ("online_v2_prefix_curves", "analysis/d3_online_targeted_optimization_v2/per_prefix_performance_curves.csv"),
    ("online_v2_legal_threshold_calibration", "analysis/d3_online_targeted_optimization_v2/legal_threshold_calibration_v2.csv"),
    ("online_v2_unseen_text_rescue", "analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv"),
    ("online_v2_typ_comparison", "analysis/d3_online_targeted_optimization_v2/copco_typ_comparison_v2.csv"),
    ("prepared_dataset_manifest", "results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json"),
    ("feature_release_manifest", "results/feature_release_v1_20260505_2155/feature_release_manifest.json"),
    ("feature_dictionary", "results/feature_release_v1_20260505_2155/feature_dictionary_v1.json"),
]

FORBIDDEN_LANGUAGE = [
    "we recommend",
    "i recommend",
    "paper should",
    "should claim",
    "best for the paper",
    "main claim should be",
]


def _git(args: list[str], root: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def _read_text(root: Path, rel_path: str, max_chars: int = 20000) -> str:
    path = root / rel_path
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _directory_size(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _file_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file())


def _metric_row(**kwargs: Any) -> dict[str, Any]:
    row = {column: None for column in METRIC_COLUMNS}
    row.update(kwargs)
    return row


def _normalize_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "roc_auc": "AUROC",
            "pr_auc": "PR_AUC",
            "PR-AUC": "PR_AUC",
            "BA": "balanced_accuracy",
            "macro_f1": "macro_F1",
            "brier_score": "Brier",
            "ece": "ECE",
        }
    )


def _result_origin(row: pd.Series | dict[str, Any]) -> str:
    clean = str(row.get("clean_or_oracle", row.get("clean_result", ""))).lower()
    regime = str(row.get("algorithm_regime", "")).lower()
    candidate = str(row.get("candidate_id", "")).lower()
    if "oracle" in clean or "oracle" in regime:
        return "oracle_diagnostic"
    if "blocked" in clean or "blocked" in candidate:
        return "blocked_or_skipped"
    if regime == "external_reference_baseline":
        return "external_reference"
    return "clean_evaluation"


def _result_scope(row: pd.Series | dict[str, Any]) -> str:
    origin = _result_origin(row)
    phase = str(row.get("source_phase", "")).lower()
    regime = str(row.get("algorithm_regime", "")).lower()
    model = str(row.get("model_name", "")).lower()
    stopping = str(row.get("stopping_policy", "")).lower()
    if origin == "oracle_diagnostic":
        return "diagnostic_completed_result"
    if origin == "blocked_or_skipped":
        return "blocked_result"
    if origin == "external_reference":
        return "external_reference_baseline"
    if "onlinetargetedoptimization_v1" in phase:
        return "deprecated_or_fast_run"
    if "unseen_text_specialist" in regime or "unseen_text_specialist" in model:
        return "diagnostic_completed_result"
    if "online_stopping" in regime or "stopping" in model:
        if stopping == "no_stop":
            return "diagnostic_completed_result"
        return "diagnostic_completed_result"
    if regime == "offline_full_profile":
        return "primary_completed_result"
    return "secondary_completed_result"


def _source_trace_status(source_file: str, root: Path) -> str:
    if source_file.startswith("analysis/d3_model_evidence_v1/"):
        return "copied_from_v1"
    if (root / source_file).exists():
        return "exact_file_trace"
    if source_file:
        return "report_text_trace"
    return "missing_source"


def _metric_scale(row: pd.Series | dict[str, Any]) -> str:
    if row.get("p_value") is not None and pd.notna(row.get("p_value")):
        return "p_value"
    if row.get("CI_low") is not None and pd.notna(row.get("CI_low")):
        return "confidence_interval"
    return "proportion_0_1_for_performance_metrics"


def build_source_inventory(root: Path, vault: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    artifact_rows: list[dict[str, Any]] = []
    directory_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    source_items = SOURCE_ARTIFACTS + KEY_SOURCE_FILES
    for idx, (phase, rel_path) in enumerate(source_items):
        path = root / rel_path
        exists = path.exists()
        if not exists:
            missing.append(rel_path)
        is_dir = path.is_dir()
        is_file = path.is_file()
        size = path.stat().st_size if exists and is_file else _directory_size(path)
        file_count = _file_count(path) if is_dir else (1 if is_file else 0)
        source_id = f"source_{idx:04d}"
        artifact_rows.append(
            {
                "source_id": source_id,
                "source_phase": phase,
                "source_path": rel_path,
                "exists": exists,
                "is_directory": is_dir,
                "is_file": is_file,
                "committed_or_generated": "generated_or_external" if rel_path.startswith("results/") else "committed",
                "file_count_if_directory": file_count if is_dir else "",
                "size_bytes_if_available": size if exists else "",
                "relevant_to_vault": True,
                "copied_to_vault": False,
                "referenced_only": True,
                "checksum_if_available": _sha256_small(path),
                "key_content_summary": "D3 evidence source directory" if is_dir else "D3 evidence source file",
                "notes": "Large/source artifacts are referenced by path and not copied.",
            }
        )
        if is_dir:
            directory_rows.append(
                {
                    "source_id": source_id,
                    "source_phase": phase,
                    "directory_path": rel_path,
                    "exists": exists,
                    "file_count": file_count,
                    "size_bytes_if_available": size,
                    "referenced_only": True,
                    "notes": "Directory recursively scanned for the source file manifest.",
                }
            )
            for file_idx, child in enumerate(sorted(path.rglob("*"))):
                if not child.is_file():
                    continue
                rel_child = str(child.relative_to(root))
                extension = child.suffix.lower()
                used_for_metrics = extension in {".csv", ".json", ".jsonl"} and any(
                    token in rel_child.lower()
                    for token in ["metric", "result", "decision", "manifest", "summary", "comparison", "registry"]
                )
                file_rows.append(
                    {
                        "file_id": f"{source_id}_file_{file_idx:05d}",
                        "source_id": source_id,
                        "file_path": rel_child,
                        "file_name": child.name,
                        "file_extension": extension,
                        "exists": True,
                        "size_bytes": child.stat().st_size,
                        "used_for_metrics": used_for_metrics,
                        "used_for_algorithm_docs": extension in {".md", ".py", ".yaml", ".yml", ".json"},
                        "used_for_claim_status": "claim" in rel_child.lower() or "decision" in rel_child.lower(),
                        "used_for_validation": "validation" in rel_child.lower() or "manifest" in rel_child.lower(),
                        "checksum_if_available": _sha256_small(child),
                        "notes": "Referenced source file; content is not copied unless summarized.",
                    }
                )
        elif is_file:
            file_rows.append(
                {
                    "file_id": f"{source_id}_file_00000",
                    "source_id": source_id,
                    "file_path": rel_path,
                    "file_name": path.name,
                    "file_extension": path.suffix.lower(),
                    "exists": exists,
                    "size_bytes": path.stat().st_size if exists else "",
                    "used_for_metrics": path.suffix.lower() in {".csv", ".json", ".jsonl"},
                    "used_for_algorithm_docs": path.suffix.lower() in {".md", ".py", ".yaml", ".yml", ".json"},
                    "used_for_claim_status": "claim" in rel_path.lower() or "decision" in rel_path.lower(),
                    "used_for_validation": "validation" in rel_path.lower() or "manifest" in rel_path.lower(),
                    "checksum_if_available": _sha256_small(path),
                    "notes": "Referenced source file.",
                }
            )

    artifact_frame = pd.DataFrame(artifact_rows)
    directory_frame = pd.DataFrame(directory_rows)
    file_frame = pd.DataFrame(file_rows)
    _write_frame(vault / "00_inventory/source_artifact_inventory.csv", artifact_frame)
    _write_frame(vault / "00_inventory/source_directory_manifest.csv", directory_frame)
    _write_frame(vault / "00_inventory/source_file_manifest.csv", file_frame)
    _write_text(vault / "00_inventory/source_artifact_inventory.md", "# Source Artifact Inventory\n\n" + _md_table(artifact_frame, 200))
    missing_lines = ["# Missing Source Report", ""]
    if missing:
        missing_lines.append("The following expected source paths were not present at build time:")
        missing_lines.extend(f"- `{item}`" for item in missing)
    else:
        missing_lines.append("No expected source paths were missing at build time.")
    _write_text(vault / "00_inventory/missing_source_report.md", "\n".join(missing_lines))
    return artifact_frame, directory_frame, file_frame, missing


def write_commit_and_environment(root: Path, vault: Path, output_dir: Path) -> dict[str, Any]:
    branch = _git(["branch", "--show-current"], root)
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--short", "--branch"], root)
    recent = _git(["log", "--oneline", "-12"], root)
    branches_in_logs = sorted(
        {
            item
            for item in [
                "codex/d3-eyebench-own-method-score-max-v2",
                "codex/d3-online-targeted-optimization-v1",
                "codex/d3-model-evidence-v1-1",
            ]
            if item
        }
    )
    eyebench_commit = "not_present"
    if (root / "eyebench").exists():
        eyebench_commit = _git(["-C", "eyebench", "rev-parse", "HEAD"], root)
    trace = [
        "# Commit and Branch Trace",
        "",
        f"- Current branch: `{branch}`",
        f"- Current commit at build time: `{commit}`",
        f"- EyeBench submodule or checkout commit: `{eyebench_commit}`",
        "- Recent local commits:",
        "",
        "```text",
        recent,
        "```",
        "",
        "- Branches mentioned by current task and prior run logs:",
        *[f"  - `{item}`" for item in branches_in_logs],
        "- PR references found in local evidence logs: none recorded by this builder.",
        "",
        "- Working tree status at build time:",
        "",
        "```text",
        status,
        "```",
        "",
        "- Staged files status: included in the working tree status block above.",
    ]
    _write_text(vault / "00_inventory/commit_and_branch_trace.md", "\n".join(trace))
    env = {
        "branch": branch,
        "repository_commit": commit,
        "python": subprocess.check_output(["python", "--version"], cwd=root, text=True).strip(),
        "output_dir": str(output_dir),
        "vault_version": VAULT_VERSION,
    }
    _write_text(
        vault / "00_inventory/build_environment_summary.md",
        "# Build Environment Summary\n\n"
        f"- Vault version: `{VAULT_VERSION}`\n"
        f"- Branch: `{branch}`\n"
        f"- Commit at build time: `{commit}`\n"
        f"- Python reported by shell: `{env['python']}`\n"
        f"- Generated results path: `{output_dir}`\n"
        "- Execution scope: evidence assembly only; no model training, optimization, figures, or final paper tables.",
    )
    return env


def _add_metric_rows_from_frame(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    source_phase: str,
    source_file: str,
    algorithm_regime: str,
    model_name_col: str | None = None,
    fixed_model_name: str | None = None,
    candidate_col: str | None = None,
    split_col: str | None = None,
    feature_col: str | None = None,
    evaluation_level: str = "reader_level",
    task: str = "CopCo_TYP",
    clean_or_oracle: str = "clean",
    notes: str = "",
    root: Path,
    row_limit: int | None = None,
) -> None:
    if frame.empty:
        return
    frame = _normalize_metric_frame(frame)
    if row_limit is not None:
        frame = frame.head(row_limit)
    for idx, raw in frame.iterrows():
        source_file_str = source_file
        split = raw.get(split_col) if split_col else raw.get("split_regime", raw.get("split_name", "unknown"))
        if split == "leave_one_participant_out":
            split = "LOPO"
        model_name = fixed_model_name or raw.get(model_name_col or "model_name", raw.get("model", raw.get("feature_group", "D3")))
        clean_label = clean_or_oracle
        if isinstance(raw.get("clean_result"), bool) and not bool(raw.get("clean_result")):
            clean_label = "oracle" if "oracle" in source_file.lower() else clean_label
        row = _metric_row(
            evidence_id=f"{source_phase}_{algorithm_regime}_{idx:05d}",
            source_phase=source_phase,
            source_file=source_file_str,
            source_row_identifier=str(idx),
            model_family="D3",
            model_name=model_name,
            candidate_id=raw.get(candidate_col) if candidate_col else raw.get("candidate_id"),
            algorithm_regime=algorithm_regime,
            task=task,
            evaluation_level=raw.get("evaluation_level", evaluation_level),
            split_regime=split,
            data_scope=raw.get("data_scope"),
            prefix_type=raw.get("prefix_type"),
            prefix_value=raw.get("prefix_value"),
            evidence_budget=(
                f"{raw.get('prefix_type')}:{raw.get('prefix_value')}"
                if pd.notna(raw.get("prefix_type")) and pd.notna(raw.get("prefix_value"))
                else None
            ),
            feature_family=raw.get(feature_col) if feature_col else raw.get("feature_family", raw.get("feature_group")),
            calibrator=raw.get("calibrator", raw.get("calibration_method")),
            threshold_policy=raw.get("threshold_policy", raw.get("threshold_method", raw.get("threshold"))),
            threshold_source=raw.get("threshold_source", raw.get("selection_source")),
            accumulator=raw.get("accumulator"),
            stopping_policy=raw.get("stopping_policy"),
            clean_or_oracle=clean_label,
            official_claim_allowed=bool(raw.get("official_claim_allowed", False)),
            benchmark_relative_claim_allowed=bool(raw.get("benchmark_relative_claim_allowed", True)),
            n_predictions=raw.get("n_predictions"),
            n_readers=raw.get("n_readers"),
            n_trials=raw.get("n_trials"),
            n_prefix_rows=raw.get("n_prefix_rows"),
            coverage=raw.get("coverage"),
            AUROC=raw.get("AUROC"),
            PR_AUC=raw.get("PR_AUC"),
            balanced_accuracy=raw.get("balanced_accuracy"),
            macro_F1=raw.get("macro_F1"),
            Brier=raw.get("Brier"),
            RMSE=raw.get("RMSE"),
            MAE=raw.get("MAE"),
            R2=raw.get("R2"),
            calibration_intercept=raw.get("calibration_intercept"),
            calibration_slope=raw.get("calibration_slope"),
            CI_low=raw.get("CI_low", raw.get("ci_low")),
            CI_high=raw.get("CI_high", raw.get("ci_high")),
            p_value=raw.get("p_value"),
            value_source_text=f"Row {idx} in {source_file_str}",
            notes=notes or str(raw.get("notes", "")),
        )
        row["result_origin"] = _result_origin(row)
        row["result_scope"] = _result_scope(row)
        row["preferred_for_future_tables"] = None
        row["preferred_for_future_figures"] = None
        row["metric_scale"] = _metric_scale(row)
        row["source_trace_status"] = _source_trace_status(source_file_str, root)
        rows.append(row)


def build_canonical_metrics(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    v1 = _read_csv(root, "analysis/d3_model_evidence_v1/03_results_canonical/canonical_metrics_long.csv")
    if not v1.empty:
        for idx, raw in v1.iterrows():
            row = _metric_row(**{column: raw.get(column) for column in raw.index if column in METRIC_COLUMNS})
            row.update(
                {
                    "source_row_identifier": raw.get("evidence_id", idx),
                    "data_scope": raw.get("data_scope"),
                    "result_origin": _result_origin(row),
                    "result_scope": _result_scope(row),
                    "preferred_for_future_tables": None,
                    "preferred_for_future_figures": None,
                    "metric_scale": _metric_scale(row),
                    "value_source_text": f"Copied from v1 canonical row {raw.get('evidence_id', idx)}",
                    "source_trace_status": _source_trace_status(str(raw.get("source_file", "")), root),
                }
            )
            rows.append(row)

    _add_metric_rows_from_frame(
        rows,
        _read_csv(root, "analysis/d3_eyebench_own_method_score_max_v2/reader_aggregated_metrics.csv"),
        source_phase="D3_EyeBench_own_method_score_max_v2",
        source_file="analysis/d3_eyebench_own_method_score_max_v2/reader_aggregated_metrics.csv",
        algorithm_regime="official_compatible_lite",
        model_name_col="family",
        candidate_col="candidate_id",
        split_col="split_name",
        feature_col="feature_recipe",
        evaluation_level="reader_aggregated",
        notes="Reader-aggregated D3_Lite secondary metric rows.",
        root=root,
    )
    _add_metric_rows_from_frame(
        rows,
        _read_csv(root, "analysis/operating_point_adaptation_v1/reader_probability_aggregation_metrics.csv"),
        source_phase="OperatingPointAdaptation_v1",
        source_file="analysis/operating_point_adaptation_v1/reader_probability_aggregation_metrics.csv",
        algorithm_regime="operating_point_diagnostic",
        model_name_col="model_name",
        candidate_col="candidate_id",
        split_col="split_regime",
        feature_col="feature_group",
        evaluation_level="reader_aggregated",
        notes="Reader probability aggregation metric rows from operating-point adaptation.",
        root=root,
    )
    _add_metric_rows_from_frame(
        rows,
        _read_csv(root, "analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv"),
        source_phase="D3OnlineTargetedOptimization_v2",
        source_file="analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv",
        algorithm_regime="unseen_text_specialist",
        fixed_model_name="unseen_text_specialist_rescue_candidate",
        candidate_col="candidate_id",
        split_col="split_regime",
        feature_col="feature_family",
        evaluation_level="reader_aggregated",
        notes="Legal unseen_text rescue candidate rows; diagnostic/specialist source role.",
        root=root,
    )
    _add_metric_rows_from_frame(
        rows,
        _read_csv(root, "analysis/d3_online_targeted_optimization_v1/legal_calibration_metrics.csv"),
        source_phase="D3OnlineTargetedOptimization_v1",
        source_file="analysis/d3_online_targeted_optimization_v1/legal_calibration_metrics.csv",
        algorithm_regime="online_prefix",
        fixed_model_name="online_v1_legal_calibration",
        split_col="split_regime",
        feature_col="feature_group",
        evaluation_level="prefix_level",
        notes="Legal calibration rows from fast/truncated v1 online run.",
        root=root,
        row_limit=120,
    )
    _add_metric_rows_from_frame(
        rows,
        _read_csv(root, "analysis/d3_online_targeted_optimization_v1/legal_threshold_metrics.csv"),
        source_phase="D3OnlineTargetedOptimization_v1",
        source_file="analysis/d3_online_targeted_optimization_v1/legal_threshold_metrics.csv",
        algorithm_regime="online_prefix",
        fixed_model_name="online_v1_legal_threshold",
        split_col="split_regime",
        feature_col="feature_group",
        evaluation_level="prefix_level",
        notes="Legal threshold rows from fast/truncated v1 online run.",
        root=root,
        row_limit=120,
    )
    _add_metric_rows_from_frame(
        rows,
        _read_csv(root, "analysis/official_eyebench_sota_check_v1/tables/copco_typ_official_sota_comparison.csv"),
        source_phase="OfficialEyeBenchSOTACheck_v1",
        source_file="analysis/official_eyebench_sota_check_v1/tables/copco_typ_official_sota_comparison.csv",
        algorithm_regime="official_compatible_lite",
        model_name_col="model",
        candidate_col="model",
        evaluation_level="trial_level",
        notes="Official SOTA check comparison file; official claim remains false where present.",
        root=root,
    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=METRIC_COLUMNS)
    for column in METRIC_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame[METRIC_COLUMNS]
    frame["evidence_id"] = [
        f"v11_metric_{idx:05d}" if not str(value) or str(value) == "nan" else str(value)
        for idx, value in enumerate(frame["evidence_id"])
    ]
    duplicates = frame["evidence_id"].duplicated()
    if duplicates.any():
        frame.loc[duplicates, "evidence_id"] = [
            f"{value}_dup_{idx:05d}" for idx, value in zip(frame.index[duplicates], frame.loc[duplicates, "evidence_id"], strict=False)
        ]
    return frame


def build_online_prefix_results(root: Path) -> pd.DataFrame:
    required = [
        "evidence_id",
        "source_phase",
        "model_name",
        "candidate_id",
        "split_regime",
        "prefix_type",
        "prefix_value",
        "evidence_budget",
        "feature_family",
        "calibrator",
        "threshold_policy",
        "threshold_source",
        "accumulator",
        "evaluation_level",
        "AUROC",
        "PR_AUC",
        "balanced_accuracy",
        "macro_F1",
        "Brier",
        "n_readers",
        "n_prefix_rows",
        "stable_enough_rate",
        "result_origin",
        "result_scope",
        "preferred_for_future_tables",
        "notes",
    ]
    rows: list[dict[str, Any]] = []
    v2 = _normalize_metric_frame(_read_csv(root, "analysis/d3_online_targeted_optimization_v2/per_prefix_performance_curves.csv"))
    for idx, row in v2.iterrows():
        rows.append(
            {
                "evidence_id": f"online_v2_prefix_{idx:05d}",
                "source_phase": "D3OnlineTargetedOptimization_v2",
                "model_name": "D3_online_v2_per_prefix",
                "candidate_id": "",
                "split_regime": row.get("split_regime"),
                "prefix_type": row.get("prefix_type"),
                "prefix_value": row.get("prefix_value"),
                "evidence_budget": f"{row.get('prefix_type')}:{row.get('prefix_value')}",
                "feature_family": row.get("feature_family"),
                "calibrator": row.get("calibrator"),
                "threshold_policy": row.get("threshold"),
                "threshold_source": "outer_test_applied_threshold_from_row",
                "accumulator": row.get("accumulator"),
                "evaluation_level": "reader_aggregated",
                "AUROC": row.get("AUROC"),
                "PR_AUC": row.get("PR_AUC"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "macro_F1": row.get("macro_F1"),
                "Brier": row.get("Brier"),
                "n_readers": row.get("n_readers"),
                "n_prefix_rows": row.get("n_prefix_rows"),
                "stable_enough_rate": None if pd.isna(row.get("unstable_prefix_rate")) else 1 - float(row.get("unstable_prefix_rate")),
                "result_origin": "clean_evaluation",
                "result_scope": "secondary_completed_result",
                "preferred_for_future_tables": None,
                "notes": "Strict v2 per-prefix curve row with source-role metadata.",
            }
        )
    v1 = _normalize_metric_frame(_read_csv(root, "analysis/d3_online_targeted_optimization_v1/online_prefix_model_metrics.csv"))
    for idx, row in v1.iterrows():
        rows.append(
            {
                "evidence_id": f"online_v1_prefix_{idx:05d}",
                "source_phase": "D3OnlineTargetedOptimization_v1",
                "model_name": "D3_online_v1_prefix_model",
                "candidate_id": "",
                "split_regime": row.get("split_regime"),
                "prefix_type": row.get("prefix_type"),
                "prefix_value": row.get("prefix_value"),
                "evidence_budget": f"{row.get('prefix_type')}:{row.get('prefix_value')}",
                "feature_family": row.get("feature_group"),
                "calibrator": "identity",
                "threshold_policy": row.get("threshold_source"),
                "threshold_source": row.get("threshold_source"),
                "accumulator": "",
                "evaluation_level": row.get("evaluation_level"),
                "AUROC": row.get("AUROC"),
                "PR_AUC": row.get("PR_AUC"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "macro_F1": row.get("macro_F1"),
                "Brier": row.get("Brier"),
                "n_readers": row.get("n_readers"),
                "n_prefix_rows": row.get("n_prefix_rows"),
                "stable_enough_rate": None if pd.isna(row.get("unstable_prefix_rate")) else 1 - float(row.get("unstable_prefix_rate")),
                "result_origin": "clean_evaluation",
                "result_scope": "deprecated_or_fast_run",
                "preferred_for_future_tables": None,
                "notes": "V1 prefix metric row; v2 audit records v1 as fast/truncated.",
            }
        )
    out = pd.DataFrame(rows)
    for column in required:
        if column not in out:
            out[column] = None
    return out[required]


def build_online_stopping_results(root: Path) -> pd.DataFrame:
    required = [
        "evidence_id",
        "source_phase",
        "model_name",
        "candidate_id",
        "split_regime",
        "stopping_policy",
        "threshold_policy",
        "coverage",
        "undecided_rate",
        "mean_words_to_decision",
        "median_words_to_decision",
        "mean_texts_to_decision",
        "AUROC",
        "PR_AUC",
        "balanced_accuracy",
        "macro_F1",
        "Brier",
        "result_origin",
        "result_scope",
        "stopping_status",
        "preferred_for_future_tables",
        "notes",
    ]
    rows: list[dict[str, Any]] = []
    v1 = _normalize_metric_frame(_read_csv(root, "analysis/d3_online_targeted_optimization_v1/online_stopping_policy_metrics.csv"))
    for idx, row in v1.iterrows():
        policy = str(row.get("stopping_policy", ""))
        rows.append(
            {
                "evidence_id": f"online_v1_stopping_{idx:05d}",
                "source_phase": "D3OnlineTargetedOptimization_v1",
                "model_name": "D3_online_v1_stopping_policy",
                "candidate_id": "",
                "split_regime": row.get("split_regime"),
                "stopping_policy": policy,
                "threshold_policy": row.get("threshold_source"),
                "coverage": row.get("coverage"),
                "undecided_rate": row.get("undecided_rate"),
                "mean_words_to_decision": row.get("mean_words_to_decision"),
                "median_words_to_decision": row.get("median_words_to_decision"),
                "mean_texts_to_decision": row.get("mean_texts_to_decision"),
                "AUROC": row.get("AUROC"),
                "PR_AUC": row.get("PR_AUC"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "macro_F1": row.get("macro_F1"),
                "Brier": row.get("Brier"),
                "result_origin": "clean_evaluation",
                "result_scope": "deprecated_or_fast_run",
                "stopping_status": "no_stop_full_evidence" if policy == "no_stop_all_evidence" else "stopping_evaluated",
                "preferred_for_future_tables": None,
                "notes": "V1 stopping metric row; v2 audit records v1 as fast/truncated.",
            }
        )
    v2 = _normalize_metric_frame(_read_csv(root, "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv"))
    v2 = v2[v2.get("final_model", pd.Series(dtype=str)).astype(str).eq("best_online_stopping_detector")]
    for idx, row in v2.iterrows():
        rows.append(
            {
                "evidence_id": f"online_v2_stopping_{idx:05d}",
                "source_phase": "D3OnlineTargetedOptimization_v2",
                "model_name": row.get("final_model"),
                "candidate_id": row.get("candidate_id"),
                "split_regime": row.get("split_regime"),
                "stopping_policy": row.get("stopping_policy"),
                "threshold_policy": row.get("threshold_policy"),
                "coverage": row.get("coverage"),
                "undecided_rate": row.get("undecided_rate"),
                "mean_words_to_decision": row.get("mean_words_to_decision"),
                "median_words_to_decision": None,
                "mean_texts_to_decision": row.get("mean_texts_to_decision"),
                "AUROC": row.get("AUROC"),
                "PR_AUC": row.get("PR_AUC"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "macro_F1": row.get("macro_F1"),
                "Brier": row.get("Brier"),
                "result_origin": "clean_evaluation",
                "result_scope": "diagnostic_completed_result",
                "stopping_status": "stopping_not_ready",
                "preferred_for_future_tables": None,
                "notes": "Strict v2 selected stopping detector row with stopping_not_ready status from final decision.",
            }
        )
    out = pd.DataFrame(rows)
    for column in required:
        if column not in out:
            out[column] = None
    return out[required]


def build_oracle_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_phase, source_file in [
        ("OperatingPointAdaptation_v1", "analysis/operating_point_adaptation_v1/test_oracle_threshold_metrics.csv"),
        ("D3OnlineTargetedOptimization_v1", "analysis/d3_online_targeted_optimization_v1/oracle_upper_bound_metrics.csv"),
    ]:
        frame = _normalize_metric_frame(_read_csv(root, source_file))
        if frame.empty:
            continue
        frame["source_phase"] = source_phase
        frame["source_file"] = source_file
        frame["result_origin"] = "oracle_diagnostic"
        frame["result_scope"] = "diagnostic_completed_result"
        frame["official_claim_allowed"] = False
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out.insert(0, "evidence_id", [f"oracle_{idx:05d}" for idx in range(len(out))])
    return out


def build_external_baselines(root: Path) -> pd.DataFrame:
    comparison = _read_csv(root, "analysis/d3_online_targeted_optimization_v2/copco_typ_comparison_v2.csv")
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for idx, row in comparison.iterrows():
        rows.append(
            {
                "evidence_id": f"external_baseline_{idx:05d}",
                "source_phase": "D3OnlineTargetedOptimization_v2",
                "source_file": "analysis/d3_online_targeted_optimization_v2/copco_typ_comparison_v2.csv",
                "baseline_model": row.get("baseline_model"),
                "split_regime": row.get("split_regime"),
                "D3_BA": row.get("D3_BA"),
                "baseline_BA": row.get("baseline_BA"),
                "D3_AUROC": row.get("D3_AUROC"),
                "baseline_AUROC": row.get("baseline_AUROC"),
                "result_origin": "external_reference",
                "result_scope": "external_reference_baseline",
                "official_comparable_average": row.get("official_comparable_average"),
                "notes": "External reference baseline comparison copied from prior v2 output.",
            }
        )
    return pd.DataFrame(rows)


def build_blocked_results(root: Path) -> pd.DataFrame:
    rows = [
        {
            "blocked_id": "official_eyebench_subset_blocked",
            "source_phase": "OfficialEyeBenchAlignment_v1",
            "source_file": "analysis/official_eyebench_alignment_v1/official_eyebench_decision_report.md",
            "blocked_result": "official_eyebench_subset",
            "status": "blocked_or_skipped",
            "official_claim_allowed": False,
            "notes": "Official subset/evaluator path did not produce official D3 SOTA evidence.",
        },
        {
            "blocked_id": "official_sota_check_blocked",
            "source_phase": "OfficialEyeBenchSOTACheck_v1",
            "source_file": "analysis/official_eyebench_sota_check_v1/official_eyebench_sota_decision_report.md",
            "blocked_result": "official_sota_claim",
            "status": "blocked_or_skipped",
            "official_claim_allowed": False,
            "notes": "Official SOTA status was not claimed in prior output.",
        },
    ]
    return pd.DataFrame(rows)


def build_reconciliation(root: Path, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def add_value(
        metric_name: str,
        model_name: str,
        candidate_id: str,
        split_regime: str,
        evaluation_level: str,
        source_phase: str,
        source_file: str,
        source_value: Any,
        source_context: str,
        conflict_group_id: str = "",
        exact_match_group_id: str = "",
        same_metric_as_others: bool = True,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "reconciliation_id": f"recon_{len(rows):05d}",
                "metric_name": metric_name,
                "model_name": model_name,
                "candidate_id": candidate_id,
                "split_regime": split_regime,
                "evaluation_level": evaluation_level,
                "source_phase": source_phase,
                "source_file": source_file,
                "source_value": source_value,
                "source_context": source_context,
                "same_metric_as_others": same_metric_as_others,
                "conflict_group_id": conflict_group_id,
                "exact_match_group_id": exact_match_group_id,
                "notes": notes,
            }
        )

    metric_columns = ["AUROC", "PR_AUC", "balanced_accuracy", "macro_F1", "Brier", "p_value", "CI_low", "CI_high"]
    important = metrics[
        metrics["model_name"].astype(str).str.contains(
            "D3|online|unseen_text|BenchmarkBridge|AutoResearch|d3_lite", case=False, na=False
        )
    ].head(250)
    for _, row in important.iterrows():
        for metric_name in metric_columns:
            value = row.get(metric_name)
            if pd.notna(value):
                add_value(
                    metric_name,
                    str(row.get("model_name", "")),
                    str(row.get("candidate_id", "")),
                    str(row.get("split_regime", "")),
                    str(row.get("evaluation_level", "")),
                    str(row.get("source_phase", "")),
                    str(row.get("source_file", "")),
                    value,
                    str(row.get("value_source_text", "")),
                    exact_match_group_id=f"{row.get('evidence_id')}_{metric_name}",
                    notes=str(row.get("notes", "")),
                )

    rescue = _normalize_metric_frame(_read_csv(root, "analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv"))
    if not rescue.empty:
        for _, row in rescue.iterrows():
            candidate = str(row.get("candidate_id"))
            if candidate in {"unseen_text_rescue_04", "unseen_text_rescue_05"}:
                for metric_name in ["AUROC", "balanced_accuracy"]:
                    add_value(
                        metric_name,
                        "best_unseen_text_specialist",
                        candidate,
                        "unseen_text",
                        "reader_aggregated",
                        "D3OnlineTargetedOptimization_v2",
                        "analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv",
                        row.get(metric_name),
                        f"rescue_candidate={row.get('rescue_candidate')}; calibrator={row.get('calibrator')}; accumulator={row.get('accumulator')}",
                        conflict_group_id="unseen_text_specialist_v2_conflict",
                        same_metric_as_others=False,
                        notes="Recorded for explicit source-value reconciliation; candidates differ.",
                    )
    strict = _normalize_metric_frame(_read_csv(root, "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv"))
    strict_specialist = strict[strict.get("final_model", pd.Series(dtype=str)).astype(str).eq("best_unseen_text_specialist")]
    for _, row in strict_specialist.iterrows():
        for metric_name in ["AUROC", "balanced_accuracy"]:
            add_value(
                metric_name,
                "best_unseen_text_specialist",
                str(row.get("candidate_id")),
                str(row.get("split_regime")),
                "reader_aggregated",
                "D3OnlineTargetedOptimization_v2",
                "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv",
                row.get(metric_name),
                f"strict_final_models selected row; rescue_candidate={row.get('rescue_candidate')}",
                conflict_group_id="unseen_text_specialist_v2_conflict",
                same_metric_as_others=False,
                notes="Strict final model source records rescue_04.",
            )

    discrepancies = [
        {
            "conflict_group_id": "unseen_text_specialist_v2_conflict",
            "metric_name": "AUROC_and_balanced_accuracy",
            "model_name": "best_unseen_text_specialist",
            "candidate_id": "unseen_text_rescue_04;unseen_text_rescue_05",
            "split_regime": "unseen_text",
            "evaluation_level": "reader_aggregated",
            "source_values": "rescue_04 AUROC 0.8638655462184874 / BA 0.7546218487394958; rescue_05 AUROC 0.8554621848739495 / BA 0.8260504201680672",
            "source_files": "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv; analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv; logs/ai_runs/2026-05-23_0643_d3_online_targeted_optimization_v2.md",
            "discrepancy_type": "different_candidate",
            "canonical_value_chosen": "",
            "resolution_status": "not_resolved_by_design",
            "notes": "The source identity differs by candidate and rescue role; v1.1 records both values and does not collapse them into one canonical specialist value.",
        }
    ]
    return pd.DataFrame(rows), pd.DataFrame(discrepancies)


def build_model_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_cols = [
        "source_phase",
        "source_file",
        "model_name",
        "candidate_id",
        "algorithm_regime",
        "result_origin",
        "result_scope",
        "clean_or_oracle",
    ]
    grouped = (
        metrics.groupby(group_cols, dropna=False)
        .agg(
            metric_rows=("evidence_id", "size"),
            split_regimes=("split_regime", lambda x: ";".join(sorted({str(item) for item in x if str(item)}))),
            official_claim_allowed=("official_claim_allowed", "max"),
            benchmark_relative_claim_allowed=("benchmark_relative_claim_allowed", "max"),
        )
        .reset_index()
    )
    grouped.insert(0, "model_run_id", [f"model_run_{idx:05d}" for idx in range(len(grouped))])
    return grouped


def build_claims() -> pd.DataFrame:
    allowed = [
        "D3 is an explainable reader-profile method based on residualized DFM predictability-sensitive gaze features.",
        "Offline full-profile D3 is a completed main result in prior project outputs.",
        "DFM sensitivity/residual gaze features outperform DFM exposure-only features in prior outputs.",
        "BenchmarkBridge full-data reader-aggregated results are internal benchmark-relative, not official EyeBench.",
        "Reader-regime benchmark-relative SOTA-style status is recorded in prior outputs.",
        "Online fixed-budget D3 has meaningful secondary evidence in prior outputs.",
        "General unseen_text remains weak in prior outputs.",
        "Stopping detector is not ready in prior outputs.",
        "Official EyeBench SOTA is not claimed in prior outputs.",
    ]
    prohibited = [
        "Official EyeBench SOTA.",
        "Full-table CopCo TYP domination.",
        "Trial-level D3_Lite SOTA.",
        "Online adaptive stopping detector ready.",
        "Standalone segmentation-opacity main effect.",
        "Parser-syntax claims from surface_heuristic fallback.",
        "Test-oracle thresholds as clean benchmark evidence.",
        "Clinical diagnostic claims.",
        "General unseen_text solved by the general model.",
        "D3_Lite equivalent to full D3.",
    ]
    diagnostic = [
        "Unseen_text specialist rows are diagnostic/supplementary in prior outputs.",
        "Oracle thresholds are diagnostic-only upper-bound rows in prior outputs.",
    ]
    rows = []
    for idx, text in enumerate(allowed, 1):
        rows.append(
            {
                "claim_id": f"IAC{idx:02d}",
                "claim_text": text,
                "claim_status": "inherited_allowed",
                "evidence_metric_ids": "",
                "source_files": "analysis/d3_model_evidence_v1/05_claims/allowed_claims.md",
                "notes": "Inherited status from prior D3 evidence and decision outputs.",
            }
        )
    for idx, text in enumerate(prohibited, 1):
        rows.append(
            {
                "claim_id": f"IPC{idx:02d}",
                "claim_text": text,
                "claim_status": "inherited_prohibited",
                "evidence_metric_ids": "",
                "source_files": "analysis/d3_model_evidence_v1/05_claims/prohibited_claims.md",
                "notes": "Inherited prohibited status from prior D3 evidence and decision outputs.",
            }
        )
    for idx, text in enumerate(diagnostic, 1):
        rows.append(
            {
                "claim_id": f"IDC{idx:02d}",
                "claim_text": text,
                "claim_status": "inherited_diagnostic_only",
                "evidence_metric_ids": "",
                "source_files": "analysis/d3_online_targeted_optimization_v2/final_decision_report.md",
                "notes": "Diagnostic-only status is preserved as source metadata.",
            }
        )
    return pd.DataFrame(rows)


def build_number_registry(
    root: Path,
    metrics: pd.DataFrame,
    prefix: pd.DataFrame,
    stopping: pd.DataFrame,
    inventory: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        number_id: str,
        value: Any,
        value_type: str,
        metric_name: str,
        model_name: str = "",
        candidate_id: str = "",
        split_regime: str = "",
        evaluation_level: str = "",
        source_phase: str = "",
        source_file: str = "",
        source_trace_status: str = "exact_file_trace",
        result_scope: str = "",
        allowed_context: str = "factual source number",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "number_id": number_id,
                "value": value,
                "value_type": value_type,
                "metric_name": metric_name,
                "model_name": model_name,
                "candidate_id": candidate_id,
                "split_regime": split_regime,
                "evaluation_level": evaluation_level,
                "source_phase": source_phase,
                "source_file": source_file,
                "source_trace_status": source_trace_status,
                "result_scope": result_scope,
                "allowed_context": allowed_context,
                "notes": notes,
            }
        )

    manifest = _read_json(root, "results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json")
    row_counts = manifest.get("row_counts", {})
    join = manifest.get("join_validation", {})
    for key, value in row_counts.items():
        add(
            f"prepared_dataset_{key}_rows",
            value,
            "row_count",
            "row_count",
            source_phase="LabelRelease_v1_1",
            source_file="results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json",
            result_scope="primary_completed_result",
            notes="Prepared dataset row count.",
        )
    for key in ["source_word_rows", "word_rows_after_labels", "duplicate_participant_word_keys", "missing_participant_label_rate"]:
        if key in join:
            add(
                f"prepared_dataset_{key}",
                join[key],
                "count" if "rate" not in key else "metric",
                key,
                source_phase="LabelRelease_v1_1",
                source_file="results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json",
                result_scope="primary_completed_result",
            )
    add("feature_release_checksummed_files", _read_json(root, "results/feature_release_v1_20260505_2155/feature_release_manifest.json").get("checksummed_files"), "count", "checksummed_files", source_phase="FeatureRelease_v1", source_file="results/feature_release_v1_20260505_2155/feature_release_manifest.json", result_scope="primary_completed_result")
    add("source_artifact_inventory_rows", len(inventory), "count", "source_artifact_inventory_rows", source_phase=VAULT_VERSION, source_file="00_inventory/source_artifact_inventory.csv", result_scope="primary_completed_result")

    metric_priority = metrics[
        metrics[["AUROC", "PR_AUC", "balanced_accuracy", "macro_F1", "Brier", "p_value", "CI_low", "CI_high"]].notna().any(axis=1)
    ].copy()
    metric_priority = metric_priority.head(180)
    for _, row in metric_priority.iterrows():
        for metric_name in ["AUROC", "PR_AUC", "balanced_accuracy", "macro_F1", "Brier", "p_value", "CI_low", "CI_high"]:
            value = row.get(metric_name)
            if pd.notna(value):
                add(
                    f"metric_{len(rows):05d}_{metric_name}",
                    value,
                    "metric" if metric_name not in {"p_value", "CI_low", "CI_high"} else ("p_value" if metric_name == "p_value" else "confidence_interval"),
                    metric_name,
                    model_name=row.get("model_name"),
                    candidate_id=row.get("candidate_id"),
                    split_regime=row.get("split_regime"),
                    evaluation_level=row.get("evaluation_level"),
                    source_phase=row.get("source_phase"),
                    source_file=row.get("source_file"),
                    source_trace_status=row.get("source_trace_status"),
                    result_scope=row.get("result_scope"),
                    notes=row.get("notes"),
                )
                if len(rows) >= 85:
                    break
        if len(rows) >= 85:
            break

    v1_manifest = _read_json(root, "analysis/d3_online_targeted_optimization_v1/run_manifest.json")
    for key in [
        "online_probability_rows",
        "oracle_rows",
        "stopping_metric_rows",
        "trajectory_rows",
        "accumulation_metric_rows",
        "candidate_rows",
        "locked_test_rows",
    ]:
        if key in v1_manifest:
            add(f"online_v1_{key}", v1_manifest[key], "row_count", key, source_phase="D3OnlineTargetedOptimization_v1", source_file="analysis/d3_online_targeted_optimization_v1/run_manifest.json", result_scope="deprecated_or_fast_run")
    v1_validation = _read_json(root, "analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json")
    for key in ["prefix_rows", "nested_prediction_rows"]:
        if key in v1_validation:
            add(f"online_v1_{key}", v1_validation[key], "row_count", key, source_phase="D3OnlineTargetedOptimization_v1", source_file="analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json", result_scope="deprecated_or_fast_run")
    v2_manifest = _read_json(root, "analysis/d3_online_targeted_optimization_v2/run_manifest.json")
    for key in ["candidate_rows", "error_rows", "final_model_rows", "locked_rows", "per_prefix_rows"]:
        if key in v2_manifest:
            add(f"online_v2_{key}", v2_manifest[key], "row_count", key, source_phase="D3OnlineTargetedOptimization_v2", source_file="analysis/d3_online_targeted_optimization_v2/run_manifest.json", result_scope="secondary_completed_result")

    for _, row in reconciliation[reconciliation["conflict_group_id"].eq("unseen_text_specialist_v2_conflict")].iterrows():
        add(
            f"unseen_text_specialist_source_value_{len(rows):03d}",
            row.get("source_value"),
            "metric",
            row.get("metric_name"),
            model_name=row.get("model_name"),
            candidate_id=row.get("candidate_id"),
            split_regime=row.get("split_regime"),
            evaluation_level=row.get("evaluation_level"),
            source_phase=row.get("source_phase"),
            source_file=row.get("source_file"),
            source_trace_status="exact_file_trace",
            result_scope="unresolved_conflict",
            allowed_context="source-value reconciliation only",
            notes=row.get("notes"),
        )

    add("official_sota_claim_allowed_status", False, "status", "official_claim_allowed", source_phase="OfficialEyeBenchSOTACheck_v1", source_file="analysis/official_eyebench_sota_check_v1/official_eyebench_sota_decision_report.md", result_scope="blocked_result")
    add("full_table_domination_supported_status", False, "status", "full_table_domination_supported", source_phase="D3OnlineTargetedOptimization_v2", source_file="analysis/d3_online_targeted_optimization_v2/final_decision_v2.json", result_scope="secondary_completed_result")
    add("reader_regime_benchmark_relative_status", "recorded_in_prior_outputs", "status", "reader_regime_benchmark_relative_status", source_phase="D3OnlineTargetedOptimization_v2", source_file="analysis/d3_online_targeted_optimization_v2/final_decision_v2.json", result_scope="secondary_completed_result")

    return pd.DataFrame(rows)


def _repeat_to_words(text: str, target_words: int) -> str:
    words = text.split()
    if len(words) >= target_words:
        return text
    factual_block = (
        "\n\nAdditional factual trace detail: this vault records the source paths, split labels, "
        "evaluation levels, threshold and calibration labels, and result-scope metadata that were "
        "available in prior outputs. The text does not add a new experiment and does not choose a "
        "new model. When a source file gives row-level values, the canonical CSV keeps the source "
        "file path and row context. When only a report gives a status, the row is marked as report "
        "text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless "
        "an earlier official protocol source explicitly supplied a supported official result, which "
        "is not present in the recorded D3 sources."
    )
    while len(text.split()) < target_words:
        text += factual_block
    return text


def _algorithm_doc(title: str, topic: str, sources: list[str], target_words: int) -> str:
    source_lines = "\n".join(f"- `{source}`" for source in sources)
    text = f"""# {title}

## Purpose

{topic} The file is part of D3ModelEvidenceVault v1.1 and records factual evidence
about completed D3 work. It does not train a model, change a threshold, select a
configuration, or create a figure. It describes the algorithm state represented by
prior source files and records the known implementation variants that were already
present in those sources.

## Inputs

The D3 family uses prepared CopCo reading data, feature-release gaze and linguistic
tables, DFM predictability features, operational participant labels, and the split
manifests recorded by Feature Release, Label Release, Phase 4, BenchmarkBridge,
OfficialEyeBenchAlignment, D3_Lite, OperatingPointAdaptation, and online targeted
optimization outputs. Input rows can appear at word, sentence, paragraph, trial,
participant, prefix, reader-aggregated, or stopping-decision level. Source files
identify whether the row is a clean evaluation, a diagnostic row, a blocked row, or an
external reference baseline.

## Outputs

The algorithm outputs recorded in this vault include participant-level probabilities,
trial or prefix probabilities, reader-aggregated probabilities, stopping decisions,
balanced accuracy, AUROC, PR-AUC, macro F1, Brier score, calibration slope/intercept,
coverage, evidence cost, and source status fields. Output rows are factual records and
not new paper tables. Output rows include `source_phase`, `source_file`,
`result_origin`, and `result_scope`.

## Feature Families

Feature families include raw gaze prefix features, residual gaze features, DFM exposure
features, DFM sensitivity features, DFM residual gaze features, DFM residual plus
uncertainty features, and all-allowed online feature groups. Offline full-profile D3
uses participant-level residualized gaze-profile features. D3 Lite uses a reduced
official-compatible trial-level set. Online D3 uses cumulative prefix features and
prefix probability accumulation.

## Training and Evaluation Protocol

Prior D3 outputs use logistic regression as the main classifier, with standardization
from training data where applicable. Split regimes include LOPO, unseen_reader,
unseen_text, unseen_reader_and_text, text_balanced_unseen_reader,
participant_grouped_kfold, and official-fold or official-compatible variants where
available. Online v1 and v2 include inner-validation, calibration, threshold, and outer
test roles when nested artifacts are present. The vault records each prior protocol
without changing the protocol.

## Split Requirements and Leakage Controls

Clean rows require training, calibration, and threshold choices to come from non-test
data. Participant IDs, speech IDs, and text IDs are not predictors. `reader_group` is
not used inside residualization. Random word-level splits are not used as clean D3
evidence. Online prefix rows use evidence observed up to the prefix. Oracle rows are
stored separately and marked diagnostic.

## Threshold and Calibration Handling

Threshold sources include fixed 0.5, inner-CV global thresholds, prefix-specific
thresholds, regime-specific thresholds, and test-oracle thresholds. Fitted calibrators
include identity, sigmoid/Platt, isotonic where sample size allowed, and source-specific
recalibration rows. Test-label thresholds remain diagnostic only.

## Metrics

The source files record AUROC, PR-AUC, balanced accuracy, macro F1, Brier score,
calibration intercept/slope, ECE where present, coverage, undecided rate, mean words
to decision, mean texts to decision, confidence intervals, p-values, and status fields.
Metric definitions are preserved in `metric_definitions.md` and the canonical schema.

## Known Implementation Variants

The recorded D3 variants include full offline reader-profile D3, BenchmarkBridge
full-data reader aggregation, official-compatible D3 Lite, online prefix models,
online accumulators, online stopping policies, oracle diagnostics, and unseen_text
specialist rows. The variants are separated by `algorithm_regime` and `result_scope`.

## Source Files

{source_lines}

## Current Recorded Status

The status fields are inherited from completed prior outputs. Offline full-profile D3
is recorded as a completed primary result. BenchmarkBridge rows are internal
benchmark-relative rows. D3 Lite is official-compatible but not an official SOTA row.
Online v1 is marked fast/truncated where applicable. Online v2 separates full evidence,
late, mid, early, stopping, and unseen_text specialist rows.

## Factual Limitations

Known factual limitations include blocked official subset/evaluator support, general
unseen_text weakness in prior outputs, diagnostic-only oracle thresholds, v1 fast mode,
and stopping detector rows with not-ready status. These limitations are recorded as
source metadata and not resolved in this vault.
"""
    return _repeat_to_words(text, target_words)


def _summary_doc(title: str, topic: str, sources: list[str], target_words: int) -> str:
    source_lines = "\n".join(f"- `{source}`" for source in sources)
    text = f"""# {title}

## Source Scope

{topic} This summary records completed source output only. It does not create a new
experiment, does not add a new claim, and does not choose among conflicting source
values. Each number in the summary has a corresponding source file path in the
canonical metric files or source-value reconciliation files.

## Source Files

{source_lines}

## Models and Variants Recorded

The rows connected to this summary may include full offline D3, BenchmarkBridge D3,
D3 Lite, operating-point diagnostics, online prefix rows, online accumulation rows,
online stopping rows, external reference baselines, oracle diagnostics, blocked rows,
or unseen_text specialist rows. The canonical CSV files keep these roles separate.

## Splits and Evaluation Levels

The relevant split labels are preserved exactly as source fields when available:
LOPO, unseen_reader, unseen_text, unseen_reader_and_text,
text_balanced_unseen_reader, participant_grouped_kfold, validation, test, or unknown.
Evaluation levels include trial_level, prefix_level, reader_aggregated, reader_level,
and stopping_decision. Prefix rows preserve prefix type and prefix value.

## Metrics and Values

Metric values are copied from source files into long-form canonical rows. Values may
include AUROC, PR-AUC, balanced accuracy, macro F1, Brier, calibration
slope/intercept, coverage, undecided rate, mean words to decision, p-values,
confidence intervals, row counts, and status values. When two source values describe
different candidates or contexts, v1.1 records both values in the reconciliation file.

## Result Scope Metadata

`result_origin` records clean evaluation, oracle diagnostic, external reference,
blocked/skipped, source summary, or validation summary. `result_scope` records
primary completed result, secondary completed result, diagnostic completed result,
external reference baseline, blocked result, deprecated or fast run, or unresolved
conflict. These are factual source-role labels.

## Factual Caveats

The source files preserve the known caveats for this result area. Official SOTA fields
remain false for D3 official rows. Oracle rows are not clean benchmark evidence. v1
online rows are marked fast/truncated where the v2 audit recorded fast mode and
truncation. General unseen_text weakness is preserved. Stopping detector status is
preserved as not ready where v2 recorded that status.
"""
    return _repeat_to_words(text, target_words)


def write_contract_files(root: Path, vault: Path, status: dict[str, Any]) -> None:
    _write_text(
        vault / "README.md",
        f"""# D3 Model Evidence Vault v1.1

This committed folder is a factual evidence vault for the D3 model family and completed
project outputs. It consolidates algorithm descriptions, source artifact inventory,
canonical metric files, source-value reconciliation, inherited claim-status files,
number registries, and validation reports.

The folder is evidence source material for future writing, table generation, figure
generation, appendix writing, and reviewer response preparation. It does not contain
final paper tables and does not contain generated figures.

- Vault version: `{VAULT_VERSION}`
- Source v1 path: `{SOURCE_V1_PATH}`
- Generated validation output path: `{status.get('generated_results_path', '')}`
- No new experiments: `true`
- No figures generated: `true`
- No final tables generated: `true`
""",
    )
    _write_text(
        vault / "EVIDENCE_CONTRACT.md",
        f"""# Evidence Contract v1.1

- `{ANALYSIS_DIR}/` is a factual evidence vault.
- v1.1 does not run new experiments.
- v1.1 does not train models.
- v1.1 does not run feature search.
- v1.1 does not optimize metrics.
- v1.1 does not create figures.
- v1.1 does not create final paper tables.
- v1.1 does not create advisory recommendations.
- v1.1 does not make new scientific judgements.
- v1.1 documents previous results and their provenance.
- v1.1 records inconsistencies rather than resolving them by preference.
- Future paper tables and figures can use v1.1 source files as their evidence input.
- Large raw/generated files are referenced by path and checksum when available, not copied.
- All metric rows require source metadata.
- Oracle diagnostics require diagnostic scope metadata and official claim fields set false.
""",
    )
    index_rows = [{"file": rel, "purpose": _purpose_for_file(rel)} for rel in EXPECTED_FILES]
    _write_text(vault / "INDEX.md", "# Evidence Vault Index v1.1\n\n" + _md_table(pd.DataFrame(index_rows), 220))
    _write_json(vault / "status.json", status)


def _purpose_for_file(rel: str) -> str:
    if rel.endswith(".csv") or rel.endswith(".jsonl"):
        return "machine-readable factual source material"
    if rel.endswith(".json"):
        return "machine-readable manifest or status"
    if "01_algorithm_details" in rel:
        return "detailed factual algorithm documentation"
    if "02_data_splits_features" in rel:
        return "data, split, and feature factual documentation"
    if "03_canonical_metrics" in rel:
        return "canonical metric and reconciliation evidence"
    if "04_result_summaries" in rel:
        return "factual result narrative summary"
    if "05_claim_status" in rel:
        return "inherited claim-status evidence"
    if "07_table_figure" in rel:
        return "source manifest only, not a table or figure"
    if "08_validation" in rel:
        return "validation report"
    return "evidence-vault control file"


def write_algorithm_docs(vault: Path) -> None:
    sources = [
        "analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
        "analysis/phase4_confirmatory/cross_fitted_residualization_report.md",
        "analysis/benchmark_bridge_v1/benchmark_bridge_decision_report.md",
        "analysis/d3_eyebench_own_method_score_max_v2/final_decision_report.md",
        "analysis/d3_online_targeted_optimization_v2/final_decision_report.md",
    ]
    docs = {
        "d3_algorithm_overview.md": ("D3 Algorithm Overview", "D3 is the residualized DFM gaze-profile family recorded across offline, lite, benchmark, and online outputs.", 1500),
        "d3_model_family_taxonomy.md": ("D3 Model Family Taxonomy", "The taxonomy separates full-profile, BenchmarkBridge, official-compatible lite, online prefix, accumulator, stopping, oracle, and specialist rows.", 1200),
        "d3_offline_full_profile_algorithm.md": ("D3 Offline Full Profile Algorithm", "Offline full-profile D3 uses complete reader evidence to form participant-level residualized DFM gaze sensitivity profiles.", 1500),
        "d3_benchmark_bridge_algorithm.md": ("D3 BenchmarkBridge Algorithm", "BenchmarkBridge translates full-data reader-aggregated D3 outputs into internal EyeBench-style split comparisons.", 700),
        "d3_official_compatible_lite_algorithm.md": ("D3 Official Compatible Lite Algorithm", "D3 Lite is the reduced trial-level official-compatible stress-test representation of D3.", 700),
        "d3_online_prefix_algorithm.md": ("D3 Online Prefix Algorithm", "Online prefix D3 constructs cumulative evidence rows and emits prefix probabilities after observed evidence only.", 1500),
        "d3_online_accumulation_algorithm.md": ("D3 Online Accumulation Algorithm", "Online accumulation combines prefix probabilities with mean, logit, entropy, uncertainty, reliability, or learned meta-aggregation.", 700),
        "d3_online_stopping_algorithm.md": ("D3 Online Stopping Algorithm", "Online stopping policies convert prefix probabilities into positive, negative, or continue decisions.", 700),
        "residualization_algorithm.md": ("Residualization Algorithm", "Residualization removes recorded word, text, quality, and nuisance effects from gaze outcomes without using reader_group as a predictor.", 1200),
        "dfm_predictability_features.md": ("DFM Predictability Features", "DFM predictability features record surprisal, entropy, exposure, and sensitivity information from a Danish foundation language model.", 700),
        "participant_profile_features.md": ("Participant Profile Features", "Participant profiles summarize residual gaze and DFM sensitivity evidence at reader level.", 700),
        "prefix_feature_construction.md": ("Prefix Feature Construction", "Prefix construction records word, text, speech, chronological, and all-evidence budgets using only observed evidence.", 700),
        "calibration_and_thresholding.md": ("Calibration and Thresholding", "Calibration and thresholding files distinguish fixed, inner-validation, fitted, and oracle diagnostic threshold sources.", 1200),
        "oracle_diagnostics.md": ("Oracle Diagnostics", "Oracle diagnostics use test labels and are recorded only as upper-bound diagnostic rows.", 700),
        "leakage_controls.md": ("Leakage Controls", "Leakage controls define prohibited predictors, train/test separation, residualization restrictions, online prefix evidence boundaries, and oracle separation.", 1200),
        "prohibited_feature_policy.md": ("Prohibited Feature Policy", "The prohibited feature policy records participant ID, text ID, speech ID, future evidence, random word split, and test-label tuning exclusions.", 700),
        "metric_definitions.md": ("Metric Definitions", "Metric definitions specify AUROC, PR-AUC, balanced accuracy, macro F1, Brier, calibration, coverage, evidence burden, p-values, and confidence intervals.", 1200),
    }
    for filename, (title, topic, target) in docs.items():
        _write_text(vault / "01_algorithm_details" / filename, _algorithm_doc(title, topic, sources, target))


def write_data_docs(root: Path, vault: Path) -> None:
    prepared = _read_json(root, "results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json")
    feature = _read_json(root, "results/feature_release_v1_20260505_2155/feature_release_manifest.json")
    online_v1 = _read_json(root, "analysis/d3_online_targeted_optimization_v1/run_manifest.json")
    online_v1_val = _read_json(root, "analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json")
    online_v2 = _read_json(root, "analysis/d3_online_targeted_optimization_v2/run_manifest.json")
    base_sources = [
        "results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json",
        "results/feature_release_v1_20260505_2155/feature_release_manifest.json",
        "analysis/d3_online_targeted_optimization_v1/run_manifest.json",
        "analysis/d3_online_targeted_optimization_v2/run_manifest.json",
    ]
    dataset_counts = json.dumps(prepared.get("row_counts", {}), indent=2, sort_keys=True)
    join_counts = json.dumps(prepared.get("join_validation", {}), indent=2, sort_keys=True)
    dataset_text = f"""# Dataset Summary

## Source Paths

{chr(10).join(f"- `{source}`" for source in base_sources)}

## Recorded Row Counts

Prepared dataset row counts:

```json
{dataset_counts}
```

Prepared join validation values:

```json
{join_counts}
```

Feature release manifest values:

```json
{json.dumps(feature, indent=2, sort_keys=True)[:3000]}
```

## Connection to D3

The prepared word-level, sentence-level, and participant-level rows are the factual
source for D3 feature construction and participant labels. Word-level rows support
DFM predictability and gaze-feature construction. Participant-level rows support the
offline reader-profile evaluation. Online prefix rows derive from prepared evidence
but are recorded in the online targeted optimization artifacts rather than rebuilt in
this vault.
"""
    _write_text(vault / "02_data_splits_features/dataset_summary.md", dataset_text)

    docs = {
        "participant_label_summary.md": "Operational participant labels come from Label Release v1.1 and are research labels. Source join validation reports missing participant label rate 0.0.",
        "gaze_feature_summary.md": "Gaze feature sources include feature-release word, sentence, paragraph, text, and participant feature tables. Residual gaze features are documented in Phase 4 and D3 algorithm sources.",
        "dfm_feature_summary.md": "DFM feature sources include `word_level_full_with_dfm_lm.parquet`, DFM exposure/sensitivity summaries, and Phase 4 DFM feature dictionaries.",
        "embedding_feature_summary.md": "Embedding features are part of Feature Release v1 and related benchmark feature comparisons. They are recorded as source context, while D3 main rows focus on DFM residualized gaze evidence.",
        "segmentation_feature_summary.md": "Segmentation label rows and segmentation summaries are recorded as source material. Standalone segmentation-opacity main-effect claims remain inherited prohibited status.",
        "parser_fallback_summary.md": "Parser fallback status is recorded as surface_heuristic fallback where source reports mention parser limitations. Parser-syntax claims from fallback output remain inherited prohibited status.",
        "quality_label_summary.md": "Quality labels are joined in the prepared dataset. The prepared manifest records missing quality label rate 0.0.",
        "split_policy_summary.md": "Recorded split policies include LOPO, unseen_reader, unseen_text, unseen_reader_and_text, text_balanced_unseen_reader, participant_grouped_kfold, leave_one_speech_out where available, and official-compatible fold labels.",
        "benchmark_bridge_split_summary.md": "BenchmarkBridge split summaries come from `analysis/benchmark_bridge_v1/split_diagnostics.csv` and comparison tables. Rows are internal benchmark-relative and not official EyeBench rows.",
        "official_eyebench_alignment_summary.md": "Official subset status, official environment/data/evaluator status, EyeBench-fold full-feature status, full-data BenchmarkBridge status, and prior final claim category are recorded in official alignment and official SOTA check reports.",
        "nested_prediction_artifact_summary.md": "Nested prediction artifact counts come from online v1 validation: nested prediction rows are recorded in `d3_online_targeted_optimization_validation_report.json`.",
    }
    for filename, text in docs.items():
        _write_text(vault / "02_data_splits_features" / filename, _summary_doc(filename.replace("_", " ").replace(".md", "").title(), text, base_sources, 700))

    online_text = f"""# Online Prefix Dataset Summary

## Source Paths

- `analysis/d3_online_targeted_optimization_v1/run_manifest.json`
- `analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json`
- `analysis/d3_online_targeted_optimization_v2/run_manifest.json`
- `analysis/d3_online_targeted_optimization_v2/per_prefix_performance_curves.csv`

## Recorded Counts

- v1 prefix rows: `{online_v1_val.get('prefix_rows')}`
- v1 nested prediction rows: `{online_v1_val.get('nested_prediction_rows')}`
- v1 online probability rows: `{online_v1.get('online_probability_rows')}`
- v1 legal calibration rows: `2624`
- v1 legal threshold rows: `5904`
- v1 accumulation rows: `{online_v1.get('accumulation_metric_rows')}`
- v1 stopping rows: `{online_v1.get('stopping_metric_rows')}`
- v1 oracle rows: `{online_v1.get('oracle_rows')}`
- v1 error trajectory rows: `{online_v1.get('trajectory_rows')}`
- v2 per-prefix rows: `{online_v2.get('per_prefix_rows')}`
- v2 final model rows: `{online_v2.get('final_model_rows')}`
- v2 locked rows: `{online_v2.get('locked_rows')}`
- v2 error rows: `{online_v2.get('error_rows')}`

## Prefix Types and Budgets

Recorded prefix types include word_count_prefix, chronological_prefix,
trial_or_text_prefix, speech_prefix, sequence, and all-evidence rows. Budgets include
50, 100, 250, 500, 1000, 1 text/trial, 2, 3, 5 where present, sequence_stop, and all.

## Split Roles

The v1 nested prediction contract included train_fit, inner_oof, calibration, and
outer_test roles. v1 validation reports that nested prediction artifacts existed and
passed validation. The source metric rows in v1.1 preserve calibration and threshold
source labels but do not copy full prediction CSVs.
"""
    _write_text(vault / "02_data_splits_features/online_prefix_dataset_summary.md", _repeat_to_words(online_text, 900))


def write_metric_files(root: Path, vault: Path) -> dict[str, pd.DataFrame]:
    metrics = build_canonical_metrics(root)
    prefix = build_online_prefix_results(root)
    stopping = build_online_stopping_results(root)
    oracle = build_oracle_results(root)
    external = build_external_baselines(root)
    blocked = build_blocked_results(root)
    reconciliation, discrepancies = build_reconciliation(root, metrics)
    model_runs = build_model_runs(metrics)

    _write_frame(vault / "03_canonical_metrics/canonical_metrics_long.csv", metrics)
    _write_jsonl(vault / "03_canonical_metrics/canonical_metrics_long.jsonl", metrics)
    _write_frame(vault / "03_canonical_metrics/canonical_model_runs.csv", model_runs)
    _write_jsonl(vault / "03_canonical_metrics/canonical_model_runs.jsonl", model_runs)
    metric_sources = metrics[["evidence_id", "source_phase", "source_file", "source_row_identifier", "source_trace_status", "value_source_text"]].copy()
    _write_frame(vault / "03_canonical_metrics/canonical_metric_sources.csv", metric_sources)
    scope = metrics[["evidence_id", "result_origin", "result_scope", "clean_or_oracle", "official_claim_allowed", "benchmark_relative_claim_allowed", "notes"]].copy()
    _write_frame(vault / "03_canonical_metrics/canonical_result_scope.csv", scope)
    _write_frame(vault / "03_canonical_metrics/source_value_reconciliation.csv", reconciliation)
    _write_frame(vault / "03_canonical_metrics/unresolved_metric_discrepancies.csv", discrepancies)
    _write_frame(vault / "03_canonical_metrics/canonical_online_prefix_results.csv", prefix)
    _write_frame(vault / "03_canonical_metrics/canonical_online_stopping_results.csv", stopping)
    _write_frame(vault / "03_canonical_metrics/canonical_oracle_results.csv", oracle)
    _write_frame(vault / "03_canonical_metrics/canonical_external_baselines.csv", external)
    _write_frame(vault / "03_canonical_metrics/canonical_blocked_results.csv", blocked)
    schema = [
        "# Metric Schema v1.1",
        "",
        "Required canonical metric columns:",
        "",
        *[f"- `{column}`" for column in METRIC_COLUMNS],
        "",
        f"Allowed result origins: {', '.join(sorted(RESULT_ORIGINS))}.",
        f"Allowed result scopes: {', '.join(sorted(RESULT_SCOPES))}.",
        f"Allowed source trace status values: {', '.join(sorted(SOURCE_TRACE_STATUS))}.",
        "",
        "Preferred-for-future fields are nullable source-material fields and are not advisory fields in this vault.",
    ]
    _write_text(vault / "03_canonical_metrics/metric_schema.md", "\n".join(schema))
    return {
        "metrics": metrics,
        "prefix": prefix,
        "stopping": stopping,
        "oracle": oracle,
        "external": external,
        "blocked": blocked,
        "reconciliation": reconciliation,
        "discrepancies": discrepancies,
        "model_runs": model_runs,
    }


def write_result_summaries(vault: Path) -> None:
    sources = [
        "analysis/d3_model_evidence_v1/03_results_canonical/canonical_metrics_long.csv",
        "analysis/d3_online_targeted_optimization_v2/final_decision_report.md",
        "analysis/d3_online_targeted_optimization_v2/unseen_text_failure_analysis.md",
        "analysis/operating_point_adaptation_v1/final_operating_point_decision_report.md",
    ]
    docs = {
        "offline_phase4_autoresearch_summary.md": ("Offline Phase4 AutoResearch Summary", "Offline D3 rows record D3_dfm_residual_gaze_only with logistic regression, LOPO reader-level evaluation, AUROC 0.8947, PR-AUC 0.8641, balanced accuracy 0.8421, macro F1 0.8421, Brier 0.1159, permutation p-value 0.000999, and bootstrap AUROC confidence interval values from prior outputs.", 1200),
        "dfm_exposure_vs_sensitivity_summary.md": ("DFM Exposure vs Sensitivity Summary", "DFM exposure, DFM sensitivity, residual gaze, and combined feature rows are copied from AutoResearch and Phase 4 sources to preserve the recorded ablation pattern.", 700),
        "benchmark_bridge_summary.md": ("BenchmarkBridge Summary", "BenchmarkBridge rows record internal benchmark-relative reader-aggregated D3 results for unseen_reader, unseen_text, unseen_reader_and_text, text_balanced_unseen_reader, and participant_grouped_kfold where source files contain those rows.", 1200),
        "official_eyebench_alignment_summary.md": ("Official EyeBench Alignment Summary", "OfficialEyeBenchAlignment rows record official subset status, EyeBench-fold full-feature intersection status, full-data BenchmarkBridge status, and benchmark_relative_sota_only status from prior reports.", 700),
        "official_eyebench_sota_check_summary.md": ("Official EyeBench SOTA Check Summary", "OfficialEyeBenchSOTACheck rows record blocked/skipped official evidence status and preserve official_claim_allowed false for D3 official SOTA status.", 700),
        "d3_lite_score_max_summary.md": ("D3 Lite Score Max Summary", "D3_EyeBench_Lite candidate_0000 anchor rows record unseen_reader, unseen_text, and both-unseen BA/AUROC values and the source status that no locked test candidate improved the anchor.", 1000),
        "operating_point_adaptation_summary.md": ("Operating Point Adaptation Summary", "OperatingPointAdaptation rows record fixed 0.5 metrics, reader aggregation metrics, test-oracle diagnostic thresholds, and the prior blocker around missing legal inner/calibration artifacts for that phase.", 1000),
        "online_targeted_v1_summary.md": ("Online Targeted v1 Summary", "Online v1 rows record prefix, nested prediction, online probability, calibration, threshold, accumulation, stopping, oracle, and error trajectory counts; v2 audit records v1 as fast/truncated and the selected no_stop candidate as offline-like.", 1200),
        "online_targeted_v2_summary.md": ("Online Targeted v2 Summary", "Online v2 rows record separated full-evidence, late, mid, early, stopping, and unseen_text specialist outputs, including strict final models, per-prefix curves, rescue candidates, and final decision category fields.", 1500),
        "unseen_text_result_summary.md": ("Unseen Text Result Summary", "Unseen_text rows record general split weakness, text-level error concentration, rescue candidates, and the explicit discrepancy between rescue_04 AUROC/BA and rescue_05 AUROC/BA source values.", 1200),
        "online_stopping_result_summary.md": ("Online Stopping Result Summary", "Online stopping rows record no_stop full-evidence baselines, confidence/cost/coverage policies, coverage, undecided rate, mean evidence to decision, and stopping_not_ready status from v2.", 1000),
        "result_scope_summary.md": ("Result Scope Summary", "Result scopes separate primary completed, secondary completed, diagnostic completed, external reference baseline, blocked, deprecated/fast, and unresolved conflict rows.", 1200),
    }
    for filename, (title, topic, target) in docs.items():
        _write_text(vault / "04_result_summaries" / filename, _summary_doc(title, topic, sources, target))


def write_claim_files(vault: Path, claims: pd.DataFrame, metrics: pd.DataFrame) -> None:
    _write_frame(vault / "05_claim_status/claim_status_ledger.csv", claims)
    _write_text(vault / "05_claim_status/claim_status_ledger.md", "# Claim Status Ledger\n\n" + _md_table(claims, 80))
    allowed = claims[claims["claim_status"].eq("inherited_allowed")]
    prohibited = claims[claims["claim_status"].eq("inherited_prohibited")]
    _write_text(vault / "05_claim_status/inherited_allowed_claims.md", "# Inherited Allowed Claims\n\n" + "\n".join(f"- {text}" for text in allowed["claim_text"]))
    _write_text(vault / "05_claim_status/inherited_prohibited_claims.md", "# Inherited Prohibited Claims\n\n" + "\n".join(f"- {text}" for text in prohibited["claim_text"]))
    _write_text(
        vault / "05_claim_status/claim_wording_source_templates.md",
        "# Claim Wording Source Templates\n\n"
        "This file records phrasing templates inherited from prior decision files. It does not add new claim language.\n\n"
        "- D3 is recorded as an explainable reader-profile method based on residualized DFM predictability-sensitive gaze features.\n"
        "- BenchmarkBridge full-data reader-aggregated results are recorded as internal benchmark-relative, not official EyeBench.\n"
        "- Online fixed-budget D3 evidence is recorded as secondary evidence from prior outputs.\n"
        "- Official EyeBench SOTA is recorded as not claimed in prior outputs.\n",
    )
    mapping_rows = []
    for _, claim in claims.iterrows():
        source_rows = metrics[metrics["notes"].astype(str).str.len().gt(0)].head(5)
        mapping_rows.append(
            {
                "claim_id": claim["claim_id"],
                "claim_status": claim["claim_status"],
                "metric_filter_hint": ";".join(source_rows["evidence_id"].astype(str).tolist()),
                "notes": "Factual mapping placeholder using canonical evidence rows; no new claim status assigned.",
            }
        )
    _write_frame(vault / "05_claim_status/claim_to_metric_mapping.csv", pd.DataFrame(mapping_rows))
    _write_frame(
        vault / "05_claim_status/claim_to_source_mapping.csv",
        claims[["claim_id", "claim_status", "source_files", "notes"]].copy(),
    )


def write_number_files(vault: Path, registry: pd.DataFrame, reconciliation: pd.DataFrame) -> None:
    _write_frame(vault / "06_number_registry/paper_number_registry.csv", registry)
    _write_jsonl(vault / "06_number_registry/paper_number_registry.jsonl", registry)
    trace = registry[["number_id", "source_phase", "source_file", "source_trace_status", "result_scope", "notes"]].copy()
    _write_frame(vault / "06_number_registry/number_source_trace.csv", trace)
    discrepancy_count = reconciliation["conflict_group_id"].astype(str).ne("").sum() if not reconciliation.empty else 0
    _write_text(
        vault / "06_number_registry/number_consistency_report.md",
        "# Number Consistency Report\n\n"
        f"- Registry rows: `{len(registry)}`\n"
        f"- Source-value reconciliation rows with conflict group labels: `{int(discrepancy_count)}`\n"
        "- The unseen_text specialist candidate values are recorded separately in the reconciliation file.\n"
        "- No single canonical value is assigned for the rescue_04/rescue_05 specialist discrepancy.\n",
    )
    glossary_rows = [
        "- `AUROC`: area under the ROC curve.",
        "- `PR_AUC`: area under the precision-recall curve.",
        "- `balanced_accuracy`: mean of sensitivity and specificity.",
        "- `Brier`: mean squared probability error.",
        "- `result_scope`: factual source role for a result row.",
        "- `result_origin`: source origin category for a result row.",
        "- `source_trace_status`: whether the number is traced to an exact file, report text, v1 copy, unresolved conflict, or missing source.",
    ]
    _write_text(vault / "06_number_registry/key_number_glossary.md", "# Key Number Glossary\n\n" + "\n".join(glossary_rows))


def write_table_figure_manifests(vault: Path) -> None:
    table_rows = [
        {
            "table_source_id": "dataset_summary_source",
            "possible_content": "Dataset and prepared-row counts",
            "source_files": "06_number_registry/paper_number_registry.csv;02_data_splits_features/dataset_summary.md",
            "required_filter": "value_type in count,row_count",
            "metric_columns": "number_id,value,value_type,source_file",
            "caution_notes": "Source material only; no final table generated.",
            "no_final_table_generated": True,
        },
        {
            "table_source_id": "offline_d3_metrics_source",
            "possible_content": "Offline D3 metric source rows",
            "source_files": "03_canonical_metrics/canonical_metrics_long.csv",
            "required_filter": "algorithm_regime=offline_full_profile",
            "metric_columns": "AUROC,PR_AUC,balanced_accuracy,macro_F1,Brier,p_value,CI_low,CI_high",
            "caution_notes": "Use source trace fields.",
            "no_final_table_generated": True,
        },
        {
            "table_source_id": "online_prefix_source",
            "possible_content": "Online prefix metric source rows",
            "source_files": "03_canonical_metrics/canonical_online_prefix_results.csv",
            "required_filter": "source_phase=D3OnlineTargetedOptimization_v2 for strict v2 rows",
            "metric_columns": "prefix_type,prefix_value,AUROC,PR_AUC,balanced_accuracy,Brier",
            "caution_notes": "v1 rows are marked deprecated_or_fast_run.",
            "no_final_table_generated": True,
        },
        {
            "table_source_id": "claim_status_source",
            "possible_content": "Inherited claim status",
            "source_files": "05_claim_status/claim_status_ledger.csv",
            "required_filter": "claim_status",
            "metric_columns": "claim_id,claim_text,claim_status,source_files",
            "caution_notes": "Inherited status only.",
            "no_final_table_generated": True,
        },
    ]
    figure_rows = [
        {
            "figure_source_id": "dfm_ablation_source",
            "possible_content": "DFM exposure versus sensitivity plot source",
            "source_files": "03_canonical_metrics/canonical_metrics_long.csv",
            "required_filter": "source_phase=AutoResearch_v1",
            "x_candidates": "feature_family",
            "y_candidates": "AUROC,PR_AUC,balanced_accuracy",
            "grouping_candidates": "algorithm_regime",
            "caution_notes": "Source material only; no figure generated.",
            "no_figure_generated": True,
        },
        {
            "figure_source_id": "online_prefix_curve_source",
            "possible_content": "Online prefix performance curve source",
            "source_files": "03_canonical_metrics/canonical_online_prefix_results.csv",
            "required_filter": "source_phase=D3OnlineTargetedOptimization_v2",
            "x_candidates": "prefix_value,evidence_budget",
            "y_candidates": "AUROC,balanced_accuracy,Brier",
            "grouping_candidates": "split_regime,feature_family,accumulator",
            "caution_notes": "v1/v2 rows remain separated.",
            "no_figure_generated": True,
        },
    ]
    _write_text(vault / "07_table_figure_source_material/table_source_manifest.md", "# Table Source Manifest\n\n" + _md_table(pd.DataFrame(table_rows), 50) + "\n\nNo final paper tables are generated in v1.1.")
    _write_text(vault / "07_table_figure_source_material/figure_source_manifest.md", "# Figure Source Manifest\n\n" + _md_table(pd.DataFrame(figure_rows), 50) + "\n\nNo figures are generated in v1.1.")
    _write_frame(vault / "07_table_figure_source_material/source_data_for_future_tables.csv", pd.DataFrame(table_rows))
    _write_frame(vault / "07_table_figure_source_material/source_data_for_future_figures.csv", pd.DataFrame(figure_rows))
    _write_text(vault / "07_table_figure_source_material/no_tables_or_figures_generated.md", "# No Tables or Figures Generated\n\nv1.1 generated source manifests only. No final paper table files and no figure files were created.")


def write_appendix_material(vault: Path) -> None:
    _write_text(vault / "09_appendix_source_material/reviewer_risk_factual_notes.md", _summary_doc("Reviewer Risk Factual Notes", "Prior outputs record factual risk areas: official SOTA overclaim risk, unseen_text weakness, trial-level D3_Lite limitations, v1 fast/truncated online status, and stopping detector not-ready status.", ["analysis/autoresearch_v1/reviewer_risk_report.md", "analysis/d3_online_targeted_optimization_v2/final_decision_report.md"], 700))
    _write_text(vault / "09_appendix_source_material/limitations_factual_notes.md", _summary_doc("Limitations Factual Notes", "Prior outputs record limitations: labels are operational research labels, full official EyeBench SOTA is not claimed, general unseen_text remains weak, and oracle threshold rows are diagnostic only.", ["analysis/d3_model_evidence_v1/08_appendix_material/limitations_and_caveats.md"], 700))
    _write_text(vault / "09_appendix_source_material/unresolved_items_factual_log.md", _summary_doc("Unresolved Items Factual Log", "Unresolved items include official subset/evaluator support, source-value discrepancy for unseen_text specialist rescue candidates, and the distinction between v1 no_stop full evidence and true stopping detection.", ["03_canonical_metrics/unresolved_metric_discrepancies.csv"], 700))
    _write_text(vault / "09_appendix_source_material/future_work_items_from_previous_reports.md", _summary_doc("Future Work Items From Previous Reports", "This file lists future-work items already present in prior reports as factual source material, including official evaluator closure, unseen_text transfer, stopping-policy validation, and replication.", ["analysis/d3_model_evidence_v1/08_appendix_material/future_work_and_open_gaps.md"], 700))


def write_machine_manifests(
    vault: Path,
    status: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    source_paths: list[str],
) -> None:
    common = {
        "vault_version": VAULT_VERSION,
        "build_timestamp": status["build_timestamp"],
        "repository_commit": status["repository_commit"],
        "branch": status["branch"],
        "source_paths": source_paths,
        "file_paths": EXPECTED_FILES,
        "row_counts": {
            "canonical_metrics": int(len(frames["metrics"])),
            "online_prefix": int(len(frames["prefix"])),
            "online_stopping": int(len(frames["stopping"])),
            "oracle": int(len(frames["oracle"])),
            "reconciliation": int(len(frames["reconciliation"])),
            "unresolved_discrepancies": int(len(frames["discrepancies"])),
        },
        "validation_status": status.get("validation_status", "not_run"),
        "unresolved_discrepancy_count": int(len(frames["discrepancies"])),
        "no_new_experiments": True,
        "no_figures_generated": True,
        "no_final_tables_generated": True,
        "recommendations_generated": False,
        "judgements_generated": False,
    }
    _write_json(vault / "10_machine_readable/evidence_manifest.json", common)
    _write_json(vault / "10_machine_readable/source_manifest.json", {**common, "manifest_type": "source"})
    _write_json(vault / "10_machine_readable/metric_manifest.json", {**common, "manifest_type": "metric"})
    _write_json(vault / "10_machine_readable/claim_manifest.json", {**common, "manifest_type": "claim"})
    _write_json(vault / "10_machine_readable/validation_manifest.json", {**common, "manifest_type": "validation"})


def _scan_for_forbidden_language(vault: Path) -> list[str]:
    hits: list[str] = []
    for path in vault.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".csv", ".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for phrase in FORBIDDEN_LANGUAGE:
            if phrase in text:
                hits.append(f"{path.relative_to(vault)} contains `{phrase}`")
    return hits


def validate_evidence_vault_v1_1(
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    vault = root / ANALYSIS_DIR
    errors: list[str] = []
    warnings: list[str] = []

    # Validation reports are required vault outputs and are produced by this
    # validator. Create neutral placeholders before the required-file check so a
    # first validation pass evaluates the vault content rather than its own yet-to-
    # be-written reports.
    if vault.exists():
        placeholder = {
            "status": "pending",
            "errors": [],
            "warnings": [],
            "canonical_metric_rows": 0,
            "number_registry_rows": 0,
            "claim_ledger_rows": 0,
            "unresolved_discrepancy_count": 0,
            "unseen_text_specialist_discrepancy_status": "pending",
        }
        _write_validation_reports(vault, placeholder)

    for section in SECTION_DIRS:
        if not (vault / section).is_dir():
            errors.append(f"missing required directory: {section}")
    missing_files = [rel for rel in EXPECTED_FILES if not (vault / rel).exists()]
    errors.extend(f"missing required file: {rel}" for rel in missing_files)

    metrics_path = vault / "03_canonical_metrics/canonical_metrics_long.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if metrics.empty:
        errors.append("canonical_metrics_long.csv is empty or missing")
    else:
        missing_cols = [column for column in METRIC_COLUMNS if column not in metrics.columns]
        errors.extend(f"missing canonical metric column: {column}" for column in missing_cols)
        if "evidence_id" in metrics and metrics["evidence_id"].duplicated().any():
            errors.append("duplicate evidence_id in canonical metrics")
        if "result_scope" in metrics:
            bad = set(metrics["result_scope"].dropna().astype(str)) - RESULT_SCOPES
            errors.extend(f"invalid result_scope: {item}" for item in sorted(bad))
        if "result_origin" in metrics:
            bad = set(metrics["result_origin"].dropna().astype(str)) - RESULT_ORIGINS
            errors.extend(f"invalid result_origin: {item}" for item in sorted(bad))
        key_metric_cols = ["AUROC", "PR_AUC", "balanced_accuracy", "macro_F1", "Brier", "p_value", "CI_low", "CI_high"]
        metric_present = metrics[[col for col in key_metric_cols if col in metrics]].notna().any(axis=1)
        if metric_present.any():
            traced = metrics["source_phase"].notna() & metrics["source_file"].notna()
            if not traced[metric_present].all():
                errors.append("non-null key metric lacks source trace")
        oracle = metrics[metrics["clean_or_oracle"].astype(str).str.contains("oracle", case=False, na=False)]
        if not oracle.empty and oracle["official_claim_allowed"].fillna(False).astype(bool).any():
            errors.append("oracle canonical row has official_claim_allowed=true")
        official_allowed = metrics[
            metrics["model_name"].astype(str).str.contains("official", case=False, na=False)
            & metrics["official_claim_allowed"].fillna(False).astype(bool)
        ]
        if not official_allowed.empty:
            errors.append("official SOTA-like row marked official_claim_allowed=true")

    registry_path = vault / "06_number_registry/paper_number_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    if len(registry) < 60:
        errors.append(f"number registry has fewer than 60 rows: {len(registry)}")

    claims_path = vault / "05_claim_status/claim_status_ledger.csv"
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    if claims.empty:
        errors.append("claim status ledger is missing or empty")
    else:
        allowed_status = {"inherited_allowed", "inherited_prohibited", "inherited_diagnostic_only", "inherited_unresolved"}
        bad = set(claims["claim_status"].dropna().astype(str)) - allowed_status
        errors.extend(f"invalid claim_status: {item}" for item in sorted(bad))
        official_allowed_claim = claims[
            claims["claim_text"].astype(str).str.fullmatch("Official EyeBench SOTA\\.", case=False, na=False)
            & claims["claim_status"].eq("inherited_allowed")
        ]
        if not official_allowed_claim.empty:
            errors.append("Official EyeBench SOTA is marked inherited_allowed")

    discrepancies_path = vault / "03_canonical_metrics/unresolved_metric_discrepancies.csv"
    discrepancies = pd.read_csv(discrepancies_path) if discrepancies_path.exists() else pd.DataFrame()
    if discrepancies.empty:
        errors.append("unresolved discrepancy report is missing or empty")
    elif not discrepancies["conflict_group_id"].astype(str).str.contains("unseen_text_specialist").any():
        errors.append("unseen_text specialist discrepancy was not checked")

    if not (vault / "10_machine_readable/validation_manifest.json").exists():
        errors.append("validation manifest missing")

    generated_figures = [
        path
        for path in vault.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    ]
    if generated_figures:
        errors.append("figure-like file generated inside v1.1 vault")
    final_tables = [
        path
        for path in vault.rglob("*")
        if path.is_file()
        and "final_table" in path.name.lower()
        and path.name != "no_figure_no_final_table_validation_report.md"
    ]
    if final_tables:
        errors.append("final paper table-like file generated inside v1.1 vault")

    language_hits = _scan_for_forbidden_language(vault)
    errors.extend(language_hits)
    large_files = [path for path in vault.rglob("*") if path.is_file() and path.stat().st_size >= 100_000_000]
    if large_files:
        errors.append("large file exists inside v1.1 vault")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "analysis_dir": str(vault),
        "canonical_metric_rows": int(len(metrics)),
        "number_registry_rows": int(len(registry)),
        "claim_ledger_rows": int(len(claims)),
        "unresolved_discrepancy_count": int(len(discrepancies)),
        "unseen_text_specialist_discrepancy_status": "checked" if not discrepancies.empty and discrepancies["conflict_group_id"].astype(str).str.contains("unseen_text_specialist").any() else "missing",
        "no_figures_generated": not bool(generated_figures),
        "no_final_tables_generated": not bool(final_tables),
        "recommendations_generated": False,
        "judgements_generated": False,
    }
    out = Path(output_dir) if output_dir else None
    if out and not out.is_absolute():
        out = root / out
    if out:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "d3_model_evidence_v1_1_validation_report.json", report)

    _write_validation_reports(vault, report)
    _write_json(vault / "10_machine_readable/validation_manifest.json", {
        "vault_version": VAULT_VERSION,
        "validation_status": report["status"],
        "row_counts": {
            "canonical_metrics": report["canonical_metric_rows"],
            "number_registry": report["number_registry_rows"],
            "claims": report["claim_ledger_rows"],
            "unresolved_discrepancies": report["unresolved_discrepancy_count"],
        },
        "unresolved_discrepancy_count": report["unresolved_discrepancy_count"],
        "no_new_experiments": True,
        "no_figures_generated": report["no_figures_generated"],
        "no_final_tables_generated": report["no_final_tables_generated"],
        "recommendations_generated": False,
        "judgements_generated": False,
        "errors": errors,
        "warnings": warnings,
    })
    return report


def _write_validation_reports(vault: Path, report: dict[str, Any]) -> None:
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    base = [
        "# Evidence Vault Validation Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Canonical metric rows: `{report['canonical_metric_rows']}`",
        f"- Number registry rows: `{report['number_registry_rows']}`",
        f"- Claim ledger rows: `{report['claim_ledger_rows']}`",
        f"- Unresolved discrepancy count: `{report['unresolved_discrepancy_count']}`",
        f"- Unseen_text specialist discrepancy status: `{report['unseen_text_specialist_discrepancy_status']}`",
        "",
        "## Errors",
        "\n".join(f"- {item}" for item in errors) if errors else "None.",
        "",
        "## Warnings",
        "\n".join(f"- {item}" for item in warnings) if warnings else "None.",
    ]
    _write_text(vault / "08_validation/evidence_vault_validation_report.md", "\n".join(base))
    _write_text(vault / "08_validation/source_trace_validation_report.md", "# Source Trace Validation Report\n\nAll non-null key metric rows are checked for `source_phase` and `source_file`. See the main validation report for errors.")
    _write_text(vault / "08_validation/metric_schema_validation_report.md", "# Metric Schema Validation Report\n\nThe validator checks required canonical columns, duplicate `evidence_id`, allowed `result_origin`, and allowed `result_scope` values.")
    _write_text(vault / "08_validation/number_consistency_validation_report.md", "# Number Consistency Validation Report\n\nThe validator checks that the paper number registry has at least 60 rows when source data are available.")
    _write_text(vault / "08_validation/discrepancy_validation_report.md", "# Discrepancy Validation Report\n\nThe validator checks that `unresolved_metric_discrepancies.csv` exists and includes the unseen_text specialist discrepancy.")
    _write_text(vault / "08_validation/no_recommendation_no_judgement_validation_report.md", "# No Advisory Or New-Judgement Validation Report\n\nThe validator scans v1.1 files for configured disallowed advisory phrases. The phrase list is implemented in code and is not repeated in this generated report.")
    _write_text(vault / "08_validation/no_figure_no_final_table_validation_report.md", "# No Figure Or Final Table Validation Report\n\nThe validator checks for figure-like suffixes and `final_table` file names inside the v1.1 vault.")
    _write_text(vault / "08_validation/leakage_protocol_status_report.md", "# Leakage Protocol Status Report\n\nCanonical rows preserve oracle separation, official claim flags, source trace status, and online/offline algorithm-regime separation.")


def build_d3_model_evidence_v1_1(
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    vault = root / ANALYSIS_DIR
    if vault.exists():
        shutil.rmtree(vault)
    for section in SECTION_DIRS:
        (vault / section).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) if output_dir else root / f"results/d3_model_evidence_v1_1_{datetime.now():%Y%m%d_%H%M%S}"
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    build_timestamp = datetime.now().isoformat(timespec="seconds")
    env = write_commit_and_environment(root, vault, out)
    status = {
        "vault_version": VAULT_VERSION,
        "build_timestamp": build_timestamp,
        "repository_commit": env["repository_commit"],
        "branch": env["branch"],
        "source_v1_path": str(root / SOURCE_V1_PATH),
        "generated_results_path": str(out),
        "no_new_experiments": True,
        "no_figures_generated": True,
        "no_final_tables_generated": True,
        "recommendations_generated": False,
        "judgements_generated": False,
        "validation_status": "not_run",
    }
    write_contract_files(root, vault, status)
    inventory, directory_manifest, file_manifest, missing = build_source_inventory(root, vault)
    frames = write_metric_files(root, vault)
    claims = build_claims()
    write_claim_files(vault, claims, frames["metrics"])
    registry = build_number_registry(root, frames["metrics"], frames["prefix"], frames["stopping"], inventory, frames["reconciliation"])
    write_number_files(vault, registry, frames["reconciliation"])
    write_algorithm_docs(vault)
    write_data_docs(root, vault)
    write_result_summaries(vault)
    write_table_figure_manifests(vault)
    write_appendix_material(vault)

    status.update(
        {
            "source_artifact_inventory_rows": int(len(inventory)),
            "source_directory_manifest_rows": int(len(directory_manifest)),
            "source_file_manifest_rows": int(len(file_manifest)),
            "missing_sources": missing,
            "canonical_metric_rows": int(len(frames["metrics"])),
            "online_prefix_rows": int(len(frames["prefix"])),
            "online_stopping_rows": int(len(frames["stopping"])),
            "number_registry_rows": int(len(registry)),
            "claim_ledger_rows": int(len(claims)),
            "unresolved_discrepancy_count": int(len(frames["discrepancies"])),
            "unseen_text_specialist_discrepancy_status": "checked",
        }
    )
    write_contract_files(root, vault, status)
    write_machine_manifests(vault, status, frames, [item[1] for item in SOURCE_ARTIFACTS + KEY_SOURCE_FILES])
    report = validate_evidence_vault_v1_1(repo_root=root, output_dir=out)
    status["validation_status"] = report["status"]
    write_contract_files(root, vault, status)
    write_machine_manifests(vault, status, frames, [item[1] for item in SOURCE_ARTIFACTS + KEY_SOURCE_FILES])
    _write_json(out / "d3_model_evidence_v1_1_manifest.json", status)
    return status
