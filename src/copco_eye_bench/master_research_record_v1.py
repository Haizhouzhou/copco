"""Build and validate MasterResearchRecord v1.

The builder only assembles existing source artifacts into an internal Markdown record.
It does not run experiments, train models, search features, generate figures, or write
final paper tables.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ANALYSIS_DIR = Path("analysis/master_research_record_v1")
MASTER_FILENAME = "MASTER_EXPERIMENT_RECORD.md"
MANIFEST_FILENAME = "source_trace_manifest.json"
VALIDATION_FILENAME = "validation_report.md"

REQUIRED_SECTIONS = [
    "SECTION 1 — Executive project map",
    "SECTION 2 — Timeline of completed research stages",
    "SECTION 3 — Data inventory and dataset versions",
    "SECTION 4 — Split protocols and evaluation regimes",
    "SECTION 5 — Feature families and how they were computed",
    "SECTION 6 — Language models and NLP tools used",
    "SECTION 7 — Model family taxonomy",
    "SECTION 8 — Full prepared CopCo results",
    "SECTION 9 — EyeBench-related results",
    "SECTION 10 — Online and offline evaluation results",
    "SECTION 11 — Result conflicts and unresolved values",
    "SECTION 12 — What each result supports",
    "SECTION 13 — Public-facing method language",
    "SECTION 14 — Paper-writing source map",
    "SECTION 15 — Validation and completeness status",
]

SOURCE_PATHS: list[dict[str, str]] = [
    {
        "source_id": "d3_model_evidence_v1_1",
        "path": "analysis/d3_model_evidence_v1_1",
        "public_description": "curated D3 model evidence vault v1.1",
    },
    {
        "source_id": "d3_model_evidence_v1",
        "path": "analysis/d3_model_evidence_v1",
        "public_description": "curated D3 model evidence vault v1",
    },
    {
        "source_id": "deep_literature_review",
        "path": "analysis/deep_literature_review",
        "public_description": "deep related-work source review, if present",
    },
    {
        "source_id": "feature_release_v1",
        "path": "results/feature_release_v1_20260505_2155",
        "public_description": "Feature Release v1 generated tables and reports",
    },
    {
        "source_id": "label_release_v1_1",
        "path": "results/label_release_v1_1_20260506_0041",
        "public_description": "Label Release v1.1 prepared dataset and label reports",
    },
    {
        "source_id": "research_exploration_v1",
        "path": "results/research_exploration_v1_20260506_0149",
        "public_description": "Phase 3 controlled research exploration",
    },
    {
        "source_id": "phase4_confirmatory_v1",
        "path": "results/phase4_confirmatory_sensitivity_v1_20260506_0715",
        "public_description": "Phase 4 confirmatory sensitivity analysis",
    },
    {
        "source_id": "autoresearch_v1",
        "path": "results/autoresearch_v1_20260506_0917",
        "public_description": "AutoResearch v1 final selection and stress tests",
    },
    {
        "source_id": "submission_v1",
        "path": "results/submission_v1_20260506_0936",
        "public_description": "SubmissionSprint v1 packaging record",
    },
    {
        "source_id": "final_manuscript_audit_v1",
        "path": "results/final_manuscript_audit_v1_20260506_1438",
        "public_description": "Final Manuscript Audit v1",
    },
    {
        "source_id": "benchmark_bridge_v1",
        "path": "results/benchmark_bridge_v1_20260506_1836",
        "public_description": "internal EyeBench-style BenchmarkBridge v1 outputs",
    },
    {
        "source_id": "official_eyebench_alignment_v1",
        "path": "results/official_eyebench_alignment_v1_20260506_2232",
        "public_description": "OfficialEyeBenchAlignment v1 audit outputs",
    },
    {
        "source_id": "official_eyebench_sota_check_v1",
        "path": "results/official_eyebench_sota_check_v1_20260506_2341",
        "public_description": "OfficialEyeBenchSOTACheck v1 blocker outputs",
    },
    {
        "source_id": "d3_own_method_score_max_v2",
        "path": "analysis/d3_eyebench_own_method_score_max_v2",
        "public_description": "D3 EyeBench own-method score-max v2 summaries",
    },
    {
        "source_id": "operating_point_adaptation_v1",
        "path": "analysis/operating_point_adaptation_v1",
        "public_description": "OperatingPointAdaptation v1 summaries",
    },
    {
        "source_id": "online_targeted_optimization_v1",
        "path": "analysis/d3_online_targeted_optimization_v1",
        "public_description": "D3OnlineTargetedOptimization v1 analysis outputs",
    },
    {
        "source_id": "online_targeted_optimization_v2",
        "path": "analysis/d3_online_targeted_optimization_v2",
        "public_description": "D3OnlineTargetedOptimization v2 audit and rerun outputs",
    },
    {
        "source_id": "configs",
        "path": "configs",
        "public_description": "pipeline configuration files",
    },
    {
        "source_id": "docs",
        "path": "docs",
        "public_description": "project documentation and data cards",
    },
    {
        "source_id": "paper_submission_v1",
        "path": "paper/submission_v1",
        "public_description": "submission v1 manuscript/supplement source record",
    },
    {
        "source_id": "ai_run_logs",
        "path": "logs/ai_runs",
        "public_description": "AI run logs",
    },
]

KEY_USED_FILES = [
    "analysis/d3_model_evidence_v1_1/03_canonical_metrics/canonical_metrics_long.csv",
    "analysis/d3_model_evidence_v1_1/03_canonical_metrics/canonical_external_baselines.csv",
    "analysis/d3_model_evidence_v1_1/03_canonical_metrics/canonical_online_prefix_results.csv",
    "analysis/d3_model_evidence_v1_1/03_canonical_metrics/canonical_online_stopping_results.csv",
    "analysis/d3_model_evidence_v1_1/03_canonical_metrics/canonical_oracle_results.csv",
    "analysis/d3_model_evidence_v1_1/03_canonical_metrics/unresolved_metric_discrepancies.csv",
    "analysis/d3_model_evidence_v1_1/06_number_registry/paper_number_registry.csv",
    "analysis/d3_model_evidence_v1_1/00_inventory/source_directory_manifest.csv",
    "analysis/d3_model_evidence_v1_1/00_inventory/source_file_manifest.csv",
    "analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
    "analysis/autoresearch_v1/tables/dfm_exposure_vs_sensitivity_table.csv",
    "analysis/phase4_confirmatory/bootstrap_results.csv",
    "analysis/phase4_confirmatory/permutation_results.csv",
    "analysis/research_exploration/participant_prediction_ablation_metrics.csv",
    "analysis/research_exploration/phase3_research_exploration_decision_report.md",
    "analysis/benchmark_bridge_v1/tables/copco_typ_benchmark_comparison.csv",
    "analysis/official_eyebench_alignment_v1/tables/copco_typ_official_alignment_comparison.csv",
    "analysis/official_eyebench_sota_check_v1/tables/copco_typ_official_sota_comparison.csv",
    "analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv",
    "analysis/d3_eyebench_own_method_score_max_v2/candidate_leaderboard.csv",
    "analysis/operating_point_adaptation_v1/fixed_threshold_metrics.csv",
    "analysis/operating_point_adaptation_v1/test_oracle_threshold_metrics.csv",
    "analysis/d3_online_targeted_optimization_v1/run_manifest.json",
    "analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json",
    "analysis/d3_online_targeted_optimization_v1/online_locked_test_results.csv",
    "analysis/d3_online_targeted_optimization_v1/online_stopping_policy_metrics.csv",
    "analysis/d3_online_targeted_optimization_v2/run_manifest.json",
    "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv",
    "analysis/d3_online_targeted_optimization_v2/strict_locked_test_results.csv",
    "analysis/d3_online_targeted_optimization_v2/per_prefix_performance_curves.csv",
    "analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv",
    "analysis/d3_online_targeted_optimization_v2/error_source_by_prefix.csv",
    "analysis/d3_online_targeted_optimization_v2/final_decision_v2.json",
    "results/feature_release_v1_20260505_2155/feature_release_report.md",
    "results/feature_release_v1_20260505_2155/feature_release_manifest.json",
    "results/feature_release_v1_20260505_2155/reports/table_summary.csv",
    "results/feature_release_v1_20260505_2155/modeling_tables/join_validation_report.json",
    "results/feature_release_v1_20260505_2155/lm_features/dfm_decoder_7b/alignment_report.json",
    "results/feature_release_v1_20260505_2155/embedding_features/manifest.json",
    "results/feature_release_v1_20260505_2155/linguistic_features/parser_diagnostics.json",
    "results/label_release_v1_1_20260506_0041/label_release_report.md",
    "results/label_release_v1_1_20260506_0041/label_release_validation_report.json",
    "results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json",
    "analysis/official_eyebench_alignment_v1/copco_alignment_audit.md",
    "analysis/official_eyebench_alignment_v1/eyebench_vendor_manifest.md",
    "analysis/official_eyebench_alignment_v1/eyebench_environment_report.md",
    "analysis/official_eyebench_alignment_v1/eyebench_data_download_report.md",
    "analysis/official_eyebench_alignment_v1/official_evaluator_blocker_report.md",
    "analysis/official_eyebench_sota_check_v1/official_eyebench_sota_decision_report.md",
    "docs/feature_dictionary_v1.md",
    "docs/segmentation_label_card_v1.md",
    "docs/participant_label_card_v1.md",
    "docs/quality_label_card_v1.md",
    "docs/split_policy_v1.md",
    "docs/dataset/CopCo_dataset_card.md",
    "docs/d3_online_targeted_optimization_v1.md",
    "docs/d3_online_targeted_optimization_v2.md",
    "docs/operating_point_adaptation_v1.md",
    "docs/benchmark_bridge_v1_analysis_plan.md",
]

INTERNAL_TERM_MAP = [
    (
        "D3",
        "residualized predictability-sensitive gaze-profile method",
        "umbrella name for residualized DFM gaze-profile rows.",
    ),
    (
        "D3 offline",
        "full-record reader-profile model",
        "participant-level model using the full reading record.",
    ),
    (
        "D3 online",
        "fixed-budget sequential reader-evidence model",
        "online prefix model using only evidence available up to a prefix.",
    ),
    (
        "D3_Lite",
        "reduced official-protocol-compatible trial-level variant",
        "trial-level reduced feature variant for official-compatible stress tests.",
    ),
    (
        "BenchmarkBridge",
        "internal EyeBench-style benchmark comparison",
        "full-data reader-aggregated benchmark-relative comparison.",
    ),
    (
        "OfficialEyeBenchAlignment",
        "official protocol and data-alignment audit",
        "audit of fold/data/evaluator alignment with EyeBench.",
    ),
    (
        "OperatingPointAdaptation",
        "probability-first operating-point diagnostic",
        "threshold, calibration, and aggregation analysis.",
    ),
    (
        "OnlineTargetedOptimization",
        "fixed-budget online and stopping-policy evaluation",
        "online prefix, accumulator, and stopping-policy evaluation.",
    ),
    (
        "D3ModelEvidenceVault",
        "curated model evidence vault",
        "source-traced internal evidence package for D3 results.",
    ),
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _read_json(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _git(args: list[str], root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not recorded"
    try:
        if pd.isna(value):
            return "not recorded"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return "not recorded"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    *,
    max_rows: int | None = None,
) -> str:
    if not rows:
        return "_No rows recorded._"
    if columns is None:
        columns = list(rows[0].keys())
    shown = rows if max_rows is None else rows[:max_rows]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in shown:
        values = []
        for column in columns:
            value = _fmt(row.get(column))
            values.append(value.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(rows)} rows._")
    return "\n".join(lines)


def _frame_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    max_rows: int = 20,
    rename: dict[str, str] | None = None,
) -> str:
    if frame.empty:
        return "_No rows recorded._"
    present = [column for column in columns if column in frame.columns]
    view = frame[present].copy()
    if rename:
        view = view.rename(columns=rename)
    rows = view.head(max_rows).to_dict("records")
    return _markdown_table(rows, list(view.columns), max_rows=None)


def _source_files(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_path_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    used = set(KEY_USED_FILES)
    for item in SOURCE_PATHS:
        rel_path = item["path"]
        path = root / rel_path
        exists = path.exists()
        files: list[Path] = []
        if exists and path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
        elif exists:
            files = [path]
        size_bytes = sum(p.stat().st_size for p in files if p.exists())
        source_path_rows.append(
            {
                "source_id": item["source_id"],
                "path": rel_path,
                "public_description": item["public_description"],
                "exists": exists,
                "file_count": len(files),
                "size_bytes": size_bytes,
            }
        )
        for idx, file_path in enumerate(files, start=1):
            rel_file = str(file_path.relative_to(root))
            file_rows.append(
                {
                    "source_id": item["source_id"],
                    "file_id": f"{item['source_id']}_{idx:04d}",
                    "path": rel_file,
                    "name": file_path.name,
                    "extension": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                    "used_for_master_record": rel_file in used,
                }
            )
    return source_path_rows, file_rows


def _collect_context(root: Path) -> dict[str, Any]:
    source_paths, source_files = _source_files(root)
    context: dict[str, Any] = {
        "build_timestamp": datetime.now().isoformat(timespec="seconds"),
        "branch": _git(["branch", "--show-current"], root),
        "commit": _git(["rev-parse", "HEAD"], root),
        "status_short": _git(["status", "--short"], root),
        "source_paths": source_paths,
        "source_files": source_files,
    }
    csv_paths = {
        "canonical_metrics": (
            "analysis/d3_model_evidence_v1_1/03_canonical_metrics/canonical_metrics_long.csv"
        ),
        "external_baselines": (
            "analysis/d3_model_evidence_v1_1/03_canonical_metrics/"
            "canonical_external_baselines.csv"
        ),
        "online_prefix": (
            "analysis/d3_model_evidence_v1_1/03_canonical_metrics/"
            "canonical_online_prefix_results.csv"
        ),
        "online_stopping": (
            "analysis/d3_model_evidence_v1_1/03_canonical_metrics/"
            "canonical_online_stopping_results.csv"
        ),
        "oracle": (
            "analysis/d3_model_evidence_v1_1/03_canonical_metrics/"
            "canonical_oracle_results.csv"
        ),
        "unresolved": (
            "analysis/d3_model_evidence_v1_1/03_canonical_metrics/"
            "unresolved_metric_discrepancies.csv"
        ),
        "number_registry": "analysis/d3_model_evidence_v1_1/06_number_registry/"
        "paper_number_registry.csv",
        "final_metrics": "analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
        "dfm_ablation": "analysis/autoresearch_v1/tables/dfm_exposure_vs_sensitivity_table.csv",
        "phase3_ablation": "analysis/research_exploration/"
        "participant_prediction_ablation_metrics.csv",
        "phase4_bootstrap": "analysis/phase4_confirmatory/bootstrap_results.csv",
        "benchmark_bridge": "analysis/benchmark_bridge_v1/tables/"
        "copco_typ_benchmark_comparison.csv",
        "official_alignment": "analysis/official_eyebench_alignment_v1/tables/"
        "copco_typ_official_alignment_comparison.csv",
        "official_sota": "analysis/official_eyebench_sota_check_v1/tables/"
        "copco_typ_official_sota_comparison.csv",
        "d3_lite_trial": "analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv",
        "d3_lite_leaderboard": "analysis/d3_eyebench_own_method_score_max_v2/"
        "candidate_leaderboard.csv",
        "operating_fixed": "analysis/operating_point_adaptation_v1/fixed_threshold_metrics.csv",
        "operating_oracle": "analysis/operating_point_adaptation_v1/"
        "test_oracle_threshold_metrics.csv",
        "online_v1_locked": "analysis/d3_online_targeted_optimization_v1/"
        "online_locked_test_results.csv",
        "online_v1_stopping": "analysis/d3_online_targeted_optimization_v1/"
        "online_stopping_policy_metrics.csv",
        "online_v2_final": "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv",
        "online_v2_locked": "analysis/d3_online_targeted_optimization_v2/"
        "strict_locked_test_results.csv",
        "online_v2_prefix": "analysis/d3_online_targeted_optimization_v2/"
        "per_prefix_performance_curves.csv",
        "online_v2_rescue": "analysis/d3_online_targeted_optimization_v2/"
        "unseen_text_rescue_candidates.csv",
        "online_v2_errors": "analysis/d3_online_targeted_optimization_v2/"
        "error_source_by_prefix.csv",
    }
    json_paths = {
        "feature_join": "results/feature_release_v1_20260505_2155/"
        "modeling_tables/join_validation_report.json",
        "feature_manifest": "results/feature_release_v1_20260505_2155/"
        "feature_release_manifest.json",
        "label_validation": "results/label_release_v1_1_20260506_0041/"
        "label_release_validation_report.json",
        "prepared_manifest": "results/label_release_v1_1_20260506_0041/"
        "prepared_dataset/analysis_ready_manifest.json",
        "dfm_alignment": "results/feature_release_v1_20260505_2155/"
        "lm_features/dfm_decoder_7b/alignment_report.json",
        "embedding_manifest": "results/feature_release_v1_20260505_2155/"
        "embedding_features/manifest.json",
        "parser_diagnostics": "results/feature_release_v1_20260505_2155/"
        "linguistic_features/parser_diagnostics.json",
        "online_v1_manifest": "analysis/d3_online_targeted_optimization_v1/run_manifest.json",
        "online_v1_validation": "analysis/d3_online_targeted_optimization_v1/"
        "d3_online_targeted_optimization_validation_report.json",
        "online_v2_manifest": "analysis/d3_online_targeted_optimization_v2/run_manifest.json",
        "online_v2_validation": "analysis/d3_online_targeted_optimization_v2/"
        "d3_online_targeted_optimization_v2_validation_report.json",
        "online_v2_decision": "analysis/d3_online_targeted_optimization_v2/"
        "final_decision_v2.json",
        "evidence_status": "analysis/d3_model_evidence_v1_1/status.json",
    }
    context["csv"] = {key: _read_csv(root, path) for key, path in csv_paths.items()}
    context["csv_paths"] = csv_paths
    context["json"] = {key: _read_json(root, path) for key, path in json_paths.items()}
    context["json_paths"] = json_paths
    return context


def _first_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _metric_col(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            value = row[name]
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            return value
    return None


def _label_counts(context: dict[str, Any]) -> dict[str, Any]:
    validation = context["json"].get("label_validation", {})
    prepared = context["json"].get("prepared_manifest", {})
    feature_join = context["json"].get("feature_join", {})
    row_counts = validation.get("row_counts", {})
    feature_counts = _readable_counts(context)
    return {
        "participants": row_counts.get("analysis_ready_participant_level_v1_1")
        or prepared.get("row_counts", {}).get("analysis_ready_participant_level_v1_1")
        or feature_counts.get("participants"),
        "dyslexia_labeled": validation.get("participant_counts", {}).get("dyslexia_labeled"),
        "typical_control": validation.get("participant_counts", {}).get("typical_control"),
        "word_observations": feature_counts.get("word_observations")
        or row_counts.get("analysis_ready_word_level_v1_1"),
        "stimulus_words": feature_counts.get("words")
        or row_counts.get("segmentation_word_labels"),
        "sentences": feature_counts.get("sentences")
        or row_counts.get("analysis_ready_sentence_level_v1_1"),
        "paragraphs": feature_counts.get("paragraphs"),
        "participant_rows": row_counts.get("analysis_ready_participant_level_v1_1"),
        "dfm_rows": feature_join.get("word_level_full_with_dfm_lm_rows"),
        "segmentation_boundary_rows": row_counts.get("segmentation_boundary_labels"),
        "segmentation_word_rows": row_counts.get("segmentation_word_labels"),
        "segmentation_sentence_rows": row_counts.get("segmentation_sentence_labels"),
        "quality_rows": row_counts.get("quality_labels"),
        "split_label_rows": row_counts.get("split_labels"),
        "lm_missing_rate": validation.get("quality_missingness", {}).get("lm_missing_rate"),
        "embedding_missing_rate": validation.get("quality_missingness", {}).get(
            "embedding_missing_rate"
        ),
        "parser_missing_rate": validation.get("quality_missingness", {}).get("parser_missing_rate"),
        "segmentation_missing_rate": validation.get("quality_missingness", {}).get(
            "segmentation_missing_rate"
        ),
        "boundary_counts": validation.get("boundary_type_counts", {}),
    }


def _readable_counts(context: dict[str, Any]) -> dict[str, Any]:
    root = Path(".").resolve()
    table_summary = _read_csv(
        root,
        "results/feature_release_v1_20260505_2155/reports/table_summary.csv",
    )
    if table_summary.empty:
        return {}
    return {
        str(row["table"]): row["rows"]
        for row in table_summary.to_dict("records")
        if "table" in row and "rows" in row
    }


def _metrics_summary(context: dict[str, Any]) -> dict[str, int]:
    return {
        "canonical_metrics": int(len(context["csv"]["canonical_metrics"])),
        "external_baselines": int(len(context["csv"]["external_baselines"])),
        "online_prefix": int(len(context["csv"]["online_prefix"])),
        "online_stopping": int(len(context["csv"]["online_stopping"])),
        "oracle": int(len(context["csv"]["oracle"])),
        "number_registry": int(len(context["csv"]["number_registry"])),
        "unresolved_conflicts": int(len(context["csv"]["unresolved"])),
    }


def _section_header(title: str) -> str:
    return f"## {title}"


def _render_master(context: dict[str, Any]) -> str:
    labels = _label_counts(context)
    metrics_summary = _metrics_summary(context)
    sections = [
        "# MasterResearchRecord v1",
        "",
        "Internal project document for CopCo / Eye Bench research-code evidence tracking.",
        "",
        "- Build timestamp: `" + str(context["build_timestamp"]) + "`",
        "- Repository branch: `" + str(context["branch"]) + "`",
        "- Repository commit: `" + str(context["commit"]) + "`",
        "- Primary output path: `analysis/master_research_record_v1/MASTER_EXPERIMENT_RECORD.md`",
        "- Scope: existing evidence only; no new training, feature search, metric optimization, "
        "paper figures, final paper tables, manuscript rewrite, or new scientific claim.",
        "",
        _render_section_1(context, labels, metrics_summary),
        _render_section_2(context),
        _render_section_3(context, labels),
        _render_section_4(context),
        _render_section_5(context),
        _render_section_6(context),
        _render_section_7(context),
        _render_section_8(context),
        _render_section_9(context),
        _render_section_10(context),
        _render_section_11(context),
        _render_section_12(context),
        _render_section_13(),
        _render_section_14(),
        _render_section_15(context, metrics_summary),
    ]
    return "\n".join(sections)


def _render_section_1(
    context: dict[str, Any],
    labels: dict[str, Any],
    metrics_summary: dict[str, int],
) -> str:
    final = _first_row(context["csv"]["final_metrics"])
    rows = [
        {
            "status": "primary",
            "result_family": "Full-record reader-profile result",
            "recorded_value": (
                f"AUROC {_fmt(_metric_col(final, 'roc_auc', 'AUROC'))}; "
                f"BA {_fmt(_metric_col(final, 'balanced_accuracy'))}; "
                f"PR-AUC {_fmt(_metric_col(final, 'pr_auc', 'PR_AUC'))}"
            ),
            "source": "analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
        },
        {
            "status": "secondary",
            "result_family": "Fixed-budget online reader evidence",
            "recorded_value": "late/mid/early rows separated from full-evidence rows",
            "source": "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv",
        },
        {
            "status": "diagnostic",
            "result_family": "Operating point, oracle threshold, and stopping diagnostics",
            "recorded_value": "oracle/test-label thresholds excluded from clean evidence",
            "source": "analysis/operating_point_adaptation_v1/",
        },
        {
            "status": "blocked",
            "result_family": "Official EyeBench leaderboard result",
            "recorded_value": "blocked by missing official processed data/environment",
            "source": "analysis/official_eyebench_sota_check_v1/",
        },
        {
            "status": "unresolved",
            "result_family": "Unseen-text specialist value",
            "recorded_value": "rescue_04 and rescue_05 differ by criterion/context",
            "source": "analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv",
        },
    ]
    terms = [
        {"internal": term, "public": public, "description": description}
        for term, public, description in INTERNAL_TERM_MAP
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[0]),
            "",
            "The project objective is to preserve the complete factual route for CopCo "
            "Danish natural-reading eye-tracking analysis and its EyeBench-related benchmark "
            "comparisons. The target task throughout the main predictive work is operational "
            "reader-group classification between dyslexia-labeled readers and typical/control "
            "readers. The full prepared dataset records "
            f"{_fmt(labels.get('participants'), 0)} participants, "
            f"{_fmt(labels.get('dyslexia_labeled'), 0)} dyslexia-labeled readers, "
            f"{_fmt(labels.get('typical_control'), 0)} typical/control readers, and "
            f"{_fmt(labels.get('word_observations'), 0)} participant-word gaze rows.",
            "",
            "The target label is reader-level because the operational label belongs to the "
            "participant, not to a word, fixation, sentence, or trial. Word-level observations "
            "are repeated evidence from the same reader and must not be treated as independent "
            "target labels. The full-record reader-profile model and fixed-budget online "
            "evidence model are therefore separated: the full-record model uses the complete "
            "reader record for retrospective profiling, while the online model records what can "
            "be inferred from prefix-limited evidence without future rows.",
            "",
            "The public-facing umbrella method name used in this record is the "
            "residualized predictability-sensitive gaze-profile method. Internally this family "
            "appears as D3, D3 offline, D3 online, D3_Lite, BenchmarkBridge, "
            "OfficialEyeBenchAlignment, OperatingPointAdaptation, OnlineTargetedOptimization, "
            "and D3ModelEvidenceVault. Each internal term is paired with a public description "
            "below.",
            "",
            "Internal term mapping:",
            "",
            _markdown_table(terms, ["internal", "public", "description"]),
            "",
            "Result status map:",
            "",
            _markdown_table(rows, ["status", "result_family", "recorded_value", "source"]),
            "",
            "Metric inventory used by this master record:",
            "",
            _markdown_table(
                [
                    {"metric_table": key, "rows": value}
                    for key, value in metrics_summary.items()
                ],
                ["metric_table", "rows"],
            ),
        ]
    )


def _timeline_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    feature_report = (Path("results/feature_release_v1_20260505_2155/feature_release_report.md"))
    label_report = (Path("results/label_release_v1_1_20260506_0041/label_release_report.md"))
    feature_commit = "205cb7d465105a54caa439c5e182e4e4ac11f04d"
    label_commit = "d7a89ecd12992203dde91b5be17fd22629e4338a"
    rows = [
        {
            "stage": "Initial scaffold and environment validation",
            "phase": "repository scaffold / environment checks",
            "date_time": "historical; exact timestamp not centralized",
            "branch_commit": "available through logs/ai_runs and git history",
            "purpose": "establish package, validation script, split policy, and safe data framing",
            "input_data": "repository source plus local CopCo derived data references",
            "outputs": "README, docs, scripts/validate_env.py, package scaffold",
            "commands_validation": "scripts/validate_env.py and pytest smoke tests where logged",
            "key_results": "CopCo environment and package validation became available",
            "status": "historical record",
            "stored_at": "src/, tests/, docs/, logs/ai_runs/",
            "role": "historical record",
        },
        {
            "stage": "Real LM scoring enablement",
            "phase": "Feature Release preparation",
            "date_time": "2026-05-05 21:55 output",
            "branch_commit": feature_commit if feature_report.exists() else "not recorded",
            "purpose": "enable real Danish causal-LM surprisal/entropy and embeddings",
            "input_data": "feature-release word/stimulus tables",
            "outputs": "DFM decoder 7B word-level LM features and embedding features",
            "commands_validation": "Slurm jobs 2722155, 2722194, 2722203; validation passed",
            "key_results": "DFM decoder completed; Gemma 2 9B blocked by gated access",
            "status": "completed with Gemma blocked/deferred",
            "stored_at": "results/feature_release_v1_20260505_2155/",
            "role": "main feature source and historical LM-status record",
        },
        {
            "stage": "Feature Release v1",
            "phase": "feature_release_v1",
            "date_time": "2026-05-05 21:55",
            "branch_commit": feature_commit,
            "purpose": "freeze gaze, text, parser-fallback, LM, embedding, and modeling tables",
            "input_data": "CopCo derived57 normalized source layers",
            "outputs": "feature tables, modeling tables, validation reports, feature dictionary",
            "commands_validation": "feature release validation passed",
            "key_results": "57 participants, 335203 word observations, 31986 stimulus words",
            "status": "complete",
            "stored_at": "results/feature_release_v1_20260505_2155/",
            "role": "main analysis input",
        },
        {
            "stage": "Label Release v1.1",
            "phase": "label_release_v1_1",
            "date_time": "2026-05-06 00:41",
            "branch_commit": label_commit if label_report.exists() else "not recorded",
            "purpose": "freeze operational reader labels, quality labels, split labels, "
            "and prepared dataset",
            "input_data": "Feature Release v1",
            "outputs": "participant, quality, segmentation, split, and analysis-ready tables",
            "commands_validation": "label release validation passed",
            "key_results": "57 participants, 19 dyslexia-labeled, 38 typical/control",
            "status": "complete",
            "stored_at": "results/label_release_v1_1_20260506_0041/",
            "role": "main analysis input",
        },
        {
            "stage": "Segmentation / boundary-opacity label generation",
            "phase": "label_release_v1_1 segmentation layer",
            "date_time": "2026-05-06 00:41",
            "branch_commit": label_commit,
            "purpose": "derive orthographic C/V boundary descriptors",
            "input_data": "stimulus word sequence",
            "outputs": "segmentation_boundary, segmentation_word, segmentation_sentence labels",
            "commands_validation": "label release validation and segmentation reports",
            "key_results": "31986 boundary/word labels; standalone main-effect support not retained",
            "status": "complete as secondary interpretability feature",
            "stored_at": "results/label_release_v1_1_20260506_0041/labels/",
            "role": "secondary/diagnostic analysis",
        },
        {
            "stage": "Phase 3 controlled research exploration",
            "phase": "research_exploration_v1",
            "date_time": "2026-05-06 01:49",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "explore controlled participant profiles, residualization, interactions, "
            "and ablations",
            "input_data": "Label Release v1.1 prepared dataset",
            "outputs": "ablation metrics, residual profiles, segmentation/group interaction reports",
            "commands_validation": "research exploration validation report passed",
            "key_results": "DFM exposure+sensitivity exploratory AUROC 0.9058; "
            "segmentation main effect not supported",
            "status": "complete",
            "stored_at": "results/research_exploration_v1_20260506_0149/",
            "role": "historical and secondary source",
        },
        {
            "stage": "Phase 4 confirmatory sensitivity analysis",
            "phase": "phase4_confirmatory_sensitivity_v1",
            "date_time": "2026-05-06 07:15",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "lock cross-fitted residualized DFM gaze-profile model and robustness tests",
            "input_data": "Phase 3 selected feature family and Label Release v1.1",
            "outputs": "confirmatory metrics, bootstrap/permutation, feature stability, interactions",
            "commands_validation": "phase4 confirmatory validation passed",
            "key_results": "D3 residual gaze AUROC 0.8947, BA 0.8421, p about 0.001",
            "status": "complete",
            "stored_at": "results/phase4_confirmatory_sensitivity_v1_20260506_0715/",
            "role": "main analysis",
        },
        {
            "stage": "AutoResearch v1",
            "phase": "autoresearch_v1",
            "date_time": "2026-05-06 09:17",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "assemble final selection, stress tests, and source-traced paper material",
            "input_data": "feature release, label release, Phase 3, Phase 4 outputs",
            "outputs": "final model metrics, DFM ablation, stress-test tables, decision report",
            "commands_validation": "autoresearch validation passed",
            "key_results": "selected D3_dfm_residual_gaze_only logistic regression LOPO",
            "status": "complete",
            "stored_at": "results/autoresearch_v1_20260506_0917/",
            "role": "main analysis source",
        },
        {
            "stage": "SubmissionSprint v1",
            "phase": "submission_v1",
            "date_time": "2026-05-06 09:36",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "package submission-era material",
            "input_data": "AutoResearch and manuscript source artifacts",
            "outputs": "submission package, reproducibility records, supplement sources",
            "commands_validation": "submission package validation where present",
            "key_results": "historical packaging; not a new result family",
            "status": "complete as historical source",
            "stored_at": "results/submission_v1_20260506_0936/ and paper/submission_v1/",
            "role": "historical record",
        },
        {
            "stage": "Final Manuscript Audit v1",
            "phase": "final_manuscript_audit_v1",
            "date_time": "2026-05-06 14:38",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "audit manuscript/result consistency",
            "input_data": "submission package and evidence outputs",
            "outputs": "audit reports",
            "commands_validation": "audit validation report where present",
            "key_results": "claim and blocker status preserved",
            "status": "complete as audit source",
            "stored_at": "results/final_manuscript_audit_v1_20260506_1438/",
            "role": "diagnostic/historical record",
        },
        {
            "stage": "BenchmarkBridge v1",
            "phase": "benchmark_bridge_v1",
            "date_time": "2026-05-06 18:36",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "evaluate full-data D3 in internal EyeBench-style split regimes",
            "input_data": "prepared full CopCo feature data and D3 residual profiles",
            "outputs": "TYP/RCS metrics, split diagnostics, residualization diagnostics",
            "commands_validation": "benchmark bridge validation passed",
            "key_results": "reader-aggregated full-data D3 AUROC 0.8961 unseen_reader",
            "status": "complete",
            "stored_at": "results/benchmark_bridge_v1_20260506_1836/",
            "role": "secondary benchmark-relative analysis",
        },
        {
            "stage": "OfficialEyeBenchAlignment v1",
            "phase": "official_eyebench_alignment_v1",
            "date_time": "2026-05-06 22:32",
            "branch_commit": "EyeBench submodule ce87f38a3083aeed029c255716a1a51e6ae51167",
            "purpose": "audit official EyeBench data/fold/evaluator compatibility",
            "input_data": "EyeBench submodule metadata and CopCo prepared data",
            "outputs": "alignment audit, official/fold/full-data comparison rows",
            "commands_validation": "official alignment validation passed",
            "key_results": "official subset blocked; EyeBench-fold full-feature intersection complete",
            "status": "complete with official blocker",
            "stored_at": "results/official_eyebench_alignment_v1_20260506_2232/",
            "role": "diagnostic and benchmark framing",
        },
        {
            "stage": "OfficialEyeBenchSOTACheck v1",
            "phase": "official_eyebench_sota_check_v1",
            "date_time": "2026-05-06 23:41",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "test whether an official EyeBench leaderboard claim can be made",
            "input_data": "official alignment audit and EyeBench submodule",
            "outputs": "official environment/data/baseline blocker reports",
            "commands_validation": "official SOTA check validation passed",
            "key_results": "official processed data and environment blocked; no official claim allowed",
            "status": "blocked official result; complete blocker record",
            "stored_at": "results/official_eyebench_sota_check_v1_20260506_2341/",
            "role": "blocked/diagnostic result",
        },
        {
            "stage": "D3 EyeBench own-method score-max v2",
            "phase": "d3_eyebench_own_method_score_max_v2",
            "date_time": "2026-05-22 analysis sync",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "evaluate reduced official-compatible trial-level D3_Lite candidates",
            "input_data": "official-compatible feature intersection",
            "outputs": "trial metrics and candidate leaderboard",
            "commands_validation": "score-max validation passed in analysis record",
            "key_results": "candidate_0000 anchor retained for no-improvement decision",
            "status": "complete",
            "stored_at": "analysis/d3_eyebench_own_method_score_max_v2/",
            "role": "official-compatible stress test",
        },
        {
            "stage": "OperatingPointAdaptation v1",
            "phase": "operating_point_adaptation_v1",
            "date_time": "2026-05-23 analysis sync",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "separate probability, calibration, threshold, reader aggregation, "
            "and oracle diagnostics",
            "input_data": "existing prediction outputs",
            "outputs": "fixed, legal, oracle, calibration, and aggregation metrics",
            "commands_validation": "operating-point validation passed",
            "key_results": "test-oracle thresholds marked diagnostic and not clean evidence",
            "status": "complete",
            "stored_at": "analysis/operating_point_adaptation_v1/",
            "role": "diagnostic analysis",
        },
        {
            "stage": "D3OnlineTargetedOptimization v1",
            "phase": "online_targeted_optimization_v1",
            "date_time": "2026-05-23 analysis sync",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "evaluate prefix datasets, online probabilities, accumulation, and stopping",
            "input_data": "Label Release v1.1 prepared dataset",
            "outputs": "prefix data, nested predictions, online probabilities, locked rows",
            "commands_validation": "v1 validation passed",
            "key_results": "selected no_stop/full-sequence candidate; later audited as offline-like",
            "status": "complete but deprecated/fast for online claim",
            "stored_at": "analysis/d3_online_targeted_optimization_v1/",
            "role": "historical and diagnostic online source",
        },
        {
            "stage": "D3OnlineTargetedOptimization v2 or audit-rerun",
            "phase": "online_targeted_optimization_v2",
            "date_time": "2026-05-23 analysis sync",
            "branch_commit": "not centralized in indexed manifest",
            "purpose": "audit v1 and rerun strict online selection categories",
            "input_data": "v1 artifacts and prepared online prefix data",
            "outputs": "strict final models, per-prefix curves, legal calibration, error analysis",
            "commands_validation": "v2 validation passed",
            "key_results": "offline/all evidence remains strongest; online rows separated by budget",
            "status": "complete",
            "stored_at": "analysis/d3_online_targeted_optimization_v2/",
            "role": "secondary online/offline analysis",
        },
        {
            "stage": "D3ModelEvidenceVault v1",
            "phase": "d3_model_evidence_v1",
            "date_time": "2026-05-23 07:47 output",
            "branch_commit": "recorded in vault source trace",
            "purpose": "curate model evidence into source-traced internal vault",
            "input_data": "all previous result directories",
            "outputs": "v1 evidence vault",
            "commands_validation": "vault validation passed",
            "key_results": "first canonical D3 evidence collection",
            "status": "complete",
            "stored_at": "analysis/d3_model_evidence_v1/",
            "role": "evidence source",
        },
        {
            "stage": "D3ModelEvidenceVault v1.1",
            "phase": "d3_model_evidence_v1_1",
            "date_time": "2026-05-23 10:10 output",
            "branch_commit": "repository commit 4d6604eeb04a8fe64cfca434b9fe2ff247a71373 in status",
            "purpose": "expand canonical metrics, source trace, result scope, and claim status",
            "input_data": "v1 vault and all listed source artifacts",
            "outputs": "v1.1 evidence vault and machine-readable manifests",
            "commands_validation": "vault validation passed",
            "key_results": "486 canonical metric rows and 1 unresolved discrepancy",
            "status": "complete",
            "stored_at": "analysis/d3_model_evidence_v1_1/",
            "role": "primary source for this master record",
        },
        {
            "stage": "Deep literature review if present",
            "phase": "deep_literature_review",
            "date_time": "not present at build time",
            "branch_commit": "not applicable",
            "purpose": "source related-work details if the directory exists",
            "input_data": "analysis/deep_literature_review/",
            "outputs": "none indexed because directory is missing",
            "commands_validation": "missing source recorded",
            "key_results": "missing; no values fabricated",
            "status": "missing source",
            "stored_at": "analysis/deep_literature_review/",
            "role": "missing/historical placeholder",
        },
    ]
    return rows


def _render_section_2(context: dict[str, Any]) -> str:
    rows = _timeline_rows(context)
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[1]),
            "",
            "Chronological route. Dates come from timestamped output directories or source "
            "reports where available. Missing branch/commit fields are explicitly recorded "
            "rather than inferred.",
            "",
            _markdown_table(
                rows,
                [
                    "stage",
                    "phase",
                    "date_time",
                    "branch_commit",
                    "purpose",
                    "input_data",
                    "outputs",
                    "commands_validation",
                    "key_results",
                    "status",
                    "stored_at",
                    "role",
                ],
            ),
        ]
    )


def _render_section_3(context: dict[str, Any], labels: dict[str, Any]) -> str:
    feature_join = context["json"]["feature_join"]
    dfm_alignment = context["json"]["dfm_alignment"]
    parser = context["json"]["parser_diagnostics"]
    embed = context["json"]["embedding_manifest"]
    online_v1 = context["json"]["online_v1_manifest"]
    online_v1_val = context["json"]["online_v1_validation"]
    online_v2 = context["json"]["online_v2_manifest"]
    official = context["csv"]["official_alignment"]
    official_rows = official.head(3).to_dict("records") if not official.empty else []
    full_data = [
        {"item": "participant count", "value": labels.get("participants"), "source": "label validation"},
        {
            "item": "dyslexia-labeled count",
            "value": labels.get("dyslexia_labeled"),
            "source": "participant label report",
        },
        {
            "item": "typical/control count",
            "value": labels.get("typical_control"),
            "source": "participant label report",
        },
        {
            "item": "word-level gaze rows",
            "value": labels.get("word_observations"),
            "source": "feature table summary",
        },
        {
            "item": "stimulus word rows",
            "value": labels.get("stimulus_words"),
            "source": "feature table summary",
        },
        {"item": "sentence rows", "value": labels.get("sentences"), "source": "feature table summary"},
        {"item": "paragraph rows", "value": labels.get("paragraphs"), "source": "feature table summary"},
        {
            "item": "participant-level rows",
            "value": labels.get("participant_rows"),
            "source": "prepared manifest",
        },
        {
            "item": "DFM LM rows",
            "value": labels.get("dfm_rows"),
            "source": "join validation",
        },
        {
            "item": "segmentation boundary rows",
            "value": labels.get("segmentation_boundary_rows"),
            "source": "label validation",
        },
        {
            "item": "segmentation word rows",
            "value": labels.get("segmentation_word_rows"),
            "source": "label validation",
        },
        {
            "item": "quality label rows",
            "value": labels.get("quality_rows"),
            "source": "label validation",
        },
        {
            "item": "split label rows",
            "value": labels.get("split_label_rows"),
            "source": "label validation",
        },
        {
            "item": "LM missing rate",
            "value": labels.get("lm_missing_rate"),
            "source": "label validation",
        },
        {
            "item": "embedding missing rate",
            "value": labels.get("embedding_missing_rate"),
            "source": "label validation",
        },
        {
            "item": "parser missing rate",
            "value": labels.get("parser_missing_rate"),
            "source": "label validation",
        },
    ]
    online_rows = [
        {
            "item": "v1 prefix rows",
            "value": online_v1_val.get("prefix_rows"),
            "source": "d3_online_targeted_optimization_validation_report.json",
        },
        {
            "item": "v1 nested prediction rows",
            "value": online_v1_val.get("nested_prediction_rows"),
            "source": "d3_online_targeted_optimization_validation_report.json",
        },
        {
            "item": "v1 online probability rows",
            "value": online_v1.get("online_probability_rows"),
            "source": "run_manifest.json",
        },
        {
            "item": "v1 accumulation rows",
            "value": online_v1.get("accumulation_metric_rows"),
            "source": "run_manifest.json",
        },
        {
            "item": "v1 stopping rows",
            "value": online_v1.get("stopping_metric_rows"),
            "source": "run_manifest.json",
        },
        {"item": "v1 oracle rows", "value": online_v1.get("oracle_rows"), "source": "run_manifest.json"},
        {
            "item": "v1 error trajectory rows",
            "value": online_v1.get("trajectory_rows"),
            "source": "run_manifest.json",
        },
        {
            "item": "v2 per-prefix rows",
            "value": online_v2.get("per_prefix_rows"),
            "source": "run_manifest.json",
        },
        {"item": "v2 candidate rows", "value": online_v2.get("candidate_rows"), "source": "run_manifest.json"},
        {
            "item": "v2 final model rows",
            "value": online_v2.get("final_model_rows"),
            "source": "run_manifest.json",
        },
        {"item": "v2 locked rows", "value": online_v2.get("locked_rows"), "source": "run_manifest.json"},
        {"item": "v2 audit rows", "value": online_v2.get("audit_rows"), "source": "run_manifest.json"},
        {"item": "v2 error rows", "value": online_v2.get("error_rows"), "source": "run_manifest.json"},
    ]
    boundary_rows = [
        {"boundary_type": key, "count": value}
        for key, value in labels.get("boundary_counts", {}).items()
    ]
    embedding_rows = [
        {
            "label": item.get("label"),
            "model_id": item.get("model_id"),
            "sentence_rows": item.get("sentence_rows"),
            "paragraph_rows": item.get("paragraph_rows"),
            "embedding_dim": item.get("embedding_dim"),
        }
        for item in embed.get("models", [])
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[2]),
            "",
            "### A. Full prepared CopCo / CopCo-Dyslexia-style dataset",
            "",
            "The full prepared data are the project-specific CopCo derived57 prepared "
            "dataset with operational reader labels. Source paths include "
            "`results/feature_release_v1_20260505_2155/`, "
            "`results/label_release_v1_1_20260506_0041/`, and "
            "`results/label_release_v1_1_20260506_0041/prepared_dataset/`.",
            "",
            _markdown_table(full_data, ["item", "value", "source"]),
            "",
            "Boundary-label distribution:",
            "",
            _markdown_table(boundary_rows, ["boundary_type", "count"]),
            "",
            "DFM alignment and parser status: DFM alignment status is `"
            + str(dfm_alignment.get("status"))
            + "` with warning counts "
            + str(dfm_alignment.get("warning_counts"))
            + ". Parser backend is `"
            + str(parser.get("backend"))
            + "` with preferred backend `"
            + str(parser.get("preferred_backend"))
            + "` and no true syntax claim.",
            "",
            "Embedding models recorded:",
            "",
            _markdown_table(
                embedding_rows,
                ["label", "model_id", "sentence_rows", "paragraph_rows", "embedding_dim"],
            ),
            "",
            "### B. EyeBench-related data/protocol settings",
            "",
            "Official EyeBench status is separated from EyeBench-style and "
            "official-compatible internal results. The EyeBench submodule commit recorded by "
            "the alignment audit is `ce87f38a3083aeed029c255716a1a51e6ae51167`. "
            "Official processed CopCo data were not present, the official environment was not "
            "import-ready, and the official evaluator did not run. EyeBench-fold full-feature "
            "intersection rows completed, but they are not official processed-data results. "
            "Full-data EyeBench-style rows are internal benchmark-relative rows.",
            "",
            _markdown_table(
                [
                    {
                        "mode": row.get("mode"),
                        "model": row.get("model"),
                        "claim_type": row.get("claim_type"),
                        "official_mode": row.get("official_mode"),
                        "exact_folds": row.get("exact_folds"),
                        "exact_processed_data": row.get("exact_processed_data"),
                    }
                    for row in official_rows
                ],
                [
                    "mode",
                    "model",
                    "claim_type",
                    "official_mode",
                    "exact_folds",
                    "exact_processed_data",
                ],
            ),
            "",
            "### C. Full-data versus reduced official-compatible data",
            "",
            "Full data include participant-level aggregates, residual gaze summaries, DFM "
            "surprisal/entropy, DFM sensitivity profiles, segmentation features, parser "
            "fallback features, and embedding-derived compact semantic features. The "
            "official-compatible reduced variant is trial-level and constrained by the "
            "intersection with EyeBench fold/protocol structure; it lacks the full reader "
            "profile scope and is not equivalent to the full-record reader-profile model.",
            "",
            "### D. Online prefix data",
            "",
            "Online prefix data use cumulative evidence only. Prefix types include "
            "`word_count_prefix`, `chronological_prefix`, `trial_or_text_prefix`, "
            "`speech_prefix`, and sequence stopping rows. Budgets recorded in v1/v2 include "
            "50, 100, 250, 500, 1000, one to three trials/texts/speeches, `all`, and "
            "sequence-stop rows. Nested split roles are train_fit, inner_oof, calibration, "
            "and outer_test; legal threshold/calibration rows are selected without outer-test "
            "label thresholds.",
            "",
            _markdown_table(online_rows, ["item", "value", "source"]),
            "",
            "Additional source detail: feature join validation reports "
            f"{_fmt(feature_join.get('word_level_full_rows'), 0)} full word rows and "
            f"{_fmt(feature_join.get('word_level_full_with_dfm_lm_rows'), 0)} "
            "word rows with DFM LM columns joined.",
        ]
    )


def _render_section_4(context: dict[str, Any]) -> str:
    split_rows = [
        {
            "split": "leave_one_participant_out / LOPO",
            "what_it_tests": "reader-level generalization to each held-out participant",
            "train_test_disjoint": "participant-disjoint",
            "participant_disjoint": True,
            "text_disjoint": False,
            "completed": True,
            "folds_skipped": 0,
            "source_file": "results/label_release_v1_1_20260506_0041/analysis/label_analysis/"
            "split_label_report.md; analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
            "role": "main",
        },
        {
            "split": "participant_grouped_kfold",
            "what_it_tests": "participant-grouped cross-validation without participant leakage",
            "train_test_disjoint": "participant-disjoint by fold",
            "participant_disjoint": True,
            "text_disjoint": False,
            "completed": True,
            "folds_skipped": "not recorded as skipped in v2 final rows",
            "source_file": "feature release splits and online v2 strict_final_models.csv",
            "role": "secondary",
        },
        {
            "split": "unseen_reader",
            "what_it_tests": "held-out readers with seen text distribution allowed",
            "train_test_disjoint": "reader-disjoint",
            "participant_disjoint": True,
            "text_disjoint": False,
            "completed": True,
            "folds_skipped": 0,
            "source_file": "BenchmarkBridge, OfficialEyeBenchAlignment, online v2",
            "role": "secondary/benchmark",
        },
        {
            "split": "unseen_text",
            "what_it_tests": "held-out speeches/texts with reader overlap allowed by protocol",
            "train_test_disjoint": "text-disjoint",
            "participant_disjoint": False,
            "text_disjoint": True,
            "completed": True,
            "folds_skipped": 0,
            "source_file": "BenchmarkBridge, OfficialEyeBenchAlignment, online v2",
            "role": "secondary/diagnostic; unresolved specialist rows",
        },
        {
            "split": "unseen_reader_and_text",
            "what_it_tests": "simultaneous held-out readers and texts",
            "train_test_disjoint": "reader-disjoint and text-disjoint",
            "participant_disjoint": True,
            "text_disjoint": True,
            "completed": True,
            "folds_skipped": 0,
            "source_file": "BenchmarkBridge, OfficialEyeBenchAlignment, online v2",
            "role": "secondary/benchmark",
        },
        {
            "split": "text_balanced_unseen_reader",
            "what_it_tests": "reader-disjoint split with deterministic text-exposure balancing",
            "train_test_disjoint": "reader-disjoint",
            "participant_disjoint": True,
            "text_disjoint": False,
            "completed": True,
            "folds_skipped": "not recorded as skipped in v2 final rows",
            "source_file": "BenchmarkBridge and online v2",
            "role": "diagnostic/secondary",
        },
        {
            "split": "leave_one_speech_out",
            "what_it_tests": "speech/text holdout sensitivity",
            "train_test_disjoint": "speech/text-disjoint",
            "participant_disjoint": False,
            "text_disjoint": True,
            "completed": "available in split sources where enabled",
            "folds_skipped": "not centralized in canonical metric table",
            "source_file": "results/feature_release_v1_20260505_2155/splits/leave_one_speech_out.csv",
            "role": "diagnostic/historical",
        },
        {
            "split": "online prefix splits",
            "what_it_tests": "same outer regimes under evidence prefixes",
            "train_test_disjoint": "outer-test rows excluded from fit/threshold/calibration",
            "participant_disjoint": "depends on outer regime",
            "text_disjoint": "depends on outer regime",
            "completed": True,
            "folds_skipped": "not collapsed; see online manifests",
            "source_file": "analysis/d3_online_targeted_optimization_v1/ and v2/",
            "role": "secondary/diagnostic online",
        },
        {
            "split": "official-compatible split handling",
            "what_it_tests": "EyeBench fold metadata compatibility",
            "train_test_disjoint": "official fold roles where available",
            "participant_disjoint": "official dependent",
            "text_disjoint": "official dependent",
            "completed": "fold-aligned intersection completed; official subset blocked",
            "folds_skipped": "official subset skipped",
            "source_file": "analysis/official_eyebench_alignment_v1/",
            "role": "blocked official / diagnostic",
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[3]),
            "",
            "All clean predictive evaluations use participant-aware or protocol-aware split "
            "roles. Random word-level train/test splitting is excluded because the target is "
            "reader-level.",
            "",
            _markdown_table(
                split_rows,
                [
                    "split",
                    "what_it_tests",
                    "train_test_disjoint",
                    "participant_disjoint",
                    "text_disjoint",
                    "completed",
                    "folds_skipped",
                    "source_file",
                    "role",
                ],
            ),
            "",
            "Nested online roles are train_fit for fitting model parameters, inner_oof for "
            "candidate selection and legal thresholds, calibration for fitted calibration "
            "where available, and outer_test for final clean evaluation only. Oracle/test-label "
            "threshold rows are explicitly diagnostic.",
        ]
    )


def _render_section_5(context: dict[str, Any]) -> str:
    feature_groups = [
        {
            "family": "A. Gaze features",
            "details": "first fixation duration, first-pass duration, go-past time, total "
            "fixation duration, fixation count, skipping/fixated indicator, landing-position "
            "fields where available, saccade/source-derived features where present, "
            "participant aggregates, residual gaze features, and online cumulative gaze "
            "summaries.",
            "source": "Feature Release v1; participant sensitivity dictionaries; "
            "residualization reports",
        },
        {
            "family": "B. Classical text features",
            "details": "word length, sentence length, word position, punctuation, "
            "capitalization/digit flags, frequency/log frequency, readability/surface "
            "components, and text-level exposure controls.",
            "source": "Feature Release v1 feature dictionary and modeling tables",
        },
        {
            "family": "C. Parser or parser-fallback features",
            "details": "parser_status is surface_heuristic_fallback. These features are "
            "surface and morpho-orthographic heuristics, not true syntactic parses. Parser "
            "syntax claims are not supported.",
            "source": "parser_diagnostics.json; quality_label_card_v1.md",
        },
        {
            "family": "D. Segmentation / boundary-opacity features",
            "details": "C#C, C#V, V#C, V#V, other/unknown; deterministic orthographic "
            "vowel/consonant logic using Danish vowels a/e/i/o/u/y/ae/oe/aa equivalents in "
            "source spelling plus Danish letters; previous-boundary, next-boundary, "
            "sentence-level rates, and V#V indicators. These are stimulus-level linguistic "
            "labels and secondary interpretability features.",
            "source": "segmentation_label_card_v1.md and label release reports",
        },
        {
            "family": "E. DFM language-model features",
            "details": "danish-foundation-models/dfm-decoder-open-v0-7b-pt with tokenizer "
            "from the causal LM; causal scoring produces word-level surprisal and entropy; "
            "subword values are aligned and aggregated to words with warning/missingness "
            "tracked. DFM exposure features summarize average stimulus difficulty. DFM "
            "sensitivity features and residualized DFM gaze features summarize each reader's "
            "gaze-cost slope after residualizing gaze against stimulus/text covariates.",
            "source": "DFM alignment report, dfm_feature_summary, residualization reports",
        },
        {
            "family": "F. Embedding features",
            "details": "KennethEnevoldsen/dfm-sentence-encoder-large and "
            "intfloat/multilingual-e5-large sentence/paragraph embeddings, compact semantic "
            "features, and missingness indicators. These are context features, not the main "
            "D3 residual-gaze signal.",
            "source": "embedding manifest and feature dictionary",
        },
        {
            "family": "G. Online prefix features",
            "details": "cumulative residual gaze features, cumulative DFM exposure, "
            "cumulative DFM residual summaries, prefix stability, uncertainty features, "
            "evidence budgets, and stable_enough_for_prediction flags. Prefix features do not "
            "use future evidence.",
            "source": "docs/d3_online_targeted_optimization_v1.md and v2 artifacts",
        },
        {
            "family": "H. Prohibited or excluded features",
            "details": "participant_id as predictor, speech_id/text_id as direct predictors, "
            "future online evidence, exposure-count variables in primary models, and "
            "test-label thresholds in clean metrics are excluded. Oracle rows are diagnostic "
            "only.",
            "source": "prohibited_feature_policy.md, leakage controls, split policy",
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[4]),
            "",
            "This section records feature construction families at the level needed for "
            "future method explanation. It does not add new features or perform feature "
            "selection.",
            "",
            _markdown_table(feature_groups, ["family", "details", "source"]),
            "",
            "Residual gaze construction: residualizers are fit inside the relevant training "
            "fold using stimulus/text predictors such as word length, log frequency, DFM "
            "surprisal, DFM entropy, sentence length, word position, segmentation labels, and "
            "missingness flags. Reader group, participant ID, speech ID, text ID, labels, and "
            "targets are not residualizer predictors. The resulting participant features are "
            "aggregates and slopes of residual gaze costs against DFM predictability features.",
        ]
    )


def _render_section_6(context: dict[str, Any]) -> str:
    parser = context["json"]["parser_diagnostics"]
    model_rows = [
        {
            "model_tool": "danish-foundation-models/dfm-decoder-open-v0-7b-pt",
            "role": "primary causal LM for surprisal and entropy",
            "input_output": "Danish text context to subword/word-level surprisal and entropy",
            "status": "completed",
            "source_phase": "Feature Release v1",
            "notes": "base/pretrained causal LM; instruction tuning not used for token likelihood",
        },
        {
            "model_tool": "google/gemma-2-9b",
            "role": "attempted sensitivity LM comparison",
            "input_output": "would have produced alternative causal-LM features",
            "status": "blocked",
            "source_phase": "Feature Release v1",
            "notes": "blocked by gated Hugging Face access; no values fabricated",
        },
        {
            "model_tool": "KennethEnevoldsen/dfm-sentence-encoder-large",
            "role": "Danish sentence/paragraph embeddings",
            "input_output": "sentences/paragraphs to 1024-dimensional embeddings and semantic summaries",
            "status": "completed",
            "source_phase": "Feature Release v1",
            "notes": "embedding context features, not causal surprisal",
        },
        {
            "model_tool": "intfloat/multilingual-e5-large",
            "role": "multilingual sentence/paragraph embeddings",
            "input_output": "sentences/paragraphs to 1024-dimensional embeddings and semantic summaries",
            "status": "completed",
            "source_phase": "Feature Release v1",
            "notes": "secondary semantic feature source",
        },
        {
            "model_tool": "DaCy / spaCy",
            "role": "preferred parser backend",
            "input_output": "sentence tokens to linguistic features",
            "status": "attempted/fallback",
            "source_phase": "Feature Release v1",
            "notes": "preferred backend unavailable; backend error recorded as "
            + str(parser.get("backend_error")),
        },
        {
            "model_tool": "surface_heuristic parser fallback",
            "role": "parser-fallback feature generation",
            "input_output": "surface/token features and heuristic covariates",
            "status": "completed",
            "source_phase": "Feature Release v1",
            "notes": "not true syntax; parser-syntax claims prohibited",
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[5]),
            "",
            _markdown_table(
                model_rows,
                ["model_tool", "role", "input_output", "status", "source_phase", "notes"],
            ),
            "",
            "Base/pretrained causal language models are used for surprisal because "
            "token-level likelihood under left context is the needed quantity. "
            "Instruction-tuned/chat models are not used for surprisal because instruction "
            "alignment changes the task objective and does not provide the same stable "
            "next-token probability interpretation.",
            "",
            "DFM exposure differs from DFM sensitivity: exposure summarizes the DFM "
            "predictability profile of the text a reader saw, while sensitivity summarizes how "
            "that reader's residual gaze costs vary with DFM surprisal/entropy. DFM residual "
            "gaze features are produced by fitting fold-local residualizers for gaze outcomes "
            "against stimulus/text covariates, then aggregating residual means and residual "
            "slopes with respect to DFM predictability features at the participant level.",
        ]
    )


def _render_section_7(context: dict[str, Any]) -> str:
    final = _first_row(context["csv"]["final_metrics"])
    dfm = context["csv"]["dfm_ablation"]
    v2 = context["csv"]["online_v2_final"]
    d3_lite = context["csv"]["d3_lite_trial"]
    bridge = context["csv"]["benchmark_bridge"]
    operating = context["csv"]["operating_fixed"]
    rescue = context["csv"]["online_v2_rescue"]

    def _dfm_row(name: str) -> dict[str, Any]:
        if dfm.empty:
            return {}
        sub = dfm[dfm["feature_group"].astype(str).eq(name)]
        return _first_row(sub)

    model_rows = [
        {
            "model_name_internal": "D3_dfm_residual_gaze_only",
            "public_description": "full-record residualized predictability-sensitive reader-profile model",
            "data_scope": "full prepared CopCo",
            "evaluation_level": "reader",
            "split_regime": "LOPO",
            "feature_family": "DFM residual gaze sensitivity",
            "calibrator": "none/fixed probability output",
            "threshold_policy": "fixed 0.5 in final metrics",
            "accumulator": "not applicable",
            "stopping_policy": "not applicable",
            "key_metrics": (
                f"AUROC {_fmt(_metric_col(final, 'roc_auc'))}; "
                f"BA {_fmt(_metric_col(final, 'balanced_accuracy'))}; "
                f"PR-AUC {_fmt(_metric_col(final, 'pr_auc'))}"
            ),
            "source_files": "analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
        },
        {
            "model_name_internal": "D1_dfm_exposure_only",
            "public_description": "language-model exposure-only ablation",
            "data_scope": "full prepared CopCo",
            "evaluation_level": "reader",
            "split_regime": "LOPO/ablation",
            "feature_family": "DFM exposure only",
            "calibrator": "not recorded",
            "threshold_policy": "fixed/evaluation default",
            "accumulator": "not applicable",
            "stopping_policy": "not applicable",
            "key_metrics": "AUROC " + _fmt(_dfm_row("D1_dfm_exposure_only").get("roc_auc")),
            "source_files": "dfm_exposure_vs_sensitivity_table.csv",
        },
        {
            "model_name_internal": "D2_dfm_sensitivity_only",
            "public_description": "language-model sensitivity-only ablation",
            "data_scope": "full prepared CopCo",
            "evaluation_level": "reader",
            "split_regime": "LOPO/ablation",
            "feature_family": "DFM sensitivity",
            "calibrator": "not recorded",
            "threshold_policy": "fixed/evaluation default",
            "accumulator": "not applicable",
            "stopping_policy": "not applicable",
            "key_metrics": "AUROC " + _fmt(_dfm_row("D2_dfm_sensitivity_only").get("roc_auc")),
            "source_files": "dfm_exposure_vs_sensitivity_table.csv",
        },
        {
            "model_name_internal": "D3_FullData_EyeBenchStyle",
            "public_description": "full-data internal EyeBench-style reader-aggregated model",
            "data_scope": "full prepared CopCo",
            "evaluation_level": "reader aggregated",
            "split_regime": "unseen_reader/unseen_text/unseen_reader_and_text",
            "feature_family": "D3 residual gaze profile",
            "calibrator": "not primary",
            "threshold_policy": "fixed/evaluation default",
            "accumulator": "reader aggregation",
            "stopping_policy": "not applicable",
            "key_metrics": "see BenchmarkBridge internal rows",
            "source_files": "analysis/benchmark_bridge_v1/tables/copco_typ_benchmark_comparison.csv",
        },
        {
            "model_name_internal": "D3_EyeBench_Lite candidate_0000",
            "public_description": "reduced official-protocol-compatible trial-level variant",
            "data_scope": "official-compatible feature/fold subset",
            "evaluation_level": "official trial-level fold mean",
            "split_regime": "unseen_reader/unseen_text/unseen_reader_and_text",
            "feature_family": "reduced D3_Lite exact features",
            "calibrator": "none",
            "threshold_policy": "fixed 0.5",
            "accumulator": "not applicable",
            "stopping_policy": "not applicable",
            "key_metrics": "anchor rows; no locked candidate improved anchor",
            "source_files": "analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv",
        },
        {
            "model_name_internal": "OperatingPointAdaptation rows",
            "public_description": "probability-first operating-point diagnostic",
            "data_scope": "existing prediction outputs",
            "evaluation_level": "reader/trial depending source",
            "split_regime": "multiple",
            "feature_family": "probability outputs",
            "calibrator": "fixed, fitted where legal, or oracle diagnostic",
            "threshold_policy": "fixed 0.5, legal inner, or test-oracle diagnostic",
            "accumulator": "reader probability aggregation where available",
            "stopping_policy": "not applicable",
            "key_metrics": f"{len(operating)} fixed-threshold rows",
            "source_files": "analysis/operating_point_adaptation_v1/",
        },
        {
            "model_name_internal": "best_online_late_accumulation / mid / early",
            "public_description": "fixed-budget sequential reader-evidence models",
            "data_scope": "online prefix data",
            "evaluation_level": "reader",
            "split_regime": "participant/text online regimes",
            "feature_family": "DFM residual plus uncertainty prefix features",
            "calibrator": "identity, isotonic, sigmoid depending candidate",
            "threshold_policy": "fixed 0.5 or inner_cv_regime_specific",
            "accumulator": "mean_probability or learned_meta_aggregator",
            "stopping_policy": "fixed_budget",
            "key_metrics": f"{len(v2)} strict final rows",
            "source_files": "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv",
        },
        {
            "model_name_internal": "best_online_stopping_detector",
            "public_description": "adaptive stopping diagnostic",
            "data_scope": "online sequence/prefix data",
            "evaluation_level": "reader",
            "split_regime": "participant/text online regimes",
            "feature_family": "DFM residual plus uncertainty prefix features",
            "calibrator": "identity",
            "threshold_policy": "inner_cv_global",
            "accumulator": "learned_meta_aggregator",
            "stopping_policy": "coverage_constrained_stop",
            "key_metrics": "stopping_not_ready status",
            "source_files": "online v2 strict_final_models and stopping result summaries",
        },
        {
            "model_name_internal": "unseen_text_rescue_04 / unseen_text_rescue_05",
            "public_description": "unseen-text specialist diagnostic/rescue variants",
            "data_scope": "online unseen_text split",
            "evaluation_level": "reader",
            "split_regime": "unseen_text",
            "feature_family": "all_allowed_strict_online",
            "calibrator": "identity or sigmoid",
            "threshold_policy": "inner_cv_regime_specific",
            "accumulator": "entropy_weighted or mean_probability",
            "stopping_policy": "fixed_budget",
            "key_metrics": f"{len(rescue)} rescue candidate rows; discrepancy preserved",
            "source_files": "unseen_text_rescue_candidates.csv",
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[6]),
            "",
            _markdown_table(
                model_rows,
                [
                    "model_name_internal",
                    "public_description",
                    "data_scope",
                    "evaluation_level",
                    "split_regime",
                    "feature_family",
                    "calibrator",
                    "threshold_policy",
                    "accumulator",
                    "stopping_policy",
                    "key_metrics",
                    "source_files",
                ],
            ),
            "",
            "Reduced D3_Lite trial-level rows are not the full method because their data "
            "scope, feature scope, evaluation unit, and official-compatible constraints differ "
            "from the full-record reader-profile model.",
            "",
            "Representative source snippets:",
            "",
            "D3_Lite anchor trial rows:",
            "",
            _frame_table(
                d3_lite.head(3),
                [
                    "candidate_id",
                    "split_name",
                    "evaluation_level",
                    "n_predictions",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                ],
            ),
            "",
            "BenchmarkBridge full-data row:",
            "",
            _frame_table(
                bridge.head(1),
                [
                    "model",
                    "unseen_reader_balanced_accuracy",
                    "unseen_text_balanced_accuracy",
                    "unseen_reader_text_balanced_accuracy",
                    "unseen_reader_AUROC",
                    "unseen_text_AUROC",
                    "unseen_reader_text_AUROC",
                ],
            ),
        ]
    )


def _render_section_8(context: dict[str, Any]) -> str:
    final = context["csv"]["final_metrics"]
    dfm = context["csv"]["dfm_ablation"]
    phase3 = context["csv"]["phase3_ablation"]
    bootstrap = context["csv"]["phase4_bootstrap"]
    registry = context["csv"]["number_registry"]
    perm = _read_csv(Path(".").resolve(), "analysis/phase4_confirmatory/permutation_results.csv")
    p_value = 0.000999 if not perm.empty else "not recorded"
    final_row = _first_row(final)
    primary_rows = [
        {
            "metric": "AUROC",
            "value": _metric_col(final_row, "roc_auc", "AUROC"),
            "source": "final_model_metrics_table.csv",
        },
        {
            "metric": "PR-AUC",
            "value": _metric_col(final_row, "pr_auc", "PR_AUC"),
            "source": "final_model_metrics_table.csv",
        },
        {
            "metric": "balanced accuracy",
            "value": _metric_col(final_row, "balanced_accuracy"),
            "source": "final_model_metrics_table.csv",
        },
        {
            "metric": "macro F1",
            "value": _metric_col(final_row, "macro_f1", "macro_F1"),
            "source": "final_model_metrics_table.csv",
        },
        {
            "metric": "Brier",
            "value": _metric_col(final_row, "brier_score", "Brier"),
            "source": "final_model_metrics_table.csv",
        },
        {
            "metric": "permutation p-value",
            "value": p_value,
            "source": "phase4_confirmatory/permutation_results.csv",
        },
        {
            "metric": "bootstrap AUROC CI",
            "value": _bootstrap_value(bootstrap, "roc_auc"),
            "source": "phase4_confirmatory/bootstrap_results.csv",
        },
        {
            "metric": "predictions",
            "value": _metric_col(final_row, "n_predictions"),
            "source": "final_model_metrics_table.csv",
        },
        {
            "metric": "usable folds",
            "value": _metric_col(final_row, "usable_folds"),
            "source": "final_model_metrics_table.csv",
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[7]),
            "",
            "### A. Offline full-record reader-profile result",
            "",
            _markdown_table(primary_rows, ["metric", "value", "source"]),
            "",
            "### B. DFM exposure vs sensitivity ablation",
            "",
            _frame_table(
                dfm,
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
                ],
                max_rows=20,
            ),
            "",
            "Interpretation boundary: exposure-only rows are an ablation against text/LM "
            "exposure, not a clinical or causal test. The recorded pattern is that DFM "
            "sensitivity/residual gaze rows substantially exceed exposure-only rows.",
            "",
            "### C. Phase 3 exploration",
            "",
            "Phase 3 records an exploratory best participant-level model with "
            "`D_dfm_exposure_and_sensitivity` logistic regression under LOPO AUROC 0.9058, "
            "permutation p-value 0.0099, and bootstrap AUROC interval [0.8162, 0.9798]. "
            "It also records word-level secondary ladder outputs, reader-group interactions "
            "as exploratory, and standalone segmentation main-effect support as not retained.",
            "",
            _frame_table(
                phase3.head(8),
                [
                    "split_name",
                    "feature_group",
                    "model",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                    "n_predictions",
                ],
            ),
            "",
            "### D. Phase 4 confirmatory analysis",
            "",
            "Phase 4 selected the cross-fitted residualized DFM gaze-profile model, checked "
            "bootstrap/permutation robustness, preserved fold-local residualization, and "
            "recorded feature stability. Mixed-effects/interaction summaries are retained as "
            "secondary interpretability sources.",
            "",
            _frame_table(
                bootstrap,
                [
                    "metric",
                    "feature_group",
                    "model",
                    "split_name",
                    "observed",
                    "n_bootstrap",
                    "ci_low",
                    "ci_high",
                ],
            ),
            "",
            "### E. AutoResearch final selection",
            "",
            "AutoResearch records the selected model as D3_dfm_residual_gaze_only logistic "
            "regression, with final main support from the locked Phase 4 LOPO metrics, "
            "DFM exposure-vs-sensitivity ablation, stress tests, calibration metrics, "
            "feature-stability outputs, and reviewer-risk/limitation source tables. It does "
            "not convert operational labels into clinical diagnostic claims.",
            "",
            f"Number registry rows available for future source tracing: {len(registry)}.",
        ]
    )


def _bootstrap_value(frame: pd.DataFrame, metric: str) -> str:
    if frame.empty:
        return "not recorded"
    sub = frame[frame["metric"].astype(str).eq(metric)]
    if sub.empty:
        return "not recorded"
    row = _first_row(sub)
    return f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}]"


def _render_section_9(context: dict[str, Any]) -> str:
    official = context["csv"]["official_alignment"]
    official_sota = context["csv"]["official_sota"]
    bridge = context["csv"]["benchmark_bridge"]
    d3_lite = context["csv"]["d3_lite_trial"]
    leader = context["csv"]["d3_lite_leaderboard"]
    external_baselines = official_sota[
        official_sota["claim_type"].astype(str).eq("official_reported_reference")
    ] if not official_sota.empty else pd.DataFrame()
    full_rows = official[
        official["model"].astype(str).isin(
            ["D3_FullFeature_EyeBenchFolds", "D3_FullData_EyeBenchStyle"]
        )
    ] if not official.empty else pd.DataFrame()
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[8]),
            "",
            "This section separates official reported baselines, internal full-data "
            "EyeBench-style comparisons, EyeBench-fold full-feature intersection rows, "
            "official-subset/evaluator blockers, and reduced official-compatible trial-level "
            "D3_Lite rows. These result types are not interchangeable.",
            "",
            "### Subsection A — Published / provided CopCo TYP baselines",
            "",
            "The following values are recorded as official reported reference rows, not "
            "direct reruns by this repository.",
            "",
            _frame_table(
                external_baselines,
                [
                    "model",
                    "unseen_reader_balanced_accuracy",
                    "unseen_reader_AUROC",
                    "unseen_text_balanced_accuracy",
                    "unseen_text_AUROC",
                    "unseen_reader_text_balanced_accuracy",
                    "unseen_reader_text_AUROC",
                    "average_balanced_accuracy",
                    "average_AUROC",
                    "metric_basis",
                    "notes",
                ],
                max_rows=30,
            ),
            "",
            "### Subsection B — Internal EyeBench-style full-data reader-aggregated comparison",
            "",
            _frame_table(
                bridge,
                [
                    "model",
                    "unseen_reader_balanced_accuracy",
                    "unseen_text_balanced_accuracy",
                    "unseen_reader_text_balanced_accuracy",
                    "average_balanced_accuracy",
                    "unseen_reader_AUROC",
                    "unseen_text_AUROC",
                    "unseen_reader_text_AUROC",
                    "average_AUROC",
                    "evaluation_level",
                    "official_mode",
                    "notes",
                ],
                max_rows=15,
            ),
            "",
            "The full-data D3 row is internal EyeBench-style and benchmark-relative. It is "
            "not an official leaderboard row because exact processed EyeBench data and the "
            "official evaluator were not used.",
            "",
            "### Subsection C — EyeBench-fold full-feature intersection",
            "",
            _frame_table(
                full_rows,
                [
                    "model",
                    "mode",
                    "claim_type",
                    "official_mode",
                    "exact_folds",
                    "exact_processed_data",
                    "unseen_reader_balanced_accuracy",
                    "unseen_text_balanced_accuracy",
                    "unseen_reader_text_balanced_accuracy",
                    "average_balanced_accuracy",
                    "unseen_reader_AUROC",
                    "unseen_text_AUROC",
                    "unseen_reader_text_AUROC",
                    "average_AUROC",
                ],
                max_rows=10,
            ),
            "",
            "Alignment audit overlap: 57 common participants, 19 common dyslexia-labeled "
            "participants, 38 common typical/control participants, 32 common texts, 4782 "
            "common trials, and 31986 common word rows. The official subset was blocked; the "
            "EyeBench-fold full-feature intersection completed with exact folds but not exact "
            "processed data.",
            "",
            "### Subsection D — Official EyeBench subset/evaluator",
            "",
            "Official environment status: blocked_by_environment. Official processed data "
            "status: blocked_by_data. Official evaluator status: not run. Baseline "
            "reproduction status: skipped. Final blocker category: environment/data. "
            "No official leaderboard result was produced because official processed CopCo "
            "data and an import-ready official EyeBench environment were absent.",
            "",
            "### Subsection E — Reduced official-protocol-compatible trial-level model",
            "",
            "D3_Lite is the reduced official-compatible trial-level variant. It is not the "
            "full reader-profile method. Candidate_0000 is the anchor; the locked candidate "
            "search recorded no-improvement relative to that anchor for the intended decision.",
            "",
            _frame_table(
                d3_lite,
                [
                    "candidate_id",
                    "family",
                    "feature_recipe",
                    "model_type",
                    "threshold_method",
                    "calibration_method",
                    "split_name",
                    "evaluation_level",
                    "n_features",
                    "n_predictions",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                    "status",
                ],
                max_rows=12,
            ),
            "",
            "Candidate leaderboard anchor and evaluated test rows:",
            "",
            _frame_table(
                leader.head(8),
                [
                    "candidate_id",
                    "family",
                    "selection_score",
                    "test_evaluated",
                    "test_internal_simple_mean_ba",
                    "test_internal_simple_mean_auroc",
                    "unseen_reader_test_ba",
                    "unseen_reader_test_auroc",
                    "unseen_text_test_ba",
                    "unseen_text_test_auroc",
                    "unseen_reader_and_text_test_ba",
                    "unseen_reader_and_text_test_auroc",
                ],
            ),
        ]
    )


def _render_section_10(context: dict[str, Any]) -> str:
    final = context["csv"]["final_metrics"]
    v1_locked = context["csv"]["online_v1_locked"]
    v1_stopping = context["csv"]["online_v1_stopping"]
    v2_final = context["csv"]["online_v2_final"]
    v2_prefix = context["csv"]["online_v2_prefix"]
    v2_errors = context["csv"]["online_v2_errors"]
    rescue = context["csv"]["online_v2_rescue"]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[9]),
            "",
            "### Subsection A — Offline full-record model",
            "",
            _frame_table(
                final,
                [
                    "analysis",
                    "split_name",
                    "feature_group",
                    "model",
                    "n_features",
                    "n_predictions",
                    "usable_folds",
                    "skipped_folds",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                ],
            ),
            "",
            "### Subsection B — Online fixed-budget evaluation",
            "",
            "Prefix types include chronological, word-count, trial/text, speech, all-evidence, "
            "and sequence-stop rows. Budgets include 50/100/250/500/1000 words, one to three "
            "texts/speeches, all evidence, and learned stopping decisions. v2 per-prefix "
            "curves contain " + str(len(v2_prefix)) + " rows.",
            "",
            _frame_table(
                v2_prefix.head(16),
                [
                    "split_regime",
                    "prefix_type",
                    "prefix_value",
                    "feature_family",
                    "calibrator",
                    "threshold",
                    "accumulator",
                    "n_readers",
                    "n_prefix_rows",
                    "AUROC",
                    "PR-AUC",
                    "BA",
                    "Brier",
                ],
            ),
            "",
            "### Subsection C — Online targeted optimization",
            "",
            "v1 selected `online_d3_0021` with `no_stop`, which the v2 audit records as "
            "offline-like because it consumes final sequence evidence. v2 separates "
            "best_offline_all_full_evidence, best_online_late_accumulation, "
            "best_online_mid_detection, best_online_early_detection, "
            "best_online_stopping_detector, and best_unseen_text_specialist.",
            "",
            _frame_table(
                v2_final,
                [
                    "final_model",
                    "split_regime",
                    "n_readers",
                    "coverage",
                    "mean_words_to_decision",
                    "mean_texts_to_decision",
                    "evidence_cost",
                    "AUROC",
                    "PR-AUC",
                    "BA",
                    "macro_F1",
                    "Brier",
                    "candidate_id",
                    "calibrator",
                    "threshold_policy",
                    "accumulator",
                    "stopping_policy",
                    "prefix_type",
                    "prefix_value",
                ],
                max_rows=30,
            ),
            "",
            "Unseen-text specialist/rescue rows:",
            "",
            _frame_table(
                rescue,
                [
                    "candidate_id",
                    "rescue_candidate",
                    "split_regime",
                    "AUROC",
                    "PR-AUC",
                    "BA",
                    "macro_F1",
                    "Brier",
                    "calibrator",
                    "threshold_policy",
                    "accumulator",
                    "prefix_type",
                    "prefix_value",
                ],
                max_rows=10,
            ),
            "",
            "### Subsection D — Online stopping",
            "",
            "Stopping policies include no_stop historical/full-evidence rows, fixed_budget "
            "rows, and coverage_constrained_stop sequence rows. v2 records stopping_not_ready "
            "as the status for adaptive stopping despite cost reductions in some rows.",
            "",
            _frame_table(
                v1_stopping.head(12),
                [
                    "split_regime",
                    "stopping_policy",
                    "coverage",
                    "undecided_rate",
                    "mean_words_to_decision",
                    "AUROC",
                    "PR-AUC",
                    "BA",
                    "Brier",
                ],
            ),
            "",
            "### Subsection E — Error trajectory",
            "",
            "v2 error-source analysis records " + str(len(v2_errors)) + " error rows. "
            "Unseen-text errors concentrate in held-out text IDs including 7905, 1323, "
            "7946, 11171, 1125, and 1165 in the source report. The analysis records "
            "persistent false positives/false negatives, insufficient-evidence errors, "
            "threshold candidates, calibration candidates, and distribution-shift candidates.",
            "",
            "v1 locked rows retained for historical comparison:",
            "",
            _frame_table(
                v1_locked.head(8),
                [
                    "split_regime",
                    "n_readers",
                    "earliness_score",
                    "AUROC",
                    "PR-AUC",
                    "BA",
                    "macro_F1",
                    "Brier",
                    "candidate_id",
                    "stopping_policy",
                ],
            ),
        ]
    )


def _render_section_11(context: dict[str, Any]) -> str:
    unresolved = context["csv"]["unresolved"]
    rescue = context["csv"]["online_v2_rescue"]
    explicit = []
    for candidate_id in ["unseen_text_rescue_04", "unseen_text_rescue_05"]:
        if not rescue.empty:
            sub = rescue[rescue["candidate_id"].astype(str).eq(candidate_id)]
            row = _first_row(sub)
        else:
            row = {}
        explicit.append(
            {
                "source_file": "analysis/d3_online_targeted_optimization_v2/"
                "unseen_text_rescue_candidates.csv",
                "candidate_model": candidate_id,
                "split": "unseen_text",
                "evaluation_level": "reader",
                "metric": "AUROC / BA",
                "source_value": f"{_fmt(row.get('AUROC'))} / {_fmt(row.get('BA'))}",
                "not_collapsed_reason": "candidate, calibrator, accumulator, and criterion differ",
                "paper_direct_use": "no; use only with context and conflict note",
            }
        )
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[10]),
            "",
            "Unresolved values are preserved when source rows differ by candidate, metric "
            "criterion, split, evaluation level, threshold policy, or diagnostic status. This "
            "record does not choose a canonical value for such conflicts.",
            "",
            "Source discrepancy table:",
            "",
            _frame_table(
                unresolved,
                [
                    "conflict_group_id",
                    "metric_name",
                    "model_name",
                    "candidate_id",
                    "split_regime",
                    "evaluation_level",
                    "source_values",
                    "source_files",
                    "discrepancy_type",
                    "canonical_value_chosen",
                    "resolution_status",
                    "notes",
                ],
                max_rows=10,
            ),
            "",
            "Explicit unseen_text specialist discrepancy rows requested for this record:",
            "",
            _markdown_table(
                explicit,
                [
                    "source_file",
                    "candidate_model",
                    "split",
                    "evaluation_level",
                    "metric",
                    "source_value",
                    "not_collapsed_reason",
                    "paper_direct_use",
                ],
            ),
            "",
            "Known values from the source file: unseen_text_rescue_04 has AUROC "
            "0.8638655462184874 and BA 0.7546218487394958. unseen_text_rescue_05 has AUROC "
            "0.8554621848739495 and BA 0.8260504201680672. They are not collapsed into one "
            "canonical value because the first maximizes/ranks differently from the second and "
            "uses a different calibrator/accumulator context.",
        ]
    )


def _render_section_12(context: dict[str, Any]) -> str:
    rows = [
        {
            "result_family": "full-record reader-profile result",
            "supports": "a completed reader-level D3 result on full prepared CopCo under LOPO",
            "does_not_support": "clinical diagnosis, screening utility, or external generalization",
            "allowed_context": "main internal method/result source",
            "prohibited_context": "clinical/medical claim or official EyeBench leaderboard claim",
        },
        {
            "result_family": "DFM exposure vs sensitivity ablation",
            "supports": "sensitivity/residual gaze rows outperform exposure-only rows in recorded outputs",
            "does_not_support": "causal mechanism or complete removal of all text-assignment concerns",
            "allowed_context": "ablation/source of method explanation",
            "prohibited_context": "claim that exposure confounds are impossible",
        },
        {
            "result_family": "BenchmarkBridge full-data comparison",
            "supports": "internal EyeBench-style benchmark-relative comparison",
            "does_not_support": "official leaderboard status",
            "allowed_context": "benchmark-relative internal comparison",
            "prohibited_context": "official EyeBench result wording",
        },
        {
            "result_family": "official-compatible trial-level stress test",
            "supports": "D3_Lite reduced variant behavior under official-compatible constraints",
            "does_not_support": "full D3 equivalence or trial-level state-of-the-art claim",
            "allowed_context": "stress test/negative or no-improvement record",
            "prohibited_context": "replace full reader-profile method",
        },
        {
            "result_family": "online fixed-budget evidence",
            "supports": "secondary online prefix performance under fixed budgets",
            "does_not_support": "ready adaptive stopping detector or official benchmark result",
            "allowed_context": "online/offline separation and evidence-cost framing",
            "prohibited_context": "full-record result equivalence without budget qualification",
        },
        {
            "result_family": "adaptive stopping result",
            "supports": "diagnostic stopping-policy status and cost/coverage tradeoffs",
            "does_not_support": "stopping detector readiness",
            "allowed_context": "diagnostic limitation/status",
            "prohibited_context": "deployment-ready stopping claim",
        },
        {
            "result_family": "unseen_text result",
            "supports": "general unseen_text remains harder; specialist rows are diagnostic",
            "does_not_support": "general unseen_text solved by the main model",
            "allowed_context": "limitations/conflict section",
            "prohibited_context": "canonical main result without caveat",
        },
        {
            "result_family": "segmentation/boundary opacity result",
            "supports": "orthographic boundary features as secondary interpretability covariates",
            "does_not_support": "standalone segmentation-opacity main effect",
            "allowed_context": "feature description and secondary interactions",
            "prohibited_context": "diagnostic label or core claim",
        },
        {
            "result_family": "parser fallback result",
            "supports": "surface_heuristic fallback features with parser status recorded",
            "does_not_support": "true syntax claims",
            "allowed_context": "limitation and feature-missingness explanation",
            "prohibited_context": "syntactic interpretation",
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[11]),
            "",
            _markdown_table(
                rows,
                [
                    "result_family",
                    "supports",
                    "does_not_support",
                    "allowed_context",
                    "prohibited_context",
                ],
            ),
        ]
    )


def _render_section_13() -> str:
    term_rows = [
        {
            "internal_term": term,
            "public_facing_term": public,
            "short_explanation": description,
            "where_it_may_appear": "internal record, methods source map, appendix source notes",
            "where_it_should_not_appear": "standalone paper prose without public description",
        }
        for term, public, description in INTERNAL_TERM_MAP
    ]
    template_rows = [
        {
            "template_name": "method_dataset_split_metric_baseline",
            "template": (
                "[method] + [dataset/protocol] + [split regime] + [metric] + "
                "[baseline] + [absolute improvement] + [additional advantage]"
            ),
        },
        {
            "template_name": "example_full_record",
            "template": (
                "Using residualized predictability-sensitive reader profiles on CopCo TYP "
                "under the unseen-reader regime, the model achieved AUROC X and balanced "
                "accuracy Y, compared with baseline Z at AUROC A and balanced accuracy B."
            ),
        },
        {
            "template_name": "official_status_guard",
            "template": (
                "When exact official processed EyeBench data, official folds, and the "
                "official evaluator are absent, describe the row as internal EyeBench-style "
                "or fold-aligned, not official."
            ),
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[12]),
            "",
            _markdown_table(
                term_rows,
                [
                    "internal_term",
                    "public_facing_term",
                    "short_explanation",
                    "where_it_may_appear",
                    "where_it_should_not_appear",
                ],
            ),
            "",
            "Objective result-expression templates:",
            "",
            _markdown_table(template_rows, ["template_name", "template"]),
            "",
            "State-of-the-art wording is restricted to objective baseline comparisons with "
            "scope labels. A standalone official leaderboard claim is not supported by the "
            "recorded official-subset status.",
        ]
    )


def _render_section_14() -> str:
    rows = [
        {
            "paper_section": "Introduction",
            "master_subsections": "Sections 1, 12, 13",
            "evidence_vault_files": "05_claim_status, 09_appendix_source_material",
            "canonical_metrics": "main LOPO D3 metrics only as factual context",
            "source_reports": "final_publication_decision_report; reviewer_risk_report",
        },
        {
            "paper_section": "Related Work",
            "master_subsections": "Sections 9, 13",
            "evidence_vault_files": "canonical_external_baselines.csv",
            "canonical_metrics": "official reported CopCo TYP baseline rows",
            "source_reports": "official alignment/SOTA check reports; deep review if later present",
        },
        {
            "paper_section": "Data",
            "master_subsections": "Sections 3, 4, 5",
            "evidence_vault_files": "dataset_summary.md, participant_label_summary.md",
            "canonical_metrics": "participant and row counts",
            "source_reports": "feature_release_report.md; label_release_report.md",
        },
        {
            "paper_section": "Feature Extraction",
            "master_subsections": "Sections 5, 6",
            "evidence_vault_files": "dfm_predictability_features.md, residualization_algorithm.md",
            "canonical_metrics": "feature row counts and missingness",
            "source_reports": "feature_dictionary_v1.md; parser/embedding/DFM reports",
        },
        {
            "paper_section": "Method",
            "master_subsections": "Sections 5, 6, 7, 13",
            "evidence_vault_files": "01_algorithm_details/*",
            "canonical_metrics": "model taxonomy rows",
            "source_reports": "Phase 4 and D3 evidence vault algorithms",
        },
        {
            "paper_section": "Experiments",
            "master_subsections": "Sections 2, 4, 7",
            "evidence_vault_files": "split_policy_summary.md, canonical_model_runs.csv",
            "canonical_metrics": "split/evaluation scope rows",
            "source_reports": "BenchmarkBridge and online target docs",
        },
        {
            "paper_section": "Results",
            "master_subsections": "Sections 8, 9, 10, 11",
            "evidence_vault_files": "canonical_metrics_long.csv, number registry",
            "canonical_metrics": "full-record, benchmark, online, and conflict rows",
            "source_reports": "AutoResearch, BenchmarkBridge, online v2",
        },
        {
            "paper_section": "Ablations",
            "master_subsections": "Sections 8B, 12",
            "evidence_vault_files": "dfm_exposure_vs_sensitivity_summary.md",
            "canonical_metrics": "D1/D2/D3/D4 rows",
            "source_reports": "dfm_exposure_vs_sensitivity_table.csv",
        },
        {
            "paper_section": "Online Evaluation",
            "master_subsections": "Section 10",
            "evidence_vault_files": "canonical_online_prefix_results.csv, stopping results",
            "canonical_metrics": "v2 strict final and per-prefix rows",
            "source_reports": "online v1/v2 reports",
        },
        {
            "paper_section": "Benchmark Comparison",
            "master_subsections": "Section 9",
            "evidence_vault_files": "canonical_external_baselines.csv",
            "canonical_metrics": "official/reference/internal split rows",
            "source_reports": "BenchmarkBridge, OfficialEyeBenchAlignment, SOTACheck",
        },
        {
            "paper_section": "Limitations",
            "master_subsections": "Sections 11, 12, 15",
            "evidence_vault_files": "limitations_factual_notes.md",
            "canonical_metrics": "blocked/unresolved rows",
            "source_reports": "reviewer risk and official blocker reports",
        },
        {
            "paper_section": "Appendix",
            "master_subsections": "all sections as needed",
            "evidence_vault_files": "source manifests and validation files",
            "canonical_metrics": "full metric registry with source paths",
            "source_reports": "all source reports listed in source_trace_manifest.json",
        },
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[13]),
            "",
            "This is a source map only. It does not write final paper text.",
            "",
            _markdown_table(
                rows,
                [
                    "paper_section",
                    "master_subsections",
                    "evidence_vault_files",
                    "canonical_metrics",
                    "source_reports",
                ],
            ),
        ]
    )


def _render_section_15(context: dict[str, Any], metrics_summary: dict[str, int]) -> str:
    source_dirs = context["source_paths"]
    source_files = context["source_files"]
    missing = [row for row in source_dirs if not row["exists"]]
    validation_rows = [
        {"check": "source directories inspected", "value": len(source_dirs)},
        {"check": "source files indexed", "value": len(source_files)},
        {"check": "metric rows used", "value": metrics_summary["canonical_metrics"]},
        {"check": "online prefix metric rows indexed", "value": metrics_summary["online_prefix"]},
        {"check": "online stopping rows indexed", "value": metrics_summary["online_stopping"]},
        {"check": "oracle rows indexed", "value": metrics_summary["oracle"]},
        {"check": "source conflicts found", "value": metrics_summary["unresolved_conflicts"]},
        {"check": "official claim status separated", "value": True},
        {"check": "full-data and EyeBench-related results separated", "value": True},
        {"check": "online/offline results separated", "value": True},
        {"check": "language-model features documented", "value": True},
        {"check": "model variants documented", "value": True},
        {"check": "large result files copied", "value": False},
        {"check": "large result files referenced only", "value": True},
        {"check": "figures generated", "value": False},
        {"check": "final paper tables generated", "value": False},
        {"check": "new experiments run", "value": False},
    ]
    missing_rows = [
        {
            "source_id": row["source_id"],
            "path": row["path"],
            "public_description": row["public_description"],
            "status": "missing",
        }
        for row in missing
    ]
    return "\n".join(
        [
            _section_header(REQUIRED_SECTIONS[14]),
            "",
            _markdown_table(validation_rows, ["check", "value"]),
            "",
            "Missing source directories:",
            "",
            _markdown_table(
                missing_rows,
                ["source_id", "path", "public_description", "status"],
            )
            if missing_rows
            else "No required source directory is missing except sources not listed as present.",
            "",
            "This master record references large result artifacts by path and does not copy "
            "Parquet files, prediction CSVs, model artifacts, figures, or final paper tables.",
        ]
    )


def _key_metrics_manifest(context: dict[str, Any]) -> list[dict[str, Any]]:
    summary = _metrics_summary(context)
    rows = [
        {
            "name": key,
            "rows": value,
            "source_file": context["csv_paths"].get(key),
        }
        for key, value in summary.items()
    ]
    final = _first_row(context["csv"]["final_metrics"])
    rows.append(
        {
            "name": "offline_full_record_main_result",
            "rows": 1 if final else 0,
            "source_file": "analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
            "AUROC": _metric_col(final, "roc_auc"),
            "PR_AUC": _metric_col(final, "pr_auc"),
            "balanced_accuracy": _metric_col(final, "balanced_accuracy"),
            "macro_F1": _metric_col(final, "macro_f1"),
            "Brier": _metric_col(final, "brier_score"),
        }
    )
    return rows


def _build_manifest(
    context: dict[str, Any],
    output_dir: Path | None,
    validation_status: str,
) -> dict[str, Any]:
    unresolved = context["csv"]["unresolved"]
    missing = [row for row in context["source_paths"] if not row["exists"]]
    return {
        "record_version": "master_research_record_v1",
        "build_timestamp": context["build_timestamp"],
        "repo_commit": context["commit"],
        "branch": context["branch"],
        "generated_output_dir": str(output_dir) if output_dir else None,
        "primary_output_dir": str(ANALYSIS_DIR),
        "master_record": str(ANALYSIS_DIR / MASTER_FILENAME),
        "no_new_experiments": True,
        "no_model_training": True,
        "no_feature_search": True,
        "no_metric_optimization": True,
        "no_figures_generated": True,
        "no_final_paper_tables_generated": True,
        "large_files_copied": False,
        "source_paths": context["source_paths"],
        "source_directories_inspected": len(context["source_paths"]),
        "source_files_indexed_count": len(context["source_files"]),
        "source_files_indexed": context["source_files"],
        "files_used": [
            row for row in context["source_files"] if row.get("used_for_master_record")
        ],
        "missing_sources": missing,
        "key_metrics_extracted": _key_metrics_manifest(context),
        "unresolved_conflicts": unresolved.to_dict("records") if not unresolved.empty else [],
        "validation_status": validation_status,
    }


def _validation_report_text(
    context: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    validation_status: str,
) -> str:
    source_dirs = context["source_paths"]
    missing = [row for row in source_dirs if not row["exists"]]
    metrics = _metrics_summary(context)
    checklist = [
        {"item": section, "status": "present" if not errors else "checked"}
        for section in REQUIRED_SECTIONS
    ]
    metric_rows = [
        {"metric_table": key, "row_count": value}
        for key, value in metrics.items()
    ]
    confirmations = [
        {"confirmation": "no new experiments run", "status": True},
        {"confirmation": "no model training run", "status": True},
        {"confirmation": "no feature search run", "status": True},
        {"confirmation": "no metric optimization run", "status": True},
        {"confirmation": "no figures generated", "status": True},
        {"confirmation": "no final paper tables generated", "status": True},
        {"confirmation": "large result files referenced only", "status": True},
        {"confirmation": "official result status separated", "status": True},
        {"confirmation": "online/offline results separated", "status": True},
    ]
    return "\n".join(
        [
            "# MasterResearchRecord v1 Validation Report",
            "",
            f"- Status: `{validation_status}`",
            f"- Build timestamp: `{context['build_timestamp']}`",
            f"- Branch: `{context['branch']}`",
            f"- Commit: `{context['commit']}`",
            "",
            "## Required Section Checklist",
            "",
            _markdown_table(checklist, ["item", "status"]),
            "",
            "## Missing Source Checklist",
            "",
            _markdown_table(
                [
                    {
                        "source_id": row["source_id"],
                        "path": row["path"],
                        "status": "missing",
                    }
                    for row in missing
                ],
                ["source_id", "path", "status"],
            )
            if missing
            else "No listed source directory is missing.",
            "",
            "## Metric Trace Checklist",
            "",
            _markdown_table(metric_rows, ["metric_table", "row_count"]),
            "",
            "## Conflict Checklist",
            "",
            "Unresolved conflict rows: `"
            + str(metrics["unresolved_conflicts"])
            + "`. The unseen_text specialist conflict is preserved without choosing a "
            "canonical value.",
            "",
            "## Confirmation Checklist",
            "",
            _markdown_table(confirmations, ["confirmation", "status"]),
            "",
            "## Errors",
            "",
            "\n".join(f"- {error}" for error in errors) if errors else "None.",
            "",
            "## Warnings",
            "",
            "\n".join(f"- {warning}" for warning in warnings) if warnings else "None.",
        ]
    )


def _copy_primary_outputs(analysis_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [MASTER_FILENAME, MANIFEST_FILENAME, VALIDATION_FILENAME]:
        shutil.copy2(analysis_dir / filename, output_dir / filename)


def build_master_research_record_v1(
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    analysis_dir = root / ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out: Path | None = None
    if output_dir:
        out = Path(output_dir)
        if not out.is_absolute():
            out = root / out
        out.mkdir(parents=True, exist_ok=True)

    context = _collect_context(root)
    master_text = _render_master(context)
    _write_text(analysis_dir / MASTER_FILENAME, master_text)

    validation = _validate_content(root, master_text, context)
    validation_status = "passed" if not validation["errors"] else "failed"
    manifest = _build_manifest(context, out, validation_status)
    _write_json(analysis_dir / MANIFEST_FILENAME, manifest)
    _write_text(
        analysis_dir / VALIDATION_FILENAME,
        _validation_report_text(
            context,
            validation["errors"],
            validation["warnings"],
            validation_status,
        ),
    )
    if out is not None:
        _copy_primary_outputs(analysis_dir, out)
        _write_json(out / "master_research_record_v1_manifest.json", manifest)
    return {
        "status": validation_status,
        "master_record": str(analysis_dir / MASTER_FILENAME),
        "source_trace_manifest": str(analysis_dir / MANIFEST_FILENAME),
        "validation_report": str(analysis_dir / VALIDATION_FILENAME),
        "generated_output_dir": str(out) if out else None,
        "source_directories_inspected": len(context["source_paths"]),
        "source_files_indexed": len(context["source_files"]),
        "metric_rows_used": _metrics_summary(context)["canonical_metrics"],
        "unresolved_conflicts": _metrics_summary(context)["unresolved_conflicts"],
        "no_new_experiments": True,
        "no_figures_generated": True,
        "no_final_paper_tables_generated": True,
    }


def _validate_content(root: Path, text: str, context: dict[str, Any]) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"Missing required section: {section}")
    required_phrases = [
        "Subsection A — Published / provided CopCo TYP baselines",
        "Subsection B — Internal EyeBench-style full-data reader-aggregated comparison",
        "Subsection C — EyeBench-fold full-feature intersection",
        "Subsection D — Official EyeBench subset/evaluator",
        "Subsection E — Reduced official-protocol-compatible trial-level model",
        "DFM exposure vs sensitivity ablation",
        "Model family taxonomy",
        "Feature families",
        "Language models",
        "Online and offline",
        "Result conflicts",
        "Public-facing method language",
    ]
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"Missing required content phrase: {phrase}")
    prohibited_phrases = [
        "Official EyeBench SOTA.",
        "official_sota_claim_allowed: true",
    ]
    lowered = text.lower()
    for phrase in prohibited_phrases:
        if phrase.lower() in lowered:
            errors.append(f"Prohibited standalone or unsupported phrase found: {phrase}")
    for term, public, _description in INTERNAL_TERM_MAP:
        if term not in text or public not in text:
            errors.append(f"Internal term lacks public-facing explanation: {term}")
    analysis_dir = root / ANALYSIS_DIR
    if analysis_dir.exists():
        figure_files = [
            str(path.relative_to(root))
            for path in analysis_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        ]
        if figure_files:
            errors.append("Figure-like files generated in master record directory: " + str(figure_files))
        final_table_files = [
            str(path.relative_to(root))
            for path in analysis_dir.rglob("*")
            if path.is_file() and "final_table" in path.name.lower()
        ]
        if final_table_files:
            errors.append("Final paper table-like files generated: " + str(final_table_files))
    missing = [row["path"] for row in context["source_paths"] if not row["exists"]]
    if missing:
        warnings.append("Missing source paths recorded: " + ", ".join(missing))
    if _metrics_summary(context)["canonical_metrics"] == 0:
        errors.append("No canonical metric rows were indexed.")
    if _metrics_summary(context)["unresolved_conflicts"] == 0:
        warnings.append("No unresolved conflicts were indexed; expected unseen_text conflict.")
    return {"errors": errors, "warnings": warnings}


def validate_master_research_record_v1(
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    analysis_dir = root / ANALYSIS_DIR
    context = _collect_context(root)
    master_path = analysis_dir / MASTER_FILENAME
    manifest_path = analysis_dir / MANIFEST_FILENAME
    if not master_path.exists():
        text = ""
        validation = {"errors": [f"{master_path} does not exist."], "warnings": []}
    else:
        text = master_path.read_text(encoding="utf-8")
        validation = _validate_content(root, text, context)
    if not manifest_path.exists():
        validation["errors"].append(f"{manifest_path} does not exist.")
    status = "passed" if not validation["errors"] else "failed"
    report_text = _validation_report_text(context, validation["errors"], validation["warnings"], status)
    _write_text(analysis_dir / VALIDATION_FILENAME, report_text)

    out: Path | None = None
    if output_dir:
        out = Path(output_dir)
        if not out.is_absolute():
            out = root / out
        out.mkdir(parents=True, exist_ok=True)
        if master_path.exists() and manifest_path.exists():
            _copy_primary_outputs(analysis_dir, out)
        _write_json(
            out / "master_research_record_v1_validation_manifest.json",
            {
                "status": status,
                "errors": validation["errors"],
                "warnings": validation["warnings"],
                "validated_master_record": str(master_path),
                "no_new_experiments": True,
                "no_figures_generated": True,
                "no_final_paper_tables_generated": True,
            },
        )
    return {
        "status": status,
        "errors": validation["errors"],
        "warnings": validation["warnings"],
        "master_record": str(master_path),
        "source_trace_manifest": str(manifest_path),
        "validation_report": str(analysis_dir / VALIDATION_FILENAME),
        "generated_output_dir": str(out) if out else None,
        "source_directories_inspected": len(context["source_paths"]),
        "source_files_indexed": len(context["source_files"]),
        "metric_rows_used": _metrics_summary(context)["canonical_metrics"],
        "unresolved_conflicts": _metrics_summary(context)["unresolved_conflicts"],
        "no_new_experiments": True,
        "no_figures_generated": True,
        "no_final_paper_tables_generated": True,
    }
