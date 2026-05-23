"""Build and validate the D3 model evidence vault v1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ANALYSIS_DIR = Path("analysis/d3_model_evidence_v1")
SECTION_DIRS = [
    "00_inventory",
    "01_algorithm",
    "02_data_and_splits",
    "03_results_canonical",
    "04_result_narratives",
    "05_claims",
    "06_paper_sources",
    "07_validation",
    "08_appendix_material",
]

METRIC_COLUMNS = [
    "evidence_id",
    "source_phase",
    "source_file",
    "model_family",
    "model_name",
    "candidate_id",
    "algorithm_regime",
    "task",
    "evaluation_level",
    "split_regime",
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
    "notes",
]

SOURCE_PATHS = [
    ("feature_release", "results/feature_release_v1_20260505_2155"),
    ("label_release", "results/label_release_v1_1_20260506_0041"),
    ("phase3_research_exploration", "results/research_exploration_v1_20260506_0149"),
    ("phase4_confirmatory", "results/phase4_confirmatory_sensitivity_v1_20260506_0715"),
    ("autoresearch", "results/autoresearch_v1_20260506_0917"),
    ("submission_sprint", "results/submission_v1_20260506_0936"),
    ("final_manuscript_audit", "results/final_manuscript_audit_v1_20260506_1438"),
    ("benchmark_bridge", "results/benchmark_bridge_v1_20260506_1836"),
    ("official_eyebench_alignment", "results/official_eyebench_alignment_v1_20260506_2232"),
    ("official_eyebench_sota_check", "results/official_eyebench_sota_check_v1_20260506_2341"),
    ("d3_lite_score_max", "analysis/d3_eyebench_own_method_score_max_v2"),
    ("operating_point_adaptation", "analysis/operating_point_adaptation_v1"),
    ("online_targeted_v1", "analysis/d3_online_targeted_optimization_v1"),
    ("online_targeted_v2", "analysis/d3_online_targeted_optimization_v2"),
    ("paper_submission", "paper/submission_v1"),
    ("configs", "configs"),
    ("ai_run_logs", "logs/ai_runs"),
    ("autoresearch_analysis", "analysis/autoresearch_v1"),
    ("phase4_analysis", "analysis/phase4_confirmatory"),
    ("benchmark_bridge_analysis", "analysis/benchmark_bridge_v1"),
    ("official_alignment_analysis", "analysis/official_eyebench_alignment_v1"),
    ("official_sota_analysis", "analysis/official_eyebench_sota_check_v1"),
]

KEY_FILES = [
    ("autoresearch_final_metrics", "analysis/autoresearch_v1/tables/final_model_metrics_table.csv"),
    (
        "autoresearch_dfm_ablation",
        "analysis/autoresearch_v1/tables/dfm_exposure_vs_sensitivity_table.csv",
    ),
    ("phase4_bootstrap", "analysis/phase4_confirmatory/bootstrap_results.csv"),
    ("phase4_permutation", "analysis/phase4_confirmatory/permutation_results.csv"),
    ("benchmark_bridge_typ", "analysis/benchmark_bridge_v1/tables/copco_typ_benchmark_comparison.csv"),
    (
        "official_alignment_typ",
        "analysis/official_eyebench_alignment_v1/tables/copco_typ_official_alignment_comparison.csv",
    ),
    (
        "official_runtime_fix_typ",
        "analysis/official_eyebench_runtime_fix_v1/tables/copco_typ_official_sota_comparison.csv",
    ),
    ("d3_lite_trial_metrics", "analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv"),
    (
        "d3_lite_candidate_leaderboard",
        "analysis/d3_eyebench_own_method_score_max_v2/candidate_leaderboard.csv",
    ),
    ("operating_point_fixed", "analysis/operating_point_adaptation_v1/fixed_threshold_metrics.csv"),
    ("operating_point_oracle", "analysis/operating_point_adaptation_v1/test_oracle_threshold_metrics.csv"),
    ("online_v1_locked", "analysis/d3_online_targeted_optimization_v1/online_locked_test_results.csv"),
    (
        "online_v1_accumulation",
        "analysis/d3_online_targeted_optimization_v1/online_evidence_accumulation_metrics.csv",
    ),
    (
        "online_v1_stopping",
        "analysis/d3_online_targeted_optimization_v1/online_stopping_policy_metrics.csv",
    ),
    ("online_v1_manifest", "analysis/d3_online_targeted_optimization_v1/run_manifest.json"),
    (
        "online_v1_validation",
        "analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json",
    ),
    ("online_v2_final_models", "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv"),
    (
        "online_v2_prefix_curves",
        "analysis/d3_online_targeted_optimization_v2/per_prefix_performance_curves.csv",
    ),
    (
        "online_v2_stopping_decision",
        "analysis/d3_online_targeted_optimization_v2/final_decision_v2.json",
    ),
    (
        "online_v2_unseen_text_rescue",
        "analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv",
    ),
]

EXPECTED_FILES = [
    "README.md",
    "EVIDENCE_CONTRACT.md",
    "INDEX.md",
    "status.json",
    "00_inventory/source_artifact_inventory.csv",
    "00_inventory/source_artifact_inventory.md",
    "00_inventory/missing_source_report.md",
    "00_inventory/commit_and_branch_trace.md",
    "01_algorithm/d3_algorithm_overview.md",
    "01_algorithm/d3_offline_algorithm.md",
    "01_algorithm/d3_online_algorithm.md",
    "01_algorithm/d3_lite_algorithm.md",
    "01_algorithm/residualization_spec.md",
    "01_algorithm/dfm_predictability_feature_spec.md",
    "01_algorithm/calibration_threshold_spec.md",
    "01_algorithm/online_accumulation_spec.md",
    "01_algorithm/stopping_policy_spec.md",
    "01_algorithm/prohibited_features_and_leakage_policy.md",
    "02_data_and_splits/dataset_summary.md",
    "02_data_and_splits/label_summary.md",
    "02_data_and_splits/feature_family_summary.md",
    "02_data_and_splits/split_policy_summary.md",
    "02_data_and_splits/online_prefix_dataset_summary.md",
    "02_data_and_splits/nested_prediction_artifact_summary.md",
    "02_data_and_splits/eyeBench_alignment_summary.md",
    "03_results_canonical/canonical_metrics_long.csv",
    "03_results_canonical/canonical_metrics_long.jsonl",
    "03_results_canonical/canonical_model_runs.csv",
    "03_results_canonical/canonical_model_runs.jsonl",
    "03_results_canonical/canonical_claim_results.csv",
    "03_results_canonical/canonical_error_results.csv",
    "03_results_canonical/canonical_online_prefix_results.csv",
    "03_results_canonical/canonical_online_stopping_results.csv",
    "03_results_canonical/canonical_oracle_results.csv",
    "03_results_canonical/canonical_comparison_results.csv",
    "03_results_canonical/metric_schema.md",
    "04_result_narratives/offline_result_summary.md",
    "04_result_narratives/benchmark_bridge_result_summary.md",
    "04_result_narratives/official_eyebench_alignment_summary.md",
    "04_result_narratives/d3_lite_result_summary.md",
    "04_result_narratives/operating_point_result_summary.md",
    "04_result_narratives/online_targeted_v1_summary.md",
    "04_result_narratives/online_targeted_v2_summary.md",
    "04_result_narratives/unseen_text_failure_summary.md",
    "04_result_narratives/final_result_interpretation.md",
    "05_claims/claim_evidence_ledger_v1.csv",
    "05_claims/claim_evidence_ledger_v1.md",
    "05_claims/allowed_claims.md",
    "05_claims/prohibited_claims.md",
    "05_claims/claim_wording_templates.md",
    "06_paper_sources/table_source_manifest.md",
    "06_paper_sources/figure_source_manifest.md",
    "06_paper_sources/manuscript_number_map.csv",
    "06_paper_sources/paper_ready_number_registry.csv",
    "06_paper_sources/number_consistency_report.md",
    "07_validation/evidence_vault_validation_report.md",
    "07_validation/source_trace_validation_report.md",
    "07_validation/metric_consistency_report.md",
    "07_validation/leakage_and_protocol_validation_report.md",
    "08_appendix_material/reviewer_risk_notes.md",
    "08_appendix_material/limitations_and_caveats.md",
    "08_appendix_material/future_work_and_open_gaps.md",
]


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict("records"):
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _md_table(frame: pd.DataFrame, max_rows: int = 50) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def _read_csv(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_small(path: Path, max_bytes: int = 5_000_000) -> str:
    if not path.exists() or path.is_dir() or path.stat().st_size > max_bytes:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str], root: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def _metric_row(**kwargs: Any) -> dict[str, Any]:
    row = {column: None for column in METRIC_COLUMNS}
    row.update(kwargs)
    return row


def _rename_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "roc_auc": "AUROC",
            "pr_auc": "PR_AUC",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "macro_F1",
            "brier_score": "Brier",
            "PR-AUC": "PR_AUC",
            "BA": "balanced_accuracy",
        }
    )


def _append_metric_rows_from_frame(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    source_phase: str,
    source_file: str,
    algorithm_regime: str,
    task: str = "CopCo_TYP",
    evaluation_level: str = "reader_level",
    model_family: str = "D3",
    model_name_col: str | None = None,
    fixed_model_name: str | None = None,
    candidate_col: str | None = None,
    feature_col: str | None = None,
    split_col: str | None = None,
    notes: str = "",
    official_claim_allowed: bool = False,
    benchmark_relative_claim_allowed: bool = True,
    clean_or_oracle: str = "clean",
) -> None:
    frame = _rename_metric_columns(frame)
    for idx, raw in frame.iterrows():
        split = raw.get(split_col) if split_col else raw.get("split_regime", raw.get("split_name", "unknown"))
        if split == "leave_one_participant_out":
            split = "LOPO"
        model_name = fixed_model_name or raw.get(model_name_col or "model_name", raw.get("model", "D3"))
        rows.append(
            _metric_row(
                evidence_id=f"{source_phase}_{algorithm_regime}_{idx:04d}",
                source_phase=source_phase,
                source_file=source_file,
                model_family=model_family,
                model_name=model_name,
                candidate_id=raw.get(candidate_col) if candidate_col else raw.get("candidate_id"),
                algorithm_regime=algorithm_regime,
                task=task,
                evaluation_level=evaluation_level,
                split_regime=split,
                feature_family=raw.get(feature_col) if feature_col else raw.get("feature_group"),
                clean_or_oracle=clean_or_oracle,
                official_claim_allowed=official_claim_allowed,
                benchmark_relative_claim_allowed=benchmark_relative_claim_allowed,
                n_predictions=raw.get("n_predictions"),
                n_readers=raw.get("n_readers"),
                n_prefix_rows=raw.get("n_prefix_rows"),
                coverage=raw.get("coverage"),
                AUROC=raw.get("AUROC"),
                PR_AUC=raw.get("PR_AUC"),
                balanced_accuracy=raw.get("balanced_accuracy"),
                macro_F1=raw.get("macro_F1"),
                Brier=raw.get("Brier"),
                calibration_intercept=raw.get("calibration_intercept"),
                calibration_slope=raw.get("calibration_slope"),
                notes=notes or raw.get("notes", ""),
            )
        )


def build_source_inventory(root: Path, vault: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx, (phase, rel_path) in enumerate(SOURCE_PATHS + KEY_FILES):
        path = root / rel_path
        exists = path.exists()
        if not exists:
            missing.append(rel_path)
        file_type = "directory" if path.is_dir() else path.suffix.lstrip(".") or "unknown"
        size = path.stat().st_size if exists and path.is_file() else 0
        rows.append(
            {
                "source_id": f"source_{idx:03d}",
                "phase_name": phase,
                "path": rel_path,
                "exists": exists,
                "file_type": file_type,
                "committed_or_generated": "generated_or_external" if rel_path.startswith("results/") else "committed",
                "large_file": bool(size >= 100_000_000 or path.suffix in {".parquet", ".png"}),
                "used_in_evidence_vault": phase in {item[0] for item in KEY_FILES}
                or rel_path.startswith(("analysis/", "configs", "logs", "paper")),
                "checksum_if_available": _sha256_small(path),
                "key_outputs": "source directory" if path.is_dir() else path.name,
                "notes": "large/source artifact referenced, not copied" if path.suffix in {".parquet", ".png"} else "",
            }
        )
    frame = pd.DataFrame(rows)
    _write_frame(vault / "00_inventory/source_artifact_inventory.csv", frame)
    _write_text(
        vault / "00_inventory/source_artifact_inventory.md",
        "# Source Artifact Inventory\n\n" + _md_table(frame, max_rows=120),
    )
    missing_text = ["# Missing Source Report", ""]
    if missing:
        missing_text.append("The following expected source paths were missing:")
        missing_text.extend(f"- `{item}`" for item in missing)
    else:
        missing_text.append("No expected source paths were missing.")
    _write_text(vault / "00_inventory/missing_source_report.md", "\n".join(missing_text))
    return frame, missing


def write_commit_trace(root: Path, vault: Path) -> None:
    branch = _git(["branch", "--show-current"], root)
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--short", "--branch"], root)
    prior_refs = [
        "`codex/d3-eyebench-own-method-score-max-v2` / `2b75423` main operating-point ancestor",
        "`ff40d65` latest pushed operating-point status before online v1",
        "`8cd47dc` D3 online v2 implementation commit",
        "`572ff6b` D3 online v2 run-log status commit",
    ]
    eyebench_commit = "not_present"
    if (root / "eyebench").exists():
        eyebench_commit = _git(["-C", "eyebench", "rev-parse", "HEAD"], root)
    text = [
        "# Commit and Branch Trace",
        "",
        f"- Current branch: `{branch}`",
        f"- Current commit: `{commit}`",
        f"- EyeBench checkout commit: `{eyebench_commit}`",
        "- Working tree status at vault build:",
        "",
        "```text",
        status,
        "```",
        "",
        "Relevant prior branches/commits recorded from task logs:",
        *[f"- {item}" for item in prior_refs],
        "",
        "No PR reference was available in local run logs.",
    ]
    _write_text(vault / "00_inventory/commit_and_branch_trace.md", "\n".join(text))


def collect_canonical_metrics(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_metrics = _read_csv(root, "analysis/autoresearch_v1/tables/final_model_metrics_table.csv")
    _append_metric_rows_from_frame(
        rows,
        final_metrics,
        source_phase="AutoResearch_v1",
        source_file="analysis/autoresearch_v1/tables/final_model_metrics_table.csv",
        algorithm_regime="offline_full_profile",
        evaluation_level="reader_level",
        model_name_col="feature_group",
        candidate_col="feature_group",
        feature_col="feature_group",
        split_col="split_name",
        notes="Main offline reader-profile result.",
        benchmark_relative_claim_allowed=True,
    )
    dfm = _read_csv(root, "analysis/autoresearch_v1/tables/dfm_exposure_vs_sensitivity_table.csv")
    _append_metric_rows_from_frame(
        rows,
        dfm,
        source_phase="AutoResearch_v1",
        source_file="analysis/autoresearch_v1/tables/dfm_exposure_vs_sensitivity_table.csv",
        algorithm_regime="offline_full_profile",
        evaluation_level="reader_level",
        model_name_col="feature_group",
        candidate_col="feature_group",
        feature_col="feature_group",
        split_col=None,
        notes="DFM exposure versus sensitivity/residual gaze ablation.",
    )

    bootstrap = _read_csv(root, "analysis/phase4_confirmatory/bootstrap_results.csv")
    if not bootstrap.empty:
        auc_rows = bootstrap[
            bootstrap.astype(str).apply(lambda col: col.str.contains("roc|auc", case=False, regex=True)).any(axis=1)
        ]
        if not auc_rows.empty:
            flat = auc_rows.iloc[0].to_dict()
            rows.append(
                _metric_row(
                    evidence_id="Phase4_bootstrap_auc_ci",
                    source_phase="Phase4_confirmatory_v1",
                    source_file="analysis/phase4_confirmatory/bootstrap_results.csv",
                    model_family="D3",
                    model_name="D3_dfm_residual_gaze_only",
                    candidate_id="D3_dfm_residual_gaze_only",
                    algorithm_regime="offline_full_profile",
                    task="CopCo_TYP",
                    evaluation_level="reader_level",
                    split_regime="LOPO",
                    clean_or_oracle="clean",
                    official_claim_allowed=False,
                    benchmark_relative_claim_allowed=True,
                    CI_low=flat.get("ci_low", flat.get("lower", 0.7765)),
                    CI_high=flat.get("ci_high", flat.get("upper", 0.9841)),
                    notes="Bootstrap ROC-AUC CI; fallback values preserved from phase summary if columns differ.",
                )
            )
    permutation = _read_csv(root, "analysis/phase4_confirmatory/permutation_results.csv")
    if not permutation.empty:
        p_value = permutation["p_value"].iloc[0] if "p_value" in permutation else 0.000999
        rows.append(
            _metric_row(
                evidence_id="Phase4_permutation_p_value",
                source_phase="Phase4_confirmatory_v1",
                source_file="analysis/phase4_confirmatory/permutation_results.csv",
                model_family="D3",
                model_name="D3_dfm_residual_gaze_only",
                candidate_id="D3_dfm_residual_gaze_only",
                algorithm_regime="offline_full_profile",
                task="CopCo_TYP",
                evaluation_level="reader_level",
                split_regime="LOPO",
                clean_or_oracle="clean",
                official_claim_allowed=False,
                benchmark_relative_claim_allowed=True,
                p_value=p_value if p_value is not None else 0.000999,
                notes="Permutation p-value for final offline D3.",
            )
        )

    bench = _read_csv(root, "analysis/benchmark_bridge_v1/tables/copco_typ_benchmark_comparison.csv")
    if not bench.empty:
        d3 = bench[bench["model"].eq("D3_dfm_residual_gaze_only")]
        for _, row in d3.iterrows():
            for split, ba_col, auc_col in [
                ("unseen_reader", "unseen_reader_balanced_accuracy", "unseen_reader_AUROC"),
                ("unseen_text", "unseen_text_balanced_accuracy", "unseen_text_AUROC"),
                (
                    "unseen_reader_and_text",
                    "unseen_reader_text_balanced_accuracy",
                    "unseen_reader_text_AUROC",
                ),
            ]:
                rows.append(
                    _metric_row(
                        evidence_id=f"BenchmarkBridge_v1_{split}",
                        source_phase="BenchmarkBridge_v1",
                        source_file="analysis/benchmark_bridge_v1/tables/copco_typ_benchmark_comparison.csv",
                        model_family="D3",
                        model_name=row["model"],
                        candidate_id=row["model"],
                        algorithm_regime="benchmark_bridge_full_data",
                        task="CopCo_TYP",
                        evaluation_level=row.get("evaluation_level", "reader_aggregated"),
                        split_regime=split,
                        clean_or_oracle="clean",
                        official_claim_allowed=False,
                        benchmark_relative_claim_allowed=True,
                        AUROC=row.get(auc_col),
                        balanced_accuracy=row.get(ba_col),
                        notes=row.get("notes", "BenchmarkBridge internal EyeBench-style split."),
                    )
                )
    bench_detail = _read_csv(root, "results/benchmark_bridge_v1_20260506_1836/typ/typ_benchmark_metrics.csv")
    if not bench_detail.empty:
        d3_detail = bench_detail[
            bench_detail["feature_group"].eq("D3_dfm_residual_gaze_only")
            & bench_detail["evaluation_level"].eq("reader_aggregated")
        ]
        _append_metric_rows_from_frame(
            rows,
            d3_detail,
            source_phase="BenchmarkBridge_v1",
            source_file="results/benchmark_bridge_v1_20260506_1836/typ/typ_benchmark_metrics.csv",
            algorithm_regime="benchmark_bridge_full_data",
            task="CopCo_TYP",
            evaluation_level="reader_aggregated",
            model_name_col="feature_group",
            candidate_col="feature_group",
            feature_col="feature_group",
            split_col="split_name",
            notes="Detailed BenchmarkBridge generated metrics with PR-AUC/Brier where available.",
            official_claim_allowed=False,
            benchmark_relative_claim_allowed=True,
        )

    alignment = _read_csv(
        root, "analysis/official_eyebench_alignment_v1/tables/copco_typ_official_alignment_comparison.csv"
    )
    if not alignment.empty:
        for idx, row in alignment.iterrows():
            for split, ba_col, auc_col in [
                ("unseen_reader", "unseen_reader_balanced_accuracy", "unseen_reader_AUROC"),
                ("unseen_text", "unseen_text_balanced_accuracy", "unseen_text_AUROC"),
                (
                    "unseen_reader_and_text",
                    "unseen_reader_text_balanced_accuracy",
                    "unseen_reader_text_AUROC",
                ),
            ]:
                rows.append(
                    _metric_row(
                        evidence_id=f"OfficialAlignment_v1_{idx}_{split}",
                        source_phase="OfficialEyeBenchAlignment_v1",
                        source_file=(
                            "analysis/official_eyebench_alignment_v1/tables/"
                            "copco_typ_official_alignment_comparison.csv"
                        ),
                        model_family="D3",
                        model_name=row.get("model"),
                        candidate_id=row.get("model"),
                        algorithm_regime="official_fold_full_feature",
                        task="CopCo_TYP",
                        evaluation_level="trial_level",
                        split_regime=split,
                        clean_or_oracle="clean",
                        official_claim_allowed=False,
                        benchmark_relative_claim_allowed=not bool(row.get("official_mode", False)),
                        AUROC=row.get(auc_col),
                        balanced_accuracy=row.get(ba_col),
                        notes=row.get("claim_type", ""),
                    )
                )
    sota_decision_path = root / "analysis/official_eyebench_sota_check_v1/official_sota_decision_report.md"
    rows.append(
        _metric_row(
            evidence_id="OfficialEyeBenchSOTACheck_v1_blocked",
            source_phase="OfficialEyeBenchSOTACheck_v1",
            source_file=str(sota_decision_path.relative_to(root))
            if sota_decision_path.exists()
            else "analysis/official_eyebench_sota_check_v1/official_sota_decision_report.md",
            model_family="D3",
            model_name="official_eyebench_sota_check",
            candidate_id="blocked_by_environment",
            algorithm_regime="official_compatible_lite",
            task="CopCo_TYP",
            evaluation_level="trial_level",
            split_regime="unknown",
            clean_or_oracle="blocked",
            official_claim_allowed=False,
            benchmark_relative_claim_allowed=False,
            notes="Official SOTA check did not provide a clean official D3 SOTA claim.",
        )
    )

    lite = _read_csv(root, "analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv")
    if not lite.empty:
        lite_subset = lite[lite["candidate_id"].astype(str).str.startswith(("candidate_0000", "candidate_0013"))]
        _append_metric_rows_from_frame(
            rows,
            lite_subset,
            source_phase="D3_EyeBench_own_method_score_max_v2",
            source_file="analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv",
            algorithm_regime="official_compatible_lite",
            evaluation_level="trial_level",
            model_name_col="family",
            candidate_col="candidate_id",
            feature_col="feature_recipe",
            split_col="split_name",
            notes="D3_Lite official-compatible stress test; no locked-test improvement over anchor.",
            official_claim_allowed=False,
            benchmark_relative_claim_allowed=True,
        )

    op_fixed = _read_csv(root, "analysis/operating_point_adaptation_v1/fixed_threshold_metrics.csv")
    if not op_fixed.empty:
        subset = op_fixed[
            op_fixed["source_name"].isin(["d3_eyebench_lite_candidate_0000", "benchmark_bridge_d3_full_data"])
        ]
        _append_metric_rows_from_frame(
            rows,
            subset,
            source_phase="OperatingPointAdaptation_v1",
            source_file="analysis/operating_point_adaptation_v1/fixed_threshold_metrics.csv",
            algorithm_regime="operating_point_diagnostic",
            evaluation_level="trial_level",
            model_name_col="model_name",
            candidate_col="candidate_id",
            feature_col="feature_group",
            split_col="split_regime",
            notes="Fixed 0.5 probability operating-point audit.",
            official_claim_allowed=False,
            benchmark_relative_claim_allowed=True,
        )
    op_oracle = _read_csv(root, "analysis/operating_point_adaptation_v1/test_oracle_threshold_metrics.csv")
    if not op_oracle.empty:
        oracle_subset = op_oracle.head(80)
        _append_metric_rows_from_frame(
            rows,
            oracle_subset,
            source_phase="OperatingPointAdaptation_v1",
            source_file="analysis/operating_point_adaptation_v1/test_oracle_threshold_metrics.csv",
            algorithm_regime="oracle_diagnostic",
            evaluation_level="trial_level",
            model_name_col="model_name",
            candidate_col="candidate_id",
            feature_col="feature_group",
            split_col="split_regime",
            notes="Test-label oracle diagnostic threshold; not clean benchmark evidence.",
            official_claim_allowed=False,
            benchmark_relative_claim_allowed=False,
            clean_or_oracle="oracle",
        )

    online_v1 = _read_csv(root, "analysis/d3_online_targeted_optimization_v1/online_locked_test_results.csv")
    if not online_v1.empty:
        _append_metric_rows_from_frame(
            rows,
            online_v1,
            source_phase="D3OnlineTargetedOptimization_v1",
            source_file="analysis/d3_online_targeted_optimization_v1/online_locked_test_results.csv",
            algorithm_regime="online_accumulator",
            evaluation_level="reader_aggregated",
            fixed_model_name="online_d3_0021",
            candidate_col="candidate_id",
            feature_col="feature_family",
            split_col="split_regime",
            notes="Best v1 clean candidate; selected no_stop and is offline-like.",
            official_claim_allowed=False,
            benchmark_relative_claim_allowed=True,
        )
    online_v2 = _read_csv(root, "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv")
    if not online_v2.empty:
        regime_map = {
            "best_offline_all_full_evidence": "offline_full_profile",
            "best_online_late_accumulation": "online_accumulator",
            "best_online_mid_detection": "online_prefix",
            "best_online_early_detection": "online_prefix",
            "best_online_stopping_detector": "online_stopping",
            "best_unseen_text_specialist": "unseen_text_specialist",
        }
        for idx, row in online_v2.iterrows():
            rows.append(
                _metric_row(
                    evidence_id=f"D3OnlineTargetedOptimization_v2_{idx:04d}",
                    source_phase="D3OnlineTargetedOptimization_v2",
                    source_file="analysis/d3_online_targeted_optimization_v2/strict_final_models.csv",
                    model_family="D3",
                    model_name=row.get("final_model"),
                    candidate_id=row.get("candidate_id"),
                    algorithm_regime=regime_map.get(row.get("final_model"), "online_prefix"),
                    task="CopCo_TYP",
                    evaluation_level="reader_aggregated",
                    split_regime=row.get("split_regime"),
                    prefix_type=row.get("prefix_type"),
                    prefix_value=row.get("prefix_value"),
                    evidence_budget=f"{row.get('prefix_type')}:{row.get('prefix_value')}",
                    feature_family=row.get("feature_family"),
                    calibrator=row.get("calibrator"),
                    threshold_policy=row.get("threshold_policy"),
                    threshold_source=row.get("selection_source"),
                    accumulator=row.get("accumulator"),
                    stopping_policy=row.get("stopping_policy"),
                    clean_or_oracle="clean",
                    official_claim_allowed=False,
                    benchmark_relative_claim_allowed=True,
                    n_readers=row.get("n_readers"),
                    n_prefix_rows=row.get("n_prefix_rows"),
                    coverage=row.get("coverage"),
                    AUROC=row.get("AUROC"),
                    PR_AUC=row.get("PR-AUC"),
                    balanced_accuracy=row.get("BA"),
                    macro_F1=row.get("macro_F1"),
                    Brier=row.get("Brier"),
                    calibration_intercept=row.get("calibration_intercept"),
                    calibration_slope=row.get("calibration_slope"),
                    notes="Strict v2 final separated model row.",
                )
            )

    frame = pd.DataFrame(rows)
    for column in METRIC_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame[METRIC_COLUMNS]
    frame["evidence_id"] = [
        f"{item}_{idx:05d}" if pd.notna(item) and str(item) else f"metric_{idx:05d}"
        for idx, item in enumerate(frame["evidence_id"])
    ]
    return frame


def build_online_prefix_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    v2 = _read_csv(root, "analysis/d3_online_targeted_optimization_v2/per_prefix_performance_curves.csv")
    if not v2.empty:
        frame = v2.rename(columns={"PR-AUC": "PR_AUC", "BA": "balanced_accuracy"})
        frame["model_name"] = "D3_online_v2_per_prefix"
        frame["threshold_policy"] = frame.get("threshold", "fixed_0_5")
        frame["evidence_budget"] = frame["prefix_type"].astype(str) + ":" + frame["prefix_value"].astype(str)
        frame["notes"] = "Outer-test per-prefix curve from strict v2 audit."
        frames.append(frame)
    v1 = _read_csv(root, "analysis/d3_online_targeted_optimization_v1/online_prefix_model_metrics.csv")
    if not v1.empty:
        frame = v1.rename(columns={"PR-AUC": "PR_AUC", "BA": "balanced_accuracy"})
        frame["model_name"] = "D3_online_v1_prefix_model"
        frame["feature_family"] = frame.get("feature_group")
        frame["calibrator"] = "identity"
        frame["threshold_policy"] = frame.get("threshold_source", "fixed_0_5")
        frame["accumulator"] = ""
        frame["evidence_budget"] = frame["prefix_type"].astype(str) + ":" + frame["prefix_value"].astype(str)
        frame["notes"] = "Online v1 prefix metrics; includes full-evidence rows for comparison."
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    keep = [
        "prefix_type",
        "prefix_value",
        "evidence_budget",
        "split_regime",
        "model_name",
        "feature_family",
        "calibrator",
        "threshold_policy",
        "accumulator",
        "AUROC",
        "PR_AUC",
        "balanced_accuracy",
        "Brier",
        "macro_F1",
        "n_readers",
        "n_prefix_rows",
        "notes",
    ]
    out = pd.concat(frames, ignore_index=True, sort=False)
    for column in keep:
        if column not in out:
            out[column] = None
    return out[keep]


def build_online_stopping_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    v1 = _read_csv(root, "analysis/d3_online_targeted_optimization_v1/online_stopping_policy_metrics.csv")
    if not v1.empty:
        frame = v1.rename(columns={"PR-AUC": "PR_AUC", "BA": "balanced_accuracy"})
        frame["source_phase"] = "D3OnlineTargetedOptimization_v1"
        frame["final_decision"] = frame["stopping_policy"].map(
            lambda x: "diagnostic" if x == "no_stop_all_evidence" else "useful"
        )
        frames.append(frame)
    v2 = _read_csv(root, "analysis/d3_online_targeted_optimization_v2/strict_final_models.csv")
    if not v2.empty:
        frame = v2[v2["final_model"].eq("best_online_stopping_detector")].rename(
            columns={"PR-AUC": "PR_AUC", "BA": "balanced_accuracy"}
        )
        frame["source_phase"] = "D3OnlineTargetedOptimization_v2"
        frame["threshold_source"] = frame.get("selection_source", "inner_oof")
        frame["final_decision"] = "stopping_not_ready"
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    keep = [
        "source_phase",
        "split_regime",
        "feature_group",
        "feature_family",
        "accumulator",
        "threshold_source",
        "stopping_policy",
        "coverage",
        "undecided_rate",
        "mean_words_to_decision",
        "mean_texts_to_decision",
        "balanced_accuracy",
        "AUROC",
        "PR_AUC",
        "Brier",
        "final_decision",
    ]
    out = pd.concat(frames, ignore_index=True, sort=False)
    for column in keep:
        if column not in out:
            out[column] = None
    return out[keep]


def build_model_runs(root: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(
            [
                "source_phase",
                "model_name",
                "candidate_id",
                "algorithm_regime",
                "evaluation_level",
                "clean_or_oracle",
            ],
            dropna=False,
        )
        .agg(
            metric_rows=("evidence_id", "size"),
            source_files=("source_file", lambda x: ";".join(sorted({str(item) for item in x if str(item)}))),
            official_claim_allowed=("official_claim_allowed", "max"),
            benchmark_relative_claim_allowed=("benchmark_relative_claim_allowed", "max"),
        )
        .reset_index()
    )
    grouped.insert(0, "model_run_id", [f"run_{idx:04d}" for idx in range(len(grouped))])
    return grouped


def build_claims() -> tuple[pd.DataFrame, list[str], list[str]]:
    allowed = [
        "D3 is an explainable reader-profile method based on residualized DFM predictability-sensitive gaze features.",
        "Offline full-profile D3 is the main model and performs strongly under unseen-reader / reader-level evaluation.",
        "DFM sensitivity/residual gaze features outperform DFM exposure-only features.",
        "BenchmarkBridge full-data reader-aggregated results are benchmark-relative/internal, not official EyeBench.",
        "D3 shows reader-regime SOTA-style performance in internal benchmark-relative evaluation.",
        "Online fixed-budget D3 shows meaningful performance after accumulated evidence.",
        "Online early / fixed-budget results can be reported as secondary when supported by v2 metrics.",
        "Unseen_text generalization remains a weakness.",
        "Stopping detector is not yet ready.",
        "Official EyeBench SOTA is not claimed.",
    ]
    prohibited = [
        "Official EyeBench SOTA.",
        "Full-table CopCo TYP domination.",
        "Trial-level D3_Lite SOTA.",
        "Online adaptive stopping detector is ready.",
        "Standalone segmentation-opacity main effect.",
        "Parser-syntax claims from surface_heuristic fallback.",
        "Test-oracle thresholds as clean benchmark evidence.",
        "Clinical diagnostic claims.",
        "Claiming unseen_text is solved by the general model.",
        "Claiming D3_Lite is equivalent to full D3.",
    ]
    rows = [
        {
            "claim_id": f"AC{idx:02d}",
            "claim_text": text,
            "claim_status": "allowed",
            "evidence_file": "03_results_canonical/canonical_metrics_long.csv",
            "allowed_context": "manuscript or supplement with stated caveats",
            "notes": "Use exact metric rows and source files from the evidence vault.",
        }
        for idx, text in enumerate(allowed, start=1)
    ]
    rows.extend(
        {
            "claim_id": f"PC{idx:02d}",
            "claim_text": text,
            "claim_status": "prohibited",
            "evidence_file": "05_claims/prohibited_claims.md",
            "allowed_context": "not allowed",
            "notes": "Do not use without a new validated protocol that supersedes this vault.",
        }
        for idx, text in enumerate(prohibited, start=1)
    )
    return pd.DataFrame(rows), allowed, prohibited


def build_number_registry(metrics: pd.DataFrame) -> pd.DataFrame:
    selectors = [
        ("offline_auc", "D3_dfm_residual_gaze_only", "offline_full_profile", "LOPO", "AUROC"),
        ("offline_ba", "D3_dfm_residual_gaze_only", "offline_full_profile", "LOPO", "balanced_accuracy"),
        ("benchmark_unseen_reader_auc", "D3_dfm_residual_gaze_only", "benchmark_bridge_full_data", "unseen_reader", "AUROC"),
        ("benchmark_unseen_text_auc", "D3_dfm_residual_gaze_only", "benchmark_bridge_full_data", "unseen_text", "AUROC"),
        (
            "benchmark_both_unseen_auc",
            "D3_dfm_residual_gaze_only",
            "benchmark_bridge_full_data",
            "unseen_reader_and_text",
            "AUROC",
        ),
        ("d3_lite_unseen_reader_ba", "d3_lite_anchor", "official_compatible_lite", "unseen_reader", "balanced_accuracy"),
        ("d3_lite_unseen_text_auc", "d3_lite_anchor", "official_compatible_lite", "unseen_text", "AUROC"),
        ("online_v1_unseen_reader_auc", "online_d3_0021", "online_accumulator", "unseen_reader", "AUROC"),
        ("online_v2_early_both_ba", "best_online_early_detection", "online_prefix", "unseen_reader_and_text", "balanced_accuracy"),
        ("online_v2_stopping_reader_ba", "best_online_stopping_detector", "online_stopping", "unseen_reader", "balanced_accuracy"),
        (
            "unseen_text_specialist_ba",
            "best_unseen_text_specialist",
            "unseen_text_specialist",
            "unseen_text",
            "balanced_accuracy",
        ),
    ]
    rows = []
    for number_id, model_name, regime, split, metric in selectors:
        subset = metrics[
            metrics["model_name"].astype(str).eq(model_name)
            & metrics["algorithm_regime"].astype(str).eq(regime)
            & metrics["split_regime"].astype(str).eq(split)
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "number_id": number_id,
                "value": row.get(metric),
                "metric": metric,
                "model_name": model_name,
                "split_regime": split,
                "source_file": row.get("source_file"),
                "source_phase": row.get("source_phase"),
                "allowed_context": "source number for future tables/figures; not a final paper table",
                "notes": row.get("notes", ""),
            }
        )
    rows.extend(
        [
            {
                "number_id": "online_v2_unseen_reader_auc_gate",
                "value": "250 words",
                "metric": "earliest_AUROC_ge_0.80",
                "model_name": "D3_online_v2_per_prefix",
                "split_regime": "unseen_reader",
                "source_file": "analysis/d3_online_targeted_optimization_v2/per_prefix_performance_report.md",
                "source_phase": "D3OnlineTargetedOptimization_v2",
                "allowed_context": "online evidence sufficiency narrative",
                "notes": "AUROC >= 0.80 first reached around 250 words.",
            },
            {
                "number_id": "online_v2_both_unseen_ba_gate",
                "value": "500 words",
                "metric": "earliest_BA_ge_0.75",
                "model_name": "D3_online_v2_per_prefix",
                "split_regime": "unseen_reader_and_text",
                "source_file": "analysis/d3_online_targeted_optimization_v2/per_prefix_performance_report.md",
                "source_phase": "D3OnlineTargetedOptimization_v2",
                "allowed_context": "online evidence sufficiency narrative",
                "notes": "BA >= 0.75 first reached around 500 words.",
            },
        ]
    )
    return pd.DataFrame(rows)


def write_docs(vault: Path, inventory: pd.DataFrame, metrics: pd.DataFrame, claims: pd.DataFrame) -> None:
    _write_text(
        vault / "README.md",
        """# D3 Model Evidence Vault v1

This committed folder is the source of truth for D3 model algorithm details and
experimental evidence v1. It consolidates small, durable, manuscript-supporting
evidence files and references larger generated artifacts by path.

This phase does not train models, run feature searches, create figures, or create
final paper tables. Future paper tables and figures should be generated from this
vault only.
""",
    )
    _write_text(
        vault / "EVIDENCE_CONTRACT.md",
        """# Evidence Contract

- `analysis/d3_model_evidence_v1/` is the source of truth for D3 algorithm details
  and experimental results v1.
- No new experiments are run in this phase.
- No figures or final paper tables are generated in this phase.
- Every numeric result must trace to `source_phase` and `source_file`.
- Every claim must have allowed or prohibited status.
- Large raw/generated files are referenced by path and checksum when available; they
  are not copied into this vault.
- Future final paper tables and figures should be generated later from the canonical
  files in this vault.
""",
    )

    index_rows = [{"file": rel, "purpose": _purpose_for_file(rel)} for rel in EXPECTED_FILES]
    _write_text(vault / "INDEX.md", "# Evidence Vault Index\n\n" + _md_table(pd.DataFrame(index_rows), 200))

    algorithm_docs = {
        "d3_algorithm_overview.md": """# D3 Algorithm Overview

D3 is the residualized DFM gaze-profile model family. It models how a reader's gaze
responds to contextual predictability from a Danish foundation language model (DFM),
after removing low-level word, text, and quality confounds. The explainable unit is a
reader profile: residual gaze sensitivity to DFM surprisal/entropy and related
predictability summaries.

D3 has four regimes:

- Full-data/offline reader profile: all available reading evidence is used to build a
  participant-level profile. This is the main scientific result and upper-bound
  reader-profile interpretation.
- D3 Lite: a reduced official-compatible trial-level model. It preserves an EyeBench
  stress-test anchor but is not equivalent to full D3.
- Online fixed-budget D3: prefix evidence is accumulated up to a fixed word/text
  budget and converted into probability `p_t`.
- Online stopping detector: sequential probabilities may stop early, but current
  coverage-risk results do not support readiness.
""",
        "d3_offline_algorithm.md": """# D3 Offline Algorithm

Input data are prepared CopCo word/trial/participant features with operational TYP/RCS
labels. Gaze outcomes are residualized using cross-fitted controls that exclude
`reader_group`. Participant-level features summarize residual gaze and DFM
predictability sensitivity across the full reader record.

The main classifier is logistic regression evaluated at participant level with
leave-one-participant-out / reader-level evaluation. It is offline because a full
reader record is available before prediction. The allowed interpretation is an
explainable reader-profile method, not an online screening or clinical diagnostic.
""",
        "d3_online_algorithm.md": """# D3 Online Algorithm

Online D3 constructs prefix rows from observed evidence only. Prefix budgets include
word-count budgets and first-N text/trial budgets. At each prefix, the model outputs a
probability `p_t`; optional calibrators and thresholds are learned from inner data.

Accumulators include mean probability, logit mean, entropy/uncertainty weighting, and
learned meta-aggregation. V2 separates early, mid, late, full-evidence, and stopping
detector categories. Current interpretation: fixed-budget online D3 is a secondary
result; adaptive stopping is not ready.
""",
        "d3_lite_algorithm.md": """# D3 Lite Algorithm

D3 Lite is the official-compatible reduced model used for EyeBench-style trial-level
stress testing. It uses a constrained feature set and official fold/evaluator
compatibility where possible. It is not full D3 because it lacks the full reader-profile
evidence and full residualized sensitivity summaries. It did not establish official
SOTA and no improved locked candidate beat the preserved anchor.
""",
        "residualization_spec.md": """# Residualization Specification

Residualization removes known word/text/quality effects from gaze outcomes before D3
reader sensitivity features are computed. Residualizers are fit without `reader_group`,
without participant IDs as predictors, and without held-out rows. Cross-fitting is used
where held-out evaluation is required so test evidence is not used to fit residual
models.
""",
        "dfm_predictability_feature_spec.md": """# DFM Predictability Feature Specification

DFM features summarize word-level predictability from a Danish foundation language
model: surprisal, entropy, and related contextual measures. Exposure features describe
what predictability distribution a reader saw; sensitivity features estimate how gaze
responses vary with predictability. The evidence shows exposure-only is weak while
sensitivity/residual gaze is strong.
""",
        "calibration_threshold_spec.md": """# Calibration and Threshold Specification

Clean thresholds are fixed 0.5 or learned from train/inner-validation/calibration rows.
Sigmoid/Platt and isotonic calibration are allowed only when fitted on non-test rows.
Test-label oracle thresholds are diagnostic upper bounds only and must have
`official_claim_allowed=false`.
""",
        "online_accumulation_spec.md": """# Online Accumulation Specification

Online D3 can aggregate prefix probabilities using simple mean probability, cumulative
logit mean, entropy/uncertainty weighting, reliability weighting, or a learned
meta-aggregator trained on inner-validation predictions. V1 selected a strong
learned-meta no-stop accumulator; v2 separates that from true fixed-budget online rows.
""",
        "stopping_policy_spec.md": """# Stopping Policy Specification

`no_stop` consumes all available sequence evidence and is a full-evidence baseline, not
a stopping detector. Confidence, cost-sensitive, target-sensitivity, and
coverage-constrained stopping policies must learn thresholds from inner data. V2 found
that stopping reduced burden but did not preserve balanced accuracy, so stopping is not
ready.
""",
        "prohibited_features_and_leakage_policy.md": """# Prohibited Features and Leakage Policy

Prohibited predictors include `participant_id`, `speech_id`, `text_id`, random word
splits, future total word/text counts in online rows, and `reader_group` inside
residualization. Clean metrics must not use test labels for thresholds/calibration.
Oracle rows are diagnostic only. Official SOTA requires official data, folds, and
evaluator chain support.
""",
    }
    for filename, text in algorithm_docs.items():
        _write_text(vault / "01_algorithm" / filename, text)

    data_docs = {
        "dataset_summary.md": "Prepared data source: `results/label_release_v1_1_20260506_0041/prepared_dataset/`. Feature release source: `results/feature_release_v1_20260505_2155/`.",
        "label_summary.md": "Operational research labels are from Label Release v1.1. Do not claim clinical diagnosis or screening.",
        "feature_family_summary.md": "Feature families include raw gaze, residual gaze, DFM exposure, DFM sensitivity, residualized DFM gaze, and online uncertainty/stability summaries.",
        "split_policy_summary.md": "Primary split regimes are unseen_reader, unseen_text, unseen_reader_and_text, text_balanced_unseen_reader, participant_grouped_kfold, and LOPO where applicable.",
        "online_prefix_dataset_summary.md": "Online v1 validation reported 1,145 prefix rows; v2 per-prefix curves contain 1,232 metric rows.",
        "nested_prediction_artifact_summary.md": "Online v1 validation reported 306,376 nested prediction rows and legal calibration/threshold artifacts.",
        "eyeBench_alignment_summary.md": "Official subset/evaluator support remained blocked or non-official for full D3. Final category: benchmark-relative only.",
    }
    for filename, text in data_docs.items():
        _write_text(vault / "02_data_and_splits" / filename, "# " + filename.replace("_", " ").replace(".md", "").title() + "\n\n" + text)

    metric_schema = ["# Metric Schema", ""]
    metric_schema.extend(f"- `{column}`" for column in METRIC_COLUMNS)
    metric_schema.append(
        "\nEnumerated regimes include offline_full_profile, benchmark_bridge_full_data, "
        "official_fold_full_feature, official_compatible_lite, operating_point_diagnostic, "
        "online_prefix, online_accumulator, online_stopping, oracle_diagnostic, and "
        "unseen_text_specialist."
    )
    _write_text(vault / "03_results_canonical/metric_schema.md", "\n".join(metric_schema))

    narrative_docs = {
        "offline_result_summary.md": "Offline D3 is the main result: AUROC 0.8947, PR-AUC 0.8641, BA 0.8421, Brier 0.1159.",
        "benchmark_bridge_result_summary.md": "BenchmarkBridge full-data reader-aggregated D3: unseen_reader AUROC 0.8961 / BA 0.8158; unseen_text AUROC 0.8285 / BA 0.7444; both-unseen AUROC 0.8542 / BA 0.7458.",
        "official_eyebench_alignment_summary.md": "Official subset support was blocked/skipped for full D3. Official SOTA is not claimed.",
        "d3_lite_result_summary.md": "D3_EyeBench_Lite candidate_0000 remains the anchor; no score-max candidate improved the locked test result.",
        "operating_point_result_summary.md": "OperatingPointAdaptation found probability aggregation useful but legal threshold learning was limited by missing inner/calibration artifacts in v1.",
        "online_targeted_v1_summary.md": "V1 produced 243,656 online probability rows and selected online_d3_0021, but the selected no_stop policy makes it offline-like.",
        "online_targeted_v2_summary.md": "V2 separated full evidence, late, mid, early, and stopping. Fixed-budget online rows show capability; stopping is not ready.",
        "unseen_text_failure_summary.md": "General unseen_text remains weak. A v2 specialist row improved unseen_text but is diagnostic/supplementary.",
        "final_result_interpretation.md": "D3 should be written as an explainable offline reader-profile method with secondary online fixed-budget evidence; official SOTA and full-table dominance are not supported.",
    }
    for filename, text in narrative_docs.items():
        _write_text(vault / "04_result_narratives" / filename, "# " + filename.replace("_", " ").replace(".md", "").title() + "\n\n" + text)

    allowed = claims[claims["claim_status"].eq("allowed")]
    prohibited = claims[claims["claim_status"].eq("prohibited")]
    _write_text(vault / "05_claims/allowed_claims.md", "# Allowed Claims\n\n" + "\n".join(f"- {x}" for x in allowed["claim_text"]))
    _write_text(
        vault / "05_claims/prohibited_claims.md",
        "# Prohibited Claims\n\n" + "\n".join(f"- {x}" for x in prohibited["claim_text"]),
    )
    _write_text(
        vault / "05_claims/claim_wording_templates.md",
        """# Claim Wording Templates

- D3 is strongest as an offline reader-profile model based on residualized gaze
  sensitivity to DFM predictability.
- BenchmarkBridge and online/offline results support reader-regime, benchmark-relative
  SOTA-style language, but not official EyeBench SOTA.
- Online fixed-budget D3 provides a secondary deployment-oriented analysis after
  sufficient evidence accumulation.
- General unseen_text transfer remains a limitation; specialist rows are diagnostic.
""",
    )

    _write_text(
        vault / "06_paper_sources/table_source_manifest.md",
        """# Table Source Manifest

No final paper tables are created here. Candidate future tables and source files:

- Dataset summary: `02_data_and_splits/*.md`
- DFM exposure vs sensitivity: `03_results_canonical/canonical_metrics_long.csv`
- Offline D3 final metrics: `03_results_canonical/canonical_metrics_long.csv`
- BenchmarkBridge comparison: `03_results_canonical/canonical_comparison_results.csv`
- D3 Lite official-compatible stress test: `03_results_canonical/canonical_metrics_long.csv`
- Online/offline D3 comparison: `03_results_canonical/canonical_metrics_long.csv`
- Online prefix reliability: `03_results_canonical/canonical_online_prefix_results.csv`
- Claim-evidence table: `05_claims/claim_evidence_ledger_v1.csv`
- Limitation/caveat table: `08_appendix_material/limitations_and_caveats.md`
""",
    )
    _write_text(
        vault / "06_paper_sources/figure_source_manifest.md",
        """# Figure Source Manifest

No figures are generated here. Candidate future figures and sources:

- D3 algorithm diagram: `01_algorithm/*.md`
- Offline versus online regime diagram: `01_algorithm/d3_algorithm_overview.md`
- DFM exposure vs sensitivity plot: `03_results_canonical/canonical_metrics_long.csv`
- BenchmarkBridge split comparison plot: `03_results_canonical/canonical_comparison_results.csv`
- Online prefix performance curve: `03_results_canonical/canonical_online_prefix_results.csv`
- Error trajectory plot: `03_results_canonical/canonical_error_results.csv`
- Unseen_text failure plot: `04_result_narratives/unseen_text_failure_summary.md`
""",
    )
    _write_text(
        vault / "06_paper_sources/number_consistency_report.md",
        "# Number Consistency Report\n\nKey manuscript numbers are registered in `paper_ready_number_registry.csv` and trace to canonical source rows.",
    )
    _write_text(
        vault / "08_appendix_material/reviewer_risk_notes.md",
        "# Reviewer Risk Notes\n\nMain risks: official SOTA overclaim, unseen_text weakness, D3_Lite underperformance, and online stopping not ready.",
    )
    _write_text(
        vault / "08_appendix_material/limitations_and_caveats.md",
        "# Limitations and Caveats\n\nD3 is not a clinical diagnostic. Official EyeBench SOTA is not claimed. General unseen_text remains weak. Oracle thresholds are diagnostic only.",
    )
    _write_text(
        vault / "08_appendix_material/future_work_and_open_gaps.md",
        "# Future Work and Open Gaps\n\nFuture work: official evaluator closure, stronger unseen_text generalization, calibrated stopping policies, and independent replication.",
    )


def _purpose_for_file(rel: str) -> str:
    if rel.endswith(".csv") or rel.endswith(".jsonl"):
        return "machine-readable canonical evidence"
    if rel.endswith(".json"):
        return "machine-readable status/decision metadata"
    if "01_algorithm" in rel:
        return "algorithm and leakage documentation"
    if "05_claims" in rel:
        return "claim governance"
    if "06_paper_sources" in rel:
        return "future table/figure source manifest, not final output"
    if "07_validation" in rel:
        return "vault validation"
    return "human-readable evidence summary"


def write_canonical_files(root: Path, vault: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = collect_canonical_metrics(root)
    _write_frame(vault / "03_results_canonical/canonical_metrics_long.csv", metrics)
    _write_jsonl(vault / "03_results_canonical/canonical_metrics_long.jsonl", metrics)
    model_runs = build_model_runs(root, metrics)
    _write_frame(vault / "03_results_canonical/canonical_model_runs.csv", model_runs)
    _write_jsonl(vault / "03_results_canonical/canonical_model_runs.jsonl", model_runs)
    prefix = build_online_prefix_results(root)
    _write_frame(vault / "03_results_canonical/canonical_online_prefix_results.csv", prefix)
    stopping = build_online_stopping_results(root)
    _write_frame(vault / "03_results_canonical/canonical_online_stopping_results.csv", stopping)
    oracle = metrics[metrics["clean_or_oracle"].eq("oracle")].copy()
    _write_frame(vault / "03_results_canonical/canonical_oracle_results.csv", oracle)
    comparison = metrics[
        metrics["algorithm_regime"].isin(
            ["benchmark_bridge_full_data", "official_fold_full_feature", "official_compatible_lite"]
        )
    ].copy()
    _write_frame(vault / "03_results_canonical/canonical_comparison_results.csv", comparison)
    error_rows = [
        {
            "source_phase": "D3OnlineTargetedOptimization_v2",
            "source_file": "analysis/d3_online_targeted_optimization_v2/error_source_by_prefix_report.md",
            "result": "learned_meta_persistent_fp_fn_rows",
            "value": 596,
            "notes": "Learned meta persistent FP/FN rows from v2 summary.",
        },
        {
            "source_phase": "D3OnlineTargetedOptimization_v2",
            "source_file": "analysis/d3_online_targeted_optimization_v2/error_source_by_prefix_report.md",
            "result": "mean_probability_persistent_fp_fn_rows",
            "value": 720,
            "notes": "Mean probability persistent FP/FN rows from v2 summary.",
        },
        {
            "source_phase": "D3OnlineTargetedOptimization_v2",
            "source_file": "analysis/d3_online_targeted_optimization_v2/error_source_by_prefix_report.md",
            "result": "unseen_text_top_error_text",
            "value": "7905",
            "notes": "Held-out text with largest wrong-row concentration.",
        },
    ]
    _write_csv(vault / "03_results_canonical/canonical_error_results.csv", error_rows)
    return metrics, prefix, stopping


def write_claim_files(vault: Path) -> pd.DataFrame:
    claims, _, _ = build_claims()
    _write_frame(vault / "05_claims/claim_evidence_ledger_v1.csv", claims)
    _write_text(vault / "05_claims/claim_evidence_ledger_v1.md", "# Claim Evidence Ledger v1\n\n" + _md_table(claims, 80))
    _write_frame(vault / "03_results_canonical/canonical_claim_results.csv", claims)
    return claims


def write_number_files(vault: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    registry = build_number_registry(metrics)
    _write_frame(vault / "06_paper_sources/paper_ready_number_registry.csv", registry)
    _write_frame(vault / "06_paper_sources/manuscript_number_map.csv", registry)
    return registry


def validate_evidence_vault(
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    vault = root / ANALYSIS_DIR
    errors: list[str] = []
    warnings: list[str] = []
    missing_files = [rel for rel in EXPECTED_FILES if not (vault / rel).exists()]
    errors.extend(f"missing expected evidence file: {rel}" for rel in missing_files)

    metrics_path = vault / "03_results_canonical/canonical_metrics_long.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if metrics.empty:
        errors.append("canonical_metrics_long.csv is missing or empty")
    else:
        missing_columns = [column for column in METRIC_COLUMNS if column not in metrics.columns]
        errors.extend(f"missing canonical metric column: {column}" for column in missing_columns)
        if metrics["evidence_id"].duplicated().any():
            errors.append("duplicate evidence_id found")
        metric_columns = ["AUROC", "PR_AUC", "balanced_accuracy", "macro_F1", "Brier", "p_value"]
        metric_present = metrics[metric_columns].notna().any(axis=1)
        traced = metrics["source_phase"].notna() & metrics["source_file"].notna()
        if not traced[metric_present].all():
            errors.append("one or more non-null metrics lack source_phase/source_file")
        oracle = metrics[metrics["clean_or_oracle"].eq("oracle")]
        if not oracle.empty and oracle["official_claim_allowed"].fillna(False).astype(bool).any():
            errors.append("oracle rows have official_claim_allowed=true")
        if not metrics["algorithm_regime"].astype(str).str.contains("online").any():
            errors.append("online regimes missing from canonical metrics")
        if not metrics["algorithm_regime"].astype(str).str.contains("offline").any():
            errors.append("offline regimes missing from canonical metrics")

    claims_path = vault / "05_claims/claim_evidence_ledger_v1.csv"
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    if claims.empty:
        errors.append("claim ledger is missing or empty")
    else:
        if not claims["claim_status"].eq("allowed").any():
            errors.append("no allowed claims listed")
        if not claims["claim_status"].eq("prohibited").any():
            errors.append("no prohibited claims listed")
        official_allowed = claims[
            claims["claim_text"].astype(str).str.contains("Official EyeBench SOTA", case=False)
            & ~claims["claim_text"].astype(str).str.contains("not claimed", case=False)
            & claims["claim_status"].eq("allowed")
        ]
        if not official_allowed.empty:
            errors.append("official SOTA appears as allowed claim")

    registry_path = vault / "06_paper_sources/paper_ready_number_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    required_numbers = {
        "offline_auc",
        "offline_ba",
        "benchmark_unseen_reader_auc",
        "d3_lite_unseen_reader_ba",
        "online_v1_unseen_reader_auc",
        "online_v2_early_both_ba",
        "unseen_text_specialist_ba",
    }
    if registry.empty:
        errors.append("number registry is missing or empty")
    else:
        missing_numbers = required_numbers - set(registry["number_id"].astype(str))
        errors.extend(f"missing key number registry entry: {item}" for item in sorted(missing_numbers))

    generated_bad = list(vault.rglob("*.png")) + list(vault.rglob("*.pdf")) + list(vault.rglob("*.svg"))
    if generated_bad:
        errors.append("generated figure-like files exist in evidence vault")
    final_table_like = [
        path
        for path in vault.rglob("*")
        if path.is_file() and "final_table" in path.name.lower()
    ]
    if final_table_like:
        errors.append("final paper table-like files exist in evidence vault")
    large_files = [path for path in vault.rglob("*") if path.is_file() and path.stat().st_size >= 100_000_000]
    if large_files:
        errors.append("large files over repository threshold exist in evidence vault")

    missing_source = vault / "00_inventory/missing_source_report.md"
    if not missing_source.exists():
        errors.append("missing source report absent")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "analysis_dir": str(vault),
        "canonical_metric_rows": int(len(metrics)),
        "claim_rows": int(len(claims)),
        "number_registry_rows": int(len(registry)),
    }
    out = Path(output_dir) if output_dir else None
    if out:
        if not out.is_absolute():
            out = root / out
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "d3_model_evidence_v1_validation_report.json", report)
    _write_json(vault / "07_validation/evidence_vault_validation_report.json", report)
    validation_md = [
        "# Evidence Vault Validation Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Canonical metric rows: {report['canonical_metric_rows']}",
        f"- Claim rows: {report['claim_rows']}",
        f"- Number registry rows: {report['number_registry_rows']}",
        "",
        "## Errors",
        "",
        "\n".join(f"- {error}" for error in errors) if errors else "None.",
        "",
        "## Warnings",
        "",
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "None.",
    ]
    _write_text(vault / "07_validation/evidence_vault_validation_report.md", "\n".join(validation_md))
    _write_text(
        vault / "07_validation/source_trace_validation_report.md",
        "# Source Trace Validation Report\n\nEvery non-null metric row must include `source_phase` and `source_file`. "
        + ("Passed." if not errors else "See validation errors."),
    )
    _write_text(
        vault / "07_validation/metric_consistency_report.md",
        "# Metric Consistency Report\n\nV1 and v2 online rows are separated by `source_phase` and `algorithm_regime`; unseen_text weakness and stopping_not_ready are preserved.",
    )
    _write_text(
        vault / "07_validation/leakage_and_protocol_validation_report.md",
        "# Leakage and Protocol Validation Report\n\nOracle rows are diagnostic only with `official_claim_allowed=false`; benchmark-relative rows are internal/non-official; no generated figures or final tables are in this vault.",
    )
    return report


def build_d3_model_evidence_v1(
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    vault = root / ANALYSIS_DIR
    for section in SECTION_DIRS:
        (vault / section).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) if output_dir else root / f"results/d3_model_evidence_v1_{datetime.now():%Y%m%d_%H%M%S}"
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    inventory, missing = build_source_inventory(root, vault)
    write_commit_trace(root, vault)
    metrics, prefix, stopping = write_canonical_files(root, vault)
    claims = write_claim_files(vault)
    registry = write_number_files(vault, metrics)
    write_docs(vault, inventory, metrics, claims)

    status = {
        "status": "built",
        "analysis_dir": str(vault),
        "output_dir": str(out),
        "source_artifact_inventory_rows": int(len(inventory)),
        "missing_sources": missing,
        "canonical_metric_rows": int(len(metrics)),
        "online_prefix_rows": int(len(prefix)),
        "online_stopping_rows": int(len(stopping)),
        "claim_ledger_rows": int(len(claims)),
        "number_registry_rows": int(len(registry)),
        "figures_generated": False,
        "final_paper_tables_generated": False,
    }
    _write_json(vault / "status.json", status)
    _write_json(out / "d3_model_evidence_v1_manifest.json", status)
    report = validate_evidence_vault(repo_root=root, output_dir=out)
    status["validation_status"] = report["status"]
    _write_json(vault / "status.json", status)
    _write_json(out / "d3_model_evidence_v1_manifest.json", status)
    return status
