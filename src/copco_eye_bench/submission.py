"""SubmissionSprint v1 packaging for the frozen AutoResearch result."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir
from .research_exploration import _format_value, _markdown_table, _pd


FINAL_TITLE = (
    "Predictability-Sensitive Gaze Profiles for Dyslexia-Labeled Reader Prediction "
    "in Danish Natural Reading"
)
FINAL_MAIN_CLAIM = (
    "Participant-level DFM predictability sensitivity and cross-fitted residualized "
    "gaze-cost profiles distinguish dyslexia-labeled and typical/control readers in "
    "Danish natural reading."
)
FINAL_MODEL_GROUP = "D3_dfm_residual_gaze_only"
FINAL_MODEL = "logistic_regression"
FINAL_SPLIT = "leave_one_participant_out"

SUBMISSION_TABLES = {
    "dataset_summary": {
        "source": "tables/dataset_summary_table.csv",
        "caption": "Dataset summary for the frozen Label Release v1.1 prepared tables.",
        "label": "tab:dataset-summary",
    },
    "feature_label_release_summary": {
        "source": "tables/feature_release_summary_table.csv",
        "caption": "Feature and label release provenance summary.",
        "label": "tab:feature-label-release",
    },
    "final_model_metrics": {
        "source": "final_model/final_model_metrics.csv",
        "caption": "Locked final participant-level model metrics.",
        "label": "tab:final-model-metrics",
    },
    "dfm_exposure_vs_sensitivity": {
        "source": "stress_tests/dfm_exposure_vs_sensitivity.csv",
        "caption": "DFM exposure-only and sensitivity/residual gaze comparisons.",
        "label": "tab:dfm-exposure-sensitivity",
    },
    "robustness_tests": {
        "source": "tables/robustness_summary_table.csv",
        "caption": "Permutation, bootstrap, and influence robustness summary.",
        "label": "tab:robustness",
    },
    "calibration_influence_summary": {
        "source": None,
        "caption": "Calibration and participant influence summary.",
        "label": "tab:calibration-influence",
    },
    "feature_stability": {
        "source": "stress_tests/feature_stability.csv",
        "caption": "Stable standardized coefficients for the locked final model.",
        "label": "tab:feature-stability",
    },
    "interaction_synthesis": {
        "source": "tables/interaction_synthesis_table.csv",
        "caption": "Focused reader-group interaction synthesis.",
        "label": "tab:interaction-synthesis",
    },
    "reviewer_risk_summary": {
        "source": "tables/reviewer_risk_table.csv",
        "caption": "Reviewer-risk summary and submission blocking status.",
        "label": "tab:reviewer-risk",
    },
    "main_claims_evidence": {
        "source": None,
        "caption": "Main claims and evidence ledger summary.",
        "label": "tab:claims-evidence",
    },
}

SUBMISSION_FIGURES = {
    "pipeline_overview": {
        "source": "figures/pipeline_overview.png",
        "caption": "End-to-end frozen analysis and submission package flow.",
        "label": "fig:pipeline-overview",
    },
    "cross_fitted_residualization_schematic": {
        "source": None,
        "caption": "Cross-fitted residualization inside the participant-held-out split.",
        "label": "fig:crossfit-schematic",
    },
    "dfm_exposure_vs_sensitivity": {
        "source": "figures/dfm_exposure_vs_sensitivity_auc.png",
        "caption": "DFM exposure-only versus DFM sensitivity and residual gaze models.",
        "label": "fig:dfm-exposure-sensitivity",
    },
    "final_roc": {
        "source": "figures/final_model_roc_curve.png",
        "caption": "Final participant-level ROC curve.",
        "label": "fig:final-roc",
    },
    "final_pr": {
        "source": "figures/final_model_pr_curve.png",
        "caption": "Final participant-level precision-recall curve.",
        "label": "fig:final-pr",
    },
    "permutation_null": {
        "source": "figures/permutation_null_distribution.png",
        "caption": "Permutation null distribution for the locked final model.",
        "label": "fig:permutation-null",
    },
    "bootstrap_auc": {
        "source": "figures/bootstrap_auc_distribution.png",
        "caption": "Bootstrap uncertainty for the final model AUC estimates.",
        "label": "fig:bootstrap-auc",
    },
    "feature_stability": {
        "source": "figures/feature_stability_coefficients.png",
        "caption": "Coefficient stability for DFM residual gaze features.",
        "label": "fig:feature-stability",
    },
    "calibration": {
        "source": "figures/calibration_plot.png",
        "caption": "Calibration summary for the final participant-level predictions.",
        "label": "fig:calibration",
    },
    "interaction_summary": {
        "source": "figures/interaction_effects_summary.png",
        "caption": "Focused controlled interaction estimates.",
        "label": "fig:interaction-summary",
    },
    "participant_error": {
        "source": "figures/participant_error_analysis.png",
        "caption": "Participant-level prediction error and influence audit.",
        "label": "fig:participant-error",
    },
    "text_exposure_audit": {
        "source": "figures/text_exposure_vs_prediction_audit.png",
        "caption": "Text exposure and prediction audit.",
        "label": "fig:text-exposure-audit",
    },
}

MANUSCRIPT_SECTIONS = [
    "00_abstract",
    "01_introduction",
    "02_related_work",
    "03_data",
    "04_features_and_labels",
    "05_methods",
    "06_results",
    "07_analysis_and_interpretation",
    "08_limitations",
    "09_conclusion",
]

SUPPLEMENT_SECTIONS = [
    "01_feature_dictionary",
    "02_final_model_feature_list",
    "03_dfm_exposure_vs_sensitivity",
    "04_robustness_results",
    "05_permutation_details",
    "06_bootstrap_details",
    "07_participant_influence_error",
    "08_calibration_details",
    "09_text_exposure_audit",
    "10_lm_warning_audit",
    "11_boundary_opacity_labels",
    "12_segmentation_null_result",
    "13_word_level_secondary_ladder",
    "14_split_policy",
    "15_reproducibility_commands",
    "16_dataset_caveats",
    "17_reviewer_risk_notes",
]

REPRODUCIBILITY_FILES = [
    "README_REPRODUCE.md",
    "reproduce_submission_package.sh",
    "reproduce_autoresearch_v1.sh",
    "slurm_submission_package.sh",
    "environment_summary.md",
    "command_manifest.md",
    "input_output_manifest.md",
    "commit_trace.md",
    "data_not_committed_notice.md",
    "artifact_manifest.json",
    "checksums.json",
]


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=str))


def _write_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _git_sha(repo_root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _configured_path(config: dict[str, Any], dotted: str, repo_root: str | Path) -> Path:
    value = get_nested(config, dotted)
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(repo_root).resolve() / path
    return path.resolve()


def _dirs(config: dict[str, Any], output_dir: str | Path | None, repo_root: str | Path) -> dict[str, Path]:
    out = (
        Path(output_dir).resolve()
        if output_dir
        else timestamped_output_dir(config, repo_root=repo_root).resolve()
    )
    root = Path(repo_root).resolve()
    return {
        "result_root": out,
        "paper": _configured_path(config, "submission.output_layout.paper_dir", repo_root),
        "analysis": _configured_path(config, "submission.output_layout.analysis_dir", repo_root),
        "autoresearch": _configured_path(config, "submission.frozen_inputs.autoresearch_dir", repo_root),
        "autoresearch_analysis": _configured_path(
            config, "submission.frozen_inputs.autoresearch_analysis_dir", repo_root
        ),
        "repo_root": root,
    }


def _ensure_layout(dirs: dict[str, Path]) -> None:
    for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
        base.mkdir(parents=True, exist_ok=True)
    for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
        for sub in [
            "tables",
            "figures",
            "manuscript",
            "supplement",
            "reproducibility",
            "reviewer_risk",
            "decision",
        ]:
            (base / sub).mkdir(parents=True, exist_ok=True)
    (dirs["paper"] / "sections").mkdir(parents=True, exist_ok=True)
    (dirs["paper"] / "supplement_sections").mkdir(parents=True, exist_ok=True)


def _copy_to_all(src: Path, rel: str, dirs: dict[str, Path]) -> None:
    for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _table_tex(name: str, frame: Any, caption: str, label: str) -> str:
    columns = list(frame.columns[: min(len(frame.columns), 6)])
    visible = frame[columns].head(12).copy()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{_tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\toprule",
        " & ".join(_tex_escape(col) for col in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in visible.iterrows():
        lines.append(" & ".join(_tex_escape(_format_value(row[col])) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _save_table_bundle(name: str, frame: Any, caption: str, label: str, dirs: dict[str, Path]) -> None:
    for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
        _write_csv(base / "tables" / f"{name}.csv", frame)
        _write_text(
            base / "tables" / f"{name}.md",
            "# " + caption + "\n\n" + _markdown_table(frame.to_dict("records"), list(frame.columns), max_rows=40),
        )
        _write_text(base / "tables" / f"{name}.tex", _table_tex(name, frame, caption, label))


def _load_package_data(config: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    pd = _pd()
    root = dirs["autoresearch"]
    return {
        "manifest": _read_json(root / "manifest.json"),
        "run_summary": _read_json(root / "run_summary.json"),
        "final_decision": _read_json(root / "decision" / "final_decision.json"),
        "final_model_manifest": _read_json(root / "final_model" / "final_model_manifest.json"),
        "metrics": pd.read_csv(root / "final_model" / "final_model_metrics.csv"),
        "feature_dictionary_text": (root / "final_model" / "final_model_feature_dictionary.md").read_text(
            encoding="utf-8"
        ),
        "dfm": pd.read_csv(root / "stress_tests" / "dfm_exposure_vs_sensitivity.csv"),
        "bootstrap": pd.read_csv(root / "stress_tests" / "bootstrap_results.csv"),
        "permutation": pd.read_csv(root / "stress_tests" / "permutation_results.csv"),
        "calibration": pd.read_csv(root / "stress_tests" / "calibration_summary.csv"),
        "influence": pd.read_csv(root / "stress_tests" / "influence_analysis.csv"),
        "text_exposure": pd.read_csv(root / "stress_tests" / "text_exposure_sensitivity.csv"),
        "feature_stability": pd.read_csv(root / "stress_tests" / "feature_stability.csv"),
        "interactions": pd.read_csv(root / "tables" / "interaction_synthesis_table.csv"),
        "reviewer_risk": pd.read_csv(root / "tables" / "reviewer_risk_table.csv"),
    }


def _final_metric(data: dict[str, Any]) -> dict[str, Any]:
    return data["metrics"].iloc[0].to_dict()


def _validate_locked_inputs(config: dict[str, Any], dirs: dict[str, Path], data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = get_nested(config, "submission.expected_metrics", {})
    metric = _final_metric(data)
    selected = data["final_model_manifest"]
    if selected.get("selected_feature_group") != FINAL_MODEL_GROUP:
        errors.append("selected feature group changed from frozen D3")
    if selected.get("selected_model") != FINAL_MODEL:
        errors.append("selected model changed from frozen logistic regression")
    if selected.get("split_name") != FINAL_SPLIT:
        errors.append("selected split changed from frozen LOPO")
    tolerance = float(expected.get("tolerance", 0.0005))
    for key in [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "macro_f1",
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
    ]:
        if key in expected and abs(float(metric[key]) - float(expected[key])) > tolerance:
            errors.append(f"metric mismatch for {key}: {metric[key]} != {expected[key]}")
    for key in ["n_predictions", "skipped_folds"]:
        if key in expected and int(metric[key]) != int(expected[key]):
            errors.append(f"metric count mismatch for {key}: {metric[key]} != {expected[key]}")
    prohibited = set(get_nested(config, "submission.prohibited_variables", []))
    features = set(selected.get("features", []))
    if prohibited.intersection(features):
        errors.append("primary model contains prohibited variables")
    if not dirs["autoresearch"].exists():
        errors.append(f"missing frozen AutoResearch input: {dirs['autoresearch']}")
    return errors


def _build_claim_ledger(dirs: dict[str, Path], data: dict[str, Any]) -> Any:
    pd = _pd()
    metric = _final_metric(data)
    perm_p = (int((data["permutation"]["roc_auc"] >= metric["roc_auc"]).sum()) + 1) / (
        int(data["permutation"]["roc_auc"].notna().sum()) + 1
    )
    rows = [
        {
            "claim_id": "C01",
            "claim_text": "DFM residual gaze profiles predict participant group.",
            "claim_category": "main",
            "evidence_file": "tables/final_model_metrics.csv",
            "evidence_table_figure": "Table 3; Figures 4-5",
            "metric_statistic": f"ROC-AUC {metric['roc_auc']:.4f}; PR-AUC {metric['pr_auc']:.4f}",
            "sample_size": "57 participants",
            "caveat": "Operational labels; no external dataset.",
            "manuscript_section": "Results",
            "status": "supported",
        },
        {
            "claim_id": "C02",
            "claim_text": "DFM sensitivity dominates DFM exposure.",
            "claim_category": "supporting",
            "evidence_file": "tables/dfm_exposure_vs_sensitivity.csv",
            "evidence_table_figure": "Table 4; Figure 3",
            "metric_statistic": "D1 ROC-AUC 0.4238; D2 0.8892; D3 0.8947",
            "sample_size": "57 participants",
            "caveat": "Same frozen dataset; not an external replication.",
            "manuscript_section": "Results",
            "status": "supported",
        },
        {
            "claim_id": "C03",
            "claim_text": "Prediction survives permutation and bootstrap robustness.",
            "claim_category": "supporting",
            "evidence_file": "tables/robustness_tests.csv",
            "evidence_table_figure": "Table 5; Figures 6-7",
            "metric_statistic": f"Permutation p={perm_p:.6f}; ROC-AUC CI [0.7765, 0.9841]",
            "sample_size": "57 participants; 1000 permutations; 2000 bootstraps",
            "caveat": "Bootstrap resamples participants from the same dataset.",
            "manuscript_section": "Results",
            "status": "supported",
        },
        {
            "claim_id": "C04",
            "claim_text": "Cross-fitted residualization avoids using held-out participants in residual fitting.",
            "claim_category": "supporting",
            "evidence_file": "decision/final_decision.json",
            "evidence_table_figure": "Figure 2",
            "metric_statistic": "LOPO residualization described and validated.",
            "sample_size": "57 LOPO folds",
            "caveat": "Implementation is documented; external audit remains useful.",
            "manuscript_section": "Methods",
            "status": "supported",
        },
        {
            "claim_id": "C05",
            "claim_text": "Exposure-count variables are absent from the primary model.",
            "claim_category": "supporting",
            "evidence_file": "final_model/final_model_manifest.json",
            "evidence_table_figure": "Table 3",
            "metric_statistic": "No prohibited exposure-count variables in D3.",
            "sample_size": "12 final features",
            "caveat": "Text exposure audits remain in appendix.",
            "manuscript_section": "Methods",
            "status": "supported",
        },
        {
            "claim_id": "C06",
            "claim_text": "Raw speed does not dominate.",
            "claim_category": "supporting",
            "evidence_file": "stress_tests/feature_stability.csv",
            "evidence_table_figure": "Table 7; Figure 8",
            "metric_statistic": "D3 uses DFM residual gaze slopes, not raw speed/global duration aggregates.",
            "sample_size": "57 LOPO folds",
            "caveat": "Residual gaze costs can still reflect reading dynamics related to speed.",
            "manuscript_section": "Analysis and Interpretation",
            "status": "supported",
        },
        {
            "claim_id": "C07",
            "claim_text": "DFM surprisal interactions provide explanatory support.",
            "claim_category": "secondary",
            "evidence_file": "tables/interaction_synthesis.csv",
            "evidence_table_figure": "Table 8; Figure 10",
            "metric_statistic": "Several controlled DFM surprisal interactions survive.",
            "sample_size": "word-level controlled rows from Phase 4",
            "caveat": "Interaction models are secondary and cluster-robust fallback models are documented.",
            "manuscript_section": "Analysis and Interpretation",
            "status": "partially_supported",
        },
        {
            "claim_id": "C08",
            "claim_text": "Boundary opacity is secondary.",
            "claim_category": "secondary",
            "evidence_file": "tables/interaction_synthesis.csv",
            "evidence_table_figure": "Table 8; Figure 10",
            "metric_statistic": "Boundary-opacity interaction retained only for interpretation.",
            "sample_size": "word-level controlled rows from Phase 4",
            "caveat": "Labels are deterministic orthographic proxies.",
            "manuscript_section": "Discussion",
            "status": "partially_supported",
        },
        {
            "claim_id": "C09",
            "claim_text": "Standalone segmentation main effect is not supported.",
            "claim_category": "appendix",
            "evidence_file": "decision/final_decision.json",
            "evidence_table_figure": "Supplement Section 12",
            "metric_statistic": "Standalone segmentation framing dropped.",
            "sample_size": "frozen Phase 4 decision",
            "caveat": "Pronunciation-aware labels deferred.",
            "manuscript_section": "Limitations",
            "status": "appendix_only",
        },
        {
            "claim_id": "C10",
            "claim_text": "Word-level classification is secondary.",
            "claim_category": "appendix",
            "evidence_file": "analysis/autoresearch_v1/manuscript/10_appendix_plan.md",
            "evidence_table_figure": "Supplement Section 13",
            "metric_statistic": "Participant label is the target; participant-level prediction is primary.",
            "sample_size": "335203 word rows are not independent participant labels",
            "caveat": "No random word-level split is used.",
            "manuscript_section": "Methods",
            "status": "appendix_only",
        },
    ]
    frame = pd.DataFrame(rows)
    for base in [dirs["result_root"], dirs["analysis"]]:
        _write_csv(base / "claim_evidence_ledger.csv", frame)
        _write_text(
            base / "claim_evidence_ledger.md",
            "# Claim-Evidence Ledger\n\n"
            + _markdown_table(frame.to_dict("records"), list(frame.columns), max_rows=40),
        )
    return frame


def _build_tables(dirs: dict[str, Path], data: dict[str, Any], ledger: Any) -> dict[str, str]:
    pd = _pd()
    root = dirs["autoresearch"]
    written = {}
    for name, spec in SUBMISSION_TABLES.items():
        if name == "calibration_influence_summary":
            calibration = data["calibration"].head(1).copy()
            influence = data["influence"]
            frame = pd.DataFrame(
                [
                    {
                        "brier_score": _final_metric(data)["brier_score"],
                        "calibration_intercept": _final_metric(data)["calibration_intercept"],
                        "calibration_slope": _final_metric(data)["calibration_slope"],
                        "misclassified_participants": int(influence["misclassified"].sum()),
                        "high_leverage_participants": int(influence["high_leverage_flag"].sum()),
                        "calibration_bins": int(len(data["calibration"]) - 1),
                    }
                ]
            )
            if not calibration.empty:
                frame["mean_predicted"] = calibration.iloc[0].get("mean_predicted")
                frame["observed_rate"] = calibration.iloc[0].get("observed_rate")
        elif name == "main_claims_evidence":
            frame = ledger[
                [
                    "claim_id",
                    "claim_text",
                    "claim_category",
                    "metric_statistic",
                    "status",
                ]
            ].copy()
        else:
            frame = pd.read_csv(root / str(spec["source"]))
        _save_table_bundle(name, frame, str(spec["caption"]), str(spec["label"]), dirs)
        written[name] = str(Path("tables") / f"{name}.csv")
    return written


def _build_crossfit_figure(path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.axis("off")
        boxes = [
            (0.05, 0.55, "Training\nparticipants"),
            (0.32, 0.55, "Fit expected\ngaze model"),
            (0.59, 0.55, "Held-out\nparticipant rows"),
            (0.32, 0.15, "Predict expected\ngaze"),
            (0.59, 0.15, "Aggregate residual\nDFM slopes"),
            (0.83, 0.35, "Participant\nprediction"),
        ]
        for x, y, text in boxes:
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black"},
                fontsize=10,
            )
        arrows = [
            ((0.14, 0.55), (0.23, 0.55)),
            ((0.41, 0.55), (0.50, 0.55)),
            ((0.59, 0.47), (0.40, 0.24)),
            ((0.41, 0.15), (0.50, 0.15)),
            ((0.68, 0.15), (0.77, 0.32)),
        ]
        for start, end in arrows:
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, lw=1.2))
        ax.text(0.50, 0.92, "Reader group is never used in residualization", ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
    except Exception as exc:
        _write_text(path.with_suffix(".md"), f"# Figure skipped\n\nCould not render schematic: `{exc}`")


def _build_figures(dirs: dict[str, Path]) -> dict[str, str]:
    written = {}
    root = dirs["autoresearch"]
    for name, spec in SUBMISSION_FIGURES.items():
        rel = Path("figures") / f"{name}.png"
        if spec["source"] is None:
            for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
                _build_crossfit_figure(base / rel)
        else:
            src = root / str(spec["source"])
            if src.exists():
                _copy_to_all(src, str(rel), dirs)
            else:
                for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
                    _write_text(base / "figures" / f"{name}.md", f"# Figure skipped\n\nMissing source `{src}`.")
        written[name] = str(rel)
    return written


def _latex_main() -> str:
    inputs = "\n".join(rf"\input{{sections/{section}.tex}}" for section in MANUSCRIPT_SECTIONS)
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\title{{{_tex_escape(FINAL_TITLE)}}}
\author{{CopCo Eye Bench Research Package}}
\date{{}}
\begin{{document}}
\maketitle
{inputs}
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}"""


def _figure_tex(name: str) -> str:
    spec = SUBMISSION_FIGURES[name]
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=0.92\linewidth]{{figures/{name}.png}}",
            rf"\caption{{{_tex_escape(spec['caption'])}}}",
            rf"\label{{{spec['label']}}}",
            r"\end{figure}",
        ]
    )


def _metric_sentence(metric: dict[str, Any]) -> str:
    return (
        f"ROC-AUC {metric['roc_auc']:.4f}, PR-AUC {metric['pr_auc']:.4f}, "
        f"balanced accuracy {metric['balanced_accuracy']:.4f}, macro F1 {metric['macro_f1']:.4f}, "
        f"Brier score {metric['brier_score']:.4f}, calibration intercept "
        f"{metric['calibration_intercept']:.4f}, and calibration slope {metric['calibration_slope']:.4f}"
    )


def _manuscript_sections(data: dict[str, Any]) -> dict[str, str]:
    metric = _final_metric(data)
    metrics_text = _metric_sentence(metric)
    return {
        "00_abstract": rf"""\begin{{abstract}}
We study whether Danish natural-reading eye movements can distinguish dyslexia-labeled
and typical/control readers at the participant level. Using the frozen CopCo
dyslexia-labeled reader package, we evaluate participant-level prediction from
Danish Foundation Models (DFM) predictability sensitivity and cross-fitted
residualized gaze-cost profiles. The locked final model is a logistic regression over
\texttt{{D3\_dfm\_residual\_gaze\_only}} features evaluated with leave-one-participant-out
validation. It obtains {metrics_text} over 57 participants. A DFM exposure-only model
is weak (ROC-AUC 0.4238, PR-AUC 0.3685), while DFM sensitivity-only and residual gaze
models are strong, indicating that prediction is driven by participant sensitivity
rather than text exposure. Robustness checks include 1,000 label permutations
($p=0.000999$), participant bootstrap intervals for ROC-AUC [0.7765, 0.9841] and PR-AUC
[0.7083, 0.9728], and leave-one-dyslexia-labeled sensitivity. Boundary-opacity
interactions are retained as secondary interpretability evidence. The study remains
limited by 57 participants, operational label provenance, and the absence of an
independent external validation dataset.
\end{{abstract}}""",
        "01_introduction": r"""\section{Introduction}
Eye movements provide a time-resolved behavioral trace of reading. Fixation durations,
refixations, skipping, and go-past measures respond to lexical, contextual, and reader
factors, making natural-reading gaze data a useful setting for studying how linguistic
pressure differs across reader groups \cite{rayner1998eye,duchowski2017eye}.
Danish natural reading is especially relevant because orthography, morphology, and
vocalic boundary patterns create pressures that are not identical to English-centered
benchmarks. Prior Danish dyslexia-reader prediction from eye movements has shown that
reader group can be predicted from gaze behavior, but the interpretability of such
predictions and the linguistic pressures driving them remain underdeveloped.

Language-model predictability offers a bridge between psycholinguistic reading-time
theory and modern NLP. The central question in this paper is not only whether
dyslexia-labeled and typical/control readers can be distinguished, but whether the
distinction is carried by text exposure, global reading speed, or participant-level
sensitivity to DFM predictability. We answer this with a frozen confirmatory analysis:
DFM predictability sensitivity and cross-fitted residualized gaze-cost profiles provide
the main predictive signal.""",
        "02_related_work": r"""\section{Related Work}
Natural-reading eye-tracking corpora support analyses of reading behavior under
connected text rather than isolated word presentation \cite{kennedy2003d,kliegl2006tracking}.
CopCo provides Danish natural-reading material that can be aligned with gaze,
linguistic, and language-model features. We treat the current labels as operational
research labels and do not claim clinical diagnosis.

Reader-level prediction from eye movements has been explored in dyslexia and reading
difficulty settings, including Danish work on gaze-based prediction. Our contribution is
not a broader model search, but an interpretable confirmatory package that separates
text exposure from participant-level predictability sensitivity. LM surprisal has long
been connected to reading time and processing difficulty \cite{hale2001probabilistic,
levy2008expectation,smith2013effect}. We use DFM surprisal and entropy as contextual
predictability features, then ask whether participant gaze costs vary with these
features. Danish vocalic and boundary-opacity literature motivates the secondary
orthographic boundary-opacity analysis, but the current labels are orthographic proxies
rather than pronunciation-aware phonological labels. Explainable reader-level prediction
requires grouped validation, leakage controls, and feature-stability analysis rather
than random word-level splits.""",
        "03_data": r"""\section{Data}
The frozen prepared dataset contains 57 participants: 19 dyslexia-labeled and 38
typical/control readers. The prepared tables contain 335,203 word-level rows, 1,986
sentence-level rows, and 57 participant-level rows. Table~\ref{tab:dataset-summary}
summarizes the dataset, and Table~\ref{tab:feature-label-release} summarizes the frozen
feature and label releases.

The analysis uses Feature Release v1 and Label Release v1.1. Quality labels record LM
missingness, parser status, and label availability. The parser status is
\texttt{surface\_heuristic\_fallback}, so no parser-syntax claim is made. Segmentation
labels are deterministic orthographic boundary-opacity proxies, not pronunciation-aware
labels. Text assignment is not randomized; this motivates the DFM exposure-only and
text-exposure sensitivity checks. All predictive validation is participant-grouped, with
leave-one-participant-out as the primary split policy.""",
        "04_features_and_labels": r"""\section{Features and Labels}
The feature package combines gaze measures, lexical covariates, DFM word-level
surprisal and entropy, orthographic boundary-opacity proxies, and participant-level
sensitivity profiles. The DFM features are based on
\texttt{danish-foundation-models/dfm-decoder-open-v0-7b-pt}. Participant labels are
operational research labels with two groups: dyslexia-labeled and typical/control.

The final model uses only cross-fitted DFM residual gaze-cost features. It excludes
direct exposure-count variables such as number of words read, number of speeches,
number of word rows, total word rows, and word-observation count. All selected final
features are documented in the supplement and in Table~\ref{tab:feature-stability}.""",
        "05_methods": r"""\section{Methods}
The locked primary model is \texttt{D3\_dfm\_residual\_gaze\_only}: a standardized
logistic regression trained and evaluated with leave-one-participant-out validation.
The prediction unit is the participant, and each fold produces one held-out participant
prediction.

Cross-fitted residualization is performed inside the held-out participant split
(Figure~\ref{fig:crossfit-schematic}). For each held-out participant, expected gaze
models are fit using only training participants, then used to predict expected gaze for
the held-out participant's word rows. Residual gaze costs are aggregated into
participant-level DFM surprisal and entropy sensitivity slopes. Reader group is never
used in residualization.

The confirmatory stress tests separate DFM exposure from DFM sensitivity, remove
exposure-count and raw-speed/global-duration variables, audit text exposure, estimate
calibration, run permutation and bootstrap uncertainty checks, assess participant
influence, and summarize coefficient stability. No random word-level split is used.""",
        "06_results": rf"""\section{{Results}}
The final participant-level model obtains {metrics_text} with 57 predictions and zero
skipped folds (Table~\ref{{tab:final-model-metrics}}, Figure~\ref{{fig:final-roc}},
Figure~\ref{{fig:final-pr}}). The DFM exposure-only model is weak, while sensitivity
and residual gaze models are strong (Table~\ref{{tab:dfm-exposure-sensitivity}},
Figure~\ref{{fig:dfm-exposure-sensitivity}}). This supports the interpretation that
the signal reflects participant-level predictability sensitivity rather than the text
assigned to a participant.

The permutation test uses 1,000 valid permutations and gives $p=0.000999$.
Bootstrap intervals are [0.7765, 0.9841] for ROC-AUC and [0.7083, 0.9728] for PR-AUC
(Table~\ref{{tab:robustness}}, Figure~\ref{{fig:permutation-null}},
Figure~\ref{{fig:bootstrap-auc}}). Calibration and influence summaries are reported in
Table~\ref{{tab:calibration-influence}}, Figure~\ref{{fig:calibration}}, and
Figure~\ref{{fig:participant-error}}. Feature stability is shown in
Table~\ref{{tab:feature-stability}} and Figure~\ref{{fig:feature-stability}}.""",
        "07_analysis_and_interpretation": r"""\section{Analysis and Interpretation}
DFM sensitivity matters because it captures how participants' residual gaze costs vary
with contextual predictability after lexical and text-level covariates are controlled
within the cross-fitted residualization pipeline. The exposure-only comparison rejects
the simpler explanation that participants are separated merely by reading different
text. The primary model also excludes direct exposure-count variables, and the
raw-speed/global-duration family is not the selected feature family.

Focused reader-group interactions provide interpretability rather than the central
claim. DFM surprisal interactions provide the strongest explanatory support, word-length
interactions provide secondary support, and previous-boundary opacity is retained as a
secondary interpretability feature (Table~\ref{tab:interaction-synthesis},
Figure~\ref{fig:interaction-summary}). Text-exposure audits are shown in
Figure~\ref{fig:text-exposure-audit}. The claim ledger in Table~\ref{tab:claims-evidence}
defines which results are main, secondary, appendix-only, deferred, or dropped.""",
        "08_limitations": r"""\section{Limitations}
The main limitation is sample size: the analysis contains 57 participants, including 19
dyslexia-labeled readers. Labels are operational research labels and should not be
interpreted as clinical diagnosis, screening, or medical validation. The package does
not include an independent external dataset, so generalization beyond this Danish
natural-reading setting remains open.

Text assignment is not randomized, which is why the paper emphasizes exposure-only and
text-exposure audits. LM alignment warnings are recorded and should be considered when
interpreting individual token-level features. Boundary-opacity labels are deterministic
orthographic proxies, not pronunciation-aware segmentation labels. Parser status is
\texttt{surface\_heuristic\_fallback}, so parser-syntax claims are deferred. Gemma
sensitivity is pending gated access and is not used in the main claim. Reviewer risks
are summarized in Table~\ref{tab:reviewer-risk}.""",
        "09_conclusion": r"""\section{Conclusion}
The frozen confirmatory package supports one main claim: participant-level DFM
predictability sensitivity and cross-fitted residualized gaze-cost profiles distinguish
dyslexia-labeled and typical/control readers in Danish natural reading. The result is
robust under participant-level validation and is not explained by DFM exposure-only
features or direct exposure-count variables. Boundary opacity and focused interaction
analyses are useful for interpretation, but they are secondary to the participant-level
DFM residual gaze-profile result.""",
    }


def _write_manuscript(dirs: dict[str, Path], data: dict[str, Any]) -> list[str]:
    sections = _manuscript_sections(data)
    for section, text in sections.items():
        _write_text(dirs["paper"] / "sections" / f"{section}.tex", text)
        _write_text(dirs["analysis"] / "manuscript" / f"{section}.md", _tex_to_md(section, text))
        _write_text(dirs["result_root"] / "manuscript" / f"{section}.md", _tex_to_md(section, text))
    _write_text(dirs["paper"] / "main.tex", _latex_main())
    _write_text(dirs["paper"] / "references.bib", _references_bib())
    _write_text(dirs["result_root"] / "manuscript" / "main.tex", _latex_main())
    _write_text(dirs["result_root"] / "manuscript" / "references.bib", _references_bib())
    return [f"sections/{section}.tex" for section in MANUSCRIPT_SECTIONS]


def _tex_to_md(section: str, text: str) -> str:
    cleaned = re.sub(r"\\section\{([^}]+)\}", r"# \1", text)
    cleaned = cleaned.replace(r"\begin{abstract}", "# Abstract").replace(r"\end{abstract}", "")
    cleaned = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", cleaned)
    cleaned = re.sub(r"\\cite\{([^}]+)\}", r"[\1]", cleaned)
    cleaned = re.sub(r"Table~\\ref\{([^}]+)\}", r"Table (\1)", cleaned)
    cleaned = re.sub(r"Figure~\\ref\{([^}]+)\}", r"Figure (\1)", cleaned)
    cleaned = cleaned.replace("\\_", "_").replace("$", "")
    return cleaned.strip()


def _references_bib() -> str:
    return """@article{rayner1998eye,
  title={Eye movements in reading and information processing},
  author={Rayner, Keith},
  journal={Psychological Bulletin},
  year={1998}
}

@book{duchowski2017eye,
  title={Eye Tracking Methodology},
  author={Duchowski, Andrew T.},
  year={2017},
  publisher={Springer}
}

@article{kennedy2003d,
  title={The Dundee corpus},
  author={Kennedy, Alan and Pynte, Jo{\"e}l},
  journal={Proceedings of the 12th European Conference on Eye Movements},
  year={2003}
}

@article{kliegl2006tracking,
  title={Tracking the mind during reading},
  author={Kliegl, Reinhold and Nuthmann, Antje and Engbert, Ralf},
  journal={Journal of Experimental Psychology: General},
  year={2006}
}

@inproceedings{hale2001probabilistic,
  title={A probabilistic Earley parser as a psycholinguistic model},
  author={Hale, John},
  booktitle={NAACL},
  year={2001}
}

@article{levy2008expectation,
  title={Expectation-based syntactic comprehension},
  author={Levy, Roger},
  journal={Cognition},
  year={2008}
}

@article{smith2013effect,
  title={The effect of word predictability on reading time is logarithmic},
  author={Smith, Nathaniel J. and Levy, Roger},
  journal={Cognition},
  year={2013}
}
"""


def _supplement_tex() -> str:
    inputs = "\n".join(
        rf"\input{{supplement_sections/{section}.tex}}" for section in SUPPLEMENT_SECTIONS
    )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\title{{Supplement: {_tex_escape(FINAL_TITLE)}}}
\author{{CopCo Eye Bench Research Package}}
\date{{}}
\begin{{document}}
\maketitle
{inputs}
\end{{document}}"""


def _supplement_sections(data: dict[str, Any]) -> dict[str, str]:
    metric = _final_metric(data)
    feature_list = "\n".join(
        f"\\item \\texttt{{{_tex_escape(feature)}}}"
        for feature in data["final_model_manifest"].get("features", [])
    )
    return {
        "01_feature_dictionary": "\\section{Full Feature Dictionary}\n"
        + _tex_escape(data["feature_dictionary_text"][:2500]),
        "02_final_model_feature_list": "\\section{Final Model Feature List}\n\\begin{itemize}\n"
        + feature_list
        + "\n\\end{itemize}",
        "03_dfm_exposure_vs_sensitivity": "\\section{DFM Exposure Versus Sensitivity}\n"
        "The full DFM comparison is provided in the submission tables. Exposure-only "
        "features are not selected for the main paper result.",
        "04_robustness_results": "\\section{Robustness Results}\n"
        f"The final model has ROC-AUC {metric['roc_auc']:.4f} and PR-AUC {metric['pr_auc']:.4f}. "
        "Robustness includes permutation, bootstrap, and influence analyses.",
        "05_permutation_details": "\\section{Permutation Details}\n"
        "The permutation test reuses the frozen AutoResearch v1 output with 1,000 valid "
        "label permutations and the standard +1 correction.",
        "06_bootstrap_details": "\\section{Bootstrap Details}\n"
        "The bootstrap intervals are participant-level uncertainty summaries for ROC-AUC "
        "and PR-AUC.",
        "07_participant_influence_error": "\\section{Participant Influence and Error Analysis}\n"
        "Participant-level misclassification and high-leverage audits are included in the "
        "generated tables and figures.",
        "08_calibration_details": "\\section{Calibration Details}\n"
        f"The Brier score is {metric['brier_score']:.4f}; calibration intercept is "
        f"{metric['calibration_intercept']:.4f}; calibration slope is "
        f"{metric['calibration_slope']:.4f}.",
        "09_text_exposure_audit": "\\section{Text-Exposure Audit}\n"
        "The text-exposure audit checks prediction scores and errors against speech count, "
        "word rows, DFM exposure, segmentation exposure, and comprehension where available.",
        "10_lm_warning_audit": "\\section{LM Warning Audit}\n"
        "LM missingness and warning fields are frozen in Label Release v1.1 and summarized "
        "in AutoResearch v1.",
        "11_boundary_opacity_labels": "\\section{Boundary-Opacity Label Construction}\n"
        "Boundary-opacity labels are deterministic orthographic proxies based on vowel and "
        "consonant patterns across word boundaries.",
        "12_segmentation_null_result": "\\section{Segmentation Null-Result Details}\n"
        "Standalone segmentation-opacity main-effect framing is not selected and is treated "
        "as a dropped main-paper result.",
        "13_word_level_secondary_ladder": "\\section{Word-Level Secondary Ladder}\n"
        "Word-level classification is not the central result because the target label is "
        "participant-level and word rows are not independent participant labels.",
        "14_split_policy": "\\section{Split Policy}\n"
        "The primary split is leave-one-participant-out. Random word-level splitting is "
        "forbidden and absent from the submission package.",
        "15_reproducibility_commands": "\\section{Reproducibility Commands}\n"
        "The reproducibility capsule includes commands to validate the frozen releases, "
        "rerun AutoResearch v1, build this package, and validate the package.",
        "16_dataset_caveats": "\\section{Dataset Caveats}\n"
        "The package has 57 participants, operational labels, Danish-only material, and no "
        "independent external validation dataset.",
        "17_reviewer_risk_notes": "\\section{Extended Reviewer-Risk Notes}\n"
        "Reviewer-risk and rebuttal-preparation notes are mirrored under analysis/submission_v1.",
    }


def _write_supplement(dirs: dict[str, Path], data: dict[str, Any]) -> list[str]:
    sections = _supplement_sections(data)
    for section, text in sections.items():
        _write_text(dirs["paper"] / "supplement_sections" / f"{section}.tex", text)
        _write_text(dirs["analysis"] / "supplement" / f"{section}.md", _tex_to_md(section, text))
        _write_text(dirs["result_root"] / "supplement" / f"{section}.md", _tex_to_md(section, text))
    _write_text(dirs["paper"] / "supplement.tex", _supplement_tex())
    _write_text(dirs["result_root"] / "supplement" / "supplement.tex", _supplement_tex())
    return [f"supplement_sections/{section}.tex" for section in SUPPLEMENT_SECTIONS]


def _build_reviewer_notes(dirs: dict[str, Path]) -> None:
    simulation = """# Reviewer Simulation

## Reviewer 1: NLP/ML

- Likely score: borderline accept to accept.
- Strengths: leakage-aware LOPO validation, frozen result, permutation/bootstrap robustness,
  reproducibility capsule.
- Concerns: small participant count, potential feature-selection narrative, external
  generalization.
- Manuscript changes needed: foreground locked model selection and no random word-level
  splits.
- Appendix evidence needed: feature dictionary, split policy, permutation/bootstrap details.
- Rebuttal-ready response: the main claim is participant-level and uses frozen
  participant-grouped validation; exposure-only and count-variable checks reject the
  most direct leakage/confound explanations.

## Reviewer 2: Psycholinguistics / Eye Tracking

- Likely score: weak accept to accept.
- Strengths: residualized gaze-cost profiles, DFM predictability bridge, focused
  interaction synthesis.
- Concerns: mixed-effects fallback, interpretability of residual slopes, calibration with
  small N.
- Manuscript changes needed: explain gaze outcomes and residualization in plain language.
- Appendix evidence needed: interaction synthesis and calibration details.
- Rebuttal-ready response: word-level interactions are secondary; the primary result is a
  participant-level confirmatory prediction result with cross-fitted residualization.

## Reviewer 3: Danish / Reading / Dyslexia Domain

- Likely score: weak accept if limitations are clear.
- Strengths: Danish natural reading focus, careful label language, boundary-opacity
  interpretability.
- Concerns: operational label provenance, no clinical claim, orthographic boundary proxy,
  Gemma pending.
- Manuscript changes needed: make label provenance and Danish-only scope explicit.
- Appendix evidence needed: segmentation construction, null standalone segmentation result,
  dataset caveats.
- Rebuttal-ready response: the paper explicitly avoids clinical screening claims and treats
  boundary opacity as secondary interpretation, not the main effect.
"""
    rebuttal = """# Rebuttal Preparation

The central rebuttal position is that the paper reports a frozen confirmatory package, not
an exploratory modeling search. The selected model, metrics, robustness checks, and
scientific decisions were fixed in AutoResearch v1.

## Likely Objection: Leakage Or Text Exposure

Response: the primary model is participant-level LOPO, contains no direct exposure-count
variables, and DFM exposure-only performs poorly. Cross-fitted residualization fits gaze
models only on training participants for each held-out participant.

## Likely Objection: Small Participant Count

Response: the paper frames the result as a strong internal natural-reading finding and
does not claim clinical validation. Bootstrap, permutation, and influence analyses are
reported, and external validation is listed as future work.

## Likely Objection: Segmentation Interpretation

Response: standalone segmentation main-effect framing is dropped. Boundary opacity is a
secondary interpretability feature, and pronunciation-aware labels are deferred.

## Likely Objection: Parser Or Syntax Claims

Response: parser status is surface heuristic fallback; parser-syntax claims are explicitly
deferred.
"""
    target = """# Submission Target Analysis

## ACL / EMNLP / COLING Main Conference

- Fit: moderate.
- Strengths: NLP plus cognition bridge, reproducible validation, LM predictability.
- Weaknesses: small N and no external dataset.
- Required emphasis: methodological rigor, leakage controls, and interpretability.
- Reviewer risks: overclaiming clinical/dyslexia utility.
- Recommended framing: concise NLP-for-reading analysis, not a benchmark leaderboard.

## Findings-Style Venue

- Fit: strong.
- Strengths: complete confirmatory result and careful limitations.
- Weaknesses: not a broad model contribution.
- Required emphasis: frozen analysis package and scientific claim boundaries.
- Reviewer risks: demand for external validation.
- Recommended framing: focused empirical finding with robust appendices.

## NLP + Cognitive Modeling Workshop

- Fit: very strong.
- Strengths: DFM predictability, gaze behavior, cross-fitted residual profiles.
- Weaknesses: narrower audience and less archival weight.
- Required emphasis: psycholinguistic interpretation and reproducibility.
- Reviewer risks: less interest in resource engineering.
- Recommended framing: methods-forward cognitive NLP contribution.

## LREC / COLING-Style Resource Venue

- Fit: moderate to strong.
- Strengths: prepared release pipeline, labels, manifests, reproducibility capsule.
- Weaknesses: primary result is analysis rather than a public raw dataset release.
- Required emphasis: resource and validation package.
- Reviewer risks: data availability constraints.
- Recommended framing: reproducible research package over Danish natural reading.

## Psycholinguistics / Behavioral Journal

- Fit: moderate.
- Strengths: eye-tracking and predictability sensitivity.
- Weaknesses: ML framing and operational labels may need deeper behavioral theory.
- Required emphasis: gaze measures, residualization, and cautious interpretation.
- Reviewer risks: model validation norms differ from NLP venues.
- Recommended framing: computational psycholinguistics case study.

## Recommendation

Primary target type: Findings-style NLP venue or cognitive NLP workshop, depending on
desired risk. Fallback target type: LREC/COLING-style resource venue. Writing strategy:
lead with the frozen participant-level DFM sensitivity result, keep clinical and
standalone segmentation claims out of the main story, and move diagnostics to the
supplement.
"""
    final = """# Final Submission Decision Report

1. Ready for full manuscript drafting: `ready_for_full_manuscript_revision`.
2. The result is strong enough to be the main paper claim as an internal confirmatory
   participant-level result.
3. Final main claim: participant-level DFM predictability sensitivity and cross-fitted
   residualized gaze-cost profiles distinguish dyslexia-labeled and typical/control
   readers in Danish natural reading.
4. Final title: Predictability-Sensitive Gaze Profiles for Dyslexia-Labeled Reader
   Prediction in Danish Natural Reading.
5. Contributions: prepared Danish natural-reading pipeline; cross-fitted residualized
   sensitivity-profile method; DFM sensitivity rather than exposure evidence; secondary
   interaction interpretation.
6. Strongest evidence: D3 LOPO ROC-AUC 0.8947, PR-AUC 0.8641, permutation p=0.000999,
   ROC-AUC bootstrap lower bound 0.7765.
7. Weakest evidence: small participant count, operational labels, no external dataset.
8. Appendix: full feature dictionary, robustness, calibration, influence, text exposure,
   segmentation null, word-level secondary ladder.
9. Remove from main paper: random word-level prediction, standalone segmentation
   main-effect framing, parser-syntax claims, clinical utility claims.
10. Target strategy: primary Findings-style or cognitive NLP venue; fallback resource venue.
11. Remaining before actual submission: prose tightening, venue formatting, citation
   polishing, and optional PDF compilation.
12. Do not change: selected model, frozen metrics, main claim, no-random-split policy,
   and segmentation-as-secondary decision.
"""
    for name, text in [
        ("reviewer_simulation.md", simulation),
        ("rebuttal_preparation.md", rebuttal),
        ("submission_target_analysis.md", target),
        ("final_submission_decision_report.md", final),
    ]:
        _write_text(dirs["analysis"] / name, text)
        _write_text(dirs["result_root"] / "decision" / name, text)


def _build_reproducibility(config: dict[str, Any], dirs: dict[str, Path], output_dir: Path) -> None:
    inputs = get_nested(config, "submission.frozen_inputs", {})
    scripts = {
        "reproduce_submission_package.sh": """#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/submission_v1_reproduced}"
conda run -n copco copco-build-submission-package --config configs/submission_v1.yaml --output-dir "$OUT" --allow-existing-output
conda run -n copco copco-validate-submission-package --config configs/submission_v1.yaml --output-dir "$OUT"
""",
        "reproduce_autoresearch_v1.sh": """#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/autoresearch_v1_reproduced}"
conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir "$OUT"
conda run -n copco copco-validate-autoresearch --config configs/autoresearch_v1.yaml --output-dir "$OUT"
""",
        "slurm_submission_package.sh": """#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/submission_v1_$(date +%Y%m%d_%H%M)}"
CLAIM_RESOURCE_LOG_DIR=logs/submission_v1_resource_logs \\
  ~/bin/claim_best_immediate_resource.sh --mode cpu \\
  --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" \\
  "cd $(pwd) && conda run -n copco copco-build-submission-package --config configs/submission_v1.yaml --output-dir $OUT"
""",
    }
    docs = {
        "README_REPRODUCE.md": "# Reproduce Submission Package\n\nRun `reproduce_submission_package.sh` to rebuild and validate only the submission package. Run `reproduce_autoresearch_v1.sh` to regenerate the frozen AutoResearch v1 source package. Generated `results/` directories are not committed.\n",
        "environment_summary.md": f"# Environment Summary\n\n- Environment: `copco`\n- Git SHA: `{_git_sha(dirs['repo_root'])}`\n- SLURM job id: `{os.environ.get('SLURM_JOB_ID', '')}`\n",
        "command_manifest.md": "# Command Manifest\n\n- `conda run -n copco python scripts/validate_env.py`\n- `conda run -n copco copco-build-submission-package --config configs/submission_v1.yaml --output-dir results/submission_v1_<timestamp>`\n- `conda run -n copco copco-validate-submission-package --config configs/submission_v1.yaml --output-dir results/submission_v1_<timestamp>`\n",
        "input_output_manifest.md": "# Input Output Manifest\n\n"
        + _markdown_table(
            [{"input": key, "path": value} for key, value in inputs.items()],
            ["input", "path"],
        )
        + f"\n\nOutput directory: `{output_dir}`\n",
        "commit_trace.md": f"# Commit Trace\n\n- Submission package built from git SHA `{_git_sha(dirs['repo_root'])}`.\n- Frozen AutoResearch source: `results/autoresearch_v1_20260506_0917`.\n",
        "data_not_committed_notice.md": "# Data Not Committed Notice\n\nThe full generated `results/submission_v1_*` directory, frozen result directories, Parquet files, embeddings, and model artifacts are not committed. The small paper and analysis mirrors are committed.\n",
    }
    artifact_manifest = {
        "paper_dir": str(dirs["paper"]),
        "analysis_dir": str(dirs["analysis"]),
        "result_dir": str(output_dir),
        "committed_expected": ["paper/submission_v1", "analysis/submission_v1"],
        "not_committed_expected": ["results/submission_v1_*", "large Parquet/model artifacts"],
    }
    for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
        repro = base / "reproducibility"
        for name, text in scripts.items():
            path = repro / name
            _write_text(path, text)
            path.chmod(0o755)
        for name, text in docs.items():
            _write_text(repro / name, text)
        _write_json(repro / "artifact_manifest.json", artifact_manifest)


def _write_checksums(dirs: dict[str, Path]) -> dict[str, str]:
    checksums = {}
    for path in sorted(dirs["result_root"].rglob("*")):
        if not path.is_file() or path.name == "checksums.json":
            continue
        if path.stat().st_size >= 100_000_000:
            continue
        checksums[str(path.relative_to(dirs["result_root"]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for base in [dirs["result_root"], dirs["paper"], dirs["analysis"]]:
        _write_json(base / "reproducibility" / "checksums.json", checksums)
    return checksums


def _write_main_asset_index(dirs: dict[str, Path]) -> None:
    table_refs = "\n".join(
        rf"\input{{tables/{name}.tex}}" for name in SUBMISSION_TABLES
    )
    figure_refs = "\n".join(_figure_tex(name) for name in SUBMISSION_FIGURES)
    _write_text(dirs["paper"] / "sections" / "10_assets.tex", table_refs + "\n\n" + figure_refs)


def _insert_assets_into_results_section(dirs: dict[str, Path]) -> None:
    results_path = dirs["paper"] / "sections" / "06_results.tex"
    text = results_path.read_text(encoding="utf-8")
    inserts = [
        r"\input{tables/final_model_metrics.tex}",
        r"\input{tables/dfm_exposure_vs_sensitivity.tex}",
        r"\input{tables/robustness_tests.tex}",
        r"\input{tables/calibration_influence_summary.tex}",
        _figure_tex("final_roc"),
        _figure_tex("final_pr"),
        _figure_tex("dfm_exposure_vs_sensitivity"),
        _figure_tex("permutation_null"),
        _figure_tex("bootstrap_auc"),
        _figure_tex("calibration"),
        _figure_tex("participant_error"),
        _figure_tex("feature_stability"),
    ]
    _write_text(results_path, text + "\n\n" + "\n\n".join(inserts))
    intro_path = dirs["paper"] / "sections" / "03_data.tex"
    _write_text(
        intro_path,
        intro_path.read_text(encoding="utf-8")
        + "\n\n"
        + r"\input{tables/dataset_summary.tex}"
        + "\n"
        + r"\input{tables/feature_label_release_summary.tex}"
        + "\n"
        + _figure_tex("pipeline_overview"),
    )
    interp_path = dirs["paper"] / "sections" / "07_analysis_and_interpretation.tex"
    _write_text(
        interp_path,
        interp_path.read_text(encoding="utf-8")
        + "\n\n"
        + r"\input{tables/feature_stability.tex}"
        + "\n"
        + r"\input{tables/interaction_synthesis.tex}"
        + "\n"
        + r"\input{tables/main_claims_evidence.tex}"
        + "\n"
        + _figure_tex("cross_fitted_residualization_schematic")
        + "\n"
        + _figure_tex("interaction_summary")
        + "\n"
        + _figure_tex("text_exposure_audit"),
    )
    lim_path = dirs["paper"] / "sections" / "08_limitations.tex"
    _write_text(
        lim_path,
        lim_path.read_text(encoding="utf-8")
        + "\n\n"
        + r"\input{tables/reviewer_risk_summary.tex}",
    )


def build_submission_package(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    if not bool(get_nested(config, "submission.no_new_core_labels", True)):
        raise ValueError("SubmissionSprint must not add new core labels")
    if not bool(get_nested(config, "submission.no_new_feature_families", True)):
        raise ValueError("SubmissionSprint must not add new feature families")
    dirs = _dirs(config, output_dir, repo_root)
    out = dirs["result_root"]
    if out.exists() and any(out.iterdir()) and not allow_existing_output:
        raise FileExistsError(f"output directory already exists and is not empty: {out}")
    _ensure_layout(dirs)
    data = _load_package_data(config, dirs)
    locked_errors = _validate_locked_inputs(config, dirs, data)
    if locked_errors:
        bug = "# SubmissionSprint Frozen Input Bug Report\n\n" + "\n".join(
            f"- {error}" for error in locked_errors
        )
        _write_text(out / "decision" / "bug_report.md", bug)
        raise ValueError(f"frozen submission inputs failed validation: {locked_errors}")
    ledger = _build_claim_ledger(dirs, data)
    tables = _build_tables(dirs, data, ledger)
    figures = _build_figures(dirs)
    manuscript = _write_manuscript(dirs, data)
    _insert_assets_into_results_section(dirs)
    supplement = _write_supplement(dirs, data)
    _build_reviewer_notes(dirs)
    _build_reproducibility(config, dirs, out)
    checksums = _write_checksums(dirs)
    manifest = {
        "run_type": "submission_v1",
        "status": "complete",
        "output_dir": str(out),
        "paper_dir": str(dirs["paper"]),
        "analysis_dir": str(dirs["analysis"]),
        "git_sha": _git_sha(repo_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "final_title": FINAL_TITLE,
        "final_main_claim": FINAL_MAIN_CLAIM,
        "selected_model": data["final_model_manifest"],
        "final_metrics": _final_metric(data),
        "tables": tables,
        "figures": figures,
        "manuscript_sections": manuscript,
        "supplement_sections": supplement,
        "claim_count": int(len(ledger)),
        "checksums_count": len(checksums),
        "large_outputs_not_for_commit": ["results/submission_v1_*/"],
    }
    _write_json(out / "manifest.json", manifest)
    _write_json(out / "run_summary.json", {"status": "complete", "validation_expected": "run validator"})
    return manifest


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _extract_citations(text: str) -> set[str]:
    cites: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        cites.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return cites


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", text))


def validate_submission_package(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    pd = _pd()
    dirs = _dirs(config, output_dir, repo_root)
    out = dirs["result_root"]
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "manifest.json",
        "main.tex",
        "supplement.tex",
        "references.bib",
        "claim_evidence_ledger.md",
        "claim_evidence_ledger.csv",
        "reviewer_simulation.md",
        "rebuttal_preparation.md",
        "submission_target_analysis.md",
        "final_submission_decision_report.md",
    ]
    for rel in required:
        candidates = [out / rel, dirs["paper"] / rel, dirs["analysis"] / rel]
        if not any(path.exists() for path in candidates):
            errors.append(f"missing required file: {rel}")
    for section in MANUSCRIPT_SECTIONS:
        if not (dirs["paper"] / "sections" / f"{section}.tex").exists():
            errors.append(f"missing manuscript section: {section}")
        if not (dirs["analysis"] / "manuscript" / f"{section}.md").exists():
            errors.append(f"missing manuscript markdown mirror: {section}")
    for section in SUPPLEMENT_SECTIONS:
        if not (dirs["paper"] / "supplement_sections" / f"{section}.tex").exists():
            errors.append(f"missing supplement section: {section}")
        if not (dirs["analysis"] / "supplement" / f"{section}.md").exists():
            errors.append(f"missing supplement markdown mirror: {section}")
    for name, spec in SUBMISSION_TABLES.items():
        for base in [dirs["paper"], dirs["analysis"]]:
            if not (base / "tables" / f"{name}.csv").exists():
                errors.append(f"missing table: {base / 'tables' / f'{name}.csv'}")
        manuscript = _all_manuscript_text(dirs)
        if str(spec["label"]) not in manuscript:
            errors.append(f"table not referenced in manuscript: {name}")
    for name, spec in SUBMISSION_FIGURES.items():
        for base in [dirs["paper"], dirs["analysis"]]:
            if not (base / "figures" / f"{name}.png").exists() and not (
                base / "figures" / f"{name}.md"
            ).exists():
                errors.append(f"missing figure or skip report: {name}")
        manuscript = _all_manuscript_text(dirs)
        if str(spec["label"]) not in manuscript:
            errors.append(f"figure not referenced in manuscript: {name}")
    try:
        data = _load_package_data(config, dirs)
        errors.extend(_validate_locked_inputs(config, dirs, data))
    except Exception as exc:
        errors.append(f"could not load frozen AutoResearch data: {exc}")
        data = None
    ledger_path = dirs["analysis"] / "claim_evidence_ledger.csv"
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        required_claims = {f"C{idx:02d}" for idx in range(1, 11)}
        if set(ledger["claim_id"]) != required_claims:
            errors.append("claim ledger does not cover required claims C01-C10")
        main_claims = ledger[ledger["claim_category"].eq("main")]
        if main_claims.empty:
            errors.append("claim ledger has no main claim")
        abstract = _read_text_if_exists(dirs["paper"] / "sections" / "00_abstract.tex")
        coverage = {
            "DFM predictability sensitivity": "DFM residual gaze",
            "cross-fitted residualized gaze": "Cross-fitted residualization",
            "exposure-only": "DFM sensitivity dominates DFM exposure",
            "Boundary-opacity": "Boundary opacity is secondary",
        }
        for phrase, ledger_phrase in coverage.items():
            if phrase.lower() in abstract.lower() and not ledger["claim_text"].str.contains(
                ledger_phrase, case=False, regex=False
            ).any():
                errors.append(f"abstract claim not represented in ledger: {phrase}")
    else:
        errors.append("claim ledger csv missing")
    feature_text = _read_text_if_exists(dirs["autoresearch"] / "final_model" / "final_model_feature_dictionary.md")
    if data is not None:
        for feature in data["final_model_manifest"].get("features", []):
            if feature not in feature_text:
                errors.append(f"selected feature missing interpretation: {feature}")
    manuscript_text = _all_manuscript_text(dirs)
    expected = get_nested(config, "submission.expected_metrics", {})
    metric_strings = [
        f"{float(expected.get(key, value)):.4f}"
        for key, value in [
            ("roc_auc", 0.8947368421),
            ("pr_auc", 0.8640879081),
            ("balanced_accuracy", 0.8421052632),
            ("brier_score", 0.1159416341),
            ("calibration_intercept", -0.5321293913),
            ("calibration_slope", 0.8693048814),
        ]
    ]
    for value in metric_strings:
        if value not in manuscript_text:
            errors.append(f"final metric not present in manuscript: {value}")
    forbidden_main_phrases = [
        "standalone segmentation main effect is the main",
        "parser syntax main result",
        "random word-level prediction is the main",
        "validated clinical screening",
        "validated clinical diagnosis",
        "validated medical utility",
    ]
    lower = manuscript_text.lower()
    for phrase in forbidden_main_phrases:
        if phrase in lower:
            errors.append(f"prohibited main-claim phrase found: {phrase}")
    if "TODO" in manuscript_text:
        errors.append("TODO placeholder found in main manuscript")
    if "D1 exposure-only model is successful" in manuscript_text:
        errors.append("exposure-only model is described as successful")
    if "random_word" in lower or (
        "random word-level split" in lower
        and "forbidden" not in lower
        and "no random word-level split" not in lower
    ):
        warnings.append("random word-level split phrase appears outside the stated prohibition")
    citations = _extract_citations(manuscript_text)
    bib = _bib_keys(_read_text_if_exists(dirs["paper"] / "references.bib"))
    missing_cites = sorted(citations - bib)
    if missing_cites:
        errors.append(f"citations missing from references.bib: {missing_cites}")
    for name in REPRODUCIBILITY_FILES:
        if not (dirs["paper"] / "reproducibility" / name).exists():
            errors.append(f"missing reproducibility file: {name}")
    staged_large = _staged_large_files(repo_root)
    if staged_large:
        errors.append(f"large files staged for commit: {staged_large}")
    report = {"status": "passed" if not errors else "failed", "errors": errors, "warnings": warnings}
    _write_json(out / "validation_report.json", report)
    return report


def _all_manuscript_text(dirs: dict[str, Path]) -> str:
    parts = [_read_text_if_exists(dirs["paper"] / "main.tex")]
    for section in MANUSCRIPT_SECTIONS:
        parts.append(_read_text_if_exists(dirs["paper"] / "sections" / f"{section}.tex"))
    return "\n".join(parts)


def _staged_large_files(repo_root: str | Path) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "-z"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    paths = [Path(p.decode()) for p in output.split(b"\0") if p]
    large = []
    for rel in paths:
        path = Path(repo_root) / rel
        if path.exists() and path.is_file() and path.stat().st_size >= 100_000_000:
            large.append(str(rel))
    return large
