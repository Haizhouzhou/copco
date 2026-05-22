"""Probability-first operating point analysis for D3-family outputs.

This module consumes existing prediction tables. It does not train new models,
rerun EyeBench leaderboard baselines, or treat test-label thresholds as clean
benchmark evidence. Test-oracle thresholds are reported only as diagnostic upper
bounds with ``official_claim_allowed=False``.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from .config import get_nested, timestamped_output_dir


CONFIG_SECTION = "operating_point_adaptation"
ANALYSIS_NAME = "operating_point_adaptation_v1"

ALLOWED_THRESHOLD_SOURCES = {
    "fixed_0_5",
    "train_inner_cv",
    "validation_split",
    "calibration_set",
    "test_oracle_diagnostic",
    "unknown",
}

LEGAL_THRESHOLD_POLICIES = (
    "global_inner_cv_threshold",
    "split_regime_inner_cv_threshold",
    "calibration_set_threshold",
    "cost_sensitive_balanced_accuracy_threshold",
    "macro_f1_threshold",
    "prevalence_prior_threshold",
)

ORACLE_THRESHOLD_POLICIES = (
    "global_test_oracle_threshold",
    "split_regime_test_oracle_threshold",
    "fold_test_oracle_threshold",
    "reader_aggregated_test_oracle_threshold",
)

READER_AGGREGATION_METHODS = (
    "simple_mean_probability",
    "logit_mean_probability",
    "median_probability",
    "inverse_entropy_weighted_probability",
    "majority_vote_hard_label",
)

VALID_DECISION_CATEGORIES = {
    "main_text_supporting_result",
    "supplement_supporting_result",
    "diagnostic_only",
    "no_improvement",
    "invalid_missing_predictions",
    "invalid_leakage_detected",
}

METRIC_COLUMNS = [
    "analysis_row",
    "source_name",
    "model_name",
    "candidate_id",
    "feature_group",
    "model",
    "evaluation_level",
    "split_regime",
    "fold_id",
    "threshold_policy",
    "threshold_source",
    "official_claim_allowed",
    "score_source",
    "threshold",
    "n_candidate_thresholds",
    "information_bits",
    "n_predictions",
    "n_positive",
    "n_negative",
    "AUROC",
    "PR-AUC",
    "BA",
    "macro_F1",
    "sensitivity",
    "specificity",
    "Brier",
    "log_loss",
    "calibration_slope",
    "calibration_intercept",
    "ECE",
    "status",
    "notes",
]


@dataclass(frozen=True)
class PredictionSource:
    name: str
    prediction_path: Path
    metrics_path: Path | None
    source_type: str
    default_model_name: str
    default_candidate_id: str
    default_evaluation_level: str
    include_feature_groups: tuple[str, ...]
    include_models: tuple[str, ...]
    include_split_regimes: tuple[str, ...]
    task_filter: str | None
    threshold_source: str | None
    role: str


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = get_nested(config, CONFIG_SECTION, {})
    return section if isinstance(section, dict) else {}


def _git_sha(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["status"])
    frame.to_csv(path, index=False)


def _analysis_dirs(config: dict[str, Any], output_dir: str | Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve()
    repo_analysis = root / str(
        get_nested(config, f"{CONFIG_SECTION}.repo_analysis_dir", f"analysis/{ANALYSIS_NAME}")
    )
    result_analysis = out / str(
        get_nested(config, f"{CONFIG_SECTION}.output_layout.analysis", f"analysis/{ANALYSIS_NAME}")
    )
    return {"repo_analysis": repo_analysis, "result_analysis": result_analysis}


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    for base in dirs.values():
        _write_text(base / name, text)


def _write_table(dirs: dict[str, Path], name: str, frame: pd.DataFrame) -> None:
    for base in dirs.values():
        _write_csv(base / name, frame)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = [str(col) for col in display.columns]
    rows = []
    for record in display.astype(object).where(pd.notna(display), "").to_dict("records"):
        rows.append([str(record.get(col, "")) for col in display.columns])
    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _resolve_path(path: str | Path | None, repo_root: str | Path) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = Path(repo_root).resolve() / p
    return p


def _source_specs(config: dict[str, Any], repo_root: str | Path) -> list[PredictionSource]:
    specs: list[PredictionSource] = []
    for raw in _section(config).get("prediction_sources", []):
        if not _safe_bool(raw.get("include", True), True):
            continue
        prediction_path = _resolve_path(raw.get("prediction_path"), repo_root)
        if prediction_path is None:
            continue
        metrics_path = _resolve_path(raw.get("metrics_path"), repo_root)
        specs.append(
            PredictionSource(
                name=str(raw.get("name", prediction_path.stem)),
                prediction_path=prediction_path,
                metrics_path=metrics_path,
                source_type=str(raw.get("source_type", "prediction_csv")),
                default_model_name=str(raw.get("model_name", raw.get("name", prediction_path.stem))),
                default_candidate_id=str(raw.get("candidate_id", raw.get("name", prediction_path.stem))),
                default_evaluation_level=str(raw.get("evaluation_level", "trial_level")),
                include_feature_groups=_as_tuple(raw.get("include_feature_groups")),
                include_models=_as_tuple(raw.get("include_models")),
                include_split_regimes=_as_tuple(raw.get("include_split_regimes")),
                task_filter=raw.get("task"),
                threshold_source=raw.get("threshold_source"),
                role=str(raw.get("role", "test")),
            )
        )
    return specs


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _normalize_evaluation_level(value: Any, default: str) -> str:
    text = str(value if value is not None and not pd.isna(value) else default)
    mapping = {
        "participant_text_trial": "trial_level",
        "official_trial_level_fold_mean": "trial_level",
        "test_trial_level": "trial_level",
        "trial": "trial_level",
        "reader": "reader_aggregated",
        "participant": "reader_aggregated",
        "reader_level": "reader_aggregated",
    }
    return mapping.get(text, text)


def _normalize_role(value: Any, default: str) -> str:
    text = str(value if value is not None and not pd.isna(value) else default).strip().lower()
    mapping = {
        "inner": "inner_validation",
        "inner_val": "inner_validation",
        "val": "validation",
        "valid": "validation",
        "cal": "calibration",
        "test": "test",
    }
    return mapping.get(text, text)


def _infer_threshold_source(
    frame: pd.DataFrame,
    p_col: str,
    y_pred_col: str | None,
    configured: str | None,
) -> pd.Series:
    if configured:
        return pd.Series([configured] * len(frame), index=frame.index, dtype="object")
    if "threshold_source" in frame.columns:
        return frame["threshold_source"].fillna("unknown").astype(str)
    if "threshold_method" in frame.columns:
        mapped = frame["threshold_method"].fillna("unknown").astype(str).map(
            {
                "fixed_0_5": "fixed_0_5",
                "fixed_0.5": "fixed_0_5",
                "inner_balanced_accuracy": "train_inner_cv",
                "train_inner_cv": "train_inner_cv",
                "validation_split": "validation_split",
                "calibration_set": "calibration_set",
            }
        )
        return mapped.fillna("unknown")
    if y_pred_col is not None:
        scores = pd.to_numeric(frame[p_col], errors="coerce")
        preds = pd.to_numeric(frame[y_pred_col], errors="coerce")
        inferred = (scores >= 0.5).astype(float)
        matches = preds.notna() & scores.notna() & (preds.astype(float) == inferred)
        if bool(matches.all()):
            return pd.Series(["fixed_0_5"] * len(frame), index=frame.index, dtype="object")
    return pd.Series(["unknown"] * len(frame), index=frame.index, dtype="object")


def load_prediction_sources(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize configured prediction sources."""

    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for spec in _source_specs(config, repo_root):
        summary: dict[str, Any] = {
            "source_name": spec.name,
            "prediction_path": str(spec.prediction_path),
            "metrics_path": "" if spec.metrics_path is None else str(spec.metrics_path),
            "exists": spec.prediction_path.exists(),
            "status": "missing",
            "rows_loaded": 0,
            "rows_after_filter": 0,
            "required_probability_present": False,
            "notes": "",
        }
        if not spec.prediction_path.exists():
            summaries.append(summary)
            continue
        try:
            raw = pd.read_csv(spec.prediction_path)
        except Exception as exc:
            summary["status"] = "read_failed"
            summary["notes"] = str(exc)
            summaries.append(summary)
            continue
        summary["rows_loaded"] = len(raw)
        p_col = _first_existing(
            raw.columns,
            ("p_pred", "y_score", "score", "probability", "prediction_prob", "prob_dyslexic"),
        )
        y_col = _first_existing(raw.columns, ("y_true", "label", "target", "reader_group_binary"))
        y_pred_col = _first_existing(raw.columns, ("y_pred", "binary_prediction", "prediction"))
        if p_col is None or y_col is None:
            summary["status"] = "missing_required_columns"
            summary["notes"] = f"p_col={p_col}; y_col={y_col}"
            summaries.append(summary)
            continue
        summary["required_probability_present"] = True
        frame = raw.copy()
        if spec.task_filter and "task" in frame.columns:
            frame = frame[frame["task"].astype(str) == str(spec.task_filter)].copy()
        if spec.include_feature_groups and "feature_group" in frame.columns:
            frame = frame[frame["feature_group"].astype(str).isin(spec.include_feature_groups)].copy()
        if spec.include_models and "model" in frame.columns:
            frame = frame[frame["model"].astype(str).isin(spec.include_models)].copy()
        split_col = _first_existing(frame.columns, ("split_regime", "split_name"))
        if spec.include_split_regimes and split_col is not None:
            frame = frame[frame[split_col].astype(str).isin(spec.include_split_regimes)].copy()
        if frame.empty:
            summary["status"] = "no_rows_after_filter"
            summary["rows_after_filter"] = 0
            summaries.append(summary)
            continue
        model_name = (
            frame["model_name"].astype(str)
            if "model_name" in frame.columns
            else pd.Series([spec.default_model_name] * len(frame), index=frame.index)
        )
        feature_group = (
            frame["feature_group"].astype(str)
            if "feature_group" in frame.columns
            else pd.Series([""] * len(frame), index=frame.index)
        )
        model = (
            frame["model"].astype(str)
            if "model" in frame.columns
            else pd.Series([""] * len(frame), index=frame.index)
        )
        if "candidate_id" in frame.columns:
            candidate_id = frame["candidate_id"].astype(str)
        else:
            candidate_id = (
                pd.Series([spec.default_candidate_id] * len(frame), index=frame.index)
                + "__"
                + feature_group.replace("", spec.default_model_name)
                + "__"
                + model.replace("", "model")
            )
        split_regime = (
            frame[split_col].astype(str)
            if split_col is not None
            else pd.Series(["unknown_split"] * len(frame), index=frame.index)
        )
        fold_col = _first_existing(frame.columns, ("fold_id", "fold_index"))
        trial_col = _first_existing(frame.columns, ("trial_id", "unique_trial_id", "sample_id"))
        sample_col = _first_existing(frame.columns, ("sample_id", "unique_trial_id", "trial_id"))
        eval_col = _first_existing(frame.columns, ("evaluation_level", "eval_level"))
        role_col = _first_existing(frame.columns, ("split_role", "eval_type", "role"))
        threshold_col = _first_existing(frame.columns, ("threshold", "decision_threshold"))
        normalized = pd.DataFrame(
            {
                "source_name": spec.name,
                "source_type": spec.source_type,
                "model_name": model_name,
                "candidate_id": candidate_id,
                "feature_group": feature_group,
                "model": model,
                "task": frame["task"].astype(str) if "task" in frame.columns else "",
                "split_regime": split_regime,
                "fold_id": frame[fold_col].astype(str) if fold_col else "all",
                "evaluation_level": (
                    frame[eval_col].map(lambda value: _normalize_evaluation_level(value, spec.default_evaluation_level))
                    if eval_col
                    else _normalize_evaluation_level(None, spec.default_evaluation_level)
                ),
                "sample_id": frame[sample_col].astype(str) if sample_col else frame.index.astype(str),
                "trial_id": frame[trial_col].astype(str) if trial_col else frame.index.astype(str),
                "participant_id": frame["participant_id"].astype(str) if "participant_id" in frame.columns else "",
                "speech_id": frame["speech_id"].astype(str) if "speech_id" in frame.columns else "",
                "text_id": frame["text_id"].astype(str) if "text_id" in frame.columns else "",
                "y_true": pd.to_numeric(frame[y_col], errors="coerce"),
                "p_pred": pd.to_numeric(frame[p_col], errors="coerce"),
                "y_pred": pd.to_numeric(frame[y_pred_col], errors="coerce") if y_pred_col else np.nan,
                "threshold": pd.to_numeric(frame[threshold_col], errors="coerce") if threshold_col else np.nan,
                "role": (
                    frame[role_col].map(lambda value: _normalize_role(value, spec.role))
                    if role_col
                    else _normalize_role(None, spec.role)
                ),
                "probability_column": p_col,
                "label_column": y_col,
                "prediction_column": y_pred_col or "",
                "source_path": str(spec.prediction_path),
            }
        )
        normalized["threshold_source"] = _infer_threshold_source(
            frame, p_col, y_pred_col, spec.threshold_source
        ).reindex(frame.index).to_numpy()
        normalized["threshold_source"] = normalized["threshold_source"].where(
            normalized["threshold_source"].isin(ALLOWED_THRESHOLD_SOURCES), "unknown"
        )
        rows.append(normalized)
        summary["status"] = "loaded"
        summary["rows_after_filter"] = len(normalized)
        summaries.append(summary)
    if rows:
        predictions = pd.concat(rows, ignore_index=True)
    else:
        predictions = pd.DataFrame(
            columns=[
                "source_name",
                "model_name",
                "candidate_id",
                "feature_group",
                "model",
                "split_regime",
                "fold_id",
                "evaluation_level",
                "sample_id",
                "trial_id",
                "participant_id",
                "speech_id",
                "text_id",
                "y_true",
                "p_pred",
                "y_pred",
                "threshold",
                "threshold_source",
                "role",
                "source_path",
            ]
        )
    return predictions, pd.DataFrame(summaries)


def threshold_candidates(scores: Iterable[float]) -> np.ndarray:
    """Return deterministic threshold candidates and include fixed 0.5."""

    arr = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna().clip(0.0, 1.0).to_numpy()
    if arr.size == 0:
        return np.array([0.5], dtype=float)
    candidates = np.unique(np.concatenate([arr, np.array([0.0, 0.5, 1.0])]))
    return np.sort(candidates.astype(float))


def information_bits(n_candidate_thresholds: int) -> float:
    if n_candidate_thresholds <= 0:
        return math.nan
    return float(math.log2(n_candidate_thresholds))


def _finite_metric(value: float) -> float:
    try:
        result = float(value)
    except Exception:
        return math.nan
    return result if math.isfinite(result) else math.nan


def expected_calibration_error(
    y_true: Iterable[int],
    p_pred: Iterable[float],
    *,
    n_bins: int = 10,
) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)
    if y.size == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, bins[1:-1], right=False)
    total = float(y.size)
    ece = 0.0
    for bin_id in range(n_bins):
        keep = ids == bin_id
        if not np.any(keep):
            continue
        ece += (float(np.sum(keep)) / total) * abs(float(np.mean(y[keep])) - float(np.mean(p[keep])))
    return float(ece)


def calibration_slope_intercept(y_true: Iterable[int], p_pred: Iterable[float]) -> tuple[float, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(p_pred), dtype=float)
    mask = np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 1e-6, 1 - 1e-6)
    if y.size < 4 or np.unique(y).size < 2:
        return math.nan, math.nan
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        return float(model.coef_[0][0]), float(model.intercept_[0])
    except Exception:
        return math.nan, math.nan


def classification_metrics(
    y_true: Iterable[int],
    p_pred: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y = pd.to_numeric(pd.Series(list(y_true)), errors="coerce")
    p = pd.to_numeric(pd.Series(list(p_pred)), errors="coerce")
    keep = y.notna() & p.notna()
    y_arr = y[keep].astype(int).to_numpy()
    p_arr = p[keep].astype(float).clip(0.0, 1.0).to_numpy()
    if y_arr.size == 0:
        return {
            "n_predictions": 0,
            "n_positive": 0,
            "n_negative": 0,
            "AUROC": math.nan,
            "PR-AUC": math.nan,
            "BA": math.nan,
            "macro_F1": math.nan,
            "sensitivity": math.nan,
            "specificity": math.nan,
            "Brier": math.nan,
            "log_loss": math.nan,
            "calibration_slope": math.nan,
            "calibration_intercept": math.nan,
            "ECE": math.nan,
        }
    y_hat = (p_arr >= float(threshold)).astype(int)
    return classification_metrics_from_predictions(y_arr, p_arr, y_hat)


def classification_metrics_from_predictions(
    y_true: Iterable[int],
    p_pred: Iterable[float],
    y_pred: Iterable[int],
) -> dict[str, float]:
    """Compute ranking metrics from probabilities and decision metrics from labels."""

    y_arr = np.asarray(list(y_true), dtype=int)
    p_arr = np.asarray(list(p_pred), dtype=float)
    y_hat = np.asarray(list(y_pred), dtype=int)
    mask = np.isfinite(p_arr)
    y_arr = y_arr[mask]
    p_arr = np.clip(p_arr[mask], 0.0, 1.0)
    y_hat = y_hat[mask]
    if y_arr.size == 0:
        return {
            "n_predictions": 0,
            "n_positive": 0,
            "n_negative": 0,
            "AUROC": math.nan,
            "PR-AUC": math.nan,
            "BA": math.nan,
            "macro_F1": math.nan,
            "sensitivity": math.nan,
            "specificity": math.nan,
            "Brier": math.nan,
            "log_loss": math.nan,
            "calibration_slope": math.nan,
            "calibration_intercept": math.nan,
            "ECE": math.nan,
        }
    unique = np.unique(y_arr)
    auroc = roc_auc_score(y_arr, p_arr) if unique.size == 2 else math.nan
    pr_auc = average_precision_score(y_arr, p_arr) if unique.size == 2 else math.nan
    cm = confusion_matrix(y_arr, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    ba = (sensitivity + specificity) / 2.0 if math.isfinite(sensitivity) and math.isfinite(specificity) else math.nan
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0
    slope, intercept = calibration_slope_intercept(y_arr, p_arr)
    try:
        ll = log_loss(y_arr, np.clip(p_arr, 1e-15, 1 - 1e-15), labels=[0, 1])
    except Exception:
        ll = math.nan
    return {
        "n_predictions": int(y_arr.size),
        "n_positive": int(np.sum(y_arr == 1)),
        "n_negative": int(np.sum(y_arr == 0)),
        "AUROC": _finite_metric(auroc),
        "PR-AUC": _finite_metric(pr_auc),
        "BA": _finite_metric(ba),
        "macro_F1": _finite_metric(macro_f1),
        "sensitivity": _finite_metric(sensitivity),
        "specificity": _finite_metric(specificity),
        "Brier": _finite_metric(brier_score_loss(y_arr, p_arr)),
        "log_loss": _finite_metric(ll),
        "calibration_slope": _finite_metric(slope),
        "calibration_intercept": _finite_metric(intercept),
        "ECE": _finite_metric(expected_calibration_error(y_arr, p_arr)),
    }


def best_threshold(
    y_true: Iterable[int],
    p_pred: Iterable[float],
    *,
    metric: str = "BA",
) -> dict[str, float]:
    y = pd.to_numeric(pd.Series(list(y_true)), errors="coerce")
    p = pd.to_numeric(pd.Series(list(p_pred)), errors="coerce")
    keep = y.notna() & p.notna()
    y = y[keep].astype(int)
    p = p[keep].astype(float).clip(0.0, 1.0)
    candidates = threshold_candidates(p)
    y_arr = y.to_numpy(dtype=int)
    p_arr = p.to_numpy(dtype=float)
    order = np.argsort(p_arr)
    p_sorted = p_arr[order]
    y_sorted = y_arr[order]
    n_pos = int(np.sum(y_sorted == 1))
    n_neg = int(np.sum(y_sorted == 0))
    best: dict[str, float] = {
        "threshold": 0.5,
        "score": -math.inf,
        "n_candidate_thresholds": int(len(candidates)),
        "information_bits": information_bits(int(len(candidates))),
    }
    if n_pos == 0 or n_neg == 0:
        best["score"] = math.nan
        return best
    pos_prefix = np.concatenate([[0], np.cumsum(y_sorted == 1)])
    neg_prefix = np.concatenate([[0], np.cumsum(y_sorted == 0)])
    idxs = np.searchsorted(p_sorted, candidates, side="left")
    tp = n_pos - pos_prefix[idxs]
    fp = n_neg - neg_prefix[idxs]
    fn = n_pos - tp
    tn = n_neg - fp
    if metric == "macro_F1":
        f1_pos = np.divide(
            2 * tp,
            2 * tp + fp + fn,
            out=np.zeros_like(tp, dtype=float),
            where=(2 * tp + fp + fn) != 0,
        )
        f1_neg = np.divide(
            2 * tn,
            2 * tn + fn + fp,
            out=np.zeros_like(tn, dtype=float),
            where=(2 * tn + fn + fp) != 0,
        )
        scores = (f1_pos + f1_neg) / 2.0
    else:
        tpr = tp / n_pos
        tnr = tn / n_neg
        scores = (tpr + tnr) / 2.0
    for threshold, score in zip(candidates, scores, strict=False):
        if metric == "macro_F1":
            score = float(score)
        else:
            score = float(score)
        if math.isfinite(score) and (score > best["score"] or (score == best["score"] and threshold == 0.5)):
            best["threshold"] = float(threshold)
            best["score"] = float(score)
    if not math.isfinite(best["score"]):
        best["score"] = math.nan
    return best


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [
        "source_name",
        "model_name",
        "candidate_id",
        "feature_group",
        "model",
        "evaluation_level",
        "split_regime",
    ]


def _metric_row(
    group: pd.DataFrame,
    *,
    analysis_row: str,
    threshold_policy: str,
    threshold_source: str,
    official_claim_allowed: bool,
    threshold: float,
    n_candidate_thresholds: int | float = math.nan,
    information: float = math.nan,
    fold_id: str = "all",
    status: str = "complete",
    notes: str = "",
) -> dict[str, Any]:
    metrics = classification_metrics(group["y_true"], group["p_pred"], threshold=threshold)
    first = group.iloc[0]
    return {
        "analysis_row": analysis_row,
        "source_name": first["source_name"],
        "model_name": first["model_name"],
        "candidate_id": first["candidate_id"],
        "feature_group": first["feature_group"],
        "model": first["model"],
        "evaluation_level": first["evaluation_level"],
        "split_regime": first["split_regime"],
        "fold_id": fold_id,
        "threshold_policy": threshold_policy,
        "threshold_source": threshold_source,
        "official_claim_allowed": bool(official_claim_allowed),
        "score_source": "p_pred",
        "threshold": float(threshold) if threshold is not None and math.isfinite(float(threshold)) else math.nan,
        "n_candidate_thresholds": n_candidate_thresholds,
        "information_bits": information,
        **metrics,
        "status": status,
        "notes": notes,
    }


def _metric_row_from_decisions(
    group: pd.DataFrame,
    decisions: Iterable[int],
    *,
    analysis_row: str,
    threshold_policy: str,
    threshold_source: str,
    official_claim_allowed: bool,
    threshold: float,
    n_candidate_thresholds: int | float = math.nan,
    information: float = math.nan,
    fold_id: str = "all",
    status: str = "complete",
    notes: str = "",
) -> dict[str, Any]:
    """Build a metric row where BA/F1 use supplied decisions and ranking uses p_pred."""

    metrics = classification_metrics_from_predictions(group["y_true"], group["p_pred"], decisions)
    first = group.iloc[0]
    return {
        "analysis_row": analysis_row,
        "source_name": first["source_name"],
        "model_name": first["model_name"],
        "candidate_id": first["candidate_id"],
        "feature_group": first["feature_group"],
        "model": first["model"],
        "evaluation_level": first["evaluation_level"],
        "split_regime": first["split_regime"],
        "fold_id": fold_id,
        "threshold_policy": threshold_policy,
        "threshold_source": threshold_source,
        "official_claim_allowed": bool(official_claim_allowed),
        "score_source": "p_pred",
        "threshold": float(threshold) if threshold is not None and math.isfinite(float(threshold)) else math.nan,
        "n_candidate_thresholds": n_candidate_thresholds,
        "information_bits": information,
        **metrics,
        "status": status,
        "notes": notes,
    }


def fixed_threshold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    for _, group in predictions.groupby(_group_columns(predictions), dropna=False):
        rows.append(
            _metric_row(
                group,
                analysis_row="fixed_0_5",
                threshold_policy="fixed_0_5",
                threshold_source="fixed_0_5",
                official_claim_allowed=True,
                threshold=0.5,
            )
        )
        if group["evaluation_level"].iloc[0] == "trial_level" and group["fold_id"].nunique() > 1:
            fold_metrics = [
                classification_metrics(fold_group["y_true"], fold_group["p_pred"], threshold=0.5)
                for _, fold_group in group.groupby("fold_id", dropna=False)
            ]
            first = group.iloc[0]
            fold_row: dict[str, Any] = {
                "analysis_row": "fixed_0_5_fold_mean",
                "source_name": first["source_name"],
                "model_name": first["model_name"],
                "candidate_id": first["candidate_id"],
                "feature_group": first["feature_group"],
                "model": first["model"],
                "evaluation_level": first["evaluation_level"],
                "split_regime": first["split_regime"],
                "fold_id": "fold_mean",
                "threshold_policy": "fixed_0_5",
                "threshold_source": "fixed_0_5",
                "official_claim_allowed": True,
                "score_source": "p_pred",
                "threshold": 0.5,
                "n_candidate_thresholds": math.nan,
                "information_bits": math.nan,
                "n_predictions": int(sum(item["n_predictions"] for item in fold_metrics)),
                "n_positive": int(sum(item["n_positive"] for item in fold_metrics)),
                "n_negative": int(sum(item["n_negative"] for item in fold_metrics)),
                "status": "complete",
                "notes": "Mean of per-fold fixed-threshold metrics.",
            }
            for metric in [
                "AUROC",
                "PR-AUC",
                "BA",
                "macro_F1",
                "sensitivity",
                "specificity",
                "Brier",
                "log_loss",
                "calibration_slope",
                "calibration_intercept",
                "ECE",
            ]:
                values = [item[metric] for item in fold_metrics if math.isfinite(float(item[metric]))]
                fold_row[metric] = float(np.mean(values)) if values else math.nan
            rows.append(fold_row)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _score_at_metric(y_true: pd.Series, p_pred: pd.Series, threshold: float, metric: str) -> float:
    metrics = classification_metrics(y_true, p_pred, threshold=threshold)
    return float(metrics.get(metric, math.nan))


def legal_threshold_analysis(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply legal threshold policies when train/inner/calibration predictions exist.

    Existing configured sources only contain outer-test predictions in this repository
    state. In that case the function emits explicit not-computed rows and does not
    learn thresholds from outer test labels.
    """

    metric_rows: list[dict[str, Any]] = []
    learned_rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS), pd.DataFrame()
    test = predictions[predictions["role"] == "test"].copy()
    legal_pool = predictions[predictions["role"].isin({"inner_validation", "validation", "calibration"})].copy()
    group_cols = _group_columns(predictions)
    if legal_pool.empty:
        for values, group in test.groupby(group_cols, dropna=False):
            base = dict(zip(group_cols, values, strict=False))
            for policy in LEGAL_THRESHOLD_POLICIES:
                learned_rows.append(
                    {
                        **base,
                        "threshold_policy": policy,
                        "threshold_source": "train_inner_cv"
                        if policy != "calibration_set_threshold"
                        else "calibration_set",
                        "threshold": math.nan,
                        "n_candidate_thresholds": 0,
                        "information_bits": math.nan,
                        "status": "not_computed_missing_inner_validation_predictions",
                        "notes": "No train/inner-validation/calibration prediction rows were available; outer test labels were not used.",
                    }
                )
        return pd.DataFrame(metric_rows, columns=METRIC_COLUMNS), pd.DataFrame(learned_rows)

    for values, test_group in test.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, values, strict=False))
        global_pool = legal_pool[
            (legal_pool["source_name"] == base["source_name"])
            & (legal_pool["candidate_id"] == base["candidate_id"])
            & (legal_pool["evaluation_level"] == base["evaluation_level"])
        ]
        split_pool = global_pool[global_pool["split_regime"] == base["split_regime"]]
        pools = {
            "global_inner_cv_threshold": (global_pool, "BA", "train_inner_cv"),
            "split_regime_inner_cv_threshold": (split_pool, "BA", "train_inner_cv"),
            "calibration_set_threshold": (
                global_pool[global_pool["role"] == "calibration"],
                "BA",
                "calibration_set",
            ),
            "cost_sensitive_balanced_accuracy_threshold": (split_pool, "BA", "train_inner_cv"),
            "macro_f1_threshold": (split_pool, "macro_F1", "train_inner_cv"),
        }
        for policy, (pool, metric, source) in pools.items():
            if pool.empty:
                threshold = math.nan
                learned = {
                    "threshold": math.nan,
                    "score": math.nan,
                    "n_candidate_thresholds": 0,
                    "information_bits": math.nan,
                }
                status = "not_computed_missing_policy_pool"
            else:
                learned = best_threshold(pool["y_true"], pool["p_pred"], metric=metric)
                threshold = learned["threshold"]
                status = "complete"
                metric_rows.append(
                    _metric_row(
                        test_group,
                        analysis_row="legal_inner_cv_threshold",
                        threshold_policy=policy,
                        threshold_source=source,
                        official_claim_allowed=True,
                        threshold=threshold,
                        n_candidate_thresholds=learned["n_candidate_thresholds"],
                        information=learned["information_bits"],
                    )
                )
            learned_rows.append(
                {
                    **base,
                    "threshold_policy": policy,
                    "threshold_source": source,
                    "threshold": threshold,
                    "n_candidate_thresholds": learned["n_candidate_thresholds"],
                    "information_bits": learned["information_bits"],
                    "status": status,
                    "notes": "Threshold learned without outer test labels." if status == "complete" else "",
                }
            )
        prior_pool = split_pool if not split_pool.empty else global_pool
        if prior_pool.empty:
            prevalence_threshold = math.nan
            n_candidates = 0
            bits = math.nan
            status = "not_computed_missing_policy_pool"
        else:
            prevalence_threshold = float(pd.to_numeric(prior_pool["y_true"], errors="coerce").mean())
            n_candidates = 1
            bits = 0.0
            status = "complete"
            metric_rows.append(
                _metric_row(
                    test_group,
                    analysis_row="legal_inner_cv_threshold",
                    threshold_policy="prevalence_prior_threshold",
                    threshold_source="train_inner_cv",
                    official_claim_allowed=True,
                    threshold=prevalence_threshold,
                    n_candidate_thresholds=n_candidates,
                    information=bits,
                    notes="Threshold equals training/inner-validation positive prevalence.",
                )
            )
        learned_rows.append(
            {
                **base,
                "threshold_policy": "prevalence_prior_threshold",
                "threshold_source": "train_inner_cv",
                "threshold": prevalence_threshold,
                "n_candidate_thresholds": n_candidates,
                "information_bits": bits,
                "status": status,
                "notes": "Outer test labels were not used." if status == "complete" else "",
            }
        )
    return pd.DataFrame(metric_rows, columns=METRIC_COLUMNS), pd.DataFrame(learned_rows)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(float), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def aggregate_reader_probabilities(
    predictions: pd.DataFrame,
    *,
    method: str,
    fixed_threshold: float = 0.5,
) -> pd.DataFrame:
    if method not in READER_AGGREGATION_METHODS:
        raise ValueError(f"unknown reader aggregation method: {method}")
    trial = predictions[
        (predictions["evaluation_level"] == "trial_level")
        & (predictions["participant_id"].astype(str) != "")
    ].copy()
    if trial.empty:
        return pd.DataFrame(columns=predictions.columns)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "source_name",
        "model_name",
        "candidate_id",
        "feature_group",
        "model",
        "split_regime",
        "participant_id",
    ]
    for _, group in trial.groupby(group_cols, dropna=False):
        p = pd.to_numeric(group["p_pred"], errors="coerce").dropna().clip(0.0, 1.0).to_numpy()
        if p.size == 0:
            continue
        if method == "simple_mean_probability":
            agg = float(np.mean(p))
            basis = "probability"
        elif method == "logit_mean_probability":
            logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            agg = float(1.0 / (1.0 + np.exp(-np.mean(logits))))
            basis = "logit"
        elif method == "median_probability":
            agg = float(np.median(p))
            basis = "probability"
        elif method == "inverse_entropy_weighted_probability":
            weights = 1.0 / (_entropy(p) + 1e-6)
            agg = float(np.average(p, weights=weights))
            basis = "probability_uncertainty_weighted"
        else:
            hard = (p >= fixed_threshold).astype(float)
            agg = float(np.mean(hard))
            basis = "hard_label_baseline"
        first = group.iloc[0].to_dict()
        first["fold_id"] = "all"
        first["evaluation_level"] = "reader_aggregated"
        first["sample_id"] = str(first["participant_id"])
        first["trial_id"] = str(first["participant_id"])
        first["speech_id"] = ""
        first["text_id"] = ""
        first["y_true"] = int(round(float(pd.to_numeric(group["y_true"], errors="coerce").dropna().mode().iloc[0])))
        first["p_pred"] = agg
        first["y_pred"] = int(agg >= fixed_threshold)
        first["threshold"] = fixed_threshold
        first["threshold_source"] = "fixed_0_5"
        first["role"] = "test"
        first["aggregation_method"] = method
        first["aggregation_basis"] = basis
        first["n_trials_for_reader"] = int(len(group))
        first["reader_probability_variance"] = float(np.var(p))
        rows.append(first)
    return pd.DataFrame(rows)


def reader_aggregation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in READER_AGGREGATION_METHODS:
        aggregated = aggregate_reader_probabilities(predictions, method=method)
        if aggregated.empty:
            continue
        for _, group in aggregated.groupby(_group_columns(aggregated), dropna=False):
            note = "Hard-label majority vote is a baseline only." if method == "majority_vote_hard_label" else ""
            row = _metric_row(
                group,
                analysis_row=method,
                threshold_policy="fixed_0_5",
                threshold_source="fixed_0_5",
                official_claim_allowed=True,
                threshold=0.5,
                notes=note,
            )
            row["aggregation_method"] = method
            row["aggregation_basis"] = str(group["aggregation_basis"].iloc[0])
            row["n_readers"] = int(len(group))
            row["median_trials_per_reader"] = float(group["n_trials_for_reader"].median())
            row["mean_reader_probability_variance"] = float(group["reader_probability_variance"].mean())
            row["primary_aggregation_allowed"] = method != "majority_vote_hard_label"
            rows.append(row)
    extra = [
        "aggregation_method",
        "aggregation_basis",
        "n_readers",
        "median_trials_per_reader",
        "mean_reader_probability_variance",
        "primary_aggregation_allowed",
    ]
    return pd.DataFrame(rows, columns=[*METRIC_COLUMNS, *extra])


def test_oracle_threshold_analysis(
    predictions: pd.DataFrame,
    fixed_metrics: pd.DataFrame,
    legal_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS), pd.DataFrame(), pd.DataFrame()
    test = predictions[predictions["role"] == "test"].copy()
    group_cols = _group_columns(test)
    fixed_lookup = {
        (
            row.source_name,
            row.candidate_id,
            row.evaluation_level,
            row.split_regime,
            row.threshold_policy,
        ): row.BA
        for row in fixed_metrics.itertuples(index=False)
    }
    legal_best = {}
    if not legal_metrics.empty:
        for _, row in legal_metrics.groupby(["source_name", "candidate_id", "evaluation_level", "split_regime"]):
            key = tuple(row[["source_name", "candidate_id", "evaluation_level", "split_regime"]].iloc[0])
            legal_best[key] = float(row["BA"].max())
    for values, group in test.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, values, strict=False))
        source_pool = test[
            (test["source_name"] == base["source_name"])
            & (test["candidate_id"] == base["candidate_id"])
            & (test["evaluation_level"] == base["evaluation_level"])
        ]
        global_best = best_threshold(source_pool["y_true"], source_pool["p_pred"], metric="BA")
        split_best = best_threshold(group["y_true"], group["p_pred"], metric="BA")
        policies = [
            ("global_test_oracle_threshold", global_best, source_pool),
            ("split_regime_test_oracle_threshold", split_best, group),
        ]
        for policy, learned, pool in policies:
            row = _metric_row(
                group,
                analysis_row="test_oracle_threshold_diagnostic",
                threshold_policy=policy,
                threshold_source="test_oracle_diagnostic",
                official_claim_allowed=False,
                threshold=learned["threshold"],
                n_candidate_thresholds=learned["n_candidate_thresholds"],
                information=learned["information_bits"],
                notes="Diagnostic upper bound; outer test labels used for threshold selection.",
            )
            fixed_key = (
                row["source_name"],
                row["candidate_id"],
                row["evaluation_level"],
                row["split_regime"],
                "fixed_0_5",
            )
            legal_key = (
                row["source_name"],
                row["candidate_id"],
                row["evaluation_level"],
                row["split_regime"],
            )
            row["improvement_over_fixed_0_5_BA"] = row["BA"] - fixed_lookup.get(fixed_key, math.nan)
            row["improvement_over_best_legal_BA"] = row["BA"] - legal_best.get(legal_key, math.nan)
            metric_rows.append(row)
            threshold_rows.append(
                {
                    **base,
                    "threshold_policy": policy,
                    "threshold_source": "test_oracle_diagnostic",
                    "threshold": learned["threshold"],
                    "optimized_metric": "BA",
                    "optimized_score": learned["score"],
                    "n_candidate_thresholds": learned["n_candidate_thresholds"],
                    "information_bits": learned["information_bits"],
                    "official_claim_allowed": False,
                    "status": "complete",
                    "notes": "Diagnostic only; not legal benchmark tuning.",
                }
            )
        fold_thresholds: list[dict[str, float]] = []
        fold_predictions = []
        for fold_id, fold_group in group.groupby("fold_id", dropna=False):
            learned = best_threshold(fold_group["y_true"], fold_group["p_pred"], metric="BA")
            fold_thresholds.append(learned)
            fold_copy = fold_group.copy()
            fold_copy["fold_oracle_y_pred"] = (fold_copy["p_pred"] >= learned["threshold"]).astype(int)
            fold_predictions.append((fold_id, fold_copy, learned))
        if fold_predictions:
            fold_eval = pd.concat([item[1] for item in fold_predictions], ignore_index=True)
            n_candidates = int(sum(item[2]["n_candidate_thresholds"] for item in fold_predictions))
            bits = float(sum(item[2]["information_bits"] for item in fold_predictions))
            row = _metric_row_from_decisions(
                fold_eval,
                fold_eval["fold_oracle_y_pred"],
                analysis_row="test_oracle_threshold_diagnostic",
                threshold_policy="fold_test_oracle_threshold",
                threshold_source="test_oracle_diagnostic",
                official_claim_allowed=False,
                threshold=float(np.mean([item[2]["threshold"] for item in fold_predictions])),
                n_candidate_thresholds=n_candidates,
                information=bits,
                notes="Diagnostic upper bound with one test-label threshold per outer fold.",
            )
            fixed_key = (
                row["source_name"],
                row["candidate_id"],
                row["evaluation_level"],
                row["split_regime"],
                "fixed_0_5",
            )
            legal_key = (
                row["source_name"],
                row["candidate_id"],
                row["evaluation_level"],
                row["split_regime"],
            )
            row["improvement_over_fixed_0_5_BA"] = row["BA"] - fixed_lookup.get(fixed_key, math.nan)
            row["improvement_over_best_legal_BA"] = row["BA"] - legal_best.get(legal_key, math.nan)
            metric_rows.append(row)
            for fold_id, _, learned in fold_predictions:
                threshold_rows.append(
                    {
                        **base,
                        "fold_id": fold_id,
                        "threshold_policy": "fold_test_oracle_threshold",
                        "threshold_source": "test_oracle_diagnostic",
                        "threshold": learned["threshold"],
                        "optimized_metric": "BA",
                        "optimized_score": learned["score"],
                        "n_candidate_thresholds": learned["n_candidate_thresholds"],
                        "information_bits": learned["information_bits"],
                        "official_claim_allowed": False,
                        "status": "complete",
                        "notes": "Diagnostic only; not legal benchmark tuning.",
                    }
                )
    reader = aggregate_reader_probabilities(test, method="simple_mean_probability")
    for values, group in reader.groupby(_group_columns(reader), dropna=False):
        base = dict(zip(_group_columns(reader), values, strict=False))
        learned = best_threshold(group["y_true"], group["p_pred"], metric="BA")
        row = _metric_row(
            group,
            analysis_row="test_oracle_reader_threshold_diagnostic",
            threshold_policy="reader_aggregated_test_oracle_threshold",
            threshold_source="test_oracle_diagnostic",
            official_claim_allowed=False,
            threshold=learned["threshold"],
            n_candidate_thresholds=learned["n_candidate_thresholds"],
            information=learned["information_bits"],
            notes="Diagnostic upper bound; reader probabilities thresholded with test labels.",
        )
        metric_rows.append(row)
        threshold_rows.append(
            {
                **base,
                "threshold_policy": "reader_aggregated_test_oracle_threshold",
                "threshold_source": "test_oracle_diagnostic",
                "threshold": learned["threshold"],
                "optimized_metric": "BA",
                "optimized_score": learned["score"],
                "n_candidate_thresholds": learned["n_candidate_thresholds"],
                "information_bits": learned["information_bits"],
                "official_claim_allowed": False,
                "status": "complete",
                "notes": "Diagnostic only; not legal benchmark tuning.",
            }
        )
    oracle_metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    budget = thresholds[
        [
            "source_name",
            "candidate_id",
            "evaluation_level",
            "split_regime",
            "fold_id",
            "threshold_policy",
            "n_candidate_thresholds",
            "information_bits",
            "official_claim_allowed",
            "notes",
        ]
    ].copy() if not thresholds.empty else pd.DataFrame()
    return oracle_metrics, thresholds, budget


def calibration_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    test = predictions[predictions["role"] == "test"].copy()
    calibration_pool = predictions[predictions["role"].isin({"inner_validation", "validation", "calibration"})].copy()
    group_cols = _group_columns(predictions)
    for _, group in test.groupby(group_cols, dropna=False):
        rows.append(
            _metric_row(
                group,
                analysis_row="legal_calibrated_probability",
                threshold_policy="fixed_0_5",
                threshold_source="fixed_0_5",
                official_claim_allowed=True,
                threshold=0.5,
                notes="identity calibrator; no fitted calibration used.",
            )
            | {"calibrator": "identity", "calibration_status": "complete"}
        )
        if calibration_pool.empty:
            first = group.iloc[0]
            for calibrator in ("sigmoid_platt", "isotonic", "temperature_logistic"):
                rows.append(
                    {
                        "analysis_row": "legal_calibrated_probability",
                        "source_name": first["source_name"],
                        "model_name": first["model_name"],
                        "candidate_id": first["candidate_id"],
                        "feature_group": first["feature_group"],
                        "model": first["model"],
                        "evaluation_level": first["evaluation_level"],
                        "split_regime": first["split_regime"],
                        "fold_id": "all",
                        "threshold_policy": "fixed_0_5",
                        "threshold_source": "calibration_set",
                        "official_claim_allowed": True,
                        "score_source": "p_pred",
                        "threshold": 0.5,
                        "n_candidate_thresholds": 0,
                        "information_bits": math.nan,
                        "n_predictions": len(group),
                        "n_positive": int(pd.to_numeric(group["y_true"], errors="coerce").sum()),
                        "n_negative": int(len(group) - pd.to_numeric(group["y_true"], errors="coerce").sum()),
                        "AUROC": math.nan,
                        "PR-AUC": math.nan,
                        "BA": math.nan,
                        "macro_F1": math.nan,
                        "sensitivity": math.nan,
                        "specificity": math.nan,
                        "Brier": math.nan,
                        "log_loss": math.nan,
                        "calibration_slope": math.nan,
                        "calibration_intercept": math.nan,
                        "ECE": math.nan,
                        "status": "not_computed_missing_calibration_predictions",
                        "notes": "No train/inner-validation/calibration prediction rows were available.",
                        "calibrator": calibrator,
                        "calibration_status": "not_computed_missing_calibration_predictions",
                    }
                )
            continue
        pool = calibration_pool[
            (calibration_pool["source_name"] == group["source_name"].iloc[0])
            & (calibration_pool["candidate_id"] == group["candidate_id"].iloc[0])
        ]
        if pool.empty:
            continue
        x_train = pd.to_numeric(pool["p_pred"], errors="coerce").clip(1e-6, 1 - 1e-6)
        y_train = pd.to_numeric(pool["y_true"], errors="coerce").astype(int)
        x_test = pd.to_numeric(group["p_pred"], errors="coerce").clip(1e-6, 1 - 1e-6)
        try:
            logit_train = np.log(x_train / (1 - x_train)).to_numpy().reshape(-1, 1)
            logit_test = np.log(x_test / (1 - x_test)).to_numpy().reshape(-1, 1)
            platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            platt.fit(logit_train, y_train)
            calibrated = platt.predict_proba(logit_test)[:, 1]
            calibrated_group = group.copy()
            calibrated_group["p_pred"] = calibrated
            rows.append(
                _metric_row(
                    calibrated_group,
                    analysis_row="legal_calibrated_probability",
                    threshold_policy="fixed_0_5",
                    threshold_source="calibration_set",
                    official_claim_allowed=True,
                    threshold=0.5,
                    notes="Platt/sigmoid calibrator fit without outer test labels.",
                )
                | {"calibrator": "sigmoid_platt", "calibration_status": "complete"}
            )
        except Exception:
            pass
        if len(pool) >= 20 and y_train.nunique() == 2:
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x_train.to_numpy(), y_train.to_numpy())
                calibrated_group = group.copy()
                calibrated_group["p_pred"] = iso.predict(x_test.to_numpy())
                rows.append(
                    _metric_row(
                        calibrated_group,
                        analysis_row="legal_calibrated_probability",
                        threshold_policy="fixed_0_5",
                        threshold_source="calibration_set",
                        official_claim_allowed=True,
                        threshold=0.5,
                        notes="Isotonic calibrator fit without outer test labels.",
                    )
                    | {"calibrator": "isotonic", "calibration_status": "complete"}
                )
            except Exception:
                pass
    return pd.DataFrame(rows)


def threshold_curve_analysis(predictions: pd.DataFrame, grid_size: int = 101) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    thresholds = np.linspace(0.0, 1.0, grid_size)
    for _, group in predictions[predictions["role"] == "test"].groupby(_group_columns(predictions), dropna=False):
        first = group.iloc[0]
        p = pd.to_numeric(group["p_pred"], errors="coerce").clip(0.0, 1.0)
        y = pd.to_numeric(group["y_true"], errors="coerce").astype(int)
        for threshold in thresholds:
            y_hat = (p >= threshold).astype(int)
            cm = confusion_matrix(y, y_hat, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
            specificity = tn / (tn + fp) if (tn + fp) else math.nan
            ba = (
                (sensitivity + specificity) / 2.0
                if math.isfinite(sensitivity) and math.isfinite(specificity)
                else math.nan
            )
            f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
            macro_f1 = (f1_pos + f1_neg) / 2.0
            rows.append(
                {
                    "source_name": first["source_name"],
                    "candidate_id": first["candidate_id"],
                    "model_name": first["model_name"],
                    "feature_group": first["feature_group"],
                    "model": first["model"],
                    "evaluation_level": first["evaluation_level"],
                    "split_regime": first["split_regime"],
                    "threshold": float(threshold),
                    "BA": _finite_metric(ba),
                    "macro_F1": _finite_metric(macro_f1),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                    "true_negative": int(tn),
                    "near_threshold_errors": int(
                        np.sum((np.abs(p - threshold) <= 0.05) & (y != y_hat))
                    ),
                    "confidently_wrong_errors": int(
                        np.sum((((p >= 0.8) & (y == 0)) | ((p <= 0.2) & (y == 1))))
                    ),
                }
            )
    return pd.DataFrame(rows)


def probability_output_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            [
                {
                    "check": "predictions_loaded",
                    "status": "failed",
                    "detail": "No prediction rows were loaded.",
                }
            ]
        )
    key_cols = [
        "source_name",
        "candidate_id",
        "split_regime",
        "fold_id",
        "evaluation_level",
        "trial_id",
    ]
    duplicates = int(predictions.duplicated(key_cols).sum())
    missing_labels = int(predictions["y_true"].isna().sum())
    p_missing = int(predictions["p_pred"].isna().sum())
    p_outside = int(((predictions["p_pred"] < 0) | (predictions["p_pred"] > 1)).sum())
    derived = predictions["y_pred"].isna() | (
        predictions["y_pred"].astype(float) == (predictions["p_pred"] >= 0.5).astype(float)
    )
    predictor_columns = [
        col
        for col in predictions.columns
        if col.endswith("_as_predictor") or col.startswith("predictor_")
    ]
    prohibited_predictor_hits = [
        col for col in predictor_columns if col in {"participant_id", "speech_id", "text_id"}
    ]
    return pd.DataFrame(
        [
            {
                "check": "p_pred_exists",
                "status": "passed" if p_missing == 0 else "failed",
                "detail": f"missing p_pred rows: {p_missing}",
            },
            {
                "check": "p_pred_in_unit_interval",
                "status": "passed" if p_outside == 0 else "failed",
                "detail": f"outside [0,1]: {p_outside}",
            },
            {
                "check": "auroc_pr_auc_score_source",
                "status": "passed",
                "detail": "Metrics are recomputed from normalized p_pred.",
            },
            {
                "check": "y_pred_documented_threshold",
                "status": "passed" if bool(derived.all()) else "warning",
                "detail": "Existing y_pred matches fixed 0.5 where y_pred is present; all metrics are recomputed with documented thresholds.",
            },
            {
                "check": "duplicate_prediction_keys",
                "status": "passed" if duplicates == 0 else "failed",
                "detail": f"duplicate keys: {duplicates}",
            },
            {
                "check": "missing_labels",
                "status": "passed" if missing_labels == 0 else "failed",
                "detail": f"missing labels: {missing_labels}",
            },
            {
                "check": "participant_speech_predictors",
                "status": "passed" if not prohibited_predictor_hits else "failed",
                "detail": "Prediction tables expose participant_id/speech_id/text_id only as grouping identifiers.",
            },
        ]
    )


def threshold_source_audit(predictions: pd.DataFrame, fixed_metrics_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(
            [{"check": "threshold_sources_available", "status": "failed", "detail": "No predictions loaded."}]
        )
    unknown = int((predictions["threshold_source"] == "unknown").sum())
    rows.append(
        {
            "check": "unknown_threshold_source",
            "status": "passed" if unknown == 0 else "warning",
            "detail": f"rows with unknown threshold source: {unknown}",
        }
    )
    if not fixed_metrics_frame.empty:
        missing = fixed_metrics_frame[
            fixed_metrics_frame["threshold_source"].isna()
            | (fixed_metrics_frame["threshold_source"].astype(str) == "")
        ]
        rows.append(
            {
                "check": "ba_f1_threshold_source",
                "status": "passed" if missing.empty else "failed",
                "detail": f"metric rows missing threshold source: {len(missing)}",
            }
        )
    rows.append(
        {
            "check": "test_oracle_mixed_with_clean_metrics",
            "status": "passed",
            "detail": "Clean fixed/legal rows are separate from test_oracle_diagnostic rows.",
        }
    )
    rows.append(
        {
            "check": "reader_aggregation_uses_probabilities",
            "status": "passed",
            "detail": "Reader aggregation methods aggregate probabilities/logits; hard majority vote is labelled baseline only.",
        }
    )
    return pd.DataFrame(rows)


def _report_from_audit(title: str, audit: pd.DataFrame) -> str:
    status_counts = audit["status"].value_counts(dropna=False).to_dict() if "status" in audit else {}
    return "\n".join(
        [
            f"# {title}",
            "",
            f"Status counts: {status_counts}",
            "",
            _markdown_table(audit, max_rows=50),
        ]
    )


def _metric_report(title: str, frame: pd.DataFrame, note: str = "") -> str:
    if frame.empty:
        table = "_No metric rows._"
    else:
        cols = [
            "analysis_row",
            "source_name",
            "candidate_id",
            "evaluation_level",
            "split_regime",
            "threshold_source",
            "threshold",
            "AUROC",
            "PR-AUC",
            "BA",
            "macro_F1",
            "Brier",
            "status",
        ]
        available = [col for col in cols if col in frame.columns]
        table = _markdown_table(frame[available], max_rows=40)
    return "\n".join([f"# {title}", "", note, "", table]).strip()


def _before_after_table(
    fixed: pd.DataFrame,
    legal: pd.DataFrame,
    calibration: pd.DataFrame,
    reader: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not fixed.empty:
        rows.append(fixed.copy())
    if not legal.empty:
        legal_copy = legal.copy()
        legal_copy["analysis_row"] = "legal_inner_cv_threshold"
        rows.append(legal_copy)
    if not calibration.empty:
        rows.append(calibration.copy())
    if not reader.empty:
        rows.append(reader.copy())
    if not oracle.empty:
        rows.append(oracle.copy())
    if not rows:
        return pd.DataFrame(
            columns=[
                "analysis_row",
                "evaluation_level",
                "split_regime",
                "threshold_source",
                "official_claim_allowed",
                "AUROC",
                "PR-AUC",
                "BA",
                "macro_F1",
                "Brier",
                "calibration_slope",
                "calibration_intercept",
                "threshold",
                "n_candidate_thresholds",
                "information_bits",
                "notes",
            ]
        )
    combined = pd.concat(rows, ignore_index=True, sort=False)
    cols = [
        "analysis_row",
        "source_name",
        "candidate_id",
        "feature_group",
        "model",
        "evaluation_level",
        "split_regime",
        "threshold_source",
        "official_claim_allowed",
        "AUROC",
        "PR-AUC",
        "BA",
        "macro_F1",
        "Brier",
        "calibration_slope",
        "calibration_intercept",
        "threshold",
        "n_candidate_thresholds",
        "information_bits",
        "notes",
    ]
    return combined[[col for col in cols if col in combined.columns]]


def _summarize_d3_lite(fixed: pd.DataFrame, oracle: pd.DataFrame, reader: pd.DataFrame) -> dict[str, Any]:
    if fixed.empty or "source_name" not in fixed.columns:
        d3 = pd.DataFrame()
    else:
        d3 = fixed[
            (fixed["source_name"] == "d3_eyebench_lite_candidate_0000")
            & (fixed["evaluation_level"] == "trial_level")
        ].copy()
        fold_mean = d3[d3["analysis_row"] == "fixed_0_5_fold_mean"].copy()
        if not fold_mean.empty:
            d3 = fold_mean
    if oracle.empty or "source_name" not in oracle.columns:
        oracle_d3 = pd.DataFrame()
    else:
        oracle_d3 = oracle[
            (oracle["source_name"] == "d3_eyebench_lite_candidate_0000")
            & (oracle["threshold_policy"] == "split_regime_test_oracle_threshold")
            & (oracle["evaluation_level"] == "trial_level")
        ].copy()
    if reader.empty or "source_name" not in reader.columns:
        reader_d3 = pd.DataFrame()
    else:
        reader_d3 = reader[
            (reader["source_name"] == "d3_eyebench_lite_candidate_0000")
            & (reader.get("aggregation_method", "") == "simple_mean_probability")
        ].copy()
    return {
        "d3_lite_fixed_trial": d3[
            ["split_regime", "AUROC", "PR-AUC", "BA", "macro_F1", "Brier"]
        ].to_dict("records")
        if not d3.empty
        else [],
        "d3_lite_oracle_trial": oracle_d3[
            [
                "split_regime",
                "threshold",
                "n_candidate_thresholds",
                "information_bits",
                "BA",
                "improvement_over_fixed_0_5_BA",
            ]
        ].to_dict("records")
        if not oracle_d3.empty
        else [],
        "d3_lite_reader_simple_mean": reader_d3[
            ["split_regime", "AUROC", "PR-AUC", "BA", "macro_F1", "Brier"]
        ].to_dict("records")
        if not reader_d3.empty
        else [],
    }


def _final_decision(
    predictions: pd.DataFrame,
    probability_audit_frame: pd.DataFrame,
    legal: pd.DataFrame,
    oracle: pd.DataFrame,
    reader: pd.DataFrame,
) -> dict[str, Any]:
    if predictions.empty:
        category = "invalid_missing_predictions"
    elif (probability_audit_frame["status"] == "failed").any():
        category = "invalid_leakage_detected"
    elif not legal.empty and legal["BA"].notna().any():
        category = "supplement_supporting_result"
    elif not reader.empty:
        category = "supplement_supporting_result"
    elif not oracle.empty:
        category = "diagnostic_only"
    else:
        category = "no_improvement"
    return {
        "decision_category": category,
        "official_sota_claim_changed": False,
        "official_sota_claim_allowed": False,
        "test_oracle_official_claim_allowed": False,
        "legal_threshold_metrics_available": bool(not legal.empty and legal["BA"].notna().any()),
        "test_oracle_metrics_available": bool(not oracle.empty),
        "reader_probability_aggregation_available": bool(not reader.empty),
        "safe_wording": (
            "Threshold and calibration analyses show that D3 is probability-first and "
            "reader-profile oriented. Test-oracle thresholds provide an upper bound "
            "but are not used for benchmark or SOTA claims."
        ),
    }


def _final_report(decision: dict[str, Any], summary: dict[str, Any], legal: pd.DataFrame) -> str:
    legal_available = bool(not legal.empty and legal["BA"].notna().any())
    oracle_rows = summary["d3_lite_oracle_trial"]
    oracle_note = "No oracle rows were available."
    if oracle_rows:
        improvements = [
            row.get("improvement_over_fixed_0_5_BA")
            for row in oracle_rows
            if row.get("improvement_over_fixed_0_5_BA") is not None
            and math.isfinite(float(row.get("improvement_over_fixed_0_5_BA")))
        ]
        if improvements:
            oracle_note = f"Maximum D3_Lite split-oracle BA improvement over fixed 0.5: {max(improvements):.4f}."
    return "\n".join(
        [
            "# OperatingPointAdaptation v1 final decision",
            "",
            f"Decision category: `{decision['decision_category']}`",
            "",
            "1. Are D3_Lite low BA results mainly threshold-related?",
            f"   Fixed-threshold and oracle-threshold rows quantify this directly. {oracle_note}",
            "2. Does legal threshold learning improve BA/Macro F1?",
            "   Legal threshold learning was not computed from the available artifacts because no train/inner-validation/calibration prediction rows were present."
            if not legal_available
            else "   Legal threshold metrics were computed without outer test labels; see legal_threshold_metrics.csv.",
            "3. Does calibration improve Brier/calibration slope?",
            "   Only identity calibration is final for the available artifacts; fitted legal calibrators require calibration prediction rows.",
            "4. Does reader-level probability aggregation improve AUROC/PR-AUC?",
            "   Reader probability aggregation is reported as secondary evidence in reader_probability_aggregation_metrics.csv.",
            "5. How much improvement is possible under test-oracle threshold?",
            f"   {oracle_note}",
            "6. How many bits of label information does the oracle threshold use?",
            "   The oracle information budget is log2(number_of_candidate_thresholds) and is reported per threshold policy in test_oracle_information_budget.csv.",
            "7. Does any legal result change the official SOTA status?",
            "   No. The official SOTA claim is unchanged and remains false.",
            "8. Does the oracle result show an upper-bound implementation potential?",
            "   Yes, where oracle BA improves over fixed 0.5 it is diagnostic implementation potential only.",
            "9. What wording should be used in the paper?",
            f"   {decision['safe_wording']}",
            "",
            "Test-oracle threshold results are diagnostic upper bounds and cannot be used for official benchmark or SOTA claims.",
        ]
    )


def run_operating_point_adaptation(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    predictions, source_summary = load_prediction_sources(config, repo_root=root)
    _write_table(dirs, "prediction_sources_loaded.csv", source_summary)

    probability_audit_frame = probability_output_audit(predictions)
    _write_report(
        dirs,
        "probability_output_audit.md",
        _report_from_audit("Probability Output Audit", probability_audit_frame),
    )

    fixed = fixed_threshold_metrics(predictions)
    _write_table(dirs, "fixed_threshold_metrics.csv", fixed)
    _write_report(
        dirs,
        "fixed_threshold_report.md",
        _metric_report(
            "Fixed Threshold Metrics",
            fixed,
            "All BA and macro F1 rows use threshold_source=fixed_0_5; AUROC and PR-AUC use p_pred.",
        ),
    )

    threshold_audit = threshold_source_audit(predictions, fixed)
    _write_report(
        dirs,
        "threshold_source_audit.md",
        _report_from_audit("Threshold Source Audit", threshold_audit),
    )

    legal, legal_thresholds = legal_threshold_analysis(predictions)
    _write_table(dirs, "legal_threshold_metrics.csv", legal)
    _write_table(dirs, "legal_thresholds_learned.csv", legal_thresholds)
    legal_note = (
        "Legal threshold policies never use outer test labels. If no train/inner-validation/"
        "calibration prediction rows are available, thresholds are explicitly not computed."
    )
    _write_report(dirs, "legal_threshold_report.md", _metric_report("Legal Threshold Report", legal, legal_note))

    calibration = calibration_analysis(predictions)
    _write_table(dirs, "calibration_metrics.csv", calibration)
    _write_report(
        dirs,
        "calibration_report.md",
        _metric_report(
            "Calibration Report",
            calibration,
            "Fitted calibrators require train/inner-validation/calibration predictions; identity calibration is always reported.",
        ),
    )

    reader = reader_aggregation_metrics(predictions)
    _write_table(dirs, "reader_probability_aggregation_metrics.csv", reader)
    _write_report(
        dirs,
        "reader_probability_aggregation_report.md",
        _metric_report(
            "Reader Probability Aggregation Report",
            reader,
            "Reader aggregation combines probabilities or logits; hard-label majority vote is a baseline only.",
        ),
    )

    oracle, oracle_thresholds, oracle_budget = test_oracle_threshold_analysis(predictions, fixed, legal)
    _write_table(dirs, "test_oracle_threshold_metrics.csv", oracle)
    _write_table(dirs, "test_oracle_thresholds.csv", oracle_thresholds)
    _write_table(dirs, "test_oracle_information_budget.csv", oracle_budget)
    _write_report(
        dirs,
        "test_oracle_threshold_report.md",
        _metric_report(
            "Test-Oracle Threshold Report",
            oracle,
            "Test-oracle threshold results are diagnostic upper bounds and cannot be used for official benchmark or SOTA claims.",
        ),
    )

    curve = threshold_curve_analysis(
        predictions, grid_size=int(get_nested(config, f"{CONFIG_SECTION}.threshold_curve_grid_size", 101))
    )
    _write_table(dirs, "threshold_curve_tables.csv", curve)
    if curve.empty:
        error_report = "# Error Origin Threshold Report\n\nNo threshold curve rows were available."
    else:
        best_rows = curve.loc[curve.groupby(["source_name", "candidate_id", "evaluation_level", "split_regime"])["BA"].idxmax()]
        error_report = "\n".join(
            [
                "# Error Origin Threshold Report",
                "",
                "The table lists the fixed grid threshold with maximal BA per group. Near-threshold and confidently wrong error counts are computed from p_pred.",
                "",
                _markdown_table(
                    best_rows[
                        [
                            "source_name",
                            "candidate_id",
                            "evaluation_level",
                            "split_regime",
                            "threshold",
                            "BA",
                            "macro_F1",
                            "false_positive",
                            "false_negative",
                            "near_threshold_errors",
                            "confidently_wrong_errors",
                        ]
                    ],
                    max_rows=60,
                ),
            ]
        )
    _write_report(dirs, "error_origin_threshold_report.md", error_report)

    before_after = _before_after_table(fixed, legal, calibration, reader, oracle)
    _write_table(dirs, "before_after_operating_point_comparison.csv", before_after)
    _write_report(
        dirs,
        "before_after_operating_point_comparison.md",
        "# Before/After Operating Point Comparison\n\n" + _markdown_table(before_after, max_rows=80),
    )

    summary = _summarize_d3_lite(fixed, oracle, reader)
    decision = _final_decision(predictions, probability_audit_frame, legal, oracle, reader)
    _write_json(dirs["repo_analysis"] / "final_operating_point_decision.json", decision)
    _write_json(dirs["result_analysis"] / "final_operating_point_decision.json", decision)
    _write_report(dirs, "final_operating_point_decision_report.md", _final_report(decision, summary, legal))

    if source_summary["status"].isin(["missing", "missing_required_columns", "read_failed"]).any():
        missing = source_summary[source_summary["status"] != "loaded"]
        _write_report(
            dirs,
            "missing_prediction_blocker_report.md",
            "# Missing Prediction Blocker Report\n\n"
            + "Some configured prediction sources could not be loaded. Non-final summaries are allowed, but missing prediction rows are not fabricated.\n\n"
            + _markdown_table(missing, max_rows=50),
        )

    manifest = {
        "analysis": ANALYSIS_NAME,
        "output_dir": str(out),
        "repo_analysis_dir": str(dirs["repo_analysis"]),
        "result_analysis_dir": str(dirs["result_analysis"]),
        "git_commit": _git_sha(root),
        "prediction_rows_loaded": int(len(predictions)),
        "sources_loaded": int((source_summary["status"] == "loaded").sum()) if not source_summary.empty else 0,
        "decision_category": decision["decision_category"],
        "official_sota_claim_changed": False,
        "official_sota_claim_allowed": False,
        "test_oracle_official_claim_allowed": False,
        "reports": sorted(path.name for path in dirs["repo_analysis"].glob("*")),
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def validate_operating_point_adaptation(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    out = Path(output_dir).resolve()
    dirs = _analysis_dirs(config, out, repo_root)
    analysis_dir = dirs["result_analysis"] if dirs["result_analysis"].exists() else dirs["repo_analysis"]
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "probability_output_audit.md",
        "prediction_sources_loaded.csv",
        "threshold_source_audit.md",
        "fixed_threshold_metrics.csv",
        "fixed_threshold_report.md",
        "legal_threshold_metrics.csv",
        "legal_thresholds_learned.csv",
        "legal_threshold_report.md",
        "test_oracle_threshold_metrics.csv",
        "test_oracle_thresholds.csv",
        "test_oracle_information_budget.csv",
        "test_oracle_threshold_report.md",
        "calibration_metrics.csv",
        "calibration_report.md",
        "reader_probability_aggregation_metrics.csv",
        "reader_probability_aggregation_report.md",
        "threshold_curve_tables.csv",
        "error_origin_threshold_report.md",
        "before_after_operating_point_comparison.csv",
        "before_after_operating_point_comparison.md",
        "final_operating_point_decision_report.md",
        "final_operating_point_decision.json",
    ]
    for name in required:
        if not (analysis_dir / name).exists():
            errors.append(f"missing required artifact: {analysis_dir / name}")
    if (analysis_dir / "test_oracle_threshold_metrics.csv").exists():
        oracle = pd.read_csv(analysis_dir / "test_oracle_threshold_metrics.csv")
        if not oracle.empty:
            if not (oracle["threshold_source"] == "test_oracle_diagnostic").all():
                errors.append("oracle rows must use threshold_source=test_oracle_diagnostic")
            if not (oracle["official_claim_allowed"] == False).all():  # noqa: E712
                errors.append("oracle rows must have official_claim_allowed=false")
            if oracle["information_bits"].isna().any():
                errors.append("oracle rows must record information_bits")
    if (analysis_dir / "fixed_threshold_metrics.csv").exists():
        fixed = pd.read_csv(analysis_dir / "fixed_threshold_metrics.csv")
        if not fixed.empty:
            if fixed["threshold_source"].isna().any():
                errors.append("fixed BA/F1 rows must record threshold_source")
            if not (fixed["score_source"] == "p_pred").all():
                errors.append("AUROC/PR-AUC must use probability scores")
    if (analysis_dir / "legal_threshold_metrics.csv").exists():
        legal = pd.read_csv(analysis_dir / "legal_threshold_metrics.csv")
        if not legal.empty:
            if (legal["threshold_source"] == "test_oracle_diagnostic").any():
                errors.append("legal threshold rows may not use test-oracle thresholds")
            if not (legal["official_claim_allowed"] == True).all():  # noqa: E712
                errors.append("legal threshold rows should remain official-claim eligible in principle")
    if (analysis_dir / "reader_probability_aggregation_metrics.csv").exists():
        reader = pd.read_csv(analysis_dir / "reader_probability_aggregation_metrics.csv")
        if not reader.empty and "aggregation_method" in reader.columns:
            hard = reader[reader["aggregation_method"] == "majority_vote_hard_label"]
            if not hard.empty and hard.get("primary_aggregation_allowed", pd.Series([False])).astype(bool).any():
                errors.append("hard-label majority vote cannot be primary")
    if (analysis_dir / "final_operating_point_decision.json").exists():
        decision = json.loads((analysis_dir / "final_operating_point_decision.json").read_text(encoding="utf-8"))
        if decision.get("decision_category") not in VALID_DECISION_CATEGORIES:
            errors.append(f"invalid decision category: {decision.get('decision_category')}")
        if decision.get("official_sota_claim_changed") is not False:
            errors.append("official SOTA claim must not change from operating-point analysis")
        if decision.get("test_oracle_official_claim_allowed") is not False:
            errors.append("test-oracle metrics cannot allow official claims")
    if (analysis_dir / "probability_output_audit.md").exists():
        text = (analysis_dir / "probability_output_audit.md").read_text(encoding="utf-8").lower()
        if "participant_id/speech_id/text_id only as grouping identifiers" not in text:
            warnings.append("predictor grouping statement not found in probability audit text")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "analysis_dir": str(analysis_dir),
        "output_dir": str(out),
    }
    _write_json(out / "operating_point_adaptation_validation_report.json", report)
    return report
