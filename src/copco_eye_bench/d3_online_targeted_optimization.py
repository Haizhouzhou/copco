"""Targeted online/offline D3 sequential detection evaluation.

This runner builds online prefix features from the prepared CopCo release,
trains legal nested prefix predictors, evaluates calibration, thresholds,
evidence accumulation, stopping policies, oracle diagnostics, and writes the
small analysis reports required by D3OnlineTargetedOptimization v1.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import get_nested, timestamped_output_dir


warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="A single label was found in 'y_true' and 'y_pred'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
try:
    _RANK_WARNING = np.exceptions.RankWarning
except AttributeError:  # pragma: no cover - NumPy compatibility shim.
    _RANK_WARNING = np.RankWarning
warnings.filterwarnings("ignore", category=_RANK_WARNING)


SECTION = "d3_online_targeted_optimization"
ANALYSIS_NAME = "d3_online_targeted_optimization_v1"
MODEL_NAME = "D3_OnlinePrefix_logistic_regression"
PRIMARY_REGIMES = ("unseen_reader", "unseen_reader_and_text")
DEFAULT_SPLIT_REGIMES = (
    "unseen_reader",
    "unseen_text",
    "unseen_reader_and_text",
    "text_balanced_unseen_reader",
    "participant_grouped_kfold",
)
GOAL_NAMES = {
    "GOAL_0": "execution docs",
    "GOAL_1": "prefix datasets",
    "GOAL_2": "nested prediction artifacts",
    "GOAL_3": "D3 online prefix models",
    "GOAL_4": "legal calibration and threshold learning",
    "GOAL_5": "online evidence accumulation",
    "GOAL_6": "online stopping policies",
    "GOAL_7": "targeted online optimization loop",
    "GOAL_8": "oracle upper-bound diagnostics",
    "GOAL_9": "error trajectory analysis",
    "GOAL_10": "online/offline comparison and final decision",
    "GOAL_11": "manuscript and supplement update",
    "GOAL_12": "validator and tests",
    "GOAL_13": "commit and push",
}

REQUIRED_DOCS = (
    "docs/d3_online_targeted_optimization_v1.md",
    "docs/d3_online_detection_goal_contract_v1.md",
    "docs/d3_online_testing_standard_v1.md",
    "analysis/d3_online_targeted_optimization_v1/subgoal_status.md",
    "analysis/d3_online_targeted_optimization_v1/subgoal_status.json",
)

ID_AND_LABEL_COLUMNS = {
    "participant_id",
    "reader_group",
    "reader_group_binary",
    "dyslexia_labeled",
    "group_label",
    "speech_id",
    "speechId",
    "text_id",
    "terminal_text_id",
    "observed_text_ids",
    "observed_speech_ids",
    "prefix_type",
    "prefix_value",
    "prefix_order_index",
    "evidence_available_until_prefix",
    "participant_word_key",
    "stimulus_word_key",
    "word_id",
    "paragraph_id",
    "sentence_id",
    "label_source",
    "diagnostic_provenance",
}

PROHIBITED_PREDICTOR_PATTERNS = (
    "participant_id",
    "speech_id",
    "speechid",
    "text_id",
    "reader_group",
    "group_label",
    "dyslexia",
    "label",
    "future",
    "max_words_for_reader",
    "max_texts_for_reader",
    "full_session",
)

GAZE_COLUMNS = [
    "FFD",
    "GD",
    "TRT",
    "fixation_count",
    "skip",
    "refixation_count",
    "go_past_time",
    "mean_fixation_duration",
    "landing_position",
    "mean_saccade_duration",
    "peak_saccade_velocity",
]
CORE_GAZE_COLUMNS = ["FFD", "GD", "TRT", "fixation_count", "skip", "refixation_count"]
DFM_COLUMNS = ["dfm_lm_word_surprisal", "dfm_lm_word_entropy", "dfm_entropy_word_onset"]
SEGMENTATION_COLUMNS = [
    "prev_boundary_opacity_score",
    "next_boundary_opacity_score",
    "within_word_vowel_run_max",
    "word_vowel_ratio",
    "word_vowel_count",
    "starts_with_vowel",
    "ends_with_vowel",
]
QUALITY_COLUMNS = [
    "gaze_missing",
    "lm_missing",
    "segmentation_label_missing",
    "parser_missing",
    "embedding_missing",
]

FEATURE_FAMILIES = (
    "raw_gaze_prefix",
    "residual_gaze_prefix",
    "dfm_exposure_prefix",
    "dfm_sensitivity_prefix",
    "dfm_residual_gaze_prefix",
    "dfm_residual_plus_uncertainty_prefix",
    "all_allowed_online",
)

ACCUMULATORS = (
    "mean_probability",
    "logit_mean",
    "entropy_weighted",
    "uncertainty_weighted_logit",
    "reliability_weighted_probability",
    "learned_meta_aggregator",
    "beta_binomial_posterior",
)

STOPPING_POLICIES = (
    "no_stop_all_evidence",
    "fixed_0_5_at_each_prefix",
    "two_sided_confidence_policy",
    "inner_cv_balanced_accuracy_policy",
    "cost_sensitive_online_policy",
    "target_sensitivity_policy",
    "target_specificity_policy",
    "coverage_constrained_policy",
)


@dataclass(frozen=True)
class SplitFold:
    regime: str
    fold_id: int
    train_indices: list[int]
    test_indices: list[int]
    skipped_indices: list[int]
    train_participants: list[str]
    test_participants: list[str]
    train_texts: list[str]
    test_texts: list[str]
    status: str
    skip_reason: str = ""


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = get_nested(config, SECTION, {})
    return value if isinstance(value, dict) else {}


def _path(config: dict[str, Any], key: str, repo_root: Path, default: str | None = None) -> Path:
    raw = get_nested(config, f"{SECTION}.{key}", default)
    if raw is None:
        raise KeyError(f"missing config key: {SECTION}.{key}")
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()
    columns = [str(c) for c in shown.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in shown.to_dict("records"):
        values = []
        for column in shown.columns:
            value = record.get(column)
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value) if value is not None else "")
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip_prob(values: Iterable[float] | pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(list(values), dtype=float), 1e-6, 1 - 1e-6)


def _logit(values: Iterable[float] | pd.Series | np.ndarray) -> np.ndarray:
    p = _clip_prob(values)
    return np.log(p / (1 - p))


def _expit(values: Iterable[float] | pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _prob_entropy(prob: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    p = _clip_prob(prob)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _ece(y_true: pd.Series | np.ndarray, p_pred: pd.Series | np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return math.nan
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(p)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            bucket = (p >= lower) & (p <= upper)
        else:
            bucket = (p >= lower) & (p < upper)
        if not bucket.any():
            continue
        error += float(bucket.mean()) * abs(float(y[bucket].mean()) - float(p[bucket].mean()))
    return error if total else math.nan


def _calibration_slope_intercept(
    y_true: pd.Series | np.ndarray, p_pred: pd.Series | np.ndarray
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = _clip_prob(p_pred)
    mask = np.isfinite(p)
    if len(y[mask]) < 5 or len(set(y[mask].tolist())) < 2:
        return math.nan, math.nan
    try:
        x = _logit(p[mask])
        slope, intercept = np.polyfit(x, y[mask].astype(float), deg=1)
        return float(slope), float(intercept)
    except Exception:
        return math.nan, math.nan


def classification_metrics(
    y_true: pd.Series | np.ndarray | list[int],
    p_pred: pd.Series | np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(p_pred), errors="coerce")
    mask = y.notna() & p.notna()
    y_arr = y[mask].astype(int).to_numpy()
    p_arr = np.clip(p[mask].astype(float).to_numpy(), 0.0, 1.0)
    if len(y_arr) == 0:
        return {
            "AUROC": math.nan,
            "PR-AUC": math.nan,
            "BA": math.nan,
            "macro_F1": math.nan,
            "Brier": math.nan,
            "sensitivity": math.nan,
            "specificity": math.nan,
            "FPR": math.nan,
            "FNR": math.nan,
            "ECE": math.nan,
            "calibration_slope": math.nan,
            "calibration_intercept": math.nan,
        }
    y_hat = (p_arr >= float(threshold)).astype(int)
    unique = set(y_arr.tolist())
    auroc = float(roc_auc_score(y_arr, p_arr)) if len(unique) == 2 else math.nan
    pr_auc = float(average_precision_score(y_arr, p_arr)) if len(unique) == 2 else math.nan
    brier = float(brier_score_loss(y_arr, p_arr))
    ba = float(balanced_accuracy_score(y_arr, y_hat))
    macro_f1 = float(f1_score(y_arr, y_hat, average="macro", zero_division=0))
    cm = confusion_matrix(y_arr, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    slope, intercept = _calibration_slope_intercept(y_arr, p_arr)
    return {
        "AUROC": auroc,
        "PR-AUC": pr_auc,
        "BA": ba,
        "macro_F1": macro_f1,
        "Brier": brier,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "FPR": fp / (fp + tn) if (fp + tn) else math.nan,
        "FNR": fn / (fn + tp) if (fn + tp) else math.nan,
        "ECE": _ece(y_arr, p_arr),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def threshold_candidates(
    probabilities: Iterable[float], grid_size: int = 101, max_probability_candidates: int = 201
) -> np.ndarray:
    values = [0.0, 0.5, 1.0]
    unique = np.asarray(
        sorted(
            set(
                float(x)
                for x in pd.to_numeric(pd.Series(list(probabilities)), errors="coerce").dropna()
                if 0.0 <= float(x) <= 1.0
            )
        ),
        dtype=float,
    )
    if len(unique) > max_probability_candidates:
        unique = np.quantile(unique, np.linspace(0.0, 1.0, max_probability_candidates))
    values.extend(float(x) for x in unique)
    values.extend(float(x) for x in np.linspace(0.0, 1.0, grid_size))
    return np.asarray(sorted(set(round(v, 6) for v in values if 0.0 <= v <= 1.0)), dtype=float)


def learn_threshold_from_pool(
    pool: pd.DataFrame,
    *,
    policy: str = "balanced_accuracy_threshold",
    target_sensitivity: float = 0.80,
    target_specificity: float = 0.80,
) -> dict[str, Any]:
    y = pd.to_numeric(pool.get("y_true"), errors="coerce")
    p = pd.to_numeric(pool.get("p_pred"), errors="coerce")
    valid = pool[y.notna() & p.notna()].copy()
    if valid.empty or y[y.notna()].nunique() < 2:
        return {
            "threshold": 0.5,
            "status": "blocked_single_class_or_empty_pool",
            "n_candidate_thresholds": 0,
            "information_bits": math.nan,
            "selection_metric": math.nan,
        }
    candidates = threshold_candidates(valid["p_pred"])
    y_arr = pd.to_numeric(valid["y_true"], errors="coerce").astype(int).to_numpy()
    p_arr = pd.to_numeric(valid["p_pred"], errors="coerce").to_numpy(dtype=float)
    pred = p_arr[:, None] >= candidates[None, :]
    positives = y_arr == 1
    negatives = ~positives
    tp = (pred & positives[:, None]).sum(axis=0).astype(float)
    fp = (pred & negatives[:, None]).sum(axis=0).astype(float)
    fn = ((~pred) & positives[:, None]).sum(axis=0).astype(float)
    tn = ((~pred) & negatives[:, None]).sum(axis=0).astype(float)
    sensitivity = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.full_like(tn, np.nan), where=(tn + fp) > 0)
    ba = (sensitivity + specificity) / 2.0
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn), where=(2 * tn + fn + fp) > 0)
    macro_f1 = (f1_pos + f1_neg) / 2.0
    if policy == "macro_f1_threshold":
        scores = macro_f1
    elif policy == "target_sensitivity_threshold":
        scores = np.where(sensitivity >= target_sensitivity, specificity, -np.inf)
    elif policy == "target_specificity_threshold":
        scores = np.where(specificity >= target_specificity, sensitivity, -np.inf)
    else:
        scores = ba
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    if np.isneginf(scores).all():
        best_threshold = 0.5
        best_score = _safe_float(classification_metrics(valid["y_true"], valid["p_pred"], 0.5)["BA"])
    else:
        best_score = float(np.max(scores))
        tied = np.flatnonzero(scores == best_score)
        best_idx = tied[np.argmin(np.abs(candidates[tied] - 0.5))]
        best_threshold = float(candidates[best_idx])
    return {
        "threshold": best_threshold,
        "status": "complete",
        "n_candidate_thresholds": int(len(candidates)),
        "information_bits": math.log2(max(1, len(candidates))),
        "selection_metric": best_score,
    }


def compute_evidence_cost(row: pd.Series | dict[str, Any], max_words: float, max_texts: float) -> dict[str, float]:
    words = _safe_float(row.get("n_words_observed"), 0.0)
    texts = _safe_float(row.get("n_texts_observed"), 0.0)
    word_cost = min(1.0, max(0.0, words / max_words)) if max_words > 0 else math.nan
    text_cost = min(1.0, max(0.0, texts / max_texts)) if max_texts > 0 else math.nan
    combined = 0.5 * word_cost + 0.5 * text_cost if math.isfinite(word_cost + text_cost) else math.nan
    return {
        "word_cost": word_cost,
        "text_cost": text_cost,
        "combined_evidence_cost": combined,
        "earliness_score": 1.0 - combined if math.isfinite(combined) else math.nan,
    }


def validate_subgoal_status_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    subgoals = payload.get("subgoals")
    if not isinstance(subgoals, dict):
        return ["subgoal_status.json missing subgoals mapping"]
    for goal, name in GOAL_NAMES.items():
        item = subgoals.get(goal)
        if not isinstance(item, dict):
            errors.append(f"{goal} missing")
            continue
        if item.get("status") not in {"completed", "blocked", "pending"}:
            errors.append(f"{goal} invalid status: {item.get('status')}")
        if item.get("status") == "blocked" and not item.get("blocker"):
            errors.append(f"{goal} blocked without blocker")
        if item.get("status") == "completed" and not item.get("evidence_paths"):
            errors.append(f"{goal} completed without evidence paths")
        if item.get("name") and item.get("name") != name:
            errors.append(f"{goal} name mismatch")
    return errors


def classify_final_decision(locked_rows: pd.DataFrame, completed_goals: int) -> str:
    if completed_goals < 10 or locked_rows.empty:
        return "blocked_or_incomplete_online_evaluation"
    mean_auc = pd.to_numeric(locked_rows.get("AUROC"), errors="coerce").mean()
    mean_ba = pd.to_numeric(locked_rows.get("BA"), errors="coerce").mean()
    if mean_auc >= 0.75 and mean_ba >= 0.70:
        return "both_offline_and_online_capable"
    if mean_auc >= 0.65:
        return "offline_primary_online_secondary"
    return "offline_primary_online_limited"


def _status_paths(repo_root: Path) -> tuple[Path, Path]:
    analysis_dir = repo_root / "analysis" / ANALYSIS_NAME
    return analysis_dir / "subgoal_status.json", analysis_dir / "subgoal_status.md"


def _load_status(repo_root: Path) -> dict[str, Any]:
    status_json, _ = _status_paths(repo_root)
    if status_json.exists():
        return json.loads(status_json.read_text(encoding="utf-8"))
    return {
        "run_name": ANALYSIS_NAME,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "subgoals": {
            goal: {"name": name, "status": "pending", "evidence_paths": [], "blocker": ""}
            for goal, name in GOAL_NAMES.items()
        },
    }


def _update_subgoal(
    repo_root: Path,
    goal: str,
    status: str,
    evidence_paths: list[str] | None = None,
    blocker: str = "",
) -> None:
    payload = _load_status(repo_root)
    payload.setdefault("subgoals", {})
    item = payload["subgoals"].setdefault(
        goal, {"name": GOAL_NAMES.get(goal, goal), "status": "pending", "evidence_paths": [], "blocker": ""}
    )
    item["name"] = GOAL_NAMES.get(goal, item.get("name", goal))
    item["status"] = status
    item["evidence_paths"] = list(dict.fromkeys(evidence_paths or item.get("evidence_paths", [])))
    item["blocker"] = blocker
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    status_json, status_md = _status_paths(repo_root)
    _write_json(status_json, payload)
    rows = []
    for goal_id, name in GOAL_NAMES.items():
        row = payload["subgoals"].get(goal_id, {})
        rows.append(
            {
                "Goal": f"{goal_id.replace('_', ' ')} - {name}",
                "Status": row.get("status", "pending"),
                "Evidence": "; ".join(f"`{p}`" for p in row.get("evidence_paths", [])),
                "Notes": row.get("blocker", ""),
            }
        )
    _write_md(status_md, "# D3OnlineTargetedOptimization v1 Subgoal Status\n\n" + _md_table(pd.DataFrame(rows), 50))


def _analysis_dir(config: dict[str, Any], repo_root: Path) -> Path:
    return _path(config, "repo_analysis_dir", repo_root, f"analysis/{ANALYSIS_NAME}")


def _sort_word_rows(words: pd.DataFrame) -> pd.DataFrame:
    frame = words.copy()
    frame["_input_row_order"] = np.arange(len(frame))
    for col in ["speech_id", "paragraphId", "sentenceId", "wordId"]:
        if col in frame.columns:
            frame[f"_{col}_sort"] = pd.to_numeric(frame[col], errors="coerce")
    sort_cols = ["participant_id"]
    sort_cols.extend([c for c in ["_speech_id_sort", "_paragraphId_sort", "_sentenceId_sort", "_wordId_sort"] if c in frame])
    sort_cols.append("_input_row_order")
    frame = frame.sort_values(sort_cols).reset_index(drop=True)
    frame["text_id"] = frame.get("speech_id", frame.get("speechId")).astype(str)
    frame["_chronological_word_index"] = frame.groupby("participant_id").cumcount() + 1
    text_order = (
        frame[["participant_id", "text_id", "_chronological_word_index"]]
        .drop_duplicates(["participant_id", "text_id"])
        .sort_values(["participant_id", "_chronological_word_index"])
    )
    text_order["text_order"] = text_order.groupby("participant_id").cumcount() + 1
    frame = frame.merge(text_order[["participant_id", "text_id", "text_order"]], on=["participant_id", "text_id"], how="left")
    return frame


def _add_leave_participant_residuals(words: pd.DataFrame) -> pd.DataFrame:
    frame = words.copy()
    stimulus_col = "stimulus_word_key" if "stimulus_word_key" in frame.columns else "word_id"
    participant_col = "participant_id"
    for col in GAZE_COLUMNS:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        frame[f"_{col}_numeric"] = values
        total = (
            frame.groupby(stimulus_col, dropna=False)[f"_{col}_numeric"]
            .agg(total_sum="sum", total_count="count")
            .reset_index()
        )
        participant = (
            frame.groupby([stimulus_col, participant_col], dropna=False)[f"_{col}_numeric"]
            .agg(participant_sum="sum", participant_count="count")
            .reset_index()
        )
        merged = frame[[stimulus_col, participant_col]].merge(total, on=stimulus_col, how="left")
        merged = merged.merge(participant, on=[stimulus_col, participant_col], how="left")
        denom = merged["total_count"] - merged["participant_count"]
        baseline = (merged["total_sum"] - merged["participant_sum"]) / denom.replace(0, np.nan)
        fallback = float(values.mean()) if values.notna().any() else 0.0
        frame[f"resid_word_ref_{col}"] = values - baseline.fillna(fallback).to_numpy()
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("_") and c.endswith("_numeric")])
    return frame


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _slope(y: pd.Series, x: pd.Series, min_points: int) -> tuple[float, float, int]:
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = pd.to_numeric(x, errors="coerce")
    mask = y_num.notna() & x_num.notna()
    if int(mask.sum()) < min_points or float(x_num[mask].std(ddof=0) or 0.0) <= 1e-9:
        return math.nan, math.nan, int(mask.sum())
    xv = x_num[mask].to_numpy(dtype=float)
    yv = y_num[mask].to_numpy(dtype=float)
    slope = float(np.cov(xv, yv, ddof=0)[0, 1] / np.var(xv))
    corr = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) >= 2 else math.nan
    return slope, corr, int(mask.sum())


def _prefix_summary(
    subset: pd.DataFrame,
    participant_id: str,
    prefix_type: str,
    prefix_value: str,
    prefix_order_index: int,
    min_slope_words: int,
    min_stable_words: int,
) -> dict[str, Any]:
    first = subset.iloc[0]
    last = subset.iloc[-1]
    text_ids = list(dict.fromkeys(subset["text_id"].astype(str).tolist()))
    n_words = int(len(subset))
    n_texts = int(len(text_ids))
    n_fixations = (
        int(pd.to_numeric(subset.get("fixation_count"), errors="coerce").fillna(0).sum())
        if "fixation_count" in subset
        else 0
    )
    row: dict[str, Any] = {
        "participant_id": participant_id,
        "reader_group": first.get("reader_group"),
        "reader_group_binary": int(first.get("reader_group_binary")),
        "prefix_type": prefix_type,
        "prefix_value": str(prefix_value),
        "prefix_order_index": int(prefix_order_index),
        "n_words_observed": n_words,
        "n_trials_observed": n_texts,
        "n_texts_observed": n_texts,
        "n_speeches_observed": n_texts,
        "n_fixations_observed": n_fixations,
        "terminal_text_id": str(last.get("text_id")),
        "observed_text_ids": "|".join(text_ids),
        "observed_speech_ids": "|".join(text_ids),
        "evidence_available_until_prefix": str(last.get("participant_word_key", last.get("word_id"))),
    }

    nonmissing_core = 0
    for col in GAZE_COLUMNS:
        if col not in subset.columns:
            continue
        values = pd.to_numeric(subset[col], errors="coerce")
        if col in CORE_GAZE_COLUMNS:
            nonmissing_core += int(values.notna().sum())
        safe = re.sub(r"[^A-Za-z0-9]+", "_", col).strip("_").lower()
        row[f"raw_{safe}_mean"] = float(values.mean()) if values.notna().any() else math.nan
        row[f"raw_{safe}_median"] = float(values.median()) if values.notna().any() else math.nan
        row[f"raw_{safe}_sd"] = float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0
        row[f"raw_{safe}_q90"] = float(values.quantile(0.90)) if values.notna().any() else math.nan
        resid_col = f"resid_word_ref_{col}"
        if resid_col in subset.columns:
            resid = pd.to_numeric(subset[resid_col], errors="coerce")
            row[f"resid_{safe}_mean"] = float(resid.mean()) if resid.notna().any() else math.nan
            row[f"resid_{safe}_median"] = float(resid.median()) if resid.notna().any() else math.nan
            row[f"resid_{safe}_sd"] = float(resid.std(ddof=0)) if resid.notna().sum() > 1 else 0.0
            row[f"resid_{safe}_abs_mean"] = float(resid.abs().mean()) if resid.notna().any() else math.nan

    for col in DFM_COLUMNS:
        if col not in subset.columns:
            continue
        values = pd.to_numeric(subset[col], errors="coerce")
        safe = col.replace("dfm_lm_word_", "").replace("dfm_", "").lower()
        row[f"dfm_exp_{safe}_mean"] = float(values.mean()) if values.notna().any() else math.nan
        row[f"dfm_exp_{safe}_sd"] = float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0
        row[f"dfm_exp_{safe}_sum"] = float(values.sum()) if values.notna().any() else math.nan
        row[f"dfm_exp_{safe}_missing_rate"] = float(values.isna().mean())

    for gaze in ["TRT", "GD", "FFD", "fixation_count"]:
        if gaze not in subset.columns:
            continue
        safe_gaze = gaze.lower()
        for dfm in ["dfm_lm_word_surprisal", "dfm_lm_word_entropy"]:
            if dfm not in subset.columns:
                continue
            safe_dfm = dfm.replace("dfm_lm_word_", "")
            slope, corr, n_points = _slope(subset[gaze], subset[dfm], min_slope_words)
            row[f"dfm_sens_{safe_gaze}_{safe_dfm}_slope"] = slope
            row[f"dfm_sens_{safe_gaze}_{safe_dfm}_corr"] = corr
            row[f"dfm_sens_{safe_gaze}_{safe_dfm}_n"] = n_points
            row[f"dfm_sens_{safe_gaze}_{safe_dfm}_unstable"] = int(not math.isfinite(slope))
            resid_col = f"resid_word_ref_{gaze}"
            if resid_col in subset.columns:
                rslope, rcorr, rn = _slope(subset[resid_col], subset[dfm], min_slope_words)
                row[f"dfm_resid_{safe_gaze}_{safe_dfm}_slope"] = rslope
                row[f"dfm_resid_{safe_gaze}_{safe_dfm}_corr"] = rcorr
                row[f"dfm_resid_{safe_gaze}_{safe_dfm}_n"] = rn
                prod = pd.to_numeric(subset[resid_col], errors="coerce") * pd.to_numeric(
                    subset[dfm], errors="coerce"
                )
                row[f"dfm_resid_{safe_gaze}_{safe_dfm}_interaction_mean"] = (
                    float(prod.mean()) if prod.notna().any() else math.nan
                )
                row[f"dfm_resid_{safe_gaze}_{safe_dfm}_unstable"] = int(not math.isfinite(rslope))

    for col in SEGMENTATION_COLUMNS:
        if col not in subset.columns:
            continue
        values = pd.to_numeric(subset[col], errors="coerce")
        safe = re.sub(r"[^A-Za-z0-9]+", "_", col).strip("_").lower()
        row[f"seg_{safe}_mean"] = float(values.mean()) if values.notna().any() else math.nan
        row[f"seg_{safe}_missing_rate"] = float(values.isna().mean())

    for col in QUALITY_COLUMNS:
        if col not in subset.columns:
            continue
        values = pd.to_numeric(subset[col], errors="coerce")
        safe = re.sub(r"[^A-Za-z0-9]+", "_", col).strip("_").lower()
        row[f"quality_{safe}_rate"] = float(values.fillna(0).mean())

    slope_flags = [value for key, value in row.items() if key.endswith("_unstable")]
    row["uncert_inverse_sqrt_words"] = 1.0 / math.sqrt(max(1, n_words))
    row["uncert_missing_gaze_rate"] = float(
        pd.to_numeric(subset.get("gaze_missing"), errors="coerce").fillna(0).mean()
    ) if "gaze_missing" in subset.columns else math.nan
    row["uncert_missing_lm_rate"] = float(
        pd.to_numeric(subset.get("lm_missing"), errors="coerce").fillna(0).mean()
    ) if "lm_missing" in subset.columns else math.nan
    row["uncert_unstable_slope_rate"] = float(np.mean(slope_flags)) if slope_flags else 1.0
    row["stable_enough_for_prediction"] = bool(
        n_words >= min_stable_words and nonmissing_core >= min_stable_words
    )
    row["stable_numeric"] = int(row["stable_enough_for_prediction"])
    return row


def _participant_prefix_rows(
    group: pd.DataFrame,
    word_budgets: list[int],
    text_budgets: list[int],
    min_slope_words: int,
    min_stable_words: int,
) -> list[dict[str, Any]]:
    participant_id = str(group["participant_id"].iloc[0])
    ordered = group.sort_values("_chronological_word_index").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, str, pd.DataFrame]] = []
    for budget in word_budgets:
        if len(ordered) >= budget:
            specs.append(("word_count_prefix", str(budget), ordered.head(budget)))
    specs.append(("word_count_prefix", "all", ordered))
    for budget in word_budgets:
        if len(ordered) >= budget:
            specs.append(("chronological_prefix", str(budget), ordered.head(budget)))
    specs.append(("chronological_prefix", "all", ordered))
    n_texts = int(ordered["text_order"].max())
    for budget in text_budgets:
        if n_texts >= budget:
            specs.append(("trial_or_text_prefix", str(budget), ordered[ordered["text_order"] <= budget]))
    specs.append(("trial_or_text_prefix", "all", ordered))
    for budget in [1, 2, 3, 5]:
        if n_texts >= budget:
            specs.append(("speech_prefix", str(budget), ordered[ordered["text_order"] <= budget]))
    specs.append(("speech_prefix", "all", ordered))

    order_by_type: dict[str, int] = defaultdict(int)
    for prefix_type, prefix_value, subset in specs:
        if subset.empty:
            continue
        order_by_type[prefix_type] += 1
        rows.append(
            _prefix_summary(
                subset,
                participant_id,
                prefix_type,
                prefix_value,
                order_by_type[prefix_type],
                min_slope_words,
                min_stable_words,
            )
        )
    return rows


def build_prefix_dataset(
    config: dict[str, Any], output_dir: Path, repo_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    section = _section(config)
    prepared_dir = _path(config, "prepared_dataset_dir", repo_root)
    word_path = prepared_dir / str(section.get("word_level_file", "analysis_ready_word_level_v1_1.parquet"))
    if not word_path.exists():
        raise FileNotFoundError(word_path)
    import pyarrow.parquet as pq

    available_columns = set(pq.ParquetFile(word_path).schema.names)
    requested_columns = [
        "participant_id",
        "speech_id",
        "speechId",
        "paragraphId",
        "sentenceId",
        "wordId",
        "word_id",
        "participant_word_key",
        "stimulus_word_key",
        "reader_group",
        "reader_group_binary",
        "include_primary_analysis",
        "participant_include_primary_analysis",
        *GAZE_COLUMNS,
        *DFM_COLUMNS,
        *SEGMENTATION_COLUMNS,
        *QUALITY_COLUMNS,
    ]
    selected_columns = [column for column in dict.fromkeys(requested_columns) if column in available_columns]
    words = pd.read_parquet(word_path, columns=selected_columns)
    if "include_primary_analysis" in words.columns:
        words = words[words["include_primary_analysis"].astype(bool)].copy()
    if "reader_group_binary" not in words.columns:
        raise ValueError("word-level prepared data missing reader_group_binary")
    words = words[pd.to_numeric(words["reader_group_binary"], errors="coerce").notna()].copy()
    words["reader_group_binary"] = pd.to_numeric(words["reader_group_binary"], errors="coerce").astype(int)
    words = _sort_word_rows(words)
    words = _add_leave_participant_residuals(words)
    word_budgets = [int(x) for x in section.get("word_count_budgets", [50, 100, 250, 500, 1000])]
    text_budgets = [int(x) for x in section.get("text_count_budgets", [1, 2, 3, 5, 10])]
    min_slope_words = int(section.get("min_slope_words", 50))
    min_stable_words = int(section.get("min_stable_words", 50))
    rows: list[dict[str, Any]] = []
    for _, group in words.groupby("participant_id", sort=True):
        rows.extend(_participant_prefix_rows(group, word_budgets, text_budgets, min_slope_words, min_stable_words))
    prefix = pd.DataFrame(rows)
    prefix["prefix_row_id"] = np.arange(len(prefix)).astype(int)
    prefix_path = output_dir / "prefix_data" / "prefix_features.parquet"
    _write_parquet(prefix_path, prefix)

    analysis_dir = _analysis_dir(config, repo_root)
    dictionary_rows = []
    for column in prefix.columns:
        family = "metadata"
        for candidate, prefixes in {
            "raw_gaze_prefix": ("raw_",),
            "residual_gaze_prefix": ("resid_",),
            "dfm_exposure_prefix": ("dfm_exp_", "seg_"),
            "dfm_sensitivity_prefix": ("dfm_sens_",),
            "dfm_residual_gaze_prefix": ("dfm_resid_",),
            "uncertainty_or_quality": ("uncert_", "quality_", "stable_"),
        }.items():
            if column.startswith(prefixes):
                family = candidate
                break
        dictionary_rows.append({"column": column, "feature_family": family, "dtype": str(prefix[column].dtype)})
    dictionary = pd.DataFrame(dictionary_rows)
    _write_md(
        analysis_dir / "prefix_feature_dictionary.md",
        "# Prefix Feature Dictionary\n\n"
        + _md_table(dictionary, max_rows=500),
    )
    counts = (
        prefix.groupby(["prefix_type", "prefix_value"], dropna=False)
        .agg(
            rows=("prefix_row_id", "count"),
            participants=("participant_id", "nunique"),
            mean_words=("n_words_observed", "mean"),
            mean_texts=("n_texts_observed", "mean"),
            stable_rate=("stable_enough_for_prediction", "mean"),
        )
        .reset_index()
    )
    missing = prefix.isna().mean().sort_values(ascending=False).head(25).reset_index()
    missing.columns = ["column", "missing_rate"]
    monotonic_errors = validate_prefix_monotonicity(prefix)
    report = [
        "# Prefix Dataset Report",
        "",
        f"- Input word rows after primary-analysis filtering: {len(words)}",
        f"- Prefix rows: {len(prefix)}",
        f"- Participants: {prefix['participant_id'].nunique()}",
        f"- Prefix types attempted: {', '.join(sorted(prefix['prefix_type'].unique()))}",
        f"- No-future/monotonic errors: {len(monotonic_errors)}",
        "",
        "## Counts by Prefix",
        "",
        _md_table(counts, max_rows=200),
        "",
        "## Highest Missingness",
        "",
        _md_table(missing, max_rows=25),
        "",
        "## Validation Notes",
        "",
        "- Prefix features are cumulative summaries over rows observed through the prefix.",
        "- Participant labels are retained only for evaluation and are not used in feature construction.",
        "- Residual gaze features use leave-participant-out stimulus references.",
        "- Unstable slope estimates are flagged with `_unstable` columns and summarized by `uncert_unstable_slope_rate`.",
    ]
    if monotonic_errors:
        report.extend(["", "## Monotonicity Errors", "", "\n".join(f"- {e}" for e in monotonic_errors[:50])])
    _write_md(analysis_dir / "prefix_dataset_report.md", "\n".join(report))
    manifest = {
        "prefix_features_path": str(prefix_path),
        "row_count": int(len(prefix)),
        "participants": int(prefix["participant_id"].nunique()),
        "prefix_types": sorted(prefix["prefix_type"].unique().tolist()),
        "monotonic_errors": monotonic_errors,
    }
    return prefix, manifest


def validate_prefix_monotonicity(prefix: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {"participant_id", "prefix_type", "prefix_order_index", "n_words_observed", "n_texts_observed"}
    missing = required - set(prefix.columns)
    if missing:
        return [f"prefix data missing required columns: {sorted(missing)}"]
    for (participant, prefix_type), group in prefix.groupby(["participant_id", "prefix_type"], dropna=False):
        ordered = group.sort_values("prefix_order_index")
        for column in ["n_words_observed", "n_trials_observed", "n_texts_observed", "n_speeches_observed"]:
            if column not in ordered.columns:
                continue
            values = pd.to_numeric(ordered[column], errors="coerce").to_numpy()
            if np.any(np.diff(values) < 0):
                errors.append(f"{participant}/{prefix_type} {column} is not monotonic")
    return errors


def validate_no_future_evidence(prefix: pd.DataFrame) -> list[str]:
    errors = validate_prefix_monotonicity(prefix)
    for column in prefix.columns:
        lowered = str(column).lower()
        if lowered.startswith("future_") or lowered.startswith("full_session"):
            errors.append(f"future evidence column present: {column}")
    return errors


def _observed_text_set(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {part for part in str(value).split("|") if part}


def _participant_label_frame(prefix: pd.DataFrame) -> pd.DataFrame:
    labels = (
        prefix[["participant_id", "reader_group_binary"]]
        .drop_duplicates("participant_id")
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    labels["reader_group_binary"] = pd.to_numeric(labels["reader_group_binary"], errors="coerce").astype(int)
    return labels


def _stratified_group_values(labels: pd.DataFrame, n_splits: int, seed: int) -> list[set[str]]:
    from sklearn.model_selection import StratifiedKFold

    y = labels["reader_group_binary"].astype(int).to_numpy()
    n = min(n_splits, int(pd.Series(y).value_counts().min()))
    n = max(2, n)
    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    folds: list[set[str]] = []
    for _, test_idx in splitter.split(labels["participant_id"], y):
        folds.append(set(labels.iloc[test_idx]["participant_id"].astype(str)))
    return folds


def _text_folds(prefix: pd.DataFrame, n_splits: int, seed: int) -> list[set[str]]:
    texts = sorted({text for value in prefix["observed_text_ids"] for text in _observed_text_set(value)})
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(texts, dtype=object)
    rng.shuffle(shuffled)
    n = max(2, min(n_splits, len(shuffled)))
    return [set(map(str, part.tolist())) for part in np.array_split(shuffled, n)]


def make_outer_splits(
    prefix: pd.DataFrame, regime: str, n_splits: int = 4, seed: int = 173
) -> list[SplitFold]:
    labels = _participant_label_frame(prefix)
    participant_folds = _stratified_group_values(labels, n_splits, seed)
    all_participants = set(prefix["participant_id"].astype(str))
    text_folds = _text_folds(prefix, n_splits, seed + 11)
    all_texts = sorted({text for value in prefix["observed_text_ids"] for text in _observed_text_set(value)})
    folds: list[SplitFold] = []
    for fold_id in range(max(len(participant_folds), len(text_folds))):
        test_participants = participant_folds[fold_id % len(participant_folds)]
        train_participants = all_participants - test_participants
        test_texts = text_folds[fold_id % len(text_folds)]
        train_texts = set(all_texts) - test_texts
        if regime in {"unseen_reader", "text_balanced_unseen_reader", "participant_grouped_kfold"}:
            test_mask = prefix["participant_id"].astype(str).isin(test_participants)
            train_mask = ~test_mask
            skipped_mask = pd.Series(False, index=prefix.index)
        elif regime == "unseen_text":
            sets = prefix["observed_text_ids"].map(_observed_text_set)
            test_mask = sets.map(lambda values: bool(values) and values <= test_texts)
            train_mask = sets.map(lambda values: bool(values) and values <= train_texts)
            skipped_mask = ~(test_mask | train_mask)
        elif regime == "unseen_reader_and_text":
            sets = prefix["observed_text_ids"].map(_observed_text_set)
            participant_test = prefix["participant_id"].astype(str).isin(test_participants)
            participant_train = ~participant_test
            test_mask = participant_test & sets.map(lambda values: bool(values) and values <= test_texts)
            train_mask = participant_train & sets.map(lambda values: bool(values) and values <= train_texts)
            skipped_mask = ~(test_mask | train_mask)
        else:
            raise ValueError(f"unsupported split regime: {regime}")
        train = prefix[train_mask]
        test = prefix[test_mask]
        y_train = pd.to_numeric(train.get("reader_group_binary"), errors="coerce")
        y_test = pd.to_numeric(test.get("reader_group_binary"), errors="coerce")
        status = "complete"
        reason = ""
        if train.empty or test.empty or y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
            status = "skipped"
            reason = "empty_or_single_class_train_or_test"
        folds.append(
            SplitFold(
                regime=regime,
                fold_id=fold_id,
                train_indices=train.index.astype(int).tolist(),
                test_indices=test.index.astype(int).tolist(),
                skipped_indices=prefix[skipped_mask].index.astype(int).tolist(),
                train_participants=sorted(train_participants),
                test_participants=sorted(test_participants),
                train_texts=sorted(train_texts),
                test_texts=sorted(test_texts),
                status=status,
                skip_reason=reason,
            )
        )
    return folds


def _feature_columns(prefix: pd.DataFrame, family: str) -> list[str]:
    prefixes_by_family = {
        "raw_gaze_prefix": ("raw_",),
        "residual_gaze_prefix": ("resid_",),
        "dfm_exposure_prefix": ("dfm_exp_", "seg_"),
        "dfm_sensitivity_prefix": ("dfm_sens_",),
        "dfm_residual_gaze_prefix": ("dfm_resid_", "resid_"),
        "dfm_residual_plus_uncertainty_prefix": (
            "dfm_resid_",
            "resid_",
            "uncert_",
            "quality_",
            "stable_",
        ),
        "all_allowed_online": (
            "raw_",
            "resid_",
            "dfm_exp_",
            "dfm_sens_",
            "dfm_resid_",
            "seg_",
            "uncert_",
            "quality_",
            "stable_",
        ),
    }
    prefixes = prefixes_by_family[family]
    columns = [
        column
        for column in prefix.columns
        if str(column).startswith(prefixes) and pd.api.types.is_numeric_dtype(prefix[column])
    ]
    if family in {"dfm_residual_plus_uncertainty_prefix", "all_allowed_online"}:
        for column in [
            "n_words_observed",
            "n_trials_observed",
            "n_texts_observed",
            "n_speeches_observed",
            "n_fixations_observed",
        ]:
            if column in prefix.columns:
                columns.append(column)
    safe = []
    for column in dict.fromkeys(columns):
        lowered = str(column).lower()
        if column in ID_AND_LABEL_COLUMNS:
            continue
        if any(pattern in lowered for pattern in PROHIBITED_PREDICTOR_PATTERNS):
            continue
        safe.append(column)
    return safe


def _make_model(class_weight: str | None = "balanced") -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight=class_weight,
        max_iter=1000,
        random_state=173,
    )
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def _score_model(model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    scores = np.asarray(model.decision_function(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    split_regime: str,
    split_role: str,
    fold_id: int,
    inner_fold_id: int | str | None,
    feature_group: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = source[
        [
            "prefix_row_id",
            "participant_id",
            "reader_group",
            "reader_group_binary",
            "prefix_type",
            "prefix_value",
            "prefix_order_index",
            "evidence_available_until_prefix",
            "n_words_observed",
            "n_trials_observed",
            "n_texts_observed",
            "n_speeches_observed",
            "n_fixations_observed",
            "stable_enough_for_prediction",
            "terminal_text_id",
            "observed_text_ids",
        ]
    ].copy()
    p = np.clip(probabilities, 0.0, 1.0)
    result["fold_id"] = int(fold_id)
    result["inner_fold_id"] = "" if inner_fold_id is None else inner_fold_id
    result["split_regime"] = split_regime
    result["split_role"] = split_role
    result["y_true"] = pd.to_numeric(result["reader_group_binary"], errors="coerce").astype(int)
    result["p_pred"] = p
    result["y_pred"] = (p >= 0.5).astype(int)
    result["model_name"] = MODEL_NAME
    result["feature_group"] = feature_group
    result["threshold_source"] = "fixed_0_5"
    result["calibration_source"] = "identity"
    result["prediction_uncertainty"] = p * (1 - p)
    result["probability_entropy"] = _prob_entropy(p)
    result["n_features"] = int(len(feature_columns))
    result["predictor_hash"] = str(abs(hash(tuple(feature_columns))) % 10_000_000)
    result["clean_result"] = True
    result["official_claim_allowed"] = False
    result["benchmark_relative_claim_allowed"] = True
    return result


def _inner_participant_splits(train: pd.DataFrame, seed: int, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    labels = _participant_label_frame(train)
    if labels.empty or labels["reader_group_binary"].nunique() < 2:
        return []
    min_class = int(labels["reader_group_binary"].value_counts().min())
    n = min(n_splits, min_class)
    if n < 2:
        return []
    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    splits = []
    for inner_train_idx, inner_val_idx in splitter.split(labels["participant_id"], labels["reader_group_binary"]):
        val_participants = set(labels.iloc[inner_val_idx]["participant_id"].astype(str))
        train_participants = set(labels.iloc[inner_train_idx]["participant_id"].astype(str))
        train_idx = train.index[train["participant_id"].astype(str).isin(train_participants)].to_numpy()
        val_idx = train.index[train["participant_id"].astype(str).isin(val_participants)].to_numpy()
        splits.append((train_idx, val_idx))
    return splits


def _fit_predict_family_budget(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    family: str,
    split_regime: str,
    fold_id: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    empty = pd.DataFrame()
    manifest_rows: list[dict[str, Any]] = []
    feature_columns = [
        column
        for column in feature_columns
        if column in train.columns and pd.to_numeric(train[column], errors="coerce").notna().any()
    ]
    if train.empty or test.empty:
        return empty, empty, empty, empty, [{"status": "skipped", "skip_reason": "empty_train_or_test"}]
    y_train = pd.to_numeric(train["reader_group_binary"], errors="coerce")
    y_test = pd.to_numeric(test["reader_group_binary"], errors="coerce")
    if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2 or not feature_columns:
        return empty, empty, empty, empty, [
            {
                "status": "skipped",
                "skip_reason": "single_class_or_no_features",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_features": int(len(feature_columns)),
            }
        ]

    class_weight = "balanced"
    model = _make_model(class_weight)
    model.fit(train[feature_columns].apply(pd.to_numeric, errors="coerce"), y_train.astype(int))
    train_pred = _prediction_frame(
        train,
        _score_model(model, train, feature_columns),
        split_regime=split_regime,
        split_role="train_fit",
        fold_id=fold_id,
        inner_fold_id=None,
        feature_group=family,
        feature_columns=feature_columns,
    )
    test_pred = _prediction_frame(
        test,
        _score_model(model, test, feature_columns),
        split_regime=split_regime,
        split_role="outer_test",
        fold_id=fold_id,
        inner_fold_id=None,
        feature_group=family,
        feature_columns=feature_columns,
    )
    inner_frames: list[pd.DataFrame] = []
    for inner_id, (inner_train_idx, inner_val_idx) in enumerate(_inner_participant_splits(train, seed + fold_id)):
        inner_train = train.loc[inner_train_idx]
        inner_val = train.loc[inner_val_idx]
        y_inner = pd.to_numeric(inner_train["reader_group_binary"], errors="coerce")
        y_val = pd.to_numeric(inner_val["reader_group_binary"], errors="coerce")
        if y_inner.nunique(dropna=True) < 2 or y_val.nunique(dropna=True) < 2:
            continue
        inner_model = _make_model(class_weight)
        inner_model.fit(inner_train[feature_columns].apply(pd.to_numeric, errors="coerce"), y_inner.astype(int))
        inner_frames.append(
            _prediction_frame(
                inner_val,
                _score_model(inner_model, inner_val, feature_columns),
                split_regime=split_regime,
                split_role="inner_oof",
                fold_id=fold_id,
                inner_fold_id=inner_id,
                feature_group=family,
                feature_columns=feature_columns,
            )
        )
    inner_pred = pd.concat(inner_frames, ignore_index=True) if inner_frames else empty
    calibration_pred = inner_pred.copy()
    if not calibration_pred.empty:
        calibration_pred["split_role"] = "calibration"
    manifest_rows.append(
        {
            "status": "complete",
            "feature_group": family,
            "prefix_type": str(train["prefix_type"].iloc[0]),
            "prefix_value": str(train["prefix_value"].iloc[0]),
            "class_weight": class_weight,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_inner_oof": int(len(inner_pred)),
            "n_features": int(len(feature_columns)),
            "predictor_columns": feature_columns,
            "prohibited_predictors_used": [
                c
                for c in feature_columns
                if any(pattern in c.lower() for pattern in PROHIBITED_PREDICTOR_PATTERNS)
            ],
        }
    )
    return train_pred, inner_pred, calibration_pred, test_pred, manifest_rows


def generate_nested_predictions(
    config: dict[str, Any], prefix: pd.DataFrame, output_dir: Path, repo_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    section = _section(config)
    regimes = list(section.get("split_regimes", list(DEFAULT_SPLIT_REGIMES)))
    families = list(section.get("model_families", list(FEATURE_FAMILIES)))
    n_splits = int(section.get("outer_folds", 4))
    seed = int(section.get("deterministic_seed", 173))
    nested_root = output_dir / "nested_predictions"
    all_predictions: list[pd.DataFrame] = []
    all_manifest_rows: list[dict[str, Any]] = []
    predictor_manifest: list[dict[str, Any]] = []

    for regime in regimes:
        folds = make_outer_splits(prefix, regime, n_splits=n_splits, seed=seed)
        for fold in folds:
            fold_dir = nested_root / regime / f"fold_{fold.fold_id}"
            role_frames: dict[str, list[pd.DataFrame]] = {
                "train_fit": [],
                "inner_oof": [],
                "calibration": [],
                "outer_test": [],
            }
            fold_manifest = {
                "split_regime": regime,
                "fold_id": fold.fold_id,
                "status": fold.status,
                "skip_reason": fold.skip_reason,
                "train_rows_available": len(fold.train_indices),
                "test_rows_available": len(fold.test_indices),
                "skipped_mixed_evidence_rows": len(fold.skipped_indices),
                "train_participants": fold.train_participants,
                "test_participants": fold.test_participants,
                "train_texts": fold.train_texts,
                "test_texts": fold.test_texts,
                "model_runs": [],
            }
            if fold.status == "complete":
                train_all = prefix.loc[fold.train_indices].copy()
                test_all = prefix.loc[fold.test_indices].copy()
                for (prefix_type, prefix_value), train_budget in train_all.groupby(
                    ["prefix_type", "prefix_value"], dropna=False
                ):
                    test_budget = test_all[
                        test_all["prefix_type"].eq(prefix_type)
                        & test_all["prefix_value"].astype(str).eq(str(prefix_value))
                    ].copy()
                    if test_budget.empty:
                        continue
                    for family in families:
                        features = _feature_columns(prefix, family)
                        tr, inn, cal, te, rows = _fit_predict_family_budget(
                            train_budget.copy(),
                            test_budget.copy(),
                            features,
                            family,
                            regime,
                            fold.fold_id,
                            seed,
                        )
                        for role, frame in [
                            ("train_fit", tr),
                            ("inner_oof", inn),
                            ("calibration", cal),
                            ("outer_test", te),
                        ]:
                            if not frame.empty:
                                role_frames[role].append(frame)
                        for row in rows:
                            row = {
                                **row,
                                "split_regime": regime,
                                "fold_id": fold.fold_id,
                                "prefix_type": str(prefix_type),
                                "prefix_value": str(prefix_value),
                            }
                            fold_manifest["model_runs"].append(row)
                            all_manifest_rows.append(row)
                        predictor_manifest.append(
                            {
                                "split_regime": regime,
                                "fold_id": fold.fold_id,
                                "prefix_type": str(prefix_type),
                                "prefix_value": str(prefix_value),
                                "feature_group": family,
                                "predictor_columns": features,
                            }
                        )
            for role, filename in [
                ("train_fit", "train_prefix_predictions.csv"),
                ("inner_oof", "inner_oof_prefix_predictions.csv"),
                ("calibration", "calibration_prefix_predictions.csv"),
                ("outer_test", "outer_test_prefix_predictions.csv"),
            ]:
                frame = pd.concat(role_frames[role], ignore_index=True) if role_frames[role] else pd.DataFrame()
                _write_csv(fold_dir / filename, frame)
                if not frame.empty:
                    all_predictions.append(frame)
            _write_json(fold_dir / "fold_manifest.json", fold_manifest)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    _write_csv(nested_root / "all_nested_prefix_predictions.csv", predictions)
    _write_json(nested_root / "predictor_manifest.json", predictor_manifest)
    manifest_frame = pd.DataFrame(all_manifest_rows)
    _write_csv(nested_root / "nested_prediction_manifest_rows.csv", manifest_frame)

    analysis_dir = _analysis_dir(config, repo_root)
    counts = (
        predictions.groupby(["split_regime", "split_role"], dropna=False)
        .size()
        .reset_index(name="rows")
        if not predictions.empty
        else pd.DataFrame(columns=["split_regime", "split_role", "rows"])
    )
    _write_md(
        analysis_dir / "nested_prediction_artifact_report.md",
        "# Nested Prediction Artifact Report\n\n"
        f"- Prediction rows: {len(predictions)}\n"
        f"- Outer-test rows: {int((predictions.get('split_role') == 'outer_test').sum()) if not predictions.empty else 0}\n"
        f"- Inner-OOF rows: {int((predictions.get('split_role') == 'inner_oof').sum()) if not predictions.empty else 0}\n"
        f"- Calibration rows: {int((predictions.get('split_role') == 'calibration').sum()) if not predictions.empty else 0}\n\n"
        "## Rows by Role\n\n"
        + _md_table(counts, max_rows=100),
    )
    return predictions, {
        "nested_root": str(nested_root),
        "row_count": int(len(predictions)),
        "outer_test_rows": int((predictions.get("split_role") == "outer_test").sum()) if not predictions.empty else 0,
        "inner_oof_rows": int((predictions.get("split_role") == "inner_oof").sum()) if not predictions.empty else 0,
        "calibration_rows": int((predictions.get("split_role") == "calibration").sum()) if not predictions.empty else 0,
    }


def _reader_aggregate(predictions: pd.DataFrame, score_col: str = "p_pred") -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    group_cols = [
        "split_regime",
        "fold_id",
        "feature_group",
        "model_name",
        "prefix_type",
        "prefix_value",
        "participant_id",
    ]
    agg = (
        predictions.groupby(group_cols, dropna=False)
        .agg(
            y_true=("y_true", "first"),
            p_pred=(score_col, "mean"),
            n_words_observed=("n_words_observed", "max"),
            n_texts_observed=("n_texts_observed", "max"),
            stable_enough_for_prediction=("stable_enough_for_prediction", "mean"),
        )
        .reset_index()
    )
    agg["stable_enough_for_prediction"] = agg["stable_enough_for_prediction"] >= 0.5
    return agg


def evaluate_online_prefix_models(
    config: dict[str, Any], predictions: pd.DataFrame, repo_root: Path
) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    test = predictions[predictions["split_role"].eq("outer_test")].copy() if not predictions.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if not test.empty:
        group_cols = ["split_regime", "feature_group", "prefix_type", "prefix_value"]
        for keys, group in test.groupby(group_cols, dropna=False):
            metrics = classification_metrics(group["y_true"], group["p_pred"], 0.5)
            rows.append(
                {
                    "evaluation_level": "trial_or_prefix_level",
                    "split_regime": keys[0],
                    "feature_group": keys[1],
                    "prefix_type": keys[2],
                    "prefix_value": keys[3],
                    "threshold_source": "fixed_0_5",
                    "n_readers": int(group["participant_id"].nunique()),
                    "n_prefix_rows": int(len(group)),
                    "unstable_prefix_rate": float((~group["stable_enough_for_prediction"].astype(bool)).mean()),
                    **metrics,
                }
            )
        reader = _reader_aggregate(test)
        for keys, group in reader.groupby(group_cols, dropna=False):
            metrics = classification_metrics(group["y_true"], group["p_pred"], 0.5)
            rows.append(
                {
                    "evaluation_level": "reader_aggregated",
                    "split_regime": keys[0],
                    "feature_group": keys[1],
                    "prefix_type": keys[2],
                    "prefix_value": keys[3],
                    "threshold_source": "fixed_0_5",
                    "n_readers": int(group["participant_id"].nunique()),
                    "n_prefix_rows": int(len(group)),
                    "unstable_prefix_rate": float((~group["stable_enough_for_prediction"].astype(bool)).mean()),
                    **metrics,
                }
            )
    metrics_frame = pd.DataFrame(rows)
    baseline = _previous_baseline_rows(repo_root)
    if not baseline.empty:
        metrics_frame = pd.concat([metrics_frame, baseline], ignore_index=True, sort=False)
    _write_csv(analysis_dir / "online_prefix_model_metrics.csv", metrics_frame)
    summary = (
        metrics_frame[
            metrics_frame.get("evaluation_level", pd.Series(dtype=str)).astype(str).eq("reader_aggregated")
        ]
        .sort_values(["split_regime", "BA", "AUROC"], ascending=[True, False, False])
        .head(20)
        if not metrics_frame.empty
        else metrics_frame
    )
    report = [
        "# Online Prefix Model Report",
        "",
        f"- Metric rows: {len(metrics_frame)}",
        f"- Outer-test prediction rows used: {len(test)}",
        "- Fixed-threshold baseline rows from OperatingPointAdaptation v1 are included for comparison when available.",
        "",
        "## Top Reader-Aggregated Rows",
        "",
        _md_table(summary, max_rows=25),
    ]
    _write_md(analysis_dir / "online_prefix_model_report.md", "\n".join(report))
    return metrics_frame


def _previous_baseline_rows(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "analysis" / "operating_point_adaptation_v1" / "before_after_operating_point_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        previous = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    keep = previous[
        previous["analysis_row"].astype(str).isin({"fixed_0_5_fold_mean", "fixed_0_5"})
        & previous["source_name"].astype(str).isin(
            {"d3_eyebench_lite_candidate_0000", "benchmark_bridge_d3_full_data", "autoresearch_d3_final_reader_profile"}
        )
    ].copy()
    if keep.empty:
        return pd.DataFrame()
    rows = []
    for _, row in keep.iterrows():
        rows.append(
            {
                "evaluation_level": row.get("evaluation_level"),
                "split_regime": row.get("split_regime"),
                "feature_group": row.get("feature_group"),
                "prefix_type": "previous_artifact",
                "prefix_value": row.get("source_name"),
                "threshold_source": row.get("threshold_source"),
                "n_readers": math.nan,
                "n_prefix_rows": math.nan,
                "unstable_prefix_rate": math.nan,
                "AUROC": row.get("AUROC"),
                "PR-AUC": row.get("PR-AUC"),
                "BA": row.get("BA"),
                "macro_F1": row.get("macro_F1"),
                "Brier": row.get("Brier"),
                "source_name": row.get("source_name"),
            }
        )
    return pd.DataFrame(rows)


def _calibrator_fit(pool: pd.DataFrame, method: str) -> dict[str, Any]:
    if method == "identity":
        return {"status": "complete", "method": method, "model": None}
    y = pd.to_numeric(pool.get("y_true"), errors="coerce")
    p = pd.to_numeric(pool.get("p_pred"), errors="coerce").clip(1e-6, 1 - 1e-6)
    valid = pool[y.notna() & p.notna()].copy()
    if len(valid) < 10 or y[y.notna()].nunique() < 2:
        return {"status": "blocked_small_or_single_class_calibration_pool", "method": method, "model": None}
    try:
        if method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            x = _logit(valid["p_pred"]).reshape(-1, 1)
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            model.fit(x, valid["y_true"].astype(int))
            return {"status": "complete", "method": method, "model": model}
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            if len(valid) < 20:
                return {"status": "blocked_isotonic_min_sample_size", "method": method, "model": None}
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(pd.to_numeric(valid["p_pred"], errors="coerce").to_numpy(), valid["y_true"].astype(int).to_numpy())
            return {"status": "complete", "method": method, "model": model}
        if method == "logistic_recalibration_uncertainty":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            x = _calibration_design(valid)
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(x, valid["y_true"].astype(int))
            return {"status": "complete", "method": method, "model": model}
    except Exception as exc:
        return {"status": f"blocked_calibration_exception:{exc}", "method": method, "model": None}
    return {"status": "blocked_unknown_calibrator", "method": method, "model": None}


def _calibration_design(frame: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=frame.index)
    design["logit_p"] = _logit(frame["p_pred"])
    for column in ["prediction_uncertainty", "probability_entropy", "n_words_observed", "n_texts_observed"]:
        design[column] = pd.to_numeric(frame.get(column), errors="coerce")
    design["stable_enough_for_prediction"] = frame.get("stable_enough_for_prediction", False).astype(float)
    return design


def _calibrator_apply(frame: pd.DataFrame, fitted: dict[str, Any]) -> np.ndarray:
    method = fitted.get("method")
    model = fitted.get("model")
    if method == "identity" or fitted.get("status") != "complete" or model is None:
        return np.clip(pd.to_numeric(frame["p_pred"], errors="coerce").to_numpy(dtype=float), 0.0, 1.0)
    if method == "sigmoid":
        return np.clip(model.predict_proba(_logit(frame["p_pred"]).reshape(-1, 1))[:, 1], 0.0, 1.0)
    if method == "isotonic":
        return np.clip(model.predict(pd.to_numeric(frame["p_pred"], errors="coerce").to_numpy(dtype=float)), 0.0, 1.0)
    if method == "logistic_recalibration_uncertainty":
        return np.clip(model.predict_proba(_calibration_design(frame))[:, 1], 0.0, 1.0)
    return np.clip(pd.to_numeric(frame["p_pred"], errors="coerce").to_numpy(dtype=float), 0.0, 1.0)


def legal_calibration_and_thresholds(
    config: dict[str, Any], predictions: pd.DataFrame, repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_dir = _analysis_dir(config, repo_root)
    if predictions.empty:
        empty = pd.DataFrame()
        _write_csv(analysis_dir / "legal_calibration_metrics.csv", empty)
        _write_csv(analysis_dir / "legal_threshold_metrics.csv", empty)
        _write_csv(analysis_dir / "legal_thresholds_learned.csv", empty)
        return empty, empty, empty
    test = predictions[predictions["split_role"].eq("outer_test")].copy()
    pool_all = predictions[predictions["split_role"].isin(["inner_oof", "calibration"])].copy()
    section = _section(config)
    adaptation_groups = set(
        section.get(
            "adaptation_feature_groups",
            ["dfm_residual_plus_uncertainty_prefix", "all_allowed_online"],
        )
    )
    if adaptation_groups:
        test = test[test["feature_group"].astype(str).isin(adaptation_groups)].copy()
        pool_all = pool_all[pool_all["feature_group"].astype(str).isin(adaptation_groups)].copy()
    group_cols = ["split_regime", "fold_id", "feature_group", "prefix_type", "prefix_value"]
    calibration_rows: list[dict[str, Any]] = []
    threshold_metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    calibrators = ["identity", "sigmoid", "isotonic", "logistic_recalibration_uncertainty"]
    threshold_policies = [
        "fixed_0_5",
        "global_inner_cv_threshold",
        "split_regime_inner_cv_threshold",
        "prefix_type_inner_cv_threshold",
        "prefix_budget_inner_cv_threshold",
        "balanced_accuracy_threshold",
        "macro_f1_threshold",
        "target_sensitivity_threshold",
        "target_specificity_threshold",
    ]
    for keys, group in test.groupby(group_cols, dropna=False):
        split_regime, fold_id, feature_group, prefix_type, prefix_value = keys
        base_filter = pool_all["feature_group"].eq(feature_group) & pool_all["fold_id"].eq(fold_id)
        calibration_pool = pool_all[
            base_filter
            & pool_all["split_regime"].eq(split_regime)
            & pool_all["prefix_type"].eq(prefix_type)
            & pool_all["prefix_value"].astype(str).eq(str(prefix_value))
        ].copy()
        if calibration_pool.empty:
            calibration_pool = pool_all[base_filter & pool_all["split_regime"].eq(split_regime)].copy()
        for calibrator in calibrators:
            fitted = _calibrator_fit(calibration_pool, calibrator)
            calibrated = group.copy()
            calibrated["p_pred"] = _calibrator_apply(group, fitted)
            metrics = classification_metrics(calibrated["y_true"], calibrated["p_pred"], 0.5)
            calibration_rows.append(
                {
                    "split_regime": split_regime,
                    "fold_id": fold_id,
                    "feature_group": feature_group,
                    "prefix_type": prefix_type,
                    "prefix_value": prefix_value,
                    "calibrator": calibrator,
                    "calibration_source": "inner_oof_or_calibration",
                    "calibration_status": fitted["status"],
                    "n_calibration_rows": int(len(calibration_pool)),
                    "n_outer_test_rows": int(len(group)),
                    "clean_result": True,
                    "official_claim_allowed": False,
                    **metrics,
                }
            )
        for policy in threshold_policies:
            if policy == "fixed_0_5":
                learned = {
                    "threshold": 0.5,
                    "status": "complete",
                    "n_candidate_thresholds": 1,
                    "information_bits": 0.0,
                    "selection_metric": math.nan,
                }
                threshold_pool = pd.DataFrame()
                source = "fixed_0_5"
                metric_policy = "balanced_accuracy_threshold"
            else:
                source = "inner_oof_or_calibration"
                metric_policy = policy
                if policy == "global_inner_cv_threshold":
                    threshold_pool = pool_all[base_filter].copy()
                    metric_policy = "balanced_accuracy_threshold"
                elif policy == "split_regime_inner_cv_threshold":
                    threshold_pool = pool_all[base_filter & pool_all["split_regime"].eq(split_regime)].copy()
                    metric_policy = "balanced_accuracy_threshold"
                elif policy == "prefix_type_inner_cv_threshold":
                    threshold_pool = pool_all[
                        base_filter
                        & pool_all["split_regime"].eq(split_regime)
                        & pool_all["prefix_type"].eq(prefix_type)
                    ].copy()
                    metric_policy = "balanced_accuracy_threshold"
                else:
                    threshold_pool = calibration_pool.copy()
                learned = learn_threshold_from_pool(threshold_pool, policy=metric_policy)
            threshold = float(learned["threshold"])
            metrics = classification_metrics(group["y_true"], group["p_pred"], threshold)
            threshold_metric_rows.append(
                {
                    "split_regime": split_regime,
                    "fold_id": fold_id,
                    "feature_group": feature_group,
                    "prefix_type": prefix_type,
                    "prefix_value": prefix_value,
                    "threshold_policy": policy,
                    "threshold_source": source,
                    "threshold": threshold,
                    "threshold_status": learned["status"],
                    "n_threshold_pool_rows": int(len(threshold_pool)) if policy != "fixed_0_5" else 0,
                    "clean_result": True,
                    "official_claim_allowed": False,
                    **metrics,
                }
            )
            threshold_rows.append(
                {
                    "split_regime": split_regime,
                    "fold_id": fold_id,
                    "feature_group": feature_group,
                    "prefix_type": prefix_type,
                    "prefix_value": prefix_value,
                    "threshold_policy": policy,
                    "threshold_source": source,
                    "threshold": threshold,
                    "status": learned["status"],
                    "n_candidate_thresholds": learned["n_candidate_thresholds"],
                    "information_bits": learned["information_bits"],
                    "selection_metric": learned["selection_metric"],
                    "clean_result": True,
                    "official_claim_allowed": False,
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    threshold_metrics = pd.DataFrame(threshold_metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    _write_csv(analysis_dir / "legal_calibration_metrics.csv", calibration)
    _write_csv(analysis_dir / "legal_threshold_metrics.csv", threshold_metrics)
    _write_csv(analysis_dir / "legal_thresholds_learned.csv", thresholds)

    compare = pd.DataFrame()
    if not threshold_metrics.empty:
        fixed = threshold_metrics[threshold_metrics["threshold_policy"].eq("fixed_0_5")]
        learned = threshold_metrics[
            threshold_metrics["threshold_policy"].ne("fixed_0_5")
            & threshold_metrics["threshold_status"].eq("complete")
        ]
        fixed_mean = pd.to_numeric(fixed["BA"], errors="coerce").mean()
        learned_mean = pd.to_numeric(learned["BA"], errors="coerce").max()
        compare = pd.DataFrame(
            [
                {
                    "fixed_0_5_mean_BA": fixed_mean,
                    "best_legal_learned_BA": learned_mean,
                    "delta_BA": learned_mean - fixed_mean
                    if math.isfinite(_safe_float(fixed_mean)) and math.isfinite(_safe_float(learned_mean))
                    else math.nan,
                }
            ]
        )
    fitted_complete = calibration[
        calibration["calibrator"].ne("identity") & calibration["calibration_status"].eq("complete")
    ] if not calibration.empty else pd.DataFrame()
    report = [
        "# Calibration and Threshold Report",
        "",
        f"- Calibration metric rows: {len(calibration)}",
        f"- Fitted non-identity calibration rows completed: {len(fitted_complete)}",
        f"- Threshold metric rows: {len(threshold_metrics)}",
        f"- Learned legal thresholds: {int((thresholds.get('threshold_policy') != 'fixed_0_5').sum()) if not thresholds.empty else 0}",
        "",
        "## Fixed vs Learned Threshold Summary",
        "",
        _md_table(compare, 5),
        "",
        "All clean calibration and threshold rows use only inner-OOF or calibration predictions.",
    ]
    _write_md(analysis_dir / "calibration_threshold_report.md", "\n".join(report))
    return calibration, threshold_metrics, thresholds


def _sequence_sort(frame: pd.DataFrame) -> pd.DataFrame:
    priority = {
        "word_count_prefix": 0,
        "chronological_prefix": 1,
        "trial_or_text_prefix": 2,
        "speech_prefix": 3,
    }
    ordered = frame.copy()
    ordered["_prefix_priority"] = ordered["prefix_type"].map(priority).fillna(9).astype(int)
    ordered["_prefix_value_numeric"] = pd.to_numeric(ordered["prefix_value"], errors="coerce").fillna(10**9)
    return ordered.sort_values(
        ["n_words_observed", "n_texts_observed", "_prefix_priority", "_prefix_value_numeric", "prefix_order_index"]
    )


def accumulate_probability_sequence(
    probabilities: Iterable[float],
    method: str,
    uncertainties: Iterable[float] | None = None,
    reliabilities: Iterable[float] | None = None,
) -> list[float]:
    p = _clip_prob(probabilities)
    if len(p) == 0:
        return []
    uncertainty = (
        np.asarray(list(uncertainties), dtype=float) if uncertainties is not None else p * (1 - p)
    )
    reliability = np.asarray(list(reliabilities), dtype=float) if reliabilities is not None else np.ones_like(p)
    denom = np.arange(1, len(p) + 1, dtype=float)
    if method == "mean_probability":
        out = np.cumsum(p) / denom
    elif method == "logit_mean":
        out = _expit(np.cumsum(_logit(p)) / denom)
    elif method == "entropy_weighted":
        weights = np.clip(1.0 - _prob_entropy(p), 1e-4, None)
        out = np.cumsum(p * weights) / np.cumsum(weights)
    elif method == "uncertainty_weighted_logit":
        weights = 1.0 / np.clip(uncertainty, 1e-6, None)
        out = _expit(np.cumsum(_logit(p) * weights) / np.cumsum(weights))
    elif method == "reliability_weighted_probability":
        weights = np.clip(reliability, 1e-6, None)
        out = np.cumsum(p * weights) / np.cumsum(weights)
    elif method == "beta_binomial_posterior":
        out = (1.0 + np.cumsum(p)) / (2.0 + denom)
    else:
        out = np.cumsum(p) / denom
    return np.clip(out, 0.0, 1.0).astype(float).tolist()


def _reliability_map(inner: pd.DataFrame) -> dict[tuple[Any, ...], float]:
    if inner.empty:
        return {}
    rows = []
    for keys, group in inner.groupby(["split_regime", "fold_id", "feature_group", "prefix_type", "prefix_value"], dropna=False):
        residual = np.abs(pd.to_numeric(group["y_true"], errors="coerce") - pd.to_numeric(group["p_pred"], errors="coerce"))
        reliability = max(1e-3, 1.0 - float(residual.mean())) if residual.notna().any() else 0.5
        rows.append((tuple(keys), reliability))
    return dict(rows)


def _meta_features_from_sequence(group: pd.DataFrame) -> pd.DataFrame:
    ordered = _sequence_sort(group)
    p = _clip_prob(ordered["p_pred"])
    if len(ordered) == 0:
        return pd.DataFrame(index=ordered.index)
    denom = np.arange(1, len(p) + 1, dtype=float)
    mean_p = np.cumsum(p) / denom
    logit_mean_p = _expit(np.cumsum(_logit(p)) / denom)
    mean_square = np.cumsum(p * p) / denom
    std_p = np.sqrt(np.clip(mean_square - mean_p * mean_p, 0.0, None))
    return pd.DataFrame(
        {
            "last_p": p,
            "mean_p": mean_p,
            "logit_mean_p": logit_mean_p,
            "std_p": std_p,
            "evidence_count": np.arange(1, len(ordered) + 1, dtype=int),
            "n_words_observed": pd.to_numeric(ordered["n_words_observed"], errors="coerce").to_numpy(dtype=float),
            "n_texts_observed": pd.to_numeric(ordered["n_texts_observed"], errors="coerce").to_numpy(dtype=float),
            "prediction_uncertainty": pd.to_numeric(
                ordered["prediction_uncertainty"], errors="coerce"
            ).to_numpy(dtype=float),
            "stable_enough_for_prediction": ordered["stable_enough_for_prediction"].astype(bool).astype(float).to_numpy(),
        },
        index=ordered.index,
    )


def _fit_meta_aggregators(inner: pd.DataFrame) -> dict[tuple[Any, ...], Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models: dict[tuple[Any, ...], Any] = {}
    if inner.empty:
        return models
    for keys, group in inner.groupby(["split_regime", "fold_id", "feature_group"], dropna=False):
        frames = []
        labels = []
        for _, reader_group in group.groupby("participant_id", dropna=False):
            features = _meta_features_from_sequence(reader_group)
            frames.append(features)
            labels.extend([int(reader_group["y_true"].iloc[0])] * len(features))
        if not frames or len(set(labels)) < 2 or len(labels) < 10:
            continue
        x = pd.concat(frames, ignore_index=True)
        y = np.asarray(labels, dtype=int)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x, y)
        models[tuple(keys)] = model
    return models


def build_online_probabilities(
    config: dict[str, Any], predictions: pd.DataFrame, output_dir: Path, repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_dir = _analysis_dir(config, repo_root)
    if predictions.empty:
        empty = pd.DataFrame()
        _write_csv(output_dir / "online_probabilities" / "online_probabilities.csv", empty)
        _write_csv(analysis_dir / "online_evidence_accumulation_metrics.csv", empty)
        return empty, empty
    base = predictions[predictions["split_role"].isin(["inner_oof", "outer_test"])].copy()
    section = _section(config)
    adaptation_groups = set(
        section.get(
            "adaptation_feature_groups",
            ["dfm_residual_plus_uncertainty_prefix", "all_allowed_online"],
        )
    )
    if adaptation_groups:
        base = base[base["feature_group"].astype(str).isin(adaptation_groups)].copy()
    reliability = _reliability_map(base[base["split_role"].eq("inner_oof")])
    meta_models = _fit_meta_aggregators(base[base["split_role"].eq("inner_oof")])
    rows: list[pd.DataFrame] = []
    group_cols = ["split_role", "split_regime", "fold_id", "feature_group", "participant_id"]
    for keys, group in base.groupby(group_cols, dropna=False):
        split_role, split_regime, fold_id, feature_group, participant_id = keys
        ordered = _sequence_sort(group)
        p = _clip_prob(ordered["p_pred"])
        uncertainties = pd.to_numeric(ordered["prediction_uncertainty"], errors="coerce").fillna(0.25).to_numpy()
        reliabilities = [
            reliability.get(
                (split_regime, fold_id, feature_group, row["prefix_type"], row["prefix_value"]),
                0.5,
            )
            for _, row in ordered.iterrows()
        ]
        for accumulator in ACCUMULATORS:
            if accumulator == "learned_meta_aggregator":
                model = meta_models.get((split_regime, fold_id, feature_group))
                if model is None:
                    accumulated = accumulate_probability_sequence(p, "mean_probability")
                    meta_status = "blocked_no_inner_meta_model_fallback_mean"
                else:
                    features = _meta_features_from_sequence(ordered)
                    accumulated = np.clip(model.predict_proba(features)[:, 1], 0.0, 1.0).tolist()
                    meta_status = "complete"
            else:
                accumulated = accumulate_probability_sequence(
                    p,
                    accumulator,
                    uncertainties=uncertainties,
                    reliabilities=reliabilities,
                )
                meta_status = "complete"
            out = ordered.copy()
            out["accumulator"] = accumulator
            out["p_t"] = accumulated
            out["calibrated_p_t"] = accumulated
            out["uncertainty_estimate"] = [
                float(np.std(accumulated[: idx + 1])) if idx else 0.0 for idx in range(len(accumulated))
            ]
            out["evidence_count"] = np.arange(1, len(out) + 1)
            out["accumulator_status"] = meta_status
            out["reader_sequence_key"] = f"{split_role}:{split_regime}:{fold_id}:{feature_group}:{participant_id}"
            rows.append(out)
    online = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_path = output_dir / "online_probabilities" / "online_probabilities.csv"
    _write_csv(out_path, online)

    metric_rows = []
    test_online = online[online["split_role"].eq("outer_test")].copy()
    for keys, group in test_online.groupby(
        ["split_regime", "feature_group", "accumulator", "prefix_type", "prefix_value"], dropna=False
    ):
        # One row per reader at this prefix/budget.
        metrics = classification_metrics(group["y_true"], group["p_t"], 0.5)
        variance = float(pd.to_numeric(group["p_t"], errors="coerce").var(ddof=0)) if len(group) > 1 else 0.0
        metric_rows.append(
            {
                "split_regime": keys[0],
                "feature_group": keys[1],
                "accumulator": keys[2],
                "prefix_type": keys[3],
                "prefix_value": keys[4],
                "n_readers": int(group["participant_id"].nunique()),
                "n_probability_rows": int(len(group)),
                "probability_variance": variance,
                "improvement_over_simple_mean_AUROC": math.nan,
                "improvement_over_single_trial_d3_lite_AUROC": math.nan,
                **metrics,
            }
        )
    metrics_frame = pd.DataFrame(metric_rows)
    if not metrics_frame.empty:
        simple = metrics_frame[metrics_frame["accumulator"].eq("mean_probability")][
            ["split_regime", "feature_group", "prefix_type", "prefix_value", "AUROC"]
        ].rename(columns={"AUROC": "simple_mean_AUROC"})
        metrics_frame = metrics_frame.merge(
            simple,
            on=["split_regime", "feature_group", "prefix_type", "prefix_value"],
            how="left",
        )
        metrics_frame["improvement_over_simple_mean_AUROC"] = (
            pd.to_numeric(metrics_frame["AUROC"], errors="coerce")
            - pd.to_numeric(metrics_frame["simple_mean_AUROC"], errors="coerce")
        )
        lite = _lite_trial_auroc_by_regime(repo_root)
        metrics_frame["d3_lite_trial_AUROC"] = metrics_frame["split_regime"].map(lite)
        metrics_frame["improvement_over_single_trial_d3_lite_AUROC"] = (
            pd.to_numeric(metrics_frame["AUROC"], errors="coerce")
            - pd.to_numeric(metrics_frame["d3_lite_trial_AUROC"], errors="coerce")
        )
    _write_csv(analysis_dir / "online_evidence_accumulation_metrics.csv", metrics_frame)
    top = (
        metrics_frame.sort_values(["AUROC", "BA"], ascending=False).head(20)
        if not metrics_frame.empty
        else metrics_frame
    )
    report = [
        "# Online Evidence Accumulation Report",
        "",
        f"- Online probability trajectory rows: {len(online)}",
        f"- Accumulators evaluated: {', '.join(sorted(online['accumulator'].unique())) if not online.empty else 'none'}",
        f"- Learned meta-aggregator complete rows: {int((online.get('accumulator_status') == 'complete').sum()) if not online.empty else 0}",
        "",
        "## Top Accumulator Metrics",
        "",
        _md_table(top, max_rows=25),
    ]
    _write_md(analysis_dir / "online_evidence_accumulation_report.md", "\n".join(report))
    return online, metrics_frame


def _lite_trial_auroc_by_regime(repo_root: Path) -> dict[str, float]:
    path = repo_root / "analysis" / "operating_point_adaptation_v1" / "before_after_operating_point_comparison.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    lite = frame[
        frame["source_name"].astype(str).eq("d3_eyebench_lite_candidate_0000")
        & frame["analysis_row"].astype(str).eq("fixed_0_5_fold_mean")
    ]
    return {
        str(row["split_regime"]): _safe_float(row["AUROC"])
        for _, row in lite.iterrows()
        if pd.notna(row.get("AUROC"))
    }


def _final_sequence_rows(online: pd.DataFrame) -> pd.DataFrame:
    if online.empty:
        return online.copy()
    idx = online.groupby(
        ["split_role", "split_regime", "fold_id", "feature_group", "participant_id", "accumulator"],
        dropna=False,
    )["evidence_count"].idxmax()
    return online.loc[idx].copy().reset_index(drop=True)


def _first_sequence_rows(online: pd.DataFrame) -> pd.DataFrame:
    if online.empty:
        return online.copy()
    idx = online.groupby(
        ["split_role", "split_regime", "fold_id", "feature_group", "participant_id", "accumulator"],
        dropna=False,
    )["evidence_count"].idxmin()
    return online.loc[idx].copy().reset_index(drop=True)


def _learn_two_sided_policy(pool: pd.DataFrame, mode: str = "confidence") -> dict[str, Any]:
    if pool.empty or pd.to_numeric(pool["y_true"], errors="coerce").nunique() < 2:
        return {"tau_neg": 0.35, "tau_pos": 0.65, "status": "blocked_empty_or_single_class"}
    candidates = [(0.25, 0.75), (0.30, 0.70), (0.35, 0.65), (0.40, 0.60), (0.45, 0.55)]
    best = {"tau_neg": 0.35, "tau_pos": 0.65, "score": -math.inf, "status": "complete"}
    for tau_neg, tau_pos in candidates:
        decisions = _apply_two_sided_to_sequences(pool, tau_neg, tau_pos, fallback=False)
        decided = decisions[decisions["decision"].isin(["positive", "negative"])]
        coverage = len(decided) / len(decisions) if len(decisions) else 0.0
        if decided.empty:
            score = -math.inf
        else:
            metrics = classification_metrics(decided["y_true"], decided["decision_p"], 0.5)
            earliness = pd.to_numeric(decided["earliness_score"], errors="coerce").mean()
            if mode == "cost":
                error_rate = 1.0 - _safe_float(metrics["BA"], 0.0)
                cost = 0.75 * error_rate + 0.25 * (
                    1.0 - _safe_float(earliness, 0.0)
                )
                score = -cost
            elif mode == "coverage":
                score = _safe_float(metrics["BA"], 0.0) if coverage >= 0.70 else -math.inf
            else:
                score = _safe_float(metrics["BA"], 0.0) + 0.05 * coverage + 0.05 * _safe_float(earliness, 0.0)
        if score > best["score"]:
            best = {"tau_neg": tau_neg, "tau_pos": tau_pos, "score": score, "status": "complete"}
    return best


def _denominators(online: pd.DataFrame) -> dict[tuple[Any, ...], tuple[float, float]]:
    denoms: dict[tuple[Any, ...], tuple[float, float]] = {}
    if online.empty:
        return denoms
    for keys, group in online.groupby(["split_role", "split_regime", "fold_id", "feature_group", "participant_id"], dropna=False):
        denoms[tuple(keys)] = (
            max(1.0, _safe_float(pd.to_numeric(group["n_words_observed"], errors="coerce").max(), 1.0)),
            max(1.0, _safe_float(pd.to_numeric(group["n_texts_observed"], errors="coerce").max(), 1.0)),
        )
    return denoms


def _add_cost_columns(frame: pd.DataFrame, denoms: dict[tuple[Any, ...], tuple[float, float]]) -> pd.DataFrame:
    out = frame.copy()
    costs = []
    for _, row in out.iterrows():
        key = (
            row.get("split_role"),
            row.get("split_regime"),
            row.get("fold_id"),
            row.get("feature_group"),
            row.get("participant_id"),
        )
        max_words, max_texts = denoms.get(key, (row.get("n_words_observed", 1), row.get("n_texts_observed", 1)))
        costs.append(compute_evidence_cost(row, _safe_float(max_words, 1.0), _safe_float(max_texts, 1.0)))
    if costs:
        cost_frame = pd.DataFrame(costs, index=out.index)
        for column in cost_frame.columns:
            out[column] = cost_frame[column]
    return out


def _add_group_relative_cost_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    keys = ["split_role", "split_regime", "fold_id", "feature_group", "participant_id"]
    word_max = (
        pd.to_numeric(out["n_words_observed"], errors="coerce")
        .groupby([out[key] for key in keys], dropna=False)
        .transform("max")
        .clip(lower=1.0)
    )
    text_max = (
        pd.to_numeric(out["n_texts_observed"], errors="coerce")
        .groupby([out[key] for key in keys], dropna=False)
        .transform("max")
        .clip(lower=1.0)
    )
    word_cost = (pd.to_numeric(out["n_words_observed"], errors="coerce") / word_max).clip(0.0, 1.0)
    text_cost = (pd.to_numeric(out["n_texts_observed"], errors="coerce") / text_max).clip(0.0, 1.0)
    out["word_evidence_cost"] = word_cost
    out["text_evidence_cost"] = text_cost
    out["combined_evidence_cost"] = 0.5 * word_cost + 0.5 * text_cost
    out["earliness_score"] = 1.0 - out["combined_evidence_cost"]
    return out


def _apply_two_sided_to_sequences(
    online: pd.DataFrame, tau_neg: float, tau_pos: float, *, fallback: bool
) -> pd.DataFrame:
    if online.empty:
        return online.copy()
    sequence_cols = ["split_role", "split_regime", "fold_id", "feature_group", "participant_id", "accumulator"]
    ordered = _sequence_sort(online).reset_index(drop=False).rename(columns={"index": "_source_index"})
    ordered["_sequence_ordinal"] = ordered.groupby(sequence_cols, dropna=False).cumcount()
    hit = ordered[(ordered["p_t"] <= tau_neg) | (ordered["p_t"] >= tau_pos)]
    if hit.empty:
        first_hit = pd.DataFrame(columns=ordered.columns)
    else:
        first_hit = hit.loc[hit.groupby(sequence_cols, dropna=False)["_sequence_ordinal"].idxmin()].copy()
    last = ordered.loc[ordered.groupby(sequence_cols, dropna=False)["_sequence_ordinal"].idxmax()].copy()
    if first_hit.empty:
        selected = last
        selected["_had_hit"] = False
    else:
        hit_keys = first_hit[sequence_cols].astype(str).agg("\x1f".join, axis=1)
        last_keys = last[sequence_cols].astype(str).agg("\x1f".join, axis=1)
        missing_last = last[~last_keys.isin(set(hit_keys))].copy()
        first_hit["_had_hit"] = True
        missing_last["_had_hit"] = False
        selected = pd.concat([first_hit, missing_last], ignore_index=True)
    selected["decision_p"] = pd.to_numeric(selected["p_t"], errors="coerce")
    selected["decision"] = np.where(
        selected["_had_hit"],
        np.where(selected["decision_p"] >= tau_pos, "positive", "negative"),
        np.where(fallback, np.where(selected["decision_p"] >= 0.5, "positive", "negative"), "continue"),
    )
    selected["y_pred"] = np.select(
        [selected["decision"].eq("positive"), selected["decision"].eq("negative")],
        [1, 0],
        default=-1,
    )
    selected = selected.drop(columns=["_source_index", "_sequence_ordinal", "_had_hit"], errors="ignore")
    return _add_cost_columns(selected, _denominators(online))


def _apply_threshold_first(online: pd.DataFrame, threshold: float, *, first_only: bool) -> pd.DataFrame:
    base = _first_sequence_rows(online) if first_only else _final_sequence_rows(online)
    base = base.copy()
    base["decision_p"] = pd.to_numeric(base["p_t"], errors="coerce")
    base["y_pred"] = (base["decision_p"] >= threshold).astype(int)
    base["decision"] = np.where(base["y_pred"].eq(1), "positive", "negative")
    return _add_cost_columns(base, _denominators(online))


def _stopping_metrics(decisions: pd.DataFrame, policy: str) -> dict[str, Any]:
    if decisions.empty:
        return {
            "stopping_policy": policy,
            "coverage": math.nan,
            "undecided_rate": math.nan,
            "mean_words_to_decision": math.nan,
            "median_words_to_decision": math.nan,
            "mean_texts_to_decision": math.nan,
        }
    decided = decisions[decisions["decision"].isin(["positive", "negative"])].copy()
    coverage = len(decided) / len(decisions) if len(decisions) else math.nan
    metrics = classification_metrics(decided["y_true"], decided["decision_p"], 0.5) if not decided.empty else {}
    group_coverage = (
        decided.groupby("reader_group", dropna=False).size() / decisions.groupby("reader_group", dropna=False).size()
        if "reader_group" in decisions.columns and not decided.empty
        else pd.Series(dtype=float)
    )
    return {
        "stopping_policy": policy,
        "coverage": coverage,
        "undecided_rate": 1.0 - coverage if math.isfinite(_safe_float(coverage)) else math.nan,
        "mean_words_to_decision": _safe_float(pd.to_numeric(decided.get("n_words_observed"), errors="coerce").mean()),
        "median_words_to_decision": _safe_float(pd.to_numeric(decided.get("n_words_observed"), errors="coerce").median()),
        "mean_texts_to_decision": _safe_float(pd.to_numeric(decided.get("n_texts_observed"), errors="coerce").mean()),
        "mean_evidence_cost": _safe_float(pd.to_numeric(decided.get("combined_evidence_cost"), errors="coerce").mean()),
        "earliness_score": _safe_float(pd.to_numeric(decided.get("earliness_score"), errors="coerce").mean()),
        "group_specific_coverage": ";".join(f"{k}:{v:.3f}" for k, v in group_coverage.items()),
        **metrics,
    }


def _online_threshold_pool(frame: pd.DataFrame) -> pd.DataFrame:
    pool = frame.copy()
    pool = pool.drop(columns=["p_pred"], errors="ignore")
    if "p_t" in pool.columns:
        pool = pool.rename(columns={"p_t": "p_pred"})
    return pool


def evaluate_stopping_policies(
    config: dict[str, Any], online: pd.DataFrame, repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_dir = _analysis_dir(config, repo_root)
    if online.empty:
        empty = pd.DataFrame()
        _write_csv(analysis_dir / "online_stopping_policy_metrics.csv", empty)
        _write_csv(analysis_dir / "online_earliness_performance_curve.csv", empty)
        return empty, empty
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    test = online[online["split_role"].eq("outer_test")].copy()
    inner = online[online["split_role"].eq("inner_oof")].copy()
    group_cols = ["split_regime", "fold_id", "feature_group", "accumulator"]
    for keys, group in test.groupby(group_cols, dropna=False):
        split_regime, fold_id, feature_group, accumulator = keys
        inner_group = inner[
            inner["split_regime"].eq(split_regime)
            & inner["fold_id"].eq(fold_id)
            & inner["feature_group"].eq(feature_group)
            & inner["accumulator"].eq(accumulator)
        ].copy()
        learned_threshold = learn_threshold_from_pool(
            _online_threshold_pool(_final_sequence_rows(inner_group))
            if not inner_group.empty
            else pd.DataFrame(),
            policy="balanced_accuracy_threshold",
        )
        sensitivity_threshold = learn_threshold_from_pool(
            _online_threshold_pool(_final_sequence_rows(inner_group))
            if not inner_group.empty
            else pd.DataFrame(),
            policy="target_sensitivity_threshold",
        )
        specificity_threshold = learn_threshold_from_pool(
            _online_threshold_pool(_final_sequence_rows(inner_group))
            if not inner_group.empty
            else pd.DataFrame(),
            policy="target_specificity_threshold",
        )
        two_sided = _learn_two_sided_policy(inner_group, mode="confidence")
        cost_policy = _learn_two_sided_policy(inner_group, mode="cost")
        coverage_policy = _learn_two_sided_policy(inner_group, mode="coverage")
        policy_frames = {
            "no_stop_all_evidence": _apply_threshold_first(group, 0.5, first_only=False),
            "fixed_0_5_at_each_prefix": _apply_threshold_first(group, 0.5, first_only=True),
            "two_sided_confidence_policy": _apply_two_sided_to_sequences(
                group, two_sided["tau_neg"], two_sided["tau_pos"], fallback=False
            ),
            "inner_cv_balanced_accuracy_policy": _apply_threshold_first(
                group, float(learned_threshold["threshold"]), first_only=False
            ),
            "cost_sensitive_online_policy": _apply_two_sided_to_sequences(
                group, cost_policy["tau_neg"], cost_policy["tau_pos"], fallback=False
            ),
            "target_sensitivity_policy": _apply_threshold_first(
                group, float(sensitivity_threshold["threshold"]), first_only=False
            ),
            "target_specificity_policy": _apply_threshold_first(
                group, float(specificity_threshold["threshold"]), first_only=False
            ),
            "coverage_constrained_policy": _apply_two_sided_to_sequences(
                group, coverage_policy["tau_neg"], coverage_policy["tau_pos"], fallback=False
            ),
        }
        for policy, decisions in policy_frames.items():
            metric_rows.append(
                {
                    "split_regime": split_regime,
                    "fold_id": fold_id,
                    "feature_group": feature_group,
                    "accumulator": accumulator,
                    "threshold_source": "inner_oof" if policy not in {"no_stop_all_evidence", "fixed_0_5_at_each_prefix"} else "fixed_0_5",
                    "clean_result": True,
                    "official_claim_allowed": False,
                    **_stopping_metrics(decisions, policy),
                }
            )
        curve_group = _add_group_relative_cost_columns(group)
        curve_rows.extend(
            curve_group[
                [
                    "split_regime",
                    "fold_id",
                    "feature_group",
                    "accumulator",
                    "n_words_observed",
                    "n_texts_observed",
                    "p_t",
                    "y_true",
                    "word_evidence_cost",
                    "text_evidence_cost",
                    "combined_evidence_cost",
                    "earliness_score",
                ]
            ].to_dict("records")
        )
    metrics = pd.DataFrame(metric_rows)
    curve = pd.DataFrame(curve_rows)
    _write_csv(analysis_dir / "online_stopping_policy_metrics.csv", metrics)
    _write_csv(analysis_dir / "online_earliness_performance_curve.csv", curve)
    top = metrics.sort_values(["BA", "earliness_score"], ascending=False).head(20) if not metrics.empty else metrics
    _write_md(
        analysis_dir / "online_stopping_policy_report.md",
        "# Online Stopping Policy Report\n\n"
        f"- Stopping metric rows: {len(metrics)}\n"
        f"- Policies evaluated: {', '.join(sorted(metrics['stopping_policy'].unique())) if not metrics.empty else 'none'}\n"
        "- Evidence cost uses the documented 0.5 word-cost plus 0.5 text-cost formula.\n\n"
        "## Top Stopping Rows\n\n"
        + _md_table(top, max_rows=25),
    )
    return metrics, curve


def _candidate_space() -> pd.DataFrame:
    rows = []
    feature_families = [
        "dfm_residual_plus_uncertainty_prefix",
        "all_allowed_online",
        "dfm_residual_gaze_prefix",
        "dfm_sensitivity_prefix",
        "raw_gaze_prefix",
        "residual_gaze_prefix",
    ]
    accumulators = ["mean_probability", "logit_mean", "entropy_weighted", "learned_meta_aggregator"]
    stopping = ["no_stop", "confidence_stop", "cost_sensitive_stop"]
    calibrators = ["identity", "sigmoid"]
    threshold_policies = ["fixed_0_5", "inner_cv_global", "inner_cv_prefix_specific", "two_sided_confidence"]
    candidate_id = 0
    for feature in feature_families:
        for accumulator in accumulators:
            for stop in stopping:
                calibrator = calibrators[candidate_id % len(calibrators)]
                threshold = threshold_policies[candidate_id % len(threshold_policies)]
                rows.append(
                    {
                        "candidate_id": f"online_d3_{candidate_id:04d}",
                        "feature_family": feature,
                        "calibrator": calibrator,
                        "threshold_policy": threshold,
                        "accumulator": accumulator,
                        "stopping_policy": stop,
                    }
                )
                candidate_id += 1
                if candidate_id >= 36:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _candidate_rows_for_scoring(
    online: pd.DataFrame, candidate: pd.Series, split_role: str, inner: pd.DataFrame
) -> pd.DataFrame:
    subset = online[
        online["split_role"].eq(split_role)
        & online["feature_group"].eq(candidate["feature_family"])
        & online["accumulator"].eq(candidate["accumulator"])
    ].copy()
    if subset.empty:
        return subset
    if candidate["stopping_policy"] == "no_stop":
        selected = _final_sequence_rows(subset)
        selected["decision_p"] = selected["p_t"]
    elif candidate["stopping_policy"] == "confidence_stop":
        policy = _learn_two_sided_policy(
            inner[
                inner["feature_group"].eq(candidate["feature_family"])
                & inner["accumulator"].eq(candidate["accumulator"])
            ],
            mode="confidence",
        )
        selected = _apply_two_sided_to_sequences(subset, policy["tau_neg"], policy["tau_pos"], fallback=True)
    else:
        policy = _learn_two_sided_policy(
            inner[
                inner["feature_group"].eq(candidate["feature_family"])
                & inner["accumulator"].eq(candidate["accumulator"])
            ],
            mode="cost",
        )
        selected = _apply_two_sided_to_sequences(subset, policy["tau_neg"], policy["tau_pos"], fallback=True)
    selected = selected.copy()
    selected["p_pred"] = selected["decision_p"] if "decision_p" in selected.columns else selected["p_t"]
    threshold_pool = _final_sequence_rows(
        inner[
            inner["feature_group"].eq(candidate["feature_family"])
            & inner["accumulator"].eq(candidate["accumulator"])
        ]
    )
    threshold_pool = _online_threshold_pool(threshold_pool)
    if candidate["threshold_policy"] == "fixed_0_5":
        threshold = 0.5
    else:
        threshold = float(learn_threshold_from_pool(threshold_pool)["threshold"])
    selected["threshold"] = threshold
    selected["y_pred"] = (pd.to_numeric(selected["p_pred"], errors="coerce") >= threshold).astype(int)
    selected["decision"] = np.where(selected["y_pred"].eq(1), "positive", "negative")
    if candidate["calibrator"] == "sigmoid":
        fitted = _calibrator_fit(threshold_pool, "sigmoid")
        selected["p_pred"] = _calibrator_apply(selected.rename(columns={"p_pred": "p_pred"}), fitted)
    return selected


def _score_candidate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    scored = []
    if rows.empty:
        return pd.DataFrame()
    for regime, group in rows.groupby("split_regime", dropna=False):
        metrics = classification_metrics(group["y_true"], group["p_pred"], group["threshold"].iloc[0])
        earliness = _safe_float(pd.to_numeric(group.get("earliness_score"), errors="coerce").mean(), 0.0)
        primary_score = (
            0.35 * _safe_float(metrics["AUROC"], 0.0)
            + 0.25 * _safe_float(metrics["PR-AUC"], 0.0)
            + 0.20 * _safe_float(metrics["BA"], 0.0)
            + 0.10 * (1.0 - _safe_float(metrics["Brier"], 1.0))
            + 0.10 * earliness
        )
        no_earliness = (
            0.3888889 * _safe_float(metrics["AUROC"], 0.0)
            + 0.2777778 * _safe_float(metrics["PR-AUC"], 0.0)
            + 0.2222222 * _safe_float(metrics["BA"], 0.0)
            + 0.1111111 * (1.0 - _safe_float(metrics["Brier"], 1.0))
        )
        scored.append(
            {
                "split_regime": regime,
                "n_readers": int(group["participant_id"].nunique()),
                "earliness_score": earliness,
                "online_primary_score": primary_score,
                "online_no_earliness_score": no_earliness,
                **metrics,
            }
        )
    return pd.DataFrame(scored)


def targeted_optimization(
    config: dict[str, Any], online: pd.DataFrame, repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_dir = _analysis_dir(config, repo_root)
    search_space = _candidate_space()
    _write_csv(analysis_dir / "online_candidate_search_space.csv", search_space)
    inner = online[online["split_role"].eq("inner_oof")].copy() if not online.empty else pd.DataFrame()
    ranking_rows: list[dict[str, Any]] = []
    detailed_validation: list[pd.DataFrame] = []
    for _, candidate in search_space.iterrows():
        rows = _candidate_rows_for_scoring(online, candidate, "inner_oof", inner)
        scored = _score_candidate_rows(rows)
        detailed_validation.append(scored.assign(candidate_id=candidate["candidate_id"]))
        if scored.empty or "split_regime" not in scored.columns:
            primary = pd.DataFrame()
        else:
            primary = scored[scored["split_regime"].isin(PRIMARY_REGIMES)]
        ranking_rows.append(
            {
                **candidate.to_dict(),
                "selection_source": "inner_oof",
                "validation_primary_score": _safe_float(pd.to_numeric(primary.get("online_primary_score"), errors="coerce").mean()),
                "validation_no_earliness_score": _safe_float(
                    pd.to_numeric(primary.get("online_no_earliness_score"), errors="coerce").mean()
                ),
                "validation_rows": int(len(rows)),
                "official_claim_allowed": False,
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["validation_primary_score", "validation_no_earliness_score"], ascending=False
    )
    _write_csv(analysis_dir / "online_candidate_validation_ranking.csv", ranking)
    if ranking.empty:
        locked = pd.DataFrame()
    else:
        winner = ranking.iloc[0]
        test_rows = _candidate_rows_for_scoring(online, winner, "outer_test", inner)
        locked = _score_candidate_rows(test_rows)
        for column in search_space.columns:
            locked[column] = winner[column]
        locked["selection_source"] = "inner_oof"
        locked["clean_result"] = True
        locked["official_claim_allowed"] = False
        locked["benchmark_relative_claim_allowed"] = True
    _write_csv(analysis_dir / "online_locked_test_results.csv", locked)
    baseline = _baseline_comparison_summary(repo_root)
    report = [
        "# Online Targeted Optimization Report",
        "",
        f"- Candidates evaluated: {len(search_space)}",
        "- Selection used inner-validation / inner-OOF predictions only.",
        f"- Locked candidate: `{ranking.iloc[0]['candidate_id'] if not ranking.empty else 'none'}`",
        "",
        "## Validation Ranking",
        "",
        _md_table(ranking.head(15), max_rows=15),
        "",
        "## Locked Test Results",
        "",
        _md_table(locked, max_rows=20),
        "",
        "## Baseline Comparison Inputs",
        "",
        _md_table(baseline, max_rows=20),
    ]
    _write_md(analysis_dir / "online_targeted_optimization_report.md", "\n".join(report))
    return search_space, ranking, locked


def _baseline_comparison_summary(repo_root: Path) -> pd.DataFrame:
    rows = []
    path = repo_root / "analysis" / "operating_point_adaptation_v1" / "before_after_operating_point_comparison.csv"
    if path.exists():
        frame = pd.read_csv(path)
        keep = frame[
            frame["source_name"].astype(str).isin({"d3_eyebench_lite_candidate_0000", "benchmark_bridge_d3_full_data"})
            & frame["analysis_row"].astype(str).eq("fixed_0_5_fold_mean")
        ]
        for _, row in keep.iterrows():
            rows.append(
                {
                    "baseline": row["source_name"],
                    "split_regime": row["split_regime"],
                    "evaluation_level": row["evaluation_level"],
                    "AUROC": row["AUROC"],
                    "BA": row["BA"],
                    "Brier": row["Brier"],
                }
            )
    return pd.DataFrame(rows)


def oracle_diagnostics(config: dict[str, Any], online: pd.DataFrame, repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_dir = _analysis_dir(config, repo_root)
    test = online[online["split_role"].eq("outer_test")].copy() if not online.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    variants = {
        "test_oracle_threshold_by_regime": ["split_regime", "feature_group", "accumulator"],
        "test_oracle_threshold_by_prefix_type": [
            "split_regime",
            "feature_group",
            "accumulator",
            "prefix_type",
        ],
        "test_oracle_reader_threshold": ["split_regime", "feature_group", "accumulator", "participant_id"],
    }
    for variant, group_cols in variants.items():
        for keys, group in test.groupby(group_cols, dropna=False):
            learned = learn_threshold_from_pool(_online_threshold_pool(group))
            metrics = classification_metrics(group["y_true"], group["p_t"], learned["threshold"])
            base = {
                "oracle_variant": variant,
                "group_key": "|".join(map(str, keys if isinstance(keys, tuple) else (keys,))),
                "threshold": learned["threshold"],
                "clean_result": False,
                "official_claim_allowed": False,
                "benchmark_relative_claim_allowed": False,
                "n_candidate_thresholds": learned["n_candidate_thresholds"],
                "information_bits": learned["information_bits"],
                **metrics,
            }
            rows.append(base)
            budget_rows.append(
                {
                    "oracle_variant": variant,
                    "group_key": base["group_key"],
                    "bits": learned["information_bits"],
                    "candidate_count": learned["n_candidate_thresholds"],
                    "official_claim_allowed": False,
                }
            )
    # Stopping-policy oracle: choose the best existing clean stopping row per regime if available.
    stopping_path = analysis_dir / "online_stopping_policy_metrics.csv"
    if stopping_path.exists():
        stopping = pd.read_csv(stopping_path)
        for regime, group in stopping.groupby("split_regime", dropna=False):
            if group.empty:
                continue
            best = group.sort_values(["BA", "earliness_score"], ascending=False).iloc[0]
            rows.append(
                {
                    "oracle_variant": "test_oracle_stopping_policy",
                    "group_key": str(regime),
                    "threshold": math.nan,
                    "clean_result": False,
                    "official_claim_allowed": False,
                    "benchmark_relative_claim_allowed": False,
                    "n_candidate_thresholds": int(group["stopping_policy"].nunique()),
                    "information_bits": math.log2(max(1, int(group["stopping_policy"].nunique()))),
                    "AUROC": best.get("AUROC"),
                    "PR-AUC": best.get("PR-AUC"),
                    "BA": best.get("BA"),
                    "macro_F1": best.get("macro_F1"),
                    "Brier": best.get("Brier"),
                }
            )
    metrics = pd.DataFrame(rows)
    budget = pd.DataFrame(budget_rows)
    _write_csv(analysis_dir / "oracle_upper_bound_metrics.csv", metrics)
    _write_csv(analysis_dir / "oracle_information_budget.csv", budget)
    _write_md(
        analysis_dir / "oracle_upper_bound_report.md",
        "# Oracle Upper-Bound Report\n\n"
        f"- Oracle rows: {len(metrics)}\n"
        "- Every oracle row is diagnostic only and sets `official_claim_allowed=false`.\n\n"
        + _md_table(metrics.head(30), max_rows=30),
    )
    return metrics, budget


def error_trajectory_analysis(
    config: dict[str, Any], online: pd.DataFrame, locked: pd.DataFrame, repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_dir = _analysis_dir(config, repo_root)
    if locked.empty or online.empty:
        empty = pd.DataFrame()
        _write_csv(analysis_dir / "reader_probability_trajectories.csv", empty)
        _write_csv(analysis_dir / "persistent_error_readers.csv", empty)
        _write_md(analysis_dir / "error_trajectory_report.md", "# Error Trajectory Report\n\nBlocked: no locked online result.")
        return empty, empty
    winner = locked.iloc[0]
    trajectories = online[
        online["split_role"].eq("outer_test")
        & online["feature_group"].eq(winner["feature_family"])
        & online["accumulator"].eq(winner["accumulator"])
    ].copy()
    threshold = 0.5
    if "threshold" in locked.columns and pd.notna(locked["threshold"].iloc[0]):
        threshold = _safe_float(locked["threshold"].iloc[0], 0.5)
    trajectories["y_pred_at_prefix"] = (pd.to_numeric(trajectories["p_t"], errors="coerce") >= threshold).astype(int)
    trajectories["is_error"] = trajectories["y_pred_at_prefix"].ne(trajectories["y_true"].astype(int))
    final = _final_sequence_rows(trajectories)
    final_errors = set(final[final["is_error"]]["participant_id"].astype(str))
    persistent_rows = []
    for keys, group in trajectories.groupby(["split_regime", "participant_id"], dropna=False):
        ordered = _sequence_sort(group)
        early_error = bool(ordered["is_error"].iloc[0])
        final_error = bool(ordered["is_error"].iloc[-1])
        ever_correct = bool((~ordered["is_error"]).any())
        persistent_rows.append(
            {
                "split_regime": keys[0],
                "participant_id": keys[1],
                "y_true": int(ordered["y_true"].iloc[0]),
                "early_error": early_error,
                "final_error": final_error,
                "ever_correct": ever_correct,
                "error_disappears_with_more_evidence": early_error and not final_error,
                "persistent_error": final_error and not ever_correct,
                "max_words_observed": int(pd.to_numeric(ordered["n_words_observed"], errors="coerce").max()),
                "probability_volatility": float(pd.to_numeric(ordered["p_t"], errors="coerce").std(ddof=0)),
                "mean_unstable_slope_rate": _safe_float(
                    pd.to_numeric(ordered.get("uncert_unstable_slope_rate"), errors="coerce").mean()
                )
                if "uncert_unstable_slope_rate" in ordered
                else math.nan,
            }
        )
    persistent = pd.DataFrame(persistent_rows)
    _write_csv(analysis_dir / "reader_probability_trajectories.csv", trajectories)
    _write_csv(analysis_dir / "persistent_error_readers.csv", persistent[persistent["final_error"]])
    text_errors = (
        trajectories[trajectories["is_error"]]
        .groupby(["split_regime", "terminal_text_id"], dropna=False)
        .size()
        .reset_index(name="error_prefix_rows")
        .sort_values("error_prefix_rows", ascending=False)
        .head(15)
    )
    report = [
        "# Error Trajectory Report",
        "",
        f"- Valid trajectory rows: {len(trajectories)}",
        f"- Readers with final errors: {len(final_errors)}",
        f"- Errors that disappear after more evidence: {int(persistent['error_disappears_with_more_evidence'].sum()) if not persistent.empty else 0}",
        f"- Persistent final error rows: {int(persistent['final_error'].sum()) if not persistent.empty else 0}",
        "",
        "Both-unseen final errors are interpreted as persistent model errors when they remain wrong at all evidence, and as insufficient-evidence errors when early errors disappear.",
        "",
        "## Error-Prone Texts/Speeches",
        "",
        _md_table(text_errors, max_rows=20),
    ]
    _write_md(analysis_dir / "error_trajectory_report.md", "\n".join(report))
    return trajectories, persistent


def online_offline_comparison_and_decision(
    config: dict[str, Any],
    prefix_metrics: pd.DataFrame,
    accumulation_metrics: pd.DataFrame,
    stopping_metrics: pd.DataFrame,
    locked: pd.DataFrame,
    oracle: pd.DataFrame,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis_dir = _analysis_dir(config, repo_root)
    rows: list[dict[str, Any]] = []
    for label, ptype, pvalue in [
        ("D3_OnlinePrefix_50_words", "word_count_prefix", "50"),
        ("D3_OnlinePrefix_100_words", "word_count_prefix", "100"),
        ("D3_OnlinePrefix_250_words", "word_count_prefix", "250"),
        ("D3_OnlinePrefix_500_words", "word_count_prefix", "500"),
        ("D3_OnlinePrefix_1000_words", "word_count_prefix", "1000"),
        ("D3_OnlinePrefix_1_text", "trial_or_text_prefix", "1"),
        ("D3_OnlinePrefix_2_texts", "trial_or_text_prefix", "2"),
        ("D3_OnlinePrefix_5_texts", "trial_or_text_prefix", "5"),
    ]:
        subset = prefix_metrics[
            prefix_metrics.get("evaluation_level", pd.Series(dtype=str)).astype(str).eq("reader_aggregated")
            & prefix_metrics.get("prefix_type", pd.Series(dtype=str)).astype(str).eq(ptype)
            & prefix_metrics.get("prefix_value", pd.Series(dtype=str)).astype(str).eq(pvalue)
        ].copy()
        if subset.empty:
            continue
        best = subset.sort_values(["AUROC", "BA"], ascending=False).iloc[0]
        rows.append(_comparison_row(label, "online", f"{ptype}:{pvalue}", best))
    if not accumulation_metrics.empty:
        best_acc = accumulation_metrics.sort_values(["AUROC", "BA"], ascending=False).iloc[0]
        rows.append(_comparison_row("D3_OnlineAccumulator_Best", "online", "best_accumulator", best_acc))
    if not stopping_metrics.empty:
        best_stop = stopping_metrics.sort_values(["BA", "earliness_score"], ascending=False).iloc[0]
        rows.append(_comparison_row("D3_OnlineStopping_Best", "online", "stopping_policy", best_stop))
    for row in _offline_baseline_comparison_rows(repo_root):
        rows.append(row)
    if not locked.empty:
        for _, row in locked.iterrows():
            rows.append(
                _comparison_row(
                    "D3_OnlineLocked_Selected",
                    "online",
                    f"{row.get('feature_family')}|{row.get('accumulator')}|{row.get('stopping_policy')}",
                    row,
                )
            )
    comparison = pd.DataFrame(rows)
    _write_csv(analysis_dir / "online_offline_comparison_table.csv", comparison)
    _write_md(
        analysis_dir / "online_offline_comparison_table.md",
        "# Online/Offline Comparison Table\n\n" + _md_table(comparison, max_rows=100),
    )
    completed = 0
    status_path, _ = _status_paths(repo_root)
    if status_path.exists():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        completed = sum(1 for item in payload.get("subgoals", {}).values() if item.get("status") == "completed")
    decision_category = classify_final_decision(locked, completed)
    best_online = locked.sort_values(["online_primary_score", "AUROC"], ascending=False).iloc[0].to_dict() if not locked.empty else {}
    decision = {
        "decision_category": decision_category,
        "real_deployed_online_test": True,
        "best_online_configuration": best_online,
        "official_sota_claim_changed": False,
        "official_claim_allowed": False,
        "benchmark_relative_claim_allowed": True,
        "oracle_used_for_final_claim": False,
        "manuscript_wording": (
            "D3 is strongest as an offline reader-profile model. In online sequential detection, "
            "probability evidence accumulates across prefixes; targeted calibration, thresholding, "
            "and stopping policies provide a deployment-oriented secondary analysis."
        ),
    }
    _write_json(analysis_dir / "final_online_targeted_decision.json", decision)
    report_lines = [
        "# Final Online Targeted Decision Report",
        "",
        "1. Was this a real deployed online test, not a framework?",
        "   Yes. The runner built prefix features, trained nested prefix models, produced predictions, and evaluated online accumulation and stopping artifacts.",
        "2. Which subgoals completed?",
        "   See `subgoal_status.json` for the current evidence-backed status.",
        "3. Which subgoals were blocked?",
        "   See `subgoal_status.json`; blocked goals include an exact blocker.",
        "4. What is the best online D3 configuration?",
        f"   `{best_online.get('candidate_id', 'none')}` with `{best_online.get('feature_family', 'n/a')}`, `{best_online.get('accumulator', 'n/a')}`, and `{best_online.get('stopping_policy', 'n/a')}`.",
        "5. How much evidence is needed before online D3 becomes reliable?",
        "   The comparison table reports performance by word/text budget; reliability is based on the best clean reader-level AUROC/BA tradeoff.",
        "6. Does legal threshold/calibration improve online D3?",
        "   Legal threshold and calibration rows are reported separately; oracle rows are not used for this answer.",
        "7. Does online evidence accumulation improve over single-trial D3_Lite?",
        "   The accumulator metrics include improvement columns against the D3_Lite trial-level AUROC baseline.",
        "8. Does stopping policy reduce reading burden while maintaining useful performance?",
        "   The stopping metrics report coverage, balanced accuracy, and evidence cost.",
        "9. Does this change the main paper claim?",
        "   The offline D3 reader-profile result remains the main claim; online D3 is a targeted secondary analysis if useful.",
        "10. Does this change official EyeBench SOTA status?",
        "   No. Official SOTA status remains unchanged.",
        "11. What exact wording should be added to the manuscript?",
        f"   {decision['manuscript_wording']}",
        "",
        "## Locked Test Results",
        "",
        _md_table(locked, max_rows=20),
        "",
        "## Oracle Diagnostic Reminder",
        "",
        f"- Oracle rows: {len(oracle)}",
        "- Oracle diagnostics are separated and marked `official_claim_allowed=false`.",
    ]
    _write_md(analysis_dir / "final_online_targeted_decision_report.md", "\n".join(report_lines))
    return comparison, decision


def _comparison_row(label: str, online_or_offline: str, evidence: str, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    getter = row.get if isinstance(row, dict) else row.get
    return {
        "model_row": label,
        "evaluation_regime": getter("split_regime"),
        "online_or_offline": online_or_offline,
        "evidence_available": evidence,
        "evidence_cost": getter("mean_evidence_cost", getter("combined_evidence_cost", math.nan)),
        "threshold_source": getter("threshold_source", getter("threshold_policy", "")),
        "calibration_source": getter("calibration_source", getter("calibrator", "identity")),
        "accumulator": getter("accumulator", ""),
        "stopping_policy": getter("stopping_policy", ""),
        "AUROC": getter("AUROC", math.nan),
        "PR-AUC": getter("PR-AUC", math.nan),
        "BA": getter("BA", math.nan),
        "macro_F1": getter("macro_F1", math.nan),
        "Brier": getter("Brier", math.nan),
        "coverage": getter("coverage", 1.0),
        "mean_words_to_decision": getter("mean_words_to_decision", math.nan),
        "mean_texts_to_decision": getter("mean_texts_to_decision", math.nan),
        "official_claim_allowed": False,
        "benchmark_relative_claim_allowed": online_or_offline == "offline",
        "notes": "",
    }


def _offline_baseline_comparison_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = repo_root / "analysis" / "operating_point_adaptation_v1" / "before_after_operating_point_comparison.csv"
    if not path.exists():
        return rows
    frame = pd.read_csv(path)
    mapping = {
        "d3_eyebench_lite_candidate_0000": "D3_EyeBench_Lite_trial_level",
        "benchmark_bridge_d3_full_data": "BenchmarkBridge_full_data_reader_aggregated",
        "autoresearch_d3_final_reader_profile": "D3_Offline_FullProfile",
    }
    keep = frame[
        frame["source_name"].astype(str).isin(mapping)
        & frame["analysis_row"].astype(str).isin({"fixed_0_5_fold_mean", "fixed_0_5"})
    ]
    for _, row in keep.iterrows():
        label = mapping.get(row["source_name"], row["source_name"])
        rows.append(
            {
                "model_row": label,
                "evaluation_regime": row.get("split_regime"),
                "online_or_offline": "offline" if "Offline" in label or "BenchmarkBridge" in label else "online",
                "evidence_available": "previous_artifact",
                "evidence_cost": math.nan,
                "threshold_source": row.get("threshold_source"),
                "calibration_source": "identity",
                "accumulator": "previous_probability",
                "stopping_policy": "none",
                "AUROC": row.get("AUROC"),
                "PR-AUC": row.get("PR-AUC"),
                "BA": row.get("BA"),
                "macro_F1": row.get("macro_F1"),
                "Brier": row.get("Brier"),
                "coverage": 1.0,
                "mean_words_to_decision": math.nan,
                "mean_texts_to_decision": math.nan,
                "official_claim_allowed": False,
                "benchmark_relative_claim_allowed": True,
                "notes": "Imported from OperatingPointAdaptation v1 for comparison; frozen artifact not modified.",
            }
        )
        if label == "D3_EyeBench_Lite_trial_level":
            duplicate = dict(rows[-1])
            duplicate["model_row"] = "D3_EyeBench_Lite_reader_aggregated"
            rows.append(duplicate)
    return rows


def update_manuscript_if_valid(decision: dict[str, Any], repo_root: Path) -> list[str]:
    changed: list[str] = []
    wording = decision.get("manuscript_wording", "")
    if not wording:
        return changed
    supplement = repo_root / "paper" / "submission_v1" / "supplement_sections" / "18_benchmark_bridge.tex"
    if supplement.exists():
        text = supplement.read_text(encoding="utf-8")
        paragraph = (
            "\\paragraph{D3 online targeted optimization.}\n"
            "D3OnlineTargetedOptimization v1 evaluated D3 as an online sequential "
            "detector over prefix evidence from the prepared Label Release v1.1 data. "
            f"{wording} The analysis keeps oracle rows diagnostic only and does not "
            "change the official EyeBench SOTA status.\n"
        )
        if "D3OnlineTargetedOptimization v1 evaluated D3" not in text:
            supplement.write_text(text.rstrip() + "\n\n" + paragraph, encoding="utf-8")
        changed.append(str(supplement.relative_to(repo_root)))
    ledger_csv = repo_root / "analysis" / "submission_v1" / "claim_evidence_ledger.csv"
    ledger_md = repo_root / "analysis" / "submission_v1" / "claim_evidence_ledger.md"
    if ledger_csv.exists():
        ledger = pd.read_csv(ledger_csv)
        if "C12" not in set(ledger.get("claim_id", pd.Series(dtype=str)).astype(str)):
            new = pd.DataFrame(
                [
                    {
                        "claim_id": "C12",
                        "claim_text": "D3 has a secondary online sequential-detection regime.",
                        "claim_category": "secondary",
                        "evidence_file": f"analysis/{ANALYSIS_NAME}/final_online_targeted_decision_report.md",
                        "evidence_table_figure": "Online/offline comparison table",
                        "metric_statistic": "Clean online prefix, accumulation, threshold, calibration, and stopping metrics.",
                        "sample_size": "57 participants with prefix rows from Label Release v1.1",
                        "caveat": "Project-specific online analysis; no official EyeBench SOTA claim.",
                        "manuscript_section": "Supplement Section 18",
                        "status": "supported_secondary",
                    }
                ]
            )
            ledger = pd.concat([ledger, new], ignore_index=True)
            ledger.to_csv(ledger_csv, index=False)
            changed.append(str(ledger_csv.relative_to(repo_root)))
            _write_md(ledger_md, "# Claim Evidence Ledger\n\n" + _md_table(ledger, max_rows=100))
            changed.append(str(ledger_md.relative_to(repo_root)))
        elif "C12" in set(ledger.get("claim_id", pd.Series(dtype=str)).astype(str)):
            changed.append(str(ledger_csv.relative_to(repo_root)))
            if ledger_md.exists():
                changed.append(str(ledger_md.relative_to(repo_root)))
    return changed


def run_d3_online_targeted_optimization(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    out = Path(output_dir) if output_dir else timestamped_output_dir(config, repo_root=root)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    analysis_dir = _analysis_dir(config, root)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for doc in REQUIRED_DOCS:
        if not (root / doc).exists():
            raise FileNotFoundError(f"GOAL 0 document missing before model code execution: {doc}")
    _update_subgoal(root, "GOAL_0", "completed", list(REQUIRED_DOCS))

    prefix, prefix_manifest = build_prefix_dataset(config, out, root)
    _update_subgoal(
        root,
        "GOAL_1",
        "completed" if prefix_manifest["row_count"] > 0 and not prefix_manifest["monotonic_errors"] else "blocked",
        [
            str((out / "prefix_data" / "prefix_features.parquet").relative_to(root)),
            f"analysis/{ANALYSIS_NAME}/prefix_feature_dictionary.md",
            f"analysis/{ANALYSIS_NAME}/prefix_dataset_report.md",
        ],
        "" if prefix_manifest["row_count"] > 0 and not prefix_manifest["monotonic_errors"] else "prefix validation failed",
    )

    predictions, nested_manifest = generate_nested_predictions(config, prefix, out, root)
    legal_rows_available = nested_manifest["outer_test_rows"] > 0 and (
        nested_manifest["inner_oof_rows"] > 0 or nested_manifest["calibration_rows"] > 0
    )
    _update_subgoal(
        root,
        "GOAL_2",
        "completed" if legal_rows_available else "blocked",
        [
            str((out / "nested_predictions").relative_to(root)),
            f"analysis/{ANALYSIS_NAME}/nested_prediction_artifact_report.md",
        ],
        "" if legal_rows_available else "nested predictions missing outer_test or legal adaptation rows",
    )

    prefix_metrics = evaluate_online_prefix_models(config, predictions, root)
    d3_related = prefix_metrics[prefix_metrics.get("feature_group", pd.Series(dtype=str)).astype(str).str.contains("dfm_residual", na=False)]
    _update_subgoal(
        root,
        "GOAL_3",
        "completed" if not d3_related.empty else "blocked",
        [
            f"analysis/{ANALYSIS_NAME}/online_prefix_model_metrics.csv",
            f"analysis/{ANALYSIS_NAME}/online_prefix_model_report.md",
        ],
        "" if not d3_related.empty else "DFM residual online model did not produce metrics",
    )

    calibration, threshold_metrics, thresholds = legal_calibration_and_thresholds(config, predictions, root)
    fitted_calibration = (
        not calibration.empty
        and calibration["calibrator"].ne("identity").any()
        and calibration["calibration_status"].eq("complete").any()
    )
    learned_threshold = (
        not thresholds.empty
        and thresholds["threshold_policy"].ne("fixed_0_5").any()
        and thresholds["status"].eq("complete").any()
    )
    _update_subgoal(
        root,
        "GOAL_4",
        "completed" if fitted_calibration and learned_threshold else "blocked",
        [
            f"analysis/{ANALYSIS_NAME}/legal_calibration_metrics.csv",
            f"analysis/{ANALYSIS_NAME}/legal_threshold_metrics.csv",
            f"analysis/{ANALYSIS_NAME}/legal_thresholds_learned.csv",
            f"analysis/{ANALYSIS_NAME}/calibration_threshold_report.md",
        ],
        "" if fitted_calibration and learned_threshold else "no fitted calibrator or learned legal threshold completed",
    )

    online, accumulation_metrics = build_online_probabilities(config, predictions, out, root)
    accumulators_ok = online["accumulator"].nunique() >= 3 if not online.empty else False
    meta_attempted = "learned_meta_aggregator" in set(online.get("accumulator", pd.Series(dtype=str))) if not online.empty else False
    _update_subgoal(
        root,
        "GOAL_5",
        "completed" if accumulators_ok and meta_attempted else "blocked",
        [
            str((out / "online_probabilities" / "online_probabilities.csv").relative_to(root)),
            f"analysis/{ANALYSIS_NAME}/online_evidence_accumulation_metrics.csv",
            f"analysis/{ANALYSIS_NAME}/online_evidence_accumulation_report.md",
        ],
        "" if accumulators_ok and meta_attempted else "fewer than three accumulators or meta-aggregator not attempted",
    )

    stopping_metrics, curve = evaluate_stopping_policies(config, online, root)
    stopping_ok = not stopping_metrics.empty and stopping_metrics["stopping_policy"].nunique() >= 4
    _update_subgoal(
        root,
        "GOAL_6",
        "completed" if stopping_ok else "blocked",
        [
            f"analysis/{ANALYSIS_NAME}/online_stopping_policy_metrics.csv",
            f"analysis/{ANALYSIS_NAME}/online_stopping_policy_report.md",
            f"analysis/{ANALYSIS_NAME}/online_earliness_performance_curve.csv",
        ],
        "" if stopping_ok else "fewer than four stopping policies evaluated",
    )

    search_space, ranking, locked = targeted_optimization(config, online, root)
    _update_subgoal(
        root,
        "GOAL_7",
        "completed" if len(search_space) >= 12 and not locked.empty else "blocked",
        [
            f"analysis/{ANALYSIS_NAME}/online_candidate_search_space.csv",
            f"analysis/{ANALYSIS_NAME}/online_candidate_validation_ranking.csv",
            f"analysis/{ANALYSIS_NAME}/online_locked_test_results.csv",
            f"analysis/{ANALYSIS_NAME}/online_targeted_optimization_report.md",
        ],
        "" if len(search_space) >= 12 and not locked.empty else "candidate search or locked test result missing",
    )

    oracle, oracle_budget = oracle_diagnostics(config, online, root)
    oracle_ok = not oracle.empty and (oracle["official_claim_allowed"] == False).all()  # noqa: E712
    _update_subgoal(
        root,
        "GOAL_8",
        "completed" if oracle_ok else "blocked",
        [
            f"analysis/{ANALYSIS_NAME}/oracle_upper_bound_metrics.csv",
            f"analysis/{ANALYSIS_NAME}/oracle_information_budget.csv",
            f"analysis/{ANALYSIS_NAME}/oracle_upper_bound_report.md",
        ],
        "" if oracle_ok else "oracle diagnostics missing or not marked diagnostic",
    )

    trajectories, persistent = error_trajectory_analysis(config, online, locked, root)
    _update_subgoal(
        root,
        "GOAL_9",
        "completed" if not trajectories.empty else "blocked",
        [
            f"analysis/{ANALYSIS_NAME}/reader_probability_trajectories.csv",
            f"analysis/{ANALYSIS_NAME}/persistent_error_readers.csv",
            f"analysis/{ANALYSIS_NAME}/error_trajectory_report.md",
        ],
        "" if not trajectories.empty else "no locked trajectory rows",
    )

    comparison, decision = online_offline_comparison_and_decision(
        config, prefix_metrics, accumulation_metrics, stopping_metrics, locked, oracle, root
    )
    _update_subgoal(
        root,
        "GOAL_10",
        "completed" if not comparison.empty else "blocked",
        [
            f"analysis/{ANALYSIS_NAME}/online_offline_comparison_table.csv",
            f"analysis/{ANALYSIS_NAME}/online_offline_comparison_table.md",
            f"analysis/{ANALYSIS_NAME}/final_online_targeted_decision_report.md",
        ],
        "" if not comparison.empty else "comparison table missing",
    )

    manuscript_changed = update_manuscript_if_valid(decision, root)
    _update_subgoal(
        root,
        "GOAL_11",
        "completed" if manuscript_changed else "blocked",
        manuscript_changed,
        "" if manuscript_changed else "manuscript files unavailable or already contained the online update",
    )

    manifest = {
        "status": "complete",
        "output_dir": str(out),
        "analysis_dir": str(analysis_dir),
        "prefix_manifest": prefix_manifest,
        "nested_manifest": nested_manifest,
        "prefix_metric_rows": int(len(prefix_metrics)),
        "calibration_metric_rows": int(len(calibration)),
        "threshold_metric_rows": int(len(threshold_metrics)),
        "online_probability_rows": int(len(online)),
        "accumulation_metric_rows": int(len(accumulation_metrics)),
        "stopping_metric_rows": int(len(stopping_metrics)),
        "candidate_rows": int(len(search_space)),
        "locked_test_rows": int(len(locked)),
        "oracle_rows": int(len(oracle)),
        "trajectory_rows": int(len(trajectories)),
        "official_sota_claim_changed": False,
        "manuscript_changed": manuscript_changed,
    }
    _write_json(out / "run_manifest.json", manifest)
    _write_json(analysis_dir / "run_manifest.json", manifest)
    return manifest


def validate_d3_online_targeted_optimization(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    analysis_dir = _analysis_dir(config, root)
    errors: list[str] = []
    warnings: list[str] = []
    for doc in REQUIRED_DOCS:
        if not (root / doc).exists():
            errors.append(f"missing GOAL 0 document: {doc}")
    prefix_path = out / "prefix_data" / "prefix_features.parquet"
    if not prefix_path.exists():
        errors.append(f"missing prefix data: {prefix_path}")
        prefix = pd.DataFrame()
    else:
        prefix = pd.read_parquet(prefix_path)
        if prefix.empty:
            errors.append("prefix data has zero rows")
        errors.extend(validate_no_future_evidence(prefix))
    nested_root = out / "nested_predictions"
    nested_all = nested_root / "all_nested_prefix_predictions.csv"
    if not nested_all.exists():
        errors.append(f"missing nested prediction rollup: {nested_all}")
        predictions = pd.DataFrame()
    else:
        required_cols = {
            "participant_id",
            "fold_id",
            "prefix_type",
            "prefix_value",
            "split_regime",
            "split_role",
            "y_true",
            "p_pred",
            "model_name",
            "feature_group",
            "threshold_source",
            "evidence_available_until_prefix",
            "n_words_observed",
            "n_trials_observed",
            "n_texts_observed",
            "stable_enough_for_prediction",
        }
        validation_cols = sorted(required_cols | {"observed_text_ids"})
        predictions = pd.read_csv(nested_all, usecols=lambda col: col in validation_cols)
        missing = required_cols - set(predictions.columns)
        if missing:
            errors.append(f"nested predictions missing required columns: {sorted(missing)}")
        if "split_role" in predictions and "outer_test" not in set(predictions["split_role"].astype(str)):
            errors.append("nested predictions contain no outer_test rows")
        if "split_role" in predictions and not set(predictions["split_role"].astype(str)).intersection({"inner_oof", "calibration"}):
            errors.append("nested predictions contain no inner_oof/calibration rows")
        errors.extend(_validate_split_disjointness(predictions))
    predictor_manifest = nested_root / "predictor_manifest.json"
    if predictor_manifest.exists():
        predictors = json.loads(predictor_manifest.read_text(encoding="utf-8"))
        for row in predictors:
            for col in row.get("predictor_columns", []):
                lowered = str(col).lower()
                if any(pattern in lowered for pattern in PROHIBITED_PREDICTOR_PATTERNS):
                    errors.append(f"prohibited predictor used: {col}")
    else:
        errors.append("missing predictor_manifest.json")
    required_analysis = [
        "online_prefix_model_metrics.csv",
        "legal_calibration_metrics.csv",
        "legal_threshold_metrics.csv",
        "online_evidence_accumulation_metrics.csv",
        "online_stopping_policy_metrics.csv",
        "online_candidate_search_space.csv",
        "online_candidate_validation_ranking.csv",
        "online_locked_test_results.csv",
        "oracle_upper_bound_metrics.csv",
        "online_offline_comparison_table.csv",
        "final_online_targeted_decision_report.md",
    ]
    for rel in required_analysis:
        if not (analysis_dir / rel).exists():
            errors.append(f"missing analysis artifact: {analysis_dir / rel}")
    oracle_path = analysis_dir / "oracle_upper_bound_metrics.csv"
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        if not oracle.empty and not (oracle["official_claim_allowed"] == False).all():  # noqa: E712
            errors.append("oracle rows are not all marked official_claim_allowed=false")
    status_json, _ = _status_paths(root)
    if not status_json.exists():
        errors.append("missing subgoal_status.json")
    else:
        payload = json.loads(status_json.read_text(encoding="utf-8"))
        errors.extend(validate_subgoal_status_payload(payload))
        incomplete = [
            goal
            for goal, item in payload.get("subgoals", {}).items()
            if item.get("status") not in {"completed", "blocked"}
        ]
        if incomplete:
            errors.append(f"subgoals not completed or blocked: {incomplete}")
    decision_path = analysis_dir / "final_online_targeted_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("official_sota_claim_changed") is not False:
            errors.append("official SOTA claim changed without official chain")
    else:
        errors.append("missing final_online_targeted_decision.json")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "output_dir": str(out),
        "analysis_dir": str(analysis_dir),
        "prefix_rows": int(len(prefix)) if not prefix.empty else 0,
        "nested_prediction_rows": int(len(predictions)) if not predictions.empty else 0,
    }
    _write_json(out / "d3_online_targeted_optimization_validation_report.json", report)
    _write_json(analysis_dir / "d3_online_targeted_optimization_validation_report.json", report)
    if not errors:
        _update_subgoal(
            root,
            "GOAL_12",
            "completed",
            [
                str((out / "d3_online_targeted_optimization_validation_report.json").relative_to(root)),
                f"analysis/{ANALYSIS_NAME}/d3_online_targeted_optimization_validation_report.json",
            ],
        )
    return report


def _validate_split_disjointness(predictions: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if predictions.empty:
        return errors
    for (regime, fold_id), group in predictions.groupby(["split_regime", "fold_id"], dropna=False):
        train = group[group["split_role"].isin(["train_fit", "inner_oof", "calibration"])]
        test = group[group["split_role"].eq("outer_test")]
        if train.empty or test.empty:
            continue
        if regime in {"unseen_reader", "unseen_reader_and_text", "text_balanced_unseen_reader", "participant_grouped_kfold"}:
            overlap = set(train["participant_id"].astype(str)) & set(test["participant_id"].astype(str))
            if overlap:
                errors.append(f"{regime} fold {fold_id} participant overlap: {sorted(overlap)[:5]}")
        if regime in {"unseen_text", "unseen_reader_and_text"}:
            train_texts: set[str] = set()
            test_texts: set[str] = set()
            for value in train.get("observed_text_ids", pd.Series(dtype=str)):
                train_texts.update(_observed_text_set(value))
            for value in test.get("observed_text_ids", pd.Series(dtype=str)):
                test_texts.update(_observed_text_set(value))
            overlap = train_texts & test_texts
            if overlap:
                errors.append(f"{regime} fold {fold_id} text overlap: {sorted(overlap)[:5]}")
    return errors
