"""D3-family own-method score maximization on official EyeBench CopCo_TYP folds.

This runner intentionally does not reproduce official leaderboard methods. It
uses the published leaderboard only as a fixed reference and preserves the
previous D3_EyeBench_Lite adapter as candidate_0000 before testing any
D3-family extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark_bridge import _model_pipeline
from .config import get_nested, load_config, timestamped_output_dir
from .d3_eyebench_protocol_optimization import (
    BASE_DENYLIST,
    _classification_metrics_at_threshold,
    _inner_split,
    _numeric_matrix,
    _safe_numeric_columns,
)
from .official_eyebench_sota_check import (
    OFFICIAL_LITE_MODE,
    OFFICIAL_SPLITS,
    READER_LEVEL,
    RESIDUAL_OUTCOME_CANDIDATES as LITE_OUTCOME_CANDIDATES,
    RESIDUAL_PREDICTOR_CANDIDATES,
    SOTA_MODEL,
    SPLIT_TO_OFFICIAL_REGIME,
    TRIAL_LEVEL,
    _fold_cache,
    _official_reference_table,
    _score_estimator,
    _trial_residual_features,
    build_official_split_labels,
    evaluate_d3_eyebench_lite,
    load_official_processed_features,
    validate_official_split_labels,
)
from .research_exploration import _markdown_table, _np, _pd


CAMPAIGN_SECTION = "d3_eyebench_own_method_score_max"
CAMPAIGN_MODEL = "D3_EyeBench_OwnMethodScoreMaxV2"

VALID_DECISION_CATEGORIES = {
    "d3_method_improved",
    "d3_method_competitive_but_not_improved",
    "d3_method_exploratory_gain_only",
    "d3_method_not_improved",
    "blocked_by_environment",
    "blocked_by_data",
    "blocked_by_evaluator",
}

PRIMARY_SPLITS = tuple(OFFICIAL_SPLITS)

EXTRA_DENYLIST = {
    "label",
    "labels",
    "target",
    "targets",
    "y_true",
    "prediction",
    "prediction_prob",
    "binary_prediction",
    "exposure_count",
    "participant_exposure_count",
    "speech_exposure_count",
    "text_exposure_count",
}

PROHIBITED_FEATURES = BASE_DENYLIST | EXTRA_DENYLIST

EXTENDED_OUTCOME_CANDIDATES = {
    **LITE_OUTCOME_CANDIDATES,
    "landing_position": ["landing_position", "IA_FIRST_RUN_LANDING_POSITION"],
    "mean_saccade_duration": ["mean_sacc_dur"],
    "peak_saccade_velocity": ["peak_sacc_velocity"],
    "regression_in_count": ["IA_REGRESSION_IN_COUNT"],
    "regression_out_count": ["IA_REGRESSION_OUT_COUNT", "IA_REGRESSION_OUT_FULL_COUNT"],
    "last_fixation_duration": ["IA_LAST_FIXATION_DURATION"],
    "last_run_dwell_time": ["IA_LAST_RUN_DWELL_TIME"],
}

BASE_OUTCOMES = [
    "first_fixation_duration",
    "first_pass_duration",
    "go_past_time",
    "total_fixation_duration",
    "skipping",
    "fixation_count",
]

EXTENDED_OUTCOMES = [
    *BASE_OUTCOMES,
    "landing_position",
    "mean_saccade_duration",
    "peak_saccade_velocity",
    "regression_in_count",
    "regression_out_count",
    "last_fixation_duration",
    "last_run_dwell_time",
]

CONTEXT_FEATURES = [
    "word_length",
    "word_length_no_punctuation",
    "wordfreq_frequency",
    "subtlex_frequency",
    "gpt2_surprisal",
    "normalized_ID",
]

METRIC_COLUMNS = [
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
    "usable_folds",
    "skipped_folds",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
    "brier_score",
    "ece",
    "threshold",
    "status",
    "skip_reason",
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    feature_recipe: str
    model_type: str
    model_params: dict[str, Any]
    threshold_method: str
    calibration_method: str
    seed: int
    anchor_exact: bool = False
    exploratory: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "feature_recipe": self.feature_recipe,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "threshold_method": self.threshold_method,
            "calibration_method": self.calibration_method,
            "seed": self.seed,
            "anchor_exact": self.anchor_exact,
            "exploratory": self.exploratory,
            "preserves_previous_d3_lite_features": True,
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_parquet(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = get_nested(config, CAMPAIGN_SECTION, {})
    return section if isinstance(section, dict) else {}


def _to_sota_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    return {
        "run": config.get("run", {}),
        "official_eyebench_sota_check": {
            "eyebench": section.get("eyebench", {}),
            "deterministic_seed": section.get("deterministic_seed", 173),
            "split_regimes": section.get("split_regimes", list(OFFICIAL_SPLITS)),
            "prohibited_features": section.get("prohibited_features", []),
            "decision_gates": {
                "CopCo_TYP": {
                    "formatted_table": get_nested(
                        config,
                        f"{CAMPAIGN_SECTION}.published_leaderboard_snapshot.local_formatted_table",
                    )
                }
            },
        },
    }


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    repo_analysis = root / str(
        get_nested(
            config,
            f"{CAMPAIGN_SECTION}.repo_analysis_dir",
            "analysis/d3_eyebench_own_method_score_max_v2",
        )
    )
    result_analysis = out / str(
        get_nested(
            config,
            f"{CAMPAIGN_SECTION}.output_layout.analysis",
            "analysis/d3_eyebench_own_method_score_max_v2",
        )
    )
    return {"repo_analysis": repo_analysis, "result_analysis": result_analysis}


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["repo_analysis"] / name, text)
    _write_md(dirs["result_analysis"] / name, text)


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


def _git_status_short(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_d3_eyebench_own_method_score_max_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    errors: list[str] = []
    for flag in [
        "official_eyebench_data_folds_only",
        "no_leaderboard_method_reruns",
        "no_wandb_online_api",
        "no_test_label_tuning",
        "no_synthetic_outputs",
        "no_random_predictions",
        "no_full_prepared_copco_joins",
        "candidate_0000_required",
        "monotonic_best_so_far",
        "preserve_d3_lite_features_first",
    ]:
        if section.get(flag) is not True:
            errors.append(f"{CAMPAIGN_SECTION}.{flag} must be true")
    splits = section.get("split_regimes", [])
    missing_splits = sorted(set(OFFICIAL_SPLITS) - set(splits))
    if missing_splits:
        errors.append(f"missing official split regimes: {missing_splits}")
    prohibited = set(section.get("prohibited_features", []))
    missing_prohibited = sorted(PROHIBITED_FEATURES - prohibited)
    if missing_prohibited:
        errors.append(f"prohibited feature list incomplete: {missing_prohibited}")
    expected = get_nested(config, f"{CAMPAIGN_SECTION}.prior_d3_lite_anchor.expected_metrics", {})
    for split_name in OFFICIAL_SPLITS:
        row = expected.get(split_name, {}) if isinstance(expected, dict) else {}
        if _safe_float(row.get("balanced_accuracy")) is None or _safe_float(row.get("roc_auc")) is None:
            errors.append(f"missing candidate_0000 expected BA/AUROC for {split_name}")
    if int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.max_candidates", 0)) <= 0:
        errors.append("budget.max_candidates must be positive")
    if int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.test_eval_top_k", 0)) < 1:
        errors.append("budget.test_eval_top_k must be at least 1")
    return {"status": "passed" if not errors else "failed", "errors": errors}


def build_candidate_specs(config: dict[str, Any]) -> list[CandidateSpec]:
    section = _section(config)
    grid = section.get("candidate_grid", {})
    seed = int(section.get("deterministic_seed", 20260522))
    candidates = [
        CandidateSpec(
            candidate_id="candidate_0000",
            family="d3_lite_anchor",
            feature_recipe="d3_lite_exact",
            model_type="official_lite_logistic",
            model_params={"class_weight": "balanced", "threshold": 0.5},
            threshold_method="fixed_0_5",
            calibration_method="none",
            seed=seed,
            anchor_exact=True,
            exploratory=False,
        )
    ]
    feature_recipes = [str(x) for x in grid.get("feature_recipes", [])]
    models = [dict(x) for x in grid.get("models", [])]
    threshold_methods = [str(x) for x in grid.get("threshold_methods", ["inner_balanced_accuracy"])]
    calibration_methods = [str(x) for x in grid.get("calibration_methods", ["none"])]
    payloads: list[dict[str, Any]] = []
    for feature_recipe in feature_recipes:
        for model in models:
            for threshold_method in threshold_methods:
                for calibration_method in calibration_methods:
                    name = str(model.get("name", "logistic_l2"))
                    params = dict(model.get("params", {}))
                    if feature_recipe == "d3_lite_exact" and name == "official_lite_logistic":
                        continue
                    if threshold_method == "fixed_0_5" and calibration_method != "none":
                        continue
                    payload = {
                        "feature_recipe": feature_recipe,
                        "model_type": name,
                        "model_params": params,
                        "threshold_method": threshold_method,
                        "calibration_method": calibration_method,
                        "seed": seed,
                    }
                    payload["order_key"] = hashlib.sha1(
                        json.dumps(payload, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    payloads.append(payload)
    payloads = sorted(payloads, key=lambda item: item["order_key"])
    max_candidates = int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.max_candidates", len(payloads)))
    for payload in payloads[: max(0, max_candidates - 1)]:
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        idx = len(candidates)
        candidates.append(
            CandidateSpec(
                candidate_id=f"candidate_{idx:04d}_{digest}",
                family=_family_for_recipe(str(payload["feature_recipe"])),
                feature_recipe=str(payload["feature_recipe"]),
                model_type=str(payload["model_type"]),
                model_params=dict(payload["model_params"]),
                threshold_method=str(payload["threshold_method"]),
                calibration_method=str(payload["calibration_method"]),
                seed=int(payload["seed"]),
                anchor_exact=False,
                exploratory=False,
            )
        )
    return candidates


def _family_for_recipe(recipe: str) -> str:
    if recipe == "d3_lite_exact":
        return "d3_lite_calibration_variant"
    if recipe == "d3_lite_robust":
        return "d3_lite_plus_robust_residuals"
    if recipe == "d3_lite_raw_summary":
        return "d3_lite_plus_raw_gaze"
    if recipe == "d3_lite_interactions":
        return "d3_lite_plus_text_gaze_interactions"
    if recipe == "d3_lite_all":
        return "d3_lite_plus_full_official_extension"
    return "d3_family_variant"


def _outcome_column(frame: Any, outcome: str) -> str | None:
    for column in EXTENDED_OUTCOME_CANDIDATES.get(outcome, []):
        if column in frame:
            return column
    return None


def _clean_numeric(values: Any) -> Any:
    pd = _pd()
    np = _np()
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _stat_features(values: Any, prefix: str, stats: list[str]) -> dict[str, float]:
    np = _np()
    series = _clean_numeric(values)
    out: dict[str, float] = {}
    valid = series.dropna()
    for stat in stats:
        key = f"{prefix}_{stat}"
        if valid.empty:
            out[key] = np.nan
        elif stat == "mean":
            out[key] = float(valid.mean())
        elif stat == "median":
            out[key] = float(valid.median())
        elif stat == "sd":
            out[key] = float(valid.std(ddof=0))
        elif stat == "q25":
            out[key] = float(valid.quantile(0.25))
        elif stat == "q75":
            out[key] = float(valid.quantile(0.75))
        elif stat == "iqr":
            out[key] = float(valid.quantile(0.75) - valid.quantile(0.25))
        elif stat == "min":
            out[key] = float(valid.min())
        elif stat == "max":
            out[key] = float(valid.max())
        elif stat == "abs_mean":
            out[key] = float(valid.abs().mean())
    return out


def _slope(x_values: Any, y_values: Any) -> float:
    np = _np()
    x = _clean_numeric(x_values)
    y = _clean_numeric(y_values)
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return np.nan
    x_arr = x[valid].astype(float).to_numpy()
    y_arr = y[valid].astype(float).to_numpy()
    variance = float(np.var(x_arr))
    if variance <= 0:
        return np.nan
    return float(np.cov(x_arr, y_arr, ddof=0)[0, 1] / variance)


def _corr(x_values: Any, y_values: Any) -> float:
    np = _np()
    x = _clean_numeric(x_values)
    y = _clean_numeric(y_values)
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return np.nan
    x_arr = x[valid].astype(float).to_numpy()
    y_arr = y[valid].astype(float).to_numpy()
    if float(np.std(x_arr)) <= 0 or float(np.std(y_arr)) <= 0:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _trial_features_for_recipe(
    fit_ids: set[str],
    apply_ids: set[str],
    ia: Any,
    candidate: CandidateSpec,
    prohibited: set[str],
) -> tuple[Any, dict[str, Any]]:
    pd = _pd()
    np = _np()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if candidate.feature_recipe == "d3_lite_exact":
        features, diag = _trial_residual_features(fit_ids, apply_ids, ia)
        diag = dict(diag)
        diag.update(
            {
                "candidate_id": candidate.candidate_id,
                "feature_recipe": candidate.feature_recipe,
                "preserves_d3_lite_features": True,
                "feature_families": ["previous_d3_lite_residuals"],
            }
        )
        return features, diag

    word = ia.copy()
    word["unique_trial_id"] = word["unique_trial_id"].astype(str)
    fit_word = word[word["unique_trial_id"].isin(fit_ids)].copy()
    apply_word = word[word["unique_trial_id"].isin(apply_ids)].copy()
    predictors = _safe_numeric_columns(fit_word, RESIDUAL_PREDICTOR_CANDIDATES, prohibited)
    diagnostics = {
        "candidate_id": candidate.candidate_id,
        "feature_recipe": candidate.feature_recipe,
        "fit_word_rows": int(len(fit_word)),
        "apply_word_rows": int(len(apply_word)),
        "predictors": predictors,
        "heldout_reader_rows_used_for_fit": False,
        "heldout_text_rows_used_for_fit": False,
        "reader_group_used": False,
        "preserves_d3_lite_features": True,
        "feature_families": ["previous_d3_lite_residuals"],
        "skipped": False,
        "skip_reason": "",
    }
    if fit_word.empty or apply_word.empty or not predictors:
        diagnostics["skipped"] = True
        diagnostics["skip_reason"] = "no_fit_or_apply_words_or_predictors"
        return pd.DataFrame(), diagnostics
    combined = pd.concat([fit_word, apply_word], ignore_index=True)
    outcomes = BASE_OUTCOMES if candidate.feature_recipe != "d3_lite_all" else EXTENDED_OUTCOMES
    residual_columns: list[str] = []
    for outcome in outcomes:
        column = _outcome_column(fit_word, outcome)
        if column is None:
            continue
        fit_y = _clean_numeric(fit_word[column])
        valid_fit = fit_y.notna()
        if int(valid_fit.sum()) < 3 or fit_y.nunique(dropna=True) <= 1:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(_numeric_matrix(fit_word.loc[valid_fit], predictors), fit_y.loc[valid_fit])
        observed = _clean_numeric(combined[column]).to_numpy(dtype=float)
        predicted = model.predict(_numeric_matrix(combined, predictors))
        prefix = "d3_lite_resid" if outcome in BASE_OUTCOMES else "d3_fuller_resid"
        resid_name = f"{prefix}_{outcome}"
        combined[resid_name] = observed - predicted
        residual_columns.append(resid_name)
    if not residual_columns:
        diagnostics["skipped"] = True
        diagnostics["skip_reason"] = "no_residual_outcomes_available"
        return pd.DataFrame(), diagnostics

    stats = ["mean", "median", "sd"]
    if candidate.feature_recipe in {"d3_lite_robust", "d3_lite_all"}:
        stats = ["mean", "median", "sd", "q25", "q75", "iqr", "abs_mean", "min", "max"]
        diagnostics["feature_families"].append("robust_residual_distributions")
    raw_outcomes = EXTENDED_OUTCOMES if candidate.feature_recipe in {"d3_lite_raw_summary", "d3_lite_all"} else []
    if raw_outcomes:
        diagnostics["feature_families"].append("official_word_level_raw_gaze_summaries")
    interactions = candidate.feature_recipe in {"d3_lite_interactions", "d3_lite_all"}
    if interactions:
        diagnostics["feature_families"].append("text_gaze_interaction_summaries")

    rows: list[dict[str, Any]] = []
    apply_subset = combined[combined["unique_trial_id"].isin(apply_ids)].copy()
    raw_columns = [(outcome, _outcome_column(apply_subset, outcome)) for outcome in raw_outcomes]
    raw_columns = [(outcome, column) for outcome, column in raw_columns if column is not None]
    context_columns = _safe_numeric_columns(apply_subset, CONTEXT_FEATURES, prohibited)
    interaction_outcomes = [
        ("first_fixation_duration", _outcome_column(apply_subset, "first_fixation_duration")),
        ("total_fixation_duration", _outcome_column(apply_subset, "total_fixation_duration")),
        ("go_past_time", _outcome_column(apply_subset, "go_past_time")),
        ("fixation_count", _outcome_column(apply_subset, "fixation_count")),
    ]
    interaction_outcomes = [(name, col) for name, col in interaction_outcomes if col is not None]
    for trial_id, group in apply_subset.groupby("unique_trial_id", dropna=False):
        row = {"sample_id": str(trial_id), "unique_trial_id": str(trial_id)}
        for column in residual_columns:
            row.update(_stat_features(group[column], column, stats))
        for outcome, column in raw_columns:
            row.update(
                _stat_features(
                    group[column],
                    f"d3_raw_{outcome}",
                    ["mean", "median", "sd", "q25", "q75"],
                )
            )
        if interactions:
            for outcome, outcome_column in interaction_outcomes:
                for context_column in context_columns:
                    prefix = f"d3_interaction_{outcome}_by_{context_column}"
                    row[f"{prefix}_slope"] = _slope(group[context_column], group[outcome_column])
                    row[f"{prefix}_corr"] = _corr(group[context_column], group[outcome_column])
        rows.append(row)
    features = pd.DataFrame(rows)
    for column in [c for c in features.columns if c not in {"sample_id", "unique_trial_id"}]:
        values = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        features[column] = values
    return features, diagnostics


def _feature_columns(frame: Any, prohibited: set[str]) -> list[str]:
    prefixes = ("d3_lite_resid_", "d3_fuller_resid_", "d3_raw_", "d3_interaction_")
    return _safe_numeric_columns(
        frame,
        [column for column in frame.columns if str(column).startswith(prefixes)],
        prohibited,
    )


def _make_model(candidate: CandidateSpec, n_jobs: int) -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seed = int(candidate.seed)
    params = dict(candidate.model_params)
    if candidate.model_type == "official_lite_logistic":
        model = _model_pipeline("logistic_regression", task="typ", seed=seed)
    elif candidate.model_type == "logistic_l2":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=float(params.get("C", 1.0)),
                penalty="l2",
                solver=str(params.get("solver", "lbfgs")),
                class_weight=params.get("class_weight", "balanced"),
                max_iter=int(params.get("max_iter", 2000)),
                random_state=seed,
            ),
        )
    elif candidate.model_type == "logistic_elasticnet":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=float(params.get("C", 1.0)),
                penalty="elasticnet",
                solver="saga",
                l1_ratio=float(params.get("l1_ratio", 0.5)),
                class_weight=params.get("class_weight", "balanced"),
                max_iter=int(params.get("max_iter", 5000)),
                random_state=seed,
            ),
        )
    elif candidate.model_type == "random_forest_d3":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 500)),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                max_features=params.get("max_features", "sqrt"),
                class_weight=params.get("class_weight", "balanced_subsample"),
                n_jobs=max(1, int(n_jobs)),
                random_state=seed,
            ),
        )
    elif candidate.model_type == "extra_trees_d3":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=int(params.get("n_estimators", 600)),
                min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                max_features=params.get("max_features", "sqrt"),
                class_weight=params.get("class_weight", "balanced"),
                n_jobs=max(1, int(n_jobs)),
                random_state=seed,
            ),
        )
    else:
        raise ValueError(f"unsupported D3 candidate model type: {candidate.model_type}")
    if candidate.calibration_method == "sigmoid_cv3":
        return CalibratedClassifierCV(model, method="sigmoid", cv=3)
    if candidate.calibration_method == "isotonic_cv3":
        return CalibratedClassifierCV(model, method="isotonic", cv=3)
    return model


def _ece(y_true: Any, y_score: Any, bins: int = 10) -> float | None:
    np = _np()
    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_score_arr)
    if int(valid.sum()) == 0:
        return None
    y_true_arr = y_true_arr[valid]
    y_score_arr = np.clip(y_score_arr[valid], 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    n = len(y_score_arr)
    for i in range(bins):
        low = edges[i]
        high = edges[i + 1]
        mask = (y_score_arr >= low) & (y_score_arr <= high if i == bins - 1 else y_score_arr < high)
        if int(mask.sum()) == 0:
            continue
        error += float(mask.sum()) / n * abs(float(y_true_arr[mask].mean()) - float(y_score_arr[mask].mean()))
    return float(error)


def _sensitivity_specificity(y_true: Any, y_score: Any, threshold: float) -> dict[str, float | None]:
    np = _np()
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= float(threshold)).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"sensitivity": sensitivity, "specificity": specificity}


def _metrics_at_threshold(y_true: Any, y_score: Any, threshold: float) -> dict[str, Any]:
    metrics = _classification_metrics_at_threshold(y_true, y_score, threshold)
    metrics.update(_sensitivity_specificity(y_true, y_score, threshold))
    metrics["ece"] = _ece(y_true, y_score)
    return metrics


def _select_threshold(y_true: Any, y_score: Any, thresholds: list[float]) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        metric = _metrics_at_threshold(y_true, y_score, threshold)
        value = metric.get("balanced_accuracy")
        if value is None:
            continue
        tie_break = abs(float(threshold) - 0.5)
        best_tie_break = abs(float(best_threshold) - 0.5)
        if float(value) > best_score or (float(value) == best_score and tie_break < best_tie_break):
            best_score = float(value)
            best_threshold = float(threshold)
    return best_threshold, best_score


def _fit_score_fold(
    train: Any,
    test: Any,
    feature_columns: list[str],
    candidate: CandidateSpec,
    *,
    threshold: float,
    n_jobs: int,
) -> tuple[Any | None, Any, str]:
    pd = _pd()
    if train.empty or test.empty:
        return None, pd.DataFrame(), "empty_train_or_test"
    y_train = pd.to_numeric(train.get("reader_group_binary"), errors="coerce")
    y_test = pd.to_numeric(test.get("reader_group_binary"), errors="coerce")
    if y_train.nunique(dropna=True) < 2 or y_test.notna().sum() == 0 or not feature_columns:
        return None, pd.DataFrame(), "single_class_or_no_features"
    valid_train = y_train.notna()
    model = _make_model(candidate, n_jobs)
    model.fit(_numeric_matrix(train.loc[valid_train], feature_columns), y_train.loc[valid_train].astype(int))
    score = _score_estimator(model, _numeric_matrix(test, feature_columns))
    pred = test.copy()
    pred["y_true"] = y_test.astype(int).to_numpy()
    pred["y_score"] = score
    pred["threshold"] = float(threshold)
    pred["y_pred"] = (pred["y_score"].astype(float) >= float(threshold)).astype(int)
    return model, pred, ""


def _prediction_rows(pred: Any, candidate: CandidateSpec, split_name: str, fold_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pred.to_dict("records"):
        rows.append(
            {
                "mode": OFFICIAL_LITE_MODE,
                "model_name": CAMPAIGN_MODEL if not candidate.anchor_exact else SOTA_MODEL,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "feature_recipe": candidate.feature_recipe,
                "model_type": candidate.model_type,
                "threshold_method": candidate.threshold_method,
                "calibration_method": candidate.calibration_method,
                "claim_type": "d3_own_method_candidate",
                "task": "CopCo_TYP",
                "split_name": split_name,
                "fold_id": int(fold_id),
                "feature_group": candidate.feature_recipe,
                "sample_id": row.get("sample_id"),
                "unique_trial_id": row.get("unique_trial_id"),
                "participant_id": row.get("participant_id"),
                "speech_id": row.get("speech_id"),
                "text_id": row.get("text_id"),
                "y_true": int(row["y_true"]),
                "y_score": float(row["y_score"]),
                "threshold": float(row["threshold"]),
                "y_pred": int(row["y_pred"]),
                "eval_regime": SPLIT_TO_OFFICIAL_REGIME[split_name],
                "eval_type": "test",
                "fold_index": int(fold_id),
                "n_features": int(row.get("n_features", 0) or 0),
            }
        )
    return rows


def _metric_frame(predictions: Any, candidate: CandidateSpec, *, evaluation_level: str) -> Any:
    pd = _pd()
    rows: list[dict[str, Any]] = []
    for split_name in OFFICIAL_SPLITS:
        split_pred = predictions[predictions["split_name"].eq(split_name)] if not predictions.empty else pd.DataFrame()
        fold_metrics = []
        for _, fold in split_pred.groupby("fold_id", dropna=False):
            threshold = float(fold.get("threshold", pd.Series([0.5])).iloc[0])
            metric = _metrics_at_threshold(fold["y_true"], fold["y_score"], threshold)
            metric["threshold"] = threshold
            metric["n_predictions"] = int(len(fold))
            fold_metrics.append(metric)
        if fold_metrics:
            fold_frame = pd.DataFrame(fold_metrics)
            values: dict[str, Any] = {}
            for metric in [
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "macro_f1",
                "sensitivity",
                "specificity",
                "brier_score",
                "ece",
                "threshold",
            ]:
                series = pd.to_numeric(fold_frame.get(metric), errors="coerce")
                values[metric] = float(series.mean()) if series.notna().any() else None
            n_predictions = int(fold_frame["n_predictions"].sum())
            usable_folds = int(len(fold_frame))
            status = "complete"
            skip_reason = ""
        else:
            values = {
                "roc_auc": None,
                "pr_auc": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "sensitivity": None,
                "specificity": None,
                "brier_score": None,
                "ece": None,
                "threshold": None,
            }
            n_predictions = 0
            usable_folds = 0
            status = "skipped"
            skip_reason = "no_predictions"
        n_features = (
            int(pd.to_numeric(split_pred.get("n_features"), errors="coerce").max())
            if not split_pred.empty and "n_features" in split_pred
            else None
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "feature_recipe": candidate.feature_recipe,
                "model_type": candidate.model_type,
                "threshold_method": candidate.threshold_method,
                "calibration_method": candidate.calibration_method,
                "split_name": split_name,
                "evaluation_level": evaluation_level,
                "n_features": n_features,
                "n_predictions": n_predictions,
                "usable_folds": usable_folds,
                "skipped_folds": int(4 - usable_folds),
                **values,
                "status": status,
                "skip_reason": skip_reason,
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _reader_aggregate(predictions: Any) -> Any:
    if predictions.empty:
        return predictions
    return (
        predictions.groupby(["task", "split_name", "fold_id", "participant_id"], dropna=False)
        .agg(
            y_true=("y_true", "first"),
            y_score=("y_score", "mean"),
            threshold=("threshold", "first"),
            sample_id=("sample_id", "first"),
            n_features=("n_features", "max"),
        )
        .reset_index()
    )


def _fold_level_metrics(predictions: Any, candidate: CandidateSpec, level: str) -> Any:
    pd = _pd()
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    for (split_name, fold_id), fold in predictions.groupby(["split_name", "fold_id"], dropna=False):
        threshold = float(fold.get("threshold", pd.Series([0.5])).iloc[0])
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "feature_recipe": candidate.feature_recipe,
                "model_type": candidate.model_type,
                "threshold_method": candidate.threshold_method,
                "calibration_method": candidate.calibration_method,
                "split_name": split_name,
                "fold_id": int(fold_id),
                "evaluation_level": level,
                "n_predictions": int(len(fold)),
                "n_features": int(pd.to_numeric(fold.get("n_features"), errors="coerce").max())
                if "n_features" in fold
                else None,
                "threshold": threshold,
                **_metrics_at_threshold(fold["y_true"], fold["y_score"], threshold),
            }
        )
    return pd.DataFrame(rows)


def _simple_mean(metrics: Any, metric: str, level: str = TRIAL_LEVEL) -> float | None:
    pd = _pd()
    if metrics.empty:
        return None
    rows = metrics[metrics["evaluation_level"].eq(level)]
    values = pd.to_numeric(rows.get(metric), errors="coerce")
    return float(values.mean()) if values.notna().any() else None


def _metric_value(metrics: Any, split_name: str, metric: str, level: str = TRIAL_LEVEL) -> float | None:
    rows = metrics[metrics["split_name"].eq(split_name) & metrics["evaluation_level"].eq(level)]
    if rows.empty or metric not in rows:
        return None
    return _safe_float(rows.iloc[0][metric])


def _standardize_anchor_metrics(metrics: Any, candidate: CandidateSpec) -> Any:
    pd = _pd()
    rows = []
    primary = metrics[metrics["evaluation_level"].isin([TRIAL_LEVEL, READER_LEVEL])].copy()
    for _, row in primary.iterrows():
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "feature_recipe": candidate.feature_recipe,
                "model_type": candidate.model_type,
                "threshold_method": candidate.threshold_method,
                "calibration_method": candidate.calibration_method,
                "split_name": row["split_name"],
                "evaluation_level": row["evaluation_level"],
                "n_features": row.get("n_features"),
                "n_predictions": row.get("n_predictions"),
                "usable_folds": row.get("usable_folds"),
                "skipped_folds": row.get("skipped_folds"),
                "roc_auc": row.get("roc_auc"),
                "pr_auc": row.get("pr_auc"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "macro_f1": row.get("macro_f1"),
                "sensitivity": None,
                "specificity": None,
                "brier_score": row.get("brier_score"),
                "ece": None,
                "threshold": 0.5,
                "status": row.get("status"),
                "skip_reason": row.get("skip_reason"),
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _standardize_anchor_predictions(predictions: Any, candidate: CandidateSpec) -> Any:
    if predictions.empty:
        return predictions
    pred = predictions.copy()
    pred["candidate_id"] = candidate.candidate_id
    pred["family"] = candidate.family
    pred["feature_recipe"] = candidate.feature_recipe
    pred["model_type"] = candidate.model_type
    pred["threshold_method"] = candidate.threshold_method
    pred["calibration_method"] = candidate.calibration_method
    pred["threshold"] = 0.5
    pred["n_features"] = 12
    return pred


def _check_anchor_reproduction(config: dict[str, Any], anchor_metrics: Any) -> dict[str, Any]:
    expected = get_nested(config, f"{CAMPAIGN_SECTION}.prior_d3_lite_anchor.expected_metrics", {})
    tolerance = float(get_nested(config, f"{CAMPAIGN_SECTION}.prior_d3_lite_anchor.tolerance", 0.001))
    rows = []
    passed = True
    for split_name in OFFICIAL_SPLITS:
        exp = expected.get(split_name, {}) if isinstance(expected, dict) else {}
        for metric, column in [("balanced_accuracy", "balanced_accuracy"), ("roc_auc", "roc_auc")]:
            actual = _metric_value(anchor_metrics, split_name, column, TRIAL_LEVEL)
            target = _safe_float(exp.get(metric))
            delta = None if actual is None or target is None else float(actual - target)
            ok = bool(delta is not None and abs(delta) <= tolerance)
            passed = passed and ok
            rows.append(
                {
                    "split_name": split_name,
                    "metric": metric,
                    "expected": target,
                    "actual": actual,
                    "delta": delta,
                    "tolerance": tolerance,
                    "passed": ok,
                }
            )
    return {"status": "passed" if passed else "failed", "rows": rows, "tolerance": tolerance}


def _load_logistic_anchor(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    pd = _pd()
    path = Path(str(get_nested(config, f"{CAMPAIGN_SECTION}.local_logistic_anchor.metrics_path")))
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return {"status": "missing", "path": str(path), "trial_internal_simple_mean_ba": None}
    frame = pd.read_csv(path)
    rows = frame[frame["split_name"].isin(OFFICIAL_SPLITS)].copy()
    if "evaluation_level" in rows:
        rows = rows[rows["evaluation_level"].astype(str).eq(TRIAL_LEVEL)]
    ba = pd.to_numeric(rows.get("balanced_accuracy"), errors="coerce")
    auc = pd.to_numeric(rows.get("roc_auc"), errors="coerce")
    return {
        "status": "present",
        "path": str(path),
        "trial_internal_simple_mean_ba": float(ba.mean()) if ba.notna().any() else None,
        "trial_internal_simple_mean_auroc": float(auc.mean()) if auc.notna().any() else None,
        "rows": rows.to_dict("records"),
        "rerun_performed": False,
    }


def _evaluate_candidate_inner(
    config: dict[str, Any],
    candidate: CandidateSpec,
    fold_data: dict[tuple[str, int], dict[str, Any]],
    ia: Any,
    prohibited: set[str],
    feature_cache: dict[tuple[str, str, int, str, int], dict[str, Any]],
) -> tuple[dict[str, Any], Any, Any]:
    pd = _pd()
    section = _section(config)
    n_jobs = int(section.get("n_jobs", 1))
    val_fraction = float(get_nested(config, f"{CAMPAIGN_SECTION}.inner_validation.fraction", 0.25))
    threshold_count = int(get_nested(config, f"{CAMPAIGN_SECTION}.inner_validation.threshold_grid_size", 99))
    thresholds = [float(x) for x in _np().linspace(0.01, 0.99, threshold_count)]
    fold_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for (split_name, fold_id), value in sorted(fold_data.items()):
        cache_key = ("feature_recipe", candidate.feature_recipe, int(candidate.seed), split_name, int(fold_id))
        if cache_key in feature_cache:
            cached = feature_cache[cache_key]
            inner_train = cached["inner_train"]
            inner_val = cached["inner_val"]
            train_aug = cached["train_aug"]
            val_aug = cached["val_aug"]
            columns = cached["columns"]
            inner_diag = dict(cached["inner_diag"])
            feature_diag = dict(cached["feature_diag"])
            feature_diag["candidate_id"] = candidate.candidate_id
        else:
            train = value["train"].copy()
            inner_train, inner_val, inner_diag = _inner_split(
                train,
                split_name,
                int(candidate.seed) + int(fold_id) * 31 + len(split_name),
                val_fraction,
            )
            fit_ids = set(inner_train["unique_trial_id"].astype(str))
            apply_ids = fit_ids.union(set(inner_val["unique_trial_id"].astype(str)))
            features, feature_diag = _trial_features_for_recipe(
                fit_ids,
                apply_ids,
                ia,
                candidate,
                prohibited,
            )
            train_aug = (
                inner_train.merge(features, on=["sample_id", "unique_trial_id"], how="left")
                if not features.empty
                else inner_train
            )
            val_aug = (
                inner_val.merge(features, on=["sample_id", "unique_trial_id"], how="left")
                if not features.empty
                else inner_val
            )
            columns = _feature_columns(train_aug, prohibited)
            feature_cache[cache_key] = {
                "inner_train": inner_train,
                "inner_val": inner_val,
                "train_aug": train_aug,
                "val_aug": val_aug,
                "columns": columns,
                "inner_diag": dict(inner_diag),
                "feature_diag": dict(feature_diag),
            }
        _, val_pred, skip_reason = _fit_score_fold(
            train_aug,
            val_aug,
            columns,
            candidate,
            threshold=0.5,
            n_jobs=n_jobs,
        )
        if not val_pred.empty:
            threshold = 0.5
            threshold_score = None
            if candidate.threshold_method == "inner_balanced_accuracy":
                threshold, threshold_score = _select_threshold(val_pred["y_true"], val_pred["y_score"], thresholds)
            metric = _metrics_at_threshold(val_pred["y_true"], val_pred["y_score"], threshold)
            fold_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "feature_recipe": candidate.feature_recipe,
                    "model_type": candidate.model_type,
                    "threshold_method": candidate.threshold_method,
                    "calibration_method": candidate.calibration_method,
                    "split_name": split_name,
                    "fold_id": int(fold_id),
                    "n_features": int(len(columns)),
                    "n_inner_train": int(len(inner_train)),
                    "n_inner_val": int(len(inner_val)),
                    "threshold": float(threshold),
                    "threshold_selection_metric": threshold_score,
                    "status": "complete",
                    "skip_reason": "",
                    **metric,
                }
            )
        else:
            fold_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "feature_recipe": candidate.feature_recipe,
                    "model_type": candidate.model_type,
                    "threshold_method": candidate.threshold_method,
                    "calibration_method": candidate.calibration_method,
                    "split_name": split_name,
                    "fold_id": int(fold_id),
                    "n_features": int(len(columns)),
                    "n_inner_train": int(len(inner_train)),
                    "n_inner_val": int(len(inner_val)),
                    "threshold": None,
                    "threshold_selection_metric": None,
                    "roc_auc": None,
                    "pr_auc": None,
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "sensitivity": None,
                    "specificity": None,
                    "brier_score": None,
                    "ece": None,
                    "status": "skipped",
                    "skip_reason": skip_reason or feature_diag.get("skip_reason", "no_predictions"),
                }
            )
        diagnostics.append(
            {
                "candidate_id": candidate.candidate_id,
                "split_name": split_name,
                "fold_id": int(fold_id),
                **inner_diag,
                **{f"feature_{key}": value for key, value in feature_diag.items()},
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    complete = fold_frame[fold_frame["status"].eq("complete")].copy()
    split_scores = complete.groupby("split_name", dropna=False)["balanced_accuracy"].mean()
    split_aurocs = complete.groupby("split_name", dropna=False)["roc_auc"].mean()
    selection_score = float(split_scores.mean()) if not split_scores.empty else None
    auroc_score = float(split_aurocs.mean()) if not split_aurocs.empty else None
    summary = {
        **candidate.as_dict(),
        "selection_metric": "inner_validation_internal_simple_mean_balanced_accuracy",
        "selection_score": selection_score,
        "inner_internal_simple_mean_auroc": auroc_score,
        "complete_folds": int(fold_frame["status"].eq("complete").sum()) if not fold_frame.empty else 0,
        "evaluated_folds": int(len(fold_frame)),
        "mean_threshold": float(pd.to_numeric(fold_frame["threshold"], errors="coerce").mean())
        if not fold_frame.empty
        else None,
        "status": "complete" if selection_score is not None else "skipped",
        "skip_reason": "" if selection_score is not None else "no_complete_inner_validation_folds",
    }
    for split_name in OFFICIAL_SPLITS:
        summary[f"{split_name}_inner_balanced_accuracy"] = (
            float(split_scores.get(split_name)) if split_name in split_scores else None
        )
        summary[f"{split_name}_inner_roc_auc"] = (
            float(split_aurocs.get(split_name)) if split_name in split_aurocs else None
        )
    return summary, fold_frame, pd.DataFrame(diagnostics)


def _evaluate_candidate_test(
    config: dict[str, Any],
    candidate: CandidateSpec,
    fold_data: dict[tuple[str, int], dict[str, Any]],
    inner_fold_metrics: Any,
    ia: Any,
    prohibited: set[str],
    feature_cache: dict[tuple[str, str, int, str, int], dict[str, Any]],
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    pd = _pd()
    section = _section(config)
    n_jobs = int(section.get("n_jobs", 1))
    threshold_map: dict[tuple[str, int], float] = {}
    candidate_inner = (
        inner_fold_metrics[inner_fold_metrics["candidate_id"].eq(candidate.candidate_id)].copy()
        if not inner_fold_metrics.empty
        else pd.DataFrame()
    )
    for _, row in candidate_inner.iterrows():
        if row.get("threshold") == row.get("threshold"):
            threshold_map[(str(row["split_name"]), int(row["fold_id"]))] = float(row["threshold"])
    global_threshold = (
        float(pd.to_numeric(candidate_inner["threshold"], errors="coerce").mean())
        if not candidate_inner.empty
        else 0.5
    )
    if not math.isfinite(global_threshold):
        global_threshold = 0.5
    prediction_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for (split_name, fold_id), value in sorted(fold_data.items()):
        train = value["train"].copy()
        test = value["test"].copy()
        fit_ids = set(train["unique_trial_id"].astype(str))
        apply_ids = fit_ids.union(set(test["unique_trial_id"].astype(str)))
        cache_key = (
            "test_feature_recipe",
            candidate.feature_recipe,
            int(candidate.seed),
            str(split_name),
            int(fold_id),
        )
        if cache_key in feature_cache:
            cached = feature_cache[cache_key]
            features = cached["features"]
            feature_diag = dict(cached["diagnostics"])
            feature_diag["candidate_id"] = candidate.candidate_id
        else:
            features, feature_diag = _trial_features_for_recipe(fit_ids, apply_ids, ia, candidate, prohibited)
            feature_cache[cache_key] = {
                "features": features,
                "diagnostics": dict(feature_diag),
            }
        train_aug = train.merge(features, on=["sample_id", "unique_trial_id"], how="left") if not features.empty else train
        test_aug = test.merge(features, on=["sample_id", "unique_trial_id"], how="left") if not features.empty else test
        columns = _feature_columns(train_aug, prohibited)
        threshold = threshold_map.get((split_name, int(fold_id)), global_threshold)
        if candidate.threshold_method == "fixed_0_5":
            threshold = 0.5
        _, test_pred, skip_reason = _fit_score_fold(
            train_aug,
            test_aug,
            columns,
            candidate,
            threshold=threshold,
            n_jobs=n_jobs,
        )
        if not test_pred.empty:
            test_pred["n_features"] = int(len(columns))
            prediction_rows.extend(_prediction_rows(test_pred, candidate, split_name, int(fold_id)))
        train_participants = set(train["participant_id"].astype(str)) if "participant_id" in train else set()
        test_participants = set(test["participant_id"].astype(str)) if "participant_id" in test else set()
        train_texts = set(train["text_id"].astype(str)) if "text_id" in train else set()
        test_texts = set(test["text_id"].astype(str)) if "text_id" in test else set()
        diagnostics.append(
            {
                "candidate_id": candidate.candidate_id,
                "split_name": split_name,
                "fold_id": int(fold_id),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_features": int(len(columns)),
                "threshold": float(threshold),
                "status": "complete" if not test_pred.empty else "skipped",
                "skip_reason": skip_reason,
                "heldout_reader_rows_used_for_feature_fit": bool(
                    split_name in {"unseen_reader", "unseen_reader_and_text"}
                    and train_participants.intersection(test_participants)
                ),
                "heldout_text_rows_used_for_feature_fit": bool(
                    split_name in {"unseen_text", "unseen_reader_and_text"}
                    and train_texts.intersection(test_texts)
                ),
                **{f"feature_{key}": value for key, value in feature_diag.items()},
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    trial_metrics = _metric_frame(predictions, candidate, evaluation_level=TRIAL_LEVEL)
    reader_metrics = _metric_frame(_reader_aggregate(predictions), candidate, evaluation_level=READER_LEVEL)
    fold_metrics = _fold_level_metrics(predictions, candidate, TRIAL_LEVEL)
    diagnostics_frame = pd.DataFrame(diagnostics)
    return trial_metrics, reader_metrics, fold_metrics, predictions, {"diagnostics": diagnostics_frame}


def _denied_predictors_present(columns: list[str], prohibited: set[str]) -> list[str]:
    lower = {column.lower() for column in prohibited}
    return sorted(column for column in columns if column in prohibited or column.lower() in lower)


def _build_leakage_report(
    config: dict[str, Any],
    diagnostics: Any,
    feature_columns: list[str],
) -> dict[str, Any]:
    pd = _pd()
    prohibited = set(_section(config).get("prohibited_features", [])) | PROHIBITED_FEATURES
    denied = _denied_predictors_present(feature_columns, prohibited)
    report = {
        "official_eyebench_data_folds_only": True,
        "no_test_label_tuning": True,
        "selection_source": "train_inner_validation_only_for_new_candidates",
        "official_test_labels_used_for_threshold_or_hyperparameters": False,
        "no_synthetic_predictions": True,
        "no_random_predictions": True,
        "no_full_prepared_copco_join": True,
        "no_leaderboard_method_reruns": True,
        "candidate_0000_preserved": True,
        "best_so_far_initialized_from_candidate_0000": True,
        "heldout_reader_rows_used_for_feature_fit": bool(
            diagnostics.get("heldout_reader_rows_used_for_feature_fit", pd.Series(dtype=bool)).any()
        )
        if not diagnostics.empty
        else False,
        "heldout_text_rows_used_for_feature_fit": bool(
            diagnostics.get("heldout_text_rows_used_for_feature_fit", pd.Series(dtype=bool)).any()
        )
        if not diagnostics.empty
        else False,
        "reader_group_used_in_residualization": False,
        "denied_predictors_present": denied,
        "status": "passed",
    }
    if (
        report["heldout_reader_rows_used_for_feature_fit"]
        or report["heldout_text_rows_used_for_feature_fit"]
        or report["reader_group_used_in_residualization"]
        or denied
    ):
        report["status"] = "failed"
    return report


def run_d3_eyebench_own_method_score_max(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    stage: str = "full",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    section = _section(config)
    require_slurm = bool(section.get("require_slurm_job", True))
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if require_slurm and not slurm_job_id:
        raise RuntimeError("SLURM_JOB_ID is empty; refusing D3 score-max run on login node")
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, root)
    dirs["repo_analysis"].mkdir(parents=True, exist_ok=True)
    dirs["result_analysis"].mkdir(parents=True, exist_ok=True)

    config_report = validate_d3_eyebench_own_method_score_max_config(config)
    _write_json(out / "config_validation.json", config_report)
    _write_json(
        out / "preflight" / "preflight_report.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "repo_root": str(root),
            "git_sha": _git_sha(root),
            "git_status_short": _git_status_short(root),
            "slurm_job_id": slurm_job_id,
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "slurm_mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "stage": stage,
        },
    )
    candidates = build_candidate_specs(config)
    _write_json(out / "candidate_manifest.json", [candidate.as_dict() for candidate in candidates])
    _write_json(dirs["repo_analysis"] / "candidate_manifest.json", [candidate.as_dict() for candidate in candidates])
    anchor_candidate = candidates[0]
    if anchor_candidate.candidate_id != "candidate_0000" or not anchor_candidate.anchor_exact:
        raise RuntimeError("candidate_0000 is not the exact D3_Lite anchor")

    reference = _official_reference_table(_to_sota_config(config), root)
    logistic_anchor = _load_logistic_anchor(config, root)
    samples, ia = load_official_processed_features(_to_sota_config(config), out, root)
    splits = build_official_split_labels(_to_sota_config(config), out, samples, root)
    split_errors, split_summaries = validate_official_split_labels(splits)
    _write_json(out / "splits" / "official_split_validation.json", {"errors": split_errors, "summaries": split_summaries})
    if config_report["status"] != "passed" or split_errors or samples.empty or ia.empty or splits.empty:
        decision = _blocked_decision(
            "blocked_by_data" if split_errors or samples.empty or ia.empty or splits.empty else "blocked_by_environment",
            "config, official data, or official folds unavailable",
            config_report,
            split_errors,
            logistic_anchor,
            reference,
        )
        _write_final_outputs(dirs, out, decision, {}, {})
        return {"status": "blocked", "output_dir": str(out), **decision}

    fold_data = _fold_cache(samples, splits)
    anchor_metrics_raw, anchor_predictions_raw, anchor_leakage = evaluate_d3_eyebench_lite(
        _to_sota_config(config),
        out,
        dirs,
        samples,
        splits,
        ia,
    )
    anchor_metrics = _standardize_anchor_metrics(anchor_metrics_raw, anchor_candidate)
    anchor_predictions = _standardize_anchor_predictions(anchor_predictions_raw, anchor_candidate)
    anchor_check = _check_anchor_reproduction(config, anchor_metrics)
    _write_json(out / "anchor_reproduction_check.json", anchor_check)
    _write_csv(out / "typ" / "candidate_0000_trial_metrics.csv", anchor_metrics)
    _write_csv(out / "typ" / "candidate_0000_trial_predictions.csv", anchor_predictions)
    _write_csv(dirs["repo_analysis"] / "candidate_0000_trial_metrics.csv", anchor_metrics)
    _write_anchor_report(dirs, anchor_metrics, anchor_check, anchor_leakage)
    if anchor_check["status"] != "passed":
        decision = _blocked_decision(
            "blocked_by_evaluator",
            "candidate_0000 did not reproduce previous D3_EyeBench_Lite within tolerance",
            config_report,
            split_errors,
            logistic_anchor,
            reference,
        )
        decision["anchor_reproduction"] = anchor_check
        _write_mismatch_audit(dirs, anchor_metrics, anchor_check)
        _write_final_outputs(dirs, out, decision, {"anchor_metrics": anchor_metrics}, {})
        return {"status": "blocked", "output_dir": str(out), **decision}
    if stage == "anchor":
        anchor_trial = anchor_metrics[anchor_metrics["evaluation_level"].eq(TRIAL_LEVEL)].copy()
        anchor_reader = anchor_metrics[anchor_metrics["evaluation_level"].eq(READER_LEVEL)].copy()
        anchor_leaderboard = _candidate_leaderboard(
            _pd().DataFrame(),
            anchor_trial,
            anchor_reader,
            _simple_mean(anchor_trial, "balanced_accuracy"),
            _simple_mean(anchor_trial, "roc_auc"),
        )
        decision = _decision_from_results(
            config,
            config_report,
            split_errors,
            reference,
            logistic_anchor,
            anchor_candidate,
            anchor_trial,
            anchor_trial,
            anchor_reader,
            "anchor_only",
            1,
            anchor_check,
            {"status": "passed"},
        )
        _write_all_reports(
            config,
            dirs,
            out,
            decision,
            anchor_trial,
            anchor_reader,
            anchor_metrics.iloc[0:0].copy(),
            anchor_metrics.iloc[0:0].copy(),
            anchor_metrics.iloc[0:0].copy(),
            anchor_leaderboard,
            anchor_check,
            {"status": "passed"},
            [],
        )
        return {"status": "complete", "output_dir": str(out), **decision}

    prohibited = set(section.get("prohibited_features", [])) | PROHIBITED_FEATURES
    feature_cache: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    search_candidates = [candidate for candidate in candidates[1:]]
    no_improvement_rounds = int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.no_improvement_rounds", 12))
    min_delta = float(get_nested(config, f"{CAMPAIGN_SECTION}.budget.minimum_improvement_delta", 0.001))
    anchor_score = _simple_mean(anchor_metrics, "balanced_accuracy")
    anchor_auc = _simple_mean(anchor_metrics, "roc_auc")
    best_inner_score = float(anchor_score) if anchor_score is not None else -1.0
    rounds_since_improvement = 0
    stop_reason = "budget_exhausted"
    summaries: list[dict[str, Any]] = []
    inner_frames = []
    inner_diag_frames = []
    progress_path = out / "candidate_progress.csv"
    for candidate in search_candidates:
        summary, inner_fold, inner_diag = _evaluate_candidate_inner(
            config,
            candidate,
            fold_data,
            ia,
            prohibited,
            feature_cache,
        )
        summaries.append(summary)
        inner_frames.append(inner_fold)
        inner_diag_frames.append(inner_diag)
        _write_csv(progress_path, _pd().DataFrame(summaries))
        print(
            "candidate_progress",
            len(summaries),
            candidate.candidate_id,
            summary.get("selection_score"),
            flush=True,
        )
        score = summary.get("selection_score")
        if score is not None and float(score) > best_inner_score + min_delta:
            best_inner_score = float(score)
            rounds_since_improvement = 0
        else:
            rounds_since_improvement += 1
        if rounds_since_improvement >= no_improvement_rounds:
            stop_reason = "no_improvement_stopping_rule"
            break
    pd = _pd()
    candidate_summary = pd.DataFrame(summaries)
    inner_fold_metrics = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    inner_diagnostics = pd.concat(inner_diag_frames, ignore_index=True) if inner_diag_frames else pd.DataFrame()
    _write_csv(out / "inner_validation_fold_metrics.csv", inner_fold_metrics)
    _write_csv(out / "inner_validation_diagnostics.csv", inner_diagnostics)

    test_eval_top_k = int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.test_eval_top_k", 5))
    completed = candidate_summary[candidate_summary["status"].eq("complete")].copy()
    top_candidates: list[CandidateSpec] = []
    if not completed.empty:
        top_ids = (
            completed.sort_values("selection_score", ascending=False)
            .head(test_eval_top_k)["candidate_id"]
            .astype(str)
            .tolist()
        )
        by_id = {candidate.candidate_id: candidate for candidate in search_candidates}
        top_candidates = [by_id[candidate_id] for candidate_id in top_ids if candidate_id in by_id]
    all_trial_metrics = [anchor_metrics[anchor_metrics["evaluation_level"].eq(TRIAL_LEVEL)].copy()]
    all_reader_metrics = [anchor_metrics[anchor_metrics["evaluation_level"].eq(READER_LEVEL)].copy()]
    all_fold_metrics = [_fold_level_metrics(anchor_predictions, anchor_candidate, TRIAL_LEVEL)]
    all_predictions = [anchor_predictions]
    final_diagnostics = []
    feature_columns_seen: set[str] = set()
    test_feature_cache: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    for candidate in top_candidates:
        trial_metrics, reader_metrics, fold_metrics, predictions, extra = _evaluate_candidate_test(
            config,
            candidate,
            fold_data,
            inner_fold_metrics,
            ia,
            prohibited,
            test_feature_cache,
        )
        all_trial_metrics.append(trial_metrics)
        all_reader_metrics.append(reader_metrics)
        all_fold_metrics.append(fold_metrics)
        all_predictions.append(predictions)
        final_diagnostics.append(extra["diagnostics"])
        if not predictions.empty:
            feature_columns_seen.update(
                [column for column in predictions.columns if str(column).startswith("d3_")]
            )
    trial_all = pd.concat(all_trial_metrics, ignore_index=True) if all_trial_metrics else pd.DataFrame()
    reader_all = pd.concat(all_reader_metrics, ignore_index=True) if all_reader_metrics else pd.DataFrame()
    fold_all = pd.concat(all_fold_metrics, ignore_index=True) if all_fold_metrics else pd.DataFrame()
    predictions_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    diagnostics_all = pd.concat(final_diagnostics, ignore_index=True) if final_diagnostics else pd.DataFrame()
    leakage = _build_leakage_report(config, diagnostics_all, sorted(feature_columns_seen))

    leaderboard = _candidate_leaderboard(candidate_summary, trial_all, reader_all, anchor_score, anchor_auc)
    _write_csv(out / "candidate_leaderboard.csv", leaderboard)
    _write_csv(dirs["repo_analysis"] / "candidate_leaderboard.csv", leaderboard)
    _write_csv(out / "fold_level_metrics.csv", fold_all)
    _write_csv(out / "typ" / "d3_own_method_trial_metrics.csv", trial_all)
    _write_csv(out / "typ" / "d3_own_method_reader_aggregated_metrics.csv", reader_all)
    _write_csv(out / "typ" / "d3_own_method_trial_predictions.csv", predictions_all)
    _write_csv(out / "final_candidate_diagnostics.csv", diagnostics_all)
    _write_json(out / "leakage_validation_report.json", leakage)

    final_candidate, final_metrics, stop_reason = _select_final_candidate(
        config,
        candidates,
        completed,
        trial_all,
        anchor_candidate,
        stop_reason,
    )
    decision = _decision_from_results(
        config,
        config_report,
        split_errors,
        reference,
        logistic_anchor,
        final_candidate,
        final_metrics,
        trial_all,
        reader_all,
        stop_reason,
        len(summaries) + 1,
        anchor_check,
        leakage,
    )
    _write_all_reports(
        config,
        dirs,
        out,
        decision,
        trial_all,
        reader_all,
        fold_all,
        inner_fold_metrics,
        diagnostics_all,
        leaderboard,
        anchor_check,
        leakage,
        [candidate.as_dict() for candidate in top_candidates],
    )
    return {"status": "complete", "output_dir": str(out), **decision}


def _candidate_leaderboard(
    inner_summary: Any,
    trial_metrics: Any,
    reader_metrics: Any,
    anchor_ba: float | None,
    anchor_auc: float | None,
) -> Any:
    pd = _pd()
    rows: list[dict[str, Any]] = []
    candidates = []
    if not inner_summary.empty:
        candidates.extend(inner_summary.to_dict("records"))
    if not any(row.get("candidate_id") == "candidate_0000" for row in candidates):
        candidates.insert(
            0,
            {
                "candidate_id": "candidate_0000",
                "family": "d3_lite_anchor",
                "feature_recipe": "d3_lite_exact",
                "model_type": "official_lite_logistic",
                "threshold_method": "fixed_0_5",
                "calibration_method": "none",
                "selection_score": anchor_ba,
                "inner_internal_simple_mean_auroc": anchor_auc,
                "status": "complete",
            },
        )
    for item in candidates:
        candidate_id = item.get("candidate_id")
        test_rows = trial_metrics[trial_metrics["candidate_id"].astype(str).eq(str(candidate_id))]
        reader_rows = reader_metrics[reader_metrics["candidate_id"].astype(str).eq(str(candidate_id))]
        row = dict(item)
        row["test_evaluated"] = not test_rows.empty
        row["test_internal_simple_mean_ba"] = _simple_mean(test_rows, "balanced_accuracy")
        row["test_internal_simple_mean_auroc"] = _simple_mean(test_rows, "roc_auc")
        row["reader_internal_simple_mean_ba"] = _simple_mean(reader_rows, "balanced_accuracy", READER_LEVEL)
        row["reader_internal_simple_mean_auroc"] = _simple_mean(reader_rows, "roc_auc", READER_LEVEL)
        for split_name in OFFICIAL_SPLITS:
            row[f"{split_name}_test_ba"] = _metric_value(test_rows, split_name, "balanced_accuracy")
            row[f"{split_name}_test_auroc"] = _metric_value(test_rows, split_name, "roc_auc")
        rows.append(row)
    return pd.DataFrame(rows)


def _select_final_candidate(
    config: dict[str, Any],
    candidates: list[CandidateSpec],
    completed_inner: Any,
    trial_metrics: Any,
    anchor_candidate: CandidateSpec,
    stop_reason: str,
) -> tuple[CandidateSpec, Any, str]:
    min_delta = float(get_nested(config, f"{CAMPAIGN_SECTION}.budget.minimum_improvement_delta", 0.001))
    anchor_rows = trial_metrics[trial_metrics["candidate_id"].eq("candidate_0000")].copy()
    anchor_score = _simple_mean(anchor_rows, "balanced_accuracy")
    if completed_inner.empty:
        return anchor_candidate, anchor_rows, "no_completed_new_candidates"
    locked_id = str(
        completed_inner.sort_values("selection_score", ascending=False).iloc[0]["candidate_id"]
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    locked_candidate = by_id.get(locked_id, anchor_candidate)
    locked_rows = trial_metrics[trial_metrics["candidate_id"].astype(str).eq(locked_id)].copy()
    locked_score = _simple_mean(locked_rows, "balanced_accuracy")
    if (
        locked_score is not None
        and anchor_score is not None
        and float(locked_score) > float(anchor_score) + min_delta
    ):
        return locked_candidate, locked_rows, stop_reason
    return anchor_candidate, anchor_rows, "no_locked_candidate_improved_over_candidate_0000"


def _blocked_decision(
    category: str,
    reason: str,
    config_report: dict[str, Any],
    split_errors: list[str],
    logistic_anchor: dict[str, Any],
    reference: Any,
) -> dict[str, Any]:
    return {
        "final_decision_category": category,
        "reason": reason,
        "candidate_0000_reproduced": False,
        "best_candidate_id": None,
        "method_evidence_improved": False,
        "official_sota_claimed": False,
        "official_leaderboard_methods_rerun": False,
        "wandb_online_api_used": False,
        "config_validation": config_report,
        "split_errors": split_errors,
        "local_logistic_anchor": logistic_anchor,
        "published_reference_rows": reference.to_dict("records") if hasattr(reference, "to_dict") else [],
    }


def _decision_from_results(
    config: dict[str, Any],
    config_report: dict[str, Any],
    split_errors: list[str],
    reference: Any,
    logistic_anchor: dict[str, Any],
    final_candidate: CandidateSpec,
    final_metrics: Any,
    trial_all: Any,
    reader_all: Any,
    stop_reason: str,
    candidate_count: int,
    anchor_check: dict[str, Any],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    min_delta = float(get_nested(config, f"{CAMPAIGN_SECTION}.budget.minimum_improvement_delta", 0.001))
    anchor_rows = trial_all[trial_all["candidate_id"].eq("candidate_0000")].copy() if not trial_all.empty else final_metrics
    final_ba = _simple_mean(final_metrics, "balanced_accuracy")
    final_auc = _simple_mean(final_metrics, "roc_auc")
    anchor_ba = _simple_mean(anchor_rows, "balanced_accuracy")
    anchor_auc = _simple_mean(anchor_rows, "roc_auc")
    logistic_ba = logistic_anchor.get("trial_internal_simple_mean_ba")
    target_visible = get_nested(config, f"{CAMPAIGN_SECTION}.published_leaderboard_snapshot.target_balanced_accuracy")
    target_visible = _safe_float(target_visible)
    exploratory_best = _best_test_row(trial_all)
    locked_improved = bool(final_ba is not None and anchor_ba is not None and final_ba > anchor_ba + min_delta)
    exploratory_improved = bool(
        exploratory_best.get("test_internal_simple_mean_ba") is not None
        and anchor_ba is not None
        and float(exploratory_best["test_internal_simple_mean_ba"]) > anchor_ba + min_delta
    )
    if config_report.get("status") != "passed" or split_errors:
        category = "blocked_by_data"
    elif leakage.get("status") != "passed":
        category = "blocked_by_evaluator"
    elif locked_improved:
        category = "d3_method_improved"
    elif exploratory_improved:
        category = "d3_method_exploratory_gain_only"
    elif final_ba is not None and logistic_ba is not None and final_ba >= float(logistic_ba) - 0.005:
        category = "d3_method_competitive_but_not_improved"
    else:
        category = "d3_method_not_improved"
    return {
        "final_decision_category": category,
        "stop_reason": stop_reason,
        "candidate_count_evaluated": int(candidate_count),
        "candidate_0000_reproduced": anchor_check.get("status") == "passed",
        "best_so_far_initialized_from_candidate_0000": True,
        "final_best_not_worse_than_candidate_0000": bool(
            final_ba is not None and anchor_ba is not None and final_ba >= anchor_ba - 1e-12
        ),
        "best_candidate": final_candidate.as_dict(),
        "best_candidate_id": final_candidate.candidate_id,
        "best_candidate_internal_simple_mean_ba": final_ba,
        "best_candidate_internal_simple_mean_auroc": final_auc,
        "candidate_0000_internal_simple_mean_ba": anchor_ba,
        "candidate_0000_internal_simple_mean_auroc": anchor_auc,
        "delta_vs_candidate_0000_ba": None if final_ba is None or anchor_ba is None else final_ba - anchor_ba,
        "delta_vs_candidate_0000_auroc": None if final_auc is None or anchor_auc is None else final_auc - anchor_auc,
        "local_logistic_anchor": logistic_anchor,
        "published_leaderboard_snapshot": dict(
            get_nested(config, f"{CAMPAIGN_SECTION}.published_leaderboard_snapshot", {})
        ),
        "published_reference_rows": reference.to_dict("records") if hasattr(reference, "to_dict") else [],
        "visible_official_average_used_as_primary": False,
        "internal_simple_mean_used": True,
        "official_sota_claimed": False,
        "official_leaderboard_methods_rerun": False,
        "wandb_online_api_used": False,
        "test_label_tuning": False,
        "synthetic_predictions_used": False,
        "random_predictions_used": False,
        "full_prepared_copco_join_used": False,
        "method_evidence_improved": locked_improved or exploratory_improved,
        "locked_inner_validation_candidate_improved": locked_improved,
        "exploratory_test_evaluated_candidate_improved": exploratory_improved,
        "exploratory_best_test_candidate": exploratory_best,
        "published_target_visible_average_ba": target_visible,
        "leakage_status": leakage.get("status"),
    }


def _best_test_row(trial_all: Any) -> dict[str, Any]:
    pd = _pd()
    if trial_all.empty:
        return {}
    rows = []
    for candidate_id, group in trial_all.groupby("candidate_id", dropna=False):
        rows.append(
            {
                "candidate_id": candidate_id,
                "test_internal_simple_mean_ba": _simple_mean(group, "balanced_accuracy"),
                "test_internal_simple_mean_auroc": _simple_mean(group, "roc_auc"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    return (
        frame.sort_values("test_internal_simple_mean_ba", ascending=False, na_position="last")
        .iloc[0]
        .to_dict()
    )


def _write_anchor_report(
    dirs: dict[str, Path],
    anchor_metrics: Any,
    anchor_check: dict[str, Any],
    anchor_leakage: dict[str, Any],
) -> None:
    lines = [
        "# Anchor Reproduction Report",
        "",
        f"- Candidate: `candidate_0000` exact previous `{SOTA_MODEL}` adapter",
        f"- Reproduction status: `{anchor_check['status']}`",
        f"- Tolerance: `{anchor_check['tolerance']}`",
        f"- Held-out reader leakage: {anchor_leakage.get('heldout_reader_rows_used_for_fit')}",
        f"- Held-out text leakage: {anchor_leakage.get('heldout_text_rows_used_for_fit')}",
        f"- Reader group used in residualization: {anchor_leakage.get('reader_group_used')}",
        "",
        "## Expected vs Actual",
        _markdown_table(anchor_check["rows"], list(anchor_check["rows"][0]), max_rows=20),
        "",
        "## Candidate Metrics",
        _markdown_table(anchor_metrics.to_dict("records"), METRIC_COLUMNS, max_rows=20),
    ]
    _write_report(dirs, "anchor_reproduction_report.md", "\n".join(lines))


def _write_mismatch_audit(dirs: dict[str, Path], anchor_metrics: Any, anchor_check: dict[str, Any]) -> None:
    lines = [
        "# Candidate 0000 Metric/Pipeline Mismatch Audit",
        "",
        "Optimization stopped before candidate search because `candidate_0000` did not reproduce "
        "the previous D3_EyeBench_Lite metrics within tolerance.",
        "",
        "## Mismatch Rows",
        _markdown_table(anchor_check["rows"], list(anchor_check["rows"][0]), max_rows=20),
        "",
        "## Actual Metrics",
        _markdown_table(anchor_metrics.to_dict("records"), METRIC_COLUMNS, max_rows=20),
    ]
    _write_report(dirs, "metric_pipeline_mismatch_audit.md", "\n".join(lines))


def _write_final_outputs(
    dirs: dict[str, Path],
    out: Path,
    decision: dict[str, Any],
    frames: dict[str, Any],
    reports: dict[str, Any],
) -> None:
    del frames, reports
    _write_json(out / "final_decision.json", decision)
    _write_json(dirs["repo_analysis"] / "final_decision.json", decision)
    _write_report(
        dirs,
        "final_decision_report.md",
        "# Final Decision Report\n\n"
        f"- Decision category: `{decision.get('final_decision_category')}`\n"
        f"- Reason: {decision.get('reason', decision.get('stop_reason', 'n/a'))}\n"
        f"- Official SOTA claimed: {decision.get('official_sota_claimed', False)}\n",
    )


def _write_all_reports(
    config: dict[str, Any],
    dirs: dict[str, Path],
    out: Path,
    decision: dict[str, Any],
    trial_all: Any,
    reader_all: Any,
    fold_all: Any,
    inner_fold_metrics: Any,
    diagnostics_all: Any,
    leaderboard: Any,
    anchor_check: dict[str, Any],
    leakage: dict[str, Any],
    test_evaluated_candidates: list[dict[str, Any]],
) -> None:
    _write_json(out / "final_decision.json", decision)
    _write_json(dirs["repo_analysis"] / "final_decision.json", decision)
    _write_json(out / "leakage_validation_report.json", leakage)
    _write_json(dirs["repo_analysis"] / "leakage_validation_report.json", leakage)
    _write_csv(dirs["repo_analysis"] / "trial_metrics.csv", trial_all)
    _write_csv(dirs["repo_analysis"] / "reader_aggregated_metrics.csv", reader_all)
    _write_csv(dirs["repo_analysis"] / "fold_level_metrics.csv", fold_all)
    _write_csv(dirs["repo_analysis"] / "inner_validation_fold_metrics.csv", inner_fold_metrics)
    _write_csv(dirs["repo_analysis"] / "final_candidate_diagnostics.csv", diagnostics_all)
    _write_csv(out / "candidate_leaderboard.csv", leaderboard)
    _write_csv(dirs["repo_analysis"] / "candidate_leaderboard.csv", leaderboard)
    _write_metric_alignment_report(dirs, decision)
    _write_candidate_search_manifest(dirs, config, leaderboard, test_evaluated_candidates)
    _write_best_candidate_report(dirs, decision, trial_all, reader_all, fold_all)
    _write_ablation_report(dirs, leaderboard)
    _write_calibration_report(dirs, inner_fold_metrics, trial_all)
    _write_reader_report(dirs, reader_all)
    _write_both_unseen_report(dirs, trial_all, reader_all)
    _write_leakage_reports(dirs, leakage, diagnostics_all)
    _write_evidence_summary(dirs, decision, trial_all, reader_all, leaderboard)
    _write_final_decision_report(dirs, decision, anchor_check)
    manifest = {
        "status": "complete",
        "run_name": get_nested(config, "run.name", "d3_eyebench_own_method_score_max_v2"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out),
        "final_decision_category": decision["final_decision_category"],
        "best_candidate_id": decision["best_candidate_id"],
        "candidate_count_evaluated": decision["candidate_count_evaluated"],
        "candidate_0000_reproduced": decision["candidate_0000_reproduced"],
        "official_sota_claimed": False,
        "official_leaderboard_methods_rerun": False,
    }
    _write_json(out / "manifest.json", manifest)
    _write_json(dirs["repo_analysis"] / "manifest.json", manifest)


def _write_metric_alignment_report(dirs: dict[str, Path], decision: dict[str, Any]) -> None:
    lines = [
        "# Metric Alignment Report",
        "",
        "- Primary metric: per-regime trial-level balanced accuracy on official "
        "`CopCo_TYP` Test folds, with internal simple mean reported separately.",
        "- AUROC is computed from probabilities/scores, not hard labels.",
        "- The internal simple mean is not called the official leaderboard average.",
        "- Thresholds for new candidates are fixed at 0.5 or selected only on train/inner-validation.",
        "- Candidate selection for the locked candidate is based on inner validation; top-k Test "
        "evaluations are labeled exploratory evidence.",
        "",
        f"- Visible official average used as primary: {decision['visible_official_average_used_as_primary']}",
        f"- Internal simple mean used: {decision['internal_simple_mean_used']}",
        f"- Test-label tuning: {decision['test_label_tuning']}",
    ]
    _write_report(dirs, "metric_alignment_report.md", "\n".join(lines))


def _write_candidate_search_manifest(
    dirs: dict[str, Path],
    config: dict[str, Any],
    leaderboard: Any,
    test_evaluated_candidates: list[dict[str, Any]],
) -> None:
    lines = [
        "# Candidate Search Manifest",
        "",
        f"- Max candidates: {get_nested(config, f'{CAMPAIGN_SECTION}.budget.max_candidates')}",
        f"- Candidate rows: {len(leaderboard)}",
        f"- Test-evaluated new candidates: {len(test_evaluated_candidates)}",
        "- `candidate_0000` is the exact previous D3_EyeBench_Lite adapter.",
        "- `best_so_far` initializes from `candidate_0000`.",
        "- No official leaderboard methods were rerun.",
        "",
        "## Leaderboard Preview",
        _markdown_table(leaderboard.to_dict("records"), list(leaderboard.columns), max_rows=20),
    ]
    _write_report(dirs, "candidate_search_manifest.md", "\n".join(lines))


def _write_best_candidate_report(
    dirs: dict[str, Path],
    decision: dict[str, Any],
    trial_all: Any,
    reader_all: Any,
    fold_all: Any,
) -> None:
    best_id = decision["best_candidate_id"]
    trial = trial_all[trial_all["candidate_id"].astype(str).eq(str(best_id))]
    reader = reader_all[reader_all["candidate_id"].astype(str).eq(str(best_id))]
    folds = fold_all[fold_all["candidate_id"].astype(str).eq(str(best_id))] if not fold_all.empty else fold_all
    lines = [
        "# Best Candidate Report",
        "",
        f"- Best candidate: `{best_id}`",
        f"- Decision category: `{decision['final_decision_category']}`",
        f"- Internal simple mean BA: {decision['best_candidate_internal_simple_mean_ba']}",
        f"- Internal simple mean AUROC: {decision['best_candidate_internal_simple_mean_auroc']}",
        f"- Delta vs candidate_0000 BA: {decision['delta_vs_candidate_0000_ba']}",
        f"- Official SOTA claimed: {decision['official_sota_claimed']}",
        "",
        "## Trial Metrics",
        _markdown_table(trial.to_dict("records"), METRIC_COLUMNS, max_rows=20),
        "",
        "## Reader-Aggregated Metrics",
        _markdown_table(reader.to_dict("records"), METRIC_COLUMNS, max_rows=20),
        "",
        "## Fold-Level Metrics",
        _markdown_table(folds.to_dict("records"), list(folds.columns), max_rows=40) if not folds.empty else "No fold rows.",
    ]
    _write_report(dirs, "best_candidate_report.md", "\n".join(lines))


def _write_ablation_report(dirs: dict[str, Path], leaderboard: Any) -> None:
    pd = _pd()
    if leaderboard.empty or not {"family", "feature_recipe"}.issubset(leaderboard.columns):
        summary = pd.DataFrame()
    else:
        summary = (
            leaderboard.groupby(["family", "feature_recipe"], dropna=False)
            .agg(
                candidates=("candidate_id", "count"),
                best_inner_ba=("selection_score", "max"),
                best_test_ba=("test_internal_simple_mean_ba", "max"),
                best_test_auroc=("test_internal_simple_mean_auroc", "max"),
            )
            .reset_index()
        )
    lines = [
        "# Feature Family Ablation Report",
        "",
        "This table is grouped by D3 feature recipe. Test values are present only for "
        "`candidate_0000` and train/inner-validation-ranked top candidates.",
        "",
        _markdown_table(summary.to_dict("records"), list(summary.columns), max_rows=40)
        if not summary.empty
        else "No ablation rows.",
    ]
    _write_report(dirs, "feature_family_ablation_report.md", "\n".join(lines))


def _write_calibration_report(dirs: dict[str, Path], inner_fold_metrics: Any, trial_all: Any) -> None:
    lines = [
        "# Calibration And Threshold Report",
        "",
        "- `candidate_0000` uses the exact previous fixed `0.5` threshold.",
        "- New candidate thresholds are either fixed `0.5` or selected on train/inner-validation folds.",
        "- Test labels are not used for threshold selection.",
        "",
        "## Inner-Validation Threshold Rows",
        _markdown_table(inner_fold_metrics.to_dict("records"), list(inner_fold_metrics.columns), max_rows=40)
        if not inner_fold_metrics.empty
        else "No inner-validation rows.",
        "",
        "## Test Threshold Summary",
        _markdown_table(
            trial_all[
                [
                    "candidate_id",
                    "split_name",
                    "evaluation_level",
                    "threshold",
                    "balanced_accuracy",
                    "roc_auc",
                    "brier_score",
                    "ece",
                ]
            ].to_dict("records"),
            [
                "candidate_id",
                "split_name",
                "evaluation_level",
                "threshold",
                "balanced_accuracy",
                "roc_auc",
                "brier_score",
                "ece",
            ],
            max_rows=40,
        )
        if not trial_all.empty
        else "No Test rows.",
    ]
    _write_report(dirs, "calibration_threshold_report.md", "\n".join(lines))


def _write_reader_report(dirs: dict[str, Path], reader_all: Any) -> None:
    lines = [
        "# Reader Aggregation Secondary Report",
        "",
        "Reader-aggregated metrics are secondary evidence only.",
        "",
        _markdown_table(reader_all.to_dict("records"), METRIC_COLUMNS, max_rows=60)
        if not reader_all.empty
        else "No reader-aggregated rows.",
    ]
    _write_report(dirs, "reader_aggregation_secondary_report.md", "\n".join(lines))


def _write_both_unseen_report(dirs: dict[str, Path], trial_all: Any, reader_all: Any) -> None:
    trial = trial_all[trial_all["split_name"].eq("unseen_reader_and_text")] if not trial_all.empty else trial_all
    reader = reader_all[reader_all["split_name"].eq("unseen_reader_and_text")] if not reader_all.empty else reader_all
    lines = [
        "# Both-Unseen Generalization Report",
        "",
        "Both-unseen AUROC and BA are treated as robustness/generalization signals.",
        "",
        "## Trial-Level",
        _markdown_table(trial.to_dict("records"), METRIC_COLUMNS, max_rows=40) if not trial.empty else "No trial rows.",
        "",
        "## Reader-Aggregated",
        _markdown_table(reader.to_dict("records"), METRIC_COLUMNS, max_rows=40)
        if not reader.empty
        else "No reader rows.",
    ]
    _write_report(dirs, "both_unseen_generalization_report.md", "\n".join(lines))


def _write_leakage_reports(dirs: dict[str, Path], leakage: dict[str, Any], diagnostics: Any) -> None:
    lines = [
        "# Leakage Validation Report",
        "",
        f"- Status: `{leakage.get('status')}`",
        f"- Official EyeBench data/folds only: {leakage.get('official_eyebench_data_folds_only')}",
        f"- Test-label tuning: {not leakage.get('no_test_label_tuning')}",
        f"- Synthetic predictions: {not leakage.get('no_synthetic_predictions')}",
        f"- Random predictions: {not leakage.get('no_random_predictions')}",
        f"- Full prepared CopCo join: {not leakage.get('no_full_prepared_copco_join')}",
        f"- Leaderboard method reruns: {not leakage.get('no_leaderboard_method_reruns')}",
        f"- Held-out reader rows used for feature fit: {leakage.get('heldout_reader_rows_used_for_feature_fit')}",
        f"- Held-out text rows used for feature fit: {leakage.get('heldout_text_rows_used_for_feature_fit')}",
        f"- Denied predictors present: {leakage.get('denied_predictors_present')}",
    ]
    _write_report(dirs, "leakage_validation_report.md", "\n".join(lines))
    prohibited_lines = [
        "# Prohibited Predictor Report",
        "",
        f"- Denied predictors present: {leakage.get('denied_predictors_present')}",
        "- `participant_id`, `speech_id`, `text_id`, `fold_id`, exposure counts, and targets are "
        "retained only for grouping/reporting when present, not as model predictors.",
        "",
        "## Final Diagnostics Preview",
        _markdown_table(diagnostics.to_dict("records"), list(diagnostics.columns), max_rows=40)
        if not diagnostics.empty
        else "No final diagnostics rows.",
    ]
    _write_report(dirs, "prohibited_predictor_report.md", "\n".join(prohibited_lines))


def _write_evidence_summary(
    dirs: dict[str, Path],
    decision: dict[str, Any],
    trial_all: Any,
    reader_all: Any,
    leaderboard: Any,
) -> None:
    lines = [
        "# Evidence Summary For Manuscript",
        "",
        "This phase is own-method D3-family evidence generation. It does not claim official SOTA.",
        "",
        f"- Decision category: `{decision['final_decision_category']}`",
        f"- Method evidence improved: {decision['method_evidence_improved']}",
        f"- Official SOTA claimed: {decision['official_sota_claimed']}",
        f"- Best candidate: `{decision['best_candidate_id']}`",
        f"- Candidate_0000 internal simple mean BA: {decision['candidate_0000_internal_simple_mean_ba']}",
        f"- Best internal simple mean BA: {decision['best_candidate_internal_simple_mean_ba']}",
        "- Published references used only as fixed references: True",
        "",
        "## Trial Evidence",
        _markdown_table(trial_all.to_dict("records"), METRIC_COLUMNS, max_rows=60)
        if not trial_all.empty
        else "No trial rows.",
        "",
        "## Candidate Leaderboard",
        _markdown_table(leaderboard.to_dict("records"), list(leaderboard.columns), max_rows=40)
        if not leaderboard.empty
        else "No leaderboard rows.",
        "",
        "## Reader-Aggregated Evidence",
        _markdown_table(reader_all.to_dict("records"), METRIC_COLUMNS, max_rows=60)
        if not reader_all.empty
        else "No reader rows.",
    ]
    _write_report(dirs, "evidence_summary_for_manuscript.md", "\n".join(lines))


def _write_final_decision_report(
    dirs: dict[str, Path],
    decision: dict[str, Any],
    anchor_check: dict[str, Any],
) -> None:
    lines = [
        "# Final Decision Report",
        "",
        f"- Final decision category: `{decision['final_decision_category']}`",
        f"- Stop reason: `{decision['stop_reason']}`",
        f"- Candidate count evaluated: {decision['candidate_count_evaluated']}",
        f"- Candidate_0000 reproduced: {decision['candidate_0000_reproduced']}",
        f"- Best candidate: `{decision['best_candidate_id']}`",
        f"- Best not worse than candidate_0000: {decision['final_best_not_worse_than_candidate_0000']}",
        f"- Method evidence improved: {decision['method_evidence_improved']}",
        f"- Official SOTA claimed: {decision['official_sota_claimed']}",
        f"- Official leaderboard methods rerun: {decision['official_leaderboard_methods_rerun']}",
        f"- W&B online API used: {decision['wandb_online_api_used']}",
        f"- Test-label tuning: {decision['test_label_tuning']}",
        f"- Synthetic predictions used: {decision['synthetic_predictions_used']}",
        f"- Random predictions used: {decision['random_predictions_used']}",
        "",
        "## Anchor Check",
        _markdown_table(anchor_check["rows"], list(anchor_check["rows"][0]), max_rows=20),
    ]
    _write_report(dirs, "final_decision_report.md", "\n".join(lines))


def validate_d3_eyebench_own_method_score_max(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve()
    dirs = _analysis_dirs(config, out, root)
    errors: list[str] = []
    warnings: list[str] = []
    config_report = validate_d3_eyebench_own_method_score_max_config(config)
    if config_report["status"] != "passed":
        errors.extend(config_report["errors"])
    required_output = [
        out / "manifest.json",
        out / "config_validation.json",
        out / "preflight" / "preflight_report.json",
        out / "splits" / "official_split_validation.json",
        out / "anchor_reproduction_check.json",
        out / "candidate_manifest.json",
        out / "candidate_leaderboard.csv",
        out / "final_decision.json",
        out / "leakage_validation_report.json",
    ]
    for path in required_output:
        if not path.exists():
            errors.append(f"missing required output: {path}")
    required_reports = [
        "anchor_reproduction_report.md",
        "metric_alignment_report.md",
        "candidate_search_manifest.md",
        "candidate_leaderboard.csv",
        "best_candidate_report.md",
        "feature_family_ablation_report.md",
        "calibration_threshold_report.md",
        "reader_aggregation_secondary_report.md",
        "both_unseen_generalization_report.md",
        "leakage_validation_report.md",
        "prohibited_predictor_report.md",
        "evidence_summary_for_manuscript.md",
        "final_decision.json",
        "final_decision_report.md",
    ]
    for name in required_reports:
        if not (dirs["repo_analysis"] / name).exists():
            errors.append(f"missing repo analysis artifact: {dirs['repo_analysis'] / name}")
    decision: dict[str, Any] = {}
    if (out / "final_decision.json").exists():
        decision = json.loads((out / "final_decision.json").read_text(encoding="utf-8"))
        category = decision.get("final_decision_category")
        if category not in VALID_DECISION_CATEGORIES:
            errors.append(f"invalid decision category: {category}")
        if decision.get("official_leaderboard_methods_rerun") is not False:
            errors.append("decision indicates official leaderboard method rerun")
        if decision.get("wandb_online_api_used") is not False:
            errors.append("decision indicates W&B online API use")
        if decision.get("test_label_tuning") is not False:
            errors.append("decision indicates test-label tuning")
        if decision.get("synthetic_predictions_used") is not False:
            errors.append("decision indicates synthetic predictions")
        if decision.get("random_predictions_used") is not False:
            errors.append("decision indicates random predictions")
        if decision.get("candidate_0000_reproduced") is not True:
            errors.append("candidate_0000 was not reproduced")
        if decision.get("best_so_far_initialized_from_candidate_0000") is not True:
            errors.append("best_so_far was not initialized from candidate_0000")
        if decision.get("final_best_not_worse_than_candidate_0000") is not True:
            errors.append("final best is worse than candidate_0000")
        if decision.get("official_sota_claimed") is not False:
            warnings.append("official SOTA claim was made; verify strict gates separately")
    if (out / "leakage_validation_report.json").exists():
        leakage = json.loads((out / "leakage_validation_report.json").read_text(encoding="utf-8"))
        if leakage.get("status") != "passed":
            errors.append("leakage validation did not pass")
        if leakage.get("denied_predictors_present"):
            errors.append(f"denied predictors present: {leakage['denied_predictors_present']}")
    if (out / "candidate_manifest.json").exists():
        manifest = json.loads((out / "candidate_manifest.json").read_text(encoding="utf-8"))
        if not manifest or manifest[0].get("candidate_id") != "candidate_0000":
            errors.append("candidate_0000 is not first in candidate manifest")
        if not manifest or manifest[0].get("anchor_exact") is not True:
            errors.append("candidate_0000 is not marked anchor_exact")
    if bool(_section(config).get("require_slurm_job", True)) and (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        preflight = json.loads((out / "preflight" / "preflight_report.json").read_text(encoding="utf-8"))
        if not preflight.get("slurm_job_id"):
            errors.append("preflight missing Slurm job id")
        if not manifest.get("output_dir"):
            errors.append("manifest missing output_dir")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "decision_category": decision.get("final_decision_category"),
        "best_candidate_id": decision.get("best_candidate_id"),
        "official_sota_claimed": bool(decision.get("official_sota_claimed")),
    }


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run D3 EyeBench own-method score max v2.")
    parser.add_argument("--config", default="configs/d3_eyebench_own_method_score_max_v2.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--stage", choices=["anchor", "full"], default="full")
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    payload = run_d3_eyebench_own_method_score_max(
        config,
        args.output_dir,
        repo_root=args.repo_root,
        stage=args.stage,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate D3 EyeBench own-method score max v2.")
    parser.add_argument("--config", default="configs/d3_eyebench_own_method_score_max_v2.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    payload = validate_d3_eyebench_own_method_score_max(
        config,
        args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_main())
