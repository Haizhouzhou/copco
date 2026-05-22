"""Bounded protocol-aligned D3 optimization on official EyeBench CopCo_TYP folds."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir
from .official_eyebench_sota_check import (
    OFFICIAL_LITE_MODE,
    OFFICIAL_SPLITS,
    READER_LEVEL,
    SPLIT_TO_OFFICIAL_REGIME,
    TRIAL_LEVEL,
    _fold_cache,
    _official_reference_table,
    build_official_split_labels,
    load_official_processed_features,
    validate_official_split_labels,
)
from .research_exploration import _markdown_table, _np, _pd, _score_estimator


CAMPAIGN_SECTION = "d3_eyebench_protocol_optimization"
CAMPAIGN_MODEL = "D3_EyeBench_Optimized"

VALID_DECISION_CATEGORIES = {
    "official_sota_claim_allowed",
    "official_compatible_d3_improved_but_not_sota",
    "official_compatible_but_not_sota",
    "optimization_inconclusive",
    "blocked_by_environment",
    "blocked_by_data",
    "blocked_by_evaluator",
}

BASE_DENYLIST = {
    "participant_id",
    "speech_id",
    "text_id",
    "fold_id",
    "split_name",
    "split_role",
    "official_regime",
    "sample_id",
    "unique_trial_id",
    "unique_paragraph_id",
    "reader_group",
    "reader_group_binary",
    "reader_group_binary_num",
    "dyslexia",
    "dyslexia_labeled",
    "group_label",
    "RCS_score",
    "comprehension_score",
    "eyebench_rcs_score",
    "n_words_read",
    "n_speeches",
    "n_word_rows",
    "total_word_rows",
    "word_observation_count",
}

RESIDUAL_OUTCOME_CANDIDATES = {
    "first_fixation_duration": ["IA_FIRST_FIXATION_DURATION", "IA_FIRST_FIX_DURATION"],
    "first_pass_duration": ["IA_FIRST_RUN_DWELL_TIME", "IA_FIRST_FIX_DWELL_TIME"],
    "go_past_time": ["IA_SELECTIVE_REGRESSION_PATH_DURATION", "IA_REGRESSION_OUT_TIME"],
    "total_fixation_duration": ["IA_TOTAL_FIXATION_DURATION", "IA_DWELL_TIME"],
    "skipping": ["IA_SKIP", "total_skip"],
    "fixation_count": ["IA_FIXATION_COUNT", "IA_RUN_COUNT"],
}

PREDICTOR_SETS = {
    "surface": [
        "word_length",
        "word_length_no_punctuation",
        "wordfreq_frequency",
        "subtlex_frequency",
        "TRIAL_IA_COUNT",
        "normalized_ID",
        "start_of_line",
        "end_of_line",
        "is_content_word",
    ],
    "surface_surprisal": [
        "word_length",
        "word_length_no_punctuation",
        "wordfreq_frequency",
        "subtlex_frequency",
        "gpt2_surprisal",
        "TRIAL_IA_COUNT",
        "normalized_ID",
        "start_of_line",
        "end_of_line",
        "is_content_word",
    ],
    "surface_surprisal_syntax": [
        "word_length",
        "word_length_no_punctuation",
        "wordfreq_frequency",
        "subtlex_frequency",
        "gpt2_surprisal",
        "TRIAL_IA_COUNT",
        "normalized_ID",
        "start_of_line",
        "end_of_line",
        "is_content_word",
        "left_dependents_count",
        "right_dependents_count",
        "AbsDistance2Head",
        "distance_to_head",
    ],
}

OUTCOME_SETS = {
    "duration_core": [
        "first_fixation_duration",
        "first_pass_duration",
        "go_past_time",
        "total_fixation_duration",
    ],
    "duration_plus_count": [
        "first_fixation_duration",
        "first_pass_duration",
        "go_past_time",
        "total_fixation_duration",
        "fixation_count",
    ],
    "all_gaze": [
        "first_fixation_duration",
        "first_pass_duration",
        "go_past_time",
        "total_fixation_duration",
        "fixation_count",
        "skipping",
    ],
}

AGGREGATION_SETS = {
    "central": ["mean", "median"],
    "central_spread": ["mean", "median", "sd"],
    "robust_full": ["mean", "median", "sd", "q25", "q75", "iqr", "abs_mean"],
}

METRIC_COLUMNS = [
    "mode",
    "model_name",
    "claim_type",
    "task",
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
    "brier_score",
    "threshold",
    "status",
    "skip_reason",
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    residual_alpha: float
    predictor_set: str
    outcome_set: str
    aggregation_set: str
    transform: str
    classifier: str
    classifier_params: dict[str, Any]
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "residual_alpha": self.residual_alpha,
            "predictor_set": self.predictor_set,
            "outcome_set": self.outcome_set,
            "aggregation_set": self.aggregation_set,
            "transform": self.transform,
            "classifier": self.classifier,
            "classifier_params": self.classifier_params,
            "seed": self.seed,
        }

    @property
    def residual_key(self) -> str:
        payload = {
            "residual_alpha": self.residual_alpha,
            "predictor_set": self.predictor_set,
            "outcome_set": self.outcome_set,
            "aggregation_set": self.aggregation_set,
            "transform": self.transform,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


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


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    repo_analysis = root / str(
        get_nested(
            config,
            f"{CAMPAIGN_SECTION}.repo_analysis_dir",
            "analysis/d3_eyebench_protocol_aligned_optimization_v1",
        )
    )
    result_analysis = out / str(
        get_nested(
            config,
            f"{CAMPAIGN_SECTION}.output_layout.analysis",
            "analysis/d3_eyebench_protocol_aligned_optimization_v1",
        )
    )
    return {
        "repo_analysis": repo_analysis,
        "repo_tables": repo_analysis / "tables",
        "result_analysis": result_analysis,
        "result_tables": out
        / str(
            get_nested(
                config,
                f"{CAMPAIGN_SECTION}.output_layout.tables",
                "analysis/d3_eyebench_protocol_aligned_optimization_v1/tables",
            )
        ),
    }


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["repo_analysis"] / name, text)
    _write_md(dirs["result_analysis"] / name, text)


def _write_table(dirs: dict[str, Path], name: str, frame: Any) -> None:
    _write_csv(dirs["repo_tables"] / name, frame)
    _write_csv(dirs["result_tables"] / name, frame)


def validate_d3_eyebench_protocol_optimization_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    errors: list[str] = []
    for flag in [
        "official_eyebench_data_folds_only",
        "no_test_label_tuning",
        "no_synthetic_outputs",
        "no_random_predictions",
        "no_full_prepared_copco_joins",
        "no_official_baseline_reruns",
    ]:
        if section.get(flag) is not True:
            errors.append(f"{CAMPAIGN_SECTION}.{flag} must be true")
    splits = section.get("split_regimes", [])
    missing_splits = sorted(set(OFFICIAL_SPLITS) - set(splits))
    if missing_splits:
        errors.append(f"missing official split regimes: {missing_splits}")
    prohibited = set(section.get("prohibited_features", []))
    missing_prohibited = sorted(BASE_DENYLIST - prohibited)
    if missing_prohibited:
        errors.append(f"prohibited feature list incomplete: {missing_prohibited}")
    target = get_nested(config, f"{CAMPAIGN_SECTION}.published_leaderboard_snapshot.target_balanced_accuracy")
    if target is None or float(target) <= 0:
        errors.append("published leaderboard target_balanced_accuracy must be positive")
    if int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.max_candidates", 0)) <= 0:
        errors.append("budget.max_candidates must be positive")
    if int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.no_improvement_rounds", 0)) <= 0:
        errors.append("budget.no_improvement_rounds must be positive")
    return {"status": "passed" if not errors else "failed", "errors": errors}


def build_candidate_specs(config: dict[str, Any]) -> list[CandidateSpec]:
    section = _section(config)
    grid = section.get("candidate_grid", {})
    residual_alphas = [float(x) for x in grid.get("residual_alphas", [1.0])]
    predictor_sets = [str(x) for x in grid.get("predictor_sets", ["surface_surprisal_syntax"])]
    outcome_sets = [str(x) for x in grid.get("outcome_sets", ["all_gaze"])]
    aggregation_sets = [str(x) for x in grid.get("aggregation_sets", ["central_spread"])]
    transforms = [str(x) for x in grid.get("transforms", ["raw"])]
    classifiers = grid.get("classifiers", [])
    if not classifiers:
        classifiers = [{"name": "logistic_regression", "params": {"C": 1.0, "penalty": "l2"}}]
    seeds = [int(x) for x in grid.get("seeds", [int(section.get("deterministic_seed", 173))])]
    payloads: list[dict[str, Any]] = []
    for values in product(
        residual_alphas,
        predictor_sets,
        outcome_sets,
        aggregation_sets,
        transforms,
        classifiers,
        seeds,
    ):
        alpha, predictor_set, outcome_set, aggregation_set, transform, classifier, seed = values
        name = str(classifier.get("name", "logistic_regression"))
        params = dict(classifier.get("params", {}))
        payload = {
            "alpha": alpha,
            "predictor_set": predictor_set,
            "outcome_set": outcome_set,
            "aggregation_set": aggregation_set,
            "transform": transform,
            "classifier": name,
            "params": params,
            "seed": seed,
        }
        payload["order_key"] = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        payloads.append(payload)
    payloads = sorted(payloads, key=lambda item: item["order_key"])
    max_candidates = int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.max_candidates", len(payloads)))
    candidates: list[CandidateSpec] = []
    for payload in payloads[:max_candidates]:
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        candidates.append(
            CandidateSpec(
                candidate_id=f"d3opt_{len(candidates) + 1:04d}_{digest}",
                residual_alpha=float(payload["alpha"]),
                predictor_set=str(payload["predictor_set"]),
                outcome_set=str(payload["outcome_set"]),
                aggregation_set=str(payload["aggregation_set"]),
                transform=str(payload["transform"]),
                classifier=str(payload["classifier"]),
                classifier_params=dict(payload["params"]),
                seed=int(payload["seed"]),
            )
        )
    return candidates


def _safe_numeric_columns(frame: Any, candidates: list[str], prohibited: set[str]) -> list[str]:
    pd = _pd()
    np = _np()
    clean: list[str] = []
    lower_prohibited = {name.lower() for name in prohibited}
    for column in candidates:
        if column not in frame:
            continue
        if column in prohibited or column.lower() in lower_prohibited:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], pd.NA)
        if values.notna().any() and values.nunique(dropna=True) > 1:
            clean.append(column)
    return clean


def _numeric_matrix(frame: Any, columns: list[str]) -> Any:
    pd = _pd()
    np = _np()
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _outcome_column(frame: Any, outcome: str) -> str | None:
    for column in RESIDUAL_OUTCOME_CANDIDATES[outcome]:
        if column in frame:
            return column
    return None


def _transformed_outcome(values: Any, *, outcome: str, transform: str) -> Any:
    np = _np()
    pd = _pd()
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if transform == "log1p_duration" and outcome != "skipping":
        return np.log1p(np.clip(series.astype(float), a_min=0, a_max=None))
    if transform == "sqrt_duration" and outcome != "skipping":
        return np.sqrt(np.clip(series.astype(float), a_min=0, a_max=None))
    return series.astype(float)


def _trial_residual_features_variant(
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

    word = ia.copy()
    word["unique_trial_id"] = word["unique_trial_id"].astype(str)
    fit_word = word[word["unique_trial_id"].isin(fit_ids)].copy()
    apply_word = word[word["unique_trial_id"].isin(apply_ids)].copy()
    predictors = _safe_numeric_columns(
        fit_word,
        PREDICTOR_SETS.get(candidate.predictor_set, PREDICTOR_SETS["surface_surprisal_syntax"]),
        prohibited,
    )
    diagnostics = {
        "candidate_id": candidate.candidate_id,
        "residual_key": candidate.residual_key,
        "fit_word_rows": int(len(fit_word)),
        "apply_word_rows": int(len(apply_word)),
        "predictor_set": candidate.predictor_set,
        "predictors": predictors,
        "outcome_set": candidate.outcome_set,
        "aggregation_set": candidate.aggregation_set,
        "transform": candidate.transform,
        "residual_alpha": candidate.residual_alpha,
        "heldout_reader_rows_used_for_fit": False,
        "heldout_text_rows_used_for_fit": False,
        "reader_group_used": False,
        "skipped": False,
        "skip_reason": "",
    }
    if fit_word.empty or apply_word.empty or not predictors:
        diagnostics["skipped"] = True
        diagnostics["skip_reason"] = "no_fit_or_apply_words_or_predictors"
        return pd.DataFrame(), diagnostics
    combined = pd.concat([fit_word, apply_word], ignore_index=True)
    outcome_names = OUTCOME_SETS.get(candidate.outcome_set, OUTCOME_SETS["all_gaze"])
    residual_columns: list[str] = []
    for outcome in outcome_names:
        column = _outcome_column(fit_word, outcome)
        if column is None:
            continue
        fit_y = _transformed_outcome(fit_word[column], outcome=outcome, transform=candidate.transform)
        valid_fit = pd.Series(fit_y).notna()
        if int(valid_fit.sum()) < 3 or pd.Series(fit_y).nunique(dropna=True) <= 1:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=float(candidate.residual_alpha)),
        )
        model.fit(_numeric_matrix(fit_word.loc[valid_fit], predictors), pd.Series(fit_y).loc[valid_fit])
        observed = _transformed_outcome(combined[column], outcome=outcome, transform=candidate.transform)
        predicted = model.predict(_numeric_matrix(combined, predictors))
        resid_name = f"d3_resid_{outcome}"
        combined[resid_name] = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
        residual_columns.append(resid_name)
    if not residual_columns:
        diagnostics["skipped"] = True
        diagnostics["skip_reason"] = "no_residual_outcomes_available"
        return pd.DataFrame(), diagnostics
    aggregations = AGGREGATION_SETS.get(candidate.aggregation_set, AGGREGATION_SETS["central_spread"])
    aggregate_rows: list[dict[str, Any]] = []
    apply_subset = combined[combined["unique_trial_id"].isin(apply_ids)].copy()
    for trial_id, group in apply_subset.groupby("unique_trial_id", dropna=False):
        row = {"sample_id": str(trial_id), "unique_trial_id": str(trial_id)}
        for column in residual_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            prefix = f"{column}_{candidate.transform}"
            if "mean" in aggregations:
                row[f"{prefix}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            if "median" in aggregations:
                row[f"{prefix}_median"] = float(values.median()) if values.notna().any() else np.nan
            if "sd" in aggregations:
                row[f"{prefix}_sd"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
            if "q25" in aggregations:
                row[f"{prefix}_q25"] = float(values.quantile(0.25)) if values.notna().any() else np.nan
            if "q75" in aggregations:
                row[f"{prefix}_q75"] = float(values.quantile(0.75)) if values.notna().any() else np.nan
            if "iqr" in aggregations:
                row[f"{prefix}_iqr"] = (
                    float(values.quantile(0.75) - values.quantile(0.25))
                    if values.notna().any()
                    else np.nan
                )
            if "abs_mean" in aggregations:
                row[f"{prefix}_abs_mean"] = float(values.abs().mean()) if values.notna().any() else np.nan
        aggregate_rows.append(row)
    features = pd.DataFrame(aggregate_rows)
    return features, diagnostics


def _make_classifier(candidate: CandidateSpec, n_jobs: int) -> Any:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(candidate.classifier_params)
    seed = int(candidate.seed)
    if candidate.classifier == "logistic_regression":
        model = LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=str(params.get("penalty", "l2")),
            solver=str(params.get("solver", "liblinear")),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=int(params.get("max_iter", 2000)),
            random_state=seed,
        )
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    if candidate.classifier == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced_subsample"),
            n_jobs=max(1, int(n_jobs)),
            random_state=seed,
        )
        return make_pipeline(SimpleImputer(strategy="median"), model)
    if candidate.classifier == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=max(1, int(n_jobs)),
            random_state=seed,
        )
        return make_pipeline(SimpleImputer(strategy="median"), model)
    if candidate.classifier == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 2)),
            random_state=seed,
        )
        return make_pipeline(SimpleImputer(strategy="median"), model)
    raise ValueError(f"unsupported D3 classifier: {candidate.classifier}")


def _classification_metrics_at_threshold(y_true: Any, y_score: Any, threshold: float) -> dict[str, Any]:
    np = _np()
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= float(threshold)).astype(int)
    if len(y_true_arr) == 0:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "brier_score": None,
        }
    if len(set(y_true_arr.tolist())) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
            "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
            "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
        }
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
    }


def _select_threshold(y_true: Any, y_score: Any, thresholds: list[float]) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        metric = _classification_metrics_at_threshold(y_true, y_score, threshold)
        value = metric.get("balanced_accuracy")
        if value is None:
            continue
        tie_break = abs(float(threshold) - 0.5)
        best_tie_break = abs(float(best_threshold) - 0.5)
        if float(value) > best_score or (float(value) == best_score and tie_break < best_tie_break):
            best_score = float(value)
            best_threshold = float(threshold)
    return best_threshold, best_score


def _inner_split(train: Any, split_name: str, seed: int, val_fraction: float) -> tuple[Any, Any, dict[str, Any]]:
    pd = _pd()
    if train.empty:
        return train.copy(), train.copy(), {"strategy": "empty", "fallback": False}
    y = pd.to_numeric(train.get("reader_group_binary"), errors="coerce")
    valid = train[y.notna()].copy()
    if valid.empty:
        return train.copy(), train.iloc[0:0].copy(), {"strategy": "no_valid_labels", "fallback": False}
    rng = _np().random.default_rng(seed)

    def choose_groups(column: str) -> tuple[set[str], str]:
        group_stats = (
            valid.groupby(column, dropna=False)["reader_group_binary"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values([column])
        )
        group_values = group_stats[column].astype(str).tolist()
        if not group_values:
            return set(), f"{column}_empty"
        order = list(range(len(group_values)))
        rng.shuffle(order)
        target = max(1, int(round(len(valid) * val_fraction)))
        selected: set[str] = set()
        for idx in order:
            selected.add(str(group_values[idx]))
            current = valid[valid[column].astype(str).isin(selected)]
            current_y = pd.to_numeric(current["reader_group_binary"], errors="coerce")
            if len(current) >= target and current_y.nunique(dropna=True) >= 2:
                break
        return selected, column

    fallback = False
    strategy = "trial_stratified"
    if split_name == "unseen_reader":
        selected, strategy = choose_groups("participant_id")
        val_mask = valid["participant_id"].astype(str).isin(selected)
        train_mask = ~val_mask
    elif split_name == "unseen_text":
        selected, strategy = choose_groups("text_id")
        val_mask = valid["text_id"].astype(str).isin(selected)
        train_mask = ~val_mask
    elif split_name == "unseen_reader_and_text":
        selected_participants, _ = choose_groups("participant_id")
        selected_texts, _ = choose_groups("text_id")
        val_mask = valid["participant_id"].astype(str).isin(selected_participants) | valid[
            "text_id"
        ].astype(str).isin(selected_texts)
        train_mask = (~valid["participant_id"].astype(str).isin(selected_participants)) & (
            ~valid["text_id"].astype(str).isin(selected_texts)
        )
        strategy = "participant_and_text_disjoint"
    else:
        train_mask = pd.Series([True] * len(valid), index=valid.index)
        val_mask = pd.Series([False] * len(valid), index=valid.index)
    inner_train = valid[train_mask].copy()
    inner_val = valid[val_mask].copy()
    if (
        inner_train.empty
        or inner_val.empty
        or pd.to_numeric(inner_train["reader_group_binary"], errors="coerce").nunique(dropna=True) < 2
        or pd.to_numeric(inner_val["reader_group_binary"], errors="coerce").nunique(dropna=True) < 2
    ):
        fallback = True
        frames = []
        for _, group in valid.groupby("reader_group_binary", dropna=False):
            order = list(group.index)
            rng.shuffle(order)
            n_val = max(1, int(round(len(order) * val_fraction)))
            val_idx = set(order[:n_val])
            part = group.copy()
            part["_inner_role"] = ["val" if idx in val_idx else "train" for idx in part.index]
            frames.append(part)
        marked = pd.concat(frames).drop(columns=[], errors="ignore").sort_index()
        inner_train = marked[marked["_inner_role"].eq("train")].drop(columns=["_inner_role"])
        inner_val = marked[marked["_inner_role"].eq("val")].drop(columns=["_inner_role"])
        strategy = "trial_stratified_fallback"
    diagnostics = {
        "strategy": strategy,
        "fallback": fallback,
        "inner_train_samples": int(len(inner_train)),
        "inner_val_samples": int(len(inner_val)),
        "inner_train_classes": int(
            pd.to_numeric(inner_train.get("reader_group_binary"), errors="coerce").nunique(dropna=True)
        ),
        "inner_val_classes": int(
            pd.to_numeric(inner_val.get("reader_group_binary"), errors="coerce").nunique(dropna=True)
        ),
    }
    return inner_train.reset_index(drop=True), inner_val.reset_index(drop=True), diagnostics


def _feature_columns(frame: Any, prohibited: set[str]) -> list[str]:
    return _safe_numeric_columns(
        frame,
        [column for column in frame.columns if str(column).startswith("d3_resid_")],
        prohibited,
    )


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
    model = _make_classifier(candidate, n_jobs)
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
                "model_name": CAMPAIGN_MODEL,
                "candidate_id": candidate.candidate_id,
                "claim_type": "official_compatible_candidate",
                "task": "CopCo_TYP",
                "split_name": split_name,
                "fold_id": int(fold_id),
                "feature_group": "d3_protocol_aligned_residuals",
                "model": candidate.classifier,
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
            }
        )
    return rows


def _metric_frame(
    predictions: Any,
    candidate: CandidateSpec,
    *,
    evaluation_level: str,
    threshold_by_fold: dict[int, float] | None = None,
) -> Any:
    pd = _pd()
    rows = []
    for split_name in OFFICIAL_SPLITS:
        split_pred = predictions[predictions["split_name"].eq(split_name)] if not predictions.empty else pd.DataFrame()
        fold_metrics = []
        for fold_id, fold in split_pred.groupby("fold_id", dropna=False):
            threshold = (
                float(threshold_by_fold.get(int(fold_id), 0.5))
                if threshold_by_fold is not None
                else float(fold.get("threshold", pd.Series([0.5])).iloc[0])
            )
            metric = _classification_metrics_at_threshold(fold["y_true"], fold["y_score"], threshold)
            metric["threshold"] = threshold
            metric["n_predictions"] = int(len(fold))
            fold_metrics.append(metric)
        if fold_metrics:
            fold_frame = pd.DataFrame(fold_metrics)
            n_predictions = int(fold_frame["n_predictions"].sum())
            usable_folds = int(len(fold_frame))
            skipped_folds = int(4 - usable_folds)
            metric_values = {}
            for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "brier_score", "threshold"]:
                values = pd.to_numeric(fold_frame.get(metric), errors="coerce")
                metric_values[metric] = float(values.mean()) if values.notna().any() else None
            status = "complete"
            skip_reason = ""
        else:
            n_predictions = 0
            usable_folds = 0
            skipped_folds = 4
            metric_values = {
                "roc_auc": None,
                "pr_auc": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "brier_score": None,
                "threshold": None,
            }
            status = "skipped"
            skip_reason = "no_predictions"
        rows.append(
            {
                "mode": OFFICIAL_LITE_MODE,
                "model_name": CAMPAIGN_MODEL,
                "claim_type": "official_compatible_candidate",
                "task": "CopCo_TYP",
                "split_name": split_name,
                "evaluation_level": evaluation_level,
                "n_features": int(predictions.get("n_features", pd.Series([0])).max())
                if not predictions.empty and "n_features" in predictions
                else None,
                "n_predictions": n_predictions,
                "usable_folds": usable_folds,
                "skipped_folds": skipped_folds,
                **metric_values,
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
        )
        .reset_index()
    )


def _evaluate_candidate_inner(
    config: dict[str, Any],
    candidate: CandidateSpec,
    fold_data: dict[tuple[str, int], dict[str, Any]],
    ia: Any,
    prohibited: set[str],
    feature_cache: dict[tuple[str, int, str, int], dict[str, Any]],
) -> tuple[dict[str, Any], Any, Any]:
    pd = _pd()
    section = _section(config)
    n_jobs = int(section.get("n_jobs", 1))
    val_fraction = float(get_nested(config, f"{CAMPAIGN_SECTION}.inner_validation.fraction", 0.25))
    threshold_count = int(get_nested(config, f"{CAMPAIGN_SECTION}.inner_validation.threshold_grid_size", 101))
    thresholds = [float(x) for x in _np().linspace(0.01, 0.99, threshold_count)]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for (split_name, fold_id), value in sorted(fold_data.items()):
        cache_key = (candidate.residual_key, int(candidate.seed), split_name, int(fold_id))
        if cache_key in feature_cache:
            cached = feature_cache[cache_key]
            inner_train = cached["inner_train"]
            inner_val = cached["inner_val"]
            train_aug = cached["train_aug"]
            val_aug = cached["val_aug"]
            columns = cached["columns"]
            inner_diag = dict(cached["inner_diag"])
            resid_diag = dict(cached["resid_diag"])
            resid_diag["candidate_id"] = candidate.candidate_id
        else:
            train = value["train"].copy()
            inner_train, inner_val, inner_diag = _inner_split(
                train,
                split_name,
                int(candidate.seed) + int(fold_id) * 31 + len(split_name),
                val_fraction,
            )
            fit_ids = set(inner_train["unique_trial_id"].astype(str))
            apply_ids = set(inner_train["unique_trial_id"].astype(str)).union(
                set(inner_val["unique_trial_id"].astype(str))
            )
            features, resid_diag = _trial_residual_features_variant(
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
                "resid_diag": dict(resid_diag),
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
            selected_threshold, selected_ba = _select_threshold(
                val_pred["y_true"], val_pred["y_score"], thresholds
            )
            metric = _classification_metrics_at_threshold(
                val_pred["y_true"], val_pred["y_score"], selected_threshold
            )
            val_pred["threshold"] = selected_threshold
            val_pred["y_pred"] = (val_pred["y_score"].astype(float) >= selected_threshold).astype(int)
            fold_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "split_name": split_name,
                    "fold_id": int(fold_id),
                    "n_features": int(len(columns)),
                    "n_inner_train": int(len(inner_train)),
                    "n_inner_val": int(len(inner_val)),
                    "threshold": float(selected_threshold),
                    "threshold_selection_metric": float(selected_ba),
                    "status": "complete",
                    "skip_reason": "",
                    **metric,
                }
            )
            for row in _prediction_rows(val_pred, candidate, split_name, int(fold_id)):
                row["eval_type"] = "inner_validation"
                row["n_features"] = int(len(columns))
                prediction_rows.append(row)
        else:
            fold_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
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
                    "brier_score": None,
                    "status": "skipped",
                    "skip_reason": skip_reason or resid_diag.get("skip_reason", "no_predictions"),
                }
            )
        diagnostics.append(
            {
                "candidate_id": candidate.candidate_id,
                "split_name": split_name,
                "fold_id": int(fold_id),
                **inner_diag,
                **{f"residual_{key}": value for key, value in resid_diag.items()},
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    split_scores = (
        fold_frame[fold_frame["status"].eq("complete")]
        .groupby("split_name", dropna=False)["balanced_accuracy"]
        .mean()
    )
    selection_score = float(split_scores.mean()) if not split_scores.empty else None
    summary = {
        **candidate.as_dict(),
        "selection_metric": "inner_validation_fold_mean_balanced_accuracy",
        "selection_score": selection_score,
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
    return summary, fold_frame, pd.DataFrame(diagnostics)


def _evaluate_final_candidate(
    config: dict[str, Any],
    candidate: CandidateSpec,
    fold_data: dict[tuple[str, int], dict[str, Any]],
    inner_fold_metrics: Any,
    ia: Any,
    prohibited: set[str],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    pd = _pd()
    section = _section(config)
    n_jobs = int(section.get("n_jobs", 1))
    prediction_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    threshold_map = {}
    if not inner_fold_metrics.empty:
        best_rows = inner_fold_metrics[inner_fold_metrics["candidate_id"].eq(candidate.candidate_id)]
        for _, row in best_rows.iterrows():
            threshold_map[(str(row["split_name"]), int(row["fold_id"]))] = float(row["threshold"])
    global_threshold = (
        float(pd.to_numeric(best_rows["threshold"], errors="coerce").mean())
        if not inner_fold_metrics.empty and not best_rows.empty
        else 0.5
    )
    for (split_name, fold_id), value in sorted(fold_data.items()):
        train = value["train"].copy()
        test = value["test"].copy()
        fit_ids = set(train["unique_trial_id"].astype(str))
        apply_ids = set(train["unique_trial_id"].astype(str)).union(set(test["unique_trial_id"].astype(str)))
        features, resid_diag = _trial_residual_features_variant(
            fit_ids,
            apply_ids,
            ia,
            candidate,
            prohibited,
        )
        train_aug = train.merge(features, on=["sample_id", "unique_trial_id"], how="left") if not features.empty else train
        test_aug = test.merge(features, on=["sample_id", "unique_trial_id"], how="left") if not features.empty else test
        columns = _feature_columns(train_aug, prohibited)
        threshold = threshold_map.get((split_name, int(fold_id)), global_threshold)
        _, test_pred, skip_reason = _fit_score_fold(
            train_aug,
            test_aug,
            columns,
            candidate,
            threshold=threshold,
            n_jobs=n_jobs,
        )
        if not test_pred.empty:
            for row in _prediction_rows(test_pred, candidate, split_name, int(fold_id)):
                row["n_features"] = int(len(columns))
                prediction_rows.append(row)
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
                **{f"residual_{key}": value for key, value in resid_diag.items()},
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    trial_metrics = _metric_frame(predictions, candidate, evaluation_level=TRIAL_LEVEL)
    reader_metrics = _metric_frame(_reader_aggregate(predictions), candidate, evaluation_level=READER_LEVEL)
    diagnostics_frame = pd.DataFrame(diagnostics)
    leakage = {
        "no_test_label_tuning": True,
        "threshold_selection_source": "inner_validation_only",
        "official_test_labels_used_for_selection": False,
        "heldout_reader_rows_used_for_residual_fit": bool(
            diagnostics_frame.get("residual_heldout_reader_rows_used_for_fit", pd.Series(dtype=bool)).any()
        )
        if not diagnostics_frame.empty
        else False,
        "heldout_text_rows_used_for_residual_fit": bool(
            diagnostics_frame.get("residual_heldout_text_rows_used_for_fit", pd.Series(dtype=bool)).any()
        )
        if not diagnostics_frame.empty
        else False,
        "reader_group_used_in_residualization": False,
        "predictor_columns": _feature_columns(predictions, prohibited) if not predictions.empty else [],
        "denied_predictors_present": [],
        "no_synthetic_outputs": True,
        "no_random_predictions": True,
        "no_full_prepared_copco_join": True,
        "status": "passed",
    }
    if (
        leakage["heldout_reader_rows_used_for_residual_fit"]
        or leakage["heldout_text_rows_used_for_residual_fit"]
        or leakage["reader_group_used_in_residualization"]
    ):
        leakage["status"] = "failed"
    return trial_metrics, reader_metrics, predictions, {"diagnostics": diagnostics_frame, "leakage": leakage}


def _average_primary_ba(metrics: Any) -> float | None:
    pd = _pd()
    if metrics.empty:
        return None
    values = pd.to_numeric(metrics["balanced_accuracy"], errors="coerce")
    return float(values.mean()) if values.notna().any() else None


def _load_anchor(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    pd = _pd()
    path = Path(str(get_nested(config, f"{CAMPAIGN_SECTION}.local_logistic_anchor.metrics_path")))
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return {"status": "missing", "path": str(path), "average_balanced_accuracy": None, "rows": []}
    frame = pd.read_csv(path)
    primary = frame[frame["split_name"].isin(OFFICIAL_SPLITS)].copy()
    avg = float(pd.to_numeric(primary["balanced_accuracy"], errors="coerce").mean())
    return {
        "status": "present",
        "path": str(path),
        "average_balanced_accuracy": avg,
        "rows": primary.to_dict("records"),
        "rerun_performed": False,
    }


def _decision_category(
    *,
    config_report: dict[str, Any],
    split_errors: list[str],
    anchor: dict[str, Any],
    leakage: dict[str, Any],
    trial_metrics: Any,
    slurm_required: bool,
    slurm_job_id: str | None,
    target_ba: float,
) -> tuple[str, dict[str, Any]]:
    final_ba = _average_primary_ba(trial_metrics)
    official_compatible = (
        config_report.get("status") == "passed"
        and not split_errors
        and anchor.get("status") == "present"
        and leakage.get("status") == "passed"
        and final_ba is not None
        and (not slurm_required or bool(slurm_job_id))
    )
    beats_target = bool(final_ba is not None and final_ba > target_ba)
    beats_anchor = bool(
        final_ba is not None
        and anchor.get("average_balanced_accuracy") is not None
        and final_ba > float(anchor["average_balanced_accuracy"])
    )
    if config_report.get("status") != "passed" or (slurm_required and not slurm_job_id):
        category = "blocked_by_environment"
    elif split_errors:
        category = "blocked_by_data"
    elif leakage.get("status") != "passed":
        category = "optimization_inconclusive"
    elif not official_compatible:
        category = "optimization_inconclusive"
    elif beats_target:
        category = "official_sota_claim_allowed"
    elif beats_anchor:
        category = "official_compatible_d3_improved_but_not_sota"
    else:
        category = "official_compatible_but_not_sota"
    checks = {
        "official_compatible": official_compatible,
        "final_average_balanced_accuracy": final_ba,
        "target_balanced_accuracy": target_ba,
        "local_logistic_anchor_average_balanced_accuracy": anchor.get("average_balanced_accuracy"),
        "beats_official_target": beats_target,
        "beats_local_logistic_anchor": beats_anchor,
        "slurm_required": slurm_required,
        "slurm_job_id": slurm_job_id,
    }
    return category, checks


def run_d3_eyebench_protocol_optimization(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    section = _section(config)
    require_slurm = bool(section.get("require_slurm_job", True))
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if require_slurm and not slurm_job_id:
        raise RuntimeError("SLURM_JOB_ID is empty; refusing D3 optimization on login node")
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    config_report = validate_d3_eyebench_protocol_optimization_config(config)
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
        },
    )

    anchor = _load_anchor(config, root)
    reference = _official_reference_table(_to_sota_config(config), root)
    samples, ia = load_official_processed_features(_to_sota_config(config), out, root)
    splits = build_official_split_labels(_to_sota_config(config), out, samples, root)
    split_errors, split_summaries = validate_official_split_labels(splits)
    _write_json(out / "splits" / "official_split_validation.json", {"errors": split_errors, "summaries": split_summaries})
    if samples.empty or ia.empty or splits.empty:
        category = "blocked_by_data"
        decision = {
            "decision_category": category,
            "official_sota_claim_allowed": False,
            "reason": "official data/folds unavailable",
        }
        _write_json(out / "official_sota_decision.json", decision)
        _write_json(dirs["repo_analysis"] / "official_sota_decision.json", decision)
        return {"status": "blocked", "output_dir": str(out), **decision}

    candidates = build_candidate_specs(config)
    _write_json(out / "optimization" / "candidate_specs.json", [candidate.as_dict() for candidate in candidates])
    _write_json(dirs["repo_analysis"] / "candidate_specs.json", [candidate.as_dict() for candidate in candidates])
    fold_data = _fold_cache(samples, splits)
    prohibited = set(section.get("prohibited_features", [])) | BASE_DENYLIST
    no_improvement_rounds = int(get_nested(config, f"{CAMPAIGN_SECTION}.budget.no_improvement_rounds", 24))
    min_delta = float(get_nested(config, f"{CAMPAIGN_SECTION}.budget.minimum_improvement_delta", 0.001))
    feature_cache: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    best_score = -1.0
    best_candidate: CandidateSpec | None = None
    rounds_since_improvement = 0
    summaries: list[dict[str, Any]] = []
    inner_frames = []
    diagnostic_frames = []
    stop_reason = "budget_exhausted"
    target_ba = float(get_nested(config, f"{CAMPAIGN_SECTION}.published_leaderboard_snapshot.target_balanced_accuracy"))
    progress_path = out / "optimization" / "candidate_progress.csv"
    for candidate in candidates:
        summary, inner_fold, diagnostics = _evaluate_candidate_inner(
            config, candidate, fold_data, ia, prohibited, feature_cache
        )
        summaries.append(summary)
        inner_frames.append(inner_fold)
        diagnostic_frames.append(diagnostics)
        _write_csv(progress_path, _pd().DataFrame(summaries))
        print(
            "candidate_progress",
            len(summaries),
            candidate.candidate_id,
            summary.get("selection_score"),
            flush=True,
        )
        score = summary.get("selection_score")
        if score is not None and float(score) > best_score + min_delta:
            best_score = float(score)
            best_candidate = candidate
            rounds_since_improvement = 0
        else:
            rounds_since_improvement += 1
        if rounds_since_improvement >= no_improvement_rounds:
            stop_reason = "no_improvement_stopping_rule"
            break
    if best_candidate is None:
        category = "optimization_inconclusive"
        decision = {
            "decision_category": category,
            "official_sota_claim_allowed": False,
            "reason": "no candidate completed inner validation",
        }
        _write_json(out / "official_sota_decision.json", decision)
        _write_json(dirs["repo_analysis"] / "official_sota_decision.json", decision)
        return {"status": "complete", "output_dir": str(out), **decision}
    pd = _pd()
    candidate_summary = pd.DataFrame(summaries)
    inner_fold_metrics = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    inner_diagnostics = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    _write_csv(out / "optimization" / "candidate_summary.csv", candidate_summary)
    _write_csv(out / "optimization" / "inner_validation_fold_metrics.csv", inner_fold_metrics)
    _write_csv(out / "optimization" / "inner_validation_diagnostics.csv", inner_diagnostics)
    _write_table(dirs, "candidate_summary.csv", candidate_summary)
    _write_table(dirs, "inner_validation_fold_metrics.csv", inner_fold_metrics)

    trial_metrics, reader_metrics, predictions, final_extra = _evaluate_final_candidate(
        config,
        best_candidate,
        fold_data,
        inner_fold_metrics,
        ia,
        prohibited,
    )
    diagnostics = final_extra["diagnostics"]
    leakage = final_extra["leakage"]
    _write_csv(out / "typ" / "d3_optimized_trial_metrics.csv", trial_metrics)
    _write_csv(out / "typ" / "d3_optimized_reader_aggregated_metrics.csv", reader_metrics)
    _write_csv(out / "typ" / "d3_optimized_trial_predictions.csv", predictions)
    _write_csv(out / "optimization" / "final_candidate_diagnostics.csv", diagnostics)
    if not predictions.empty:
        official = predictions.copy()
        official["label"] = official["y_true"]
        official["prediction_prob"] = official["y_score"]
        official["prediction"] = official["y_pred"]
        official["binary_prediction"] = official["y_pred"]
        keep = [
            "label",
            "prediction_prob",
            "prediction",
            "binary_prediction",
            "eval_regime",
            "eval_type",
            "fold_index",
            "participant_id",
            "speech_id",
            "text_id",
            "unique_trial_id",
        ]
        _write_csv(out / "typ" / "trial_level_test_results.csv", official[keep])
    _write_table(dirs, "d3_optimized_trial_metrics.csv", trial_metrics)
    _write_table(dirs, "d3_optimized_reader_aggregated_metrics.csv", reader_metrics)
    _write_json(out / "leakage_report.json", leakage)
    _write_json(dirs["repo_analysis"] / "leakage_report.json", leakage)

    category, decision_checks = _decision_category(
        config_report=config_report,
        split_errors=split_errors,
        anchor=anchor,
        leakage=leakage,
        trial_metrics=trial_metrics,
        slurm_required=require_slurm,
        slurm_job_id=slurm_job_id,
        target_ba=target_ba,
    )
    leaderboard = dict(get_nested(config, f"{CAMPAIGN_SECTION}.published_leaderboard_snapshot", {}))
    decision = {
        "decision_category": category,
        "official_sota_claim_allowed": category == "official_sota_claim_allowed",
        "campaign_protocol": get_nested(config, f"{CAMPAIGN_SECTION}.protocol_path"),
        "published_leaderboard_snapshot": leaderboard,
        "published_reference_rows": reference.to_dict("records"),
        "local_logistic_anchor": anchor,
        "best_candidate": best_candidate.as_dict(),
        "candidate_count_evaluated": int(len(candidate_summary)),
        "stop_reason": stop_reason,
        "selection_source": "inner_validation_only",
        "test_label_tuning": False,
        "official_baseline_rerun_performed": False,
        "synthetic_outputs_used": False,
        "random_predictions_used": False,
        "full_prepared_copco_join_used": False,
        "checks": decision_checks,
    }
    _write_json(out / "official_sota_decision.json", decision)
    _write_json(dirs["repo_analysis"] / "official_sota_decision.json", decision)
    _write_reports(dirs, decision, trial_metrics, reader_metrics, candidate_summary, leakage)
    manifest = {
        "status": "complete",
        "run_name": get_nested(config, "run.name", "d3_eyebench_protocol_aligned_optimization_v1"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out),
        "repo_root": str(root),
        "git_sha": _git_sha(root),
        "slurm_job_id": slurm_job_id,
        "config_validation_status": config_report["status"],
        "split_validation_errors": split_errors,
        "candidate_count_evaluated": int(len(candidate_summary)),
        "stop_reason": stop_reason,
        "best_candidate_id": best_candidate.candidate_id,
        "best_inner_validation_balanced_accuracy": best_score,
        "final_average_balanced_accuracy": decision_checks["final_average_balanced_accuracy"],
        "decision_category": category,
        "official_sota_claim_allowed": category == "official_sota_claim_allowed",
    }
    _write_json(out / "manifest.json", manifest)
    _write_json(dirs["repo_analysis"] / "manifest.json", manifest)
    return manifest


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


def _write_reports(
    dirs: dict[str, Path],
    decision: dict[str, Any],
    trial_metrics: Any,
    reader_metrics: Any,
    candidate_summary: Any,
    leakage: dict[str, Any],
) -> None:
    best = decision["best_candidate"]
    lines = [
        "# D3 EyeBench Protocol-Aligned Optimization Report",
        "",
        f"- Decision category: `{decision['decision_category']}`",
        f"- Stop reason: `{decision['stop_reason']}`",
        f"- Candidate count evaluated: {decision['candidate_count_evaluated']}",
        f"- Best candidate: `{best['candidate_id']}`",
        f"- Selection source: `{decision['selection_source']}`",
        f"- Official baseline rerun performed: {decision['official_baseline_rerun_performed']}",
        f"- Test-label tuning: {decision['test_label_tuning']}",
        f"- Synthetic outputs used: {decision['synthetic_outputs_used']}",
        f"- Random predictions used: {decision['random_predictions_used']}",
        f"- Full prepared CopCo join used: {decision['full_prepared_copco_join_used']}",
        "",
        "## Best Candidate",
        "```json",
        json.dumps(best, indent=2, sort_keys=True),
        "```",
        "",
        "## Trial-Level Primary Metrics",
        _markdown_table(trial_metrics.to_dict("records"), METRIC_COLUMNS, max_rows=20),
        "",
        "## Reader-Aggregated Secondary Metrics",
        _markdown_table(reader_metrics.to_dict("records"), METRIC_COLUMNS, max_rows=20),
        "",
        "## Top Inner-Validation Candidates",
        _markdown_table(
            candidate_summary.sort_values("selection_score", ascending=False, na_position="last")
            .head(20)
            .to_dict("records"),
            list(candidate_summary.columns),
            max_rows=20,
        ),
    ]
    _write_report(dirs, "optimization_report.md", "\n".join(lines))
    leak_lines = [
        "# D3 EyeBench Protocol-Aligned Leakage Report",
        "",
        f"- Status: `{leakage.get('status')}`",
        f"- Threshold selection source: `{leakage.get('threshold_selection_source')}`",
        f"- Official Test labels used for selection: {leakage.get('official_test_labels_used_for_selection')}",
        f"- Held-out reader rows used for residual fit: {leakage.get('heldout_reader_rows_used_for_residual_fit')}",
        f"- Held-out text rows used for residual fit: {leakage.get('heldout_text_rows_used_for_residual_fit')}",
        f"- Reader group used in residualization: {leakage.get('reader_group_used_in_residualization')}",
        f"- Denied predictors present: {leakage.get('denied_predictors_present')}",
        f"- Synthetic outputs: {not leakage.get('no_synthetic_outputs')}",
        f"- Random predictions: {not leakage.get('no_random_predictions')}",
        f"- Full prepared CopCo join: {not leakage.get('no_full_prepared_copco_join')}",
    ]
    _write_report(dirs, "leakage_report.md", "\n".join(leak_lines))
    decision_lines = [
        "# D3 EyeBench Protocol-Aligned Decision Report",
        "",
        f"- Decision category: `{decision['decision_category']}`",
        f"- Official SOTA claim allowed: {decision['official_sota_claim_allowed']}",
        f"- Final average balanced accuracy: {decision['checks']['final_average_balanced_accuracy']:.4f}"
        if decision["checks"]["final_average_balanced_accuracy"] is not None
        else "- Final average balanced accuracy: missing",
        f"- Official target balanced accuracy: {decision['checks']['target_balanced_accuracy']:.4f}",
        "- Local Logistic anchor average balanced accuracy: "
        f"{decision['checks']['local_logistic_anchor_average_balanced_accuracy']:.4f}"
        if decision["checks"]["local_logistic_anchor_average_balanced_accuracy"] is not None
        else "- Local Logistic anchor average balanced accuracy: missing",
        f"- Beats official target: {decision['checks']['beats_official_target']}",
        f"- Beats local Logistic anchor: {decision['checks']['beats_local_logistic_anchor']}",
    ]
    _write_report(dirs, "decision_report.md", "\n".join(decision_lines))


def validate_d3_eyebench_protocol_optimization(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    config_report = validate_d3_eyebench_protocol_optimization_config(config)
    if config_report["status"] != "passed":
        errors.extend(config_report["errors"])
    required = [
        out / "manifest.json",
        out / "config_validation.json",
        out / "preflight" / "preflight_report.json",
        out / "splits" / "official_split_validation.json",
        out / "optimization" / "candidate_summary.csv",
        out / "optimization" / "inner_validation_fold_metrics.csv",
        out / "typ" / "d3_optimized_trial_metrics.csv",
        out / "typ" / "d3_optimized_reader_aggregated_metrics.csv",
        out / "typ" / "trial_level_test_results.csv",
        out / "leakage_report.json",
        out / "official_sota_decision.json",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required output: {path}")
    dirs = _analysis_dirs(config, out, root)
    for name in [
        "manifest.json",
        "official_sota_decision.json",
        "leakage_report.json",
        "optimization_report.md",
        "leakage_report.md",
        "decision_report.md",
    ]:
        if not (dirs["repo_analysis"] / name).exists():
            errors.append(f"missing repo analysis artifact: {dirs['repo_analysis'] / name}")
    decision: dict[str, Any] = {}
    if (out / "official_sota_decision.json").exists():
        decision = json.loads((out / "official_sota_decision.json").read_text(encoding="utf-8"))
        category = decision.get("decision_category")
        if category not in VALID_DECISION_CATEGORIES:
            errors.append(f"invalid decision category: {category}")
        if decision.get("test_label_tuning") is not False:
            errors.append("decision indicates test-label tuning")
        if decision.get("synthetic_outputs_used") is not False:
            errors.append("decision indicates synthetic outputs")
        if decision.get("random_predictions_used") is not False:
            errors.append("decision indicates random predictions")
        if decision.get("full_prepared_copco_join_used") is not False:
            errors.append("decision indicates full prepared CopCo join")
        if decision.get("official_baseline_rerun_performed") is not False:
            errors.append("decision indicates official baseline rerun")
        if decision.get("selection_source") != "inner_validation_only":
            errors.append("selection source is not inner validation only")
    if (out / "leakage_report.json").exists():
        leakage = json.loads((out / "leakage_report.json").read_text(encoding="utf-8"))
        if leakage.get("status") != "passed":
            errors.append("leakage report did not pass")
        if leakage.get("official_test_labels_used_for_selection"):
            errors.append("official test labels used for selection")
        if leakage.get("denied_predictors_present"):
            errors.append(f"denied predictors present: {leakage['denied_predictors_present']}")
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if bool(_section(config).get("require_slurm_job", True)) and not manifest.get("slurm_job_id"):
            errors.append("manifest missing Slurm job id")
    if (out / "typ" / "d3_optimized_trial_metrics.csv").exists():
        pd = _pd()
        metrics = pd.read_csv(out / "typ" / "d3_optimized_trial_metrics.csv")
        if set(OFFICIAL_SPLITS) - set(metrics.get("split_name", [])):
            errors.append("trial metrics missing official split rows")
        if "balanced_accuracy" not in metrics:
            errors.append("trial metrics missing balanced_accuracy")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "decision_category": decision.get("decision_category"),
        "official_sota_claim_allowed": bool(decision.get("official_sota_claim_allowed")),
    }
