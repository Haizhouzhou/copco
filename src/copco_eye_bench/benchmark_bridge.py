"""EyeBench-style BenchmarkBridge v1 evaluation for frozen CopCo DFM profiles."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir
from .research_exploration import (
    _classification_metrics,
    _format_value,
    _markdown_table,
    _merge_boundary_vocoid,
    _np,
    _pd,
    _score_estimator,
    _simple_slope,
    _with_derived_columns,
)


BENCHMARK_SPLITS = [
    "unseen_reader",
    "unseen_text",
    "unseen_reader_and_text",
    "text_balanced_unseen_reader",
    "leave_one_speech_out",
    "participant_grouped_kfold",
]

PRIMARY_TYPO_FEATURE_GROUP = "D3_dfm_residual_gaze_only"

PROHIBITED_FEATURES = {
    "n_words_read",
    "n_speeches",
    "n_word_rows",
    "total_word_rows",
    "word_observation_count",
    "participant_id",
    "speech_id",
    "text_id",
    "reader_group",
    "reader_group_binary",
    "reader_group_binary_num",
    "dyslexia_labeled",
    "group_label",
}

RESIDUALIZATION_PREDICTORS = [
    "word_length_chars",
    "log_corpus_frequency",
    "dfm_lm_word_surprisal",
    "dfm_lm_word_entropy",
    "sentence_length_words",
    "word_position_in_sentence_norm",
    "prev_boundary_opacity_score",
    "vv_indicator",
    "gaze_missing",
    "lm_missing",
    "embedding_missing",
    "parser_missing",
    "segmentation_label_missing",
]

RESIDUALIZATION_FORBIDDEN = {
    "reader_group",
    "reader_group_binary",
    "reader_group_binary_num",
    "dyslexia_labeled",
    "group_label",
    "participant_id",
    "speech_id",
    "text_id",
}

RESIDUAL_OUTCOME_SPECS = [
    ("log_ffd", "ffd", False),
    ("log_first_pass_duration", "first_pass", False),
    ("log_go_past_time", "go_past", False),
    ("log_total_fixation_duration", "total_fixation", False),
    ("skip", "skipping", True),
    ("fixation_count", "fixation_count", False),
]

RAW_GAZE_FEATURES = [
    "mean_ffd",
    "median_ffd",
    "mean_gd",
    "median_gd",
    "mean_trt",
    "median_trt",
    "mean_go_past_time",
    "mean_fixation_count",
    "skipping_rate",
    "refixation_rate",
    "trt_sd",
    "trt_q90",
    "gaze_missing_rate",
]

READING_SPEED_FEATURES = [
    "mean_ffd",
    "mean_gd",
    "mean_trt",
    "mean_go_past_time",
    "skipping_rate",
]

EXPOSURE_FEATURES = [
    "mean_word_length_exposure",
    "mean_log_frequency_exposure",
    "mean_sentence_length_exposure",
    "mean_dfm_surprisal_exposure",
    "mean_dfm_entropy_exposure",
    "mean_boundary_opacity_exposure",
    "vv_boundary_exposure_rate",
    "lm_missing_rate",
    "embedding_missing_rate",
    "parser_missing_rate",
    "segmentation_missing_rate",
]

DFM_EXPOSURE_FEATURES = [
    "mean_dfm_surprisal_exposure",
    "mean_dfm_entropy_exposure",
    "lm_missing_rate",
]

SENSITIVITY_FEATURES = [
    "sample_trt_dfm_surprisal_slope",
    "sample_trt_dfm_entropy_slope",
    "sample_go_past_dfm_surprisal_slope",
    "sample_go_past_dfm_entropy_slope",
]

SEGMENTATION_SENSITIVITY_FEATURES = [
    "sample_trt_boundary_opacity_slope",
    "sample_trt_vv_cost",
    "sample_go_past_boundary_opacity_slope",
]

TYP_METRIC_COLUMNS = [
    "task",
    "split_name",
    "feature_group",
    "model",
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
    "status",
    "skip_reason",
]

RCS_METRIC_COLUMNS = [
    "task",
    "split_name",
    "feature_group",
    "model",
    "evaluation_level",
    "target",
    "target_scale",
    "n_features",
    "n_predictions",
    "usable_folds",
    "skipped_folds",
    "rmse",
    "mae",
    "r2",
    "status",
    "skip_reason",
]


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


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["result_analysis"] / name, text)
    _write_md(dirs["repo_analysis"] / name, text)


def _write_analysis_csv(dirs: dict[str, Path], name: str, frame: Any) -> None:
    _write_csv(dirs["result_analysis"] / name, frame)
    _write_csv(dirs["repo_analysis"] / name, frame)


def _write_analysis_table(dirs: dict[str, Path], name: str, frame: Any) -> None:
    _write_csv(dirs["result_tables"] / name, frame)
    _write_csv(dirs["repo_tables"] / name, frame)


def _git_sha(repo_root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    analysis_rel = str(
        get_nested(config, "benchmark_bridge.output_layout.analysis", "analysis/benchmark_bridge_v1")
    )
    tables_rel = str(
        get_nested(
            config,
            "benchmark_bridge.output_layout.tables",
            "analysis/benchmark_bridge_v1/tables",
        )
    )
    repo_analysis = root / str(
        get_nested(config, "benchmark_bridge.repo_analysis_dir", "analysis/benchmark_bridge_v1")
    )
    return {
        "repo_analysis": repo_analysis,
        "repo_tables": repo_analysis / "tables",
        "result_analysis": out / analysis_rel,
        "result_tables": out / tables_rel,
    }


def _configured_path(config: dict[str, Any], dotted: str, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    value = get_nested(config, dotted)
    if value is None:
        raise ValueError(f"missing required config path: {dotted}")
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_inputs(config: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    pd = _pd()
    label_dir = _configured_path(config, "benchmark_bridge.frozen_inputs.label_release_dir", repo_root)
    prepared = _configured_path(
        config,
        "benchmark_bridge.frozen_inputs.prepared_dataset_dir",
        repo_root,
    )
    return {
        "label_dir": label_dir,
        "prepared_dir": prepared,
        "word": pd.read_parquet(prepared / "analysis_ready_word_level_v1_1.parquet"),
        "participant": pd.read_parquet(prepared / "analysis_ready_participant_level_v1_1.parquet"),
        "participant_labels": pd.read_parquet(label_dir / "labels" / "participant_labels_v1.parquet"),
        "splits": pd.read_parquet(label_dir / "labels" / "split_labels_v1.parquet"),
        "segmentation_boundary": pd.read_parquet(
            label_dir / "labels" / "segmentation_boundary_labels_v1.parquet"
        ),
        "quality": pd.read_parquet(label_dir / "labels" / "quality_labels_v1.parquet"),
    }


def validate_benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings_list: list[str] = []
    section = get_nested(config, "benchmark_bridge", {})
    if not isinstance(section, dict):
        errors.append("missing benchmark_bridge config section")
        section = {}
    if section.get("no_new_labels") is not True:
        errors.append("benchmark bridge must not create new labels")
    if section.get("no_feature_engineering_search") is not True:
        errors.append("benchmark bridge must not run broad feature-engineering search")
    if section.get("forbid_random_word_level_split") is not True:
        errors.append("random word-level split prohibition is not enabled")
    configured_splits = list(section.get("split_regimes", []))
    missing_splits = sorted(set(BENCHMARK_SPLITS) - set(configured_splits))
    if missing_splits:
        errors.append(f"missing split regimes: {missing_splits}")
    prohibited = set(section.get("prohibited_features", []))
    missing_prohibited = sorted(PROHIBITED_FEATURES - prohibited)
    if missing_prohibited:
        errors.append(f"config prohibited feature list incomplete: {missing_prohibited}")
    residualization = section.get("residualization", {})
    if isinstance(residualization, dict) and residualization.get("reader_group_never_used") is not True:
        errors.append("residualization.reader_group_never_used must be true")
    if RESIDUALIZATION_FORBIDDEN.intersection(RESIDUALIZATION_PREDICTORS):
        errors.append("residualization predictor constants include forbidden identifiers")
    if "CopCo_RCS" not in [task.get("name") for task in section.get("tasks", []) if isinstance(task, dict)]:
        warnings_list.append("CopCo_RCS task is not configured")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
    }


def _bool_rate(frame: Any, column: str) -> float | None:
    pd = _pd()
    if column not in frame:
        return None
    values = frame[column]
    if values.empty:
        return None
    if values.dtype == bool:
        return float(values.mean())
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())
    return float(values.astype(str).str.lower().isin(["true", "1", "yes"]).mean())


def _numeric_mean(frame: Any, column: str) -> float | None:
    pd = _pd()
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else None


def _numeric_median(frame: Any, column: str) -> float | None:
    pd = _pd()
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.median()) if values.notna().any() else None


def _numeric_std(frame: Any, column: str) -> float | None:
    pd = _pd()
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.std()) if values.notna().sum() > 1 else None


def _numeric_quantile(frame: Any, column: str, q: float) -> float | None:
    pd = _pd()
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.quantile(q)) if values.notna().any() else None


def _slope_or_zero(frame: Any, x_col: str, y_col: str, min_words: int) -> tuple[float, bool]:
    pd = _pd()
    if x_col not in frame or y_col not in frame:
        return 0.0, True
    x = pd.to_numeric(frame[x_col], errors="coerce")
    y = pd.to_numeric(frame[y_col], errors="coerce")
    if int((x.notna() & y.notna()).sum()) < min_words:
        return 0.0, True
    value = _simple_slope(frame, x_col, y_col)
    if value is None or not math.isfinite(float(value)):
        return 0.0, True
    return float(value), False


def _mean_difference_or_zero(frame: Any, mask_col: str, y_col: str) -> float:
    pd = _pd()
    if mask_col not in frame or y_col not in frame:
        return 0.0
    mask = pd.to_numeric(frame[mask_col], errors="coerce").eq(1)
    high = pd.to_numeric(frame.loc[mask, y_col], errors="coerce")
    low = pd.to_numeric(frame.loc[~mask, y_col], errors="coerce")
    if not high.notna().any() or not low.notna().any():
        return 0.0
    return float(high.mean() - low.mean())


def _base_sample_row(group: Any, sample_id: str, min_words: int) -> dict[str, Any]:
    first = group.iloc[0]
    text_id = str(first.get("speech_id"))
    n_words = int(len(group))
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "participant_id": str(first.get("participant_id")),
        "reader_group": first.get("reader_group"),
        "reader_group_binary": int(first.get("reader_group_binary")),
        "speech_id": text_id,
        "text_id": text_id,
        "paragraph_id": first.get("paragraph_id") if group["paragraph_id"].nunique(dropna=True) == 1 else None,
        "passage_id": text_id,
        "n_words_in_sample": n_words,
        "dyslexia_labeled": int(first.get("dyslexia_labeled")),
        "group_label": first.get("group_label"),
        "comprehension_score": first.get("comprehension_score"),
        "include_primary_analysis": bool(first.get("include_primary_analysis", True)),
        "include_sensitivity_analysis": bool(first.get("include_sensitivity_analysis", True)),
        "eligible_typ": bool(first.get("include_primary_analysis", True)),
        "eligible_rcs": first.get("comprehension_score") == first.get("comprehension_score"),
    }
    row.update(
        {
            "mean_ffd": _numeric_mean(group, "FFD"),
            "median_ffd": _numeric_median(group, "FFD"),
            "mean_gd": _numeric_mean(group, "GD"),
            "median_gd": _numeric_median(group, "GD"),
            "mean_trt": _numeric_mean(group, "TRT"),
            "median_trt": _numeric_median(group, "TRT"),
            "mean_go_past_time": _numeric_mean(group, "go_past_time"),
            "mean_fixation_count": _numeric_mean(group, "fixation_count"),
            "skipping_rate": _numeric_mean(group, "skip"),
            "refixation_rate": _numeric_mean(group, "refixation_count"),
            "trt_sd": _numeric_std(group, "TRT"),
            "trt_q90": _numeric_quantile(group, "TRT", 0.90),
            "mean_landing_position": _numeric_mean(group, "landing_position"),
            "mean_word_length_exposure": _numeric_mean(group, "word_length_chars"),
            "mean_log_frequency_exposure": _numeric_mean(group, "log_corpus_frequency"),
            "mean_sentence_length_exposure": _numeric_mean(group, "sentence_length_words"),
            "mean_dfm_surprisal_exposure": _numeric_mean(group, "dfm_lm_word_surprisal"),
            "mean_dfm_entropy_exposure": _numeric_mean(group, "dfm_lm_word_entropy"),
            "mean_boundary_opacity_exposure": _numeric_mean(group, "prev_boundary_opacity_score"),
            "vv_boundary_exposure_rate": _numeric_mean(group, "vv_indicator"),
            "gaze_missing_rate": _bool_rate(group, "gaze_missing"),
            "lm_missing_rate": _bool_rate(group, "lm_missing"),
            "embedding_missing_rate": _bool_rate(group, "embedding_missing"),
            "parser_missing_rate": _bool_rate(group, "parser_missing"),
            "segmentation_missing_rate": _bool_rate(group, "segmentation_label_missing"),
        }
    )
    slope_specs = [
        ("sample_trt_dfm_surprisal_slope", "dfm_lm_word_surprisal", "log_total_fixation_duration"),
        ("sample_trt_dfm_entropy_slope", "dfm_lm_word_entropy", "log_total_fixation_duration"),
        ("sample_trt_boundary_opacity_slope", "prev_boundary_opacity_score", "log_total_fixation_duration"),
        ("sample_trt_word_length_slope", "word_length_chars", "log_total_fixation_duration"),
        ("sample_go_past_dfm_surprisal_slope", "dfm_lm_word_surprisal", "log_go_past_time"),
        ("sample_go_past_dfm_entropy_slope", "dfm_lm_word_entropy", "log_go_past_time"),
        ("sample_go_past_boundary_opacity_slope", "prev_boundary_opacity_score", "log_go_past_time"),
    ]
    unstable = 0
    for name, x_col, y_col in slope_specs:
        value, is_unstable = _slope_or_zero(group, x_col, y_col, min_words)
        row[name] = value
        unstable += int(is_unstable)
    row["sample_trt_vv_cost"] = _mean_difference_or_zero(group, "vv_indicator", "log_total_fixation_duration")
    row["sample_sensitivity_unstable"] = bool(unstable)
    row["sample_sensitivity_unstable_count"] = int(unstable)
    return row


def build_participant_text_trial_table(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
) -> tuple[Any, Any]:
    pd = _pd()
    min_words = int(get_nested(config, "benchmark_bridge.residualization.min_words_for_slope", 8))
    word = _with_derived_columns(_merge_boundary_vocoid(inputs["word"], inputs["segmentation_boundary"]))
    word["participant_id"] = word["participant_id"].astype(str)
    word["speech_id"] = word["speech_id"].astype(str)
    word["text_id"] = word["speech_id"].astype(str)
    word["sample_id"] = word["participant_id"] + "::" + word["text_id"]
    include = word.get("include_primary_analysis", True)
    if not isinstance(include, bool):
        word = word[include.fillna(False).astype(bool)].copy()
    rows = [
        _base_sample_row(group, str(sample_id), min_words)
        for sample_id, group in word.groupby("sample_id", sort=True, dropna=False)
    ]
    samples = pd.DataFrame(rows).sort_values(["participant_id", "text_id"]).reset_index(drop=True)
    descriptive = build_global_descriptive_residual_features(config, word, samples)
    samples = samples.merge(descriptive, on="sample_id", how="left") if not descriptive.empty else samples
    _write_parquet(out / "data" / "participant_text_trial_features.parquet", samples)
    write_participant_text_feature_dictionary(dirs, samples)
    return samples, word


def _residual_design_matrices(train: Any, apply: Any) -> tuple[Any, Any, list[str]]:
    pd = _pd()
    train_x = train[[c for c in RESIDUALIZATION_PREDICTORS if c in train]].copy()
    apply_x = apply[train_x.columns.tolist()].copy()
    for column in train_x.columns:
        train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
        apply_x[column] = pd.to_numeric(apply_x[column], errors="coerce")
    return train_x, apply_x, train_x.columns.astype(str).tolist()


def _fit_apply_residuals(
    train: Any,
    apply: Any,
    outcome: str,
    *,
    binary: bool,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    pd = _pd()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = pd.to_numeric(train[outcome], errors="coerce")
    fit_mask = y_train.notna()
    if binary:
        fit_mask = fit_mask & y_train.isin([0, 1])
    fit_train = train.loc[fit_mask].copy()
    if fit_train.empty:
        return pd.Series(index=apply.index, dtype=float), {
            "outcome": outcome,
            "status": "skipped",
            "skip_reason": "no_training_rows",
            "n_train_rows": 0,
            "n_apply_rows": int(len(apply)),
            "n_predictors": 0,
            "uses_reader_group": False,
        }
    y_fit = pd.to_numeric(fit_train[outcome], errors="coerce")
    if binary and y_fit.nunique(dropna=True) < 2:
        return pd.Series(index=apply.index, dtype=float), {
            "outcome": outcome,
            "status": "skipped",
            "skip_reason": "insufficient_binary_outcome_variation",
            "n_train_rows": int(len(fit_train)),
            "n_apply_rows": int(len(apply)),
            "n_predictors": 0,
            "uses_reader_group": False,
        }
    x_train, x_apply, columns = _residual_design_matrices(fit_train, apply)
    if RESIDUALIZATION_FORBIDDEN.intersection(columns):
        raise ValueError("forbidden columns leaked into residualization design")
    if binary:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=False),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        )
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), Ridge(alpha=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_fit.astype(int) if binary else y_fit)
    prediction = model.predict_proba(x_apply)[:, 1] if binary else model.predict(x_apply)
    y_apply = pd.to_numeric(apply[outcome], errors="coerce")
    residual = pd.Series(y_apply.to_numpy(dtype=float) - prediction, index=apply.index, dtype=float)
    return residual, {
        "outcome": outcome,
        "status": "complete",
        "skip_reason": "",
        "n_train_rows": int(len(fit_train)),
        "n_apply_rows": int(len(apply)),
        "n_predictors": int(len(columns)),
        "uses_reader_group": False,
    }


def _fit_apply_residuals_pair(
    train: Any,
    train_apply: Any,
    test_apply: Any,
    outcome: str,
    *,
    binary: bool,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    pd = _pd()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = pd.to_numeric(train[outcome], errors="coerce")
    fit_mask = y_train.notna()
    if binary:
        fit_mask = fit_mask & y_train.isin([0, 1])
    fit_train = train.loc[fit_mask].copy()
    base_diag = {
        "outcome": outcome,
        "n_apply_train_rows": int(len(train_apply)),
        "n_apply_test_rows": int(len(test_apply)),
        "uses_reader_group": False,
    }
    if fit_train.empty:
        empty_train = pd.Series(index=train_apply.index, dtype=float)
        empty_test = pd.Series(index=test_apply.index, dtype=float)
        return empty_train, empty_test, {
            **base_diag,
            "status": "skipped",
            "skip_reason": "no_training_rows",
            "n_train_rows": 0,
            "n_predictors": 0,
        }
    y_fit = pd.to_numeric(fit_train[outcome], errors="coerce")
    if binary and y_fit.nunique(dropna=True) < 2:
        empty_train = pd.Series(index=train_apply.index, dtype=float)
        empty_test = pd.Series(index=test_apply.index, dtype=float)
        return empty_train, empty_test, {
            **base_diag,
            "status": "skipped",
            "skip_reason": "insufficient_binary_outcome_variation",
            "n_train_rows": int(len(fit_train)),
            "n_predictors": 0,
        }
    x_fit, x_train_apply, columns = _residual_design_matrices(fit_train, train_apply)
    _, x_test_apply, _ = _residual_design_matrices(fit_train, test_apply)
    if RESIDUALIZATION_FORBIDDEN.intersection(columns):
        raise ValueError("forbidden columns leaked into residualization design")
    if binary:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=False),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        )
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), Ridge(alpha=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_fit, y_fit.astype(int) if binary else y_fit)
    train_prediction = model.predict_proba(x_train_apply)[:, 1] if binary else model.predict(x_train_apply)
    test_prediction = model.predict_proba(x_test_apply)[:, 1] if binary else model.predict(x_test_apply)
    train_y_apply = pd.to_numeric(train_apply[outcome], errors="coerce")
    test_y_apply = pd.to_numeric(test_apply[outcome], errors="coerce")
    train_residual = pd.Series(
        train_y_apply.to_numpy(dtype=float) - train_prediction,
        index=train_apply.index,
        dtype=float,
    )
    test_residual = pd.Series(
        test_y_apply.to_numpy(dtype=float) - test_prediction,
        index=test_apply.index,
        dtype=float,
    )
    return train_residual, test_residual, {
        **base_diag,
        "status": "complete",
        "skip_reason": "",
        "n_train_rows": int(len(fit_train)),
        "n_predictors": int(len(columns)),
    }


def _aggregate_residual_sample(group: Any, sample_id: str, min_words: int, *, prefix: str) -> dict[str, Any]:
    pd = _pd()
    row: dict[str, Any] = {"sample_id": sample_id}
    unstable = 0
    for _, outcome_prefix, _ in RESIDUAL_OUTCOME_SPECS:
        source = f"{prefix}_{outcome_prefix}_residual"
        if source not in group:
            continue
        values = pd.to_numeric(group[source], errors="coerce")
        row[f"{source}_mean"] = float(values.mean()) if values.notna().any() else None
        row[f"{source}_median"] = float(values.median()) if values.notna().any() else None
        row[f"{source}_sd"] = float(values.std()) if values.notna().sum() > 1 else None
        for x_name, x_col in [
            ("dfm_surprisal", "dfm_lm_word_surprisal"),
            ("dfm_entropy", "dfm_lm_word_entropy"),
            ("word_length", "word_length_chars"),
            ("boundary_opacity", "prev_boundary_opacity_score"),
        ]:
            value, is_unstable = _slope_or_zero(group, x_col, source, min_words)
            row[f"{source}_{x_name}_slope"] = value
            unstable += int(is_unstable)
        row[f"{source}_high_opacity_cost"] = _mean_difference_or_zero(
            group.assign(high_opacity=pd.to_numeric(group["prev_boundary_opacity_score"], errors="coerce").eq(3)),
            "high_opacity",
            source,
        )
        row[f"{source}_vv_cost"] = _mean_difference_or_zero(group, "vv_indicator", source)
    row[f"{prefix}_residual_unstable_count"] = int(unstable)
    row[f"{prefix}_residual_unstable"] = bool(unstable)
    return row


def build_global_descriptive_residual_features(config: dict[str, Any], word: Any, samples: Any) -> Any:
    pd = _pd()
    seed = int(get_nested(config, "benchmark_bridge.deterministic_seed", 97))
    min_words = int(get_nested(config, "benchmark_bridge.residualization.min_words_for_slope", 8))
    data = word.copy()
    for outcome, prefix, binary in RESIDUAL_OUTCOME_SPECS:
        residual, _ = _fit_apply_residuals(data, data, outcome, binary=binary, seed=seed)
        data[f"global_{prefix}_residual"] = residual
    rows = [
        _aggregate_residual_sample(group, str(sample_id), min_words, prefix="global")
        for sample_id, group in data.groupby("sample_id", sort=True, dropna=False)
    ]
    frame = pd.DataFrame(rows)
    keep = ["sample_id", *[c for c in frame.columns if c.startswith("global_")]]
    return frame[keep] if not frame.empty else frame


def write_participant_text_feature_dictionary(dirs: dict[str, Path], samples: Any) -> None:
    rows = []
    categories = [
        ("identifiers", ["sample_id", "participant_id", "speech_id", "text_id", "paragraph_id", "passage_id"]),
        ("targets", ["reader_group_binary", "comprehension_score"]),
        ("raw_gaze", RAW_GAZE_FEATURES),
        ("reading_speed", READING_SPEED_FEATURES),
        ("dfm_exposure", DFM_EXPOSURE_FEATURES),
        ("exposure", EXPOSURE_FEATURES),
        ("sensitivity", SENSITIVITY_FEATURES + SEGMENTATION_SENSITIVITY_FEATURES),
        ("descriptive_global_residual", [c for c in samples.columns if c.startswith("global_")]),
        ("quality", [c for c in samples.columns if c.endswith("_missing_rate") or "unstable" in c]),
    ]
    columns = set(samples.columns.astype(str))
    for category, names in categories:
        for name in names:
            if name in columns:
                rows.append(
                    {
                        "feature": name,
                        "category": category,
                        "primary_predictor_allowed": name not in PROHIBITED_FEATURES
                        and not name.startswith("global_"),
                        "notes": (
                            "descriptive only; split-specific cross-fitted residual features are "
                            "used for primary benchmark models"
                            if name.startswith("global_")
                            else ""
                        ),
                    }
                )
    text = "\n".join(
        [
            "# Participant-Text Feature Dictionary",
            "",
            "Rows are participant-by-speech sample units. Identifiers are retained for grouping, "
            "fold construction, diagnostics, and reports; they are not primary model predictors.",
            "",
            _markdown_table(
                rows,
                ["feature", "category", "primary_predictor_allowed", "notes"],
                max_rows=300,
            ),
        ]
    )
    _write_report(dirs, "participant_text_feature_dictionary.md", text)


def _assign_stratified_participant_folds(samples: Any, n_folds: int) -> dict[str, int]:
    participants = (
        samples[["participant_id", "reader_group_binary"]]
        .drop_duplicates("participant_id")
        .sort_values(["reader_group_binary", "participant_id"])
    )
    fold_map: dict[str, int] = {}
    for _, group in participants.groupby("reader_group_binary", sort=True):
        for idx, participant_id in enumerate(group["participant_id"].astype(str).tolist()):
            fold_map[participant_id] = idx % n_folds
    return fold_map


def _assign_text_folds(samples: Any, n_folds: int) -> dict[str, int]:
    counts = samples.groupby("text_id", dropna=False)["participant_id"].nunique().sort_values(ascending=False)
    return {str(text_id): idx % n_folds for idx, text_id in enumerate(counts.index.astype(str).tolist())}


def _append_split_rows(
    rows: list[dict[str, Any]],
    samples: Any,
    *,
    split_name: str,
    fold_id: int,
    train_mask: Any,
    test_mask: Any,
    seed: int,
) -> None:
    train = samples.loc[train_mask].copy()
    test = samples.loc[test_mask].copy()
    train_y = train["reader_group_binary"].dropna()
    test_y = test["reader_group_binary"].dropna()
    split_valid = bool(not train.empty and not test.empty and train_y.nunique() >= 2 and not test_y.empty)
    train_p = set(train["participant_id"].astype(str))
    test_p = set(test["participant_id"].astype(str))
    train_t = set(train["text_id"].astype(str))
    test_t = set(test["text_id"].astype(str))
    for _, sample in samples.iterrows():
        in_train = bool(train_mask.loc[sample.name])
        in_test = bool(test_mask.loc[sample.name])
        if in_train and in_test:
            role = "invalid_overlap"
        elif in_train:
            role = "train"
        elif in_test:
            role = "test"
        else:
            role = "exclude"
        rows.append(
            {
                "split_name": split_name,
                "fold_id": int(fold_id),
                "sample_id": sample["sample_id"],
                "participant_id": sample["participant_id"],
                "speech_id": sample["speech_id"],
                "text_id": sample["text_id"],
                "reader_group": sample["reader_group"],
                "reader_group_binary": sample["reader_group_binary"],
                "split_role": role,
                "include_in_fold": role in {"train", "test"},
                "n_train_samples": int(len(train)),
                "n_test_samples": int(len(test)),
                "n_train_participants": int(len(train_p)),
                "n_test_participants": int(len(test_p)),
                "n_train_texts": int(len(train_t)),
                "n_test_texts": int(len(test_t)),
                "n_train_dyslexia_labeled": int(pd_numeric_sum(train, "reader_group_binary")),
                "n_test_dyslexia_labeled": int(pd_numeric_sum(test, "reader_group_binary")),
                "n_train_typical_control": int(len(train) - pd_numeric_sum(train, "reader_group_binary")),
                "n_test_typical_control": int(len(test) - pd_numeric_sum(test, "reader_group_binary")),
                "participant_overlap": bool(train_p.intersection(test_p)),
                "text_overlap": bool(train_t.intersection(test_t)),
                "split_valid": split_valid,
                "skip_reason": "" if split_valid else "empty_or_single_class_training_fold",
                "split_seed": int(seed),
                "split_version": "benchmark_bridge_v1_internal_matched",
            }
        )


def pd_numeric_sum(frame: Any, column: str) -> float:
    pd = _pd()
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def build_benchmark_splits(config: dict[str, Any], out: Path, samples: Any) -> Any:
    pd = _pd()
    seed = int(get_nested(config, "benchmark_bridge.deterministic_seed", 97))
    rows: list[dict[str, Any]] = []
    samples = samples.reset_index(drop=True).copy()
    samples["participant_id"] = samples["participant_id"].astype(str)
    samples["text_id"] = samples["text_id"].astype(str)
    participants = sorted(samples["participant_id"].unique().tolist())
    texts = sorted(samples["text_id"].unique().tolist())

    for fold_id, participant_id in enumerate(participants):
        test_mask = samples["participant_id"].eq(participant_id)
        train_mask = ~test_mask
        _append_split_rows(
            rows,
            samples,
            split_name="unseen_reader",
            fold_id=fold_id,
            train_mask=train_mask,
            test_mask=test_mask,
            seed=seed,
        )

    for fold_id, text_id in enumerate(texts):
        test_mask = samples["text_id"].eq(text_id)
        train_mask = ~test_mask
        for split_name in ["unseen_text", "leave_one_speech_out"]:
            _append_split_rows(
                rows,
                samples,
                split_name=split_name,
                fold_id=fold_id,
                train_mask=train_mask,
                test_mask=test_mask,
                seed=seed,
            )

    grouped_folds = int(get_nested(config, "benchmark_bridge.split_policy.participant_grouped_kfold_folds", 5))
    participant_folds = _assign_stratified_participant_folds(samples, grouped_folds)
    for fold_id in range(grouped_folds):
        test_participants = {pid for pid, fold in participant_folds.items() if fold == fold_id}
        test_mask = samples["participant_id"].isin(test_participants)
        train_mask = ~test_mask
        for split_name in ["participant_grouped_kfold", "text_balanced_unseen_reader"]:
            _append_split_rows(
                rows,
                samples,
                split_name=split_name,
                fold_id=fold_id,
                train_mask=train_mask,
                test_mask=test_mask,
                seed=seed,
            )

    strict_folds = int(get_nested(config, "benchmark_bridge.split_policy.unseen_reader_and_text_folds", 5))
    strict_participant_folds = _assign_stratified_participant_folds(samples, strict_folds)
    strict_text_folds = _assign_text_folds(samples, strict_folds)
    for fold_id in range(strict_folds):
        test_participants = {pid for pid, fold in strict_participant_folds.items() if fold == fold_id}
        test_texts = {text for text, fold in strict_text_folds.items() if fold == fold_id}
        test_mask = samples["participant_id"].isin(test_participants) & samples["text_id"].isin(test_texts)
        train_mask = ~samples["participant_id"].isin(test_participants) & ~samples["text_id"].isin(test_texts)
        _append_split_rows(
            rows,
            samples,
            split_name="unseen_reader_and_text",
            fold_id=fold_id,
            train_mask=train_mask,
            test_mask=test_mask,
            seed=seed,
        )

    splits = pd.DataFrame(rows)
    _write_parquet(out / "splits" / "benchmark_split_labels.parquet", splits)
    return splits


def validate_split_labels(splits: Any) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    if splits["split_name"].astype(str).str.contains("random", case=False, na=False).any():
        errors.append("random split label found")
    for (split_name, fold_id), fold in splits.groupby(["split_name", "fold_id"], dropna=False):
        train = fold[fold["split_role"].eq("train")]
        test = fold[fold["split_role"].eq("test")]
        train_participants = set(train["participant_id"].astype(str))
        test_participants = set(test["participant_id"].astype(str))
        train_texts = set(train["text_id"].astype(str))
        test_texts = set(test["text_id"].astype(str))
        if split_name in {"unseen_reader", "text_balanced_unseen_reader", "participant_grouped_kfold"}:
            if train_participants.intersection(test_participants):
                errors.append(f"participant overlap in {split_name} fold {fold_id}")
        if split_name in {"unseen_text", "leave_one_speech_out"}:
            if train_texts.intersection(test_texts):
                errors.append(f"text overlap in {split_name} fold {fold_id}")
        if split_name == "unseen_reader_and_text":
            if train_participants.intersection(test_participants):
                errors.append(f"participant overlap in {split_name} fold {fold_id}")
            if train_texts.intersection(test_texts):
                errors.append(f"text overlap in {split_name} fold {fold_id}")
        train_y = train["reader_group_binary"].dropna()
        test_y = test["reader_group_binary"].dropna()
        if not train.empty and train_y.nunique() < 2:
            errors.append(f"TYP train fold lacks both classes: {split_name} fold {fold_id}")
        if test.empty or test_y.empty:
            errors.append(f"test fold has no valid labels: {split_name} fold {fold_id}")
        summaries.append(
            {
                "split_name": split_name,
                "fold_id": int(fold_id),
                "train_samples": int(len(train)),
                "test_samples": int(len(test)),
                "train_participants": int(len(train_participants)),
                "test_participants": int(len(test_participants)),
                "train_texts": int(len(train_texts)),
                "test_texts": int(len(test_texts)),
                "train_positive": int(pd_numeric_sum(train, "reader_group_binary")),
                "test_positive": int(pd_numeric_sum(test, "reader_group_binary")),
            }
        )
    return errors, summaries


def _feature_columns(samples: Any) -> dict[str, list[str]]:
    columns = set(samples.columns.astype(str))
    residual_columns = [
        column
        for column in columns
        if column.startswith("crossfit_")
        and ("dfm_surprisal_slope" in column or "dfm_entropy_slope" in column)
    ]
    residual_summary_columns = [
        column
        for column in columns
        if column.startswith("crossfit_") and column.endswith(("_mean", "_median", "_sd"))
    ]
    segmentation_columns = [
        "mean_boundary_opacity_exposure",
        "vv_boundary_exposure_rate",
        *SEGMENTATION_SENSITIVITY_FEATURES,
        *[
            column
            for column in columns
            if column.startswith("crossfit_")
            and ("boundary_opacity" in column or "high_opacity" in column or "vv_cost" in column)
        ],
    ]
    all_allowed = _unique(
        RAW_GAZE_FEATURES
        + SENSITIVITY_FEATURES
        + SEGMENTATION_SENSITIVITY_FEATURES
        + EXPOSURE_FEATURES
        + residual_columns
        + residual_summary_columns
    )
    groups = {
        "reading_speed": READING_SPEED_FEATURES,
        "reading_speed_baseline": READING_SPEED_FEATURES,
        "raw_gaze_baseline": RAW_GAZE_FEATURES,
        "D1_dfm_exposure_only": DFM_EXPOSURE_FEATURES,
        "D2_dfm_sensitivity_only": SENSITIVITY_FEATURES + residual_columns,
        "D3_dfm_residual_gaze_only": residual_columns,
        "D4_dfm_exposure_plus_sensitivity": DFM_EXPOSURE_FEATURES
        + SENSITIVITY_FEATURES
        + residual_columns,
        "no_dfm_baseline": [
            column
            for column in all_allowed
            if "dfm" not in column.lower() and "surprisal" not in column and "entropy" not in column
        ],
        "no_segmentation_baseline": [
            column
            for column in all_allowed
            if column not in segmentation_columns
            and "boundary" not in column
            and "opacity" not in column
            and "vv_" not in column
        ],
        "all_allowed_non_exposure": [column for column in all_allowed if column not in EXPOSURE_FEATURES],
    }
    return {
        name: [column for column in _unique(group) if column in columns and column not in PROHIBITED_FEATURES]
        for name, group in groups.items()
    }


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _aggregate_fold_residual_features(config: dict[str, Any], data: Any, *, prefix: str) -> Any:
    pd = _pd()
    min_words = int(get_nested(config, "benchmark_bridge.residualization.min_words_for_slope", 8))
    rows = [
        _aggregate_residual_sample(group, str(sample_id), min_words, prefix=prefix)
        for sample_id, group in data.groupby("sample_id", sort=True, dropna=False)
    ]
    return pd.DataFrame(rows)


def _build_one_fold_cache_entry(
    config: dict[str, Any],
    samples: Any,
    word: Any,
    fold_item: tuple[int, tuple[tuple[str, int], Any]],
) -> tuple[tuple[str, int] | None, dict[str, Any] | None, dict[str, Any]]:
    fold_index, ((split_name, fold_id), fold) = fold_item
    seed = int(get_nested(config, "benchmark_bridge.deterministic_seed", 97))
    train_ids = set(fold[fold["split_role"].eq("train")]["sample_id"].astype(str))
    test_ids = set(fold[fold["split_role"].eq("test")]["sample_id"].astype(str))
    train_word = word[word["sample_id"].isin(train_ids)].copy()
    test_word = word[word["sample_id"].isin(test_ids)].copy()
    fold_diag = {
        "split_name": split_name,
        "fold_id": int(fold_id),
        "train_word_rows": int(len(train_word)),
        "test_word_rows": int(len(test_word)),
        "train_samples": int(len(train_ids)),
        "test_samples": int(len(test_ids)),
        "heldout_reader_rows_used_for_fit": False,
        "heldout_text_rows_used_for_fit": False,
        "reader_group_used": False,
        "status": "complete",
        "skip_reason": "",
    }
    if train_word.empty or test_word.empty:
        fold_diag["status"] = "skipped"
        fold_diag["skip_reason"] = "empty_word_rows"
        return None, None, fold_diag
    train_participants = set(train_word["participant_id"].astype(str))
    test_participants = set(test_word["participant_id"].astype(str))
    train_texts = set(train_word["text_id"].astype(str))
    test_texts = set(test_word["text_id"].astype(str))
    if split_name in {"unseen_reader", "text_balanced_unseen_reader", "participant_grouped_kfold"}:
        fold_diag["heldout_reader_rows_used_for_fit"] = bool(
            train_participants.intersection(test_participants)
        )
    if split_name in {"unseen_text", "leave_one_speech_out"}:
        fold_diag["heldout_text_rows_used_for_fit"] = bool(train_texts.intersection(test_texts))
    if split_name == "unseen_reader_and_text":
        fold_diag["heldout_reader_rows_used_for_fit"] = bool(
            train_participants.intersection(test_participants)
        )
        fold_diag["heldout_text_rows_used_for_fit"] = bool(train_texts.intersection(test_texts))
    train_resid = train_word.copy()
    test_resid = test_word.copy()
    outcome_diags = []
    for outcome, prefix, binary in RESIDUAL_OUTCOME_SPECS:
        train_residual, test_residual, diag = _fit_apply_residuals_pair(
            train_word,
            train_word,
            test_word,
            outcome,
            binary=binary,
            seed=seed + fold_index,
        )
        train_resid[f"crossfit_{prefix}_residual"] = train_residual
        test_resid[f"crossfit_{prefix}_residual"] = test_residual
        outcome_diags.append(diag)
    train_features = _aggregate_fold_residual_features(config, train_resid, prefix="crossfit")
    test_features = _aggregate_fold_residual_features(config, test_resid, prefix="crossfit")
    train_samples = samples[samples["sample_id"].isin(train_ids)].merge(
        train_features,
        on="sample_id",
        how="left",
    )
    test_samples = samples[samples["sample_id"].isin(test_ids)].merge(
        test_features,
        on="sample_id",
        how="left",
    )
    payload = {
        "train": train_samples,
        "test": test_samples,
        "diagnostics": outcome_diags,
    }
    fold_diag["outcome_complete"] = int(
        sum(1 for row in outcome_diags if row.get("status") == "complete")
    )
    if any(row.get("uses_reader_group") for row in outcome_diags):
        fold_diag["reader_group_used"] = True
    return (str(split_name), int(fold_id)), payload, fold_diag


def build_crossfit_fold_feature_cache(
    config: dict[str, Any],
    out: Path,
    samples: Any,
    word: Any,
    splits: Any,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    pd = _pd()
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    grouped = splits[splits["include_in_fold"].astype(bool)].groupby(["split_name", "fold_id"], dropna=False)
    fold_items = list(enumerate(grouped))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus else 1
    requested_jobs = int(get_nested(config, "benchmark_bridge.residualization.max_parallel_folds", 8))
    n_jobs = max(1, min(requested_jobs, cpus, len(fold_items)))
    if n_jobs > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_build_one_fold_cache_entry)(config, samples, word, item) for item in fold_items
        )
    else:
        results = [_build_one_fold_cache_entry(config, samples, word, item) for item in fold_items]
    diagnostics = []
    for key, payload, fold_diag in results:
        diagnostics.append(fold_diag)
        if key is not None and payload is not None:
            cache[key] = payload
    diag_frame = pd.DataFrame(diagnostics)
    _write_csv(out / "residualization" / "crossfit_residualization_diagnostics.csv", diag_frame)
    report = write_crossfit_residualization_report(out, config, diag_frame)
    return cache, report


def write_crossfit_residualization_report(out: Path, config: dict[str, Any], diagnostics: Any) -> dict[str, Any]:
    pd = _pd()
    by_split = (
        diagnostics.groupby("split_name", dropna=False)
        .agg(
            folds=("fold_id", "count"),
            skipped=("status", lambda s: int((s != "complete").sum())),
            heldout_reader_rows_used_for_fit=("heldout_reader_rows_used_for_fit", "max"),
            heldout_text_rows_used_for_fit=("heldout_text_rows_used_for_fit", "max"),
            reader_group_used=("reader_group_used", "max"),
            train_word_rows=("train_word_rows", "sum"),
            test_word_rows=("test_word_rows", "sum"),
        )
        .reset_index()
        if not diagnostics.empty
        else pd.DataFrame()
    )
    payload = {
        "folds": int(len(diagnostics)),
        "heldout_reader_rows_used_for_fit": bool(
            diagnostics.get("heldout_reader_rows_used_for_fit", pd.Series(dtype=bool)).any()
        )
        if not diagnostics.empty
        else False,
        "heldout_text_rows_used_for_fit": bool(
            diagnostics.get("heldout_text_rows_used_for_fit", pd.Series(dtype=bool)).any()
        )
        if not diagnostics.empty
        else False,
        "reader_group_used": bool(diagnostics.get("reader_group_used", pd.Series(dtype=bool)).any())
        if not diagnostics.empty
        else False,
        "predictors": RESIDUALIZATION_PREDICTORS,
    }
    text = "\n".join(
        [
            "# Crossfit Residualization Report",
            "",
            "Residualizers are fit separately inside each benchmark split/fold using training "
            "word rows only, then applied to both training and test rows before sample-level "
            "aggregation. Reader group, participant ID, speech ID, text ID, and labels are not "
            "residualization predictors.",
            "",
            "## Predictors",
            "\n".join(f"- `{column}`" for column in RESIDUALIZATION_PREDICTORS),
            "",
            "## Fold Diagnostics",
            _markdown_table(
                by_split.to_dict("records") if not by_split.empty else [],
                [
                    "split_name",
                    "folds",
                    "skipped",
                    "heldout_reader_rows_used_for_fit",
                    "heldout_text_rows_used_for_fit",
                    "reader_group_used",
                    "train_word_rows",
                    "test_word_rows",
                ],
                max_rows=80,
            ),
            "",
            "## Validation",
            f"- Held-out reader rows used for residual fitting: "
            f"{payload['heldout_reader_rows_used_for_fit']}",
            f"- Held-out text rows used for residual fitting: "
            f"{payload['heldout_text_rows_used_for_fit']}",
            f"- Reader group used in residualization: {payload['reader_group_used']}",
            "- Target leakage check: primary labels and comprehension targets are not residualizer inputs.",
        ]
    )
    _write_md(out / "residualization" / "crossfit_residualization_report.md", text)
    return payload


def _clean_feature_list(frame: Any, columns: list[str]) -> list[str]:
    pd = _pd()
    clean = []
    for column in columns:
        if column in PROHIBITED_FEATURES or column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            clean.append(column)
    return clean


def _model_pipeline(model_name: str, *, task: str, seed: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if task == "typ":
        if model_name == "logistic_regression":
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(with_mean=False),
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            )
        if model_name == "linear_svm":
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(with_mean=False),
                LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed),
            )
        if model_name == "random_forest":
            return make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
    if task == "rcs":
        if model_name == "ridge_regression":
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        if model_name == "elastic_net":
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=5000),
            )
        if model_name == "random_forest":
            return make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed),
            )
    raise ValueError(f"unsupported model {model_name} for task {task}")


def _score_regressor(estimator: Any, x: Any) -> Any:
    return estimator.predict(x).astype(float)


def _regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    np = _np()
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for RCS benchmark metrics") from exc
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) == 0:
        return {"rmse": None, "mae": None, "r2": None}
    r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) >= 2 else None
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": r2,
    }


def _reader_aggregate_classification(predictions: Any) -> Any:
    if predictions.empty:
        return predictions
    return (
        predictions.groupby(
            ["task", "split_name", "fold_id", "feature_group", "model", "participant_id"],
            dropna=False,
        )
        .agg(
            y_true=("y_true", "first"),
            y_score=("y_score", "mean"),
            n_trial_predictions=("sample_id", "count"),
        )
        .reset_index()
    )


def _reader_aggregate_regression(predictions: Any) -> Any:
    if predictions.empty:
        return predictions
    return (
        predictions.groupby(
            ["task", "split_name", "fold_id", "feature_group", "model", "participant_id"],
            dropna=False,
        )
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            n_trial_predictions=("sample_id", "count"),
        )
        .reset_index()
    )


def evaluate_typ_benchmark(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    fold_cache: dict[tuple[str, int], dict[str, Any]],
) -> tuple[Any, Any]:
    pd = _pd()
    seed = int(get_nested(config, "benchmark_bridge.deterministic_seed", 97))
    feature_groups = get_nested(config, "benchmark_bridge.typ_feature_groups", [])
    if not feature_groups:
        feature_groups = ["chance_majority", PRIMARY_TYPO_FEATURE_GROUP]
    models = ["logistic_regression"]
    all_predictions = []
    metric_rows = []
    for split_name in BENCHMARK_SPLITS:
        fold_keys = [key for key in sorted(fold_cache) if key[0] == split_name]
        for feature_group in feature_groups:
            run_models = ["chance_majority"] if feature_group == "chance_majority" else models
            for model_name in run_models:
                predictions = []
                usable = 0
                skipped = 0
                n_features = 0
                skip_reason = ""
                for _, fold_id in fold_keys:
                    train = fold_cache[(split_name, fold_id)]["train"].copy()
                    test = fold_cache[(split_name, fold_id)]["test"].copy()
                    train_y = pd.to_numeric(train["reader_group_binary"], errors="coerce")
                    test_y = pd.to_numeric(test["reader_group_binary"], errors="coerce")
                    if train.empty or test.empty or train_y.nunique(dropna=True) < 2:
                        skipped += 1
                        skip_reason = "empty_or_single_class_training_fold"
                        continue
                    if model_name == "chance_majority":
                        prevalence = float(train_y.mean())
                        score = [prevalence for _ in range(len(test))]
                        columns: list[str] = []
                    else:
                        groups = _feature_columns(train)
                        columns = _clean_feature_list(train, groups.get(feature_group, []))
                        if not columns:
                            skipped += 1
                            skip_reason = "no_usable_features"
                            continue
                        model = _model_pipeline(model_name, task="typ", seed=seed + int(fold_id))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model.fit(train[columns], train_y.astype(int))
                        score = _score_estimator(model, test[columns])
                    usable += 1
                    n_features = max(n_features, len(columns))
                    for row, truth, pred in zip(test.to_dict("records"), test_y, score, strict=True):
                        predictions.append(
                            {
                                "task": "CopCo_TYP",
                                "split_name": split_name,
                                "fold_id": int(fold_id),
                                "feature_group": feature_group,
                                "model": model_name,
                                "sample_id": row["sample_id"],
                                "participant_id": row["participant_id"],
                                "speech_id": row["speech_id"],
                                "text_id": row["text_id"],
                                "y_true": int(truth),
                                "y_score": float(pred),
                                "y_pred": int(float(pred) >= 0.5),
                            }
                        )
                pred_frame = pd.DataFrame(predictions)
                all_predictions.append(pred_frame)
                for level, frame in [
                    ("participant_text_trial", pred_frame),
                    ("reader_aggregated", _reader_aggregate_classification(pred_frame)),
                ]:
                    if frame.empty:
                        metric = {
                            "n_predictions": 0,
                            "roc_auc": None,
                            "pr_auc": None,
                            "balanced_accuracy": None,
                            "macro_f1": None,
                            "brier_score": None,
                            "status": "skipped",
                            "skip_reason": skip_reason or "no_valid_predictions",
                        }
                    else:
                        metric = {
                            "n_predictions": int(len(frame)),
                            **_classification_metrics(frame["y_true"], frame["y_score"]),
                            "status": "complete",
                            "skip_reason": "",
                        }
                    metric_rows.append(
                        {
                            "task": "CopCo_TYP",
                            "split_name": split_name,
                            "feature_group": feature_group,
                            "model": model_name,
                            "evaluation_level": level,
                            "n_features": int(n_features),
                            "usable_folds": int(usable),
                            "skipped_folds": int(skipped),
                            **metric,
                        }
                    )
    metrics = pd.DataFrame(metric_rows, columns=TYP_METRIC_COLUMNS)
    non_empty_predictions = [frame for frame in all_predictions if not frame.empty]
    predictions = (
        pd.concat(non_empty_predictions, ignore_index=True)
        if non_empty_predictions
        else pd.DataFrame()
    )
    _write_csv(out / "typ" / "typ_benchmark_metrics.csv", metrics)
    _write_csv(out / "typ" / "typ_benchmark_predictions.csv", predictions)
    write_typ_report(dirs, metrics)
    return metrics, predictions


def evaluate_rcs_benchmark(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    samples: Any,
    fold_cache: dict[tuple[str, int], dict[str, Any]],
) -> tuple[Any, Any, dict[str, Any]]:
    pd = _pd()
    target = "comprehension_score"
    target_values = pd.to_numeric(samples.get(target), errors="coerce") if target in samples else pd.Series()
    if target not in samples or target_values.notna().sum() < 10 or target_values.nunique(dropna=True) < 3:
        reason = "comprehension_score target unavailable or insufficiently variable"
        metrics = pd.DataFrame(columns=RCS_METRIC_COLUMNS)
        predictions = pd.DataFrame()
        _write_csv(out / "rcs" / "rcs_benchmark_metrics.csv", metrics)
        _write_csv(out / "rcs" / "rcs_benchmark_predictions.csv", predictions)
        write_rcs_skipped_report(dirs, reason)
        return metrics, predictions, {"status": "skipped", "skip_reason": reason}
    seed = int(get_nested(config, "benchmark_bridge.deterministic_seed", 97))
    feature_groups = get_nested(config, "benchmark_bridge.rcs_feature_groups", [])
    if not feature_groups:
        feature_groups = [PRIMARY_TYPO_FEATURE_GROUP]
    models = ["ridge_regression"]
    splits = ["unseen_reader", "unseen_text", "unseen_reader_and_text"]
    all_predictions = []
    metric_rows = []
    target_scale = "raw_project_scale"
    if float(target_values.min()) >= 0.0 and float(target_values.max()) <= 1.0:
        target_scale = "raw_project_scale_0_1; EyeBench-compatible 1-10 = 1 + 9 * raw"
    for split_name in splits:
        fold_keys = [key for key in sorted(fold_cache) if key[0] == split_name]
        for feature_group in feature_groups:
            for model_name in models:
                predictions = []
                usable = 0
                skipped = 0
                n_features = 0
                skip_reason = ""
                for _, fold_id in fold_keys:
                    train = fold_cache[(split_name, fold_id)]["train"].copy()
                    test = fold_cache[(split_name, fold_id)]["test"].copy()
                    train_y = pd.to_numeric(train[target], errors="coerce")
                    test_y = pd.to_numeric(test[target], errors="coerce")
                    valid_train = train_y.notna()
                    valid_test = test_y.notna()
                    if valid_train.sum() < 3 or valid_test.sum() == 0:
                        skipped += 1
                        skip_reason = "insufficient_target_rows"
                        continue
                    train = train.loc[valid_train].copy()
                    test = test.loc[valid_test].copy()
                    train_y = train_y.loc[valid_train]
                    test_y = test_y.loc[valid_test]
                    groups = _feature_columns(train)
                    columns = _clean_feature_list(train, groups.get(feature_group, []))
                    if not columns:
                        skipped += 1
                        skip_reason = "no_usable_features"
                        continue
                    model = _model_pipeline(model_name, task="rcs", seed=seed + int(fold_id))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(train[columns], train_y.astype(float))
                    pred = _score_regressor(model, test[columns])
                    usable += 1
                    n_features = max(n_features, len(columns))
                    for row, truth, value in zip(test.to_dict("records"), test_y, pred, strict=True):
                        predictions.append(
                            {
                                "task": "CopCo_RCS",
                                "split_name": split_name,
                                "fold_id": int(fold_id),
                                "feature_group": feature_group,
                                "model": model_name,
                                "sample_id": row["sample_id"],
                                "participant_id": row["participant_id"],
                                "speech_id": row["speech_id"],
                                "text_id": row["text_id"],
                                "y_true": float(truth),
                                "y_pred": float(value),
                            }
                        )
                pred_frame = pd.DataFrame(predictions)
                all_predictions.append(pred_frame)
                for level, frame in [
                    ("participant_text_trial", pred_frame),
                    ("reader_aggregated", _reader_aggregate_regression(pred_frame)),
                ]:
                    if frame.empty:
                        metric = {
                            "n_predictions": 0,
                            "rmse": None,
                            "mae": None,
                            "r2": None,
                            "status": "skipped",
                            "skip_reason": skip_reason or "no_valid_predictions",
                        }
                    else:
                        metric = {
                            "n_predictions": int(len(frame)),
                            **_regression_metrics(frame["y_true"], frame["y_pred"]),
                            "status": "complete",
                            "skip_reason": "",
                        }
                    metric_rows.append(
                        {
                            "task": "CopCo_RCS",
                            "split_name": split_name,
                            "feature_group": feature_group,
                            "model": model_name,
                            "evaluation_level": level,
                            "target": target,
                            "target_scale": target_scale,
                            "n_features": int(n_features),
                            "usable_folds": int(usable),
                            "skipped_folds": int(skipped),
                            **metric,
                        }
                    )
    metrics = pd.DataFrame(metric_rows, columns=RCS_METRIC_COLUMNS)
    non_empty_predictions = [frame for frame in all_predictions if not frame.empty]
    predictions = (
        pd.concat(non_empty_predictions, ignore_index=True)
        if non_empty_predictions
        else pd.DataFrame()
    )
    _write_csv(out / "rcs" / "rcs_benchmark_metrics.csv", metrics)
    _write_csv(out / "rcs" / "rcs_benchmark_predictions.csv", predictions)
    write_rcs_report(dirs, metrics, target_scale)
    return metrics, predictions, {"status": "complete", "target_scale": target_scale}


def write_typ_report(dirs: dict[str, Path], metrics: Any) -> None:
    d3 = metrics[
        metrics["feature_group"].eq(PRIMARY_TYPO_FEATURE_GROUP)
        & metrics["model"].eq("logistic_regression")
        & metrics["evaluation_level"].eq("reader_aggregated")
    ]
    text = "\n".join(
        [
            "# CopCo TYP BenchmarkBridge Report",
            "",
            "Primary model: D3 DFM residual gaze-only logistic regression. Metrics are internal "
            "EyeBench-style unless the compatibility report states that official folds were used.",
            "",
            "## Reader-Aggregated D3 Metrics",
            _markdown_table(
                d3[
                    [
                        "split_name",
                        "n_predictions",
                        "roc_auc",
                        "pr_auc",
                        "balanced_accuracy",
                        "macro_f1",
                        "brier_score",
                        "skipped_folds",
                    ]
                ].to_dict("records"),
                [
                    "split_name",
                    "n_predictions",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                    "skipped_folds",
                ],
                max_rows=20,
            ),
        ]
    )
    _write_report(dirs, "copco_typ_benchmark_report.md", text)


def write_rcs_report(dirs: dict[str, Path], metrics: Any, target_scale: str) -> None:
    d3 = metrics[
        metrics["feature_group"].eq(PRIMARY_TYPO_FEATURE_GROUP)
        & metrics["model"].eq("ridge_regression")
        & metrics["evaluation_level"].eq("reader_aggregated")
    ]
    text = "\n".join(
        [
            "# CopCo RCS BenchmarkBridge Report",
            "",
            f"Target scale: {target_scale}. RCS is auxiliary and does not affect the main TYP claim.",
            "",
            "## Reader-Aggregated D3 Metrics",
            _markdown_table(
                d3[
                    [
                        "split_name",
                        "n_predictions",
                        "rmse",
                        "mae",
                        "r2",
                        "skipped_folds",
                    ]
                ].to_dict("records"),
                ["split_name", "n_predictions", "rmse", "mae", "r2", "skipped_folds"],
                max_rows=20,
            ),
        ]
    )
    _write_report(dirs, "copco_rcs_benchmark_report.md", text)


def write_rcs_skipped_report(dirs: dict[str, Path], reason: str) -> None:
    _write_report(
        dirs,
        "rcs_skipped_report.md",
        "\n".join(["# CopCo RCS Skipped Report", "", f"- Reason: {reason}"]),
    )


def write_eyebench_compatibility_report(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    del config
    candidates = []
    import_available = False
    import_error = ""
    try:
        __import__("eyebench")
        import_available = True
    except Exception as exc:
        import_error = str(exc)
    for name in ["EyeBench", "eyebench", "eye-bench"]:
        path = Path(repo_root) / name
        if path.exists():
            candidates.append(str(path))
    cli_available = shutil.which("eyebench") is not None
    official_mode_run = False
    exact_folds = False
    blocker = ""
    if not import_available and not candidates and not cli_available:
        blocker = "EyeBench package/repository/CLI not available in the local CopCo workspace."
    else:
        blocker = (
            "EyeBench artefacts appear locally available, but no exact CopCo_TYP/CopCo_RCS fold "
            "adapter is implemented in BenchmarkBridge v1."
        )
    report = {
        "official_mode_run": official_mode_run,
        "exact_eyebench_folds_used": exact_folds,
        "internal_matched_split_used": True,
        "can_be_called_official_benchmark_results": False,
        "should_be_called_benchmark_relative_only": True,
        "eyebench_import_available": import_available,
        "eyebench_import_error": import_error,
        "eyebench_cli_available": cli_available,
        "local_candidates": candidates,
        "blocker": blocker,
    }
    text = "\n".join(
        [
            "# EyeBench Compatibility Report",
            "",
            f"- official_mode_run: {report['official_mode_run']}",
            f"- exact EyeBench folds used: {report['exact_eyebench_folds_used']}",
            f"- internal matched split used: {report['internal_matched_split_used']}",
            f"- can results be called official benchmark results: "
            f"{report['can_be_called_official_benchmark_results']}",
            f"- should results be called benchmark-relative only: "
            f"{report['should_be_called_benchmark_relative_only']}",
            f"- EyeBench import available: {report['eyebench_import_available']}",
            f"- EyeBench CLI available: {report['eyebench_cli_available']}",
            f"- local candidates: {', '.join(candidates) if candidates else 'none'}",
            "",
            "## Blocker",
            blocker,
        ]
    )
    _write_report(dirs, "eyebench_compatibility_report.md", text)
    _write_json(out / "eyebench_compatibility_report.json", report)
    return report


def _metric_value(metrics: Any, split_name: str, metric: str, *, task: str) -> Any:
    model = "logistic_regression" if task == "typ" else "ridge_regression"
    row = metrics[
        metrics["split_name"].eq(split_name)
        & metrics["feature_group"].eq(PRIMARY_TYPO_FEATURE_GROUP)
        & metrics["model"].eq(model)
        & metrics["evaluation_level"].eq("reader_aggregated")
    ]
    if row.empty or metric not in row:
        return None
    value = row.iloc[0][metric]
    return float(value) if value == value else None


def _average(values: list[Any]) -> Any:
    valid = [float(value) for value in values if value is not None and value == value]
    return float(sum(valid) / len(valid)) if valid else None


def write_typ_comparison_tables(
    config: dict[str, Any],
    dirs: dict[str, Path],
    metrics: Any,
    official: dict[str, Any],
) -> Any:
    pd = _pd()
    gate = get_nested(config, "benchmark_bridge.decision_gates.CopCo_TYP", {})
    ours = {
        "model": "D3_dfm_residual_gaze_only",
        "unseen_reader_balanced_accuracy": _metric_value(metrics, "unseen_reader", "balanced_accuracy", task="typ"),
        "unseen_text_balanced_accuracy": _metric_value(metrics, "unseen_text", "balanced_accuracy", task="typ"),
        "unseen_reader_text_balanced_accuracy": _metric_value(
            metrics, "unseen_reader_and_text", "balanced_accuracy", task="typ"
        ),
        "unseen_reader_AUROC": _metric_value(metrics, "unseen_reader", "roc_auc", task="typ"),
        "unseen_text_AUROC": _metric_value(metrics, "unseen_text", "roc_auc", task="typ"),
        "unseen_reader_text_AUROC": _metric_value(metrics, "unseen_reader_and_text", "roc_auc", task="typ"),
        "evaluation_level": "reader_aggregated",
        "official_mode": bool(official.get("official_mode_run")),
        "notes": "BenchmarkBridge internal EyeBench-style split.",
    }
    ours["average_balanced_accuracy"] = _average(
        [
            ours["unseen_reader_balanced_accuracy"],
            ours["unseen_text_balanced_accuracy"],
            ours["unseen_reader_text_balanced_accuracy"],
        ]
    )
    ours["average_AUROC"] = _average(
        [ours["unseen_reader_AUROC"], ours["unseen_text_AUROC"], ours["unseen_reader_text_AUROC"]]
    )
    references = []
    for model in [
        "Chance",
        "Reading Speed",
        "Logistic Regression",
        "SVM",
        "Random Forest",
        "AhnCNN",
        "BEyeLSTM",
        "RoBERTEye-W",
        "best_reported_baseline",
    ]:
        row = {
            "model": model,
            "unseen_reader_balanced_accuracy": None,
            "unseen_text_balanced_accuracy": None,
            "unseen_reader_text_balanced_accuracy": None,
            "average_balanced_accuracy": None,
            "unseen_reader_AUROC": None,
            "unseen_text_AUROC": None,
            "unseen_reader_text_AUROC": None,
            "average_AUROC": None,
            "evaluation_level": "EyeBench reported central value",
            "official_mode": True,
            "notes": "Central value not present in frozen BenchmarkBridge prompt/config.",
        }
        if model in {"AhnCNN", "best_reported_baseline"}:
            row["unseen_reader_balanced_accuracy"] = gate.get("unseen_reader", {}).get(
                "test_balanced_accuracy"
            )
            row["unseen_reader_AUROC"] = gate.get("unseen_reader", {}).get("test_AUROC")
            row["unseen_reader_text_balanced_accuracy"] = gate.get("unseen_reader_and_text", {}).get(
                "test_balanced_accuracy"
            )
            row["unseen_reader_text_AUROC"] = gate.get("unseen_reader_and_text", {}).get("test_AUROC")
            row["average_balanced_accuracy"] = _average(
                [row["unseen_reader_balanced_accuracy"], row["unseen_reader_text_balanced_accuracy"]]
            )
            row["average_AUROC"] = _average([row["unseen_reader_AUROC"], row["unseen_reader_text_AUROC"]])
            row["notes"] = "Gate central values supplied in BenchmarkBridge v1 request."
        if model == "Chance":
            for column in [
                "unseen_reader_balanced_accuracy",
                "unseen_text_balanced_accuracy",
                "unseen_reader_text_balanced_accuracy",
                "average_balanced_accuracy",
                "unseen_reader_AUROC",
                "unseen_text_AUROC",
                "unseen_reader_text_AUROC",
                "average_AUROC",
            ]:
                row[column] = 0.5
            row["notes"] = "Analytic chance reference."
        references.append(row)
    table = pd.DataFrame([ours, *references])[
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
        ]
    ]
    _write_analysis_table(dirs, "copco_typ_benchmark_comparison.csv", table)
    md = _markdown_table(table.to_dict("records"), table.columns.tolist(), max_rows=20)
    tex = table.to_latex(index=False, float_format=lambda x: f"{x:.3f}" if x == x else "")
    _write_md(dirs["result_tables"] / "copco_typ_benchmark_comparison.md", md)
    _write_md(dirs["repo_tables"] / "copco_typ_benchmark_comparison.md", md)
    _write_md(dirs["result_tables"] / "copco_typ_benchmark_comparison.tex", tex)
    _write_md(dirs["repo_tables"] / "copco_typ_benchmark_comparison.tex", tex)
    return table


def write_rcs_comparison_tables(
    config: dict[str, Any],
    dirs: dict[str, Path],
    metrics: Any,
    official: dict[str, Any],
) -> Any:
    pd = _pd()
    gate = get_nested(config, "benchmark_bridge.decision_gates.CopCo_RCS", {})
    ours = {
        "model": "D3_dfm_residual_gaze_only",
        "unseen_reader_RMSE": _metric_value(metrics, "unseen_reader", "rmse", task="rcs"),
        "unseen_text_RMSE": _metric_value(metrics, "unseen_text", "rmse", task="rcs"),
        "unseen_reader_text_RMSE": _metric_value(metrics, "unseen_reader_and_text", "rmse", task="rcs"),
        "unseen_reader_MAE": _metric_value(metrics, "unseen_reader", "mae", task="rcs"),
        "unseen_text_MAE": _metric_value(metrics, "unseen_text", "mae", task="rcs"),
        "unseen_reader_text_MAE": _metric_value(metrics, "unseen_reader_and_text", "mae", task="rcs"),
        "unseen_reader_R2": _metric_value(metrics, "unseen_reader", "r2", task="rcs"),
        "unseen_text_R2": _metric_value(metrics, "unseen_text", "r2", task="rcs"),
        "unseen_reader_text_R2": _metric_value(metrics, "unseen_reader_and_text", "r2", task="rcs"),
        "evaluation_level": "reader_aggregated",
        "official_mode": bool(official.get("official_mode_run")),
        "notes": "BenchmarkBridge auxiliary RCS run.",
    }
    ours["average_RMSE"] = _average(
        [ours["unseen_reader_RMSE"], ours["unseen_text_RMSE"], ours["unseen_reader_text_RMSE"]]
    )
    ours["average_MAE"] = _average(
        [ours["unseen_reader_MAE"], ours["unseen_text_MAE"], ours["unseen_reader_text_MAE"]]
    )
    ours["average_R2"] = _average(
        [ours["unseen_reader_R2"], ours["unseen_text_R2"], ours["unseen_reader_text_R2"]]
    )
    rf = {
        "model": "Random Forest",
        "unseen_reader_RMSE": None,
        "unseen_text_RMSE": None,
        "unseen_reader_text_RMSE": None,
        "average_RMSE": None,
        "unseen_reader_MAE": None,
        "unseen_text_MAE": None,
        "unseen_reader_text_MAE": None,
        "average_MAE": None,
        "unseen_reader_R2": gate.get("validation_R2_approx"),
        "unseen_text_R2": None,
        "unseen_reader_text_R2": None,
        "average_R2": gate.get("average_test_R2_approx"),
        "evaluation_level": "EyeBench reported central value",
        "official_mode": True,
        "notes": "Approximate R2 gate supplied in BenchmarkBridge v1 request.",
    }
    table = pd.DataFrame([ours, rf])[
        [
            "model",
            "unseen_reader_RMSE",
            "unseen_text_RMSE",
            "unseen_reader_text_RMSE",
            "average_RMSE",
            "unseen_reader_MAE",
            "unseen_text_MAE",
            "unseen_reader_text_MAE",
            "average_MAE",
            "unseen_reader_R2",
            "unseen_text_R2",
            "unseen_reader_text_R2",
            "average_R2",
            "evaluation_level",
            "official_mode",
            "notes",
        ]
    ]
    _write_analysis_table(dirs, "copco_rcs_benchmark_comparison.csv", table)
    md = _markdown_table(table.to_dict("records"), table.columns.tolist(), max_rows=20)
    tex = table.to_latex(index=False, float_format=lambda x: f"{x:.3f}" if x == x else "")
    _write_md(dirs["result_tables"] / "copco_rcs_benchmark_comparison.md", md)
    _write_md(dirs["repo_tables"] / "copco_rcs_benchmark_comparison.md", md)
    _write_md(dirs["result_tables"] / "copco_rcs_benchmark_comparison.tex", tex)
    _write_md(dirs["repo_tables"] / "copco_rcs_benchmark_comparison.tex", tex)
    return table


def write_decision_report(
    config: dict[str, Any],
    dirs: dict[str, Path],
    typ_metrics: Any,
    rcs_metrics: Any,
    official: dict[str, Any],
) -> dict[str, Any]:
    gate = get_nested(config, "benchmark_bridge.decision_gates.CopCo_TYP", {})
    d3_reader_auc = _metric_value(typ_metrics, "unseen_reader", "roc_auc", task="typ")
    d3_reader_ba = _metric_value(typ_metrics, "unseen_reader", "balanced_accuracy", task="typ")
    d3_strict_auc = _metric_value(typ_metrics, "unseen_reader_and_text", "roc_auc", task="typ")
    d3_strict_ba = _metric_value(
        typ_metrics,
        "unseen_reader_and_text",
        "balanced_accuracy",
        task="typ",
    )
    reader_gate = gate.get("unseen_reader", {})
    strict_gate = gate.get("unseen_reader_and_text", {})
    beats_reader = bool(
        d3_reader_auc is not None
        and d3_reader_ba is not None
        and d3_reader_auc > float(reader_gate.get("test_AUROC", 1.0))
        and d3_reader_ba > float(reader_gate.get("test_balanced_accuracy", 1.0))
    )
    beats_strict = bool(
        d3_strict_auc is not None
        and d3_strict_ba is not None
        and d3_strict_auc > float(strict_gate.get("test_AUROC", 1.0))
        and d3_strict_ba > float(strict_gate.get("test_balanced_accuracy", 1.0))
    )
    official_mode = bool(official.get("official_mode_run"))
    if beats_reader and beats_strict:
        category = "main_paper_comparison"
    elif beats_reader:
        category = "appendix_comparison"
    elif not official_mode:
        category = "internal_only"
    else:
        category = "not_supported"
    rcs_d3 = rcs_metrics[
        rcs_metrics["feature_group"].eq(PRIMARY_TYPO_FEATURE_GROUP)
        & rcs_metrics["model"].eq("ridge_regression")
        & rcs_metrics["evaluation_level"].eq("reader_aggregated")
    ]
    rcs_best = None
    if not rcs_d3.empty and "r2" in rcs_d3:
        valid = rcs_d3["r2"].dropna()
        rcs_best = float(valid.max()) if not valid.empty else None
    rcs_signal = bool(rcs_best is not None and rcs_best > 0.08)
    paper_text = (
        "In an internal EyeBench-style bridge, the frozen D3 DFM residual gaze-profile "
        "model was re-evaluated under unseen-reader, unseen-text, and strict unseen-reader-plus-text "
        "splits. Because exact official EyeBench folds were not available, these results are "
        "benchmark-relative and should be reported as supplementary validation rather than official "
        "leaderboard scores."
    )
    decision = {
        "decision_category": category,
        "beats_CopCo_TYP_unseen_reader_baseline": beats_reader,
        "beats_CopCo_TYP_unseen_reader_text_baseline": beats_strict,
        "official_eyebench_compatible": official_mode,
        "main_manuscript_update_supported": category == "main_paper_comparison",
        "appendix_only_supported": category in {"appendix_comparison", "internal_only"},
        "rcs_useful_signal": rcs_signal,
        "rcs_best_reader_aggregated_r2": rcs_best,
        "rcs_affects_main_story": False,
        "changes_title_abstract_claims": False,
        "paper_text": paper_text,
    }
    rows = [
        {
            "question": "Does D3 beat the strongest listed CopCo TYP Unseen Reader baseline?",
            "answer": beats_reader,
            "evidence": f"D3 AUROC={_format_value(d3_reader_auc)}, BA={_format_value(d3_reader_ba)}",
        },
        {
            "question": "Does D3 beat the strongest listed CopCo TYP Unseen Reader + Text baseline?",
            "answer": beats_strict,
            "evidence": f"D3 AUROC={_format_value(d3_strict_auc)}, BA={_format_value(d3_strict_ba)}",
        },
        {
            "question": "Is the result official EyeBench-compatible or internal EyeBench-style?",
            "answer": "official" if official_mode else "internal EyeBench-style",
            "evidence": str(official.get("blocker", "")),
        },
        {
            "question": "Should the benchmark comparison enter the main manuscript?",
            "answer": decision["main_manuscript_update_supported"],
            "evidence": category,
        },
        {
            "question": "Should it remain appendix-only?",
            "answer": decision["appendix_only_supported"],
            "evidence": category,
        },
        {
            "question": "Does RCS show useful signal?",
            "answer": rcs_signal,
            "evidence": f"best R2={_format_value(rcs_best)}",
        },
        {
            "question": "Does RCS affect the main paper story?",
            "answer": False,
            "evidence": "RCS is auxiliary and not part of the frozen TYP main claim.",
        },
        {
            "question": "Does the bridge change the final title, abstract, or claims?",
            "answer": False,
            "evidence": "Exact official EyeBench mode was not run.",
        },
    ]
    text = "\n".join(
        [
            "# BenchmarkBridge Decision Report",
            "",
            f"- Decision category: `{category}`",
            f"- Official EyeBench-compatible: {official_mode}",
            "",
            "## Questions",
            _markdown_table(rows, ["question", "answer", "evidence"], max_rows=20),
            "",
            "## Exact Paper Text",
            paper_text,
        ]
    )
    _write_report(dirs, "benchmark_bridge_decision_report.md", text)
    return decision


def write_analysis_plan(repo_root: str | Path) -> None:
    path = Path(repo_root).resolve() / "docs" / "benchmark_bridge_v1_analysis_plan.md"
    text = """# BenchmarkBridge v1 Analysis Plan

BenchmarkBridge v1 evaluates the frozen `D3_dfm_residual_gaze_only` model under
EyeBench-style internal split regimes. It is not a feature-engineering search, label
expansion, or neural baseline rerun.

Primary task: CopCo TYP classification. Auxiliary task: CopCo RCS regression if the
frozen comprehension target is available and sufficiently variable.

Split regimes:

- `unseen_reader`: test participants are disjoint from train participants.
- `unseen_text`: test speeches/texts are disjoint from train speeches/texts.
- `unseen_reader_and_text`: test participants and test texts are both disjoint from
  the training set.
- `text_balanced_unseen_reader`: participant-disjoint internal fold with deterministic
  text-exposure balancing by fold assignment.
- `leave_one_speech_out`: one speech/text held out.
- `participant_grouped_kfold`: deterministic participant-grouped k-fold.

Residualization is fit within each split/fold on training word rows only. Reader group,
participant ID, speech ID, text ID, labels, and targets are not residualizer predictors.
Primary models never use participant or speech identifiers as predictors.

Decision gates compare D3 against the request-specified CopCo TYP AhnCNN central values
for Unseen Reader and Unseen Reader + Text. Exact official EyeBench integration is
attempted but not required; if unavailable, results are reported as internal
EyeBench-style and benchmark-relative only.
"""
    _write_md(path, text)


def run_benchmark_bridge(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    config_check = validate_benchmark_config(config)
    if config_check["status"] != "passed":
        raise ValueError(f"benchmark bridge config failed validation: {config_check['errors']}")
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=repo_root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, repo_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    write_analysis_plan(repo_root)
    inputs = _load_inputs(config, repo_root)
    samples, word = build_participant_text_trial_table(config, out, dirs, inputs)
    splits = build_benchmark_splits(config, out, samples)
    split_errors, split_summary = validate_split_labels(splits)
    _write_analysis_csv(dirs, "split_diagnostics.csv", _pd().DataFrame(split_summary))
    if split_errors:
        _write_json(out / "benchmark_bridge_split_validation_report.json", {"errors": split_errors})
        raise ValueError(f"benchmark split validation failed: {split_errors}")
    fold_cache, residualization = build_crossfit_fold_feature_cache(config, out, samples, word, splits)
    typ_metrics, typ_predictions = evaluate_typ_benchmark(config, out, dirs, fold_cache)
    rcs_metrics, rcs_predictions, rcs_status = evaluate_rcs_benchmark(
        config,
        out,
        dirs,
        samples,
        fold_cache,
    )
    official = write_eyebench_compatibility_report(config, out, dirs, repo_root)
    typ_comparison = write_typ_comparison_tables(config, dirs, typ_metrics, official)
    rcs_comparison = write_rcs_comparison_tables(config, dirs, rcs_metrics, official)
    decision = write_decision_report(config, dirs, typ_metrics, rcs_metrics, official)
    manifest = {
        "run_type": "benchmark_bridge_v1",
        "status": "complete",
        "git_sha": _git_sha(repo_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_dir": str(out),
        "config_validation": config_check,
        "frozen_inputs": get_nested(config, "benchmark_bridge.frozen_inputs", {}),
        "row_counts": {
            "word_level": int(len(inputs["word"])),
            "participant_text_trial_samples": int(len(samples)),
            "split_label_rows": int(len(splits)),
            "typ_predictions": int(len(typ_predictions)),
            "rcs_predictions": int(len(rcs_predictions)),
        },
        "split_regimes_completed": sorted(splits["split_name"].unique().tolist()),
        "residualization": residualization,
        "official_eyebench_compatibility": official,
        "rcs_status": rcs_status,
        "decision": decision,
        "large_outputs_not_for_commit": [
            "data/participant_text_trial_features.parquet",
            "splits/benchmark_split_labels.parquet",
            "typ/typ_benchmark_predictions.csv",
            "rcs/rcs_benchmark_predictions.csv",
            "results/benchmark_bridge_v1_*/",
        ],
        "small_analysis_tables": {
            "typ_comparison_rows": int(len(typ_comparison)),
            "rcs_comparison_rows": int(len(rcs_comparison)),
        },
    }
    _write_json(out / "manifest.json", manifest)
    validation = validate_benchmark_bridge(config, out, repo_root=repo_root)
    _write_json(out / "benchmark_bridge_validation_report.json", validation)
    return manifest


def validate_benchmark_bridge(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    pd = _pd()
    out = Path(output_dir).resolve()
    dirs = _analysis_dirs(config, out, repo_root)
    errors: list[str] = []
    warnings_list: list[str] = []
    config_check = validate_benchmark_config(config)
    errors.extend(config_check["errors"])
    warnings_list.extend(config_check["warnings"])
    required_paths = [
        out / "data" / "participant_text_trial_features.parquet",
        out / "splits" / "benchmark_split_labels.parquet",
        out / "residualization" / "crossfit_residualization_report.md",
        out / "typ" / "typ_benchmark_metrics.csv",
        out / "typ" / "typ_benchmark_predictions.csv",
        out / "rcs" / "rcs_benchmark_metrics.csv",
        out / "rcs" / "rcs_benchmark_predictions.csv",
        dirs["result_analysis"] / "copco_typ_benchmark_report.md",
        dirs["result_analysis"] / "eyebench_compatibility_report.md",
        dirs["result_tables"] / "copco_typ_benchmark_comparison.csv",
        dirs["result_tables"] / "copco_rcs_benchmark_comparison.csv",
        dirs["result_analysis"] / "benchmark_bridge_decision_report.md",
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"missing required artifact: {path}")
    if (out / "typ" / "typ_benchmark_metrics.csv").exists():
        metrics = pd.read_csv(out / "typ" / "typ_benchmark_metrics.csv")
        missing = sorted(set(TYP_METRIC_COLUMNS) - set(metrics.columns))
        if missing:
            errors.append(f"TYP metrics missing columns: {missing}")
        d3 = metrics[
            metrics["feature_group"].eq(PRIMARY_TYPO_FEATURE_GROUP)
            & metrics["model"].eq("logistic_regression")
            & metrics["evaluation_level"].eq("reader_aggregated")
        ]
        missing_splits = sorted(set(["unseen_reader", "unseen_text", "unseen_reader_and_text"]) - set(d3["split_name"]))
        if missing_splits:
            errors.append(f"D3 reader-aggregated TYP metrics missing splits: {missing_splits}")
    if (out / "rcs" / "rcs_benchmark_metrics.csv").exists():
        rcs = pd.read_csv(out / "rcs" / "rcs_benchmark_metrics.csv")
        missing = sorted(set(RCS_METRIC_COLUMNS) - set(rcs.columns))
        if missing:
            errors.append(f"RCS metrics missing columns: {missing}")
    if (out / "splits" / "benchmark_split_labels.parquet").exists():
        split_errors, _ = validate_split_labels(pd.read_parquet(out / "splits" / "benchmark_split_labels.parquet"))
        errors.extend(split_errors)
    if (out / "data" / "participant_text_trial_features.parquet").exists():
        samples = pd.read_parquet(out / "data" / "participant_text_trial_features.parquet")
        groups = _feature_columns(samples.assign(**{c: 0.0 for c in samples.columns if c.startswith("global_")}))
        for name, columns in groups.items():
            bad = sorted(set(columns).intersection(PROHIBITED_FEATURES))
            if bad:
                errors.append(f"feature group {name} includes prohibited predictors: {bad}")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
        "output_dir": str(out),
    }
    return report
