"""Phase 4 confirmatory sensitivity and residualized gaze-cost analyses."""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir
from .research_exploration import (
    PARTICIPANT_METRIC_COLUMNS,
    PRIMARY_EXPOSURE_COUNT_FEATURES,
    _classification_metrics,
    _format_value,
    _markdown_table,
    _merge_boundary_vocoid,
    _np,
    _pd,
    _score_estimator,
    _with_derived_columns,
)


PHASE4_RESIDUALIZATION_PREDICTORS = [
    "word_length_chars",
    "log_corpus_frequency",
    "dfm_lm_word_surprisal",
    "dfm_lm_word_entropy",
    "sentence_length_words",
    "word_position_in_sentence_norm",
    "prev_boundary_opacity_score",
    "vocoid_run_cross_boundary",
    "vv_indicator",
    "lm_missing",
    "embedding_missing",
    "parser_missing",
    "segmentation_label_missing",
    "speech_id",
]

RESIDUALIZATION_FORBIDDEN_COLUMNS = {
    "reader_group",
    "reader_group_binary",
    "reader_group_binary_num",
    "dyslexia_labeled",
    "group_label",
    "participant_id",
}

PHASE4_METRIC_COLUMNS = [
    *PARTICIPANT_METRIC_COLUMNS[:13],
    "calibration_intercept",
    "calibration_slope",
    *PARTICIPANT_METRIC_COLUMNS[13:],
    "fold_validity",
]

RAW_GAZE_FEATURES = [
    "mean_ffd",
    "median_ffd",
    "mean_gd",
    "median_gd",
    "mean_trt",
    "median_trt",
    "skipping_rate",
    "refixation_rate",
    "mean_go_past_time",
    "trt_sd",
    "trt_q90",
]

GLOBAL_SPEED_FEATURES = {
    "mean_ffd",
    "median_ffd",
    "mean_gd",
    "median_gd",
    "mean_trt",
    "median_trt",
    "mean_go_past_time",
    "trt_sd",
    "trt_q90",
}

EXPOSURE_ONLY_FEATURES = {
    "mean_word_length_exposure",
    "mean_log_frequency_exposure",
    "mean_sentence_length_exposure",
    "mean_dfm_surprisal_exposure",
    "mean_dfm_entropy_exposure",
    "mean_boundary_opacity_exposure",
    "vv_boundary_exposure_rate",
    "lm_missing_rate_phase3",
    "embedding_missing_rate_phase3",
}

LABEL_AND_ID_COLUMNS = {
    "participant_id",
    "reader_group",
    "reader_group_binary",
    "dyslexia_labeled",
    "group_label",
}

PRIMARY_EXCLUDED_FEATURES = PRIMARY_EXPOSURE_COUNT_FEATURES | LABEL_AND_ID_COLUMNS

RESIDUAL_OUTCOME_SPECS = [
    ("log_ffd", "ffd", False),
    ("log_first_pass_duration", "first_pass", False),
    ("log_go_past_time", "go_past", False),
    ("log_total_fixation_duration", "total_fixation", False),
    ("skip", "skipping", True),
    ("fixation_count", "fixation_count", False),
]

INTERACTION_TERMS = {
    "reader_group_x_word_length": "reader_group_binary_num:word_length_chars_z",
    "reader_group_x_dfm_surprisal": "reader_group_binary_num:dfm_lm_word_surprisal_z",
    "reader_group_x_previous_boundary_opacity": (
        "reader_group_binary_num:prev_boundary_opacity_score_z"
    ),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["result_analysis"] / name, text)
    _write_md(dirs["repo_analysis"] / name, text)


def _write_csv(dirs: dict[str, Path], name: str, frame: Any) -> None:
    for root in [dirs["result_analysis"], dirs["repo_analysis"]]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)


def _write_parquet(dirs: dict[str, Path], name: str, frame: Any) -> None:
    for root in [dirs["result_analysis"], dirs["repo_analysis"]]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def _git_sha(repo_root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    analysis_rel = str(
        get_nested(config, "phase4_confirmatory.output_layout.analysis", "analysis/phase4_confirmatory")
    )
    repo_analysis = root / str(
        get_nested(config, "phase4_confirmatory.repo_analysis_dir", "analysis/phase4_confirmatory")
    )
    return {
        "repo_analysis": repo_analysis,
        "result_analysis": out / analysis_rel,
    }


def _configured_path(config: dict[str, Any], dotted: str, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    value = get_nested(config, dotted)
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _prepared_dir(config: dict[str, Any], repo_root: str | Path) -> Path:
    return _configured_path(config, "phase4_confirmatory.prepared_dataset_dir", repo_root)


def _label_release_dir(config: dict[str, Any], repo_root: str | Path) -> Path:
    return _configured_path(config, "phase4_confirmatory.label_release_dir", repo_root)


def _phase3_analysis_dir(config: dict[str, Any], repo_root: str | Path) -> Path:
    return _configured_path(config, "phase4_confirmatory.phase3_analysis_dir", repo_root)


def _load_inputs(config: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    pd = _pd()
    label_dir = _label_release_dir(config, repo_root)
    prepared = _prepared_dir(config, repo_root)
    phase3_analysis = _phase3_analysis_dir(config, repo_root)
    return {
        "label_dir": label_dir,
        "prepared_dir": prepared,
        "phase3_analysis_dir": phase3_analysis,
        "word": pd.read_parquet(prepared / "analysis_ready_word_level_v1_1.parquet"),
        "participant": pd.read_parquet(prepared / "analysis_ready_participant_level_v1_1.parquet"),
        "participant_labels": pd.read_parquet(label_dir / "labels" / "participant_labels_v1.parquet"),
        "splits": pd.read_parquet(label_dir / "labels" / "split_labels_v1.parquet"),
        "segmentation_boundary": pd.read_parquet(
            label_dir / "labels" / "segmentation_boundary_labels_v1.parquet"
        ),
        "phase3_profiles": pd.read_parquet(phase3_analysis / "participant_sensitivity_profiles.parquet"),
    }


def validate_phase4_preflight(
    config: dict[str, Any], inputs: dict[str, Any] | None = None, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    data = inputs or _load_inputs(config, repo_root)
    errors: list[str] = []
    warnings_list: list[str] = []
    word = data["word"]
    participant = data["participant"]
    participant_labels = data["participant_labels"]
    splits = data["splits"]
    phase3_profiles = data["phase3_profiles"]
    expected = get_nested(config, "phase4_confirmatory.expected_row_counts", {})
    checks = {
        "word_level": int(len(word)),
        "participant_level": int(len(participant)),
        "participant_labels": int(len(participant_labels)),
        "participant_sensitivity_profiles_phase3": int(len(phase3_profiles)),
    }
    for key, actual in checks.items():
        expected_value = expected.get(key) if isinstance(expected, dict) else None
        if expected_value is not None and int(actual) != int(expected_value):
            errors.append(f"{key} row count {actual} != expected {expected_value}")
    if "participant_word_key" not in word.columns:
        errors.append("prepared word table missing participant_word_key")
    elif word["participant_word_key"].duplicated().any():
        errors.append("prepared word table has duplicate participant_word_key")
    required_dfm = {
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "dfm_lm_alignment_status",
        "dfm_lm_alignment_warning",
    }
    missing_dfm = sorted(required_dfm - set(word.columns))
    if missing_dfm:
        errors.append(f"prepared word table missing DFM columns: {missing_dfm}")
    if word.get("reader_group") is None or word["reader_group"].isna().any():
        errors.append("prepared word table has incomplete participant labels")
    if participant_labels["participant_id"].nunique() != participant["participant_id"].nunique():
        errors.append("participant label coverage does not match prepared participant table")
    if set(participant["participant_id"].astype(str)) != set(phase3_profiles["participant_id"].astype(str)):
        errors.append("Phase 3 participant sensitivity profiles do not cover all participants")
    required_phase3_residuals = {
        "trt_residual_mean",
        "surprisal_sensitivity_phase3",
        "entropy_sensitivity_phase3",
        "high_opacity_trt_residual_cost",
    }
    missing_phase3 = sorted(required_phase3_residuals - set(phase3_profiles.columns))
    if missing_phase3:
        errors.append(f"Phase 3 residualization outputs missing: {missing_phase3}")
    legal = set(
        get_nested(
            config,
            "phase4_confirmatory.leakage_policy.legal_split_names",
            [
                "leave_one_participant_out",
                "participant_grouped_kfold",
                "sensitivity_exclude_uncertain_labels",
                "text_balanced_sensitivity_lopo",
            ],
        )
    )
    split_names = set(splits["split_name"].dropna().astype(str).unique())
    illegal = sorted(split_names - legal)
    if illegal:
        errors.append(f"illegal split names present: {illegal}")
    if splits["split_name"].astype(str).str.contains("random", case=False, na=False).any():
        errors.append("random split label found")
    for (split_name, fold_id), fold in splits.groupby(["split_name", "fold_id"], dropna=False):
        train_ids = set(fold[fold["split_role"].eq("train")]["participant_id"].astype(str))
        test_ids = set(fold[fold["split_role"].eq("test")]["participant_id"].astype(str))
        if train_ids.intersection(test_ids):
            errors.append(f"participant train/test overlap in {split_name} fold {fold_id}")
    if "leave_one_participant_out" in split_names:
        lopo = splits[splits["split_name"].eq("leave_one_participant_out")]
        tests_per_fold = lopo[lopo["split_role"].eq("test")].groupby("fold_id").size()
        if not tests_per_fold.eq(1).all():
            errors.append("LOPO split does not have exactly one test participant in every fold")
        test_once = lopo[lopo["split_role"].eq("test")]["participant_id"].astype(str).value_counts()
        if not test_once.eq(1).all() or len(test_once) != participant["participant_id"].nunique():
            errors.append("LOPO split does not test every participant exactly once")
    excluded = set(
        get_nested(
            config,
            "phase4_confirmatory.leakage_policy.excluded_primary_prediction_features",
            sorted(PRIMARY_EXCLUDED_FEATURES),
        )
    )
    missing_excluded = sorted(PRIMARY_EXPOSURE_COUNT_FEATURES - excluded)
    if missing_excluded:
        errors.append(f"exposure-count variables not flagged for exclusion: {missing_excluded}")
    if RESIDUALIZATION_FORBIDDEN_COLUMNS.intersection(PHASE4_RESIDUALIZATION_PREDICTORS):
        errors.append("reader/group/target columns are listed as residualization predictors")
    parser_expected = str(
        get_nested(config, "phase4_confirmatory.parser_status_expected", "surface_heuristic_fallback")
    )
    if "parser_status" in word.columns:
        parser_values = set(word["parser_status"].dropna().astype(str).unique())
        if parser_expected not in parser_values:
            warnings_list.append(f"expected parser status {parser_expected} not observed")
    else:
        errors.append("prepared word table missing parser_status")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
        "row_counts": checks,
        "split_names": sorted(split_names),
        "exposure_count_features_flagged": sorted(PRIMARY_EXPOSURE_COUNT_FEATURES),
    }


def _zscore_values(values: Any) -> Any:
    pd = _pd()
    numeric = pd.to_numeric(values, errors="coerce")
    sd = numeric.std()
    if sd is None or sd != sd or sd == 0:
        return numeric * 0
    return (numeric - numeric.mean()) / sd


def _simple_slope(frame: Any, x_col: str, y_col: str) -> float | None:
    np = _np()
    pd = _pd()
    x = pd.to_numeric(frame[x_col], errors="coerce") if x_col in frame else None
    y = pd.to_numeric(frame[y_col], errors="coerce") if y_col in frame else None
    if x is None or y is None:
        return None
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return None
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    denom = float(np.var(xv))
    if denom == 0:
        return None
    return float(np.cov(xv, yv, bias=True)[0, 1] / denom)


def _mean_or_nan(frame: Any, column: str) -> float:
    pd = _pd()
    if frame.empty or column not in frame:
        return float("nan")
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return float(value) if value == value else float("nan")


def _residual_design_matrices(train: Any, test: Any) -> tuple[Any, Any, list[str]]:
    pd = _pd()
    numeric = [
        column
        for column in [
            "word_length_chars",
            "log_corpus_frequency",
            "dfm_lm_word_surprisal",
            "dfm_lm_word_entropy",
            "sentence_length_words",
            "word_position_in_sentence_norm",
            "prev_boundary_opacity_score",
            "vocoid_run_cross_boundary",
            "vv_indicator",
            "lm_missing",
            "embedding_missing",
            "parser_missing",
            "segmentation_label_missing",
        ]
        if column in train.columns
    ]
    train_x = train[numeric].copy()
    test_x = test[numeric].copy()
    for column in train_x.columns:
        train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
        test_x[column] = pd.to_numeric(test_x[column], errors="coerce")
    if "speech_id" in train.columns:
        train_speech = pd.get_dummies(train["speech_id"].astype(str), prefix="speech", drop_first=True)
        test_speech = pd.get_dummies(test["speech_id"].astype(str), prefix="speech", drop_first=True)
        test_speech = test_speech.reindex(columns=train_speech.columns, fill_value=0)
        train_x = pd.concat([train_x, train_speech], axis=1)
        test_x = pd.concat([test_x, test_speech], axis=1)
    return train_x, test_x, train_x.columns.astype(str).tolist()


def _fit_predict_residuals_for_fold(
    train: Any, test: Any, outcome: str, *, binary: bool, seed: int
) -> tuple[Any, dict[str, Any]]:
    pd = _pd()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = pd.to_numeric(train[outcome], errors="coerce")
    train_mask = y_train.notna()
    if binary:
        train_mask = train_mask & y_train.isin([0, 1])
    fit_train = train.loc[train_mask].copy()
    y_fit = pd.to_numeric(fit_train[outcome], errors="coerce")
    if fit_train.empty or (binary and y_fit.nunique() < 2):
        return pd.Series(index=test.index, dtype=float), {
            "outcome": outcome,
            "status": "skipped",
            "skip_reason": "insufficient_training_outcome_variation",
            "n_train_rows": int(len(fit_train)),
        }
    x_train, x_test, columns = _residual_design_matrices(fit_train, test)
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
    if binary:
        prediction = model.predict_proba(x_test)[:, 1]
    else:
        prediction = model.predict(x_test)
    y_test = pd.to_numeric(test[outcome], errors="coerce")
    residual = pd.Series(index=test.index, dtype=float)
    residual.loc[test.index] = y_test - prediction
    return residual, {
        "outcome": outcome,
        "status": "complete",
        "skip_reason": "",
        "n_train_rows": int(len(fit_train)),
        "n_test_rows": int(len(test)),
        "n_predictors": int(len(columns)),
        "uses_reader_group": False,
    }


def _aggregate_crossfit_participant(group: Any, participant_id: str) -> dict[str, Any]:
    pd = _pd()
    row: dict[str, Any] = {"participant_id": participant_id}
    for _, prefix, _ in RESIDUAL_OUTCOME_SPECS:
        column = f"crossfit_{prefix}_residual"
        if column not in group:
            continue
        values = pd.to_numeric(group[column], errors="coerce")
        row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else None
        row[f"{column}_median"] = float(values.median()) if values.notna().any() else None
        row[f"{column}_sd"] = float(values.std()) if values.notna().sum() > 1 else None
        row[f"{column}_dfm_surprisal_slope"] = _simple_slope(
            group, "dfm_lm_word_surprisal", column
        )
        row[f"{column}_dfm_entropy_slope"] = _simple_slope(group, "dfm_lm_word_entropy", column)
        row[f"{column}_word_length_slope"] = _simple_slope(group, "word_length_chars", column)
        row[f"{column}_boundary_opacity_slope"] = _simple_slope(
            group, "prev_boundary_opacity_score", column
        )
        high = group[pd.to_numeric(group["prev_boundary_opacity_score"], errors="coerce").eq(3)]
        low = group[~pd.to_numeric(group["prev_boundary_opacity_score"], errors="coerce").eq(3)]
        vv = group[group["vv_indicator"].eq(1)]
        non_vv = group[group["vv_indicator"].eq(0)]
        row[f"{column}_high_opacity_cost"] = _mean_or_nan(high, column) - _mean_or_nan(low, column)
        row[f"{column}_vv_cost"] = _mean_or_nan(vv, column) - _mean_or_nan(non_vv, column)
    return row


def build_cross_fitted_residual_profiles(
    config: dict[str, Any], dirs: dict[str, Path], word: Any
) -> tuple[Any, dict[str, Any]]:
    pd = _pd()
    seed = int(get_nested(config, "phase4_confirmatory.deterministic_seed", 41))
    data = word.copy()
    participants = sorted(data["participant_id"].astype(str).unique())
    residual_frames = []
    diagnostics = []
    for fold_id, heldout in enumerate(participants):
        train = data[~data["participant_id"].astype(str).eq(heldout)].copy()
        test = data[data["participant_id"].astype(str).eq(heldout)].copy()
        fold_ok = True
        for outcome, prefix, binary in RESIDUAL_OUTCOME_SPECS:
            residual, diag = _fit_predict_residuals_for_fold(
                train, test, outcome, binary=binary, seed=seed + fold_id
            )
            test[f"crossfit_{prefix}_residual"] = residual
            diag.update(
                {
                    "fold_id": int(fold_id),
                    "heldout_participant_id": heldout,
                    "train_contains_heldout": bool(train["participant_id"].astype(str).eq(heldout).any()),
                    "test_participant_rows": int(len(test)),
                }
            )
            if diag["status"] != "complete":
                fold_ok = False
            diagnostics.append(diag)
        test["crossfit_fold_valid"] = fold_ok
        residual_frames.append(test)
    residual_rows = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    profile_rows = [
        _aggregate_crossfit_participant(group, str(participant_id))
        for participant_id, group in residual_rows.groupby("participant_id", dropna=False)
    ]
    profiles = pd.DataFrame(profile_rows)
    _write_parquet(dirs, "participant_sensitivity_profiles_crossfit.parquet", profiles)
    diag_frame = pd.DataFrame(diagnostics)
    by_outcome = (
        diag_frame.groupby("outcome", dropna=False)
        .agg(
            folds=("fold_id", "count"),
            complete_folds=("status", lambda s: int((s == "complete").sum())),
            skipped_folds=("status", lambda s: int((s != "complete").sum())),
            uses_reader_group=("uses_reader_group", "max"),
            train_contains_heldout=("train_contains_heldout", "max"),
        )
        .reset_index()
        if not diag_frame.empty
        else pd.DataFrame()
    )
    report = "\n".join(
        [
            "# Cross-Fitted Residualization Report",
            "",
            "Each participant's expected gaze model is fit on all other participants and then "
            "applied to that held-out participant's word rows. Reader group, participant labels, "
            "and participant identifiers are never residualization predictors.",
            "",
            "## Residualization Predictors",
            "\n".join(f"- `{column}`" for column in PHASE4_RESIDUALIZATION_PREDICTORS),
            "",
            "## Fold Diagnostics By Outcome",
            _markdown_table(
                by_outcome.to_dict("records"),
                [
                    "outcome",
                    "folds",
                    "complete_folds",
                    "skipped_folds",
                    "uses_reader_group",
                    "train_contains_heldout",
                ],
            ),
            "",
            "## Validation",
            f"- Held-out participant rows used to fit their own residual model: "
            f"{bool(diag_frame.get('train_contains_heldout', pd.Series(dtype=bool)).any())}",
            f"- Reader-group variables used in residualization: "
            f"{bool(diag_frame.get('uses_reader_group', pd.Series(dtype=bool)).any())}",
            f"- Participant profiles produced: {len(profiles)}",
            "",
            "## Output",
            "- `participant_sensitivity_profiles_crossfit.parquet`",
        ]
    )
    _write_report(dirs, "cross_fitted_residualization_report.md", report)
    return profiles, {
        "profile_rows": int(len(profiles)),
        "diagnostics": diag_frame.to_dict("records"),
        "by_outcome": by_outcome.to_dict("records") if not by_outcome.empty else [],
        "reader_group_used": False,
        "heldout_rows_used_for_fit": bool(
            diag_frame.get("train_contains_heldout", pd.Series(dtype=bool)).any()
        )
        if not diag_frame.empty
        else False,
    }


def _crossfit_columns(profiles: Any, suffix: str | None = None, contains: str | None = None) -> list[str]:
    columns = []
    for column in profiles.columns.astype(str):
        if not column.startswith("crossfit_"):
            continue
        if suffix is not None and not column.endswith(suffix):
            continue
        if contains is not None and contains not in column:
            continue
        columns.append(column)
    return sorted(columns)


def phase4_feature_groups(profiles: Any | None = None) -> dict[str, list[str]]:
    available = set(profiles.columns.astype(str)) if profiles is not None else set()
    residual_aggregates = _crossfit_columns(profiles, "_mean") if profiles is not None else []
    residual_aggregates += _crossfit_columns(profiles, "_median") if profiles is not None else []
    residual_aggregates += _crossfit_columns(profiles, "_sd") if profiles is not None else []
    dfm_residual = []
    segmentation_residual = []
    length_residual = []
    if profiles is not None:
        dfm_residual = [
            column
            for column in profiles.columns.astype(str)
            if column.startswith("crossfit_")
            and ("dfm_surprisal_slope" in column or "dfm_entropy_slope" in column)
        ]
        segmentation_residual = [
            column
            for column in profiles.columns.astype(str)
            if column.startswith("crossfit_")
            and ("boundary_opacity_slope" in column or "high_opacity_cost" in column or "vv_cost" in column)
        ]
        length_residual = [
            column
            for column in profiles.columns.astype(str)
            if column.startswith("crossfit_") and "word_length_slope" in column
        ]
    groups = {
        "A_raw_gaze": RAW_GAZE_FEATURES,
        "B_residual_gaze": residual_aggregates
        + [
            "high_opacity_trt_residual_cost",
            "vv_trt_residual_cost",
            "trt_residual_mean",
            "skipping_residual_mean",
        ],
        "C_sensitivity_slopes_only": [
            "length_sensitivity",
            "frequency_sensitivity",
            "surprisal_sensitivity",
            "entropy_sensitivity",
            "length_sensitivity_phase3",
            "frequency_sensitivity_phase3",
            "surprisal_sensitivity_phase3",
            "entropy_sensitivity_phase3",
            "boundary_opacity_sensitivity_phase3",
            *dfm_residual,
            *segmentation_residual,
            *length_residual,
        ],
        "D1_dfm_exposure_only": [
            "mean_dfm_surprisal_exposure",
            "mean_dfm_entropy_exposure",
            "lm_missing_rate_phase3",
        ],
        "D2_dfm_sensitivity_only": [
            "surprisal_sensitivity",
            "entropy_sensitivity",
            "surprisal_sensitivity_phase3",
            "entropy_sensitivity_phase3",
            *dfm_residual,
        ],
        "D3_dfm_residual_gaze_only": dfm_residual,
        "D4_dfm_exposure_plus_sensitivity": [
            "mean_dfm_surprisal_exposure",
            "mean_dfm_entropy_exposure",
            "lm_missing_rate_phase3",
            "surprisal_sensitivity",
            "entropy_sensitivity",
            "surprisal_sensitivity_phase3",
            "entropy_sensitivity_phase3",
            *dfm_residual,
        ],
        "E_segmentation_exposure_only": [
            "mean_boundary_opacity_exposure",
            "vv_boundary_exposure_rate",
        ],
        "F_segmentation_sensitivity_only": [
            "boundary_opacity_sensitivity_phase3",
            "high_opacity_trt_residual_cost",
            "vv_trt_residual_cost",
            *segmentation_residual,
        ],
    }
    all_allowed = _unique(
        groups["A_raw_gaze"]
        + groups["B_residual_gaze"]
        + groups["C_sensitivity_slopes_only"]
        + groups["D4_dfm_exposure_plus_sensitivity"]
        + groups["E_segmentation_exposure_only"]
        + groups["F_segmentation_sensitivity_only"]
        + [
            "mean_word_length_exposure",
            "mean_log_frequency_exposure",
            "mean_sentence_length_exposure",
            "age",
            "comprehension_score",
        ]
    )
    groups["G_all_allowed_non_exposure"] = [
        column for column in all_allowed if column not in EXPOSURE_ONLY_FEATURES
    ]
    groups["H_all_except_dfm"] = [
        column
        for column in all_allowed
        if "dfm" not in column and "surprisal" not in column and "entropy" not in column
    ]
    groups["I_all_except_segmentation"] = [
        column
        for column in all_allowed
        if "boundary" not in column and "opacity" not in column and "vv_" not in column
    ]
    groups["J_all_except_raw_speed"] = [
        column for column in all_allowed if column not in GLOBAL_SPEED_FEATURES
    ]
    groups["K_all_except_exposure_variables"] = [
        column for column in all_allowed if column not in EXPOSURE_ONLY_FEATURES
    ]
    cleaned = {}
    for name, columns in groups.items():
        unique_columns = _unique(columns)
        if profiles is not None:
            unique_columns = [column for column in unique_columns if column in available]
        cleaned[name] = unique_columns
    return cleaned


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def assert_phase4_primary_feature_sets_safe(feature_sets: dict[str, list[str]]) -> None:
    for name, columns in feature_sets.items():
        bad = set(columns) & PRIMARY_EXCLUDED_FEATURES
        if bad:
            raise ValueError(f"feature group {name} contains excluded primary features: {sorted(bad)}")


def _feature_group_metadata(feature_sets: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows = []
    sensitivity_groups = {
        "C_sensitivity_slopes_only",
        "D2_dfm_sensitivity_only",
        "D3_dfm_residual_gaze_only",
        "F_segmentation_sensitivity_only",
    }
    primary_allowed_groups = {
        "B_residual_gaze",
        "C_sensitivity_slopes_only",
        "D2_dfm_sensitivity_only",
        "D3_dfm_residual_gaze_only",
        "F_segmentation_sensitivity_only",
        "G_all_allowed_non_exposure",
        "J_all_except_raw_speed",
        "K_all_except_exposure_variables",
    }
    for group, columns in feature_sets.items():
        for column in columns:
            could_encode_text_assignment = column in EXPOSURE_ONLY_FEATURES or column in PRIMARY_EXPOSURE_COUNT_FEATURES
            rows.append(
                {
                    "feature_group": group,
                    "feature_name": column,
                    "could_encode_text_assignment": could_encode_text_assignment,
                    "allowed_in_primary_publication_model": group in primary_allowed_groups
                    and not could_encode_text_assignment
                    and column not in PRIMARY_EXCLUDED_FEATURES,
                    "sensitivity_only": group in sensitivity_groups,
                    "feature_family": _feature_family(column),
                }
            )
    return rows


def _feature_family(column: str) -> str:
    if column in PRIMARY_EXPOSURE_COUNT_FEATURES:
        return "exposure_count"
    if column in EXPOSURE_ONLY_FEATURES:
        return "text_exposure"
    if "dfm" in column or "surprisal" in column or "entropy" in column:
        return "dfm_sensitivity_or_exposure"
    if "boundary" in column or "opacity" in column or "vv_" in column:
        return "segmentation"
    if column.startswith("crossfit_") or "residual" in column:
        return "residual_gaze"
    if column in RAW_GAZE_FEATURES:
        return "raw_gaze"
    return "covariate_or_other"


def write_dfm_feature_group_dictionary(dirs: dict[str, Path], feature_sets: dict[str, list[str]]) -> None:
    pd = _pd()
    rows = _feature_group_metadata(feature_sets)
    metadata = pd.DataFrame(rows)
    dfm_rows = metadata[metadata["feature_group"].str.startswith("D")].copy()
    _write_report(
        dirs,
        "dfm_feature_group_dictionary.md",
        "\n".join(
            [
                "# DFM Feature Group Dictionary",
                "",
                "Phase 4 separates text-level DFM exposure from participant-level DFM sensitivity. "
                "Exposure-only groups are comparison analyses, not the preferred publication model.",
                "",
                _markdown_table(
                    dfm_rows.to_dict("records"),
                    [
                        "feature_group",
                        "feature_name",
                        "feature_family",
                        "could_encode_text_assignment",
                        "allowed_in_primary_publication_model",
                        "sensitivity_only",
                    ],
                    max_rows=200,
                ),
            ]
        ),
    )
    group_summary = (
        metadata.groupby("feature_group", dropna=False)
        .agg(
            n_features=("feature_name", "count"),
            could_encode_text_assignment=("could_encode_text_assignment", "max"),
            allowed_primary_features=("allowed_in_primary_publication_model", "sum"),
            sensitivity_only=("sensitivity_only", "max"),
        )
        .reset_index()
    )
    _write_report(
        dirs,
        "dfm_exposure_vs_sensitivity_report.md",
        "\n".join(
            [
                "# DFM Exposure Vs Sensitivity Report",
                "",
                "The Phase 3 combined DFM group is decomposed into explicit exposure-only, "
                "sensitivity-only, residual-gaze-only, and combined comparison groups.",
                "",
                "## Group Summary",
                _markdown_table(
                    group_summary.to_dict("records"),
                    [
                        "feature_group",
                        "n_features",
                        "could_encode_text_assignment",
                        "allowed_primary_features",
                        "sensitivity_only",
                    ],
                    max_rows=80,
                ),
                "",
                "## Interpretation Rules",
                "- `D1_dfm_exposure_only` can encode text assignment and is not a primary "
                "publication model.",
                "- `D2_dfm_sensitivity_only` and `D3_dfm_residual_gaze_only` are "
                "sensitivity-only DFM groups.",
                "- `D4_dfm_exposure_plus_sensitivity` reproduces the combined Phase 3 family "
                "as a comparison group.",
                "- Exposure-count variables remain flagged and excluded from all feature groups.",
            ]
        ),
    )


def _combine_participant_profiles(phase3_profiles: Any, crossfit_profiles: Any, participant: Any) -> Any:
    keep = [
        column
        for column in [
            "participant_id",
            "word_observation_count",
            "n_speeches",
            "n_word_rows",
            "n_words_read",
            "mean_segmentation_opacity",
            "mean_dfm_surprisal",
            "mean_dfm_entropy",
        ]
        if column in participant.columns
    ]
    combined = phase3_profiles.merge(crossfit_profiles, on="participant_id", how="left")
    if keep:
        combined = combined.merge(participant[keep].drop_duplicates("participant_id"), on="participant_id", how="left")
    return combined


def _phase4_models(seed: int) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        ),
        "linear_svm": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000),
        ),
    }


def _calibration_metrics(y_true: Any, y_score: Any) -> dict[str, Any]:
    np = _np()
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(y) < 3 or len(set(y.tolist())) < 2:
        return {"calibration_intercept": None, "calibration_slope": None}
    eps = 1e-6
    logits = np.log(np.clip(score, eps, 1 - eps) / np.clip(1 - score, eps, 1 - eps))
    if float(np.std(logits)) == 0:
        return {"calibration_intercept": None, "calibration_slope": None}
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        model.fit(logits.reshape(-1, 1), y)
        return {
            "calibration_intercept": float(model.intercept_[0]),
            "calibration_slope": float(model.coef_[0][0]),
        }
    except Exception:
        slope, intercept = np.polyfit(logits, y, 1)
        return {"calibration_intercept": float(intercept), "calibration_slope": float(slope)}


def _phase4_classification_metrics(y_true: Any, y_score: Any) -> dict[str, Any]:
    metrics = _classification_metrics(y_true, y_score)
    metrics.update(_calibration_metrics(y_true, y_score))
    return metrics


def _text_balanced_lopo_splits(profiles: Any, seed: int) -> Any:
    pd = _pd()
    needed = {"participant_id", "reader_group", "n_words_read", "n_speeches"}
    if not needed.issubset(profiles.columns):
        return pd.DataFrame()
    data = profiles[list(needed)].copy()
    ranges = []
    for column in ["n_words_read", "n_speeches"]:
        by_group = data.groupby("reader_group")[column].agg(["min", "max"])
        if len(by_group) < 2:
            return pd.DataFrame()
        lower = float(by_group["min"].max())
        upper = float(by_group["max"].min())
        if lower > upper:
            return pd.DataFrame()
        ranges.append((column, lower, upper))
    mask = pd.Series(True, index=data.index)
    for column, lower, upper in ranges:
        values = pd.to_numeric(data[column], errors="coerce")
        mask &= values.ge(lower) & values.le(upper)
    selected = data.loc[mask].copy()
    if selected["reader_group"].nunique() < 2:
        return pd.DataFrame()
    if selected.groupby("reader_group")["participant_id"].nunique().min() < 3:
        return pd.DataFrame()
    rows = []
    participants = sorted(selected["participant_id"].astype(str).unique())
    for fold_id, heldout in enumerate(participants):
        for participant_id in participants:
            rows.append(
                {
                    "split_name": "text_balanced_sensitivity_lopo",
                    "fold_id": fold_id,
                    "participant_id": participant_id,
                    "reader_group": selected.set_index("participant_id").loc[participant_id, "reader_group"],
                    "split_role": "test" if participant_id == heldout else "train",
                    "include_in_fold": True,
                    "split_valid": True,
                    "skip_reason": "",
                    "split_seed": seed,
                    "split_version": "phase4_text_balanced_overlap_v1",
                }
            )
    return pd.DataFrame(rows)


def _prediction_splits(config: dict[str, Any], profiles: Any, splits: Any) -> Any:
    pd = _pd()
    seed = int(get_nested(config, "phase4_confirmatory.deterministic_seed", 41))
    text_balanced = _text_balanced_lopo_splits(profiles, seed)
    if text_balanced.empty:
        return splits.copy()
    missing_cols = [column for column in splits.columns if column not in text_balanced.columns]
    for column in missing_cols:
        text_balanced[column] = None
    return pd.concat([splits, text_balanced[splits.columns]], ignore_index=True)


def _evaluate_split_predictions(
    data: Any,
    splits: Any,
    feature_columns: list[str],
    model_name: str,
    *,
    split_name: str,
    seed: int,
    label_column: str = "reader_group_binary",
) -> tuple[dict[str, Any], Any]:
    pd = _pd()
    np = _np()
    predictions = []
    skipped = 0
    usable = 0
    split_rows = splits[splits["split_name"].eq(split_name)].copy()
    if "include_in_fold" in split_rows.columns:
        split_rows = split_rows[split_rows["include_in_fold"].fillna(True).astype(bool)]
    if split_rows.empty:
        metric = {
            "n_predictions": 0,
            "usable_folds": 0,
            "skipped_folds": 0,
            **_phase4_classification_metrics([], []),
            "status": "skipped",
            "skip_reason": "split_not_available",
            "fold_validity": "skipped",
        }
        return metric, pd.DataFrame()
    fold_ids = sorted(split_rows["fold_id"].dropna().unique())
    for fold_id in fold_ids:
        fold = split_rows[split_rows["fold_id"].eq(fold_id)]
        train_ids = set(fold[fold["split_role"].eq("train")]["participant_id"].astype(str))
        test_ids = set(fold[fold["split_role"].eq("test")]["participant_id"].astype(str))
        if not test_ids or not train_ids or train_ids.intersection(test_ids):
            skipped += 1
            continue
        train = data[data["participant_id"].astype(str).isin(train_ids)].copy()
        test = data[data["participant_id"].astype(str).isin(test_ids)].copy()
        if train.empty or test.empty or train[label_column].nunique() < 2:
            skipped += 1
            continue
        estimator = _phase4_models(seed + int(fold_id)).get(model_name)
        if estimator is None:
            skipped += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(train[feature_columns], train[label_column].astype(int))
        score = _score_estimator(estimator, test[feature_columns])
        score = np.asarray(score, dtype=float)
        usable += 1
        for pid, truth, pred in zip(test["participant_id"], test[label_column], score, strict=True):
            predictions.append(
                {
                    "split_name": split_name,
                    "fold_id": int(fold_id),
                    "model": model_name,
                    "participant_id": pid,
                    "y_true": int(truth),
                    "y_score": float(pred),
                    "y_pred": int(pred >= 0.5),
                    "fold_valid": True,
                }
            )
    pred_frame = pd.DataFrame(predictions)
    if pred_frame.empty:
        return {
            "n_predictions": 0,
            "usable_folds": usable,
            "skipped_folds": skipped,
            **_phase4_classification_metrics([], []),
            "status": "skipped",
            "skip_reason": "no_valid_predictions",
            "fold_validity": "skipped",
        }, pred_frame
    metric = {
        "n_predictions": int(len(pred_frame)),
        "usable_folds": usable,
        "skipped_folds": skipped,
        **_phase4_classification_metrics(pred_frame["y_true"], pred_frame["y_score"]),
        "status": "complete",
        "skip_reason": "",
        "fold_validity": "all_test_predictions_generated" if skipped == 0 else "some_folds_skipped",
    }
    return metric, pred_frame


def run_confirmatory_prediction(
    config: dict[str, Any], dirs: dict[str, Path], profiles: Any, splits: Any
) -> tuple[Any, Any, dict[str, Any]]:
    pd = _pd()
    seed = int(get_nested(config, "phase4_confirmatory.deterministic_seed", 41))
    requested_models = set(
        get_nested(
            config,
            "phase4_confirmatory.modeling.participant_prediction_models",
            ["logistic_regression", "linear_svm"],
        )
    )
    models = [name for name in ["logistic_regression", "linear_svm"] if name in requested_models]
    split_data = _prediction_splits(config, profiles, splits)
    feature_sets = phase4_feature_groups(profiles)
    assert_phase4_primary_feature_sets_safe(feature_sets)
    metrics_rows = []
    prediction_frames = []
    split_names = [
        "leave_one_participant_out",
        "participant_grouped_kfold",
        "text_balanced_sensitivity_lopo",
    ]
    for feature_group, columns in feature_sets.items():
        if not columns:
            metrics_rows.append(
                {
                    "analysis": "phase4_confirmatory_participant_prediction",
                    "split_name": "all",
                    "feature_group": feature_group,
                    "model": "all",
                    "n_features": 0,
                    "n_predictions": 0,
                    "usable_folds": 0,
                    "skipped_folds": 0,
                    **_phase4_classification_metrics([], []),
                    "status": "skipped",
                    "skip_reason": "no_available_features",
                    "fold_validity": "skipped",
                }
            )
            continue
        for split_name in split_names:
            for model_name in models:
                metric, predictions = _evaluate_split_predictions(
                    profiles,
                    split_data,
                    columns,
                    model_name,
                    split_name=split_name,
                    seed=seed,
                )
                metric.update(
                    {
                        "analysis": "phase4_confirmatory_participant_prediction",
                        "split_name": split_name,
                        "feature_group": feature_group,
                        "model": model_name,
                        "n_features": len(columns),
                    }
                )
                metrics_rows.append(metric)
                if not predictions.empty:
                    predictions["feature_group"] = feature_group
                    predictions["n_features"] = len(columns)
                    prediction_frames.append(predictions)
    metrics = pd.DataFrame(metrics_rows)
    for column in PHASE4_METRIC_COLUMNS:
        if column not in metrics.columns:
            metrics[column] = None
    metrics = metrics[PHASE4_METRIC_COLUMNS]
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    _write_csv(dirs, "confirmatory_prediction_metrics.csv", metrics)
    _write_csv(dirs, "confirmatory_predictions.csv", predictions)
    best = _select_confirmatory_model(metrics)
    top = metrics[metrics["status"].eq("complete")].sort_values(
        ["split_name", "roc_auc"], ascending=[True, False], na_position="last"
    )
    report = "\n".join(
        [
            "# Confirmatory Prediction Report",
            "",
            "Prediction is participant-level only. Exposure-count variables are excluded from every "
            "feature group; DFM exposure-only groups are retained only as explicit comparisons.",
            "",
            "## Best Confirmatory Model",
            _markdown_table(
                [best],
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
                    "skipped_folds",
                ],
            ),
            "",
            "## Top Metric Rows",
            _markdown_table(
                top.head(20).to_dict("records"),
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
                    "fold_validity",
                ],
                max_rows=20,
            ),
        ]
    )
    _write_report(dirs, "confirmatory_prediction_report.md", report)
    return metrics, predictions, {
        "best_confirmatory_model": best,
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "text_balanced_split_available": bool(
            split_data["split_name"].eq("text_balanced_sensitivity_lopo").any()
        ),
    }


def _select_confirmatory_model(metrics: Any) -> dict[str, Any]:
    complete = metrics[
        (metrics["status"].eq("complete"))
        & metrics["split_name"].eq("leave_one_participant_out")
        & metrics["model"].eq("logistic_regression")
        & metrics["roc_auc"].notna()
    ].copy()
    preferred_groups = {
        "B_residual_gaze",
        "C_sensitivity_slopes_only",
        "D2_dfm_sensitivity_only",
        "D3_dfm_residual_gaze_only",
        "F_segmentation_sensitivity_only",
        "G_all_allowed_non_exposure",
        "J_all_except_raw_speed",
        "K_all_except_exposure_variables",
    }
    preferred = complete[complete["feature_group"].isin(preferred_groups)]
    if not preferred.empty:
        return preferred.sort_values("roc_auc", ascending=False).iloc[0].to_dict()
    if complete.empty:
        complete = metrics[(metrics["status"].eq("complete")) & metrics["roc_auc"].notna()].copy()
    if complete.empty:
        return {}
    return complete.sort_values("roc_auc", ascending=False).iloc[0].to_dict()


def _metric_lookup(metrics: Any, feature_group: str, *, model: str = "logistic_regression") -> dict[str, Any]:
    rows = metrics[
        metrics["split_name"].eq("leave_one_participant_out")
        & metrics["feature_group"].eq(feature_group)
        & metrics["model"].eq(model)
        & metrics["status"].eq("complete")
    ]
    if rows.empty:
        return {}
    return rows.sort_values("roc_auc", ascending=False).iloc[0].to_dict()


def run_robustness_tests(
    config: dict[str, Any],
    dirs: dict[str, Path],
    profiles: Any,
    splits: Any,
    metrics: Any,
) -> dict[str, Any]:
    pd = _pd()
    np = _np()
    seed = int(get_nested(config, "phase4_confirmatory.deterministic_seed", 41))
    permutation_count = int(get_nested(config, "phase4_confirmatory.modeling.permutation_count", 1000))
    bootstrap_count = int(get_nested(config, "phase4_confirmatory.modeling.bootstrap_count", 2000))
    feature_sets = phase4_feature_groups(profiles)
    selected = _select_confirmatory_model(metrics)
    if not selected:
        empty = pd.DataFrame()
        _write_csv(dirs, "permutation_results.csv", empty)
        _write_csv(dirs, "bootstrap_results.csv", empty)
        _write_csv(dirs, "influence_analysis.csv", empty)
        _write_report(dirs, "robustness_report.md", "# Robustness Report\n\nNo complete model.")
        return {"status": "skipped", "skip_reason": "no_complete_model"}
    feature_group = str(selected["feature_group"])
    model_name = str(selected["model"])
    split_name = str(selected["split_name"])
    columns = feature_sets[feature_group]
    observed = float(selected["roc_auc"])
    split_data = _prediction_splits(config, profiles, splits)
    rng = np.random.default_rng(seed)
    permutation_rows = []
    for iteration in range(permutation_count):
        permuted = profiles.copy()
        permuted["reader_group_binary"] = rng.permutation(
            permuted["reader_group_binary"].astype(int).to_numpy()
        )
        metric, _ = _evaluate_split_predictions(
            permuted,
            split_data,
            columns,
            model_name,
            split_name=split_name,
            seed=seed + iteration,
        )
        permutation_rows.append(
            {
                "iteration": iteration,
                "feature_group": feature_group,
                "model": model_name,
                "split_name": split_name,
                "roc_auc": metric["roc_auc"],
                "status": metric["status"],
            }
        )
    permutation_frame = pd.DataFrame(permutation_rows)
    valid_perm = pd.to_numeric(permutation_frame["roc_auc"], errors="coerce").dropna()
    permutation_p = None
    if len(valid_perm):
        permutation_p = float((int((valid_perm >= observed).sum()) + 1) / (len(valid_perm) + 1))
    _write_csv(dirs, "permutation_results.csv", permutation_frame)
    _, selected_predictions = _evaluate_split_predictions(
        profiles,
        split_data,
        columns,
        model_name,
        split_name=split_name,
        seed=seed,
    )
    bootstrap_rows = []
    if not selected_predictions.empty:
        participant_ids = selected_predictions["participant_id"].astype(str).unique()
        roc_scores = []
        pr_scores = []
        for iteration in range(bootstrap_count):
            sample = rng.choice(participant_ids, size=len(participant_ids), replace=True)
            boot = pd.concat(
                [
                    selected_predictions[selected_predictions["participant_id"].astype(str).eq(pid)]
                    for pid in sample
                ],
                ignore_index=True,
            )
            metric = _phase4_classification_metrics(boot["y_true"], boot["y_score"])
            if metric["roc_auc"] is not None:
                roc_scores.append(float(metric["roc_auc"]))
            if metric["pr_auc"] is not None:
                pr_scores.append(float(metric["pr_auc"]))
        for metric_name, values, observed_value in [
            ("roc_auc", roc_scores, selected.get("roc_auc")),
            ("pr_auc", pr_scores, selected.get("pr_auc")),
        ]:
            if values:
                arr = np.asarray(values, dtype=float)
                bootstrap_rows.append(
                    {
                        "metric": metric_name,
                        "feature_group": feature_group,
                        "model": model_name,
                        "split_name": split_name,
                        "observed": observed_value,
                        "n_bootstrap": int(len(arr)),
                        "ci_low": float(np.quantile(arr, 0.025)),
                        "ci_high": float(np.quantile(arr, 0.975)),
                    }
                )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    _write_csv(dirs, "bootstrap_results.csv", bootstrap_frame)
    influence_rows = []
    participant_ids = sorted(profiles["participant_id"].astype(str).unique())
    for participant_id in participant_ids:
        reduced = profiles[~profiles["participant_id"].astype(str).eq(participant_id)].copy()
        reduced_splits = split_data[~split_data["participant_id"].astype(str).eq(participant_id)].copy()
        metric, _ = _evaluate_split_predictions(
            reduced,
            reduced_splits,
            columns,
            model_name,
            split_name=split_name,
            seed=seed,
        )
        group_values = profiles.loc[
            profiles["participant_id"].astype(str).eq(participant_id), "reader_group"
        ].astype(str)
        group = group_values.iloc[0] if not group_values.empty else ""
        influence_rows.append(
            {
                "removed_participant_id": participant_id,
                "removed_reader_group": group,
                "leave_one_dyslexia_labeled": group == "dyslexia_labeled",
                "feature_group": feature_group,
                "model": model_name,
                "split_name": split_name,
                "roc_auc": metric["roc_auc"],
                "pr_auc": metric["pr_auc"],
                "delta_roc_auc": None
                if metric["roc_auc"] is None
                else float(metric["roc_auc"]) - observed,
                "status": metric["status"],
            }
        )
    influence = pd.DataFrame(influence_rows)
    _write_csv(dirs, "influence_analysis.csv", influence)
    rows_of_interest = [
        _metric_lookup(metrics, group)
        for group in [
            "D1_dfm_exposure_only",
            "D2_dfm_sensitivity_only",
            "D3_dfm_residual_gaze_only",
            "D4_dfm_exposure_plus_sensitivity",
            "J_all_except_raw_speed",
            "K_all_except_exposure_variables",
        ]
    ]
    rows_of_interest = [row for row in rows_of_interest if row]
    min_leave_one_dyslexia = None
    dyslexia_influence = influence[influence["leave_one_dyslexia_labeled"].eq(True)]
    if not dyslexia_influence.empty:
        min_leave_one_dyslexia = pd.to_numeric(dyslexia_influence["roc_auc"], errors="coerce").min()
    bootstrap_roc = {}
    if not bootstrap_frame.empty:
        roc_rows = bootstrap_frame[bootstrap_frame["metric"].eq("roc_auc")]
        if not roc_rows.empty:
            bootstrap_roc = roc_rows.iloc[0].to_dict()
    report = "\n".join(
        [
            "# Robustness Report",
            "",
            "Robustness is computed for the selected confirmatory LOPO model. Label permutation "
            "shuffles participant labels only; no word-level random split is used.",
            "",
            "## Selected Model",
            _markdown_table(
                [selected],
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
            "## Permutation And Bootstrap",
            _markdown_table(
                [
                    {
                        "observed_roc_auc": observed,
                        "valid_permutations": int(len(valid_perm)),
                        "permutation_p_value": permutation_p,
                        "bootstrap_roc_auc_low": bootstrap_roc.get("ci_low"),
                        "bootstrap_roc_auc_high": bootstrap_roc.get("ci_high"),
                        "leave_one_dyslexia_min_roc_auc": min_leave_one_dyslexia,
                    }
                ],
                [
                    "observed_roc_auc",
                    "valid_permutations",
                    "permutation_p_value",
                    "bootstrap_roc_auc_low",
                    "bootstrap_roc_auc_high",
                    "leave_one_dyslexia_min_roc_auc",
                ],
            ),
            "",
            "## Required Sensitivity Rows",
            _markdown_table(
                rows_of_interest,
                ["feature_group", "roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "brier_score"],
            ),
        ]
    )
    _write_report(dirs, "robustness_report.md", report)
    return {
        "status": "complete",
        "selected": selected,
        "permutation_count": int(len(valid_perm)),
        "permutation_p_value": permutation_p,
        "bootstrap": bootstrap_rows,
        "leave_one_dyslexia_min_roc_auc": None
        if min_leave_one_dyslexia != min_leave_one_dyslexia
        else float(min_leave_one_dyslexia),
    }


def compute_feature_stability(
    config: dict[str, Any],
    dirs: dict[str, Path],
    profiles: Any,
    splits: Any,
    metrics: Any,
) -> dict[str, Any]:
    pd = _pd()
    seed = int(get_nested(config, "phase4_confirmatory.deterministic_seed", 41))
    selected = _select_confirmatory_model(metrics)
    if not selected:
        empty = pd.DataFrame()
        _write_csv(dirs, "feature_stability_by_fold.csv", empty)
        _write_report(dirs, "feature_stability_report.md", "# Feature Stability Report\n\nNo complete model.")
        return {"status": "skipped", "skip_reason": "no_complete_model"}
    feature_group = str(selected["feature_group"])
    feature_sets = phase4_feature_groups(profiles)
    columns = feature_sets[feature_group]
    split_data = _prediction_splits(config, profiles, splits)
    lopo = split_data[split_data["split_name"].eq("leave_one_participant_out")]
    coefficient_rows = []
    for fold_id in sorted(lopo["fold_id"].dropna().unique()):
        fold = lopo[lopo["fold_id"].eq(fold_id)]
        train_ids = set(fold[fold["split_role"].eq("train")]["participant_id"].astype(str))
        test_ids = set(fold[fold["split_role"].eq("test")]["participant_id"].astype(str))
        train = profiles[profiles["participant_id"].astype(str).isin(train_ids)].copy()
        if train.empty or train["reader_group_binary"].nunique() < 2:
            continue
        estimator = _phase4_models(seed + int(fold_id))["logistic_regression"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(train[columns], train["reader_group_binary"].astype(int))
        coefs = estimator.named_steps["logisticregression"].coef_[0]
        heldout = sorted(test_ids)[0] if test_ids else ""
        for feature, coefficient in zip(columns, coefs, strict=True):
            coefficient_rows.append(
                {
                    "feature_group": feature_group,
                    "fold_id": int(fold_id),
                    "heldout_participant_id": heldout,
                    "feature": feature,
                    "standardized_logistic_coefficient": float(coefficient),
                    "coefficient_sign": "positive"
                    if coefficient > 0
                    else "negative"
                    if coefficient < 0
                    else "zero",
                }
            )
    stability = pd.DataFrame(coefficient_rows)
    _write_csv(dirs, "feature_stability_by_fold.csv", stability)
    if stability.empty:
        _write_report(dirs, "feature_stability_report.md", "# Feature Stability Report\n\nNo coefficients.")
        return {"status": "skipped", "skip_reason": "no_coefficients"}
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
    stable_positive = summary[
        (summary["mean_coefficient"] > 0) & (summary["sign_stability"] >= 0.8)
    ].sort_values("abs_mean_coefficient", ascending=False)
    stable_negative = summary[
        (summary["mean_coefficient"] < 0) & (summary["sign_stability"] >= 0.8)
    ].sort_values("abs_mean_coefficient", ascending=False)
    unstable = summary[summary["sign_stability"] < 0.65].sort_values(
        "abs_mean_coefficient", ascending=False
    )
    dfm_terms = summary[
        summary["feature"].astype(str).str.contains("dfm|surprisal|entropy", case=False, regex=True)
    ]
    segmentation_terms = summary[
        summary["feature"].astype(str).str.contains("boundary|opacity|vv_", case=False, regex=True)
    ]
    raw_speed_terms = summary[summary["feature"].isin(GLOBAL_SPEED_FEATURES)]
    raw_speed_dominates = False
    if not raw_speed_terms.empty:
        raw_speed_dominates = float(raw_speed_terms["abs_mean_coefficient"].max()) >= float(
            summary["abs_mean_coefficient"].max()
        )
    report = "\n".join(
        [
            "# Feature Stability Report",
            "",
            "Standardized logistic coefficients are recomputed inside LOPO folds for the selected "
            "confirmatory feature group.",
            "",
            "## Selected Model",
            _markdown_table(
                [selected],
                ["feature_group", "model", "split_name", "roc_auc", "pr_auc", "n_predictions"],
            ),
            "",
            "## Top Stable Positive Features",
            _markdown_table(
                stable_positive.head(15).to_dict("records"),
                ["feature", "mean_coefficient", "sign_stability", "positive_rate", "n_folds"],
                max_rows=15,
            ),
            "",
            "## Top Stable Negative Features",
            _markdown_table(
                stable_negative.head(15).to_dict("records"),
                ["feature", "mean_coefficient", "sign_stability", "negative_rate", "n_folds"],
                max_rows=15,
            ),
            "",
            "## Unstable Features",
            _markdown_table(
                unstable.head(15).to_dict("records"),
                ["feature", "mean_coefficient", "sign_stability", "positive_rate", "negative_rate"],
                max_rows=15,
            ),
            "",
            "## Stability Answers",
            f"- DFM sensitivity features stable: {bool((dfm_terms['sign_stability'] >= 0.8).any()) if not dfm_terms.empty else False}",
            f"- Segmentation sensitivity features stable: {bool((segmentation_terms['sign_stability'] >= 0.8).any()) if not segmentation_terms.empty else False}",
            f"- Raw speed dominates selected model: {raw_speed_dominates}",
        ]
    )
    _write_report(dirs, "feature_stability_report.md", report)
    return {
        "status": "complete",
        "selected": selected,
        "coefficient_rows": int(len(stability)),
        "stable_dfm_features": int((dfm_terms["sign_stability"] >= 0.8).sum())
        if not dfm_terms.empty
        else 0,
        "stable_segmentation_features": int((segmentation_terms["sign_stability"] >= 0.8).sum())
        if not segmentation_terms.empty
        else 0,
        "raw_speed_dominates": raw_speed_dominates,
    }


def _prepare_interaction_frame(config: dict[str, Any], word: Any) -> Any:
    pd = _pd()
    seed = int(get_nested(config, "phase4_confirmatory.deterministic_seed", 41))
    max_rows = int(get_nested(config, "phase4_confirmatory.modeling.mixed_effects_max_rows", len(word)))
    data = word.copy()
    if max_rows > 0 and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed + 7).copy()
    data["reader_group_binary_num"] = pd.to_numeric(data["reader_group_binary"], errors="coerce")
    for column in [
        "word_length_chars",
        "log_corpus_frequency",
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "sentence_length_words",
        "word_position_in_sentence_norm",
        "prev_boundary_opacity_score",
        "vv_indicator",
    ]:
        if column in data.columns:
            data[f"{column}_z"] = _zscore_values(data[column])
    for column in ["lm_missing", "embedding_missing", "parser_missing"]:
        if column in data.columns:
            data[f"{column}_num"] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(float)
    return data


def run_confirmatory_interactions(
    config: dict[str, Any], dirs: dict[str, Path], word: Any
) -> dict[str, Any]:
    pd = _pd()
    data = _prepare_interaction_frame(config, word)
    coefficient_rows = []
    diagnostics = []
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as exc:
        coefficients = pd.DataFrame(columns=_mixed_effects_columns())
        _write_csv(dirs, "mixed_effects_coefficients.csv", coefficients)
        _write_report(
            dirs,
            "mixed_effects_interaction_report.md",
            f"# Mixed-Effects Interaction Report\n\nStatsmodels unavailable: `{exc}`.",
        )
        return {"status": "skipped", "skip_reason": "statsmodels_unavailable"}
    formula_terms = [
        "reader_group_binary_num",
        "word_length_chars_z",
        "dfm_lm_word_surprisal_z",
        "prev_boundary_opacity_score_z",
        "reader_group_binary_num:word_length_chars_z",
        "reader_group_binary_num:dfm_lm_word_surprisal_z",
        "reader_group_binary_num:prev_boundary_opacity_score_z",
        "log_corpus_frequency_z",
        "dfm_lm_word_entropy_z",
        "sentence_length_words_z",
        "word_position_in_sentence_norm_z",
        "vv_indicator_z",
        "lm_missing_num",
        "embedding_missing_num",
        "C(speech_id)",
    ]
    model_columns = [
        "participant_id",
        "speech_id",
        "reader_group_binary_num",
        "word_length_chars_z",
        "dfm_lm_word_surprisal_z",
        "prev_boundary_opacity_score_z",
        "log_corpus_frequency_z",
        "dfm_lm_word_entropy_z",
        "sentence_length_words_z",
        "word_position_in_sentence_norm_z",
        "vv_indicator_z",
        "lm_missing_num",
        "embedding_missing_num",
    ]
    outcomes = [
        ("skip", "logistic"),
        ("log_ffd", "linear"),
        ("log_first_pass_duration", "linear"),
        ("log_go_past_time", "linear"),
        ("log_total_fixation_duration", "linear"),
        ("fixation_count", "linear"),
    ]
    for outcome, kind in outcomes:
        frame = data.copy()
        if outcome not in frame.columns:
            diagnostics.append({"outcome": outcome, "status": "skipped", "reason": "missing_outcome"})
            continue
        frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
        needed = [outcome, *[column for column in model_columns if column in frame.columns]]
        frame = frame.dropna(subset=[column for column in needed if column in frame.columns])
        if frame.empty or frame["reader_group_binary_num"].nunique() < 2:
            diagnostics.append({"outcome": outcome, "status": "skipped", "reason": "insufficient_rows"})
            continue
        formula = f"{outcome} ~ " + " + ".join(formula_terms)
        warning_messages: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if kind == "logistic":
                    result = smf.glm(formula=formula, data=frame, family=sm.families.Binomial()).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": frame["participant_id"].astype(str)},
                    )
                    model_type = "cluster_robust_glm_binomial"
                else:
                    result = smf.ols(formula=formula, data=frame).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": frame["participant_id"].astype(str)},
                    )
                    model_type = "cluster_robust_ols"
                warning_messages = [str(item.message) for item in caught]
            conf = result.conf_int()
            for term, estimate in result.params.items():
                ci_low, ci_high = conf.loc[term].tolist()
                coefficient_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_kind": kind,
                        "term": term,
                        "phase4_interaction": _interaction_label(term),
                        "estimate": float(estimate),
                        "std_error": float(result.bse[term]),
                        "p_value": float(result.pvalues[term]),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "n_obs": int(result.nobs),
                        "model_type": model_type,
                        "random_effects_attempted": False,
                        "fallback_reason": (
                            "crossed participant/item mixed effects were not feasible in this "
                            "automated confirmatory runner; participant-clustered covariance and "
                            "speech fixed effects are used"
                        ),
                        "convergence_warnings": "; ".join(warning_messages[:3]),
                    }
                )
            diagnostics.append(
                {
                    "outcome": outcome,
                    "status": "complete",
                    "model_type": model_type,
                    "n_obs": int(result.nobs),
                    "warnings": "; ".join(warning_messages[:3]),
                }
            )
        except Exception as exc:
            diagnostics.append({"outcome": outcome, "status": "failed", "reason": str(exc)})
    coefficients = pd.DataFrame(coefficient_rows, columns=_mixed_effects_columns())
    _write_csv(dirs, "mixed_effects_coefficients.csv", coefficients)
    focus = coefficients[coefficients["phase4_interaction"].astype(str).ne("")].copy()
    if not focus.empty:
        focus["survives_controls"] = pd.to_numeric(focus["p_value"], errors="coerce") < 0.05
        focus["effect_direction"] = focus["estimate"].map(
            lambda value: "positive" if value > 0 else "negative" if value < 0 else "zero"
        )
    diagnostics_frame = pd.DataFrame(diagnostics)
    report = "\n".join(
        [
            "# Mixed-Effects Interaction Report",
            "",
            "The confirmatory interaction model focuses only on the Phase 3 candidates: "
            "reader group by word length, DFM surprisal, and previous-boundary opacity. "
            "Cluster-robust fallback models are used with speech fixed effects because crossed "
            "mixed effects are not feasible in this automated run.",
            "",
            "## Focus Interaction Coefficients",
            _markdown_table(
                focus[
                    [
                        "outcome",
                        "phase4_interaction",
                        "estimate",
                        "std_error",
                        "p_value",
                        "ci_low",
                        "ci_high",
                        "effect_direction",
                        "survives_controls",
                    ]
                ].to_dict("records")
                if not focus.empty
                else [],
                [
                    "outcome",
                    "phase4_interaction",
                    "estimate",
                    "std_error",
                    "p_value",
                    "ci_low",
                    "ci_high",
                    "effect_direction",
                    "survives_controls",
                ],
                max_rows=80,
            ),
            "",
            "## Diagnostics",
            _markdown_table(
                diagnostics_frame.to_dict("records"),
                ["outcome", "status", "model_type", "n_obs", "warnings", "reason"],
                max_rows=20,
            ),
            "",
            "## Framing",
            "- Interactions with controlled p-values below 0.05 are treated as confirmatory support.",
            "- Non-surviving interactions should be appendix or deferred framing, not main claims.",
        ]
    )
    _write_report(dirs, "mixed_effects_interaction_report.md", report)
    return {
        "status": "complete" if not coefficients.empty else "skipped",
        "coefficient_rows": int(len(coefficients)),
        "focus_interaction_rows": int(len(focus)),
        "surviving_focus_interactions": int(focus["survives_controls"].sum()) if not focus.empty else 0,
        "diagnostics": diagnostics,
    }


def _mixed_effects_columns() -> list[str]:
    return [
        "outcome",
        "outcome_kind",
        "term",
        "phase4_interaction",
        "estimate",
        "std_error",
        "p_value",
        "ci_low",
        "ci_high",
        "n_obs",
        "model_type",
        "random_effects_attempted",
        "fallback_reason",
        "convergence_warnings",
    ]


def _interaction_label(term: str) -> str:
    for label, statsmodels_term in INTERACTION_TERMS.items():
        if term == statsmodels_term:
            return label
    return ""


def write_segmentation_decision_report(
    dirs: dict[str, Path], metrics: Any, mixed: dict[str, Any]
) -> dict[str, Any]:
    pd = _pd()
    coeff_path = dirs["result_analysis"] / "mixed_effects_coefficients.csv"
    coefficients = pd.read_csv(coeff_path) if coeff_path.exists() else pd.DataFrame()
    boundary = pd.DataFrame()
    if not coefficients.empty:
        boundary = coefficients[
            coefficients["phase4_interaction"].eq("reader_group_x_previous_boundary_opacity")
        ].copy()
        boundary["survives_controls"] = pd.to_numeric(boundary["p_value"], errors="coerce") < 0.05
    segmentation_metric = _metric_lookup(metrics, "F_segmentation_sensitivity_only")
    dfm_metric = _metric_lookup(metrics, "D2_dfm_sensitivity_only")
    boundary_survives = bool(boundary["survives_controls"].any()) if not boundary.empty else False
    decision = "secondary_result" if boundary_survives else "appendix_result"
    if not segmentation_metric:
        decision = "defer"
    report = "\n".join(
        [
            "# Segmentation Decision Report",
            "",
            "## Decision",
            f"- Category: `{decision}`",
            "- Standalone segmentation main-effect framing: `drop`",
            "- Segmentation retained as a secondary interaction and interpretability feature.",
            "- Pronunciation-aware labels are recommended only if the boundary-opacity interaction "
            "remains meaningful in controlled models.",
            "",
            "## Boundary Opacity Beyond DFM Surprisal",
            _markdown_table(
                boundary[
                    [
                        "outcome",
                        "estimate",
                        "std_error",
                        "p_value",
                        "ci_low",
                        "ci_high",
                        "survives_controls",
                    ]
                ].to_dict("records")
                if not boundary.empty
                else [],
                ["outcome", "estimate", "std_error", "p_value", "ci_low", "ci_high", "survives_controls"],
                max_rows=30,
            ),
            "",
            "## Prediction Context",
            _markdown_table(
                [row for row in [segmentation_metric, dfm_metric] if row],
                ["feature_group", "roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "brier_score"],
            ),
        ]
    )
    _write_report(dirs, "segmentation_decision_report.md", report)
    return {
        "decision_category": decision,
        "boundary_interaction_survives_controls": boundary_survives,
        "standalone_segmentation_main_effect_framing": "drop",
        "mixed_summary": mixed,
    }


def write_phase4_publication_decision_report(
    dirs: dict[str, Path],
    metrics: Any,
    robustness: dict[str, Any],
    stability: dict[str, Any],
    mixed: dict[str, Any],
    segmentation: dict[str, Any],
) -> dict[str, Any]:
    selected = robustness.get("selected") or _select_confirmatory_model(metrics)
    d1 = _metric_lookup(metrics, "D1_dfm_exposure_only")
    d2 = _metric_lookup(metrics, "D2_dfm_sensitivity_only")
    d3 = _metric_lookup(metrics, "D3_dfm_residual_gaze_only")
    d4 = _metric_lookup(metrics, "D4_dfm_exposure_plus_sensitivity")
    j = _metric_lookup(metrics, "J_all_except_raw_speed")
    k = _metric_lookup(metrics, "K_all_except_exposure_variables")
    dfm_sensitivity_beats_exposure = _metric_value(d2, "roc_auc") is not None and (
        _metric_value(d1, "roc_auc") is None or _metric_value(d2, "roc_auc") >= _metric_value(d1, "roc_auc")
    )
    survives_crossfit = _metric_value(selected, "roc_auc") is not None and _metric_value(selected, "roc_auc") >= 0.7
    survives_no_raw_speed = _metric_value(j, "roc_auc") is not None and _metric_value(j, "roc_auc") >= 0.7
    survives_no_exposure = _metric_value(k, "roc_auc") is not None and _metric_value(k, "roc_auc") >= 0.7
    interaction_category = (
        "secondary_result" if mixed.get("surviving_focus_interactions", 0) > 0 else "appendix_result"
    )
    recommendations = [
        {
            "result": "participant-level DFM sensitivity and cross-fitted residualized gaze-cost profiles",
            "category": "main_paper_result" if survives_crossfit else "appendix_result",
        },
        {
            "result": "DFM exposure-only prediction",
            "category": "appendix_result" if d1 else "defer",
        },
        {
            "result": "boundary opacity",
            "category": segmentation["decision_category"],
        },
        {
            "result": "standalone segmentation main-effect framing",
            "category": "drop",
        },
        {
            "result": "random word-level prediction",
            "category": "drop",
        },
        {
            "result": "parser-syntax claims while parser status is surface_heuristic_fallback",
            "category": "defer",
        },
    ]
    answers = [
        {
            "question": "Does participant-level prediction survive cross-fitted residualization?",
            "answer": survives_crossfit,
            "evidence": _brief_metric(selected),
        },
        {
            "question": "Is DFM sensitivity more important than DFM exposure?",
            "answer": dfm_sensitivity_beats_exposure,
            "evidence": f"D1={_brief_metric(d1)}; D2={_brief_metric(d2)}; D3={_brief_metric(d3)}; D4={_brief_metric(d4)}",
        },
        {
            "question": "Does performance survive removal of exposure-count variables?",
            "answer": True,
            "evidence": "Exposure-count variables are excluded from every Phase 4 feature group.",
        },
        {
            "question": "Does performance survive removal of exposure-only variables?",
            "answer": survives_no_exposure,
            "evidence": _brief_metric(k),
        },
        {
            "question": "Does performance survive removal of raw speed/global-duration variables?",
            "answer": survives_no_raw_speed,
            "evidence": _brief_metric(j),
        },
        {
            "question": "Are word length, DFM surprisal, and boundary-opacity interactions stable?",
            "answer": mixed.get("surviving_focus_interactions", 0) > 0,
            "evidence": f"{mixed.get('surviving_focus_interactions', 0)} controlled focus interactions survive.",
        },
        {
            "question": "Is segmentation a main finding, secondary finding, or deferred?",
            "answer": segmentation["decision_category"],
            "evidence": "Standalone main-effect framing is dropped; boundary opacity is retained only as interaction/interpretability.",
        },
    ]
    report = "\n".join(
        [
            "# Phase 4 Publication Decision Report",
            "",
            "## Main Decision",
            "Main paper should focus on participant-level DFM predictability sensitivity and "
            "cross-fitted residualized gaze-cost profiles if the selected confirmatory model remains "
            "above the prespecified performance threshold and robustness tests remain supportive.",
            "",
            "Boundary opacity should be retained only as an interaction and interpretability feature "
            "unless stronger evidence emerges.",
            "",
            "## Required Answers",
            _markdown_table(answers, ["question", "answer", "evidence"], max_rows=20),
            "",
            "## Decision Categories",
            _markdown_table(recommendations, ["result", "category"], max_rows=20),
            "",
            "## Robustness And Stability",
            _markdown_table(
                [
                    {
                        "permutation_p_value": robustness.get("permutation_p_value"),
                        "leave_one_dyslexia_min_roc_auc": robustness.get(
                            "leave_one_dyslexia_min_roc_auc"
                        ),
                        "stable_dfm_features": stability.get("stable_dfm_features"),
                        "stable_segmentation_features": stability.get("stable_segmentation_features"),
                        "raw_speed_dominates": stability.get("raw_speed_dominates"),
                    }
                ],
                [
                    "permutation_p_value",
                    "leave_one_dyslexia_min_roc_auc",
                    "stable_dfm_features",
                    "stable_segmentation_features",
                    "raw_speed_dominates",
                ],
            ),
            "",
            "## Still Needed Before Writing",
            "- Confirm Gemma sensitivity only if gated model access is available and alignment safeguards pass.",
            "- Decide whether the cluster-robust interaction fallback is sufficient for the paper or "
            "whether a slower dedicated mixed-model run is needed.",
            "- Keep parser-syntax claims deferred while parser status remains `surface_heuristic_fallback`.",
            "",
            "## Exclude From Main Paper",
            "- Exposure-count and text-assignment proxy models.",
            "- Random word-level predictive evaluations.",
            "- Standalone segmentation main-effect framing.",
            "- Clinical screening or diagnostic claims.",
        ]
    )
    _write_report(dirs, "phase4_publication_decision_report.md", report)
    return {
        "selected_confirmatory_model": selected,
        "survives_cross_fitted_residualization": survives_crossfit,
        "dfm_sensitivity_beats_exposure": dfm_sensitivity_beats_exposure,
        "survives_no_raw_speed": survives_no_raw_speed,
        "survives_no_exposure_only_variables": survives_no_exposure,
        "recommendations": recommendations,
        "interaction_category": interaction_category,
    }


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key) if row else None
    try:
        if value is None or value != value:
            return None
        return float(value)
    except Exception:
        return None


def _brief_metric(row: dict[str, Any]) -> str:
    if not row:
        return "not available"
    return (
        f"{row.get('feature_group')} {row.get('model')} {row.get('split_name')}: "
        f"ROC-AUC={_format_value(row.get('roc_auc'))}, PR-AUC={_format_value(row.get('pr_auc'))}, "
        f"balanced accuracy={_format_value(row.get('balanced_accuracy'))}"
    )


def _write_preflight_report(dirs: dict[str, Path], preflight: dict[str, Any]) -> None:
    report = "\n".join(
        [
            "# Phase 4 Preflight Validation Report",
            "",
            f"- Status: `{preflight['status']}`",
            "",
            "## Row Counts",
            _markdown_table(
                [{"table": key, "rows": value} for key, value in preflight["row_counts"].items()],
                ["table", "rows"],
            ),
            "",
            "## Split Names",
            "\n".join(f"- `{name}`" for name in preflight["split_names"]),
            "",
            "## Exposure-Count Features Flagged For Exclusion",
            "\n".join(f"- `{name}`" for name in preflight["exposure_count_features_flagged"]),
            "",
            "## Errors",
            "\n".join(f"- {error}" for error in preflight["errors"]) if preflight["errors"] else "_None._",
            "",
            "## Warnings",
            "\n".join(f"- {warning}" for warning in preflight["warnings"])
            if preflight["warnings"]
            else "_None._",
        ]
    )
    _write_report(dirs, "phase4_preflight_validation_report.md", report)


def _summarize_existing_prediction(metrics: Any, predictions: Any) -> dict[str, Any]:
    return {
        "best_confirmatory_model": _select_confirmatory_model(metrics),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "text_balanced_split_available": bool(
            metrics["split_name"].eq("text_balanced_sensitivity_lopo").any()
        )
        if "split_name" in metrics
        else False,
        "reused_existing_outputs": True,
    }


def _summarize_existing_robustness(dirs: dict[str, Path], metrics: Any) -> dict[str, Any]:
    pd = _pd()
    selected = _select_confirmatory_model(metrics)
    permutation_path = dirs["result_analysis"] / "permutation_results.csv"
    bootstrap_path = dirs["result_analysis"] / "bootstrap_results.csv"
    influence_path = dirs["result_analysis"] / "influence_analysis.csv"
    permutation_p = None
    permutation_count = 0
    if permutation_path.exists() and selected:
        permutation = pd.read_csv(permutation_path)
        valid = pd.to_numeric(permutation.get("roc_auc"), errors="coerce").dropna()
        permutation_count = int(len(valid))
        observed = _metric_value(selected, "roc_auc")
        if observed is not None and permutation_count:
            permutation_p = float((int((valid >= observed).sum()) + 1) / (permutation_count + 1))
    bootstrap_rows = []
    if bootstrap_path.exists():
        bootstrap_rows = pd.read_csv(bootstrap_path).to_dict("records")
    leave_one_dyslexia_min = None
    if influence_path.exists():
        influence = pd.read_csv(influence_path)
        if "leave_one_dyslexia_labeled" in influence.columns:
            dyslexia = influence[influence["leave_one_dyslexia_labeled"].eq(True)]
            if not dyslexia.empty:
                value = pd.to_numeric(dyslexia["roc_auc"], errors="coerce").min()
                leave_one_dyslexia_min = None if value != value else float(value)
    return {
        "status": "complete",
        "selected": selected,
        "permutation_count": permutation_count,
        "permutation_p_value": permutation_p,
        "bootstrap": bootstrap_rows,
        "leave_one_dyslexia_min_roc_auc": leave_one_dyslexia_min,
        "reused_existing_outputs": True,
    }


def _summarize_existing_stability(dirs: dict[str, Path], metrics: Any) -> dict[str, Any]:
    pd = _pd()
    path = dirs["result_analysis"] / "feature_stability_by_fold.csv"
    selected = _select_confirmatory_model(metrics)
    if not path.exists():
        return {"status": "skipped", "skip_reason": "missing_feature_stability_by_fold"}
    stability = pd.read_csv(path)
    if stability.empty:
        return {"status": "skipped", "skip_reason": "empty_feature_stability_by_fold"}
    summary = (
        stability.groupby(["feature_group", "feature"], dropna=False)
        .agg(
            mean_coefficient=("standardized_logistic_coefficient", "mean"),
            positive_rate=("coefficient_sign", lambda s: float((s == "positive").mean())),
            negative_rate=("coefficient_sign", lambda s: float((s == "negative").mean())),
        )
        .reset_index()
    )
    summary["sign_stability"] = summary[["positive_rate", "negative_rate"]].max(axis=1)
    dfm_terms = summary[
        summary["feature"].astype(str).str.contains("dfm|surprisal|entropy", case=False, regex=True)
    ]
    segmentation_terms = summary[
        summary["feature"].astype(str).str.contains("boundary|opacity|vv_", case=False, regex=True)
    ]
    raw_speed_terms = summary[summary["feature"].isin(GLOBAL_SPEED_FEATURES)]
    summary["abs_mean_coefficient"] = summary["mean_coefficient"].abs()
    raw_speed_dominates = False
    if not raw_speed_terms.empty:
        raw_speed_dominates = float(raw_speed_terms["abs_mean_coefficient"].max()) >= float(
            summary["abs_mean_coefficient"].max()
        )
    return {
        "status": "complete",
        "selected": selected,
        "coefficient_rows": int(len(stability)),
        "stable_dfm_features": int((dfm_terms["sign_stability"] >= 0.8).sum())
        if not dfm_terms.empty
        else 0,
        "stable_segmentation_features": int((segmentation_terms["sign_stability"] >= 0.8).sum())
        if not segmentation_terms.empty
        else 0,
        "raw_speed_dominates": raw_speed_dominates,
        "reused_existing_outputs": True,
    }


def run_phase4_confirmatory(
    config: dict[str, Any], output_dir: str | Path | None = None, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    if get_nested(config, "phase4_confirmatory.no_new_core_labels", True) is not True:
        raise ValueError("Phase 4 must not generate new core labels")
    if get_nested(config, "phase4_confirmatory.no_broad_exploratory_feature_expansion", True) is not True:
        raise ValueError("Phase 4 must not run broad exploratory feature expansion")
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=repo_root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, repo_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(config, repo_root)
    preflight = validate_phase4_preflight(config, inputs, repo_root=repo_root)
    _write_preflight_report(dirs, preflight)
    if preflight["status"] != "passed":
        _write_json(out / "phase4_confirmatory_validation_report.json", preflight)
        raise ValueError(f"phase4 preflight failed: {preflight['errors']}")
    word = _with_derived_columns(_merge_boundary_vocoid(inputs["word"], inputs["segmentation_boundary"]))
    crossfit_path = dirs["result_analysis"] / "participant_sensitivity_profiles_crossfit.parquet"
    if crossfit_path.exists() and (dirs["result_analysis"] / "cross_fitted_residualization_report.md").exists():
        crossfit_profiles = _pd().read_parquet(crossfit_path)
        crossfit = {
            "profile_rows": int(len(crossfit_profiles)),
            "diagnostics": [],
            "by_outcome": [],
            "reader_group_used": False,
            "heldout_rows_used_for_fit": False,
            "reused_existing_outputs": True,
        }
    else:
        crossfit_profiles, crossfit = build_cross_fitted_residual_profiles(config, dirs, word)
    profiles = _combine_participant_profiles(
        inputs["phase3_profiles"], crossfit_profiles, inputs["participant"]
    )
    feature_sets = phase4_feature_groups(profiles)
    write_dfm_feature_group_dictionary(dirs, feature_sets)
    metrics_path = dirs["result_analysis"] / "confirmatory_prediction_metrics.csv"
    predictions_path = dirs["result_analysis"] / "confirmatory_predictions.csv"
    if (
        metrics_path.exists()
        and predictions_path.exists()
        and (dirs["result_analysis"] / "confirmatory_prediction_report.md").exists()
    ):
        metrics = _pd().read_csv(metrics_path)
        predictions = _pd().read_csv(predictions_path)
        prediction_summary = _summarize_existing_prediction(metrics, predictions)
    else:
        metrics, predictions, prediction_summary = run_confirmatory_prediction(
            config, dirs, profiles, inputs["splits"]
        )
    if (
        (dirs["result_analysis"] / "permutation_results.csv").exists()
        and (dirs["result_analysis"] / "bootstrap_results.csv").exists()
        and (dirs["result_analysis"] / "influence_analysis.csv").exists()
        and (dirs["result_analysis"] / "robustness_report.md").exists()
    ):
        robustness = _summarize_existing_robustness(dirs, metrics)
    else:
        robustness = run_robustness_tests(config, dirs, profiles, inputs["splits"], metrics)
    if (
        (dirs["result_analysis"] / "feature_stability_by_fold.csv").exists()
        and (dirs["result_analysis"] / "feature_stability_report.md").exists()
    ):
        stability = _summarize_existing_stability(dirs, metrics)
    else:
        stability = compute_feature_stability(config, dirs, profiles, inputs["splits"], metrics)
    mixed = run_confirmatory_interactions(config, dirs, word)
    segmentation = write_segmentation_decision_report(dirs, metrics, mixed)
    decision = write_phase4_publication_decision_report(
        dirs, metrics, robustness, stability, mixed, segmentation
    )
    manifest = {
        "run_type": "phase4_confirmatory_sensitivity_v1",
        "status": "complete",
        "git_sha": _git_sha(repo_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_dir": str(out),
        "label_release_dir": str(inputs["label_dir"]),
        "prepared_dataset_dir": str(inputs["prepared_dir"]),
        "phase3_analysis_dir": str(inputs["phase3_analysis_dir"]),
        "preflight": preflight,
        "row_counts": {
            "word_level": int(len(inputs["word"])),
            "participant_level": int(len(inputs["participant"])),
            "phase3_profiles": int(len(inputs["phase3_profiles"])),
            "crossfit_profiles": int(len(crossfit_profiles)),
            "prediction_rows": int(len(predictions)),
        },
        "feature_groups": {name: columns for name, columns in feature_sets.items()},
        "cross_fitted_residualization": crossfit,
        "confirmatory_prediction": prediction_summary,
        "robustness": robustness,
        "feature_stability": stability,
        "mixed_effects_interactions": mixed,
        "segmentation_decision": segmentation,
        "publication_decision": decision,
        "large_outputs_not_for_commit": [
            "analysis/phase4_confirmatory/participant_sensitivity_profiles_crossfit.parquet",
            "results/phase4_confirmatory_sensitivity_v1_*/",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    validation = validate_phase4_confirmatory(config, out, repo_root=repo_root)
    _write_json(out / "phase4_confirmatory_validation_report.json", validation)
    return manifest


def validate_phase4_confirmatory(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    pd = _pd()
    out = Path(output_dir).resolve()
    dirs = _analysis_dirs(config, out, repo_root)
    errors: list[str] = []
    warnings_list: list[str] = []
    required_reports = [
        "phase4_preflight_validation_report.md",
        "dfm_exposure_vs_sensitivity_report.md",
        "dfm_feature_group_dictionary.md",
        "cross_fitted_residualization_report.md",
        "confirmatory_prediction_report.md",
        "robustness_report.md",
        "feature_stability_report.md",
        "mixed_effects_interaction_report.md",
        "segmentation_decision_report.md",
        "phase4_publication_decision_report.md",
    ]
    for name in required_reports:
        if not (dirs["result_analysis"] / name).exists():
            errors.append(f"missing report: {name}")
    required_csv = [
        "confirmatory_prediction_metrics.csv",
        "confirmatory_predictions.csv",
        "permutation_results.csv",
        "bootstrap_results.csv",
        "influence_analysis.csv",
        "feature_stability_by_fold.csv",
        "mixed_effects_coefficients.csv",
    ]
    for name in required_csv:
        path = dirs["result_analysis"] / name
        if not path.exists():
            errors.append(f"missing csv: {name}")
    metrics_path = dirs["result_analysis"] / "confirmatory_prediction_metrics.csv"
    if metrics_path.exists():
        columns = set(pd.read_csv(metrics_path).columns)
        missing = set(PHASE4_METRIC_COLUMNS) - columns
        if missing:
            errors.append(f"confirmatory metrics missing columns: {sorted(missing)}")
    predictions_path = dirs["result_analysis"] / "confirmatory_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        lopo = predictions[
            predictions["split_name"].eq("leave_one_participant_out")
            & predictions["feature_group"].eq("D2_dfm_sensitivity_only")
            & predictions["model"].eq("logistic_regression")
        ]
        if not lopo.empty and lopo["participant_id"].duplicated().any():
            errors.append("LOPO D2 logistic predictions contain duplicate participants")
    profile_path = dirs["result_analysis"] / "participant_sensitivity_profiles_crossfit.parquet"
    if not profile_path.exists():
        errors.append("missing participant_sensitivity_profiles_crossfit.parquet")
    else:
        profiles = pd.read_parquet(profile_path)
        if "participant_id" not in profiles or profiles["participant_id"].duplicated().any():
            errors.append("crossfit participant profiles missing unique participant_id")
    try:
        inputs = _load_inputs(config, repo_root)
        preflight = validate_phase4_preflight(config, inputs, repo_root=repo_root)
        if preflight["status"] != "passed":
            errors.extend(preflight["errors"])
            warnings_list.extend(preflight["warnings"])
    except Exception as exc:
        errors.append(f"preflight reload failed: {exc}")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
        "output_dir": str(out),
    }
    return report
