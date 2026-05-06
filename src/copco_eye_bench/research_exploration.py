"""Phase 3 controlled research exploration from Label Release v1.1."""

from __future__ import annotations

import json
import math
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir


RESIDUALIZATION_PREDICTORS = [
    "word_length_chars",
    "log_corpus_frequency",
    "dfm_lm_word_surprisal",
    "dfm_lm_word_entropy",
    "sentence_length_words",
    "word_position_in_sentence_norm",
    "prev_boundary_opacity_score",
    "vocoid_run_cross_boundary",
    "vv_indicator",
    "speech_id",
]

PRIMARY_EXPOSURE_COUNT_FEATURES = {
    "n_words_read",
    "n_speeches",
    "n_word_rows",
    "word_observation_count",
    "total_word_rows",
}

PARTICIPANT_METRIC_COLUMNS = [
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
    "calibration_mean_predicted",
    "calibration_observed_rate",
    "status",
    "skip_reason",
]


def _pd() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for Phase 3 research exploration") from exc
    return pd


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for Phase 3 research exploration") from exc
    return np


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _git_sha(repo_root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _markdown_table(records: list[dict[str, Any]], columns: list[str], *, max_rows: int = 30) -> str:
    if not records or not columns:
        return "_No rows._"
    rows = records[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    if len(records) > max_rows:
        body.append("| ... | " + " | ".join("" for _ in columns[1:]) + " |")
    return "\n".join([header, sep, *body])


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    analysis_rel = str(
        get_nested(config, "research_exploration.output_layout.analysis", "analysis/research_exploration")
    )
    repo_analysis = root / str(
        get_nested(config, "research_exploration.repo_analysis_dir", "analysis/research_exploration")
    )
    result_analysis = out / analysis_rel
    figures_rel = str(
        get_nested(config, "research_exploration.output_layout.figures", "analysis/research_exploration/figures")
    )
    return {
        "repo_analysis": repo_analysis,
        "result_analysis": result_analysis,
        "repo_figures": repo_analysis / "figures",
        "result_figures": out / figures_rel,
    }


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["result_analysis"] / name, text)
    _write_md(dirs["repo_analysis"] / name, text)


def _write_csv(dirs: dict[str, Path], name: str, frame: Any) -> None:
    path = dirs["result_analysis"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    repo = dirs["repo_analysis"] / name
    repo.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(repo, index=False)


def _write_parquet(dirs: dict[str, Path], name: str, frame: Any) -> None:
    path = dirs["result_analysis"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    repo = dirs["repo_analysis"] / name
    repo.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(repo, index=False)


def _prepared_dir(config: dict[str, Any], repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    path = Path(str(get_nested(config, "research_exploration.prepared_dataset_dir")))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _label_release_dir(config: dict[str, Any], repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    path = Path(str(get_nested(config, "research_exploration.label_release_dir")))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_inputs(config: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    pd = _pd()
    label_dir = _label_release_dir(config, repo_root)
    prepared = _prepared_dir(config, repo_root)
    return {
        "label_dir": label_dir,
        "prepared_dir": prepared,
        "word": pd.read_parquet(prepared / "analysis_ready_word_level_v1_1.parquet"),
        "sentence": pd.read_parquet(prepared / "analysis_ready_sentence_level_v1_1.parquet"),
        "participant": pd.read_parquet(prepared / "analysis_ready_participant_level_v1_1.parquet"),
        "participant_labels": pd.read_parquet(label_dir / "labels" / "participant_labels_v1.parquet"),
        "segmentation_word": pd.read_parquet(label_dir / "labels" / "segmentation_word_labels_v1.parquet"),
        "segmentation_boundary": pd.read_parquet(
            label_dir / "labels" / "segmentation_boundary_labels_v1.parquet"
        ),
        "quality": pd.read_parquet(label_dir / "labels" / "quality_labels_v1.parquet"),
        "splits": pd.read_parquet(label_dir / "labels" / "split_labels_v1.parquet"),
    }


def validate_preflight(
    config: dict[str, Any], inputs: dict[str, Any] | None = None, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    data = inputs or _load_inputs(config, repo_root)
    errors: list[str] = []
    warnings_list: list[str] = []
    word = data["word"]
    participant = data["participant"]
    participant_labels = data["participant_labels"]
    segmentation_word = data["segmentation_word"]
    segmentation_boundary = data["segmentation_boundary"]
    quality = data["quality"]
    splits = data["splits"]

    expected = get_nested(config, "research_exploration.expected_row_counts", {})
    checks = {
        "word_level": len(word),
        "sentence_level": len(data["sentence"]),
        "participant_level": len(participant),
        "participant_labels": len(participant_labels),
        "segmentation_words": len(segmentation_word),
        "segmentation_boundaries": len(segmentation_boundary),
        "quality_labels": len(quality),
    }
    for key, actual in checks.items():
        expected_value = expected.get(key) if isinstance(expected, dict) else None
        if expected_value is not None and int(actual) != int(expected_value):
            errors.append(f"{key} row count {actual} != expected {expected_value}")
    if "participant_word_key" not in word or word["participant_word_key"].duplicated().any():
        errors.append("prepared word table has missing or duplicate participant_word_key")
    if word["reader_group"].isna().any():
        errors.append("prepared word table has missing participant labels")
    if word["segmentation_label_version"].isna().any():
        errors.append("prepared word table has missing segmentation labels")
    if "parser_status" not in word:
        errors.append("prepared word table missing parser_status")
    else:
        expected_parser = str(
            get_nested(config, "research_exploration.parser_status_expected", "surface_heuristic_fallback")
        )
        parser_values = set(word["parser_status"].dropna().astype(str).unique())
        if expected_parser not in parser_values:
            errors.append(f"parser_status does not include expected value {expected_parser}")
    for column in ["lm_alignment_status", "lm_alignment_warning", "lm_alignment_error", "lm_missing"]:
        if column not in word.columns:
            errors.append(f"prepared word table missing LM warning field: {column}")
    legal = set(
        get_nested(
            config,
            "research_exploration.leakage_policy.legal_split_names",
            ["leave_one_participant_out", "participant_grouped_kfold", "sensitivity_exclude_uncertain_labels"],
        )
    )
    split_names = set(splits["split_name"].dropna().astype(str).unique())
    illegal = sorted(split_names - legal)
    if illegal:
        errors.append(f"illegal split names present: {illegal}")
    if splits["split_name"].str.contains("random", case=False, na=False).any():
        errors.append("random split label found")
    for (split_name, fold_id), group in splits.groupby(["split_name", "fold_id"], dropna=False):
        train = set(group[group["split_role"].isin(["train", "include"])]["participant_id"].astype(str))
        test = set(group[group["split_role"].eq("test")]["participant_id"].astype(str))
        if train.intersection(test):
            errors.append(f"participant train/test overlap in {split_name} fold {fold_id}")
    if word["participant_label_missing"].any():
        errors.append("participant_label_missing is true for prepared word rows")
    if word["segmentation_label_missing"].any():
        errors.append("segmentation_label_missing is true for prepared word rows")
    if word["dfm_lm_alignment_warning"].fillna("").eq("non_special_token_unassigned").mean() > 0.9:
        warnings_list.append("most rows inherit non_special_token_unassigned LM alignment warnings")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
        "row_counts": checks,
        "split_names": sorted(split_names),
    }


def _with_derived_columns(frame: Any) -> Any:
    pd = _pd()
    out = frame.copy()
    out["reader_group_binary_num"] = pd.to_numeric(out["reader_group_binary"], errors="coerce")
    out["vv_indicator"] = out["prev_boundary_type_orth"].astype(str).eq("V#V").astype(int)
    out["fixated"] = 1 - pd.to_numeric(out["skip"], errors="coerce").fillna(0).clip(0, 1)
    for source, target in [
        ("FFD", "log_ffd"),
        ("GD", "log_first_pass_duration"),
        ("go_past_time", "log_go_past_time"),
        ("TRT", "log_total_fixation_duration"),
    ]:
        values = pd.to_numeric(out[source], errors="coerce") if source in out else pd.Series(index=out.index)
        out[target] = values.where(values > 0).map(lambda value: math.log1p(value) if value == value else None)
    if "fixation_count" in out:
        out["log_fixation_count"] = pd.to_numeric(out["fixation_count"], errors="coerce").map(
            lambda value: math.log1p(value) if value == value else None
        )
    return out


def _sample_word_rows(frame: Any, max_rows: int, seed: int) -> Any:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=seed).copy()


def _numeric_summary_by_group(frame: Any, columns: list[str]) -> list[dict[str, Any]]:
    pd = _pd()
    records = []
    grouped = frame.groupby("reader_group", dropna=False)
    for group, data in grouped:
        row: dict[str, Any] = {"reader_group": group}
        row["participants"] = int(data["participant_id"].nunique())
        row["word_rows"] = int(len(data))
        row["speeches_read"] = int(data["speech_id"].nunique())
        row["mean_words_per_participant"] = float(data.groupby("participant_id").size().mean())
        row["mean_speeches_per_participant"] = float(data.groupby("participant_id")["speech_id"].nunique().mean())
        for column in columns:
            if column in data:
                row[f"mean_{column}"] = float(pd.to_numeric(data[column], errors="coerce").mean())
        row["vv_exposure_rate"] = float(data["prev_boundary_type_orth"].astype(str).eq("V#V").mean())
        row["embedding_missingness"] = float(data["embedding_missing"].mean())
        row["lm_missingness"] = float(data["lm_missing"].mean())
        row["parser_status"] = ";".join(sorted(data["parser_status"].astype(str).unique()))
        if "comprehension_score" in data:
            row["mean_comprehension_score"] = float(pd.to_numeric(data["comprehension_score"], errors="coerce").mean())
        if "age" in data:
            row["mean_age"] = float(pd.to_numeric(data["age"], errors="coerce").mean())
        if "sex" in data:
            row["sex_distribution"] = ", ".join(
                f"{key}:{value}" for key, value in data[["participant_id", "sex"]].drop_duplicates()["sex"].value_counts().items()
            )
        records.append(row)
    return records


def write_text_exposure_audit(dirs: dict[str, Path], word: Any) -> dict[str, Any]:
    exposure_columns = [
        "word_length_chars",
        "log_corpus_frequency",
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "sentence_length_words",
        "prev_boundary_opacity_score",
    ]
    records = _numeric_summary_by_group(word, exposure_columns)
    flagged = sorted(PRIMARY_EXPOSURE_COUNT_FEATURES)
    report = "\n".join(
        [
            "# Text Exposure Audit",
            "",
            "This audit quantifies reader-group exposure in the prepared dataset. Exposure-count "
            "variables are documented as confounds and are excluded from primary predictive feature sets.",
            "",
            "## Exposure By Reader Group",
            _markdown_table(records, list(records[0]) if records else [], max_rows=10),
            "",
            "## Variables Flagged As Exposure Counts",
            "\n".join(f"- `{name}`" for name in flagged),
            "",
            "## Interpretation",
            "Reader groups differ in amount of text exposure and speech coverage. Later models should "
            "control text/stimulus predictors and should not treat exposure-count variables as primary "
            "predictive features.",
        ]
    )
    _write_report(dirs, "text_exposure_audit.md", report)
    return {"records": records, "flagged_features": flagged}


def write_lm_warning_audit(dirs: dict[str, Path], word: Any) -> dict[str, Any]:
    pd = _pd()
    by_speech = (
        word.groupby("speech_id", dropna=False)
        .agg(
            rows=("word_id", "count"),
            warning_rows=("lm_alignment_warning", lambda s: s.notna().sum()),
            lm_missing_rate=("lm_missing", "mean"),
        )
        .reset_index()
    )
    by_speech["warning_rate"] = by_speech["warning_rows"] / by_speech["rows"].clip(lower=1)
    by_group = (
        word.groupby("reader_group", dropna=False)
        .agg(
            rows=("word_id", "count"),
            warning_rows=("lm_alignment_warning", lambda s: s.notna().sum()),
            lm_missing_rate=("lm_missing", "mean"),
            mean_word_length=("word_length_chars", "mean"),
            mean_surprisal=("dfm_lm_word_surprisal", "mean"),
            mean_entropy=("dfm_lm_word_entropy", "mean"),
        )
        .reset_index()
    )
    by_group["warning_rate"] = by_group["warning_rows"] / by_group["rows"].clip(lower=1)
    examples = (
        word[word["lm_alignment_warning"].fillna("").eq("non_special_token_unassigned")]
        [["speech_id", "sentence_id", "word_id", "word", "lm_alignment_status", "lm_alignment_warning", "lm_missing"]]
        .head(20)
        .to_dict("records")
    )
    complete = word[~word["lm_missing"]]
    missing = word[word["lm_missing"]]
    distribution_shift = []
    for column in ["word_length_chars", "log_corpus_frequency", "sentence_length_words", "prev_boundary_opacity_score"]:
        distribution_shift.append(
            {
                "feature": column,
                "all_rows_mean": float(pd.to_numeric(word[column], errors="coerce").mean()),
                "lm_complete_mean": float(pd.to_numeric(complete[column], errors="coerce").mean()),
                "lm_missing_mean": None
                if missing.empty
                else float(pd.to_numeric(missing[column], errors="coerce").mean()),
            }
        )
    warning_missing = (
        word.groupby("lm_alignment_warning", dropna=False)
        .agg(rows=("word_id", "count"), lm_missing_rate=("lm_missing", "mean"))
        .reset_index()
        .to_dict("records")
    )
    _write_csv(dirs, "lm_warning_by_speech.csv", by_speech)
    report = "\n".join(
        [
            "# LM Warning Audit",
            "",
            "The DFM LM warning `non_special_token_unassigned` is treated as an audit flag. Rows are "
            "not excluded automatically because most contexts inherit this warning and validation found "
            "no alignment errors.",
            "",
            "## Warning And Missingness By Reader Group",
            _markdown_table(by_group.to_dict("records"), list(by_group.columns), max_rows=10),
            "",
            "## Warning And Missingness By Speech",
            _markdown_table(by_speech.to_dict("records"), list(by_speech.columns), max_rows=80),
            "",
            "## Warning Examples",
            _markdown_table(
                examples,
                ["speech_id", "sentence_id", "word_id", "word", "lm_alignment_status", "lm_alignment_warning", "lm_missing"],
            ),
            "",
            "## Do Warnings Cause Missing LM Values?",
            _markdown_table(warning_missing, ["lm_alignment_warning", "rows", "lm_missing_rate"]),
            "",
            "## Distribution Shift If LM-Missing Rows Are Excluded",
            _markdown_table(
                distribution_shift,
                ["feature", "all_rows_mean", "lm_complete_mean", "lm_missing_mean"],
            ),
        ]
    )
    _write_report(dirs, "lm_warning_audit.md", report)
    return {
        "by_group": by_group.to_dict("records"),
        "warning_missing": warning_missing,
        "distribution_shift": distribution_shift,
    }


def _zscore(series: Any) -> Any:
    pd = _pd()
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std()
    if sd is None or sd != sd or sd == 0:
        return values * 0
    return (values - values.mean()) / sd


def _design_matrix(
    data: Any,
    numeric_predictors: list[str],
    *,
    include_speech: bool = True,
    categoricals: list[str] | None = None,
    interactions_with_group: bool = False,
) -> tuple[Any, list[str]]:
    pd = _pd()
    frame = data.copy()
    columns = []
    for predictor in numeric_predictors:
        if predictor not in frame.columns:
            continue
        name = f"{predictor}_z"
        frame[name] = _zscore(frame[predictor])
        columns.append(name)
        if interactions_with_group:
            iname = f"reader_group_x_{predictor}_z"
            frame[iname] = frame["reader_group_binary_num"] * frame[name]
            columns.append(iname)
    if interactions_with_group:
        frame["reader_group_binary_num"] = pd.to_numeric(frame["reader_group_binary_num"], errors="coerce")
        columns.insert(0, "reader_group_binary_num")
    for categorical in categoricals or []:
        if categorical not in frame:
            continue
        dummies = pd.get_dummies(frame[categorical].astype(str), prefix=categorical, drop_first=True)
        frame = pd.concat([frame, dummies], axis=1)
        columns.extend(dummies.columns.tolist())
    if include_speech and "speech_id" in frame:
        speech = pd.get_dummies(frame["speech_id"].astype(str), prefix="speech", drop_first=True)
        frame = pd.concat([frame, speech], axis=1)
        columns.extend(speech.columns.tolist())
    return frame[columns], columns


def _fit_association_model(
    data: Any,
    *,
    outcome: str,
    outcome_kind: str,
    numeric_predictors: list[str],
    categoricals: list[str] | None = None,
    include_speech: bool = True,
    interactions_with_group: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pd = _pd()
    from sklearn.linear_model import LogisticRegression, Ridge

    keep = [outcome, "participant_id", "reader_group_binary_num", "speech_id", *(categoricals or []), *numeric_predictors]
    keep = [column for column in dict.fromkeys(keep) if column in data.columns]
    frame = data[keep].copy()
    frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
    frame = frame.dropna(subset=[outcome, "participant_id"])
    if frame.empty or frame["participant_id"].nunique() < 2:
        return [], {"status": "skipped", "reason": "insufficient_complete_cases"}
    x, columns = _design_matrix(
        frame,
        numeric_predictors,
        include_speech=include_speech,
        categoricals=categoricals,
        interactions_with_group=interactions_with_group,
    )
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y = frame[outcome].astype(float)
    try:
        if outcome_kind == "logistic":
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=17,
            )
            model.fit(x, y.astype(int))
            coef = model.coef_[0]
            intercept = float(model.intercept_[0])
            model_type = "regularized_logistic_regression"
        else:
            model = Ridge(alpha=1.0)
            model.fit(x, y)
            coef = model.coef_
            intercept = float(model.intercept_)
            model_type = "ridge_linear_regression"
    except Exception as exc:
        return [], {
            "outcome": outcome,
            "status": "failed",
            "reason": str(exc),
            "n_obs": int(len(frame)),
            "n_terms": int(len(columns)),
        }

    bootstrap = _bootstrap_association_coefficients(
        frame,
        outcome,
        outcome_kind,
        numeric_predictors,
        categoricals=categoricals,
        include_speech=include_speech,
        interactions_with_group=interactions_with_group,
        terms=columns,
    )
    rows = [
        {
            "outcome": outcome,
            "outcome_kind": outcome_kind,
            "term": "intercept",
            "estimate": intercept,
            "std_error": None,
            "p_value": None,
            "ci_low": None,
            "ci_high": None,
            "n_obs": int(len(frame)),
            "model_type": model_type,
        }
    ]
    for term, estimate in zip(columns, coef, strict=True):
        uncertainty = bootstrap.get(term, {})
        rows.append(
            {
                "outcome": outcome,
                "outcome_kind": outcome_kind,
                "term": term,
                "estimate": float(estimate),
                "std_error": uncertainty.get("std_error"),
                "p_value": uncertainty.get("sign_p_value"),
                "ci_low": uncertainty.get("ci_low"),
                "ci_high": uncertainty.get("ci_high"),
                "n_obs": int(len(frame)),
                "model_type": model_type,
            }
        )
    diagnostics = {
        "status": "complete",
        "model_type": model_type,
        "n_obs": int(len(frame)),
        "n_terms": int(len(columns)),
        "outcome": outcome,
        "bootstrap_replicates": int(max((item.get("n_bootstrap", 0) for item in bootstrap.values()), default=0)),
        "pseudo_r2_or_r2": None,
    }
    return rows, diagnostics


def _bootstrap_association_coefficients(
    frame: Any,
    outcome: str,
    outcome_kind: str,
    numeric_predictors: list[str],
    *,
    categoricals: list[str] | None,
    include_speech: bool,
    interactions_with_group: bool,
    terms: list[str],
    bootstrap_count: int = 12,
    max_rows: int = 20_000,
) -> dict[str, dict[str, Any]]:
    pd = _pd()
    np = _np()
    from sklearn.linear_model import LogisticRegression, Ridge

    if bootstrap_count <= 0 or frame["participant_id"].nunique() < 3:
        return {}
    rng = np.random.default_rng(17)
    participants = frame["participant_id"].astype(str).unique()
    estimates: dict[str, list[float]] = {term: [] for term in terms}
    for _ in range(bootstrap_count):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        boot = pd.concat(
            [frame[frame["participant_id"].astype(str).eq(pid)] for pid in sampled],
            ignore_index=True,
        )
        if max_rows > 0 and len(boot) > max_rows:
            boot = boot.sample(n=max_rows, random_state=int(rng.integers(0, 1_000_000))).copy()
        x_boot, cols = _design_matrix(
            boot,
            numeric_predictors,
            include_speech=include_speech,
            categoricals=categoricals,
            interactions_with_group=interactions_with_group,
        )
        x_boot = x_boot.reindex(columns=terms, fill_value=0.0)
        x_boot = x_boot.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        y_boot = pd.to_numeric(boot[outcome], errors="coerce")
        mask = y_boot.notna()
        x_boot = x_boot[mask]
        y_boot = y_boot[mask]
        if outcome_kind == "logistic":
            if y_boot.nunique() < 2:
                continue
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=17,
            )
            model.fit(x_boot, y_boot.astype(int))
            coefs = model.coef_[0]
        else:
            model = Ridge(alpha=1.0)
            model.fit(x_boot, y_boot)
            coefs = model.coef_
        for term, value in zip(terms, coefs, strict=True):
            estimates[term].append(float(value))
    out = {}
    for term, values in estimates.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        sign_p = 2 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
        out[term] = {
            "std_error": float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
            "sign_p_value": min(1.0, sign_p),
            "n_bootstrap": int(len(arr)),
        }
    return out


def _plot_segmentation_effects(dirs: dict[str, Path], word: Any) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    data = word.copy()
    grouped = (
        data.groupby("prev_boundary_opacity_score", dropna=True)
        .agg(
            skip_rate=("skip", "mean"),
            mean_trt=("TRT", "mean"),
            mean_ffd=("FFD", "mean"),
            rows=("word_id", "count"),
        )
        .reset_index()
    )
    for path in [dirs["result_figures"] / "segmentation_effects.png", dirs["repo_figures"] / "segmentation_effects.png"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
        axes[0].plot(grouped["prev_boundary_opacity_score"], grouped["skip_rate"], marker="o")
        axes[0].set_title("Skipping")
        axes[1].plot(grouped["prev_boundary_opacity_score"], grouped["mean_ffd"], marker="o")
        axes[1].set_title("FFD")
        axes[2].plot(grouped["prev_boundary_opacity_score"], grouped["mean_trt"], marker="o")
        axes[2].set_title("TRT")
        for ax in axes:
            ax.set_xlabel("Previous-boundary opacity")
        fig.savefig(path, dpi=150)
        plt.close(fig)


def run_segmentation_psycholinguistic(
    config: dict[str, Any], dirs: dict[str, Path], word: Any
) -> dict[str, Any]:
    pd = _pd()
    seed = int(get_nested(config, "research_exploration.deterministic_seed", 17))
    max_rows = int(get_nested(config, "research_exploration.modeling.max_word_model_rows", 100000))
    data = _sample_word_rows(word, max_rows, seed)
    outcomes = [
        ("skip", "logistic"),
        ("log_ffd", "linear"),
        ("log_first_pass_duration", "linear"),
        ("log_go_past_time", "linear"),
        ("log_total_fixation_duration", "linear"),
        ("fixation_count", "linear"),
    ]
    predictors = [
        "prev_boundary_opacity_score",
        "vv_indicator",
        "vocoid_run_cross_boundary",
        "word_length_chars",
        "log_corpus_frequency",
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "sentence_length_words",
        "word_position_in_sentence_norm",
    ]
    rows = []
    diagnostics = []
    for outcome, kind in outcomes:
        coef_rows, diag = _fit_association_model(
            data,
            outcome=outcome,
            outcome_kind=kind,
            numeric_predictors=predictors,
            include_speech=True,
        )
        for row in coef_rows:
            row["model_family"] = "opacity_controlled"
        rows.extend(coef_rows)
        diagnostics.append(diag)
    type_rows, type_diag = _fit_association_model(
        data,
        outcome="log_total_fixation_duration",
        outcome_kind="linear",
        numeric_predictors=[
            "word_length_chars",
            "log_corpus_frequency",
            "dfm_lm_word_surprisal",
            "dfm_lm_word_entropy",
            "sentence_length_words",
            "word_position_in_sentence_norm",
        ],
        categoricals=["prev_boundary_type_orth"],
        include_speech=True,
    )
    for row in type_rows:
        row["model_family"] = "boundary_type_controlled"
    rows.extend(type_rows)
    diagnostics.append(type_diag)
    coefficient_columns = [
        "outcome",
        "outcome_kind",
        "term",
        "estimate",
        "std_error",
        "p_value",
        "ci_low",
        "ci_high",
        "n_obs",
        "model_type",
        "model_family",
    ]
    coefficients = pd.DataFrame(rows, columns=coefficient_columns)
    _write_csv(dirs, "segmentation_psycholinguistic_coefficients.csv", coefficients)
    _plot_segmentation_effects(dirs, word)
    focus_terms = coefficients[
        coefficients["term"].isin(
            [
                "prev_boundary_opacity_score_z",
                "vv_indicator_z",
                "vocoid_run_cross_boundary_z",
                "prev_boundary_type_orth_V#V",
            ]
        )
    ].copy()
    significant = focus_terms[pd.to_numeric(focus_terms["p_value"], errors="coerce") < 0.05]
    opacity_rows = coefficients[coefficients["term"].eq("prev_boundary_opacity_score_z")]
    consistent_direction = None
    if not opacity_rows.empty:
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in opacity_rows["estimate"]]
        consistent_direction = len(set(signs)) == 1
    report = "\n".join(
        [
            "# Segmentation Psycholinguistic Report",
            "",
            f"- Model rows sampled deterministically: {len(data)} of {len(word)}",
            "- Models use participant-clustered standard errors when fit succeeds.",
            "- Speech fixed effects are included as controls.",
            "",
            "## Boundary Terms",
            _markdown_table(
                focus_terms[
                    ["outcome", "model_family", "term", "estimate", "std_error", "p_value", "ci_low", "ci_high", "n_obs", "model_type"]
                ].to_dict("records"),
                ["outcome", "model_family", "term", "estimate", "std_error", "p_value", "ci_low", "ci_high", "n_obs", "model_type"],
                max_rows=80,
            ),
            "",
            "## Diagnostics",
            _markdown_table(diagnostics, ["outcome", "status", "model_type", "n_obs", "n_terms"]),
            "",
            "## Summary",
            f"- Boundary-opacity significant terms: {len(significant)}",
            f"- Boundary-opacity direction consistent across modeled outcomes: {consistent_direction}",
            "- Effect plot: `figures/segmentation_effects.png`",
            "",
            "These results are controlled exploratory associations, not label generation. A positive "
            "duration coefficient indicates higher gaze cost; a positive skipping coefficient indicates "
            "higher skipping odds.",
        ]
    )
    _write_report(dirs, "segmentation_psycholinguistic_report.md", report)
    return {
        "coefficient_rows": int(len(coefficients)),
        "significant_boundary_terms": int(len(significant)),
        "consistent_opacity_direction": consistent_direction,
    }


def run_group_interactions(config: dict[str, Any], dirs: dict[str, Path], word: Any) -> dict[str, Any]:
    pd = _pd()
    seed = int(get_nested(config, "research_exploration.deterministic_seed", 17))
    max_rows = int(get_nested(config, "research_exploration.modeling.max_word_model_rows", 100000))
    data = _sample_word_rows(word, max_rows, seed + 1)
    interaction_predictors = [
        "word_length_chars",
        "log_corpus_frequency",
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "prev_boundary_opacity_score",
        "vv_indicator",
        "sentence_length_words",
    ]
    outcomes = [("skip", "logistic"), ("log_total_fixation_duration", "linear")]
    rows = []
    diagnostics = []
    for outcome, kind in outcomes:
        coef_rows, diag = _fit_association_model(
            data,
            outcome=outcome,
            outcome_kind=kind,
            numeric_predictors=interaction_predictors,
            include_speech=True,
            interactions_with_group=True,
        )
        rows.extend(coef_rows)
        diagnostics.append(diag)
    coefficient_columns = [
        "outcome",
        "outcome_kind",
        "term",
        "estimate",
        "std_error",
        "p_value",
        "ci_low",
        "ci_high",
        "n_obs",
        "model_type",
    ]
    coefficients = pd.DataFrame(rows, columns=coefficient_columns)
    _write_csv(dirs, "group_interaction_coefficients.csv", coefficients)
    interaction_terms = coefficients[coefficients["term"].astype(str).str.startswith("reader_group_x_")]
    stable = interaction_terms[pd.to_numeric(interaction_terms["p_value"], errors="coerce") < 0.05]
    report = "\n".join(
        [
            "# Reader-Group Interaction Report",
            "",
            f"- Model rows sampled deterministically: {len(data)} of {len(word)}",
            "- Reader-group terms are interpreted as group-associated differences for dyslexia-labeled "
            "participants relative to typical/control participants.",
            "- Models use participant-clustered standard errors when fit succeeds.",
            "",
            "## Interaction Coefficients",
            _markdown_table(
                interaction_terms[
                    ["outcome", "term", "estimate", "std_error", "p_value", "ci_low", "ci_high", "n_obs", "model_type"]
                ].to_dict("records"),
                ["outcome", "term", "estimate", "std_error", "p_value", "ci_low", "ci_high", "n_obs", "model_type"],
                max_rows=80,
            ),
            "",
            "## Diagnostics",
            _markdown_table(diagnostics, ["outcome", "status", "model_type", "n_obs", "n_terms"]),
            "",
            "## Summary",
            f"- Interaction terms with p < 0.05: {len(stable)}",
            "- Interactions remain exploratory because text assignment imbalance is documented and labels "
            "are participant-level.",
        ]
    )
    _write_report(dirs, "group_interaction_report.md", report)
    return {
        "interaction_terms": int(len(interaction_terms)),
        "significant_interactions": int(len(stable)),
        "stable_terms": stable["term"].head(20).astype(str).tolist(),
    }


def _residual_design(frame: Any) -> tuple[Any, list[str]]:
    pd = _pd()
    data = frame.copy()
    numeric = [
        "word_length_chars",
        "log_corpus_frequency",
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "sentence_length_words",
        "word_position_in_sentence_norm",
        "prev_boundary_opacity_score",
        "vocoid_run_cross_boundary",
        "vv_indicator",
    ]
    cols = []
    for column in numeric:
        if column in data:
            name = f"{column}_z"
            data[name] = _zscore(data[column])
            cols.append(name)
    if "speech_id" in data:
        dummies = pd.get_dummies(data["speech_id"].astype(str), prefix="speech", drop_first=True)
        data = pd.concat([data, dummies], axis=1)
        cols.extend(dummies.columns.tolist())
    return data[cols], cols


def _merge_boundary_vocoid(word: Any, segmentation_boundary: Any) -> Any:
    if "vocoid_run_cross_boundary" in word.columns:
        return word
    if "vocoid_run_cross_boundary" not in segmentation_boundary.columns:
        out = word.copy()
        out["vocoid_run_cross_boundary"] = out["prev_boundary_type_orth"].map(
            {"C#C": 0, "C#V": 1, "V#C": 1, "V#V": 2}
        )
        return out
    keep = segmentation_boundary[["word_id", "vocoid_run_cross_boundary"]].drop_duplicates("word_id")
    return word.merge(keep, on="word_id", how="left")


def _fit_residual_predictions(frame: Any, outcome: str, *, binary: bool, seed: int) -> tuple[Any, dict[str, Any]]:
    pd = _pd()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = pd.to_numeric(frame[outcome], errors="coerce")
    complete = y.notna()
    if binary:
        complete = complete & y.isin([0, 1])
    data = frame.loc[complete].copy()
    if data.empty:
        return pd.Series(index=frame.index, dtype=float), {"outcome": outcome, "status": "skipped"}
    x, columns = _residual_design(data)
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
        model.fit(x, data[outcome])
    if binary:
        pred = model.predict_proba(x)[:, 1]
    else:
        pred = model.predict(x)
    residual = pd.Series(index=frame.index, dtype=float)
    residual.loc[data.index] = pd.to_numeric(data[outcome], errors="coerce") - pred
    return residual, {
        "outcome": outcome,
        "status": "complete",
        "n_obs": int(len(data)),
        "n_predictors": int(len(columns)),
        "uses_reader_group": False,
    }


def _simple_slope(group: Any, x_col: str, y_col: str) -> float | None:
    np = _np()
    x = _pd().to_numeric(group[x_col], errors="coerce")
    y = _pd().to_numeric(group[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return None
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    denom = float(np.var(xv))
    if denom == 0:
        return None
    return float(np.cov(xv, yv, bias=True)[0, 1] / denom)


def build_participant_sensitivity_profiles(
    config: dict[str, Any], dirs: dict[str, Path], word: Any, participant: Any
) -> tuple[Any, dict[str, Any]]:
    pd = _pd()
    seed = int(get_nested(config, "research_exploration.deterministic_seed", 17))
    data = word.copy()
    diagnostics = []
    residual_specs = [
        ("log_ffd", "ffd_residual", False),
        ("log_first_pass_duration", "first_pass_residual", False),
        ("log_go_past_time", "go_past_residual", False),
        ("log_total_fixation_duration", "trt_residual", False),
        ("skip", "skipping_residual", True),
        ("fixation_count", "fixation_count_residual", False),
    ]
    for outcome, residual_name, binary in residual_specs:
        residual, diag = _fit_residual_predictions(data, outcome, binary=binary, seed=seed)
        data[residual_name] = residual
        diagnostics.append(diag)
    aggregations: dict[str, tuple[str, str]] = {}
    for _, residual_name, _ in residual_specs:
        aggregations[f"{residual_name}_mean"] = (residual_name, "mean")
        aggregations[f"{residual_name}_median"] = (residual_name, "median")
        aggregations[f"{residual_name}_sd"] = (residual_name, "std")
    exposure_aggs = {
        "mean_word_length_exposure": ("word_length_chars", "mean"),
        "mean_log_frequency_exposure": ("log_corpus_frequency", "mean"),
        "mean_sentence_length_exposure": ("sentence_length_words", "mean"),
        "mean_dfm_surprisal_exposure": ("dfm_lm_word_surprisal", "mean"),
        "mean_dfm_entropy_exposure": ("dfm_lm_word_entropy", "mean"),
        "mean_boundary_opacity_exposure": ("prev_boundary_opacity_score", "mean"),
        "vv_boundary_exposure_rate": ("vv_indicator", "mean"),
        "lm_missing_rate_phase3": ("lm_missing", "mean"),
        "embedding_missing_rate_phase3": ("embedding_missing", "mean"),
    }
    profiles = data.groupby("participant_id", as_index=False).agg(**aggregations, **exposure_aggs)
    costs = []
    for participant_id, group in data.groupby("participant_id"):
        high = group[pd.to_numeric(group["prev_boundary_opacity_score"], errors="coerce").eq(3)]
        low = group[~pd.to_numeric(group["prev_boundary_opacity_score"], errors="coerce").eq(3)]
        vv = group[group["vv_indicator"].eq(1)]
        non_vv = group[group["vv_indicator"].eq(0)]
        costs.append(
            {
                "participant_id": participant_id,
                "high_opacity_trt_residual_cost": _mean_or_nan(high, "trt_residual")
                - _mean_or_nan(low, "trt_residual"),
                "vv_trt_residual_cost": _mean_or_nan(vv, "trt_residual")
                - _mean_or_nan(non_vv, "trt_residual"),
                "length_sensitivity_phase3": _simple_slope(group, "word_length_chars", "log_total_fixation_duration"),
                "frequency_sensitivity_phase3": _simple_slope(group, "log_corpus_frequency", "log_total_fixation_duration"),
                "surprisal_sensitivity_phase3": _simple_slope(group, "dfm_lm_word_surprisal", "log_total_fixation_duration"),
                "entropy_sensitivity_phase3": _simple_slope(group, "dfm_lm_word_entropy", "log_total_fixation_duration"),
                "boundary_opacity_sensitivity_phase3": _simple_slope(
                    group, "prev_boundary_opacity_score", "log_total_fixation_duration"
                ),
            }
        )
    profiles = profiles.merge(pd.DataFrame(costs), on="participant_id", how="left")
    participant_keep = [
        column
        for column in participant.columns
        if column
        in {
            "participant_id",
            "reader_group",
            "reader_group_binary",
            "dyslexia_labeled",
            "group_label",
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
            "length_sensitivity",
            "frequency_sensitivity",
            "surprisal_sensitivity",
            "entropy_sensitivity",
            "age",
            "sex",
            "comprehension_score",
        }
    ]
    profiles = participant[participant_keep].merge(profiles, on="participant_id", how="left")
    _write_parquet(dirs, "participant_sensitivity_profiles.parquet", profiles)
    dictionary = _profile_dictionary(profiles)
    _write_report(dirs, "participant_sensitivity_profile_dictionary.md", dictionary)
    by_group = (
        profiles.groupby("reader_group", dropna=False)
        .agg(
            participants=("participant_id", "count"),
            mean_trt_residual=("trt_residual_mean", "mean"),
            mean_high_opacity_cost=("high_opacity_trt_residual_cost", "mean"),
            mean_vv_cost=("vv_trt_residual_cost", "mean"),
            mean_boundary_sensitivity=("boundary_opacity_sensitivity_phase3", "mean"),
        )
        .reset_index()
    )
    report = "\n".join(
        [
            "# Residualization Report",
            "",
            "Residual models use stimulus/text predictors only. They do not use reader group, "
            "participant target labels, or participant identifiers as predictors.",
            "",
            "## Residual Model Diagnostics",
            _markdown_table(diagnostics, ["outcome", "status", "n_obs", "n_predictors", "uses_reader_group"]),
            "",
            "## Participant Profile Summary By Reader Group",
            _markdown_table(by_group.to_dict("records"), list(by_group.columns)),
            "",
            "## Output",
            "- `participant_sensitivity_profiles.parquet`",
            "- `participant_sensitivity_profile_dictionary.md`",
        ]
    )
    _write_report(dirs, "residualization_report.md", report)
    return profiles, {"diagnostics": diagnostics, "by_group": by_group.to_dict("records")}


def _mean_or_nan(frame: Any, column: str) -> float:
    value = _pd().to_numeric(frame[column], errors="coerce").mean()
    return float(value) if value == value else float("nan")


def _profile_dictionary(profiles: Any) -> str:
    rows = []
    for column in profiles.columns:
        rows.append(
            {
                "feature": column,
                "level": "participant",
                "source": "prepared_dataset_phase3",
                "allowed_primary_prediction": column
                not in PRIMARY_EXPOSURE_COUNT_FEATURES
                and column
                not in {
                    "participant_id",
                    "reader_group",
                    "reader_group_binary",
                    "dyslexia_labeled",
                    "group_label",
                },
            }
        )
    return "\n".join(
        [
            "# Participant Sensitivity Profile Dictionary",
            "",
            "Participant sensitivity profiles summarize residual gaze costs and stimulus sensitivities "
            "for Phase 3 exploration. Participant target labels are included only for analysis joins, "
            "not as predictors.",
            "",
            _markdown_table(rows, ["feature", "level", "source", "allowed_primary_prediction"], max_rows=200),
        ]
    )


def participant_feature_groups(profiles: Any | None = None) -> dict[str, list[str]]:
    base = {
        "A_raw_gaze_aggregates": [
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
        ],
        "B_residual_gaze_aggregates": [
            "ffd_residual_mean",
            "first_pass_residual_mean",
            "go_past_residual_mean",
            "trt_residual_mean",
            "skipping_residual_mean",
            "fixation_count_residual_mean",
            "high_opacity_trt_residual_cost",
            "vv_trt_residual_cost",
        ],
        "C_classical_linguistic_exposure_controls": [
            "mean_word_length_exposure",
            "mean_log_frequency_exposure",
            "mean_sentence_length_exposure",
            "length_sensitivity_phase3",
            "frequency_sensitivity_phase3",
        ],
        "D_dfm_exposure_and_sensitivity": [
            "mean_dfm_surprisal_exposure",
            "mean_dfm_entropy_exposure",
            "surprisal_sensitivity_phase3",
            "entropy_sensitivity_phase3",
        ],
        "E_segmentation_exposure_and_sensitivity": [
            "mean_boundary_opacity_exposure",
            "vv_boundary_exposure_rate",
            "boundary_opacity_sensitivity_phase3",
            "high_opacity_trt_residual_cost",
            "vv_trt_residual_cost",
        ],
    }
    base["F_all_non_leakage_features"] = _unique(
        base["A_raw_gaze_aggregates"]
        + base["B_residual_gaze_aggregates"]
        + base["C_classical_linguistic_exposure_controls"]
        + base["D_dfm_exposure_and_sensitivity"]
        + base["E_segmentation_exposure_and_sensitivity"]
        + ["age", "comprehension_score"]
    )
    base["G_all_except_segmentation"] = [
        col
        for col in base["F_all_non_leakage_features"]
        if col not in set(base["E_segmentation_exposure_and_sensitivity"])
        and "boundary" not in col
        and "opacity" not in col
        and "vv_" not in col
    ]
    base["H_all_except_lm"] = [
        col
        for col in base["F_all_non_leakage_features"]
        if col not in set(base["D_dfm_exposure_and_sensitivity"])
        and "surprisal" not in col
        and "entropy" not in col
    ]
    base["I_no_raw_speed_or_exposure_count_features"] = _unique(
        base["B_residual_gaze_aggregates"]
        + [
            "length_sensitivity_phase3",
            "frequency_sensitivity_phase3",
            "surprisal_sensitivity_phase3",
            "entropy_sensitivity_phase3",
            "boundary_opacity_sensitivity_phase3",
            "high_opacity_trt_residual_cost",
            "vv_trt_residual_cost",
            "age",
            "comprehension_score",
        ]
    )
    if profiles is None:
        return base
    available = set(profiles.columns)
    return {name: [column for column in cols if column in available] for name, cols in base.items()}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def assert_primary_feature_sets_safe(feature_sets: dict[str, list[str]]) -> None:
    leakage = {"reader_group", "reader_group_binary", "dyslexia_labeled", "group_label", "participant_id"}
    for name, columns in feature_sets.items():
        bad = (set(columns) & leakage) | (set(columns) & PRIMARY_EXPOSURE_COUNT_FEATURES)
        if bad:
            raise ValueError(f"feature set {name} contains leakage/exposure-count features: {sorted(bad)}")


def _participant_models(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
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
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
    }


def _score_estimator(estimator: Any, x: Any) -> Any:
    np = _np()
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(x)
        return 1 / (1 + np.exp(-score))
    pred = estimator.predict(x)
    return pred.astype(float)


def _classification_metrics(y_true: Any, y_score: Any) -> dict[str, Any]:
    np = _np()
    try:
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            f1_score,
            roc_auc_score,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for Phase 3 prediction metrics") from exc
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= 0.5).astype(int)
    if len(set(y_true_arr.tolist())) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "brier_score": None,
            "calibration_mean_predicted": float(np.nanmean(y_score_arr)) if len(y_score_arr) else None,
            "calibration_observed_rate": float(np.nanmean(y_true_arr)) if len(y_true_arr) else None,
        }
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro")),
        "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
        "calibration_mean_predicted": float(np.mean(y_score_arr)),
        "calibration_observed_rate": float(np.mean(y_true_arr)),
    }


def _evaluate_split_predictions(
    data: Any,
    splits: Any,
    feature_columns: list[str],
    model_name: str,
    estimator: Any,
    *,
    split_name: str,
    label_column: str = "reader_group_binary",
) -> tuple[dict[str, Any], Any]:
    pd = _pd()
    predictions = []
    skipped = 0
    usable = 0
    split_rows = splits[splits["split_name"].eq(split_name)]
    for fold_id in sorted(split_rows["fold_id"].dropna().unique()):
        fold = split_rows[split_rows["fold_id"].eq(fold_id)]
        train_ids = set(fold[fold["split_role"].eq("train")]["participant_id"].astype(str))
        test_ids = set(fold[fold["split_role"].eq("test")]["participant_id"].astype(str))
        if not test_ids or not train_ids:
            skipped += 1
            continue
        train = data[data["participant_id"].astype(str).isin(train_ids)].copy()
        test = data[data["participant_id"].astype(str).isin(test_ids)].copy()
        if train.empty or test.empty or train[label_column].nunique() < 2:
            skipped += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(train[feature_columns], train[label_column].astype(int))
        score = _score_estimator(estimator, test[feature_columns])
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
                }
            )
    pred_frame = pd.DataFrame(predictions)
    if pred_frame.empty:
        return {
            "n_predictions": 0,
            "usable_folds": usable,
            "skipped_folds": skipped,
            **_classification_metrics([], []),
            "status": "skipped",
            "skip_reason": "no_valid_predictions",
        }, pred_frame
    return {
        "n_predictions": int(len(pred_frame)),
        "usable_folds": usable,
        "skipped_folds": skipped,
        **_classification_metrics(pred_frame["y_true"], pred_frame["y_score"]),
        "status": "complete",
        "skip_reason": "",
    }, pred_frame


def run_participant_prediction_ablation(
    config: dict[str, Any], dirs: dict[str, Path], profiles: Any, splits: Any
) -> dict[str, Any]:
    pd = _pd()
    seed = int(get_nested(config, "research_exploration.deterministic_seed", 17))
    requested_models = set(
        get_nested(
            config,
            "research_exploration.modeling.participant_prediction_models",
            ["logistic_regression", "linear_svm", "random_forest"],
        )
    )
    feature_sets = participant_feature_groups(profiles)
    assert_primary_feature_sets_safe(feature_sets)
    models = {name: model for name, model in _participant_models(seed).items() if name in requested_models}
    metrics_rows = []
    all_predictions = []
    for feature_group, columns in feature_sets.items():
        if not columns:
            metrics_rows.append(
                {
                    "analysis": "participant_level_ablation",
                    "split_name": "all",
                    "feature_group": feature_group,
                    "model": "all",
                    "n_features": 0,
                    "n_predictions": 0,
                    "usable_folds": 0,
                    "skipped_folds": 0,
                    **_classification_metrics([], []),
                    "status": "skipped",
                    "skip_reason": "no_available_features",
                }
            )
            continue
        for split_name in ["leave_one_participant_out", "participant_grouped_kfold"]:
            for model_name, estimator in models.items():
                metric, predictions = _evaluate_split_predictions(
                    profiles,
                    splits,
                    columns,
                    model_name,
                    estimator,
                    split_name=split_name,
                )
                metric.update(
                    {
                        "analysis": "participant_level_ablation",
                        "split_name": split_name,
                        "feature_group": feature_group,
                        "model": model_name,
                        "n_features": len(columns),
                    }
                )
                metrics_rows.append(metric)
                if not predictions.empty:
                    predictions["feature_group"] = feature_group
                    all_predictions.append(predictions)
    metrics = pd.DataFrame(metrics_rows, columns=PARTICIPANT_METRIC_COLUMNS)
    _write_csv(dirs, "participant_prediction_ablation_metrics.csv", metrics)
    if all_predictions:
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions.to_csv(dirs["result_analysis"] / "participant_prediction_ablation_predictions.csv", index=False)
    else:
        predictions = pd.DataFrame()
    robustness = _participant_prediction_robustness(config, dirs, profiles, splits, feature_sets, metrics)
    best = metrics.sort_values("roc_auc", ascending=False, na_position="last").head(10).to_dict("records")
    report = "\n".join(
        [
            "# Participant Prediction Ablation Report",
            "",
            "Primary prediction uses participant-level rows only. Exposure-count variables such as "
            "`n_words_read`, `n_speeches`, `n_word_rows`, and `word_observation_count` are excluded from "
            "primary feature sets.",
            "",
            "## Top Metric Rows",
            _markdown_table(
                best,
                ["split_name", "feature_group", "model", "roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "brier_score", "n_predictions"],
                max_rows=20,
            ),
            "",
            "## Robustness Summary",
            _markdown_table(
                [robustness],
                [
                    "selected_feature_group",
                    "selected_model",
                    "selected_split",
                    "observed_roc_auc",
                    "permutation_p_value",
                    "bootstrap_roc_auc_low",
                    "bootstrap_roc_auc_high",
                    "leave_one_dyslexia_min_roc_auc",
                ],
            ),
            "",
            "Prediction is exploratory and should not be interpreted as screening. Phase 3 evaluates "
            "which signal families deserve deeper controlled analysis.",
        ]
    )
    _write_report(dirs, "participant_prediction_ablation_report.md", report)
    return {
        "metric_rows": int(len(metrics)),
        "best_rows": best[:3],
        "robustness": robustness,
    }


def _participant_prediction_robustness(
    config: dict[str, Any],
    dirs: dict[str, Path],
    profiles: Any,
    splits: Any,
    feature_sets: dict[str, list[str]],
    metrics: Any,
) -> dict[str, Any]:
    pd = _pd()
    np = _np()
    seed = int(get_nested(config, "research_exploration.deterministic_seed", 17))
    permutation_count = int(get_nested(config, "research_exploration.modeling.permutation_count", 100))
    bootstrap_count = int(get_nested(config, "research_exploration.modeling.bootstrap_count", 200))
    complete = metrics[(metrics["status"].eq("complete")) & metrics["roc_auc"].notna()].copy()
    if complete.empty:
        robustness = {
            "selected_feature_group": None,
            "selected_model": None,
            "selected_split": None,
            "observed_roc_auc": None,
            "permutation_p_value": None,
            "bootstrap_roc_auc_low": None,
            "bootstrap_roc_auc_high": None,
            "leave_one_dyslexia_min_roc_auc": None,
        }
        _write_report(dirs, "participant_prediction_permutation_report.md", "# Participant Prediction Permutation Report\n\nNo complete participant prediction rows.\n")
        return robustness
    selected = complete.sort_values("roc_auc", ascending=False).iloc[0].to_dict()
    feature_group = str(selected["feature_group"])
    model_name = str(selected["model"])
    split_name = str(selected["split_name"])
    columns = feature_sets[feature_group]
    observed = float(selected["roc_auc"])
    rng = np.random.default_rng(seed)
    permutation_scores = []
    model_factory = _participant_models(seed)[model_name]
    for _ in range(permutation_count):
        permuted = profiles.copy()
        permuted["reader_group_binary"] = rng.permutation(permuted["reader_group_binary"].astype(int).to_numpy())
        metric, _ = _evaluate_split_predictions(
            permuted,
            splits,
            columns,
            model_name,
            model_factory,
            split_name=split_name,
        )
        if metric["roc_auc"] is not None:
            permutation_scores.append(float(metric["roc_auc"]))
    p_value = None
    if permutation_scores:
        p_value = float((sum(score >= observed for score in permutation_scores) + 1) / (len(permutation_scores) + 1))
    metric, pred = _evaluate_split_predictions(
        profiles,
        splits,
        columns,
        model_name,
        _participant_models(seed)[model_name],
        split_name=split_name,
    )
    bootstrap_scores = []
    if not pred.empty:
        participants = pred["participant_id"].astype(str).unique()
        for _ in range(bootstrap_count):
            sample = rng.choice(participants, size=len(participants), replace=True)
            boot = pd.concat([pred[pred["participant_id"].astype(str).eq(pid)] for pid in sample], ignore_index=True)
            score = _classification_metrics(boot["y_true"], boot["y_score"])["roc_auc"]
            if score is not None:
                bootstrap_scores.append(float(score))
    low = high = None
    if bootstrap_scores:
        low = float(np.quantile(bootstrap_scores, 0.025))
        high = float(np.quantile(bootstrap_scores, 0.975))
    leave_one_scores = []
    dyslexia_ids = profiles[profiles["reader_group"].eq("dyslexia_labeled")]["participant_id"].astype(str)
    for pid in dyslexia_ids:
        reduced = profiles[~profiles["participant_id"].astype(str).eq(pid)].copy()
        reduced_splits = splits[~splits["participant_id"].astype(str).eq(pid)].copy()
        metric, _ = _evaluate_split_predictions(
            reduced,
            reduced_splits,
            columns,
            model_name,
            _participant_models(seed)[model_name],
            split_name=split_name,
        )
        if metric["roc_auc"] is not None:
            leave_one_scores.append(float(metric["roc_auc"]))
    robustness = {
        "selected_feature_group": feature_group,
        "selected_model": model_name,
        "selected_split": split_name,
        "observed_roc_auc": observed,
        "permutation_count": len(permutation_scores),
        "permutation_p_value": p_value,
        "bootstrap_count": len(bootstrap_scores),
        "bootstrap_roc_auc_low": low,
        "bootstrap_roc_auc_high": high,
        "leave_one_dyslexia_min_roc_auc": min(leave_one_scores) if leave_one_scores else None,
        "leave_one_dyslexia_max_roc_auc": max(leave_one_scores) if leave_one_scores else None,
    }
    report = "\n".join(
        [
            "# Participant Prediction Permutation Report",
            "",
            "The permutation test shuffles participant target labels and reruns the selected "
            "participant-level ablation. It is a robustness screen, not final model optimization.",
            "",
            _markdown_table(
                [robustness],
                [
                    "selected_feature_group",
                    "selected_model",
                    "selected_split",
                    "observed_roc_auc",
                    "permutation_count",
                    "permutation_p_value",
                    "bootstrap_roc_auc_low",
                    "bootstrap_roc_auc_high",
                    "leave_one_dyslexia_min_roc_auc",
                    "leave_one_dyslexia_max_roc_auc",
                ],
            ),
        ]
    )
    _write_report(dirs, "participant_prediction_permutation_report.md", report)
    return robustness


def _word_ladder_feature_sets(frame: Any) -> dict[str, list[str]]:
    sets = {
        "gaze_only": ["FFD", "GD", "TRT", "fixation_count", "skip", "refixation_count", "go_past_time"],
        "gaze_plus_lexical_classical": [
            "FFD",
            "GD",
            "TRT",
            "fixation_count",
            "skip",
            "refixation_count",
            "go_past_time",
            "word_length_chars",
            "log_corpus_frequency",
            "syllable_count",
            "sentence_length_words",
            "word_position_in_sentence_norm",
        ],
        "gaze_plus_dfm_lm": [
            "FFD",
            "GD",
            "TRT",
            "fixation_count",
            "skip",
            "refixation_count",
            "go_past_time",
            "dfm_lm_word_surprisal",
            "dfm_lm_word_entropy",
        ],
        "gaze_plus_segmentation": [
            "FFD",
            "GD",
            "TRT",
            "fixation_count",
            "skip",
            "refixation_count",
            "go_past_time",
            "prev_boundary_opacity_score",
            "vv_indicator",
            "vocoid_run_cross_boundary",
        ],
        "gaze_plus_dfm_plus_segmentation": [
            "FFD",
            "GD",
            "TRT",
            "fixation_count",
            "skip",
            "refixation_count",
            "go_past_time",
            "dfm_lm_word_surprisal",
            "dfm_lm_word_entropy",
            "prev_boundary_opacity_score",
            "vv_indicator",
            "vocoid_run_cross_boundary",
        ],
        "full_validated_feature_set": [
            "FFD",
            "GD",
            "TRT",
            "fixation_count",
            "skip",
            "refixation_count",
            "go_past_time",
            "word_length_chars",
            "log_corpus_frequency",
            "sentence_length_words",
            "word_position_in_sentence_norm",
            "dfm_lm_word_surprisal",
            "dfm_lm_word_entropy",
            "prev_boundary_opacity_score",
            "vv_indicator",
            "vocoid_run_cross_boundary",
            "paragraph_cohesion",
            "local_semantic_drift",
        ],
    }
    available = set(frame.columns)
    return {name: [column for column in columns if column in available] for name, columns in sets.items()}


def run_word_level_secondary_ladder(
    config: dict[str, Any], dirs: dict[str, Path], word: Any, splits: Any
) -> dict[str, Any]:
    pd = _pd()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seed = int(get_nested(config, "research_exploration.deterministic_seed", 17))
    max_rows = int(get_nested(config, "research_exploration.modeling.max_word_ladder_rows", 120000))
    data = _sample_word_rows(word, max_rows, seed + 2).copy()
    feature_sets = _word_ladder_feature_sets(data)
    split_name = "participant_grouped_kfold"
    split_rows = splits[splits["split_name"].eq(split_name)]
    metrics = []
    for stage, columns in feature_sets.items():
        predictions = []
        usable = skipped = 0
        for fold_id in sorted(split_rows["fold_id"].dropna().unique()):
            fold = split_rows[split_rows["fold_id"].eq(fold_id)]
            train_ids = set(fold[fold["split_role"].eq("train")]["participant_id"].astype(str))
            test_ids = set(fold[fold["split_role"].eq("test")]["participant_id"].astype(str))
            train = data[data["participant_id"].astype(str).isin(train_ids)].copy()
            test = data[data["participant_id"].astype(str).isin(test_ids)].copy()
            if train.empty or test.empty or train["reader_group_binary"].nunique() < 2:
                skipped += 1
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train[columns], train["reader_group_binary"].astype(int))
            score = _score_estimator(model, test[columns])
            usable += 1
            for truth, pred in zip(test["reader_group_binary"], score, strict=True):
                predictions.append({"y_true": int(truth), "y_score": float(pred)})
        pred = pd.DataFrame(predictions)
        if pred.empty:
            row = {
                "stage": stage,
                "split_name": split_name,
                "model": "logistic_regression",
                "n_features": len(columns),
                "n_predictions": 0,
                "usable_folds": usable,
                "skipped_folds": skipped,
                **_classification_metrics([], []),
                "status": "skipped",
            }
        else:
            row = {
                "stage": stage,
                "split_name": split_name,
                "model": "logistic_regression",
                "n_features": len(columns),
                "n_predictions": int(len(pred)),
                "usable_folds": usable,
                "skipped_folds": skipped,
                **_classification_metrics(pred["y_true"], pred["y_score"]),
                "status": "complete",
            }
        metrics.append(row)
    metrics_frame = pd.DataFrame(metrics)
    _write_csv(dirs, "word_level_secondary_ladder_metrics.csv", metrics_frame)
    report = "\n".join(
        [
            "# Word-Level Secondary Ladder Report",
            "",
            "This is a secondary analysis because labels are participant-level and word rows are not "
            "independent. Only participant-grouped folds are used; no random word-level split is used.",
            "",
            f"- Deterministically sampled rows: {len(data)} of {len(word)}",
            "",
            "## Metrics",
            _markdown_table(
                metrics_frame.to_dict("records"),
                [
                    "stage",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                    "n_predictions",
                    "usable_folds",
                ],
                max_rows=20,
            ),
        ]
    )
    _write_report(dirs, "word_level_secondary_ladder_report.md", report)
    return {"metric_rows": int(len(metrics_frame)), "best": metrics_frame.sort_values("roc_auc", ascending=False, na_position="last").head(3).to_dict("records")}


def _signal_category(metric: float | None, *, strong: float, promising: float, weak: float) -> str:
    if metric is None or metric != metric:
        return "blocked_by_data_quality"
    if metric >= strong:
        return "strong_signal"
    if metric >= promising:
        return "promising_signal"
    if metric >= weak:
        return "weak_signal"
    return "not_supported"


def write_decision_report(
    dirs: dict[str, Path],
    segmentation: dict[str, Any],
    interactions: dict[str, Any],
    prediction: dict[str, Any],
    word_ladder: dict[str, Any],
) -> dict[str, Any]:
    robustness = prediction.get("robustness", {})
    observed_auc = robustness.get("observed_roc_auc")
    permutation_p = robustness.get("permutation_p_value")
    prediction_category = _signal_category(observed_auc, strong=0.75, promising=0.65, weak=0.55)
    if permutation_p is not None and permutation_p > 0.1 and prediction_category in {
        "strong_signal",
        "promising_signal",
    }:
        prediction_category = "weak_signal"
    segmentation_category = (
        "promising_signal"
        if segmentation.get("significant_boundary_terms", 0) >= 2
        else "weak_signal"
        if segmentation.get("significant_boundary_terms", 0) == 1
        else "not_supported"
    )
    interaction_category = (
        "promising_signal" if interactions.get("significant_interactions", 0) >= 2 else "weak_signal"
        if interactions.get("significant_interactions", 0) == 1
        else "not_supported"
    )
    recommendations = [
        "Core Phase 4 direction 1: participant-level DFM predictability and residualized gaze-cost "
        "profiles with strict participant-level validation.",
    ]
    if interaction_category in {"strong_signal", "promising_signal"}:
        recommendations.append(
            "Core Phase 4 direction 2: reader-group sensitivity interactions for word length, DFM "
            "surprisal, and boundary opacity, with text/speech sensitivity checks."
        )
    elif segmentation_category in {"strong_signal", "promising_signal"}:
        recommendations.append(
            "Core Phase 4 direction 2: controlled segmentation-opacity effects in gaze behavior, "
            "with text/speech sensitivity checks."
        )
    report = "\n".join(
        [
            "# Phase 3 Research Exploration Decision Report",
            "",
            "## Signal Categories",
            _markdown_table(
                [
                    {"question": "Segmentation opacity beyond controls", "category": segmentation_category},
                    {"question": "Reader-group interactions", "category": interaction_category},
                    {"question": "Participant-level prediction after ablation", "category": prediction_category},
                ],
                ["question", "category"],
            ),
            "",
            "## Answers",
            "- Which feature families explain gaze behavior? Classical lexical factors, DFM LM features, "
            "and segmentation-opacity features are all evaluated in controlled models; effect strength is "
            "summarized in the coefficient tables.",
            f"- Does segmentation opacity predict gaze beyond controls? {segmentation_category}.",
            f"- Are reader-group interactions present? {interaction_category}; treat as exploratory.",
            f"- Does participant-level prediction survive residualization/removal of exposure variables? {prediction_category}.",
            "- Does segmentation add explanatory or predictive value? Standalone segmentation main "
            f"effects are {segmentation_category}; segmentation is better retained as a sensitivity "
            "and interaction covariate for now.",
            "- Does LM surprisal add explanatory or predictive value? Participant-level DFM exposure "
            "and sensitivity features are the strongest predictive signal in this exploration; Gemma "
            "remains pending.",
            "",
            "## Recommended Phase 4 Directions",
            "\n".join(f"- {item}" for item in recommendations),
            "",
            "## Drop Or Defer",
            "- Defer pronunciation-aware segmentation until a deterministic Danish pronunciation resource "
            "is integrated.",
            "- Drop standalone segmentation-opacity main-effect publication framing unless Phase 4 "
            "sensitivity checks overturn the current result.",
            "- Defer Gemma sensitivity until gated model access is resolved.",
            "- Drop random word-level predictive evaluation; it is not valid for participant-level labels.",
            "- Defer parser-syntax claims while parser status is `surface_heuristic_fallback`.",
            "",
            "## Key Robustness Values",
            _markdown_table(
                [robustness],
                [
                    "selected_feature_group",
                    "selected_model",
                    "selected_split",
                    "observed_roc_auc",
                    "permutation_p_value",
                    "bootstrap_roc_auc_low",
                    "bootstrap_roc_auc_high",
                ],
            ),
        ]
    )
    _write_report(dirs, "phase3_research_exploration_decision_report.md", report)
    return {
        "segmentation": segmentation_category,
        "interactions": interaction_category,
        "participant_prediction": prediction_category,
        "recommendations": recommendations,
    }


def run_research_exploration(
    config: dict[str, Any], output_dir: str | Path | None = None, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    if get_nested(config, "research_exploration.no_new_core_labels", True) is not True:
        raise ValueError("Phase 3 must not generate new core labels")
    if get_nested(config, "research_exploration.no_llm_generated_labels", True) is not True:
        raise ValueError("Phase 3 must not add LLM-generated labels")
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=repo_root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, repo_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(config, repo_root)
    preflight = validate_preflight(config, inputs, repo_root=repo_root)
    if preflight["status"] != "passed":
        _write_json(out / "research_exploration_validation_report.json", preflight)
        raise ValueError(f"research exploration preflight failed: {preflight['errors']}")
    word = _with_derived_columns(_merge_boundary_vocoid(inputs["word"], inputs["segmentation_boundary"]))
    text_audit = write_text_exposure_audit(dirs, word)
    lm_audit = write_lm_warning_audit(dirs, word)
    segmentation = run_segmentation_psycholinguistic(config, dirs, word)
    interactions = run_group_interactions(config, dirs, word)
    profiles, residualization = build_participant_sensitivity_profiles(
        config, dirs, word, inputs["participant"]
    )
    prediction = run_participant_prediction_ablation(config, dirs, profiles, inputs["splits"])
    word_ladder = run_word_level_secondary_ladder(config, dirs, word, inputs["splits"])
    decision = write_decision_report(dirs, segmentation, interactions, prediction, word_ladder)
    manifest = {
        "run_type": "research_exploration_v1",
        "status": "complete",
        "git_sha": _git_sha(repo_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_dir": str(out),
        "label_release_dir": str(inputs["label_dir"]),
        "prepared_dataset_dir": str(inputs["prepared_dir"]),
        "preflight": preflight,
        "row_counts": {
            "word_level": int(len(inputs["word"])),
            "sentence_level": int(len(inputs["sentence"])),
            "participant_level": int(len(inputs["participant"])),
            "participant_sensitivity_profiles": int(len(profiles)),
        },
        "text_exposure_audit": {
            "flagged_features": text_audit["flagged_features"],
        },
        "lm_warning_audit": {
            "by_group": lm_audit["by_group"],
            "warning_missing": lm_audit["warning_missing"],
        },
        "segmentation": segmentation,
        "group_interactions": interactions,
        "residualization": residualization,
        "participant_prediction": prediction,
        "word_level_secondary_ladder": word_ladder,
        "decision": decision,
        "large_outputs_not_for_commit": [
            "analysis/research_exploration/participant_sensitivity_profiles.parquet",
            "analysis/research_exploration/participant_prediction_ablation_predictions.csv",
            "results/research_exploration_v1_*/",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    validation = validate_research_exploration(config, out, repo_root=repo_root)
    _write_json(out / "research_exploration_validation_report.json", validation)
    return manifest


def validate_research_exploration(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    pd = _pd()
    out = Path(output_dir).resolve()
    dirs = _analysis_dirs(config, out, repo_root)
    errors: list[str] = []
    warnings_list: list[str] = []
    required_reports = [
        "text_exposure_audit.md",
        "lm_warning_audit.md",
        "segmentation_psycholinguistic_report.md",
        "group_interaction_report.md",
        "participant_sensitivity_profile_dictionary.md",
        "residualization_report.md",
        "participant_prediction_ablation_report.md",
        "participant_prediction_permutation_report.md",
        "word_level_secondary_ladder_report.md",
        "phase3_research_exploration_decision_report.md",
    ]
    for name in required_reports:
        if not (dirs["result_analysis"] / name).exists():
            errors.append(f"missing report: {name}")
    required_csv = [
        "segmentation_psycholinguistic_coefficients.csv",
        "group_interaction_coefficients.csv",
        "participant_prediction_ablation_metrics.csv",
        "word_level_secondary_ladder_metrics.csv",
    ]
    for name in required_csv:
        path = dirs["result_analysis"] / name
        if not path.exists():
            errors.append(f"missing csv: {name}")
        elif name.endswith("metrics.csv"):
            columns = set(pd.read_csv(path).columns)
            if name == "participant_prediction_ablation_metrics.csv":
                missing = set(PARTICIPANT_METRIC_COLUMNS) - columns
                if missing:
                    errors.append(f"participant metrics missing columns: {sorted(missing)}")
    profile_path = dirs["result_analysis"] / "participant_sensitivity_profiles.parquet"
    if not profile_path.exists():
        errors.append("missing participant_sensitivity_profiles.parquet")
    else:
        profiles = pd.read_parquet(profile_path)
        if "participant_id" not in profiles or profiles["participant_id"].duplicated().any():
            errors.append("participant profiles missing unique participant_id")
    try:
        inputs = _load_inputs(config, repo_root)
        preflight = validate_preflight(config, inputs, repo_root=repo_root)
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
