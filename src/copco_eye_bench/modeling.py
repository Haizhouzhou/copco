"""Predictive modeling for dyslexia-labeled reader experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import get_nested


FAMILY_FEATURES = {
    "gaze": ["FFD", "GD", "TRT", "fixation_count", "skip", "refixation_count", "go_past_time"],
    "classical_linguistic": [
        "word_length_chars",
        "syllable_count",
        "word_index_in_sentence",
        "word_index_in_paragraph",
        "word_position_in_sentence_norm",
        "word_position_in_paragraph_norm",
        "sentence_length_words",
        "paragraph_length_words",
    ],
    "parser_morphosyntax": [
        "is_function_word",
        "is_noun_or_proper",
        "is_pronoun",
        "is_finite_verb",
        "is_negation",
        "dependency_distance",
    ],
}
EMPTY_METRICS: dict[str, float | None] = {"roc_auc": None, "pr_auc": None, "brier": None}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for modeling") from exc
    return pd


def _load_feature_frame(output_dir: Path) -> Any:
    pd = _require_pandas()
    for candidate in (
        output_dir / "modeling_tables" / "word_level_full_with_all_lm.parquet",
        output_dir / "modeling_tables" / "word_level_full_with_dfm_lm.parquet",
        output_dir / "modeling_tables" / "word_level_full.parquet",
    ):
        if candidate.exists():
            return pd.read_parquet(candidate)

    base_path = output_dir / "tables" / "word_observations.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"missing feature table: {base_path}")
    frame = pd.read_parquet(base_path)
    for path in sorted((output_dir / "lm_features").glob("**/*.parquet")):
        extra = pd.read_parquet(path)
        if "word_id" in extra.columns:
            frame = frame.merge(extra, on="word_id", how="left", suffixes=("", "_lm"))
        elif "sentence_id" in extra.columns:
            frame = frame.merge(extra, on="sentence_id", how="left", suffixes=("", "_lm"))
        elif "paragraph_id" in extra.columns:
            frame = frame.merge(extra, on="paragraph_id", how="left", suffixes=("", "_lm"))
    return frame


def _feature_columns(frame: Any, families: list[str]) -> list[str]:
    selected: list[str] = []
    for family in families:
        selected.extend(FAMILY_FEATURES.get(family, []))
        if family == "lm_surprisal_entropy":
            selected.extend(
                column
                for column in frame.columns
                if "surprisal" in column or "entropy" in column
            )
        elif family == "embeddings":
            selected.extend(
                column
                for column in frame.columns
                if column.startswith("embedding_")
                or "cohesion" in column
                or "centroid" in column
                or "semantic_drift" in column
            )
        elif family == "instruct_annotations":
            selected.extend(
                column
                for column in frame.columns
                if column.startswith("instruct_") or column.startswith("annotation_")
            )
    numeric = []
    for column in dict.fromkeys(selected):
        if column in frame.columns:
            numeric.append(column)
    return numeric


def _aggregate_features(frame: Any, by: list[str], feature_columns: list[str]) -> Any:
    pd = _require_pandas()
    keep = [*by, "dyslexia_labeled", *feature_columns]
    data = frame[keep].copy()
    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    grouped = data.groupby(by, dropna=False)
    means = grouped[feature_columns].mean().add_suffix("__mean")
    stds = grouped[feature_columns].std().add_suffix("__sd")
    labels = grouped["dyslexia_labeled"].first()
    aggregated = means.join(stds).join(labels).reset_index()
    aggregated["dyslexia_labeled"] = aggregated["dyslexia_labeled"].astype(int)
    return aggregated


def _classifiers(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    classifiers: dict[str, Any] = {
        "l1_logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                random_state=seed,
                max_iter=1000,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                min_samples_leaf=2,
            ),
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        classifiers["lightgbm"] = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=seed,
                verbose=-1,
            ),
        )
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier

        classifiers["xgboost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                eval_metric="logloss",
                random_state=seed,
                n_estimators=300,
                max_depth=3,
            ),
        )
    except Exception:
        pass
    return classifiers


def _positive_prob(estimator: Any, x_test: Any) -> Any:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        import numpy as np

        score = estimator.decision_function(x_test)
        return 1.0 / (1.0 + np.exp(-score))
    return estimator.predict(x_test)


def _metric_row(y_true: Any, y_score: Any) -> dict[str, float | None]:
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for model metrics") from exc

    if len(set(map(int, y_true))) < 2:
        return {"roc_auc": None, "pr_auc": None, "brier": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def _evaluate_participant_splits(
    data: Any,
    split_table: Any,
    feature_columns: list[str],
    classifier_name: str,
    classifier: Any,
    *,
    regime: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pd = _require_pandas()
    prediction_rows: list[dict[str, Any]] = []
    for fold in sorted(split_table["fold"].unique()):
        fold_rows = split_table[split_table["fold"] == fold]
        train_ids = set(fold_rows[fold_rows["split"] == "train"]["participant_id"].astype(str))
        test_ids = set(fold_rows[fold_rows["split"] == "test"]["participant_id"].astype(str))
        train = data[data["participant_id"].astype(str).isin(train_ids)].copy()
        test = data[data["participant_id"].astype(str).isin(test_ids)].copy()
        if train.empty or test.empty or train["dyslexia_labeled"].nunique() < 2:
            continue
        classifier.fit(train[feature_columns], train["dyslexia_labeled"])
        scores = _positive_prob(classifier, test[feature_columns])
        for row, score in zip(test.itertuples(index=False), scores, strict=True):
            prediction_rows.append(
                {
                    "cv_regime": regime,
                    "fold": int(fold),
                    "classifier": classifier_name,
                    "participant_id": row.participant_id,
                    "y_true": int(row.dyslexia_labeled),
                    "y_score": float(score),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        return {**EMPTY_METRICS, "status": "skipped_no_predictions"}, []
    metrics = _metric_row(predictions["y_true"], predictions["y_score"])
    metrics["status"] = "complete"
    return metrics, prediction_rows


def _evaluate_loso(
    data: Any,
    split_table: Any,
    feature_columns: list[str],
    classifier_name: str,
    classifier: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pd = _require_pandas()
    prediction_rows: list[dict[str, Any]] = []
    for fold in sorted(split_table["fold"].unique()):
        fold_rows = split_table[split_table["fold"] == fold]
        train_speeches = set(fold_rows[fold_rows["split"] == "train"]["speech_id"].astype(str))
        test_speeches = set(fold_rows[fold_rows["split"] == "test"]["speech_id"].astype(str))
        train = data[data["speech_id"].astype(str).isin(train_speeches)].copy()
        test = data[data["speech_id"].astype(str).isin(test_speeches)].copy()
        if train.empty or test.empty or train["dyslexia_labeled"].nunique() < 2:
            continue
        classifier.fit(train[feature_columns], train["dyslexia_labeled"])
        scores = _positive_prob(classifier, test[feature_columns])
        for row, score in zip(test.itertuples(index=False), scores, strict=True):
            prediction_rows.append(
                {
                    "cv_regime": "leave_one_speech_out",
                    "fold": int(fold),
                    "classifier": classifier_name,
                    "participant_id": row.participant_id,
                    "speech_id": row.speech_id,
                    "y_true": int(row.dyslexia_labeled),
                    "y_score": float(score),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        return {**EMPTY_METRICS, "status": "skipped_no_predictions"}, []
    averaged = predictions.groupby("participant_id").agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
    metrics = _metric_row(averaged["y_true"], averaged["y_score"])
    metrics["status"] = "complete"
    return metrics, prediction_rows


def run_models(config: dict[str, Any], output_dir: str | Path, *, seed: int | None = None) -> dict[str, Any]:
    """Run Models A-F where required feature families are present."""

    pd = _require_pandas()
    out = Path(output_dir).resolve()
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_feature_frame(out)
    frame = frame.dropna(subset=["dyslexia_labeled"]).copy()
    if frame.empty:
        manifest = {
            "run_type": "models",
            "status": "skipped",
            "reason": "missing_labels_after_sampling",
            "message": "Model smoke skipped: no labeled participants present after sampling",
        }
        _write_json(model_dir / "manifest.json", manifest)
        return manifest
    if frame["dyslexia_labeled"].nunique() < 2:
        label_counts = frame["dyslexia_labeled"].value_counts(dropna=False).to_dict()
        manifest = {
            "run_type": "models",
            "status": "skipped",
            "reason": "only_one_class_after_sampling",
            "message": "Model smoke skipped: only one class present after sampling",
            "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        }
        _write_json(model_dir / "manifest.json", manifest)
        return manifest
    frame["dyslexia_labeled"] = frame["dyslexia_labeled"].astype(int)

    model_families = get_nested(config, "models.families", {})
    enabled_families = get_nested(config, "models.enabled_families")
    if enabled_families:
        enabled = [str(name) for name in enabled_families]
        model_families = {name: model_families[name] for name in enabled if name in model_families}
    allow_leave_one_speech_out = bool(get_nested(config, "models.allow_leave_one_speech_out", True))
    random_seed = int(seed or get_nested(config, "cv.random_seeds", [17])[0])
    split_dir = out / "splits"
    participant_split = pd.read_csv(split_dir / "participant_grouped_folds.csv")
    lopo_split = pd.read_csv(split_dir / "leave_one_participant_out.csv")
    loso_split = pd.read_csv(split_dir / "leave_one_speech_out.csv")

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for model_name, families in model_families.items():
        features = _feature_columns(frame, list(families))
        if not features:
            skipped[model_name] = "no_available_features_for_family"
            continue
        participant_data = _aggregate_features(frame, ["participant_id"], features)
        speech_data = _aggregate_features(frame, ["participant_id", "speech_id"], features)
        feature_columns = [
            column
            for column in participant_data.columns
            if column.endswith("__mean") or column.endswith("__sd")
        ]
        if not feature_columns:
            skipped[model_name] = "no_numeric_aggregated_features"
            continue
        speech_feature_columns = [
            column for column in speech_data.columns if column.endswith("__mean") or column.endswith("__sd")
        ]

        for classifier_name, classifier in _classifiers(random_seed).items():
            for regime, split_table in (
                ("participant_grouped_5fold", participant_split),
                ("leave_one_participant_out", lopo_split),
            ):
                metrics, rows = _evaluate_participant_splits(
                    participant_data,
                    split_table,
                    feature_columns,
                    classifier_name,
                    classifier,
                    regime=regime,
                )
                metric_rows.append(
                    {
                        "model": model_name,
                        "classifier": classifier_name,
                        "cv_regime": regime,
                        "feature_count": len(feature_columns),
                        **metrics,
                    }
                )
                for row in rows:
                    prediction_rows.append({"model": model_name, **row})

            if allow_leave_one_speech_out:
                metrics, rows = _evaluate_loso(
                    speech_data,
                    loso_split,
                    speech_feature_columns,
                    classifier_name,
                    classifier,
                )
                metric_rows.append(
                    {
                        "model": model_name,
                        "classifier": classifier_name,
                        "cv_regime": "leave_one_speech_out",
                        "feature_count": len(speech_feature_columns),
                        **metrics,
                    }
                )
                for row in rows:
                    prediction_rows.append({"model": model_name, **row})

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(model_dir / "model_metrics.csv", index=False)
    predictions.to_csv(model_dir / "model_predictions.csv", index=False)
    sequence_manifest = {
        "enabled": bool(get_nested(config, "models.sequence_models.enabled", True)),
        "claim_scope": "exploratory_only",
        "status": "not_run_by_default_due_small_n",
    }
    _write_json(model_dir / "sequence_models_exploratory.json", sequence_manifest)
    manifest = {
        "run_type": "models",
        "status": "complete",
        "random_seed": random_seed,
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "skipped_models": skipped,
        "allow_leave_one_speech_out": allow_leave_one_speech_out,
        "sequence_models": sequence_manifest,
    }
    _write_json(model_dir / "manifest.json", manifest)
    return manifest
