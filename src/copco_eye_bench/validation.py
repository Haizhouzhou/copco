"""Run-artifact validation for CopCo pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .splits import assert_no_group_leakage


REQUIRED_TABLE_COLUMNS = {
    "words.parquet": {"word_id", "speech_id", "paragraph_id", "sentence_id", "word_form"},
    "sentences.parquet": {"sentence_id", "paragraph_id", "sentence_text"},
    "paragraphs.parquet": {"paragraph_id", "speech_id", "paragraph_text"},
    "participants.parquet": {"participant_id", "dyslexia_labeled"},
    "word_observations.parquet": {"participant_id", "word_id", "speech_id", "TRT"},
}

REQUIRED_METRIC_COLUMNS = {"model", "classifier", "cv_regime", "roc_auc", "pr_auc", "brier"}
REQUIRED_LM_SURPRISAL_COLUMNS = {
    "word_id",
    "speech_id",
    "paragraph_id",
    "sentence_id",
    "lm_model_id",
    "lm_tokenizer_id",
    "lm_context_mode",
    "lm_context_tokens",
    "lm_word_surprisal",
    "lm_word_entropy",
    "lm_subword_count",
    "lm_alignment_status",
    "lm_alignment_warning",
    "lm_alignment_error",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_metrics_schema(frame: Any) -> None:
    missing = sorted(REQUIRED_METRIC_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"model metrics missing columns: {missing}")


def _validate_tables(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import pandas as pd

    table_dir = output_dir / "tables"
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for filename, required in REQUIRED_TABLE_COLUMNS.items():
        path = table_dir / filename
        if not path.exists():
            errors.append(f"missing_table:{filename}")
            continue
        frame = pd.read_parquet(path)
        missing = sorted(required.difference(frame.columns))
        duplicate_count = 0
        if filename == "word_observations.parquet" and {"participant_id", "word_id"} <= set(frame.columns):
            duplicate_count = int(frame.duplicated(["participant_id", "word_id"]).sum())
            if duplicate_count:
                errors.append(f"duplicate_participant_word:{duplicate_count}")
        if filename == "words.parquet" and "word_id" in frame.columns:
            duplicate_count = int(frame.duplicated(["word_id"]).sum())
            if duplicate_count:
                errors.append(f"duplicate_word_id:{duplicate_count}")
        if missing:
            errors.append(f"{filename}:missing_columns:{missing}")
        reports.append(
            {
                "table": filename,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "missing_required_columns": missing,
                "duplicate_count": duplicate_count,
            }
        )
    return reports, errors


def _validate_splits(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import pandas as pd

    split_dir = output_dir / "splits"
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    manifest_path = output_dir / "manifest.json"
    split_skipped = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_skipped = isinstance(manifest.get("split_tables"), dict) and "skipped" in manifest[
            "split_tables"
        ]
    participant_path = split_dir / "participant_grouped_folds.csv"
    if participant_path.exists():
        splits = pd.read_csv(participant_path)
        for fold in sorted(splits["fold"].unique()):
            fold_rows = splits[splits["fold"] == fold]
            train = fold_rows[fold_rows["split"] == "train"]
            test = fold_rows[fold_rows["split"] == "test"]
            try:
                assert_no_group_leakage(train, test, ["participant_id"])
            except ValueError as exc:
                errors.append(f"participant_grouped_folds:fold_{fold}:{exc}")
        reports.append({"split_table": participant_path.name, "rows": int(len(splits))})
    elif split_skipped:
        reports.append({"split_table": participant_path.name, "status": "skipped_by_manifest"})
    else:
        errors.append("missing_split_table:participant_grouped_folds.csv")

    loso_path = split_dir / "leave_one_speech_out.csv"
    if loso_path.exists():
        splits = pd.read_csv(loso_path)
        for fold in sorted(splits["fold"].unique()):
            fold_rows = splits[splits["fold"] == fold]
            train = fold_rows[fold_rows["split"] == "train"]
            test = fold_rows[fold_rows["split"] == "test"]
            try:
                assert_no_group_leakage(train, test, ["speech_id"])
            except ValueError as exc:
                errors.append(f"leave_one_speech_out:fold_{fold}:{exc}")
        reports.append({"split_table": loso_path.name, "rows": int(len(splits))})
    return reports, errors


def _validate_metrics(output_dir: Path) -> tuple[dict[str, Any], list[str]]:
    import pandas as pd

    path = output_dir / "models" / "model_metrics.csv"
    if not path.exists():
        return {"available": False, "rows": 0}, []
    frame = pd.read_csv(path)
    try:
        validate_metrics_schema(frame)
    except ValueError as exc:
        return {"available": True, "rows": int(len(frame))}, [str(exc)]
    return {"available": True, "rows": int(len(frame))}, []


def _validate_lm_features(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import pandas as pd

    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    lm_root = output_dir / "lm_features"
    if not lm_root.exists():
        return reports, errors

    for path in sorted(lm_root.glob("**/*.parquet")):
        frame = pd.read_parquet(path)
        missing = sorted(REQUIRED_LM_SURPRISAL_COLUMNS.difference(frame.columns))
        duplicate_count = 0
        if "lm_word_surprisal" in frame.columns:
            if missing:
                errors.append(f"{path.relative_to(output_dir)}:missing_columns:{missing}")
            if "word_id" in frame.columns:
                key = ["word_id"]
                if "participant_id" in frame.columns:
                    key = ["participant_id", "word_id"]
                duplicate_count = int(frame.duplicated(key).sum())
                if duplicate_count:
                    errors.append(f"{path.relative_to(output_dir)}:duplicate_lm_word_keys:{duplicate_count}")
            for column in ("word_id", "speech_id", "paragraph_id", "sentence_id"):
                if column in frame.columns and bool(frame[column].isna().any()):
                    errors.append(f"{path.relative_to(output_dir)}:missing_stable_ids:{column}")
            if "lm_subword_count" in frame.columns and bool((frame["lm_subword_count"] <= 0).any()):
                errors.append(f"{path.relative_to(output_dir)}:zero_subword_words")
            if "lm_alignment_status" in frame.columns:
                bad = frame[~frame["lm_alignment_status"].isin(["ok", "warning"])]
                if not bad.empty:
                    errors.append(f"{path.relative_to(output_dir)}:lm_alignment_failed:{len(bad)}")
        reports.append(
            {
                "lm_feature_file": str(path.relative_to(output_dir)),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "missing_required_columns": missing if "lm_word_surprisal" in frame.columns else [],
                "duplicate_count": duplicate_count,
            }
        )

    for path in sorted(lm_root.glob("**/alignment_report_shard*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            errors.append(f"{path.relative_to(output_dir)}:alignment_report_failed")
        reports.append(
            {
                "lm_alignment_report": str(path.relative_to(output_dir)),
                "status": payload.get("status"),
                "contexts": len(payload.get("reports", [])),
            }
        )
    return reports, errors


def validate_run(output_dir: str | Path) -> dict[str, Any]:
    """Validate schemas, duplicate keys, leakage controls, metrics, and manifests."""

    out = Path(output_dir).resolve()
    errors: list[str] = []
    table_reports, table_errors = _validate_tables(out)
    split_reports, split_errors = _validate_splits(out)
    metric_report, metric_errors = _validate_metrics(out)
    lm_reports, lm_errors = _validate_lm_features(out)
    errors.extend(table_errors)
    errors.extend(split_errors)
    errors.extend(metric_errors)
    errors.extend(lm_errors)
    manifest_path = out / "manifest.json"
    manifest_available = manifest_path.exists()
    if not manifest_available:
        errors.append("missing_manifest:manifest.json")
    report = {
        "run_type": "validate_run",
        "output_dir": str(out),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "tables": table_reports,
        "splits": split_reports,
        "metrics": metric_report,
        "lm_features": lm_reports,
        "manifest_available": manifest_available,
    }
    _write_json(out / "validation_report.json", report)
    return report
