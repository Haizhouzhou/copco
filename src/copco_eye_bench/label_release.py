"""Label Release v1.1 and prepared-dataset freeze utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir
from .splits import leave_one_participant_out, participant_grouped_folds


DANISH_VOWELS = frozenset("aeiouyæøåAEIOUYÆØÅ")
ALLOWED_READER_GROUPS = {"dyslexia_labeled", "typical_control", "uncertain"}
ALLOWED_DIAGNOSTIC_PROVENANCE = {
    "formal_diagnosis",
    "self_report",
    "screening_result",
    "project_metadata",
    "unknown",
    "not_available",
}
ALLOWED_LABEL_CONFIDENCE = {"high", "medium", "low", "unknown"}

RELEASE_VERSION = "v1.1"
SEGMENTATION_VERSION = "segmentation_orthographic_v1"


def _pd() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for label-release operations") from exc
    return pd


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for label-release operations") from exc
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


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksums(root: Path, path: Path, *, max_bytes: int) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file == path:
            continue
        size = file.stat().st_size
        entry: dict[str, Any] = {
            "path": str(file.relative_to(root)),
            "bytes": int(size),
            "sha256": None,
            "skipped": False,
        }
        if size <= max_bytes:
            entry["sha256"] = _sha256(file)
        else:
            entry["skipped"] = True
            entry["skip_reason"] = f"larger_than_{max_bytes}_bytes"
        files.append(entry)
    payload = {"root": str(root), "files": files}
    _write_json(path, payload)
    return payload


def _source_dir(config: dict[str, Any], repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    source = Path(str(get_nested(config, "label_release.source_feature_release_dir")))
    if not source.is_absolute():
        source = root / source
    return source.resolve()


def _label_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    result_analysis = out / str(get_nested(config, "label_release.output_layout.analysis", "analysis/label_analysis"))
    repo_analysis = root / str(get_nested(config, "label_release.docs.analysis_dir", "analysis/label_analysis"))
    return {
        "labels": out / str(get_nested(config, "label_release.output_layout.labels", "labels")),
        "prepared": out
        / str(get_nested(config, "label_release.output_layout.prepared_dataset", "prepared_dataset")),
        "result_analysis": result_analysis,
        "repo_analysis": repo_analysis,
    }


def _write_analysis_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["result_analysis"] / name, text)
    _write_md(dirs["repo_analysis"] / name, text)


def _doc_path(config: dict[str, Any], key: str, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    path = Path(str(get_nested(config, f"label_release.docs.{key}")))
    if not path.is_absolute():
        path = root / path
    return path


def _markdown_table(records: list[dict[str, Any]], columns: list[str], *, max_rows: int = 20) -> str:
    if not records:
        return "_No rows._"
    rows = records[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if len(records) > max_rows:
        body.append(f"| ... | {' | '.join([''] * (len(columns) - 1))} |")
    return "\n".join([header, separator, *body])


def _value_counts(frame: Any, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        return []
    counts = frame[column].fillna("missing").astype(str).value_counts(dropna=False)
    total = max(1, int(len(frame)))
    return [
        {"value": value, "count": int(count), "proportion": float(count / total)}
        for value, count in counts.items()
    ]


def _mean_or_none(values: Any) -> float | None:
    numeric = _pd().to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return float(numeric.mean())


def normalize_orth_token(token: Any) -> str:
    """Strip leading/trailing non-letters for orthographic C/V labels."""

    text = "" if token is None else str(token)
    chars = list(text.strip())
    start = 0
    end = len(chars)
    while start < end and not chars[start].isalpha():
        start += 1
    while end > start and not chars[end - 1].isalpha():
        end -= 1
    return "".join(chars[start:end])


def _cv_of_char(char: str | None) -> str:
    if not char:
        return "unknown"
    if char in DANISH_VOWELS:
        return "V"
    if char.isalpha():
        return "C"
    return "other"


def _boundary_class(boundary_type: str) -> str:
    return {
        "C#C": "low",
        "C#V": "medium_low",
        "V#C": "medium_high",
        "V#V": "high",
        "other": "other",
        "unknown": "unknown",
    }.get(boundary_type, "unknown")


def _boundary_score(boundary_type: str) -> int | None:
    return {"C#C": 0, "C#V": 1, "V#C": 2, "V#V": 3}.get(boundary_type)


def leading_vowel_run(token: Any) -> int:
    normalized = normalize_orth_token(token)
    count = 0
    for char in normalized:
        if char in DANISH_VOWELS:
            count += 1
        else:
            break
    return count


def trailing_vowel_run(token: Any) -> int:
    normalized = normalize_orth_token(token)
    count = 0
    for char in reversed(normalized):
        if char in DANISH_VOWELS:
            count += 1
        else:
            break
    return count


def within_word_vowel_run_max(token: Any) -> int:
    normalized = normalize_orth_token(token)
    max_run = 0
    current = 0
    for char in normalized:
        if char in DANISH_VOWELS:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def classify_boundary(prev_word: Any, word: Any, *, sentence_initial: bool = False) -> dict[str, Any]:
    """Classify an orthographic Danish boundary using C/V categories."""

    prev_norm = normalize_orth_token(prev_word)
    word_norm = normalize_orth_token(word)
    prev_final = prev_norm[-1:] or None
    word_initial = word_norm[:1] or None
    prev_cv = _cv_of_char(prev_final)
    word_cv = _cv_of_char(word_initial)
    if sentence_initial:
        boundary_type = "unknown"
    elif prev_cv in {"unknown"} or word_cv in {"unknown"}:
        boundary_type = "unknown"
    elif prev_cv == "other" or word_cv == "other":
        boundary_type = "other"
    else:
        boundary_type = f"{prev_cv}#{word_cv}"
    return {
        "prev_word_normalized": prev_norm,
        "word_normalized": word_norm,
        "orth_prev_final_char": prev_final,
        "orth_word_initial_char": word_initial,
        "orth_prev_final_cv": prev_cv,
        "orth_word_initial_cv": word_cv,
        "orth_boundary_type": boundary_type,
        "vocoid_run_cross_boundary": int(trailing_vowel_run(prev_norm) + leading_vowel_run(word_norm)),
        "boundary_opacity_score": _boundary_score(boundary_type),
        "boundary_opacity_class": _boundary_class(boundary_type),
    }


def _bool_series(series: Any) -> Any:
    return series.fillna(False).astype(bool)


def _participant_word_key(frame: Any) -> Any:
    return frame["participant_id"].astype(str) + "::" + frame["word_id"].astype(str)


def _pick_word_column(frame: Any) -> str:
    for column in ("word", "word_form", "word_form_x", "word_form_y"):
        if column in frame.columns:
            return column
    raise ValueError("word table has no recognized word surface column")


def build_participant_labels(
    config: dict[str, Any],
    source_dir: Path,
    out: Path,
    dirs: dict[str, Path],
    *,
    repo_root: str | Path,
) -> Any:
    pd = _pd()
    participants = pd.read_parquet(source_dir / "features" / "participant_level.parquet").copy()
    gaze = pd.read_parquet(source_dir / "features" / "word_level_gaze.parquet")

    participants["participant_id"] = participants["participant_id"].astype(str)
    reader_group = []
    binary = []
    for value in participants.get("dyslexia_labeled", pd.Series([None] * len(participants))):
        if pd.isna(value):
            reader_group.append("uncertain")
            binary.append(pd.NA)
        elif int(value) == 1:
            reader_group.append("dyslexia_labeled")
            binary.append(1)
        else:
            reader_group.append("typical_control")
            binary.append(0)
    labels = pd.DataFrame({"participant_id": participants["participant_id"]})
    labels["reader_group"] = reader_group
    labels["reader_group_binary"] = pd.Series(binary, dtype="Int64")
    source_column = str(
        get_nested(config, "label_release.participant_labels.label_source_column", "label_provenance")
    )
    if source_column in participants.columns:
        labels["label_source"] = participants[source_column].fillna("project_metadata").astype(str)
    else:
        labels["label_source"] = "project_metadata"
    labels["diagnostic_provenance"] = str(
        get_nested(config, "label_release.participant_labels.diagnostic_provenance_default", "project_metadata")
    )
    labels.loc[labels["reader_group"].eq("uncertain"), "diagnostic_provenance"] = "unknown"
    labels["label_confidence"] = str(
        get_nested(config, "label_release.participant_labels.confidence_default", "medium")
    )
    labels.loc[labels["reader_group"].eq("uncertain"), "label_confidence"] = "unknown"
    labels["label_notes"] = (
        "Operational project metadata label. Use dyslexia-labeled reader wording; "
        "do not interpret as formal diagnostic status unless provenance is extended."
    )
    labels["include_primary_analysis"] = labels["reader_group"].isin(
        ["dyslexia_labeled", "typical_control"]
    )
    labels["include_sensitivity_analysis"] = labels["include_primary_analysis"]

    optional_map = {
        "age": "age",
        "sex": "sex",
        "comprehension_score": "comprehension_accuracy",
        "n_speeches": "number_of_speeches",
        "recording_batch": "recording_batch",
    }
    for output_column, source in optional_map.items():
        labels[output_column] = participants[source] if source in participants.columns else pd.NA

    observed = gaze.groupby("participant_id", as_index=False).agg(
        n_word_rows=("word_id", "count"),
        n_words_read=("word_id", "nunique"),
        observed_n_speeches=("speech_id", "nunique"),
    )
    labels = labels.merge(observed, on="participant_id", how="left")
    labels["n_speeches"] = labels["n_speeches"].fillna(labels["observed_n_speeches"])
    labels = labels.drop(columns=["observed_n_speeches"])
    labels["source_feature_release_dir"] = str(source_dir)
    labels["label_release_version"] = RELEASE_VERSION
    labels["include_primary_analysis"] = labels["include_primary_analysis"].astype(bool)
    labels["include_sensitivity_analysis"] = labels["include_sensitivity_analysis"].astype(bool)

    path = dirs["labels"] / "participant_labels_v1.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)

    report = _participant_label_report(labels)
    _write_analysis_report(dirs, "participant_label_report.md", report)
    _write_md(_doc_path(config, "participant_label_card", repo_root), _participant_label_card())
    return labels


def _participant_label_report(labels: Any) -> str:
    metadata_columns = ["age", "sex", "comprehension_score", "n_speeches", "n_words_read", "n_word_rows"]
    missing = [
        {
            "field": column,
            "missing": int(labels[column].isna().sum()) if column in labels else len(labels),
            "missing_rate": float(labels[column].isna().mean()) if column in labels else 1.0,
        }
        for column in metadata_columns
    ]
    words_by_group = (
        labels.groupby("reader_group", dropna=False)
        .agg(
            participants=("participant_id", "count"),
            mean_speeches=("n_speeches", "mean"),
            mean_word_rows=("n_word_rows", "mean"),
            total_word_rows=("n_word_rows", "sum"),
            mean_words_read=("n_words_read", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )
    age_summary = (
        labels.groupby("reader_group", dropna=False)
        .agg(mean_age=("age", "mean"), mean_comprehension=("comprehension_score", "mean"))
        .reset_index()
        .to_dict("records")
    )
    sex_summary = []
    if "sex" in labels:
        sex_summary = (
            labels.groupby(["reader_group", "sex"], dropna=False)
            .size()
            .reset_index(name="count")
            .to_dict("records")
        )
    readiness = "ready_with_caveats"
    if labels["reader_group"].isna().any() or labels["participant_id"].duplicated().any():
        readiness = "not_ready"
    return "\n".join(
        [
            "# Participant Label Report",
            "",
            f"- Total participants: {len(labels)}",
            f"- Dyslexia-labeled participants: {int(labels['reader_group'].eq('dyslexia_labeled').sum())}",
            f"- Typical/control participants: {int(labels['reader_group'].eq('typical_control').sum())}",
            f"- Uncertain participants: {int(labels['reader_group'].eq('uncertain').sum())}",
            f"- Readiness judgement: {readiness}",
            "",
            "These are participant-level target labels from operational project metadata. "
            "Use dyslexia-labeled and typical/control wording; the current provenance does not by itself "
            "support formal diagnosis wording.",
            "",
            "## Reader Group Counts",
            _markdown_table(_value_counts(labels, "reader_group"), ["value", "count", "proportion"]),
            "",
            "## Label Source Counts",
            _markdown_table(_value_counts(labels, "label_source"), ["value", "count", "proportion"]),
            "",
            "## Diagnostic Provenance Counts",
            _markdown_table(_value_counts(labels, "diagnostic_provenance"), ["value", "count", "proportion"]),
            "",
            "## Label Confidence Counts",
            _markdown_table(_value_counts(labels, "label_confidence"), ["value", "count", "proportion"]),
            "",
            "## Metadata Missingness",
            _markdown_table(missing, ["field", "missing", "missing_rate"]),
            "",
            "## Words And Speeches Read By Group",
            _markdown_table(
                words_by_group,
                ["reader_group", "participants", "mean_speeches", "mean_word_rows", "total_word_rows"],
            ),
            "",
            "## Age And Comprehension By Group",
            _markdown_table(age_summary, ["reader_group", "mean_age", "mean_comprehension"]),
            "",
            "## Sex Distribution By Group",
            _markdown_table(sex_summary, ["reader_group", "sex", "count"]),
            "",
            "## Warnings",
            "- Labels are imbalanced at participant level and should be handled with grouped splits.",
            "- Diagnostic provenance is operational project metadata in v1.1.",
            "- Classification analyses remain exploratory until participant-level validation is available.",
        ]
    )


def _participant_label_card() -> str:
    return "\n".join(
        [
            "# Participant Label Card v1",
            "",
            "## Purpose",
            "This card documents the participant-level target label used in Label Release v1.1.",
            "",
            "## Allowed Labels",
            "- `dyslexia_labeled`: participant is marked by project metadata as dyslexia-labeled.",
            "- `typical_control`: participant is marked by project metadata as typical/control.",
            "- `uncertain`: source metadata is insufficient or conflicting.",
            "",
            "## Provenance",
            "The v1.1 labels are deterministic transformations of existing project metadata. "
            "Current wording should remain dyslexia-labeled readers and typical/control readers.",
            "",
            "## Intended Use",
            "Use these labels for psycholinguistic group-difference analyses and exploratory "
            "participant-level prediction with participant-grouped splits.",
            "",
            "## Prohibited Interpretations",
            "Do not use the labels as formal diagnostic status, screening outcome, or biomarker. "
            "Do not interpret row-level word examples as independent participant labels.",
            "",
            "## Missingness Policy",
            "Participants with uncertain labels remain documented and can be excluded from primary "
            "analysis through `include_primary_analysis` while remaining available for sensitivity checks.",
        ]
    )


def build_segmentation_labels(
    config: dict[str, Any],
    source_dir: Path,
    out: Path,
    dirs: dict[str, Path],
    *,
    repo_root: str | Path,
) -> tuple[Any, Any, Any]:
    pd = _pd()
    classical = pd.read_parquet(source_dir / "features" / "word_level_classical.parquet").copy()
    sentences = pd.read_parquet(source_dir / "features" / "sentence_level.parquet").copy()
    word_col = _pick_word_column(classical)
    sort_cols = [
        column
        for column in [
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_index_in_sentence",
            "word_id",
        ]
        if column in classical.columns
    ]
    words = classical.sort_values(sort_cols).reset_index(drop=True)
    label_source = str(
        get_nested(
            config,
            "label_release.segmentation_labels.label_source",
            "deterministic_orthographic_danish_vowels",
        )
    )
    confidence = str(
        get_nested(config, "label_release.segmentation_labels.confidence", "orthographic_proxy")
    )
    include_initial = bool(
        get_nested(config, "label_release.segmentation_labels.include_sentence_initial_boundaries", True)
    )

    boundary_rows: list[dict[str, Any]] = []
    for (_, _, sentence_id), group in words.groupby(["speech_id", "paragraph_id", "sentence_id"], sort=False):
        group = group.sort_values("word_index_in_sentence" if "word_index_in_sentence" in group else "word_id")
        prev = None
        for index, row in enumerate(group.itertuples(index=False)):
            sentence_initial = index == 0
            if sentence_initial and not include_initial:
                prev = row
                continue
            previous_word = None if prev is None else getattr(prev, word_col)
            classified = classify_boundary(
                previous_word,
                getattr(row, word_col),
                sentence_initial=sentence_initial,
            )
            paragraph_initial = sentence_initial and int(getattr(row, "word_index_in_paragraph", index)) == 0
            boundary_rows.append(
                {
                    "speech_id": getattr(row, "speech_id"),
                    "paragraph_id": getattr(row, "paragraph_id"),
                    "sentence_id": sentence_id,
                    "prev_word_id": None if prev is None else getattr(prev, "word_id"),
                    "word_id": getattr(row, "word_id"),
                    "boundary_id": f"{sentence_id}_b{int(getattr(row, 'word_index_in_sentence', index))}",
                    "prev_word": previous_word,
                    "word": getattr(row, word_col),
                    **classified,
                    "is_sentence_initial_boundary": bool(sentence_initial),
                    "is_paragraph_initial_boundary": bool(paragraph_initial),
                    "segmentation_label_source": label_source,
                    "segmentation_confidence": confidence,
                    "segmentation_label_version": SEGMENTATION_VERSION,
                    "segmentation_notes": "Orthographic proxy using Danish vowel letters.",
                }
            )
            prev = row
    boundaries = pd.DataFrame(boundary_rows)

    word_rows: list[dict[str, Any]] = []
    prev_by_word = boundaries.set_index("word_id", drop=False).to_dict("index")
    next_by_prev = boundaries.dropna(subset=["prev_word_id"]).set_index("prev_word_id", drop=False).to_dict(
        "index"
    )
    for row in words.itertuples(index=False):
        word_id = getattr(row, "word_id")
        word = getattr(row, word_col)
        norm = normalize_orth_token(word)
        first = norm[:1] or None
        final = norm[-1:] or None
        prev_boundary = prev_by_word.get(word_id)
        next_boundary = next_by_prev.get(word_id)
        vowel_count = sum(1 for char in norm if char in DANISH_VOWELS)
        alpha_count = sum(1 for char in norm if char.isalpha())
        word_rows.append(
            {
                "speech_id": getattr(row, "speech_id"),
                "paragraph_id": getattr(row, "paragraph_id"),
                "sentence_id": getattr(row, "sentence_id"),
                "word_id": word_id,
                "word": word,
                "word_normalized": norm,
                "word_initial_char_orth": first,
                "word_final_char_orth": final,
                "word_initial_cv_orth": _cv_of_char(first),
                "word_final_cv_orth": _cv_of_char(final),
                "starts_with_vowel": bool(first in DANISH_VOWELS) if first else False,
                "ends_with_vowel": bool(final in DANISH_VOWELS) if final else False,
                "prev_boundary_type_orth": None if prev_boundary is None else prev_boundary["orth_boundary_type"],
                "next_boundary_type_orth": None if next_boundary is None else next_boundary["orth_boundary_type"],
                "prev_boundary_opacity_score": None
                if prev_boundary is None
                else prev_boundary["boundary_opacity_score"],
                "next_boundary_opacity_score": None
                if next_boundary is None
                else next_boundary["boundary_opacity_score"],
                "prev_boundary_opacity_class": None
                if prev_boundary is None
                else prev_boundary["boundary_opacity_class"],
                "next_boundary_opacity_class": None
                if next_boundary is None
                else next_boundary["boundary_opacity_class"],
                "within_word_vowel_run_max": int(within_word_vowel_run_max(word)),
                "word_vowel_count": int(vowel_count),
                "word_vowel_ratio": None if alpha_count == 0 else float(vowel_count / alpha_count),
                "has_vowel": bool(vowel_count > 0),
                "has_only_vowels_after_normalization": bool(alpha_count > 0 and vowel_count == alpha_count),
                "has_only_consonants_after_normalization": bool(alpha_count > 0 and vowel_count == 0),
                "segmentation_label_source": label_source,
                "segmentation_confidence": confidence,
                "segmentation_label_version": SEGMENTATION_VERSION,
            }
        )
    word_labels = pd.DataFrame(word_rows)

    sentence_rows = []
    for row in sentences.itertuples(index=False):
        sid = getattr(row, "sentence_id")
        sent_words = word_labels[word_labels["sentence_id"].astype(str) == str(sid)]
        sent_bounds = boundaries[
            (boundaries["sentence_id"].astype(str) == str(sid))
            & (~boundaries["is_sentence_initial_boundary"])
        ]
        counts = sent_bounds["orth_boundary_type"].value_counts()
        scores = _pd().to_numeric(sent_bounds["boundary_opacity_score"], errors="coerce")
        sentence_rows.append(
            {
                "speech_id": getattr(row, "speech_id"),
                "paragraph_id": getattr(row, "paragraph_id"),
                "sentence_id": sid,
                "n_words": int(len(sent_words)),
                "n_boundaries": int(len(sent_bounds)),
                "n_C_hash_C": int(counts.get("C#C", 0)),
                "n_C_hash_V": int(counts.get("C#V", 0)),
                "n_V_hash_C": int(counts.get("V#C", 0)),
                "n_V_hash_V": int(counts.get("V#V", 0)),
                "sentence_vv_boundary_rate": float(counts.get("V#V", 0) / max(1, len(sent_bounds))),
                "sentence_v_boundary_rate": float(
                    (counts.get("C#V", 0) + counts.get("V#C", 0) + counts.get("V#V", 0))
                    / max(1, len(sent_bounds))
                ),
                "sentence_mean_boundary_opacity": None if scores.notna().sum() == 0 else float(scores.mean()),
                "sentence_max_boundary_opacity": None if scores.notna().sum() == 0 else float(scores.max()),
                "sentence_high_opacity_boundary_count": int(counts.get("V#V", 0)),
                "sentence_vowel_initial_word_rate": float(sent_words["starts_with_vowel"].mean())
                if len(sent_words)
                else None,
                "sentence_vowel_final_word_rate": float(sent_words["ends_with_vowel"].mean())
                if len(sent_words)
                else None,
                "sentence_mean_within_word_vowel_run": _mean_or_none(
                    sent_words["within_word_vowel_run_max"]
                )
                if len(sent_words)
                else None,
                "sentence_max_within_word_vowel_run": None
                if sent_words.empty
                else int(sent_words["within_word_vowel_run_max"].max()),
                "segmentation_label_source": label_source,
                "segmentation_label_version": SEGMENTATION_VERSION,
            }
        )
    sentence_labels = pd.DataFrame(sentence_rows)

    label_dir = dirs["labels"]
    label_dir.mkdir(parents=True, exist_ok=True)
    boundaries.to_parquet(label_dir / "segmentation_boundary_labels_v1.parquet", index=False)
    word_labels.to_parquet(label_dir / "segmentation_word_labels_v1.parquet", index=False)
    sentence_labels.to_parquet(label_dir / "segmentation_sentence_labels_v1.parquet", index=False)

    _write_md(_doc_path(config, "segmentation_label_card", repo_root), _segmentation_label_card())
    _write_analysis_report(
        dirs,
        "segmentation_label_distribution_report.md",
        _segmentation_distribution_report(boundaries, word_labels, sentence_labels),
    )
    _write_analysis_report(
        dirs,
        "segmentation_confounds_report.md",
        _segmentation_confounds_report(source_dir, boundaries, word_labels, sentence_labels),
    )
    return boundaries, word_labels, sentence_labels


def _segmentation_label_card() -> str:
    return "\n".join(
        [
            "# Segmentation Label Card v1",
            "",
            "## Motivation",
            "Segmentation-opacity labels describe stimulus-level orthographic boundary structure for "
            "Danish natural-reading analyses. They are not participant target labels.",
            "",
            "## Deterministic Algorithm",
            "For each within-sentence word boundary, the final alphabetic character of the previous "
            "word and the initial alphabetic character of the current word are classified as C or V. "
            "Leading and trailing punctuation or quotes are stripped only for classification; original "
            "word strings are preserved.",
            "",
            "## Danish Vowel Set",
            "`a e i o u y æ ø å` and uppercase variants.",
            "",
            "## Boundary Types And Scores",
            "- `C#C`: score 0, low opacity.",
            "- `C#V`: score 1, medium-low opacity.",
            "- `V#C`: score 2, medium-high opacity.",
            "- `V#V`: score 3, high opacity.",
            "- `other` and `unknown`: no numeric opacity score.",
            "",
            "## Examples",
            "- C#C: `tak for`.",
            "- C#V: `kan ikke`.",
            "- V#C: `de går`.",
            "- V#V: `se efter`.",
            "",
            "## Limitations",
            "These are orthographic proxy labels, not pronunciation-aware syllabification or phonology. "
            "They should be interpreted as deterministic stimulus descriptors for exploratory "
            "psycholinguistic modeling.",
            "",
            "## Planned Extension",
            "A future pronunciation-aware layer can add phonological boundary labels from a Danish "
            "pronunciation lexicon. LLM-generated labels are not part of the core v1.1 release.",
            "",
            "## Prohibited Interpretations",
            "Do not treat segmentation-opacity labels as dyslexia labels, diagnostic measures, or "
            "evidence of individual reading status.",
        ]
    )


def _segmentation_distribution_report(boundaries: Any, word_labels: Any, sentence_labels: Any) -> str:
    within = boundaries[~boundaries["is_sentence_initial_boundary"]].copy()
    counts = _value_counts(within, "orth_boundary_type")
    by_speech = (
        within.groupby(["speech_id", "orth_boundary_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .head(60)
        .to_dict("records")
    )
    sentence_bins = sentence_labels.copy()
    sentence_bins["sentence_length_bin"] = _pd().cut(
        sentence_bins["n_words"], bins=[0, 5, 10, 20, 40, math.inf], include_lowest=True
    ).astype(str)
    by_bin = (
        sentence_bins.groupby("sentence_length_bin", dropna=False)
        .agg(
            sentences=("sentence_id", "count"),
            mean_opacity=("sentence_mean_boundary_opacity", "mean"),
            vv_rate=("sentence_vv_boundary_rate", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )
    examples = (
        within.sort_values(["orth_boundary_type", "boundary_id"])
        .groupby("orth_boundary_type", dropna=False)
        .head(5)[["orth_boundary_type", "prev_word", "word", "boundary_opacity_score", "boundary_id"]]
        .to_dict("records")
    )
    vv = within[within["orth_boundary_type"].eq("V#V")]
    top_vv_words = (
        _pd()
        .concat([vv["prev_word"].astype(str), vv["word"].astype(str)])
        .value_counts()
        .head(20)
        .reset_index()
        .rename(columns={"index": "word", "count": "count"})
        .to_dict("records")
        if not vv.empty
        else []
    )
    high_examples = vv[
        ["speech_id", "sentence_id", "prev_word", "word", "vocoid_run_cross_boundary"]
    ].head(20).to_dict("records")
    return "\n".join(
        [
            "# Segmentation Label Distribution Report",
            "",
            f"- Total boundary rows including sentence-initial rows: {len(boundaries)}",
            f"- Within-sentence boundary count: {len(within)}",
            f"- Total word count: {len(word_labels)}",
            f"- Total sentence count: {len(sentence_labels)}",
            "",
            "## Boundary Type Distribution",
            _markdown_table(counts, ["value", "count", "proportion"]),
            "",
            "## Distribution By Speech",
            _markdown_table(by_speech, ["speech_id", "orth_boundary_type", "count"], max_rows=60),
            "",
            "## Distribution By Sentence Length Bin",
            _markdown_table(by_bin, ["sentence_length_bin", "sentences", "mean_opacity", "vv_rate"]),
            "",
            "## Boundary Examples",
            _markdown_table(
                examples,
                ["orth_boundary_type", "prev_word", "word", "boundary_opacity_score", "boundary_id"],
                max_rows=40,
            ),
            "",
            "## Top Words Involved In V#V Boundaries",
            _markdown_table(top_vv_words, ["word", "count"], max_rows=20),
            "",
            "## High-Opacity Boundary Examples",
            _markdown_table(
                high_examples,
                ["speech_id", "sentence_id", "prev_word", "word", "vocoid_run_cross_boundary"],
                max_rows=20,
            ),
        ]
    )


def _segmentation_confounds_report(
    source_dir: Path, boundaries: Any, word_labels: Any, sentence_labels: Any
) -> str:
    pd = _pd()
    classical = pd.read_parquet(source_dir / "features" / "word_level_classical.parquet")
    joined = word_labels.merge(
        classical[
            [
                column
                for column in [
                    "word_id",
                    "word_length_chars",
                    "log_corpus_frequency",
                    "sentence_length_words",
                    "word_index_in_sentence",
                    "long_word_lix_component",
                ]
                if column in classical.columns
            ]
        ],
        on="word_id",
        how="left",
    )
    model_path = source_dir / "modeling_tables" / "word_level_full_with_dfm_lm.parquet"
    exposure_rows = []
    if model_path.exists():
        model = pd.read_parquet(model_path)
        lm_cols = [
            column
            for column in ["word_id", "dfm_lm_word_surprisal", "dfm_lm_word_entropy"]
            if column in model.columns
        ]
        if len(lm_cols) > 1:
            lm = model[lm_cols].drop_duplicates("word_id")
            joined = joined.merge(lm, on="word_id", how="left")
        exposure = model[["participant_id", "word_id", "group_label"]].merge(
            word_labels[["word_id", "prev_boundary_opacity_score", "prev_boundary_type_orth"]],
            on="word_id",
            how="left",
        )
        group_map = {"typical": "typical_control", "dyslexia_labeled": "dyslexia_labeled"}
        exposure["reader_group"] = exposure["group_label"].map(group_map).fillna(exposure["group_label"])
        exposure_rows = (
            exposure.groupby("reader_group", dropna=False)
            .agg(
                word_rows=("word_id", "count"),
                mean_prev_opacity=("prev_boundary_opacity_score", "mean"),
                vv_exposure_rate=("prev_boundary_type_orth", lambda s: s.eq("V#V").mean()),
            )
            .reset_index()
            .to_dict("records")
        )
    corr_rows = []
    target = pd.to_numeric(joined["prev_boundary_opacity_score"], errors="coerce")
    for column in [
        "word_length_chars",
        "log_corpus_frequency",
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "sentence_length_words",
        "word_index_in_sentence",
        "long_word_lix_component",
    ]:
        if column in joined.columns:
            values = pd.to_numeric(joined[column], errors="coerce")
            corr = None if target.notna().sum() < 2 or values.notna().sum() < 2 else target.corr(values)
            corr_rows.append(
                {
                    "feature": column,
                    "pearson_correlation_with_prev_opacity": None if pd.isna(corr) else float(corr),
                    "feature_missing_rate": float(values.isna().mean()),
                }
            )
    missingness = [
        {
            "field": "prev_boundary_opacity_score",
            "missing_rate": float(joined["prev_boundary_opacity_score"].isna().mean()),
        },
        {
            "field": "segmentation_word_label",
            "missing_rate": float(
                joined["segmentation_label_version"].isna().mean()
                if "segmentation_label_version" in joined
                else 1.0
            ),
        },
    ]
    return "\n".join(
        [
            "# Segmentation Confounds Report",
            "",
            "Segmentation labels are deterministic stimulus descriptors. This report quantifies "
            "associations with other stimulus variables for later controlled analyses.",
            "",
            "## Correlations With Previous-Boundary Opacity",
            _markdown_table(
                corr_rows,
                ["feature", "pearson_correlation_with_prev_opacity", "feature_missing_rate"],
            ),
            "",
            "## Reader-Group Exposure After Joining To Participant Word Rows",
            _markdown_table(
                exposure_rows,
                ["reader_group", "word_rows", "mean_prev_opacity", "vv_exposure_rate"],
            ),
            "",
            "## Missingness",
            _markdown_table(missingness, ["field", "missing_rate"]),
            "",
            "## Recommendation",
            "Segmentation labels are ready for exploratory modeling with caveats. Later analyses should "
            "control for word length, word frequency, surprisal, entropy, sentence length, word position, "
            "and text assignment because orthographic boundary opacity is not randomized.",
            f"Within-sentence boundary rows analyzed: {int((~boundaries['is_sentence_initial_boundary']).sum())}.",
        ]
    )


def build_quality_labels(
    config: dict[str, Any],
    source_dir: Path,
    dirs: dict[str, Path],
    participant_labels: Any,
    segmentation_word: Any,
    *,
    repo_root: str | Path,
) -> Any:
    pd = _pd()
    word = pd.read_parquet(source_dir / "modeling_tables" / "word_level_full_with_dfm_lm.parquet").copy()
    participant_min = participant_labels[
        [
            "participant_id",
            "reader_group",
            "include_primary_analysis",
            "include_sensitivity_analysis",
            "age",
            "sex",
            "comprehension_score",
        ]
    ].copy()
    seg_min = segmentation_word[["word_id", "segmentation_confidence", "segmentation_label_version"]].copy()
    q = word[
        [
            column
            for column in [
                "participant_id",
                "speech_id",
                "paragraph_id",
                "sentence_id",
                "word_id",
                "TRT",
                "FFD",
                "GD",
                "skip",
                "dfm_lm_word_surprisal",
                "dfm_lm_word_entropy",
                "dfm_lm_alignment_status",
                "dfm_lm_alignment_warning",
                "dfm_lm_alignment_error",
                "upos",
                "parser_backend",
                "paragraph_cohesion",
                "local_semantic_drift",
            ]
            if column in word.columns
        ]
    ].copy()
    q["participant_word_key"] = _participant_word_key(q)
    q["stimulus_word_key"] = q["word_id"].astype(str)
    q = q.merge(participant_min, on="participant_id", how="left")
    q = q.merge(seg_min, on="word_id", how="left")
    q["gaze_missing"] = q[["TRT", "FFD", "GD"]].isna().all(axis=1) if {"TRT", "FFD", "GD"}.issubset(q) else False
    q["participant_label_missing"] = q["reader_group"].isna()
    q["segmentation_label_missing"] = q["segmentation_label_version"].isna()
    q["lm_missing"] = q["dfm_lm_word_surprisal"].isna() if "dfm_lm_word_surprisal" in q else True
    q["lm_alignment_status"] = q.get("dfm_lm_alignment_status", pd.NA)
    q["lm_alignment_warning"] = q.get("dfm_lm_alignment_warning", pd.NA)
    q["lm_alignment_error"] = q.get("dfm_lm_alignment_error", pd.NA)
    q["parser_missing"] = q["upos"].isna() if "upos" in q else True
    q["parser_status"] = str(
        get_nested(config, "label_release.quality_labels.parser_status_for_surface_fallback")
        or "surface_heuristic_fallback"
    )
    q["parser_confidence"] = str(
        get_nested(config, "label_release.quality_labels.parser_confidence_for_surface_fallback")
        or "usable_for_surface_not_syntax"
    )
    q["embedding_missing"] = (
        q[["paragraph_cohesion", "local_semantic_drift"]].isna().all(axis=1)
        if {"paragraph_cohesion", "local_semantic_drift"}.issubset(q)
        else True
    )
    q["participant_metadata_missing"] = q[["age", "sex", "comprehension_score"]].isna().any(axis=1)
    q["text_assignment_balance_status"] = "documented_not_controlled"

    base_primary = _bool_series(q["include_primary_analysis"])
    base_sensitivity = _bool_series(q["include_sensitivity_analysis"])
    q["include_primary_analysis"] = (
        base_primary & ~q["participant_label_missing"] & ~q["segmentation_label_missing"]
    )
    q["include_sensitivity_analysis"] = base_sensitivity & ~q["participant_label_missing"]
    reasons = []
    for row in q.itertuples(index=False):
        row_reasons = []
        if bool(getattr(row, "participant_label_missing")):
            row_reasons.append("participant_label_missing")
        if bool(getattr(row, "segmentation_label_missing")):
            row_reasons.append("segmentation_label_missing")
        reasons.append(";".join(row_reasons))
    q["exclusion_reason"] = reasons
    bool_columns = [
        "gaze_missing",
        "participant_label_missing",
        "segmentation_label_missing",
        "lm_missing",
        "parser_missing",
        "embedding_missing",
        "participant_metadata_missing",
        "include_primary_analysis",
        "include_sensitivity_analysis",
    ]
    for column in bool_columns:
        q[column] = q[column].astype(bool)

    keep = [
        "participant_id",
        "speech_id",
        "paragraph_id",
        "sentence_id",
        "word_id",
        "stimulus_word_key",
        "participant_word_key",
        "gaze_missing",
        "participant_label_missing",
        "segmentation_label_missing",
        "segmentation_confidence",
        "lm_missing",
        "lm_alignment_status",
        "lm_alignment_warning",
        "lm_alignment_error",
        "parser_missing",
        "parser_status",
        "parser_confidence",
        "embedding_missing",
        "participant_metadata_missing",
        "text_assignment_balance_status",
        "include_primary_analysis",
        "include_sensitivity_analysis",
        "exclusion_reason",
    ]
    quality = q[keep].copy()
    quality.to_parquet(dirs["labels"] / "quality_labels_v1.parquet", index=False)
    _write_md(_doc_path(config, "quality_label_card", repo_root), _quality_label_card())
    _write_analysis_report(dirs, "quality_label_report.md", _quality_label_report(quality, q))
    return quality


def _quality_label_card() -> str:
    return "\n".join(
        [
            "# Quality Label Card v1",
            "",
            "Quality labels document whether each participant-word row has usable labels and feature "
            "coverage for primary and sensitivity analyses.",
            "",
            "## Parser Status",
            "`parser_status = surface_heuristic_fallback` means current parser-feature files are "
            "surface and morpho-orthographic heuristics. They are usable as heuristic covariates, "
            "not as true syntactic annotations.",
            "",
            "## Missingness",
            "Missing LM, embedding, parser, participant metadata, and segmentation fields are preserved "
            "as boolean flags. Rows are not dropped by the label release.",
            "",
            "## Intended Use",
            "Use `include_primary_analysis` and `include_sensitivity_analysis` to construct transparent "
            "analysis subsets. Report missingness by reader group before modeling.",
        ]
    )


def _quality_label_report(quality: Any, enriched: Any) -> str:
    flag_columns = [
        "gaze_missing",
        "participant_label_missing",
        "segmentation_label_missing",
        "lm_missing",
        "parser_missing",
        "embedding_missing",
        "participant_metadata_missing",
    ]
    overall = [
        {"flag": column, "count": int(quality[column].sum()), "rate": float(quality[column].mean())}
        for column in flag_columns
        if column in quality
    ]
    by_group = (
        enriched.groupby("reader_group", dropna=False)
        .agg(
            rows=("word_id", "count"),
            lm_missing_rate=("lm_missing", "mean"),
            embedding_missing_rate=("embedding_missing", "mean"),
            segmentation_missing_rate=("segmentation_label_missing", "mean"),
            participant_metadata_missing_rate=("participant_metadata_missing", "mean"),
        )
        .reset_index()
        .to_dict("records")
        if "reader_group" in enriched
        else []
    )
    by_speech = (
        quality.groupby("speech_id", dropna=False)
        .agg(
            rows=("word_id", "count"),
            lm_missing_rate=("lm_missing", "mean"),
            embedding_missing_rate=("embedding_missing", "mean"),
        )
        .reset_index()
        .head(80)
        .to_dict("records")
    )
    by_participant = (
        quality.groupby("participant_id", dropna=False)
        .agg(
            rows=("word_id", "count"),
            lm_missing_rate=("lm_missing", "mean"),
            embedding_missing_rate=("embedding_missing", "mean"),
            participant_metadata_missing_rate=("participant_metadata_missing", "mean"),
        )
        .reset_index()
        .head(80)
        .to_dict("records")
    )
    warnings = _value_counts(quality, "lm_alignment_warning")
    parser_status = _value_counts(quality, "parser_status")
    examples = (
        quality[quality["lm_alignment_warning"].notna()]
        [["speech_id", "sentence_id", "word_id", "lm_alignment_status", "lm_alignment_warning"]]
        .head(20)
        .to_dict("records")
    )
    return "\n".join(
        [
            "# Quality Label Report",
            "",
            f"- Quality rows: {len(quality)}",
            f"- Primary-analysis rows: {int(quality['include_primary_analysis'].sum())}",
            f"- Sensitivity-analysis rows: {int(quality['include_sensitivity_analysis'].sum())}",
            "",
            "## Missingness Overall",
            _markdown_table(overall, ["flag", "count", "rate"]),
            "",
            "## Missingness By Reader Group",
            _markdown_table(
                by_group,
                [
                    "reader_group",
                    "rows",
                    "lm_missing_rate",
                    "embedding_missing_rate",
                    "segmentation_missing_rate",
                ],
            ),
            "",
            "## Missingness By Speech",
            _markdown_table(by_speech, ["speech_id", "rows", "lm_missing_rate", "embedding_missing_rate"], max_rows=80),
            "",
            "## Missingness By Participant",
            _markdown_table(
                by_participant,
                ["participant_id", "rows", "lm_missing_rate", "embedding_missing_rate"],
                max_rows=80,
            ),
            "",
            "## Parser Status Counts",
            _markdown_table(parser_status, ["value", "count", "proportion"]),
            "",
            "## Alignment Warning Counts",
            _markdown_table(warnings, ["value", "count", "proportion"]),
            "",
            "## Alignment Warning Examples",
            _markdown_table(
                examples,
                ["speech_id", "sentence_id", "word_id", "lm_alignment_status", "lm_alignment_warning"],
            ),
            "",
            "## Recommendation",
            "Primary analyses can use rows with complete participant and segmentation labels. DFM LM "
            "missingness and alignment warnings should be reported and checked in sensitivity analyses. "
            "The parser status must remain surface_heuristic_fallback until a real parser run succeeds.",
        ]
    )


def build_split_labels(
    config: dict[str, Any],
    source_dir: Path,
    dirs: dict[str, Path],
    participant_labels: Any,
    quality: Any,
    *,
    repo_root: str | Path,
) -> Any:
    pd = _pd()
    seed = int(get_nested(config, "label_release.deterministic_seed", 17))
    participants = participant_labels.copy()
    participants["dyslexia_labeled"] = participants["reader_group_binary"].fillna(-1).astype(int)
    splits = []

    for split_name, frame in [
        ("leave_one_participant_out", leave_one_participant_out(participants)),
        (
            "participant_grouped_kfold",
            participant_grouped_folds(
                participants[participants["reader_group"].isin(["dyslexia_labeled", "typical_control"])],
                n_splits=int(get_nested(config, "label_release.split_labels.participant_grouped_kfolds", 5)),
                seed=seed,
            ),
        ),
    ]:
        tmp = frame.rename(columns={"fold": "fold_id", "split": "split_role"}).copy()
        tmp["split_name"] = split_name
        splits.append(tmp)

    uncertain = participants.copy()
    uncertain["fold_id"] = 0
    uncertain["split_role"] = uncertain["reader_group"].map(
        lambda value: "exclude" if value == "uncertain" else "include"
    )
    uncertain["split_name"] = "sensitivity_exclude_uncertain_labels"
    uncertain["dyslexia_labeled"] = uncertain["reader_group_binary"].fillna(-1).astype(int)
    splits.append(
        uncertain[
            ["split_name", "fold_id", "split_role", "participant_id", "dyslexia_labeled", "reader_group"]
        ]
    )

    split_labels = pd.concat(splits, ignore_index=True, sort=False)
    split_labels = split_labels.merge(
        participant_labels[["participant_id", "reader_group"]],
        on="participant_id",
        how="left",
        suffixes=("", "_label"),
    )
    split_labels["reader_group"] = split_labels["reader_group"].fillna(split_labels.get("reader_group_label"))
    split_labels = split_labels.drop(columns=["reader_group_label"], errors="ignore")
    word_counts = quality.groupby("participant_id", as_index=False).agg(n_word_rows=("word_id", "count"))
    split_labels = split_labels.merge(word_counts, on="participant_id", how="left")
    split_labels["include_in_fold"] = split_labels["split_role"].isin(["train", "test", "include"])
    split_labels["split_seed"] = seed
    split_labels["split_version"] = "split_policy_v1"
    split_labels["split_valid"] = True
    split_labels["skip_reason"] = ""

    count_rows = []
    for (split_name, fold_id), group in split_labels.groupby(["split_name", "fold_id"], dropna=False):
        train = group[group["split_role"].isin(["train", "include"])]
        test = group[group["split_role"].eq("test")]
        overlap = set(train["participant_id"]).intersection(set(test["participant_id"]))
        valid = not overlap
        skip_reason = "" if valid else "participant_overlap_between_train_and_test"
        counts = {
            "split_name": split_name,
            "fold_id": fold_id,
            "n_train_participants": int(train["participant_id"].nunique()),
            "n_test_participants": int(test["participant_id"].nunique()),
            "n_train_dyslexia_labeled": int(train["reader_group"].eq("dyslexia_labeled").sum()),
            "n_train_typical_control": int(train["reader_group"].eq("typical_control").sum()),
            "n_test_dyslexia_labeled": int(test["reader_group"].eq("dyslexia_labeled").sum()),
            "n_test_typical_control": int(test["reader_group"].eq("typical_control").sum()),
            "n_train_word_rows": int(train["n_word_rows"].fillna(0).sum()),
            "n_test_word_rows": int(test["n_word_rows"].fillna(0).sum()),
            "split_valid": bool(valid),
            "skip_reason": skip_reason,
        }
        count_rows.append(counts)
    counts_frame = pd.DataFrame(count_rows)
    split_labels = split_labels.drop(columns=["split_valid", "skip_reason"]).merge(
        counts_frame,
        on=["split_name", "fold_id"],
        how="left",
    )
    split_labels["include_in_fold"] = split_labels["include_in_fold"].astype(bool)
    split_labels["split_valid"] = split_labels["split_valid"].astype(bool)

    keep = [
        "split_name",
        "fold_id",
        "participant_id",
        "reader_group",
        "split_role",
        "include_in_fold",
        "n_train_participants",
        "n_test_participants",
        "n_train_dyslexia_labeled",
        "n_train_typical_control",
        "n_test_dyslexia_labeled",
        "n_test_typical_control",
        "n_train_word_rows",
        "n_test_word_rows",
        "split_valid",
        "skip_reason",
        "split_seed",
        "split_version",
    ]
    split_labels = split_labels[keep].sort_values(["split_name", "fold_id", "split_role", "participant_id"])
    split_labels.to_parquet(dirs["labels"] / "split_labels_v1.parquet", index=False)
    _write_md(_doc_path(config, "split_policy", repo_root), _split_policy_doc())
    _write_analysis_report(dirs, "split_label_report.md", _split_label_report(split_labels))
    return split_labels


def _split_policy_doc() -> str:
    return "\n".join(
        [
            "# Split Policy v1",
            "",
            "## Allowed Splits",
            "- `leave_one_participant_out`: primary participant-level evaluation split.",
            "- `participant_grouped_kfold`: secondary participant-grouped cross-validation split.",
            "- `sensitivity_exclude_uncertain_labels`: documented sensitivity subset.",
            "",
            "## Prohibited Splits",
            "Random word-level train/test splitting is not allowed because word rows from the same "
            "participant are not independent and would leak participant-level target labels.",
            "",
            "## Fold Representation",
            "Every split label row keeps a participant wholly in one fold role. Invalid folds are kept "
            "with `split_valid = false` and a `skip_reason`; they are not silently dropped.",
            "",
            "## Imbalance Handling",
            "Class counts are reported per fold. Later predictive analyses should report skipped folds, "
            "confidence intervals where feasible, and avoid interpreting exploratory classification as "
            "screening.",
        ]
    )


def _split_label_report(split_labels: Any) -> str:
    fold_summary = (
        split_labels.drop_duplicates(["split_name", "fold_id"])
        [
            [
                "split_name",
                "fold_id",
                "n_train_participants",
                "n_test_participants",
                "n_train_dyslexia_labeled",
                "n_train_typical_control",
                "n_test_dyslexia_labeled",
                "n_test_typical_control",
                "split_valid",
                "skip_reason",
            ]
        ]
        .sort_values(["split_name", "fold_id"])
        .to_dict("records")
    )
    lopo = split_labels[split_labels["split_name"].eq("leave_one_participant_out")]
    test_counts = lopo[lopo["split_role"].eq("test")]["participant_id"].value_counts()
    return "\n".join(
        [
            "# Split Label Report",
            "",
            f"- Split label rows: {len(split_labels)}",
            f"- Split names: {', '.join(sorted(split_labels['split_name'].astype(str).unique()))}",
            f"- Invalid fold rows: {int((~split_labels['split_valid']).sum())}",
            f"- LOPO participants tested exactly once: {bool((test_counts == 1).all())}",
            "- Random word-level split present: false",
            "",
            "## Fold Summary",
            _markdown_table(fold_summary, list(fold_summary[0]) if fold_summary else [], max_rows=120),
        ]
    )


def build_prepared_dataset(
    source_dir: Path,
    out: Path,
    dirs: dict[str, Path],
    participant_labels: Any,
    segmentation_word: Any,
    segmentation_sentence: Any,
    quality: Any,
) -> dict[str, Any]:
    pd = _pd()
    prep = dirs["prepared"]
    prep.mkdir(parents=True, exist_ok=True)
    word = pd.read_parquet(source_dir / "modeling_tables" / "word_level_full_with_dfm_lm.parquet").copy()
    sentence = pd.read_parquet(source_dir / "modeling_tables" / "sentence_level_full.parquet").copy()
    participants = pd.read_parquet(source_dir / "modeling_tables" / "participant_aggregates.parquet").copy()

    word["participant_word_key"] = _participant_word_key(word)
    word["stimulus_word_key"] = word["word_id"].astype(str)
    if "word" not in word.columns:
        word["word"] = word[_pick_word_column(word)]
    base_rows = len(word)
    duplicate_participant_word_keys = int(word["participant_word_key"].duplicated().sum())

    participant_cols = [
        "participant_id",
        "reader_group",
        "reader_group_binary",
        "label_source",
        "diagnostic_provenance",
        "label_confidence",
        "include_primary_analysis",
        "include_sensitivity_analysis",
        "age",
        "sex",
        "comprehension_score",
    ]
    word = word.merge(
        participant_labels[[column for column in participant_cols if column in participant_labels.columns]].rename(
            columns={
                "include_primary_analysis": "participant_include_primary_analysis",
                "include_sensitivity_analysis": "participant_include_sensitivity_analysis",
            }
        ),
        on="participant_id",
        how="left",
    )
    seg_cols = [
        column
        for column in segmentation_word.columns
        if column
        not in {
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word",
        }
    ]
    word = word.merge(segmentation_word[seg_cols], on="word_id", how="left", suffixes=("", "_seg"))
    quality_cols = [
        column
        for column in quality.columns
        if column
        not in {"participant_id", "speech_id", "paragraph_id", "sentence_id", "word_id", "stimulus_word_key"}
    ]
    word = word.merge(
        quality[["participant_id", "word_id", *quality_cols]],
        on=["participant_id", "word_id"],
        how="left",
        suffixes=("", "_quality"),
    )
    # Use quality inclusion flags as the row-level analysis flags.
    if "include_primary_analysis_quality" in word.columns:
        word["include_primary_analysis"] = word["include_primary_analysis_quality"]
    if "include_sensitivity_analysis_quality" in word.columns:
        word["include_sensitivity_analysis"] = word["include_sensitivity_analysis_quality"]

    sentence_ready = sentence.merge(segmentation_sentence, on=["speech_id", "paragraph_id", "sentence_id"], how="left")
    sentence_quality = (
        quality.groupby("sentence_id", as_index=False)
        .agg(
            quality_word_rows=("word_id", "count"),
            lm_missing_rate=("lm_missing", "mean"),
            segmentation_missing_rate=("segmentation_label_missing", "mean"),
            embedding_missing_rate=("embedding_missing", "mean"),
        )
    )
    sentence_ready = sentence_ready.merge(sentence_quality, on="sentence_id", how="left")

    exposure = word.groupby("participant_id", as_index=False).agg(
        mean_segmentation_opacity=("prev_boundary_opacity_score", "mean"),
        vv_boundary_exposure_rate=("prev_boundary_type_orth", lambda values: values.eq("V#V").mean()),
        mean_dfm_surprisal=("dfm_lm_word_surprisal", "mean"),
        mean_dfm_entropy=("dfm_lm_word_entropy", "mean"),
        lm_missing_rate=("lm_missing", "mean"),
        segmentation_missing_rate=("segmentation_label_missing", "mean"),
        embedding_missing_rate=("embedding_missing", "mean"),
        primary_analysis_row_rate=("include_primary_analysis", "mean"),
    )
    participant_ready = participants.merge(participant_labels, on="participant_id", how="left")
    participant_ready = participant_ready.merge(exposure, on="participant_id", how="left")

    word_path = prep / "analysis_ready_word_level_v1_1.parquet"
    sentence_path = prep / "analysis_ready_sentence_level_v1_1.parquet"
    participant_path = prep / "analysis_ready_participant_level_v1_1.parquet"
    word.to_parquet(word_path, index=False)
    sentence_ready.to_parquet(sentence_path, index=False)
    participant_ready.to_parquet(participant_path, index=False)

    manifest = {
        "run_type": "prepared_dataset_v1_1",
        "status": "complete",
        "output_dir": str(prep),
        "source_feature_release_dir": str(source_dir),
        "row_counts": {
            "analysis_ready_word_level_v1_1": int(len(word)),
            "analysis_ready_sentence_level_v1_1": int(len(sentence_ready)),
            "analysis_ready_participant_level_v1_1": int(len(participant_ready)),
        },
        "join_validation": {
            "source_word_rows": int(base_rows),
            "word_rows_after_labels": int(len(word)),
            "unexpected_row_loss": int(base_rows - len(word)),
            "unexpected_row_gain": int(len(word) - base_rows),
            "duplicate_participant_word_keys": duplicate_participant_word_keys,
            "duplicate_stimulus_word_keys": int(segmentation_word["word_id"].duplicated().sum()),
            "missing_participant_label_rate": float(word["reader_group"].isna().mean()),
            "missing_segmentation_word_label_rate": float(word["segmentation_label_version"].isna().mean()),
            "missing_quality_label_rate": float(word["participant_word_key_quality"].isna().mean())
            if "participant_word_key_quality" in word
            else 0.0,
            "target_labels_used_during_feature_generation": False,
        },
    }
    _write_json(prep / "analysis_ready_manifest.json", manifest)
    return manifest


def write_text_assignment_balance_report(
    dirs: dict[str, Path], word_ready: Any, sentence_ready: Any
) -> None:
    frame = word_ready.copy()
    if "lix_component" not in frame.columns and "lix_component" in sentence_ready:
        frame = frame.merge(sentence_ready[["sentence_id", "lix_component"]], on="sentence_id", how="left")
    participants = (
        frame[["participant_id", "reader_group"]].drop_duplicates().groupby("reader_group").size()
        .reset_index(name="participants")
        .to_dict("records")
    )
    speeches = (
        frame.groupby("reader_group")["speech_id"].nunique().reset_index(name="speeches_read").to_dict("records")
    )
    speech_group_participants = (
        frame.groupby(["speech_id", "reader_group"])["participant_id"]
        .nunique()
        .reset_index(name="participant_count")
        .head(120)
        .to_dict("records")
    )
    speech_group_rows = (
        frame.groupby(["speech_id", "reader_group"])["word_id"]
        .count()
        .reset_index(name="word_rows")
        .head(120)
        .to_dict("records")
    )
    exposure_cols = [
        column
        for column in [
            "word_length_chars",
            "sentence_length_words",
            "log_corpus_frequency",
            "dfm_lm_word_surprisal",
            "dfm_lm_word_entropy",
            "lix_component",
            "prev_boundary_opacity_score",
        ]
        if column in frame.columns
    ]
    exposures = frame.groupby("reader_group", dropna=False).agg(
        participants=("participant_id", "nunique"),
        mean_speeches_per_participant=("speech_id", lambda s: s.groupby(frame.loc[s.index, "participant_id"]).nunique().mean()),
        mean_word_rows_per_participant=("word_id", lambda s: s.groupby(frame.loc[s.index, "participant_id"]).count().mean()),
        vv_exposure_rate=("prev_boundary_type_orth", lambda s: s.eq("V#V").mean()),
        comprehension_score=("comprehension_score", "mean"),
        age=("age", "mean"),
        lm_missing_rate=("lm_missing", "mean"),
        segmentation_missing_rate=("segmentation_label_missing", "mean"),
    )
    for column in exposure_cols:
        exposures[f"mean_{column}"] = frame.groupby("reader_group")[column].mean()
    exposure_records = exposures.reset_index().to_dict("records")
    sex_records = (
        frame[["participant_id", "reader_group", "sex"]]
        .drop_duplicates()
        .groupby(["reader_group", "sex"], dropna=False)
        .size()
        .reset_index(name="participants")
        .to_dict("records")
        if "sex" in frame
        else []
    )
    report = "\n".join(
        [
            "# Text Assignment Balance Report",
            "",
            "This report quantifies text and stimulus exposure by reader group. It documents possible "
            "confounds for later controlled analyses; v1.1 does not try to solve them.",
            "",
            "## Participants By Reader Group",
            _markdown_table(participants, ["reader_group", "participants"]),
            "",
            "## Speeches Read By Reader Group",
            _markdown_table(speeches, ["reader_group", "speeches_read"]),
            "",
            "## Speech x Reader Group Participant Counts",
            _markdown_table(
                speech_group_participants, ["speech_id", "reader_group", "participant_count"], max_rows=120
            ),
            "",
            "## Speech x Reader Group Word Rows",
            _markdown_table(speech_group_rows, ["speech_id", "reader_group", "word_rows"], max_rows=120),
            "",
            "## Exposure And Missingness By Group",
            _markdown_table(exposure_records, list(exposure_records[0]) if exposure_records else [], max_rows=20),
            "",
            "## Sex Distribution By Group",
            _markdown_table(sex_records, ["reader_group", "sex", "participants"]),
            "",
            "## Recommendation",
            "Later analyses should include text/stimulus controls and sensitivity checks for exposure "
            "imbalance. Participant-grouped splits remain mandatory.",
        ]
    )
    _write_analysis_report(dirs, "text_assignment_balance_report.md", report)


def write_readiness_report(
    dirs: dict[str, Path],
    manifest: dict[str, Any],
    participant_labels: Any,
    segmentation_word: Any,
    quality: Any,
    split_labels: Any,
) -> None:
    participant_ready = not participant_labels["reader_group"].isna().any()
    segmentation_ready = segmentation_word["segmentation_label_version"].notna().all()
    quality_ready = not quality["participant_word_key"].duplicated().any()
    splits_ready = bool(split_labels["split_valid"].all()) and not split_labels["split_name"].str.contains(
        "random", case=False, na=False
    ).any()
    judgement = "ready_with_caveats"
    if not (participant_ready and segmentation_ready and quality_ready and splits_ready):
        judgement = "not_ready"
    report = "\n".join(
        [
            "# Prepared Dataset Readiness Report",
            "",
            f"- Participant target labels complete: {participant_ready}",
            "- Label sources/provenance documented: true",
            f"- Segmentation labels complete enough: {segmentation_ready}",
            "- Segmentation labels confounded with other linguistic features: documented; control later",
            f"- Quality labels complete: {quality_ready}",
            f"- Split labels leakage-safe: {splits_ready}",
            "- Text assignments balanced enough for exploratory analysis: documented with caveats",
            f"- Readiness judgement: {judgement}",
            "",
            "## Caveats To Carry Forward",
            "- Participant provenance is operational project metadata in v1.1.",
            "- Segmentation-opacity labels are orthographic proxies, not pronunciation-aware labels.",
            "- Parser fields remain surface_heuristic_fallback and should not be interpreted as full syntax.",
            "- DFM LM alignment warnings are documented; Gemma sensitivity remains pending because access was gated.",
            "- Text exposure and label balance require confound-controlled analysis in the next phase.",
            "",
            "## Prepared Dataset Row Counts",
            _markdown_table(
                [
                    {"table": key, "rows": value}
                    for key, value in manifest.get("row_counts", {}).items()
                ],
                ["table", "rows"],
            ),
            "",
            "## Recommended Next Analyses",
            "- Fit controlled psycholinguistic models using participant-grouped or mixed-effects designs.",
            "- Evaluate segmentation-opacity effects with word length, frequency, surprisal, entropy, and text controls.",
            "- Run parser upgrade sensitivity once DaCy/spaCy environment issues are resolved.",
            "- Keep participant-level prediction exploratory and leakage-safe.",
        ]
    )
    _write_analysis_report(dirs, "prepared_dataset_readiness_report.md", report)


def build_label_release(
    config: dict[str, Any], output_dir: str | Path | None = None, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    if get_nested(config, "label_release.no_llm_generated_labels", True) is not True:
        raise ValueError("Label Release v1.1 forbids LLM-generated core labels")
    if str(get_nested(config, "label_release.corpus_mode", "full")) != "full":
        raise ValueError("Label Release v1.1 requires full dataset mode")
    source = _source_dir(config, repo_root)
    if not source.exists():
        raise FileNotFoundError(
            f"source feature release is missing: {source}. "
            "Rebuild Feature Release v1 before running Label Release v1.1."
        )
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=repo_root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _label_dirs(config, out, repo_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    participants = build_participant_labels(config, source, out, dirs, repo_root=repo_root)
    boundaries, seg_word, seg_sentence = build_segmentation_labels(
        config, source, out, dirs, repo_root=repo_root
    )
    quality = build_quality_labels(config, source, dirs, participants, seg_word, repo_root=repo_root)
    splits = build_split_labels(config, source, dirs, participants, quality, repo_root=repo_root)
    prepared_manifest = build_prepared_dataset(
        source, out, dirs, participants, seg_word, seg_sentence, quality
    )

    word_ready = _pd().read_parquet(dirs["prepared"] / "analysis_ready_word_level_v1_1.parquet")
    sentence_ready = _pd().read_parquet(dirs["prepared"] / "analysis_ready_sentence_level_v1_1.parquet")
    write_text_assignment_balance_report(dirs, word_ready, sentence_ready)
    write_readiness_report(dirs, prepared_manifest, participants, seg_word, quality, splits)

    manifest = {
        "run_type": "label_release_v1_1",
        "status": "complete",
        "git_sha": _git_sha(repo_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "label_release_version": RELEASE_VERSION,
        "segmentation_label_version": SEGMENTATION_VERSION,
        "source_feature_release_dir": str(source),
        "output_dir": str(out),
        "row_counts": {
            "participant_labels": int(len(participants)),
            "segmentation_boundary_labels": int(len(boundaries)),
            "segmentation_word_labels": int(len(seg_word)),
            "segmentation_sentence_labels": int(len(seg_sentence)),
            "quality_labels": int(len(quality)),
            "split_labels": int(len(splits)),
            **prepared_manifest.get("row_counts", {}),
        },
        "participant_counts": dict(participants["reader_group"].value_counts().sort_index()),
        "boundary_type_counts": dict(
            boundaries.loc[~boundaries["is_sentence_initial_boundary"], "orth_boundary_type"]
            .value_counts()
            .sort_index()
        ),
        "quality_missingness": {
            "lm_missing_rate": float(quality["lm_missing"].mean()),
            "parser_missing_rate": float(quality["parser_missing"].mean()),
            "embedding_missing_rate": float(quality["embedding_missing"].mean()),
            "segmentation_missing_rate": float(quality["segmentation_label_missing"].mean()),
            "participant_label_missing_rate": float(quality["participant_label_missing"].mean()),
        },
        "prepared_dataset_manifest": prepared_manifest,
        "large_outputs_not_for_commit": [
            "labels/*.parquet",
            "prepared_dataset/*.parquet",
            "prepared_dataset/checksums.json",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    _write_md(out / "label_release_report.md", _label_release_report(manifest))

    max_bytes = int(get_nested(config, "label_release.manifest.checksum_max_bytes", 500000000))
    if bool(get_nested(config, "label_release.manifest.write_checksums", True)):
        _write_checksums(dirs["prepared"], dirs["prepared"] / "checksums.json", max_bytes=max_bytes)
        _write_checksums(out, out / "checksums.json", max_bytes=max_bytes)
    return manifest


def _label_release_report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Label Release v1.1 Report",
            "",
            f"- Status: {manifest['status']}",
            f"- Git SHA: {manifest['git_sha']}",
            f"- Slurm job ID: {manifest.get('slurm_job_id')}",
            f"- Source feature release: {manifest['source_feature_release_dir']}",
            f"- Output directory: {manifest['output_dir']}",
            "",
            "## Row Counts",
            _markdown_table(
                [{"table": key, "rows": value} for key, value in manifest["row_counts"].items()],
                ["table", "rows"],
            ),
            "",
            "## Participant Counts",
            _markdown_table(
                [
                    {"reader_group": key, "participants": value}
                    for key, value in manifest["participant_counts"].items()
                ],
                ["reader_group", "participants"],
            ),
            "",
            "## Boundary Type Counts",
            _markdown_table(
                [
                    {"orth_boundary_type": key, "count": value}
                    for key, value in manifest["boundary_type_counts"].items()
                ],
                ["orth_boundary_type", "count"],
            ),
            "",
            "## Quality Missingness",
            _markdown_table(
                [
                    {"metric": key, "value": value}
                    for key, value in manifest["quality_missingness"].items()
                ],
                ["metric", "value"],
            ),
            "",
            "## Interpretation",
            "The prepared dataset is frozen for controlled research exploration with caveats documented "
            "in the readiness report. Core labels are deterministic and do not use LLM-generated labels.",
        ]
    )


def freeze_prepared_dataset(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    source = _source_dir(config, repo_root)
    out = Path(output_dir).resolve()
    dirs = _label_dirs(config, out, repo_root)
    pd = _pd()
    participant_labels = pd.read_parquet(dirs["labels"] / "participant_labels_v1.parquet")
    seg_word = pd.read_parquet(dirs["labels"] / "segmentation_word_labels_v1.parquet")
    seg_sentence = pd.read_parquet(dirs["labels"] / "segmentation_sentence_labels_v1.parquet")
    quality = pd.read_parquet(dirs["labels"] / "quality_labels_v1.parquet")
    manifest = build_prepared_dataset(source, out, dirs, participant_labels, seg_word, seg_sentence, quality)
    _write_json(out / "prepared_dataset_freeze_manifest.json", manifest)
    return manifest


def validate_label_release(
    output_dir: str | Path, config: dict[str, Any] | None = None, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    pd = _pd()
    out = Path(output_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    labels_dir = out / "labels"
    prepared = out / "prepared_dataset"
    required_files = [
        labels_dir / "participant_labels_v1.parquet",
        labels_dir / "segmentation_boundary_labels_v1.parquet",
        labels_dir / "segmentation_word_labels_v1.parquet",
        labels_dir / "segmentation_sentence_labels_v1.parquet",
        labels_dir / "quality_labels_v1.parquet",
        labels_dir / "split_labels_v1.parquet",
        prepared / "analysis_ready_word_level_v1_1.parquet",
        prepared / "analysis_ready_sentence_level_v1_1.parquet",
        prepared / "analysis_ready_participant_level_v1_1.parquet",
        prepared / "analysis_ready_manifest.json",
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(out)}")
    if errors:
        report = {"status": "failed", "errors": errors, "warnings": warnings}
        _write_json(out / "label_release_validation_report.json", report)
        return report

    participant = pd.read_parquet(labels_dir / "participant_labels_v1.parquet")
    boundaries = pd.read_parquet(labels_dir / "segmentation_boundary_labels_v1.parquet")
    seg_word = pd.read_parquet(labels_dir / "segmentation_word_labels_v1.parquet")
    seg_sentence = pd.read_parquet(labels_dir / "segmentation_sentence_labels_v1.parquet")
    quality = pd.read_parquet(labels_dir / "quality_labels_v1.parquet")
    splits = pd.read_parquet(labels_dir / "split_labels_v1.parquet")
    word_ready = pd.read_parquet(prepared / "analysis_ready_word_level_v1_1.parquet")
    sentence_ready = pd.read_parquet(prepared / "analysis_ready_sentence_level_v1_1.parquet")
    participant_ready = pd.read_parquet(prepared / "analysis_ready_participant_level_v1_1.parquet")
    prepared_manifest = _safe_read_json(prepared / "analysis_ready_manifest.json")

    participant_required = {
        "participant_id",
        "reader_group",
        "reader_group_binary",
        "label_source",
        "diagnostic_provenance",
        "label_confidence",
        "include_primary_analysis",
        "include_sensitivity_analysis",
        "source_feature_release_dir",
        "label_release_version",
    }
    missing = sorted(participant_required.difference(participant.columns))
    if missing:
        errors.append(f"participant labels missing columns: {missing}")
    if participant["participant_id"].duplicated().any():
        errors.append("participant labels contain duplicate participant_id")
    if participant["reader_group"].isna().any():
        errors.append("participant labels contain missing reader_group")
    unexpected_groups = set(participant["reader_group"].dropna().astype(str)) - ALLOWED_READER_GROUPS
    if unexpected_groups:
        errors.append(f"unexpected reader_group values: {sorted(unexpected_groups)}")
    unexpected_provenance = set(participant["diagnostic_provenance"].dropna().astype(str)) - ALLOWED_DIAGNOSTIC_PROVENANCE
    if unexpected_provenance:
        errors.append(f"unexpected diagnostic_provenance values: {sorted(unexpected_provenance)}")
    unexpected_confidence = set(participant["label_confidence"].dropna().astype(str)) - ALLOWED_LABEL_CONFIDENCE
    if unexpected_confidence:
        errors.append(f"unexpected label_confidence values: {sorted(unexpected_confidence)}")
    if config is not None:
        expected_total = get_nested(config, "label_release.participant_labels.expected_total")
        expected_dyslexia = get_nested(config, "label_release.participant_labels.expected_dyslexia_labeled")
        expected_typical = get_nested(config, "label_release.participant_labels.expected_typical_control")
        if expected_total is not None and len(participant) != int(expected_total):
            errors.append(f"participant count {len(participant)} != expected {expected_total}")
        if (
            expected_dyslexia is not None
            and int(participant["reader_group"].eq("dyslexia_labeled").sum()) != int(expected_dyslexia)
        ):
            errors.append("dyslexia-labeled participant count does not match config expectation")
        if (
            expected_typical is not None
            and int(participant["reader_group"].eq("typical_control").sum()) != int(expected_typical)
        ):
            errors.append("typical/control participant count does not match config expectation")

    if boundaries["boundary_id"].duplicated().any():
        errors.append("segmentation boundary labels contain duplicate boundary_id")
    if seg_word["word_id"].duplicated().any():
        errors.append("segmentation word labels contain duplicate word_id")
    if seg_sentence["sentence_id"].duplicated().any():
        errors.append("segmentation sentence labels contain duplicate sentence_id")
    if "participant_word_key" not in quality or quality["participant_word_key"].isna().any():
        errors.append("quality labels missing participant_word_key")
    elif quality["participant_word_key"].duplicated().any():
        errors.append("quality labels contain duplicate participant_word_key")
    bool_columns = [
        "gaze_missing",
        "participant_label_missing",
        "segmentation_label_missing",
        "lm_missing",
        "parser_missing",
        "embedding_missing",
        "participant_metadata_missing",
        "include_primary_analysis",
        "include_sensitivity_analysis",
    ]
    for column in bool_columns:
        if column not in quality.columns:
            errors.append(f"quality labels missing boolean column {column}")
    if "surface_heuristic_fallback" not in set(quality["parser_status"].astype(str)):
        warnings.append("parser_status does not include surface_heuristic_fallback")

    if splits["split_name"].str.contains("random", case=False, na=False).any():
        errors.append("random split name found in split labels")
    for (split_name, fold_id), group in splits.groupby(["split_name", "fold_id"], dropna=False):
        train = set(group[group["split_role"].isin(["train", "include"])]["participant_id"].astype(str))
        test = set(group[group["split_role"].eq("test")]["participant_id"].astype(str))
        if train.intersection(test):
            errors.append(f"participant overlap in split {split_name} fold {fold_id}")
    lopo = splits[splits["split_name"].eq("leave_one_participant_out")]
    lopo_test_counts = lopo[lopo["split_role"].eq("test")]["participant_id"].value_counts()
    if len(lopo_test_counts) != len(participant) or not (lopo_test_counts == 1).all():
        errors.append("LOPO test coverage is not exactly once per participant")

    if word_ready["participant_word_key"].duplicated().any():
        errors.append("prepared word table contains duplicate participant_word_key")
    if "word" not in word_ready.columns:
        errors.append("prepared word table missing canonical word column")
    if "stimulus_word_key" not in word_ready.columns:
        errors.append("prepared word table missing stimulus_word_key")
    if len(word_ready) != len(quality):
        errors.append("prepared word table row count does not match quality labels")
    if participant_ready["participant_id"].duplicated().any():
        errors.append("prepared participant table contains duplicate participant_id")
    if sentence_ready["sentence_id"].duplicated().any():
        errors.append("prepared sentence table contains duplicate sentence_id")
    join_validation = prepared_manifest.get("join_validation", {})
    if join_validation.get("unexpected_row_loss", 0) != 0 or join_validation.get("unexpected_row_gain", 0) != 0:
        errors.append("prepared dataset manifest reports unexpected row loss/gain")
    if float(join_validation.get("missing_participant_label_rate", 1.0)) != 0.0:
        errors.append("participant labels did not join to all prepared word rows")
    if float(join_validation.get("missing_segmentation_word_label_rate", 1.0)) != 0.0:
        errors.append("segmentation labels did not join to all prepared word rows")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "row_counts": {
            "participant_labels": int(len(participant)),
            "segmentation_boundary_labels": int(len(boundaries)),
            "segmentation_word_labels": int(len(seg_word)),
            "segmentation_sentence_labels": int(len(seg_sentence)),
            "quality_labels": int(len(quality)),
            "split_labels": int(len(splits)),
            "analysis_ready_word_level_v1_1": int(len(word_ready)),
            "analysis_ready_sentence_level_v1_1": int(len(sentence_ready)),
            "analysis_ready_participant_level_v1_1": int(len(participant_ready)),
        },
        "participant_counts": dict(participant["reader_group"].value_counts().sort_index()),
        "boundary_type_counts": dict(
            boundaries.loc[~boundaries["is_sentence_initial_boundary"], "orth_boundary_type"]
            .value_counts()
            .sort_index()
        ),
        "quality_missingness": {
            "lm_missing_rate": float(quality["lm_missing"].mean()),
            "parser_missing_rate": float(quality["parser_missing"].mean()),
            "embedding_missing_rate": float(quality["embedding_missing"].mean()),
            "segmentation_missing_rate": float(quality["segmentation_label_missing"].mean()),
            "participant_label_missing_rate": float(quality["participant_label_missing"].mean()),
        },
        "prepared_join_validation": join_validation,
    }
    _write_json(out / "label_release_validation_report.json", report)
    return report
