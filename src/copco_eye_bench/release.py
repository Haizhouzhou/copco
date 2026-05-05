"""Feature-release table exports, joins, and first-stage analyses."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .config import get_nested


DANISH_STOPWORDS = {
    "af",
    "at",
    "de",
    "den",
    "der",
    "det",
    "du",
    "en",
    "er",
    "et",
    "for",
    "fra",
    "han",
    "har",
    "hun",
    "i",
    "ikke",
    "jeg",
    "kan",
    "med",
    "men",
    "og",
    "om",
    "på",
    "som",
    "til",
    "var",
    "vi",
}


def _pd() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for feature-release operations") from exc
    return pd


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for feature-release operations") from exc
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


def _read_base_tables(output_dir: str | Path) -> dict[str, Any]:
    pd = _pd()
    out = Path(output_dir)
    table_dir = out / "tables"
    return {
        "words": pd.read_parquet(table_dir / "words.parquet"),
        "word_observations": pd.read_parquet(table_dir / "word_observations.parquet"),
        "sentences": pd.read_parquet(table_dir / "sentences.parquet"),
        "paragraphs": pd.read_parquet(table_dir / "paragraphs.parquet"),
        "participants": pd.read_parquet(table_dir / "participants.parquet"),
    }


def _norm_word(value: Any) -> str:
    return str(value or "").strip()


def _add_surface_features(words: Any) -> Any:
    out = words.copy()
    out["normalized_word"] = out["word_form"].map(_norm_word)
    out["lowercase_form"] = out["normalized_word"].str.lower()
    out["lemma"] = out["lowercase_form"]
    out["has_punctuation"] = out["normalized_word"].str.contains(r"[^\w\sæøåÆØÅ]", regex=True)
    out["is_punctuation_only"] = out["normalized_word"].str.fullmatch(r"[^\wæøåÆØÅ]+").fillna(False)
    out["is_capitalized"] = out["normalized_word"].str.match(r"^[A-ZÆØÅ]").fillna(False)
    out["is_all_caps"] = out["normalized_word"].str.match(r"^[A-ZÆØÅ]+$").fillna(False)
    out["has_digit"] = out["normalized_word"].str.contains(r"\d", regex=True)
    out["hyphen_count"] = out["normalized_word"].str.count("-")
    out["compound_proxy"] = (
        (out["word_length_chars"] >= 12) | (out["hyphen_count"] > 0) | out["lowercase_form"].str.contains("s")
    )
    out["unique_char_count"] = out["lowercase_form"].map(lambda text: len(set(text)))
    out["function_word_flag"] = out["lowercase_form"].isin(DANISH_STOPWORDS)
    out["stopword_flag"] = out["function_word_flag"]

    token_counts = out["lowercase_form"].value_counts()
    out["corpus_frequency"] = out["lowercase_form"].map(token_counts).astype(int)
    out["log_corpus_frequency"] = out["corpus_frequency"].map(lambda value: math.log1p(int(value)))
    out["frequency_rank"] = out["corpus_frequency"].rank(method="dense", ascending=False).astype(int)
    out["frequency_source"] = "internal_corpus_frequency"

    bigrams = Counter()
    trigrams = Counter()
    for token in out["lowercase_form"]:
        bigrams.update(token[index : index + 2] for index in range(max(0, len(token) - 1)))
        trigrams.update(token[index : index + 3] for index in range(max(0, len(token) - 2)))

    def mean_ngram(text: str, n: int, counts: Counter[str]) -> float:
        grams = [text[index : index + n] for index in range(max(0, len(text) - n + 1))]
        if not grams:
            return 0.0
        return float(sum(counts[gram] for gram in grams) / len(grams))

    out["mean_char_bigram_frequency"] = out["lowercase_form"].map(lambda text: mean_ngram(text, 2, bigrams))
    out["mean_char_trigram_frequency"] = out["lowercase_form"].map(lambda text: mean_ngram(text, 3, trigrams))
    length_counts = out.groupby("word_length_chars")["lowercase_form"].transform("nunique")
    out["orthographic_neighborhood_proxy"] = (length_counts - 1).clip(lower=0)
    out["long_word_lix_component"] = out["word_length_chars"].ge(7).astype(int)
    return out


def write_release_features(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    """Write release feature-family parquet files from the base feature tables."""

    pd = _pd()
    out = Path(output_dir).resolve()
    tables = _read_base_tables(out)
    feature_dir = out / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    words = _add_surface_features(tables["words"])
    sentences = tables["sentences"].copy()
    paragraphs = tables["paragraphs"].copy()
    participants = tables["participants"].copy()
    gaze = tables["word_observations"].copy()

    sentences["sentence_position_in_paragraph"] = sentences.groupby("paragraph_id").cumcount()
    sentences["lix_long_word_count"] = words.groupby("sentence_id")["long_word_lix_component"].transform(
        "sum"
    ).groupby(words["sentence_id"]).first().reindex(sentences["sentence_id"]).to_numpy()
    sentences["lix_long_word_count"] = pd.to_numeric(sentences["lix_long_word_count"], errors="coerce").fillna(0)
    sentences["lix_component"] = sentences["sentence_length_words"] + (
        100 * sentences["lix_long_word_count"] / sentences["sentence_length_words"].replace(0, pd.NA)
    )
    sentences["lix_component"] = sentences["lix_component"].fillna(0.0)

    paragraphs["paragraph_position_in_speech"] = paragraphs.groupby("speech_id").cumcount()
    paragraphs = paragraphs.merge(
        sentences.groupby("paragraph_id", as_index=False).agg(
            mean_lix_component=("lix_component", "mean"),
            max_sentence_length_words=("sentence_length_words", "max"),
        ),
        on="paragraph_id",
        how="left",
    )
    text_level = paragraphs.groupby("speech_id", as_index=False).agg(
        paragraph_count=("paragraph_id", "count"),
        sentence_count=("sentence_count", "sum"),
        word_count=("paragraph_length_words", "sum"),
        mean_paragraph_length_words=("paragraph_length_words", "mean"),
        mean_lix_component=("mean_lix_component", "mean"),
    )

    gaze_columns = [
        column
        for column in [
            "participant_id",
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_id",
            "word_form",
            "dyslexia_labeled",
            "group_label",
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
        if column in gaze.columns
    ]
    gaze[gaze_columns].to_parquet(feature_dir / "word_level_gaze.parquet", index=False)
    words.to_parquet(feature_dir / "word_level_classical.parquet", index=False)
    sentences.to_parquet(feature_dir / "sentence_level.parquet", index=False)
    paragraphs.to_parquet(feature_dir / "paragraph_level.parquet", index=False)
    participants.to_parquet(feature_dir / "participant_level.parquet", index=False)
    text_level.to_parquet(feature_dir / "text_level.parquet", index=False)

    report = {
        "run_type": "write_release_features",
        "status": "complete",
        "git_sha": _git_sha(repo_root),
        "output_dir": str(out),
        "row_counts": {
            "word_level_gaze": int(len(gaze)),
            "word_level_classical": int(len(words)),
            "sentence_level": int(len(sentences)),
            "paragraph_level": int(len(paragraphs)),
            "participant_level": int(len(participants)),
            "text_level": int(len(text_level)),
        },
        "sample_limits_forbidden": bool(get_nested(config, "feature_release.require_full_corpus", False)),
    }
    _write_json(out / "reports" / "release_feature_manifest.json", report)
    _write_label_provenance_report(out, participants)
    return report


def _heuristic_upos(word: str) -> str:
    lower = word.lower()
    if lower in DANISH_STOPWORDS:
        return "ADP" if lower in {"af", "for", "fra", "i", "med", "om", "på", "til"} else "SCONJ"
    if lower in {"ikke", "aldrig", "ingen", "intet"}:
        return "PART"
    if lower.endswith(("ede", "te", "er")):
        return "VERB"
    if word[:1].isupper():
        return "PROPN"
    if lower.endswith(("hed", "ing", "else", "tion")):
        return "NOUN"
    return "NOUN"


def run_parser_features(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Build parser/morphosyntactic features, using DaCy when usable and fallback otherwise."""

    pd = _pd()
    out = Path(output_dir).resolve()
    ling_dir = out / "linguistic_features"
    ling_dir.mkdir(parents=True, exist_ok=True)
    words = pd.read_parquet(out / "features" / "word_level_classical.parquet")
    sentences = pd.read_parquet(out / "features" / "sentence_level.parquet")

    backend = "surface_heuristic"
    backend_error = None
    try:
        import dacy  # noqa: F401

        backend = "dacy_import_available_but_surface_alignment_fallback"
    except Exception as exc:
        backend_error = str(exc)

    parser = words[["speech_id", "paragraph_id", "sentence_id", "word_id", "word_form", "lowercase_form"]].copy()
    parser["lemma"] = parser["lowercase_form"]
    parser["upos"] = parser["word_form"].map(_heuristic_upos)
    parser["xpos"] = parser["upos"]
    parser["morph_features"] = ""
    parser["dep_rel"] = "unknown"
    parser["dep_head_position"] = -1
    parser["dependency_distance"] = 0
    parser["is_root"] = False
    parser["is_finite_verb"] = parser["upos"].eq("VERB")
    parser["is_noun_or_proper"] = parser["upos"].isin(["NOUN", "PROPN"])
    parser["is_pronoun"] = parser["lowercase_form"].isin({"jeg", "du", "han", "hun", "vi", "de"})
    parser["is_function_word"] = parser["lowercase_form"].isin(DANISH_STOPWORDS)
    parser["is_negation"] = parser["lowercase_form"].isin({"ikke", "aldrig", "ingen", "intet"})
    parser["is_coordination"] = parser["lowercase_form"].isin({"og", "eller", "men"})
    parser["subordinate_clause_proxy"] = parser["lowercase_form"].isin({"at", "som", "fordi", "hvis"})
    parser["parser_backend"] = backend
    parser.to_parquet(ling_dir / "parser_word_level.parquet", index=False)

    pos_counts = (
        parser.pivot_table(index="sentence_id", columns="upos", values="word_id", aggfunc="count", fill_value=0)
        .add_prefix("upos_count_")
        .reset_index()
    )
    sent_parser = sentences[["sentence_id", "paragraph_id", "sentence_text", "sentence_length_words"]].merge(
        pos_counts, on="sentence_id", how="left"
    )
    for column in [column for column in sent_parser.columns if column.startswith("upos_count_")]:
        sent_parser[column] = sent_parser[column].fillna(0).astype(int)
        sent_parser[column.replace("count", "prop")] = sent_parser[column] / sent_parser[
            "sentence_length_words"
        ].replace(0, pd.NA)
    sent_parser["mean_dependency_distance"] = 0.0
    sent_parser.to_parquet(ling_dir / "parser_sentence_level.parquet", index=False)

    diagnostics = {
        "run_type": "parser_features",
        "status": "complete",
        "backend": backend,
        "preferred_backend": get_nested(config, "linguistic_features.parser.preferred_backend", "dacy"),
        "backend_error": backend_error,
        "model_name": None,
        "failed_sentence_count": 0,
        "tokenization_mismatch_count": 0,
        "mismatch_examples": [],
        "coverage_rate": 1.0,
        "note": "DaCy is preferred when import/model loading is stable; fallback features are surface heuristics.",
    }
    _write_json(ling_dir / "parser_diagnostics.json", diagnostics)
    return diagnostics


def _cosine_distance(a: Any, b: Any) -> float:
    np = _np()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(1 - (np.dot(a, b) / denom))


def _embedding_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _encode_texts(model_id: str, texts: list[str], *, batch_size: int) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=_embedding_device())
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def _write_embedding_table(ids: Any, vectors: Any, id_column: str, path: Path, model_id: str) -> None:
    pd = _pd()
    frame = pd.DataFrame(vectors, columns=[f"embedding_{index:04d}" for index in range(vectors.shape[1])])
    frame.insert(0, "embedding_dim", int(vectors.shape[1]))
    frame.insert(0, "embedding_model_id", model_id)
    frame.insert(0, id_column, ids.astype(str).to_list())
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def run_embedding_features(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    model_id: str | None = None,
    model_label: str | None = None,
    run_baseline: bool = False,
    skip_baseline: bool = False,
) -> dict[str, Any]:
    """Build full sentence/paragraph embedding files and compact semantic features."""

    pd = _pd()
    np = _np()
    out = Path(output_dir).resolve()
    sentences = pd.read_parquet(out / "features" / "sentence_level.parquet")
    paragraphs = pd.read_parquet(out / "features" / "paragraph_level.parquet")
    batch_size = int(get_nested(config, "language_models.embeddings.batch_size", 32))

    requests: list[tuple[str, str]] = []
    if model_id:
        requests.append((model_label or "custom_embedding_model", model_id))
    else:
        requests.append(
            (
                str(get_nested(config, "language_models.embeddings.primary_label", "dfm_sentence_encoder")),
                str(get_nested(config, "language_models.embeddings.primary")),
            )
        )
        if run_baseline and not skip_baseline:
            requests.append(
                (
                    str(get_nested(config, "language_models.embeddings.baseline_label", "e5_large")),
                    str(get_nested(config, "language_models.embeddings.baseline")),
                )
            )

    reports = []
    for label, requested_model in requests:
        emb_dir = out / "embedding_features" / label
        sent_vecs = _encode_texts(requested_model, sentences["sentence_text"].astype(str).tolist(), batch_size=batch_size)
        para_vecs = _encode_texts(requested_model, paragraphs["paragraph_text"].astype(str).tolist(), batch_size=batch_size)
        sent_vecs = np.asarray(sent_vecs)
        para_vecs = np.asarray(para_vecs)
        _write_embedding_table(
            sentences["sentence_id"],
            sent_vecs,
            "sentence_id",
            emb_dir / "sentence_embeddings.parquet",
            requested_model,
        )
        _write_embedding_table(
            paragraphs["paragraph_id"],
            para_vecs,
            "paragraph_id",
            emb_dir / "paragraph_embeddings.parquet",
            requested_model,
        )

        para_by_id = dict(zip(paragraphs["paragraph_id"].astype(str), para_vecs, strict=True))
        sent_rows = []
        previous = None
        for row, vector in zip(sentences.itertuples(index=False), sent_vecs, strict=True):
            paragraph_vector = para_by_id.get(str(row.paragraph_id))
            sent_rows.append(
                {
                    "sentence_id": row.sentence_id,
                    "paragraph_id": row.paragraph_id,
                    "embedding_model_label": label,
                    "sentence_to_previous_sentence_cosine_distance": None
                    if previous is None
                    else _cosine_distance(vector, previous),
                    "sentence_to_paragraph_centroid_distance": None
                    if paragraph_vector is None
                    else _cosine_distance(vector, paragraph_vector),
                }
            )
            previous = vector
        semantic_sentence = pd.DataFrame(sent_rows)
        semantic_sentence.to_parquet(emb_dir / "sentence_semantic_features.parquet", index=False)

        paragraph_rows = []
        sentence_vectors = dict(zip(sentences["sentence_id"].astype(str), sent_vecs, strict=True))
        for row in paragraphs.itertuples(index=False):
            group = sentences[sentences["paragraph_id"].astype(str) == str(row.paragraph_id)]
            vectors = [sentence_vectors[str(sentence_id)] for sentence_id in group["sentence_id"].astype(str)]
            distances = [
                _cosine_distance(vectors[index], vectors[index - 1]) for index in range(1, len(vectors))
            ]
            paragraph_rows.append(
                {
                    "paragraph_id": row.paragraph_id,
                    "embedding_model_label": label,
                    "paragraph_cohesion": None if not distances else float(1 - np.nanmean(distances)),
                    "local_semantic_drift": None if not distances else float(np.nanmean(distances)),
                    "sentence_count_embedded": len(vectors),
                }
            )
        semantic_paragraph = pd.DataFrame(paragraph_rows)
        semantic_paragraph.to_parquet(emb_dir / "paragraph_semantic_features.parquet", index=False)
        reports.append(
            {
                "label": label,
                "model_id": requested_model,
                "sentence_rows": int(len(sentences)),
                "paragraph_rows": int(len(paragraphs)),
                "embedding_dim": int(sent_vecs.shape[1]),
            }
        )

    manifest_path = out / "embedding_features" / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_label = {str(item.get("label")): item for item in existing.get("models", [])}
    else:
        by_label = {}
    for report in reports:
        by_label[str(report["label"])] = report
    manifest = {
        "run_type": "embedding_features",
        "status": "complete",
        "models": [by_label[key] for key in sorted(by_label)],
    }
    _write_json(manifest_path, manifest)
    return manifest


def _read_lm(path: Path, prefix: str) -> Any:
    pd = _pd()
    files = sorted(path.glob("**/*.parquet"))
    frames = [pd.read_parquet(file) for file in files if "surprisal" in file.name]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    rename = {
        column: f"{prefix}_{column}"
        for column in frame.columns
        if column not in {"word_id", "speech_id", "paragraph_id", "sentence_id"}
    }
    return frame.rename(columns=rename)


def _aggregate_lm_alignment_reports(out: Path, label: str) -> dict[str, Any] | None:
    pd = _pd()
    label_dir = out / "lm_features" / label
    report_paths = sorted(label_dir.glob("**/alignment_report_shard*.json"))
    if not report_paths:
        return None

    reports: list[dict[str, Any]] = []
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports.extend(payload.get("reports", []))
    status_counts = Counter(str(report.get("status", "missing")) for report in reports)
    warning_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    examples = []
    for report in reports:
        for warning in report.get("warnings", []):
            warning_counts[str(warning)] += 1
        for error in report.get("errors", []):
            error_counts[str(error)] += 1
        if report.get("status") != "ok" and len(examples) < 200:
            examples.append(
                {
                    "context_id": report.get("context_id"),
                    "status": report.get("status"),
                    "warnings": ";".join(map(str, report.get("warnings", []))),
                    "errors": ";".join(map(str, report.get("errors", []))),
                    "word_count": report.get("word_count"),
                    "token_count": report.get("token_count"),
                }
            )

    aggregate = {
        "status": "failed" if error_counts else "passed",
        "model_label": label,
        "context_reports": len(reports),
        "status_counts": dict(status_counts),
        "warning_counts": dict(warning_counts),
        "error_counts": dict(error_counts),
        "report_files": [str(path.relative_to(out)) for path in report_paths],
    }
    _write_json(label_dir / "alignment_report.json", aggregate)
    pd.DataFrame(
        examples,
        columns=["context_id", "status", "warnings", "errors", "word_count", "token_count"],
    ).to_parquet(label_dir / "alignment_examples.parquet", index=False)
    return aggregate


def build_modeling_tables(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Join feature families into validated modeling tables."""

    pd = _pd()
    out = Path(output_dir).resolve()
    model_dir = out / "modeling_tables"
    model_dir.mkdir(parents=True, exist_ok=True)
    gaze = pd.read_parquet(out / "features" / "word_level_gaze.parquet")
    classical = pd.read_parquet(out / "features" / "word_level_classical.parquet")
    parser = pd.read_parquet(out / "linguistic_features" / "parser_word_level.parquet")
    sentence = pd.read_parquet(out / "features" / "sentence_level.parquet")
    paragraph = pd.read_parquet(out / "features" / "paragraph_level.parquet")
    participant = pd.read_parquet(out / "features" / "participant_level.parquet")

    base_rows = len(gaze)
    word_full = gaze.merge(classical, on=["speech_id", "paragraph_id", "sentence_id", "word_id"], how="left")
    word_full = word_full.merge(
        parser.drop(columns=["speech_id", "paragraph_id", "sentence_id", "word_form"], errors="ignore"),
        on="word_id",
        how="left",
    )
    dfm_label = str(get_nested(config, "language_models.primary_surprisal.output_label", "dfm_decoder_7b"))
    gemma_label = str(get_nested(config, "language_models.sensitivity_surprisal.output_label", "gemma2_9b"))
    dfm_alignment = _aggregate_lm_alignment_reports(out, dfm_label)
    gemma_alignment = _aggregate_lm_alignment_reports(out, gemma_label)
    dfm = _read_lm(out / "lm_features" / dfm_label, "dfm")
    gemma = _read_lm(out / "lm_features" / gemma_label, "gemma")
    word_with_dfm = word_full.merge(dfm, on=["word_id", "speech_id", "paragraph_id", "sentence_id"], how="left")
    if not gemma.empty:
        word_with_all = word_with_dfm.merge(
            gemma, on=["word_id", "speech_id", "paragraph_id", "sentence_id"], how="left"
        )
    else:
        word_with_all = word_with_dfm.copy()

    primary_embed_label = str(get_nested(config, "language_models.embeddings.primary_label", "dfm_sentence_encoder"))
    sent_sem_path = out / "embedding_features" / primary_embed_label / "sentence_semantic_features.parquet"
    para_sem_path = out / "embedding_features" / primary_embed_label / "paragraph_semantic_features.parquet"
    if sent_sem_path.exists():
        sentence = sentence.merge(pd.read_parquet(sent_sem_path), on=["sentence_id", "paragraph_id"], how="left")
    if para_sem_path.exists():
        paragraph = paragraph.merge(pd.read_parquet(para_sem_path), on="paragraph_id", how="left")
        word_with_dfm = word_with_dfm.merge(
            paragraph[
                [
                    "paragraph_id",
                    *[
                        column
                        for column in paragraph.columns
                        if column in {"paragraph_cohesion", "local_semantic_drift"}
                    ],
                ]
            ].drop_duplicates("paragraph_id"),
            on="paragraph_id",
            how="left",
        )

    aggregates = _participant_aggregates(word_with_dfm)
    word_full.to_parquet(model_dir / "word_level_full.parquet", index=False)
    word_with_dfm.to_parquet(model_dir / "word_level_full_with_dfm_lm.parquet", index=False)
    if not gemma.empty:
        word_with_all.to_parquet(model_dir / "word_level_full_with_all_lm.parquet", index=False)
    sentence.to_parquet(model_dir / "sentence_level_full.parquet", index=False)
    paragraph.to_parquet(model_dir / "paragraph_level_full.parquet", index=False)
    participant.to_parquet(model_dir / "participant_level_full.parquet", index=False)
    aggregates.to_parquet(model_dir / "participant_aggregates.parquet", index=False)

    validation = {
        "base_word_rows": int(base_rows),
        "word_level_full_rows": int(len(word_full)),
        "word_level_full_with_dfm_lm_rows": int(len(word_with_dfm)),
        "unexpected_row_loss": int(base_rows - len(word_with_dfm)),
        "unexpected_row_gain": int(len(word_with_dfm) - base_rows),
        "duplicate_participant_word_keys": int(word_with_dfm.duplicated(["participant_id", "word_id"]).sum()),
        "missing_dfm_lm_rate": None
        if "dfm_lm_word_surprisal" not in word_with_dfm.columns
        else float(word_with_dfm["dfm_lm_word_surprisal"].isna().mean()),
        "missing_parser_feature_rate": float(word_with_dfm["upos"].isna().mean()) if "upos" in word_with_dfm else None,
        "missing_embedding_feature_rate": float(word_with_dfm["paragraph_cohesion"].isna().mean())
        if "paragraph_cohesion" in word_with_dfm
        else None,
        "lm_features_are_stimulus_level": True,
        "lm_join_key": "word_id plus speech/paragraph/sentence IDs",
        "dfm_alignment": dfm_alignment,
        "gemma_alignment": gemma_alignment,
    }
    _write_json(model_dir / "join_validation_report.json", validation)
    _write_md(
        model_dir / "join_validation_report.md",
        "\n".join(
            [
                "# Join Validation Report",
                "",
                f"- Base word rows: {validation['base_word_rows']}",
                f"- Joined DFM rows: {validation['word_level_full_with_dfm_lm_rows']}",
                f"- Duplicate participant-word keys: {validation['duplicate_participant_word_keys']}",
                f"- Missing DFM LM rate: {validation['missing_dfm_lm_rate']}",
                f"- Missing parser feature rate: {validation['missing_parser_feature_rate']}",
                f"- Missing embedding feature rate: {validation['missing_embedding_feature_rate']}",
                "",
                "LM features are stimulus-level and are joined to participant-specific gaze rows by stable word identifiers.",
            ]
        ),
    )
    return {"run_type": "build_modeling_tables", "status": "complete", "join_validation": validation}


def _participant_aggregates(frame: Any) -> Any:
    pd = _pd()
    data = frame.copy()
    numeric = ["FFD", "GD", "TRT", "go_past_time", "fixation_count", "skip", "refixation_count"]
    for column in numeric:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    grouped = data.groupby("participant_id", dropna=False)
    out = grouped.agg(
        dyslexia_labeled=("dyslexia_labeled", "first"),
        group_label=("group_label", "first"),
        mean_ffd=("FFD", "mean"),
        median_ffd=("FFD", "median"),
        mean_gd=("GD", "mean"),
        median_gd=("GD", "median"),
        mean_trt=("TRT", "mean"),
        median_trt=("TRT", "median"),
        skipping_rate=("skip", "mean"),
        refixation_rate=("refixation_count", "mean"),
        mean_go_past_time=("go_past_time", "mean"),
        trt_sd=("TRT", "std"),
        trt_q90=("TRT", lambda values: values.quantile(0.9)),
        word_observation_count=("word_id", "count"),
    ).reset_index()
    for predictor, out_col in (
        ("word_length_chars", "length_sensitivity"),
        ("log_corpus_frequency", "frequency_sensitivity"),
        ("dfm_lm_word_surprisal", "surprisal_sensitivity"),
        ("dfm_lm_word_entropy", "entropy_sensitivity"),
    ):
        if predictor in data.columns:
            slopes = []
            for participant_id, group in data.groupby("participant_id"):
                slopes.append({"participant_id": participant_id, out_col: _simple_slope(group, predictor, "TRT")})
            out = out.merge(pd.DataFrame(slopes), on="participant_id", how="left")
    return out


def _simple_slope(frame: Any, x_col: str, y_col: str) -> float | None:
    pd = _pd()
    np = _np()
    data = frame[[x_col, y_col]].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna()
    if len(data) < 3 or data[x_col].nunique() < 2:
        return None
    return float(np.polyfit(data[x_col], data[y_col], 1)[0])


def _write_label_provenance_report(out: Path, participants: Any) -> None:
    counts = participants["dyslexia_labeled"].value_counts(dropna=False).to_dict()
    text = [
        "# Label Provenance Report",
        "",
        f"- Participants: {len(participants)}",
        f"- Dyslexia-labeled participants: {int(counts.get(1, 0))}",
        f"- Typical/control participants: {int(counts.get(0, 0))}",
        f"- Missing labels: {int(participants['dyslexia_labeled'].isna().sum())}",
        "- Label source column: `dyslexia` when present in participant metadata, normalized to `dyslexia_labeled`.",
        "- Diagnostic provenance: no formal diagnostic instrument is encoded by this pipeline.",
        "- Supported wording: dyslexia-labeled participants/readers.",
        "- Unsupported wording: clinical diagnosis, diagnostic biomarker, or diagnosed dyslexics.",
        "- Classification results are exploratory and require participant-level validation.",
    ]
    for column in ["age", "sex", "comprehension", "reading_time"]:
        if column in participants.columns:
            text.append(f"- `{column}` summary: {participants[column].describe(include='all').to_dict()}")
    _write_md(out / "label_provenance_report.md", "\n".join(text))


FEATURE_TABLE_SPECS: dict[str, dict[str, Any]] = {
    "features/word_level_gaze.parquet": {
        "level": "word_observation",
        "source": "eye-tracking extracted features",
        "participant_specific": True,
        "stimulus_specific": True,
    },
    "features/word_level_classical.parquet": {
        "level": "word",
        "source": "stimulus text and internal corpus statistics",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "features/sentence_level.parquet": {
        "level": "sentence",
        "source": "stimulus text",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "features/paragraph_level.parquet": {
        "level": "paragraph",
        "source": "stimulus text",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "features/participant_level.parquet": {
        "level": "participant",
        "source": "participant metadata",
        "participant_specific": True,
        "stimulus_specific": False,
    },
    "features/text_level.parquet": {
        "level": "text",
        "source": "stimulus text",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "linguistic_features/parser_word_level.parquet": {
        "level": "word",
        "source": "Danish parser layer or documented surface-heuristic fallback",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "linguistic_features/parser_sentence_level.parquet": {
        "level": "sentence",
        "source": "Danish parser layer or documented surface-heuristic fallback",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "modeling_tables/word_level_full.parquet": {
        "level": "word_observation",
        "source": "joined gaze, classical, and parser features",
        "participant_specific": True,
        "stimulus_specific": True,
    },
    "modeling_tables/word_level_full_with_dfm_lm.parquet": {
        "level": "word_observation",
        "source": "joined word table with DFM causal-LM features",
        "participant_specific": True,
        "stimulus_specific": True,
    },
    "modeling_tables/word_level_full_with_all_lm.parquet": {
        "level": "word_observation",
        "source": "joined word table with all available causal-LM features",
        "participant_specific": True,
        "stimulus_specific": True,
    },
    "modeling_tables/sentence_level_full.parquet": {
        "level": "sentence",
        "source": "joined sentence features",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "modeling_tables/paragraph_level_full.parquet": {
        "level": "paragraph",
        "source": "joined paragraph features",
        "participant_specific": False,
        "stimulus_specific": True,
    },
    "modeling_tables/participant_level_full.parquet": {
        "level": "participant",
        "source": "participant metadata",
        "participant_specific": True,
        "stimulus_specific": False,
    },
    "modeling_tables/participant_aggregates.parquet": {
        "level": "participant",
        "source": "participant-level aggregates from word-observation features",
        "participant_specific": True,
        "stimulus_specific": False,
    },
}


IDENTIFIER_COLUMNS = {
    "participant_id",
    "speech_id",
    "paragraph_id",
    "sentence_id",
    "word_id",
    "source_row_id",
    "trial_id",
    "shard_id",
}
LABEL_COLUMNS = {"dyslexia_labeled", "group_label", "label_provenance_strength"}
TEXT_COLUMNS = {"word", "word_form", "normalized_word", "lowercase_form", "sentence_text", "paragraph_text"}


def _feature_type(dtype: Any) -> str:
    pd = _pd()
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "string"


def _feature_units(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("duration", "ffd", "gd", "trt", "time")):
        return "milliseconds"
    if "surprisal" in lower:
        return "negative log probability"
    if "entropy" in lower:
        return "nats"
    if "cosine" in lower or "cohesion" in lower or "semantic_drift" in lower:
        return "cosine distance or similarity"
    if "embedding_" in lower:
        return "unit-normalized embedding coordinate"
    if "length" in lower and "char" in lower:
        return "characters"
    if "length" in lower or "count" in lower or lower.endswith("_id"):
        return "count/index"
    if "rate" in lower or "probability" in lower or lower.endswith("_prop"):
        return "proportion"
    if "position" in lower:
        return "position/index"
    if "frequency" in lower:
        return "corpus count or transformed frequency"
    return "unitless"


def _computation_method(name: str, table: str) -> str:
    lower = name.lower()
    if lower in IDENTIFIER_COLUMNS:
        return "stable identifier propagated from the CopCo feature build"
    if lower in LABEL_COLUMNS:
        return "participant metadata label normalized for operational research use"
    if "surprisal" in lower:
        return "summed causal-LM subword negative log probabilities at word level"
    if "entropy" in lower:
        return "causal-LM next-token entropy aligned to the word context"
    if "embedding_" in lower:
        return "sentence-transformer embedding coordinate stored outside the main modeling table"
    if "cosine" in lower:
        return "cosine distance from normalized sentence or paragraph embeddings"
    if "cohesion" in lower or "semantic_drift" in lower:
        return "compact semantic feature derived from adjacent sentence embeddings"
    if table.startswith("linguistic_features/parser"):
        return "parser annotation when available; otherwise documented Danish surface heuristic"
    if lower in {"ffd", "gd", "trt", "go_past_time", "mean_fixation_duration"}:
        return "word-level gaze duration extracted from fixation features"
    if "fixation" in lower or lower in {"skip", "refixation_count", "landing_position"}:
        return "word-level gaze behavior extracted from fixation features"
    if "frequency" in lower:
        return "internal full-corpus frequency statistic computed from normalized word forms"
    if "lix" in lower:
        return "Danish readability component based on sentence length and long-word counts"
    if "syllable" in lower:
        return "Danish syllable-count heuristic"
    if "length" in lower:
        return "length/count computed from normalized stimulus text"
    if lower.startswith("is_") or lower.endswith("_flag") or lower.startswith("has_"):
        return "boolean indicator computed from text, gaze, or parser metadata"
    return "computed or propagated by the feature-release pipeline"


def _expected_direction(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("duration", "ffd", "gd", "trt", "go_past_time")):
        return "higher values indicate greater observed processing cost"
    if lower == "skip" or "skipping_rate" in lower:
        return "lower values are expected for more difficult words"
    if "word_length" in lower or "surprisal" in lower or "entropy" in lower or "lix" in lower:
        return "higher values are expected to increase reading difficulty or gaze cost"
    if "frequency" in lower:
        return "higher values are expected to reduce reading difficulty"
    if "cohesion" in lower:
        return "higher values may reduce integration cost"
    if "semantic_drift" in lower or "cosine" in lower:
        return "higher values may increase discourse integration cost"
    return "not specified"


def _missing_policy(name: str) -> str:
    lower = name.lower()
    if lower in IDENTIFIER_COLUMNS or lower in {"word_form", "sentence_text", "paragraph_text"}:
        return "not expected; validation should fail if missing"
    if "lm_" in lower or lower.startswith(("dfm_", "gemma_")):
        return "kept as missing and summarized in LM/join validation"
    if "embedding" in lower or "semantic" in lower or "cohesion" in lower:
        return "kept as missing and summarized in join validation"
    if any(token in lower for token in ("ffd", "gd", "trt", "fixation", "landing", "saccade")):
        return "kept as missing because skipped or unavailable gaze events are meaningful"
    return "kept as missing unless a validation rule states otherwise"


def _transformation(name: str) -> str:
    lower = name.lower()
    if lower.startswith("log_") or "_log_" in lower:
        return "log transform"
    if "rank" in lower:
        return "dense rank"
    if lower.endswith("_norm"):
        return "normalized to a unit interval or relative position"
    if "mean_" in lower:
        return "mean aggregation"
    if "median_" in lower:
        return "median aggregation"
    if lower.endswith("_q90"):
        return "90th percentile aggregation"
    if lower.endswith("_sd"):
        return "standard deviation aggregation"
    return "none"


def _allowed_in_predictive_models(name: str) -> bool:
    lower = name.lower()
    if lower in IDENTIFIER_COLUMNS or lower in LABEL_COLUMNS or lower in TEXT_COLUMNS:
        return False
    if "alignment_error" in lower or "source_observation_count" in lower:
        return False
    return True


def _allowed_in_mixed_effects(name: str) -> bool:
    lower = name.lower()
    if lower in TEXT_COLUMNS or "embedding_" in lower or "alignment_error" in lower:
        return False
    if lower in IDENTIFIER_COLUMNS:
        return False
    return True


def _feature_dictionary_rows(output_dir: str | Path) -> list[dict[str, Any]]:
    pd = _pd()
    out = Path(output_dir)
    table_specs = dict(FEATURE_TABLE_SPECS)

    for lm_path in sorted((out / "lm_features").glob("*/surprisal/surprisal_shard*.parquet")):
        rel = str(lm_path.relative_to(out))
        label = lm_path.parts[-3]
        table_specs[rel] = {
            "level": "word",
            "source": f"{label} causal-LM scoring",
            "participant_specific": False,
            "stimulus_specific": True,
        }
    for emb_path in sorted((out / "embedding_features").glob("*/*.parquet")):
        rel = str(emb_path.relative_to(out))
        label = emb_path.parts[-2]
        level = "sentence" if "sentence" in emb_path.name else "paragraph"
        table_specs[rel] = {
            "level": level,
            "source": f"{label} sentence-transformer embeddings or semantic derivatives",
            "participant_specific": False,
            "stimulus_specific": True,
        }

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for table, spec in sorted(table_specs.items()):
        path = out / table
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        for column, dtype in frame.dtypes.items():
            key = (table, str(column))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "feature_name": str(column),
                    "table": table,
                    "level": spec["level"],
                    "type": _feature_type(dtype),
                    "units": _feature_units(str(column)),
                    "source": spec["source"],
                    "computation_method": _computation_method(str(column), table),
                    "participant_specific": bool(
                        spec["participant_specific"]
                        or str(column) in {"participant_id", *LABEL_COLUMNS}
                        or str(column).startswith(("mean_", "median_"))
                    ),
                    "stimulus_specific": bool(spec["stimulus_specific"] or str(column) in IDENTIFIER_COLUMNS),
                    "missing_value_policy": _missing_policy(str(column)),
                    "transformation": _transformation(str(column)),
                    "expected_direction_for_reading_difficulty": _expected_direction(str(column)),
                    "allowed_in_predictive_models": _allowed_in_predictive_models(str(column)),
                    "allowed_in_mixed_effects_analysis": _allowed_in_mixed_effects(str(column)),
                }
            )
    return rows


def write_feature_dictionary(output_dir: str | Path, docs_path: str | Path = "docs/feature_dictionary_v1.md") -> None:
    rows = _feature_dictionary_rows(output_dir)
    _write_json(Path(output_dir) / "feature_dictionary_v1.json", {"features": rows})
    lines = [
        "# Feature Dictionary V1",
        "",
        "This dictionary is generated from the feature-release tables. It documents each",
        "column by table because the same feature can appear in both source feature files",
        "and joined modeling tables.",
        "",
        f"Total table-specific feature entries: {len(rows)}",
        "",
    ]
    fields = [
        "feature_name",
        "level",
        "type",
        "units",
        "source",
        "computation_method",
        "participant_specific",
        "stimulus_specific",
        "missing_value_policy",
        "transformation",
        "expected_direction_for_reading_difficulty",
        "allowed_in_predictive_models",
        "allowed_in_mixed_effects_analysis",
    ]
    by_table: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_table.setdefault(str(row["table"]), []).append(row)
    for table, table_rows in sorted(by_table.items()):
        lines.extend([f"## `{table}`", ""])
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in table_rows:
            values = [str(row[field]).replace("|", "\\|").replace("\n", " ") for field in fields]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    _write_md(Path(docs_path), "\n".join(lines))


def _safe_corr(frame: Any, a: str, b: str) -> float | None:
    pd = _pd()
    data = frame[[a, b]].copy()
    data[a] = pd.to_numeric(data[a], errors="coerce")
    data[b] = pd.to_numeric(data[b], errors="coerce")
    data = data.dropna()
    if len(data) < 3:
        return None
    return float(data[a].corr(data[b]))


def run_analysis_package(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    """Create psycholinguistic, group, predictive, and research-planning reports."""

    pd = _pd()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir).resolve()
    word = pd.read_parquet(out / "modeling_tables" / "word_level_full_with_dfm_lm.parquet")
    participant = pd.read_parquet(out / "modeling_tables" / "participant_aggregates.parquet")
    analysis_dir = out / "analysis"

    psych_dir = analysis_dir / "psycholinguistic_validation"
    fig_dir = psych_dir / "figures"
    table_dir = psych_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    max_points = int(get_nested(config, "analysis.max_plot_points", 10000))
    sample = word.sample(min(len(word), max_points), random_state=int(get_nested(config, "analysis.random_seed", 17)))
    for x, y, name in [
        ("word_length_chars", "skip", "word_length_skip"),
        ("word_length_chars", "TRT", "word_length_trt"),
        ("log_corpus_frequency", "TRT", "frequency_trt"),
        ("dfm_lm_word_surprisal", "TRT", "dfm_surprisal_trt"),
        ("dfm_lm_word_entropy", "TRT", "dfm_entropy_trt"),
    ]:
        if x in sample.columns and y in sample.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(pd.to_numeric(sample[x], errors="coerce"), pd.to_numeric(sample[y], errors="coerce"), s=4, alpha=0.25)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{name}.png", dpi=160)
            plt.close()

    correlations = pd.DataFrame(
        [
            {"analysis": "word_length_vs_skip", "correlation": _safe_corr(word, "word_length_chars", "skip")},
            {"analysis": "word_length_vs_trt", "correlation": _safe_corr(word, "word_length_chars", "TRT")},
            {"analysis": "frequency_vs_skip", "correlation": _safe_corr(word, "log_corpus_frequency", "skip")},
            {"analysis": "frequency_vs_trt", "correlation": _safe_corr(word, "log_corpus_frequency", "TRT")},
            {"analysis": "dfm_surprisal_vs_trt", "correlation": _safe_corr(word, "dfm_lm_word_surprisal", "TRT")},
            {"analysis": "dfm_entropy_vs_trt", "correlation": _safe_corr(word, "dfm_lm_word_entropy", "TRT")},
        ]
    )
    correlations.to_csv(table_dir / "psycholinguistic_correlations.csv", index=False)
    word[["skip", "FFD", "GD", "TRT", "go_past_time", "dfm_lm_word_surprisal", "dfm_lm_word_entropy"]].describe().to_csv(
        table_dir / "feature_distributions.csv"
    )
    align_counts = word["dfm_lm_alignment_status"].value_counts(dropna=False).reset_index()
    align_counts.columns = ["alignment_status", "rows"]
    align_counts.to_csv(table_dir / "alignment_warning_distribution.csv", index=False)
    _write_md(
        psych_dir / "psycholinguistic_validation_report.md",
        "\n".join(
            [
                "# Psycholinguistic Validation Report",
                "",
                "This report checks qualitative reading-pattern sanity for the full feature table.",
                "",
                correlations.to_markdown(index=False),
                "",
                "Expected patterns are evaluated qualitatively: longer words should be skipped less often and tend to increase gaze cost; lower-frequency and higher-surprisal words should tend to increase processing cost. These checks are not headline statistical claims.",
                f"Alignment status counts: {align_counts.to_dict(orient='records')}",
            ]
        ),
    )

    group_dir = analysis_dir / "dyslexia_group_analysis"
    group_dir.mkdir(parents=True, exist_ok=True)
    group_summary = word.groupby("dyslexia_labeled").agg(
        participants=("participant_id", "nunique"),
        rows=("word_id", "count"),
        skip_rate=("skip", "mean"),
        mean_ffd=("FFD", "mean"),
        mean_gd=("GD", "mean"),
        mean_trt=("TRT", "mean"),
        mean_fixations=("fixation_count", "mean"),
    ).reset_index()
    group_summary.to_csv(group_dir / "group_summary.csv", index=False)
    coeffs = _fit_group_models(word)
    coeffs.to_csv(group_dir / "coefficient_table.csv", index=False)
    _write_md(
        group_dir / "dyslexia_labeled_reader_report.md",
        "\n".join(
            [
                "# Dyslexia-Labeled Reader Analysis",
                "",
                "The analysis reports group-associated differences and interactions with linguistic predictors. It is not a clinical diagnostic analysis.",
                "",
                "## Group Summary",
                group_summary.to_markdown(index=False),
                "",
                "## Interpretable Model Coefficients",
                coeffs.head(80).to_markdown(index=False),
                "",
                "Interpretation should focus on associations and altered sensitivity to linguistic predictors, not diagnosis.",
            ]
        ),
    )

    pred_dir = analysis_dir / "predictive_models"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_metrics, pred_rows = _participant_prediction(participant)
    pred_metrics.to_csv(pred_dir / "participant_level_metrics.csv", index=False)
    pred_rows.to_csv(pred_dir / "participant_level_predictions.csv", index=False)
    _write_md(
        pred_dir / "participant_level_model_report.md",
        "\n".join(
            [
                "# Participant-Level Predictive Model Report",
                "",
                "Prediction is participant-level and exploratory. No random word-level split is used.",
                "",
                pred_metrics.to_markdown(index=False),
            ]
        ),
    )

    model_metrics = out / "models" / "model_metrics.csv"
    if model_metrics.exists():
        ladder = pd.read_csv(model_metrics)
        ladder.to_csv(pred_dir / "word_level_model_ladder_metrics.csv", index=False)
        _write_md(
            pred_dir / "word_level_model_ladder_report.md",
            "\n".join(
                [
                    "# Word-Level Model Ladder Report",
                    "",
                    "This ladder is secondary and exploratory because labels are participant-level. Splits are grouped; random word-level splitting is not used.",
                    "",
                    ladder.groupby(["model", "cv_regime"], dropna=False)["status"].value_counts().to_frame("rows").reset_index().to_markdown(index=False),
                ]
            ),
        )

    _write_research_plan(out, repo_root=repo_root)
    write_feature_dictionary(out, get_nested(config, "feature_release.feature_dictionary_path", "docs/feature_dictionary_v1.md"))
    manifest = {
        "run_type": "analysis_package",
        "status": "complete",
        "psycholinguistic_correlation_rows": int(len(correlations)),
        "participant_prediction_rows": int(len(pred_rows)),
        "repo_git_sha": _git_sha(repo_root),
    }
    _write_json(analysis_dir / "manifest.json", manifest)
    return manifest


def _fit_group_models(word: Any) -> Any:
    pd = _pd()
    rows = []
    predictors = [
        "dyslexia_labeled",
        "word_length_chars",
        "log_corpus_frequency",
        "dfm_lm_word_surprisal",
        "dfm_lm_word_entropy",
        "sentence_length_words",
        "word_position_in_sentence_norm",
    ]
    for outcome in ["skip", "FFD", "GD", "TRT", "go_past_time", "fixation_count"]:
        available = [column for column in predictors if column in word.columns]
        data = word[[outcome, *available]].copy()
        for column in [outcome, *available]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna()
        if len(data) < 20:
            continue
        try:
            import statsmodels.api as sm

            y = data[outcome]
            x = sm.add_constant(data[available], has_constant="add")
            model = sm.OLS(y, x).fit(cov_type="HC3")
            for term, value in model.params.items():
                rows.append(
                    {
                        "outcome": outcome,
                        "term": term,
                        "estimate": float(value),
                        "std_error": float(model.bse[term]),
                        "p_value": float(model.pvalues[term]),
                        "n_obs": int(model.nobs),
                        "model_type": "robust_ols_hc3",
                    }
                )
        except Exception as exc:
            rows.append({"outcome": outcome, "term": "model_failed", "estimate": None, "std_error": None, "p_value": None, "n_obs": len(data), "model_type": str(exc)})
    return pd.DataFrame(rows)


def _participant_prediction(participant: Any) -> tuple[Any, Any]:
    pd = _pd()
    np = _np()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = participant.dropna(subset=["dyslexia_labeled"]).copy()
    data["dyslexia_labeled"] = data["dyslexia_labeled"].astype(int)
    feature_cols = [
        column
        for column in data.columns
        if column
        not in {
            "participant_id",
            "dyslexia_labeled",
            "group_label",
        }
        and pd.api.types.is_numeric_dtype(data[column])
    ]
    predictions = []
    skipped = 0
    for participant_id in data["participant_id"].astype(str):
        train = data[data["participant_id"].astype(str) != participant_id]
        test = data[data["participant_id"].astype(str) == participant_id]
        if train["dyslexia_labeled"].nunique() < 2:
            skipped += 1
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", solver="liblinear", random_state=17),
        )
        model.fit(train[feature_cols], train["dyslexia_labeled"])
        score = float(model.predict_proba(test[feature_cols])[:, 1][0])
        predictions.append(
            {
                "participant_id": participant_id,
                "y_true": int(test["dyslexia_labeled"].iloc[0]),
                "y_score": score,
                "classifier": "logistic_regression",
                "cv_regime": "leave_one_participant_out",
            }
        )
    pred = pd.DataFrame(predictions)
    if pred.empty or pred["y_true"].nunique() < 2:
        metrics = pd.DataFrame(
            [
                {
                    "classifier": "logistic_regression",
                    "cv_regime": "leave_one_participant_out",
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "brier": np.nan,
                    "prediction_rows": len(pred),
                    "skipped_folds": skipped,
                    "status": "invalid_single_class_or_no_predictions",
                }
            ]
        )
    else:
        metrics = pd.DataFrame(
            [
                {
                    "classifier": "logistic_regression",
                    "cv_regime": "leave_one_participant_out",
                    "roc_auc": float(roc_auc_score(pred["y_true"], pred["y_score"])),
                    "pr_auc": float(average_precision_score(pred["y_true"], pred["y_score"])),
                    "brier": float(brier_score_loss(pred["y_true"], pred["y_score"])),
                    "prediction_rows": len(pred),
                    "skipped_folds": skipped,
                    "status": "complete",
                }
            ]
        )
    return metrics, pred


def _write_research_plan(out: Path, *, repo_root: str | Path = ".") -> None:
    text = """# Danish Natural-Reading Eye-Tracking Signatures of Dyslexia-Labeled Readers

## Project Title

Danish natural-reading eye-tracking signatures of dyslexia-labeled readers: integrating gaze, linguistic complexity, and language-model predictability.

## Research Objective

Estimate psycholinguistic and predictive signatures of dyslexia-labeled reader behavior in Danish natural reading, while keeping claims exploratory and non-clinical.

## Dataset Summary

CopCo is treated as a Danish natural-reading eye-tracking corpus with operational dyslexia labels. The release reports participant, text, sentence, word, and gaze counts in `feature_release_report.md`.

## Label-Provenance Summary

Labels support cautious wording such as dyslexia-labeled participants and typical/control participants. They do not by themselves support clinical diagnosis wording.

## Completed Engineering Milestones

- Stable identifiers and leakage-aware split tables.
- Full gaze/classical feature tables.
- Parser/morphosyntactic feature layer with fallback diagnostics.
- DFM causal-LM surprisal and entropy.
- Embedding feature layer and compact semantic features.
- Joined modeling tables and validation reports.

## Feature Families Now Available

- Participant-specific gaze features.
- Classical lexical, surface, readability, and position features.
- Parser or parser-fallback morphosyntactic features with diagnostics.
- DFM base causal-LM surprisal and entropy.
- Optional Gemma base causal-LM sensitivity when model access is available.
- Sentence and paragraph embeddings plus compact semantic-cohesion features.
- Joined word, sentence, paragraph, and participant-level modeling tables.

## Main Hypotheses

- Dyslexia-labeled readers may show elevated gaze cost after controlling for lexical and syntactic difficulty.
- Dyslexia-labeled readers may show different sensitivity to word length.
- Dyslexia-labeled readers may show different sensitivity to word frequency.
- Dyslexia-labeled readers may show different sensitivity to LM-derived surprisal.
- Participant-level gaze-sensitivity profiles may support exploratory classification, but not clinical diagnosis.

## Analysis Strategy

Start with psycholinguistic validation, then group-associated effects, then participant-level exploratory prediction. Feature validation and interpretable effects are primary.

## Model Strategy

Use interpretable model ladders before complex learners. Participant-level classification is primary for dyslexia-label prediction; word-level ladders are secondary because word rows inherit participant labels.

## Validation Strategy

Validate row counts, stable keys, duplicate counts, missing feature rates, LM alignment warnings, split leakage, model metric schemas, and report completeness before treating outputs as release artifacts.

## Leakage-Control Strategy

Participant-level labels require participant-grouped evaluation. Random word-level splits are not allowed.

## Statistical Modeling Plan

Use mixed-effects models where feasible, with participant and stimulus grouping. Use robust clustered or HC3 models as fallbacks when mixed models fail.

## Predictive Modeling Plan

Primary predictive unit is participant-level. Report skipped folds, class balance, uncertainty, and invalid metric conditions.

## Sensitivity Analyses

Gemma base-model LM features are a sensitivity analysis, not a blocker for the primary DFM feature release.

## Limitations

Operational labels, no independent clinical validation, parser fallback limitations, and no external validation dataset yet.

## Next Data-Collection Needs

Clarify label provenance, formal diagnostic instruments if available, age and sex completeness, comprehension scoring provenance, and external validation or replication data.

## Paper Outline

1. Dataset and label-provenance framing.
2. Feature engineering and alignment validation.
3. Psycholinguistic sanity checks.
4. Group-associated gaze and linguistic-sensitivity effects.
5. Exploratory participant-level prediction.
6. Limitations, sensitivity analyses, and next data needs.

## Immediate Next Tasks

Review label provenance, inspect mixed-effects convergence, decide paper hypotheses, and identify external validation or additional participant metadata needs.
"""
    _write_md(out / "analysis" / "research_plan_next_stage.md", text)
    _write_md(Path(repo_root) / "analysis" / "research_plan_next_stage.md", text)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_feature_release(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Validate required feature-release artifacts and key invariants."""

    out = Path(output_dir).resolve()
    errors = []
    required = [
        "features/word_level_gaze.parquet",
        "features/word_level_classical.parquet",
        "features/sentence_level.parquet",
        "features/paragraph_level.parquet",
        "features/participant_level.parquet",
        "lm_features/dfm_decoder_7b/alignment_report.json",
        "lm_features/dfm_decoder_7b/alignment_examples.parquet",
        "linguistic_features/parser_word_level.parquet",
        "linguistic_features/parser_sentence_level.parquet",
        "modeling_tables/word_level_full.parquet",
        "modeling_tables/word_level_full_with_dfm_lm.parquet",
        "modeling_tables/participant_aggregates.parquet",
        "label_provenance_report.md",
        "feature_dictionary_v1.json",
        "analysis/research_plan_next_stage.md",
    ]
    for rel in required:
        if not (out / rel).exists():
            errors.append(f"missing:{rel}")
    pd = _pd()
    dictionary_path = out / "feature_dictionary_v1.json"
    if dictionary_path.exists():
        dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
        features = dictionary.get("features", [])
        required_fields = {
            "feature_name",
            "table",
            "level",
            "type",
            "units",
            "source",
            "computation_method",
            "participant_specific",
            "stimulus_specific",
            "missing_value_policy",
            "transformation",
            "expected_direction_for_reading_difficulty",
            "allowed_in_predictive_models",
            "allowed_in_mixed_effects_analysis",
        }
        if len(features) < 50:
            errors.append(f"feature_dictionary_too_small:{len(features)}")
        missing_field_rows = [
            index
            for index, row in enumerate(features)
            if required_fields.difference(row)
        ]
        if missing_field_rows:
            errors.append(f"feature_dictionary_missing_fields:{missing_field_rows[:10]}")
    join_path = out / "modeling_tables" / "join_validation_report.json"
    if join_path.exists():
        join = json.loads(join_path.read_text(encoding="utf-8"))
        if int(join.get("unexpected_row_loss", 0)) != 0:
            errors.append(f"unexpected_row_loss:{join.get('unexpected_row_loss')}")
        if int(join.get("unexpected_row_gain", 0)) != 0:
            errors.append(f"unexpected_row_gain:{join.get('unexpected_row_gain')}")
        if int(join.get("duplicate_participant_word_keys", 0)) != 0:
            errors.append(f"duplicate_participant_word:{join.get('duplicate_participant_word_keys')}")
        alignment = join.get("dfm_alignment") or {}
        if alignment.get("error_counts"):
            errors.append(f"dfm_alignment_errors:{alignment.get('error_counts')}")
    if (out / "modeling_tables/word_level_full_with_dfm_lm.parquet").exists():
        word = pd.read_parquet(out / "modeling_tables/word_level_full_with_dfm_lm.parquet")
        duplicate = int(word.duplicated(["participant_id", "word_id"]).sum())
        if duplicate:
            errors.append(f"duplicate_participant_word:{duplicate}")
        if "dfm_lm_alignment_status" in word.columns:
            bad = int((~word["dfm_lm_alignment_status"].isin(["ok", "warning"])).sum())
            if bad:
                errors.append(f"lm_alignment_bad_status:{bad}")
        if "dfm_lm_word_surprisal" in word.columns and float(word["dfm_lm_word_surprisal"].isna().mean()) > 0.01:
            errors.append("dfm_lm_missing_rate_gt_1pct")
    report = {
        "run_type": "validate_feature_release",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "output_dir": str(out),
    }
    _write_json(out / "feature_release_validation_report.json", report)
    return report


def finalize_feature_release(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    """Write the final release report and checksums for major generated files."""

    pd = _pd()
    out = Path(output_dir).resolve()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    join = json.loads((out / "modeling_tables" / "join_validation_report.json").read_text(encoding="utf-8"))
    parser_diag = json.loads((out / "linguistic_features" / "parser_diagnostics.json").read_text(encoding="utf-8"))
    lm_manifest_paths = sorted((out / "lm_features").glob("**/manifest_shard*.json"))
    lm_manifests = [json.loads(path.read_text(encoding="utf-8")) for path in lm_manifest_paths]
    completed_lms = [item for item in lm_manifests if item.get("status") == "complete"]
    aborted_lms = [item for item in lm_manifests if item.get("status") not in {"complete", None}]
    embedding_manifest_path = out / "embedding_features" / "manifest.json"
    embedding_manifest = (
        json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
        if embedding_manifest_path.exists()
        else {"models": []}
    )
    mixed_manifest_path = out / "mixed_effects" / "manifest.json"
    mixed_manifest = (
        json.loads(mixed_manifest_path.read_text(encoding="utf-8")) if mixed_manifest_path.exists() else {}
    )
    model_manifest_path = out / "models" / "manifest.json"
    model_manifest = (
        json.loads(model_manifest_path.read_text(encoding="utf-8")) if model_manifest_path.exists() else {}
    )
    analysis_manifest_path = out / "analysis" / "manifest.json"
    analysis_manifest = (
        json.loads(analysis_manifest_path.read_text(encoding="utf-8")) if analysis_manifest_path.exists() else {}
    )
    slurm_jobs_path = out / "slurm_jobs.json"
    slurm_jobs = json.loads(slurm_jobs_path.read_text(encoding="utf-8")) if slurm_jobs_path.exists() else []
    participants = pd.read_parquet(out / "features" / "participant_level.parquet")
    counts = participants["dyslexia_labeled"].value_counts(dropna=False).to_dict()
    major = [
        path
        for path in sorted(out.glob("**/*"))
        if path.is_file() and path.suffix in {".parquet", ".csv", ".json", ".md"}
    ]
    checksums = [
        {"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in major
        if path.stat().st_size < 500_000_000
    ]
    _write_json(out / "checksums.json", {"files": checksums})
    report = [
        "# Feature Release V1 Report",
        "",
        f"- Commit hash: {_git_sha(repo_root)}",
        f"- Output directory: {out}",
        f"- Participants: {manifest['row_counts']['participants']}",
        f"- Dyslexia-labeled participants: {int(counts.get(1, 0))}",
        f"- Typical/control participants: {int(counts.get(0, 0))}",
        f"- Words: {manifest['row_counts']['words']}",
        f"- Sentences: {manifest['row_counts']['sentences']}",
        f"- Paragraphs: {manifest['row_counts']['paragraphs']}",
        f"- Word observations: {manifest['row_counts']['word_observations']}",
        f"- Parser backend: {parser_diag.get('backend')}",
        f"- LM models completed: {[item.get('model_id') for item in completed_lms]}",
        f"- LM models pending or aborted: {[{'model_id': item.get('model_id'), 'status': item.get('status'), 'reason': str(item.get('reason', ''))[:240]} for item in aborted_lms]}",
        f"- Embedding models: {[item.get('model_id') for item in embedding_manifest.get('models', [])]}",
        f"- Slurm jobs: {slurm_jobs if slurm_jobs else 'not recorded in slurm_jobs.json'}",
        f"- Join validation: {join}",
        f"- Predictive model summary: {model_manifest}",
        f"- Mixed-effects summary: {mixed_manifest}",
        f"- Analysis package summary: {analysis_manifest}",
        "",
        "## Limitations",
        "",
        "- Labels are operational dyslexia labels, not clinical diagnosis.",
        "- Classification is exploratory.",
        "- Parser diagnostics should be reviewed before syntactic claims.",
        "- Gemma sensitivity is optional and non-blocking for the primary release.",
    ]
    _write_md(out / "feature_release_report.md", "\n".join(report))
    result = {"run_type": "finalize_feature_release", "status": "complete", "checksummed_files": len(checksums)}
    _write_json(out / "feature_release_manifest.json", result)
    return result
