"""Language-model feature generation helpers."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import get_nested


INSTRUCT_MARKERS = ("instruct", "-it", "chat")


@dataclass(frozen=True)
class WordSpan:
    word_id: str
    start: int
    end: int
    word_form: str | None = None
    speech_id: str | None = None
    paragraph_id: str | None = None
    sentence_id: str | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def align_token_offsets_to_word_spans(
    token_offsets: Iterable[tuple[int, int]], word_spans: Iterable[WordSpan]
) -> list[str | None]:
    """Assign tokenizer offsets to source words by character-span overlap."""

    spans = list(word_spans)
    assignments: list[str | None] = []
    for token_start, token_end in token_offsets:
        if token_end <= token_start:
            assignments.append(None)
            continue
        best_id: str | None = None
        best_overlap = 0
        for span in spans:
            overlap = max(0, min(token_end, span.end) - max(token_start, span.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = span.word_id
        assignments.append(best_id if best_overlap > 0 else None)
    return assignments


def _normalized_alignment_text(text: str) -> str:
    """Normalize only whitespace for source/span reconstruction checks."""

    return " ".join(str(text).split())


def rebuild_text_from_word_spans(word_spans: Iterable[WordSpan]) -> str:
    """Reconstruct the CopCo normalized text represented by ordered word spans."""

    words = []
    for span in word_spans:
        if span.word_form is None:
            return ""
        words.append(str(span.word_form))
    return " ".join(words)


def rebuild_prefix_word_spans(word_spans: Iterable[WordSpan]) -> tuple[str, list[WordSpan]]:
    """Rebuild a text prefix and fresh offsets from ordered word forms."""

    rebuilt: list[WordSpan] = []
    position = 0
    for span in word_spans:
        word = "" if span.word_form is None else str(span.word_form)
        start = position
        end = start + len(word)
        rebuilt.append(
            WordSpan(
                span.word_id,
                start,
                end,
                word,
                span.speech_id,
                span.paragraph_id,
                span.sentence_id,
            )
        )
        position = end + 1
    return " ".join(str(span.word_form) for span in rebuilt), rebuilt


def validate_token_offsets_to_word_spans(
    text: str,
    token_offsets: Iterable[tuple[int, int]],
    word_spans: Iterable[WordSpan],
    assignments: Iterable[str | None] | None = None,
    *,
    context_id: str,
) -> dict[str, Any]:
    """Validate tokenizer offset alignment before LM scores are trusted."""

    spans = list(word_spans)
    offsets = [(int(start), int(end)) for start, end in token_offsets]
    assigned = (
        list(assignments)
        if assignments is not None
        else align_token_offsets_to_word_spans(offsets, spans)
    )
    errors: list[str] = []
    warnings: list[str] = []

    if len(assigned) != len(offsets):
        errors.append("assignment_count_does_not_match_token_count")

    ids = [str(span.word_id) for span in spans]
    missing_ids = [index for index, value in enumerate(ids) if not value or value.lower() == "nan"]
    if missing_ids:
        errors.append(f"missing_stable_word_ids:{len(missing_ids)}")
    duplicate_ids = sorted({word_id for word_id in ids if ids.count(word_id) > 1})
    if duplicate_ids:
        errors.append(f"duplicate_stable_word_ids:{len(duplicate_ids)}")

    reconstructed = rebuild_text_from_word_spans(spans)
    if reconstructed and _normalized_alignment_text(reconstructed) != _normalized_alignment_text(text):
        errors.append("reconstructed_text_mismatch")

    for span in spans:
        if span.start < 0 or span.end < span.start or span.end > len(text):
            errors.append(f"invalid_word_span:{span.word_id}")
            continue
        if span.word_form is not None and text[span.start : span.end] != str(span.word_form):
            errors.append(f"word_span_text_mismatch:{span.word_id}")

    subword_counts = {word_id: 0 for word_id in ids}
    for (start, end), word_id in zip(offsets, assigned, strict=False):
        if end <= start:
            continue
        if start < 0 or end > len(text):
            warnings.append("token_offset_outside_text")
        if word_id is None:
            warnings.append("non_special_token_unassigned")
            continue
        if word_id not in subword_counts:
            errors.append(f"token_assigned_to_unknown_word:{word_id}")
            continue
        subword_counts[word_id] += 1

    zero_subword_ids = [word_id for word_id, count in subword_counts.items() if count == 0]
    if zero_subword_ids:
        errors.append(f"zero_subword_words:{len(zero_subword_ids)}")

    deduped_warnings = sorted(set(warnings))
    return {
        "context_id": context_id,
        "status": "error" if errors else ("warning" if deduped_warnings else "ok"),
        "errors": errors,
        "warnings": deduped_warnings,
        "word_count": len(spans),
        "token_count": len(offsets),
        "word_subword_counts": subword_counts,
        "reconstructed_text_matches": not reconstructed
        or _normalized_alignment_text(reconstructed) == _normalized_alignment_text(text),
    }


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for LM feature IO") from exc
    return pd


def _read_tables(output_dir: Path) -> tuple[Any, Any, Any]:
    pd = _require_pandas()
    tables = output_dir / "tables"
    words_path = tables / "words.parquet"
    paragraphs_path = tables / "paragraphs.parquet"
    sentences_path = tables / "sentences.parquet"
    missing = [str(path) for path in (words_path, paragraphs_path, sentences_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"feature tables missing for LM generation: {missing}")
    return pd.read_parquet(words_path), pd.read_parquet(paragraphs_path), pd.read_parquet(sentences_path)


def _paragraph_word_spans(words: Any, paragraph_id: str) -> list[WordSpan]:
    subset = words[words["paragraph_id"].astype(str) == str(paragraph_id)].sort_values(
        ["speechId", "paragraphId", "sentenceId", "wordId"], kind="mergesort"
    )
    return [
        WordSpan(
            str(row.word_id),
            int(row.word_start_in_paragraph),
            int(row.word_end_in_paragraph),
            str(row.word_form),
            str(row.speech_id),
            str(row.paragraph_id),
            str(row.sentence_id),
        )
        for row in subset.itertuples(index=False)
    ]


def _limit_source_by_word_budget(source: Any, max_word_tokens: int | None) -> tuple[Any, int]:
    pd = _require_pandas()
    if max_word_tokens is None:
        if "paragraph_length_words" in source.columns:
            return source, int(source["paragraph_length_words"].fillna(0).astype(int).sum())
        return source, int(len(source))
    if max_word_tokens <= 0:
        raise ValueError("--max-word-tokens must be positive when provided")
    if source.empty:
        return source, 0

    selected: list[Any] = []
    planned = 0
    for index, row in source.iterrows():
        value = row.get("paragraph_length_words", 1)
        length = 1 if pd.isna(value) else int(value)
        if planned >= max_word_tokens:
            break
        selected.append(index)
        planned += min(length, max_word_tokens - planned)
    return source.loc[selected].copy(), int(planned)


def _model_is_instruct(model_id: str) -> bool:
    lower = model_id.lower()
    return any(marker in lower for marker in INSTRUCT_MARKERS)


def _gpu_status() -> dict[str, Any]:
    script = r"""
import json
status = {"torch_available": False, "cuda_available": False, "device_count": 0}
try:
    import torch
    status["torch_available"] = True
    status["cuda_available"] = bool(torch.cuda.is_available())
    status["device_count"] = int(torch.cuda.device_count())
    status["devices"] = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
        }
        for index in range(torch.cuda.device_count())
    ]
except BaseException as exc:
    status["error"] = str(exc)
print(json.dumps(status, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except Exception as exc:
        return {
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "error": f"gpu_probe_subprocess_failed:{exc}",
        }
    if result.returncode != 0:
        return {
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "error": f"gpu_probe_returncode:{result.returncode}",
            "stderr": result.stderr.strip()[-1000:],
            "stdout": result.stdout.strip()[-1000:],
        }
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "error": f"gpu_probe_parse_failed:{exc}",
            "stderr": result.stderr.strip()[-1000:],
            "stdout": result.stdout.strip()[-1000:],
        }


def run_lm_features(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    model_id: str | None = None,
    model_label: str | None = None,
    feature_kind: str = "surprisal",
    shard_index: int = 0,
    shard_count: int = 1,
    limit_items: int | None = None,
    max_word_tokens: int | None = None,
    dry_run: bool = False,
    force_real_run: bool = False,
    require_gpu: bool = False,
) -> dict[str, Any]:
    """Generate paragraph-sharded LM features, or a manifest-only dry run."""

    pd = _require_pandas()
    out = Path(output_dir).resolve()
    chosen_model = model_id or str(
        get_nested(
            config,
            "language_models.primary_surprisal.model_id",
            "danish-foundation-models/dfm-decoder-open-v0-7b-pt",
        )
    )
    chosen_label = model_label or str(
        get_nested(config, "language_models.primary_surprisal.output_label", "")
    )
    lm_dir = out / "lm_features" / chosen_label / feature_kind if chosen_label else out / "lm_features" / feature_kind
    lm_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_type": "lm_features",
        "feature_kind": feature_kind,
        "model_id": chosen_model,
        "model_label": chosen_label or None,
        "tokenizer_id": chosen_model,
        "context_mode": str(get_nested(config, "language_models.context_mode", "paragraph")),
        "dtype": str(get_nested(config, "language_models.dtype", "float16")),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "dry_run": dry_run,
        "config_dry_run": bool(get_nested(config, "language_models.dry_run", False)),
        "force_real_run": force_real_run,
        "require_gpu": require_gpu,
        "limit_items": limit_items,
        "max_word_tokens": max_word_tokens,
        "claim_language": "base LMs only for surprisal; instruct annotations are optional ablations",
    }

    if feature_kind == "surprisal" and _model_is_instruct(chosen_model):
        raise ValueError(f"instruction-tuned model cannot be used for surprisal: {chosen_model}")

    gpu = _gpu_status()
    manifest["gpu_status"] = gpu
    if require_gpu and int(gpu.get("device_count", 0)) == 0:
        manifest["status"] = "aborted"
        manifest["reason"] = "require_gpu_set_but_torch_sees_zero_cuda_devices"
        _write_json(lm_dir / f"manifest_shard{shard_index}.json", manifest)
        raise RuntimeError(manifest["reason"])

    words, paragraphs, sentences = _read_tables(out)
    if feature_kind == "embeddings":
        source = sentences
        id_column = "sentence_id"
        text_column = "sentence_text"
    else:
        source = paragraphs
        id_column = "paragraph_id"
        text_column = "paragraph_text"

    source = source.sort_values(id_column).reset_index(drop=True)
    assigned = source.iloc[shard_index::shard_count].copy()
    if limit_items is not None:
        assigned = assigned.head(limit_items).copy()
    if max_word_tokens is None:
        config_max_word_tokens = get_nested(config, "language_models.max_word_tokens")
        if config_max_word_tokens is not None:
            max_word_tokens = int(config_max_word_tokens)
            manifest["max_word_tokens"] = max_word_tokens
    assigned, planned_word_tokens = _limit_source_by_word_budget(assigned, max_word_tokens)
    manifest["assigned_items"] = int(len(assigned))
    manifest["planned_word_tokens"] = int(planned_word_tokens)

    config_dry_run = bool(get_nested(config, "language_models.dry_run", False))
    if dry_run or (config_dry_run and not force_real_run):
        manifest["status"] = "dry_run_complete"
        manifest["output_rows"] = 0
        _write_json(lm_dir / f"manifest_shard{shard_index}.json", manifest)
        return manifest

    alignment_reports: list[dict[str, Any]] = []
    try:
        if feature_kind == "embeddings":
            rows = _run_sentence_embeddings(assigned, chosen_model, id_column, text_column)
        elif feature_kind == "surprisal":
            rows = _run_surprisal(
                words,
                assigned,
                chosen_model,
                id_column,
                text_column,
                context_mode=str(get_nested(config, "language_models.context_mode", "paragraph")),
                max_word_tokens=max_word_tokens,
                alignment_reports=alignment_reports,
            )
        else:
            manifest["status"] = "skipped"
            manifest["reason"] = f"unsupported_feature_kind:{feature_kind}"
            _write_json(lm_dir / f"manifest_shard{shard_index}.json", manifest)
            return manifest
    except Exception as exc:
        manifest["status"] = "aborted"
        manifest["reason"] = str(exc)
        _write_json(lm_dir / f"manifest_shard{shard_index}.json", manifest)
        raise

    for row in rows:
        if "shard_id" in row:
            row["shard_id"] = shard_index
    frame = pd.DataFrame(rows)
    output_path = lm_dir / f"{feature_kind}_shard{shard_index:04d}_of_{shard_count:04d}.parquet"
    frame.to_parquet(output_path, index=False)
    if alignment_reports:
        alignment_path = lm_dir / f"alignment_report_shard{shard_index}.json"
        alignment_payload = {
            "status": "passed"
            if all(report["status"] in {"ok", "warning"} for report in alignment_reports)
            else "failed",
            "reports": alignment_reports,
        }
        _write_json(alignment_path, alignment_payload)
        manifest["alignment_report_path"] = str(alignment_path)
        manifest["alignment_status_counts"] = {
            status: sum(1 for report in alignment_reports if report["status"] == status)
            for status in sorted({report["status"] for report in alignment_reports})
        }
    manifest["status"] = "complete"
    manifest["output_path"] = str(output_path)
    manifest["output_rows"] = int(len(frame))
    manifest["output_columns"] = list(frame.columns)
    _write_json(lm_dir / f"manifest_shard{shard_index}.json", manifest)
    return manifest


def _run_sentence_embeddings(source: Any, model_id: str, id_column: str, text_column: str) -> list[dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("sentence-transformers is required for embedding features") from exc

    model = SentenceTransformer(model_id)
    vectors = model.encode(source[text_column].astype(str).tolist(), normalize_embeddings=True)
    rows: list[dict[str, Any]] = []
    for row, vector in zip(source.itertuples(index=False), vectors, strict=True):
        base = {
            id_column: getattr(row, id_column),
            "model_id": model_id,
            "embedding_dim": int(len(vector)),
        }
        for index, value in enumerate(vector):
            base[f"embedding_{index:04d}"] = float(value)
        rows.append(base)
    return rows


def _run_surprisal(
    words: Any,
    source: Any,
    model_id: str,
    id_column: str,
    text_column: str,
    *,
    context_mode: str,
    max_word_tokens: int | None,
    alignment_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and transformers are required for surprisal features") from exc

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise RuntimeError("surprisal scoring requires CUDA; CPU fallback is disabled")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"fast tokenizer with offset mapping is required: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    rows: list[dict[str, Any]] = []
    remaining_words = max_word_tokens
    for row in source.itertuples(index=False):
        item_id = str(getattr(row, id_column))
        spans = _paragraph_word_spans(words, item_id)
        if remaining_words is not None:
            if remaining_words <= 0:
                break
            spans = spans[:remaining_words]
            remaining_words -= len(spans)
            text, spans = rebuild_prefix_word_spans(spans)
        else:
            text = str(getattr(row, text_column))
        if not spans:
            continue

        encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
        offsets = [(int(a), int(b)) for a, b in encoded.pop("offset_mapping")[0].tolist()]
        assignments = align_token_offsets_to_word_spans(offsets, spans)
        alignment_report = validate_token_offsets_to_word_spans(
            text, offsets, spans, assignments, context_id=item_id
        )
        alignment_reports.append(alignment_report)
        if alignment_report["errors"]:
            raise ValueError(f"LM alignment failed for {item_id}: {alignment_report['errors']}")

        input_device = next(model.parameters()).device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            logits = output.logits[0]
            input_ids = encoded["input_ids"][0]
            log_probs = torch.log_softmax(logits[:-1], dim=-1)
            next_ids = input_ids[1:]
            token_nll = -log_probs.gather(1, next_ids.unsqueeze(1)).squeeze(1)
            entropy = -(torch.softmax(logits[:-1], dim=-1) * log_probs).sum(dim=-1)

        word_values: dict[str, dict[str, Any]] = {
            span.word_id: {
                "speech_id": span.speech_id,
                "paragraph_id": span.paragraph_id,
                "sentence_id": span.sentence_id,
                "lm_word_surprisal": 0.0,
                "lm_word_entropy_sum": 0.0,
                "lm_scored_subword_count": 0,
                "lm_subword_count": int(alignment_report["word_subword_counts"].get(span.word_id, 0)),
                "lm_word_entropy_onset": math.nan,
            }
            for span in spans
        }
        # Prediction at position i-1 produces token i, so assignments shift by one.
        for token_pos, word_id in enumerate(assignments[1:]):
            if word_id is None:
                continue
            values = word_values[word_id]
            values["lm_word_surprisal"] += float(token_nll[token_pos].item())
            values["lm_word_entropy_sum"] += float(entropy[token_pos].item())
            values["lm_scored_subword_count"] += 1
            if math.isnan(values["lm_word_entropy_onset"]):
                values["lm_word_entropy_onset"] = float(entropy[token_pos].item())

        alignment_status = "warning" if alignment_report["warnings"] else "ok"
        alignment_error = ";".join(alignment_report["warnings"]) or None
        context_tokens = int(encoded["input_ids"].shape[1])
        tokenizer_id = str(getattr(tokenizer, "name_or_path", model_id))
        for word_id, values in word_values.items():
            scored = int(values["lm_scored_subword_count"])
            if scored:
                word_entropy = float(values["lm_word_entropy_sum"]) / scored
                word_surprisal = float(values["lm_word_surprisal"])
            else:
                word_entropy = math.nan
                word_surprisal = math.nan
            rows.append(
                {
                    "word_id": word_id,
                    "speech_id": values["speech_id"],
                    "paragraph_id": values["paragraph_id"] or item_id,
                    "sentence_id": values["sentence_id"],
                    "lm_model_id": model_id,
                    "lm_tokenizer_id": tokenizer_id,
                    "lm_context_mode": context_mode,
                    "lm_context_tokens": context_tokens,
                    "lm_word_surprisal": word_surprisal,
                    "lm_word_entropy": word_entropy,
                    "lm_word_entropy_onset": values["lm_word_entropy_onset"],
                    "lm_subword_count": int(values["lm_subword_count"]),
                    "lm_scored_subword_count": scored,
                    "lm_alignment_status": alignment_status,
                    "lm_alignment_warning": alignment_error,
                    "lm_alignment_error": None,
                    "shard_id": None,
                    # Backward-compatible names used by existing model/mixed-effects scaffolds.
                    "model_id": model_id,
                    "surprisal_paragraph_context": word_surprisal,
                    "entropy_word_onset": values["lm_word_entropy_onset"],
                }
            )
    return rows
