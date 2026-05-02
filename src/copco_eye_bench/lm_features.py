"""Language-model feature generation helpers."""

from __future__ import annotations

import json
import math
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
        WordSpan(str(row.word_id), int(row.word_start_in_paragraph), int(row.word_end_in_paragraph))
        for row in subset.itertuples(index=False)
    ]


def _model_is_instruct(model_id: str) -> bool:
    lower = model_id.lower()
    return any(marker in lower for marker in INSTRUCT_MARKERS)


def _gpu_status() -> dict[str, Any]:
    status: dict[str, Any] = {"torch_available": False, "cuda_available": False, "device_count": 0}
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
    except Exception as exc:
        status["error"] = str(exc)
    return status


def run_lm_features(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    model_id: str | None = None,
    feature_kind: str = "surprisal",
    shard_index: int = 0,
    shard_count: int = 1,
    limit_items: int | None = None,
    dry_run: bool = False,
    require_gpu: bool = False,
) -> dict[str, Any]:
    """Generate paragraph-sharded LM features, or a manifest-only dry run."""

    pd = _require_pandas()
    out = Path(output_dir).resolve()
    lm_dir = out / "lm_features" / feature_kind
    lm_dir.mkdir(parents=True, exist_ok=True)
    chosen_model = model_id or str(
        get_nested(
            config,
            "language_models.primary_surprisal.model_id",
            "danish-foundation-models/dfm-decoder-open-v0-7b-pt",
        )
    )
    manifest: dict[str, Any] = {
        "run_type": "lm_features",
        "feature_kind": feature_kind,
        "model_id": chosen_model,
        "dtype": str(get_nested(config, "language_models.dtype", "float16")),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "dry_run": dry_run,
        "require_gpu": require_gpu,
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
    manifest["assigned_items"] = int(len(assigned))

    if dry_run or bool(get_nested(config, "language_models.dry_run", False)):
        manifest["status"] = "dry_run_complete"
        manifest["output_rows"] = 0
        _write_json(lm_dir / f"manifest_shard{shard_index}.json", manifest)
        return manifest

    if feature_kind == "embeddings":
        rows = _run_sentence_embeddings(assigned, chosen_model, id_column, text_column)
    elif feature_kind == "surprisal":
        rows = _run_surprisal(words, assigned, chosen_model, id_column, text_column)
    else:
        manifest["status"] = "skipped"
        manifest["reason"] = f"unsupported_feature_kind:{feature_kind}"
        _write_json(lm_dir / f"manifest_shard{shard_index}.json", manifest)
        return manifest

    frame = pd.DataFrame(rows)
    output_path = lm_dir / f"{feature_kind}_shard{shard_index:04d}_of_{shard_count:04d}.parquet"
    frame.to_parquet(output_path, index=False)
    manifest["status"] = "complete"
    manifest["output_path"] = str(output_path)
    manifest["output_rows"] = int(len(frame))
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
    words: Any, source: Any, model_id: str, id_column: str, text_column: str
) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and transformers are required for surprisal features") from exc

    if not torch.cuda.is_available():
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
    for row in source.itertuples(index=False):
        item_id = str(getattr(row, id_column))
        text = str(getattr(row, text_column))
        spans = _paragraph_word_spans(words, item_id)
        encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = [(int(a), int(b)) for a, b in encoded.pop("offset_mapping")[0].tolist()]
        assignments = align_token_offsets_to_word_spans(offsets, spans)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            logits = output.logits[0]
            input_ids = encoded["input_ids"][0]
            log_probs = torch.log_softmax(logits[:-1], dim=-1)
            next_ids = input_ids[1:]
            token_nll = -log_probs.gather(1, next_ids.unsqueeze(1)).squeeze(1)
            entropy = -(torch.softmax(logits[:-1], dim=-1) * log_probs).sum(dim=-1)

        word_values: dict[str, dict[str, float]] = {}
        # Prediction at position i-1 produces token i, so assignments shift by one.
        for token_pos, word_id in enumerate(assignments[1:]):
            if word_id is None:
                continue
            values = word_values.setdefault(
                word_id,
                {
                    "surprisal_paragraph_context": 0.0,
                    "entropy_sum": 0.0,
                    "subtoken_count": 0.0,
                    "entropy_word_onset": math.nan,
                },
            )
            values["surprisal_paragraph_context"] += float(token_nll[token_pos].item())
            values["entropy_sum"] += float(entropy[token_pos].item())
            values["subtoken_count"] += 1.0
            if math.isnan(values["entropy_word_onset"]):
                values["entropy_word_onset"] = float(entropy[token_pos].item())

        for word_id, values in word_values.items():
            rows.append({"word_id": word_id, "paragraph_id": item_id, "model_id": model_id, **values})
    return rows
