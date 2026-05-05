from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.label_release import (
    DANISH_VOWELS,
    build_label_release,
    classify_boundary,
    normalize_orth_token,
    validate_label_release,
    within_word_vowel_run_max,
)


def test_danish_vowel_and_boundary_helpers() -> None:
    assert {"æ", "ø", "å", "Æ", "Ø", "Å"}.issubset(DANISH_VOWELS)
    assert normalize_orth_token('"Åben!"') == "Åben"
    assert classify_boundary("tak", "for")["orth_boundary_type"] == "C#C"
    assert classify_boundary("tak", "om")["orth_boundary_type"] == "C#V"
    assert classify_boundary("de", "går")["orth_boundary_type"] == "V#C"
    vv = classify_boundary("se", "efter")
    assert vv["orth_boundary_type"] == "V#V"
    assert vv["vocoid_run_cross_boundary"] == 2
    assert classify_boundary(None, "Hej", sentence_initial=True)["orth_boundary_type"] == "unknown"
    assert classify_boundary("", "Hej")["orth_boundary_type"] == "unknown"
    assert within_word_vowel_run_max("saaeed") == 4


def _write_minimal_feature_release(root: Path) -> None:
    for subdir in ["features", "modeling_tables", "linguistic_features"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    participants = pd.DataFrame(
        {
            "participant_id": ["P01", "P02"],
            "dyslexia_labeled": [0, 1],
            "group_label": ["typical", "dyslexia_labeled"],
            "label_provenance": ["operational_metadata", "operational_metadata"],
            "age": [29, 31],
            "sex": ["F", "M"],
            "comprehension_accuracy": [0.9, 0.8],
            "number_of_speeches": [1, 1],
        }
    )
    words = pd.DataFrame(
        {
            "speech_id": ["s1", "s1", "s1", "s1"],
            "paragraph_id": ["s1_p0"] * 4,
            "sentence_id": ["s1_p0_sent"] * 4,
            "word_id": [f"s1_p0_sent_w{i}" for i in range(4)],
            "word_form": ["Tak", "om", "se", "efter"],
            "word_length_chars": [3, 2, 2, 5],
            "word_index_in_sentence": [0, 1, 2, 3],
            "word_index_in_paragraph": [0, 1, 2, 3],
            "sentence_length_words": [4, 4, 4, 4],
            "log_corpus_frequency": [1.0, 2.0, 3.0, 1.5],
            "long_word_lix_component": [0, 0, 0, 0],
        }
    )
    gaze_rows = []
    for participant_id, dyslexia, group in [
        ("P01", 0, "typical"),
        ("P02", 1, "dyslexia_labeled"),
    ]:
        for row in words.itertuples(index=False):
            gaze_rows.append(
                {
                    "participant_id": participant_id,
                    "speech_id": row.speech_id,
                    "paragraph_id": row.paragraph_id,
                    "sentence_id": row.sentence_id,
                    "word_id": row.word_id,
                    "word_form": row.word_form,
                    "dyslexia_labeled": dyslexia,
                    "group_label": group,
                    "FFD": 100.0,
                    "GD": 120.0,
                    "TRT": 150.0,
                    "fixation_count": 1,
                    "skip": 0,
                    "refixation_count": 0,
                    "go_past_time": 150.0,
                }
            )
    gaze = pd.DataFrame(gaze_rows)
    sentence = pd.DataFrame(
        {
            "speech_id": ["s1"],
            "paragraph_id": ["s1_p0"],
            "sentence_id": ["s1_p0_sent"],
            "sentence_text": ["Tak om se efter"],
            "sentence_length_words": [4],
            "lix_component": [4.0],
        }
    )
    word_full = gaze.merge(words, on=["speech_id", "paragraph_id", "sentence_id", "word_id"], how="left")
    word_full = word_full.rename(columns={"word_form_x": "word_form_x", "word_form_y": "word_form_y"})
    word_full["dfm_lm_word_surprisal"] = 1.0
    word_full["dfm_lm_word_entropy"] = 2.0
    word_full["dfm_lm_alignment_status"] = "warning"
    word_full["dfm_lm_alignment_warning"] = "non_special_token_unassigned"
    word_full["dfm_lm_alignment_error"] = None
    word_full["upos"] = "NOUN"
    word_full["parser_backend"] = "surface_heuristic"
    word_full["paragraph_cohesion"] = 0.8
    word_full["local_semantic_drift"] = 0.2
    aggregates = pd.DataFrame(
        {
            "participant_id": ["P01", "P02"],
            "dyslexia_labeled": [0, 1],
            "group_label": ["typical", "dyslexia_labeled"],
            "mean_ffd": [100.0, 100.0],
        }
    )
    participants.to_parquet(root / "features" / "participant_level.parquet", index=False)
    words.to_parquet(root / "features" / "word_level_classical.parquet", index=False)
    gaze.to_parquet(root / "features" / "word_level_gaze.parquet", index=False)
    sentence.to_parquet(root / "features" / "sentence_level.parquet", index=False)
    word_full.to_parquet(root / "modeling_tables" / "word_level_full_with_dfm_lm.parquet", index=False)
    sentence.to_parquet(root / "modeling_tables" / "sentence_level_full.parquet", index=False)
    aggregates.to_parquet(root / "modeling_tables" / "participant_aggregates.parquet", index=False)
    (root / "linguistic_features" / "parser_diagnostics.json").write_text(
        '{"backend": "surface_heuristic"}\n', encoding="utf-8"
    )


def test_label_release_build_validate_and_prepared_dataset(tmp_path: Path) -> None:
    source = tmp_path / "feature_release"
    _write_minimal_feature_release(source)
    docs = tmp_path / "docs"
    analysis = tmp_path / "analysis" / "label_analysis"
    config = {
        "run": {"name": "label_release_v1_1", "output_root": str(tmp_path / "results")},
        "label_release": {
            "source_feature_release_dir": str(source),
            "corpus_mode": "full",
            "no_llm_generated_labels": True,
            "deterministic_seed": 17,
            "output_layout": {
                "labels": "labels",
                "prepared_dataset": "prepared_dataset",
                "analysis": "analysis/label_analysis",
            },
            "docs": {
                "participant_label_card": str(docs / "participant_label_card_v1.md"),
                "segmentation_label_card": str(docs / "segmentation_label_card_v1.md"),
                "quality_label_card": str(docs / "quality_label_card_v1.md"),
                "split_policy": str(docs / "split_policy_v1.md"),
                "analysis_dir": str(analysis),
            },
            "participant_labels": {
                "expected_total": 2,
                "expected_dyslexia_labeled": 1,
                "expected_typical_control": 1,
            },
            "split_labels": {"participant_grouped_kfolds": 2},
            "manifest": {"write_checksums": True, "checksum_max_bytes": 5_000_000},
        },
    }
    out = tmp_path / "label_release"
    manifest = build_label_release(config, out, repo_root=tmp_path)
    assert manifest["row_counts"]["participant_labels"] == 2
    assert manifest["row_counts"]["segmentation_word_labels"] == 4
    assert manifest["row_counts"]["analysis_ready_word_level_v1_1"] == 8

    validation = validate_label_release(out, config=config, repo_root=tmp_path)
    assert validation["status"] == "passed", validation["errors"]

    participant = pd.read_parquet(out / "labels" / "participant_labels_v1.parquet")
    assert participant["participant_id"].is_unique
    assert set(participant["reader_group"]) == {"typical_control", "dyslexia_labeled"}

    boundaries = pd.read_parquet(out / "labels" / "segmentation_boundary_labels_v1.parquet")
    assert boundaries["boundary_id"].is_unique
    assert "V#V" in set(boundaries["orth_boundary_type"])

    quality = pd.read_parquet(out / "labels" / "quality_labels_v1.parquet")
    assert quality["participant_word_key"].is_unique
    assert set(quality["parser_status"]) == {"surface_heuristic_fallback"}

    splits = pd.read_parquet(out / "labels" / "split_labels_v1.parquet")
    assert not splits["split_name"].str.contains("random", case=False).any()
    lopo = splits[splits["split_name"].eq("leave_one_participant_out")]
    assert (lopo[lopo["split_role"].eq("test")]["participant_id"].value_counts() == 1).all()

    ready = pd.read_parquet(out / "prepared_dataset" / "analysis_ready_word_level_v1_1.parquet")
    assert len(ready) == len(quality)
    assert ready["participant_word_key"].is_unique
    assert ready["stimulus_word_key"].notna().all()
    assert ready["word"].notna().all()
    assert ready["reader_group"].notna().all()
    assert ready["segmentation_label_version"].notna().all()
    assert (out / "prepared_dataset" / "analysis_ready_manifest.json").exists()
