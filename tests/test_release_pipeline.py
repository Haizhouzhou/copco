from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from copco_eye_bench.cli import build_features_main
from copco_eye_bench.features import build_feature_tables
from copco_eye_bench.release import (
    build_modeling_tables,
    run_parser_features,
    validate_feature_release,
    write_feature_dictionary,
    write_release_features,
)


def _write_minimal_source(root: Path) -> None:
    extracted = root / "ExtractedFeatures"
    extracted.mkdir(parents=True)
    rows = [
        ("P01", 45, 1, 1, 1, "Hej", 120, 130, 150, 1),
        ("P01", 45, 1, 1, 2, "verden", 121, 131, 151, 2),
        ("P02", 45, 1, 1, 1, "Hej", 220, 230, 250, 1),
        ("P02", 45, 1, 1, 2, "verden", 221, 231, 251, 2),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "part",
            "speechId",
            "paragraphId",
            "sentenceId",
            "wordId",
            "word",
            "word_first_fix_dur",
            "word_first_pass_dur",
            "word_total_fix_dur",
            "number_of_fixations",
        ],
    )
    frame["trialId"] = 1
    frame["char_IA_ids"] = ""
    frame["landing_position"] = 0
    frame["word_go_past_time"] = frame["word_total_fix_dur"]
    frame["word_mean_fix_dur"] = frame["word_total_fix_dur"]
    frame["word_mean_sacc_dur"] = 0
    frame["word_peak_sacc_velocity"] = 0
    frame[frame["part"] == "P01"].to_csv(extracted / "P01.csv", index=False)
    frame[frame["part"] == "P02"].to_csv(extracted / "P02.csv", index=False)
    pd.DataFrame({"subj": ["P01", "P02"], "dyslexia": [0, 1]}).to_csv(
        root / "participant_stats.csv", index=False
    )


def test_release_exports_parser_join_and_validation(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    _write_minimal_source(legacy_root)
    config = {
        "feature_release": {
            "require_full_corpus": True,
            "feature_dictionary_path": str(tmp_path / "docs" / "feature_dictionary_v1.md"),
        },
        "dataset": {
            "legacy_root": str(legacy_root),
            "extracted_features_glob": "ExtractedFeatures/P*.csv",
            "participant_stats_path": "participant_stats.csv",
        },
        "language_models": {
            "primary_surprisal": {"output_label": "dfm_decoder_7b"},
            "sensitivity_surprisal": {"output_label": "gemma2_9b"},
        },
        "cv": {"participant_grouped_folds": 2, "random_seeds": [17]},
    }
    output_dir = tmp_path / "release"
    build_feature_tables(config, output_dir, repo_root=tmp_path)
    feature_manifest = write_release_features(config, output_dir, repo_root=tmp_path)
    assert feature_manifest["sample_limits_forbidden"] is True
    parser_manifest = run_parser_features(config, output_dir)
    assert parser_manifest["status"] == "complete"

    words = pd.read_parquet(output_dir / "features" / "word_level_classical.parquet")
    lm_dir = output_dir / "lm_features" / "dfm_decoder_7b" / "surprisal"
    lm_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "word_id": row.word_id,
                "speech_id": row.speech_id,
                "paragraph_id": row.paragraph_id,
                "sentence_id": row.sentence_id,
                "word": row.word_form,
                "lm_model_id": "danish-foundation-models/dfm-decoder-open-v0-7b-pt",
                "lm_tokenizer_id": "danish-foundation-models/dfm-decoder-open-v0-7b-pt",
                "lm_context_mode": "paragraph",
                "lm_context_tokens": 8,
                "lm_word_surprisal": 1.0,
                "lm_word_entropy": 2.0,
                "lm_subword_count": 1,
                "lm_alignment_status": "warning",
                "lm_alignment_warning": "non_special_token_unassigned",
                "lm_alignment_error": None,
                "shard_id": 0,
            }
            for row in words.itertuples(index=False)
        ]
    ).to_parquet(lm_dir / "surprisal_shard0000_of_0001.parquet", index=False)
    (lm_dir / "alignment_report_shard0.json").write_text(
        """
{
  "status": "passed",
  "reports": [
    {
      "context_id": "45_p1",
      "status": "warning",
      "warnings": ["non_special_token_unassigned"],
      "errors": [],
      "word_count": 2,
      "token_count": 4
    }
  ]
}
""",
        encoding="utf-8",
    )

    join_manifest = build_modeling_tables(config, output_dir)
    assert join_manifest["join_validation"]["unexpected_row_loss"] == 0
    assert (output_dir / "lm_features" / "dfm_decoder_7b" / "alignment_report.json").exists()

    write_feature_dictionary(output_dir, tmp_path / "docs" / "feature_dictionary_v1.md")
    (output_dir / "analysis").mkdir()
    (output_dir / "analysis" / "research_plan_next_stage.md").write_text("# Plan\n", encoding="utf-8")
    validation = validate_feature_release(config, output_dir)
    assert validation["status"] == "passed", validation["errors"]


def test_feature_release_config_forbids_cli_sample_limits(tmp_path: Path) -> None:
    config_path = tmp_path / "release.yaml"
    config_path.write_text(
        """
feature_release:
  require_full_corpus: true
dataset:
  legacy_root: data
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        build_features_main(
            [
                "--config",
                str(config_path),
                "--repo-root",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--sample-participants",
                "2",
            ]
        )
