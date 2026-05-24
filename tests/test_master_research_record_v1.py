from __future__ import annotations

import json
from pathlib import Path

from copco_eye_bench.master_research_record_v1 import (
    ANALYSIS_DIR,
    MANIFEST_FILENAME,
    MASTER_FILENAME,
    REQUIRED_SECTIONS,
    VALIDATION_FILENAME,
    build_master_research_record_v1,
    validate_master_research_record_v1,
)


def _build_record(tmp_path: Path) -> Path:
    root = Path.cwd()
    report = build_master_research_record_v1(
        output_dir=tmp_path / "generated",
        repo_root=root,
    )
    assert report["status"] == "passed"
    return root / ANALYSIS_DIR


def test_master_record_required_files_sections_and_validator(tmp_path: Path) -> None:
    record_dir = _build_record(tmp_path)
    master = record_dir / MASTER_FILENAME
    manifest = record_dir / MANIFEST_FILENAME
    validation = record_dir / VALIDATION_FILENAME
    assert master.exists()
    assert manifest.exists()
    assert validation.exists()

    text = master.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text

    report = validate_master_research_record_v1(
        repo_root=Path.cwd(),
        output_dir=tmp_path / "validation",
    )
    assert report["status"] == "passed"


def test_master_manifest_source_trace_and_missing_sources(tmp_path: Path) -> None:
    record_dir = _build_record(tmp_path)
    manifest = json.loads((record_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["no_new_experiments"] is True
    assert manifest["no_figures_generated"] is True
    assert manifest["no_final_paper_tables_generated"] is True
    assert manifest["source_directories_inspected"] >= 20
    assert manifest["source_files_indexed_count"] >= 800
    assert manifest["files_used"]
    assert manifest["key_metrics_extracted"]
    missing_paths = {row["path"] for row in manifest["missing_sources"]}
    assert "analysis/deep_literature_review" in missing_paths


def test_master_content_separates_eyebench_offline_online_and_conflicts(
    tmp_path: Path,
) -> None:
    record_dir = _build_record(tmp_path)
    text = (record_dir / MASTER_FILENAME).read_text(encoding="utf-8")
    assert "Subsection A — Published / provided CopCo TYP baselines" in text
    assert "Subsection B — Internal EyeBench-style full-data reader-aggregated comparison" in text
    assert "Subsection C — EyeBench-fold full-feature intersection" in text
    assert "Subsection D — Official EyeBench subset/evaluator" in text
    assert "Subsection E — Reduced official-protocol-compatible trial-level model" in text
    assert "D3_dfm_residual_gaze_only" in text
    assert "Fixed-budget online" in text or "fixed-budget sequential reader-evidence" in text
    assert "unseen_text_rescue_04" in text
    assert "unseen_text_rescue_05" in text
    assert "Official EyeBench SOTA." not in text


def test_master_no_figures_no_final_tables_and_public_term_mapping(tmp_path: Path) -> None:
    record_dir = _build_record(tmp_path)
    assert not [
        path
        for path in record_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    ]
    assert not [
        path
        for path in record_dir.rglob("*")
        if path.is_file() and "final_table" in path.name.lower()
    ]

    text = (record_dir / MASTER_FILENAME).read_text(encoding="utf-8")
    required_pairs = {
        "D3": "residualized predictability-sensitive gaze-profile method",
        "D3 offline": "full-record reader-profile model",
        "D3 online": "fixed-budget sequential reader-evidence model",
        "D3_Lite": "reduced official-protocol-compatible trial-level variant",
        "BenchmarkBridge": "internal EyeBench-style benchmark comparison",
        "OfficialEyeBenchAlignment": "official protocol and data-alignment audit",
        "OperatingPointAdaptation": "probability-first operating-point diagnostic",
        "OnlineTargetedOptimization": "fixed-budget online and stopping-policy evaluation",
    }
    for internal, public in required_pairs.items():
        assert internal in text
        assert public in text
