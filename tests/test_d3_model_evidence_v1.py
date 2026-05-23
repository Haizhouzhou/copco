from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.d3_model_evidence_v1 import (
    ANALYSIS_DIR,
    EXPECTED_FILES,
    METRIC_COLUMNS,
    build_d3_model_evidence_v1,
    build_source_inventory,
    validate_evidence_vault,
)


def _ensure_vault(tmp_path: Path) -> Path:
    root = Path.cwd()
    build_d3_model_evidence_v1(output_dir=tmp_path / "d3_model_evidence", repo_root=root)
    return root / ANALYSIS_DIR


def test_d3_evidence_folder_structure_and_validator(tmp_path: Path) -> None:
    vault = _ensure_vault(tmp_path)
    missing = [rel for rel in EXPECTED_FILES if not (vault / rel).exists()]
    assert missing == []
    report = validate_evidence_vault(repo_root=Path.cwd(), output_dir=tmp_path / "validation")
    assert report["status"] == "passed"


def test_d3_evidence_metric_schema_and_regime_separation(tmp_path: Path) -> None:
    vault = _ensure_vault(tmp_path)
    metrics = pd.read_csv(vault / "03_results_canonical/canonical_metrics_long.csv")
    assert set(METRIC_COLUMNS).issubset(metrics.columns)
    assert not metrics["evidence_id"].duplicated().any()
    assert metrics["algorithm_regime"].astype(str).str.contains("offline").any()
    assert metrics["algorithm_regime"].astype(str).str.contains("online").any()
    oracle = metrics[metrics["clean_or_oracle"].eq("oracle")]
    assert oracle.empty or not oracle["official_claim_allowed"].fillna(False).astype(bool).any()


def test_d3_evidence_claims_and_number_registry(tmp_path: Path) -> None:
    vault = _ensure_vault(tmp_path)
    claims = pd.read_csv(vault / "05_claims/claim_evidence_ledger_v1.csv")
    assert claims["claim_status"].eq("allowed").any()
    assert claims["claim_status"].eq("prohibited").any()
    assert claims["claim_text"].str.contains("Official EyeBench SOTA").any()
    official_allowed = claims[
        claims["claim_text"].str.contains("Official EyeBench SOTA")
        & ~claims["claim_text"].str.contains("not claimed")
        & claims["claim_status"].eq("allowed")
    ]
    assert official_allowed.empty

    registry = pd.read_csv(vault / "06_paper_sources/paper_ready_number_registry.csv")
    assert {"offline_auc", "online_v2_early_both_ba", "unseen_text_specialist_ba"}.issubset(
        set(registry["number_id"])
    )


def test_d3_evidence_no_figures_or_final_tables(tmp_path: Path) -> None:
    vault = _ensure_vault(tmp_path)
    forbidden_suffixes = {".png", ".pdf", ".svg"}
    assert not [path for path in vault.rglob("*") if path.suffix.lower() in forbidden_suffixes]
    assert not [path for path in vault.rglob("*") if "final_table" in path.name.lower()]


def test_d3_source_inventory_handles_missing_paths(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    frame, missing = build_source_inventory(tmp_path, vault)
    assert not frame.empty
    assert missing
    assert (vault / "00_inventory/missing_source_report.md").exists()
