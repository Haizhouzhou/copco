from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from copco_eye_bench.d3_model_evidence_v1_1 import (
    ANALYSIS_DIR,
    EXPECTED_FILES,
    METRIC_COLUMNS,
    RESULT_ORIGINS,
    RESULT_SCOPES,
    build_d3_model_evidence_v1_1,
    validate_evidence_vault_v1_1,
)


def _build_vault(tmp_path: Path) -> Path:
    root = Path.cwd()
    build_d3_model_evidence_v1_1(output_dir=tmp_path / "generated", repo_root=root)
    return root / ANALYSIS_DIR


def test_v1_1_folder_structure_required_files_and_validator(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    missing = [rel for rel in EXPECTED_FILES if not (vault / rel).exists()]
    assert missing == []
    report = validate_evidence_vault_v1_1(repo_root=Path.cwd(), output_dir=tmp_path / "validation")
    assert report["status"] == "passed"


def test_v1_1_canonical_metric_schema_and_scope_values(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    metrics = pd.read_csv(vault / "03_canonical_metrics/canonical_metrics_long.csv")
    assert set(METRIC_COLUMNS).issubset(metrics.columns)
    assert not metrics["evidence_id"].duplicated().any()
    assert set(metrics["result_scope"].dropna().astype(str)).issubset(RESULT_SCOPES)
    assert set(metrics["result_origin"].dropna().astype(str)).issubset(RESULT_ORIGINS)
    assert metrics["algorithm_regime"].astype(str).str.contains("offline").any()
    assert metrics["algorithm_regime"].astype(str).str.contains("online").any()


def test_v1_1_claims_no_advisory_language_and_official_sota_status(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    claims = pd.read_csv(vault / "05_claim_status/claim_status_ledger.csv")
    assert {"inherited_allowed", "inherited_prohibited"}.issubset(set(claims["claim_status"]))
    official = claims[claims["claim_text"].eq("Official EyeBench SOTA.")]
    assert not official.empty
    assert not official["claim_status"].eq("inherited_allowed").any()

    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").lower()
        for path in vault.rglob("*")
        if path.is_file() and path.suffix in {".md", ".csv", ".json", ".jsonl"}
    )
    for phrase in ["we recommend", "paper should", "should claim", "best for the paper"]:
        assert phrase not in text


def test_v1_1_reconciliation_number_registry_and_oracle_flags(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    registry = pd.read_csv(vault / "06_number_registry/paper_number_registry.csv")
    assert len(registry) >= 60

    discrepancies = pd.read_csv(vault / "03_canonical_metrics/unresolved_metric_discrepancies.csv")
    assert discrepancies["conflict_group_id"].astype(str).str.contains("unseen_text_specialist").any()
    assert discrepancies["canonical_value_chosen"].fillna("").eq("").all()

    metrics = pd.read_csv(vault / "03_canonical_metrics/canonical_metrics_long.csv")
    oracle = metrics[metrics["clean_or_oracle"].astype(str).str.contains("oracle", case=False, na=False)]
    assert oracle.empty or not oracle["official_claim_allowed"].fillna(False).astype(bool).any()


def test_v1_1_no_figures_no_final_tables_and_json_manifests(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)
    assert not [
        path
        for path in vault.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    ]
    assert not [
        path
        for path in vault.rglob("*")
        if path.is_file()
        and "final_table" in path.name.lower()
        and path.name != "no_figure_no_final_table_validation_report.md"
    ]

    for name in [
        "evidence_manifest.json",
        "source_manifest.json",
        "metric_manifest.json",
        "claim_manifest.json",
        "validation_manifest.json",
    ]:
        payload = json.loads((vault / "10_machine_readable" / name).read_text(encoding="utf-8"))
        assert payload["vault_version"] == "d3_model_evidence_v1_1"
        assert payload["no_new_experiments"] is True
        assert payload["no_figures_generated"] is True
        assert payload["no_final_tables_generated"] is True
        assert payload["recommendations_generated"] is False
        assert payload["judgements_generated"] is False
