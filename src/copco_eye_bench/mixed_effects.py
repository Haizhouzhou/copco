"""Mixed-effects analyses for the CopCo dyslexia-labeled reader program."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import get_nested


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for mixed-effects analysis") from exc
    return pd


def _load_frame(output_dir: Path) -> Any:
    pd = _require_pandas()
    path = output_dir / "tables" / "word_observations.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing feature table: {path}")
    frame = pd.read_parquet(path)
    for path in sorted((output_dir / "lm_features").glob("*/*.parquet")):
        extra = pd.read_parquet(path)
        if "word_id" in extra.columns:
            frame = frame.merge(extra, on="word_id", how="left", suffixes=("", "_lm"))
    return frame


def _fit_one(frame: Any, name: str, spec: dict[str, Any], default_outcome: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    outcome = str(spec.get("outcome", default_outcome))
    fixed = [str(term) for term in spec.get("fixed_effects", [])]
    required_columns = {outcome, "participant_id", *[term for term in fixed if ":" not in term]}
    missing = sorted(column for column in required_columns if column not in frame.columns)
    if missing:
        return {"hypothesis": name, "status": "skipped", "reason": f"missing_columns:{missing}"}

    data = frame[[outcome, "participant_id", "word_id", *[term for term in fixed if ":" not in term]]].copy()
    for column in [outcome, *[term for term in fixed if ":" not in term]]:
        data[column] = _require_pandas().to_numeric(data[column], errors="coerce")
    data = data.dropna()
    if data.empty or data["participant_id"].nunique() < 2:
        return {"hypothesis": name, "status": "skipped", "reason": "insufficient_complete_cases"}

    formula = f"{outcome} ~ " + " + ".join(fixed)
    try:
        model = smf.mixedlm(
            formula,
            data,
            groups=data["participant_id"],
            vc_formula={"word_id": "0 + C(word_id)"} if "word_id" in data.columns else None,
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    except Exception as exc:
        return {"hypothesis": name, "status": "failed", "reason": str(exc), "formula": formula}

    return {
        "hypothesis": name,
        "status": "complete",
        "formula": formula,
        "n_obs": int(result.nobs),
        "aic": float(result.aic) if result.aic == result.aic else None,
        "bic": float(result.bic) if result.bic == result.bic else None,
        "converged": bool(getattr(result, "converged", False)),
        "params": {key: float(value) for key, value in result.params.items()},
        "pvalues": {key: float(value) for key, value in result.pvalues.items()},
    }


def fit_mixed_effects(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Fit configured mixed-effects models when statsmodels and columns are available."""

    out = Path(output_dir).resolve()
    report_dir = out / "mixed_effects"
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        import statsmodels  # noqa: F401
    except Exception as exc:
        manifest = {"run_type": "mixed_effects", "status": "skipped", "reason": str(exc)}
        _write_json(report_dir / "manifest.json", manifest)
        return manifest

    frame = _load_frame(out)
    if "dyslexia_labeled" not in frame.columns:
        manifest = {"run_type": "mixed_effects", "status": "skipped", "reason": "missing_labels"}
        _write_json(report_dir / "manifest.json", manifest)
        return manifest
    default_outcome = str(get_nested(config, "mixed_effects.primary_outcome", "TRT"))
    hypotheses = get_nested(config, "mixed_effects.hypotheses", {})
    reports = [_fit_one(frame, name, spec, default_outcome) for name, spec in hypotheses.items()]
    _write_json(report_dir / "hypothesis_reports.json", {"hypotheses": reports})
    manifest = {
        "run_type": "mixed_effects",
        "status": "complete",
        "hypotheses_total": len(reports),
        "hypotheses_complete": sum(1 for report in reports if report.get("status") == "complete"),
        "claim_scope": "scientific primary evidence; operational labels only",
    }
    _write_json(report_dir / "manifest.json", manifest)
    return manifest
