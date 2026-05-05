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
    for candidate in (
        output_dir / "modeling_tables" / "word_level_full_with_all_lm.parquet",
        output_dir / "modeling_tables" / "word_level_full_with_dfm_lm.parquet",
        output_dir / "modeling_tables" / "word_level_full.parquet",
    ):
        if candidate.exists():
            return pd.read_parquet(candidate)

    path = output_dir / "tables" / "word_observations.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing feature table: {path}")
    frame = pd.read_parquet(path)
    for path in sorted((output_dir / "lm_features").glob("**/*.parquet")):
        extra = pd.read_parquet(path)
        if "word_id" in extra.columns:
            frame = frame.merge(extra, on="word_id", how="left", suffixes=("", "_lm"))
    return frame


def _fit_one(frame: Any, name: str, spec: dict[str, Any], default_outcome: str) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    outcome = str(spec.get("outcome", default_outcome))
    fixed = [str(term) for term in spec.get("fixed_effects", [])]
    fixed_base_terms = sorted({piece for term in fixed for piece in term.split(":") if piece})
    required_columns = {outcome, "participant_id", *fixed_base_terms}
    missing = sorted(column for column in required_columns if column not in frame.columns)
    if missing:
        return {"hypothesis": name, "status": "skipped", "reason": f"missing_columns:{missing}"}

    data = frame[[outcome, "participant_id", "word_id", *fixed_base_terms]].copy()
    for column in [outcome, *fixed_base_terms]:
        data[column] = _require_pandas().to_numeric(data[column], errors="coerce")
    data = data.dropna()
    if data.empty or data["participant_id"].nunique() < 2:
        return {"hypothesis": name, "status": "skipped", "reason": "insufficient_complete_cases"}
    max_fit_rows = int(spec.get("max_fit_rows", 50_000))
    sampled = False
    if max_fit_rows > 0 and len(data) > max_fit_rows:
        data = data.sample(n=max_fit_rows, random_state=17).copy()
        sampled = True

    formula = f"{outcome} ~ " + " + ".join(fixed)
    word_levels = data["word_id"].nunique() if "word_id" in data.columns else 0
    word_random_limit = int(spec.get("word_random_effects_max_levels", 5_000))
    vc_formula = {"word_id": "0 + C(word_id)"} if 0 < word_levels <= word_random_limit else None
    try:
        model = smf.mixedlm(
            formula,
            data,
            groups=data["participant_id"],
            vc_formula=vc_formula,
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    except Exception as exc:
        try:
            import statsmodels.api as sm

            y = data[outcome]
            x = sm.add_constant(data[fixed_base_terms], has_constant="add")
            fallback = sm.OLS(y, x).fit(cov_type="HC3")
        except Exception as fallback_exc:
            return {
                "hypothesis": name,
                "status": "failed",
                "reason": str(exc),
                "fallback_reason": str(fallback_exc),
                "formula": formula,
                "sampled": sampled,
            }
        return {
            "hypothesis": name,
            "status": "fallback_complete",
            "reason": str(exc),
            "formula": formula,
            "n_obs": int(fallback.nobs),
            "model_type": "robust_ols_hc3",
            "sampled": sampled,
            "params": {key: float(value) for key, value in fallback.params.items()},
            "pvalues": {key: float(value) for key, value in fallback.pvalues.items()},
        }

    return {
        "hypothesis": name,
        "status": "complete",
        "formula": formula,
        "model_type": "mixedlm_participant_random_intercept"
        if vc_formula is None
        else "mixedlm_participant_and_word_random_intercepts",
        "sampled": sampled,
        "word_random_effect_policy": "included" if vc_formula else "skipped_many_word_levels",
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
    coefficients = []
    for report in reports:
        for term, estimate in report.get("params", {}).items():
            coefficients.append(
                {
                    "hypothesis": report.get("hypothesis"),
                    "status": report.get("status"),
                    "model_type": report.get("model_type"),
                    "term": term,
                    "estimate": estimate,
                    "p_value": report.get("pvalues", {}).get(term),
                    "n_obs": report.get("n_obs"),
                    "sampled": report.get("sampled"),
                }
            )
    if coefficients:
        _require_pandas().DataFrame(coefficients).to_csv(report_dir / "coefficient_table.csv", index=False)
    report_lines = [
        "# Mixed-Effects Analysis Report",
        "",
        "Models use participant grouping and operational dyslexia labels. Results are exploratory and not diagnostic.",
        "",
        *[
            (
                f"- {report.get('hypothesis')}: {report.get('status')} "
                f"({report.get('model_type', report.get('reason', 'not_fit'))})"
            )
            for report in reports
        ],
    ]
    (report_dir / "mixed_effects_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "run_type": "mixed_effects",
        "status": "complete",
        "hypotheses_total": len(reports),
        "hypotheses_complete": sum(
            1 for report in reports if report.get("status") in {"complete", "fallback_complete"}
        ),
        "claim_scope": "scientific primary evidence; operational labels only",
    }
    _write_json(report_dir / "manifest.json", manifest)
    return manifest
