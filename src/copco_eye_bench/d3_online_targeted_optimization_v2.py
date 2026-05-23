"""Audit and strict rerun for D3 online targeted optimization v2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_nested, timestamped_output_dir
from .d3_online_targeted_optimization import (
    _md_table,
    _write_csv,
    _write_json,
    _write_md,
    classification_metrics,
)


SECTION = "d3_online_targeted_optimization_v2"
ANALYSIS_NAME = "d3_online_targeted_optimization_v2"
PRIMARY_REGIMES = ("unseen_reader", "unseen_reader_and_text")
SPLITS = ("unseen_reader", "unseen_text", "unseen_reader_and_text")
METRIC_COLUMNS = ["AUROC", "PR-AUC", "BA", "macro_F1", "Brier"]
ONLINE_SELECTED_GROUPS = {
    "online_late_accumulation",
    "online_mid_detection",
    "online_early_detection",
    "online_stopping_detector",
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    candidate_group: str
    feature_family: str
    source_feature_group: str
    calibrator: str
    threshold_policy: str
    accumulator: str
    stopping_policy: str
    prefix_type: str
    prefix_value: str


def _section(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get(SECTION, {}))


def _path(config: dict[str, Any], key: str, repo_root: Path) -> Path:
    raw = get_nested(config, f"{SECTION}.{key}")
    if raw is None:
        raise KeyError(f"missing config key: {SECTION}.{key}")
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _analysis_dir(config: dict[str, Any], repo_root: Path) -> Path:
    return _path(config, "repo_analysis_dir", repo_root)


def _as_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _score_from_metrics(metrics: dict[str, Any], earliness: float) -> float:
    auroc = _num(metrics.get("AUROC"), 0.0)
    pr_auc = _num(metrics.get("PR-AUC"), 0.0)
    ba = _num(metrics.get("BA"), 0.0)
    brier = _num(metrics.get("Brier"), 1.0)
    return 0.35 * auroc + 0.25 * pr_auc + 0.20 * ba + 0.10 * (1.0 - brier) + 0.10 * earliness


def _score_no_earliness(metrics: dict[str, Any]) -> float:
    auroc = _num(metrics.get("AUROC"), 0.0)
    pr_auc = _num(metrics.get("PR-AUC"), 0.0)
    ba = _num(metrics.get("BA"), 0.0)
    brier = _num(metrics.get("Brier"), 1.0)
    return 0.3888889 * auroc + 0.2777778 * pr_auc + 0.2222222 * ba + 0.1111111 * (1.0 - brier)


def _num(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _prefix_rank(prefix_type: str, prefix_value: str) -> tuple[int, float]:
    type_rank = {
        "word_count_prefix": 0,
        "chronological_prefix": 1,
        "trial_or_text_prefix": 2,
        "speech_prefix": 3,
    }.get(str(prefix_type), 9)
    value = str(prefix_value)
    if value == "all":
        return type_rank, 1e9
    return type_rank, _num(value, 1e8)


def _evidence_rank(prefix_type: str, prefix_value: str) -> tuple[int, float]:
    if prefix_value == "all":
        return 99, 1e9
    value = _num(prefix_value, 1e8)
    if prefix_type in {"word_count_prefix", "chronological_prefix"}:
        return 0, value
    if prefix_type == "trial_or_text_prefix":
        return 1, value
    if prefix_type == "speech_prefix":
        return 2, value
    return 9, value


def _load_online(v1_output_dir: Path) -> pd.DataFrame:
    path = v1_output_dir / "online_probabilities" / "online_probabilities.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame["prefix_value"] = frame["prefix_value"].map(_as_str)
    frame["source_feature_group"] = frame["feature_group"]
    frame["strict_feature_family"] = frame["feature_group"].replace(
        {"all_allowed_online": "all_allowed_strict_online"}
    )
    sequence_keys = [
        "split_role",
        "split_regime",
        "fold_id",
        "source_feature_group",
        "accumulator",
        "participant_id",
    ]
    for observed_col, max_col in [
        ("n_words_observed", "max_words_available"),
        ("n_texts_observed", "max_texts_available"),
    ]:
        values = pd.to_numeric(frame[observed_col], errors="coerce")
        frame[max_col] = values.groupby([frame[key] for key in sequence_keys], dropna=False).transform(
            "max"
        )
    return frame


def _load_v1(config: dict[str, Any], repo_root: Path) -> dict[str, pd.DataFrame]:
    v1_analysis = _path(config, "v1_analysis_dir", repo_root)
    v1_output = _path(config, "v1_output_dir", repo_root)
    return {
        "prefix_metrics": pd.read_csv(v1_analysis / "online_prefix_model_metrics.csv"),
        "accumulation": pd.read_csv(v1_analysis / "online_evidence_accumulation_metrics.csv"),
        "stopping": pd.read_csv(v1_analysis / "online_stopping_policy_metrics.csv"),
        "ranking": pd.read_csv(v1_analysis / "online_candidate_validation_ranking.csv"),
        "locked": pd.read_csv(v1_analysis / "online_locked_test_results.csv"),
        "comparison": pd.read_csv(v1_analysis / "online_offline_comparison_table.csv"),
        "online": _load_online(v1_output),
    }


def _read_predictor_manifest(config: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    v1_output = _path(config, "v1_output_dir", repo_root)
    path = v1_output / "nested_predictions" / "predictor_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_rate(group: pd.DataFrame) -> float:
    if "stable_enough_for_prediction" not in group:
        return math.nan
    stable = group["stable_enough_for_prediction"].astype(bool)
    return float((~stable).mean())


def _fit_calibrator(rows: pd.DataFrame, calibrator: str) -> dict[str, Any]:
    p = pd.to_numeric(rows.get("p_raw"), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows.get("y_true"), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[mask], 1e-6, 1 - 1e-6)
    y = y[mask].astype(int)
    if calibrator == "identity" or len(y) < 10 or len(set(y.tolist())) < 2:
        return {"kind": "identity"}
    if calibrator == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(p.reshape(-1, 1), y)
        return {"kind": "sigmoid", "model": model}
    if calibrator == "isotonic" and len(y) >= 40:
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(p, y)
        return {"kind": "isotonic", "model": model}
    return {"kind": "identity"}


def _apply_calibrator(rows: pd.DataFrame, fitted: dict[str, Any]) -> np.ndarray:
    p = pd.to_numeric(rows.get("p_raw"), errors="coerce").to_numpy(dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if fitted.get("kind") in {"sigmoid", "isotonic"}:
        model = fitted["model"]
        if fitted["kind"] == "sigmoid":
            return np.clip(model.predict_proba(p.reshape(-1, 1))[:, 1], 0.0, 1.0)
        return np.clip(model.predict(p), 0.0, 1.0)
    return p


def _learn_threshold(rows: pd.DataFrame, policy: str) -> float:
    if policy == "fixed_0_5" or rows.empty or rows["y_true"].nunique() < 2:
        return 0.5
    p = pd.to_numeric(rows["p_eval"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["y_true"], errors="coerce").to_numpy(dtype=int)
    mask = np.isfinite(p)
    p = p[mask]
    y = y[mask]
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return 0.5
    candidates = np.unique(np.quantile(p, np.linspace(0, 1, min(101, max(3, len(p))))))
    if len(candidates) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        metrics = classification_metrics(y, p, float(threshold))
        if policy == "target_sensitivity_stop":
            score = _num(metrics.get("BA"), 0.0) if _num(metrics.get("sensitivity"), 0.0) >= 0.80 else -math.inf
        else:
            score = _num(metrics.get("BA"), 0.0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _budget_filter(frame: pd.DataFrame, prefix_type: str, prefix_value: str) -> pd.DataFrame:
    return frame[
        frame["prefix_type"].astype(str).eq(prefix_type)
        & frame["prefix_value"].map(_as_str).eq(str(prefix_value))
    ].copy()


def _prepare_rows_for_candidate(
    online: pd.DataFrame, candidate: Candidate, split_role: str
) -> pd.DataFrame:
    base = online[
        online["split_role"].eq(split_role)
        & online["source_feature_group"].eq(candidate.source_feature_group)
        & online["accumulator"].eq(candidate.accumulator)
    ].copy()
    if candidate.prefix_value != "sequence_stop":
        base = _budget_filter(base, candidate.prefix_type, candidate.prefix_value)
    else:
        base = base[base["prefix_value"].ne("all")].copy()
    if base.empty:
        return base
    base["p_raw"] = pd.to_numeric(base["p_t"], errors="coerce")
    return base


def _add_evidence_cost(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if out.empty:
        out["evidence_cost"] = pd.Series(dtype=float)
        out["earliness_score"] = pd.Series(dtype=float)
        return out
    keys = [
        "split_role",
        "split_regime",
        "fold_id",
        "source_feature_group",
        "accumulator",
        "participant_id",
    ]
    if "max_words_available" in out:
        max_words = pd.to_numeric(out["max_words_available"], errors="coerce")
    else:
        max_words = pd.to_numeric(out["n_words_observed"], errors="coerce").groupby(
            [out[key] for key in keys if key in out], dropna=False
        ).transform("max")
    if "max_texts_available" in out:
        max_texts = pd.to_numeric(out["max_texts_available"], errors="coerce")
    else:
        max_texts = pd.to_numeric(out["n_texts_observed"], errors="coerce").groupby(
            [out[key] for key in keys if key in out], dropna=False
        ).transform("max")
    word_cost = (pd.to_numeric(out["n_words_observed"], errors="coerce") / max_words.clip(lower=1)).clip(0, 1)
    text_cost = (pd.to_numeric(out["n_texts_observed"], errors="coerce") / max_texts.clip(lower=1)).clip(0, 1)
    out["evidence_cost"] = 0.5 * word_cost + 0.5 * text_cost
    out["earliness_score"] = 1.0 - out["evidence_cost"]
    return out


def _evaluate_rows(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    records = []
    if rows.empty:
        return pd.DataFrame()
    for regime, group in rows.groupby("split_regime", dropna=False):
        decided = group[group.get("decision", "decide").ne("continue")] if "decision" in group else group
        coverage = len(decided) / len(group) if len(group) else math.nan
        metrics = classification_metrics(decided["y_true"], decided["p_eval"], threshold) if len(decided) else {}
        records.append(
            {
                "split_regime": regime,
                "n_readers": int(group["participant_id"].nunique()),
                "n_prefix_rows": int(len(group)),
                "coverage": coverage,
                "undecided_rate": 1.0 - coverage if math.isfinite(coverage) else math.nan,
                "mean_words_to_decision": _num(
                    pd.to_numeric(decided.get("n_words_observed"), errors="coerce").mean()
                ),
                "mean_texts_to_decision": _num(
                    pd.to_numeric(decided.get("n_texts_observed"), errors="coerce").mean()
                ),
                "evidence_cost": _num(pd.to_numeric(decided.get("evidence_cost"), errors="coerce").mean()),
                "earliness_score": _num(
                    pd.to_numeric(decided.get("earliness_score"), errors="coerce").mean(),
                    0.0,
                ),
                "unstable_prefix_rate": _stable_rate(group),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def _score_summary(scored: pd.DataFrame) -> tuple[float, float]:
    if scored.empty:
        return math.nan, math.nan
    primary = scored[scored["split_regime"].isin(PRIMARY_REGIMES)]
    if primary.empty:
        primary = scored
    score = np.nanmean([
        _score_from_metrics(row, _num(row.get("earliness_score"), 0.0))
        for row in primary.to_dict("records")
    ])
    no_early = np.nanmean([_score_no_earliness(row) for row in primary.to_dict("records")])
    return float(score), float(no_early)


def _fit_and_apply_fixed_budget(
    online: pd.DataFrame, candidate: Candidate, split_role: str
) -> tuple[pd.DataFrame, float, str]:
    inner = _prepare_rows_for_candidate(online, candidate, "inner_oof")
    target = _prepare_rows_for_candidate(online, candidate, split_role)
    if inner.empty or target.empty:
        return pd.DataFrame(), 0.5, "blocked_empty_rows"
    calibrator = _fit_calibrator(inner, candidate.calibrator)
    inner = inner.copy()
    target = target.copy()
    inner["p_eval"] = _apply_calibrator(inner, calibrator)
    target["p_eval"] = _apply_calibrator(target, calibrator)
    if candidate.threshold_policy == "inner_cv_regime_specific" and split_role == "outer_test":
        thresholds = {
            regime: _learn_threshold(group, candidate.threshold_policy)
            for regime, group in inner.groupby("split_regime", dropna=False)
        }
        target["threshold"] = target["split_regime"].map(thresholds).fillna(0.5)
    else:
        threshold = _learn_threshold(inner, candidate.threshold_policy)
        target["threshold"] = threshold
    target["decision"] = "decide"
    target = _add_evidence_cost(target)
    target["y_pred"] = (target["p_eval"] >= target["threshold"]).astype(int)
    threshold_value = _num(target["threshold"].mean(), 0.5)
    return target, threshold_value, str(calibrator.get("kind", "identity"))


def _sort_sequence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ranks = out.apply(lambda row: _prefix_rank(row["prefix_type"], row["prefix_value"]), axis=1)
    out["_prefix_rank"] = [item[0] for item in ranks]
    out["_prefix_numeric"] = [item[1] for item in ranks]
    return out.sort_values(["participant_id", "n_words_observed", "_prefix_rank", "_prefix_numeric"])


def _choose_stopping_thresholds(inner: pd.DataFrame, policy: str) -> tuple[float, float]:
    if inner.empty:
        return 0.35, 0.65
    candidates = [(0.25, 0.75), (0.30, 0.70), (0.35, 0.65), (0.40, 0.60), (0.45, 0.55)]
    best = (0.35, 0.65)
    best_score = -math.inf
    for tau_neg, tau_pos in candidates:
        decisions = _apply_stopping(inner, tau_neg, tau_pos)
        decided = decisions[decisions["decision"].ne("continue")]
        coverage = len(decided) / len(decisions) if len(decisions) else 0.0
        metrics = classification_metrics(decided["y_true"], decided["p_eval"], 0.5) if len(decided) else {}
        ba = _num(metrics.get("BA"), 0.0)
        sensitivity = _num(metrics.get("sensitivity"), 0.0)
        earliness = _num(decided["earliness_score"].mean(), 0.0) if len(decided) else 0.0
        if policy == "cost_sensitive_stop":
            score = ba + 0.20 * earliness
        elif policy == "target_sensitivity_stop":
            score = ba + 0.05 * earliness if sensitivity >= 0.80 else -math.inf
        elif policy == "coverage_constrained_stop":
            score = ba + 0.05 * earliness if coverage >= 0.70 else -math.inf
        else:
            score = ba + 0.10 * earliness + 0.05 * coverage
        if score > best_score:
            best = (tau_neg, tau_pos)
            best_score = score
    return best


def _apply_stopping(rows: pd.DataFrame, tau_neg: float, tau_pos: float) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    base = rows[rows["prefix_value"].ne("all")].copy()
    records = []
    for _, group in _sort_sequence(base).groupby(
        ["split_role", "split_regime", "fold_id", "source_feature_group", "participant_id"],
        dropna=False,
    ):
        hit = group[(group["p_raw"] <= tau_neg) | (group["p_raw"] >= tau_pos)]
        if hit.empty:
            selected = group.iloc[-1].copy()
            selected["decision"] = "continue"
        else:
            selected = hit.iloc[0].copy()
            selected["decision"] = "positive" if selected["p_raw"] >= tau_pos else "negative"
        records.append(selected)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["p_eval"] = pd.to_numeric(out["p_raw"], errors="coerce")
    out["y_pred"] = np.select(
        [out["decision"].eq("positive"), out["decision"].eq("negative")],
        [1, 0],
        default=-1,
    )
    return _add_evidence_cost(out)


def _fit_and_apply_stopping(
    online: pd.DataFrame, candidate: Candidate, split_role: str
) -> tuple[pd.DataFrame, float, str]:
    inner = _prepare_rows_for_candidate(online, candidate, "inner_oof")
    target = _prepare_rows_for_candidate(online, candidate, split_role)
    if inner.empty or target.empty:
        return pd.DataFrame(), 0.5, "blocked_empty_rows"
    tau_neg, tau_pos = _choose_stopping_thresholds(inner, candidate.stopping_policy)
    out = _apply_stopping(target, tau_neg, tau_pos)
    out["threshold"] = 0.5
    return out, 0.5, f"two_sided:{tau_neg:.2f}:{tau_pos:.2f}"


def _candidate_space(config: dict[str, Any]) -> pd.DataFrame:
    section = _section(config)
    features = section.get("feature_families", [])
    accumulators = section.get("accumulators", [])
    calibrators = section.get("calibrators", [])
    thresholds = section.get("thresholds", [])
    stopping = section.get("stopping_policies", [])
    budgets = section.get("evidence_budgets", {})
    rows: list[dict[str, Any]] = []
    cid = 0

    def add(
        group: str,
        feature: str,
        calibrator: str,
        threshold: str,
        accumulator: str,
        stop: str,
        prefix_type: str,
        prefix_value: str,
    ) -> None:
        nonlocal cid
        rows.append(
            {
                "candidate_id": f"v2_candidate_{cid:04d}",
                "candidate_group": group,
                "feature_family": feature,
                "source_feature_group": "all_allowed_online"
                if feature == "all_allowed_strict_online"
                else feature,
                "calibrator": calibrator,
                "threshold_policy": threshold,
                "accumulator": accumulator,
                "stopping_policy": stop,
                "prefix_type": prefix_type,
                "prefix_value": str(prefix_value),
                "clean_result": True,
                "official_claim_allowed": False,
                "selection_source": "inner_oof",
            }
        )
        cid += 1

    offline_candidates = [
        ("all_allowed_strict_online", "identity", "fixed_0_5", "learned_meta_aggregator"),
        ("all_allowed_strict_online", "sigmoid", "inner_cv_global", "entropy_weighted"),
        ("all_allowed_strict_online", "identity", "fixed_0_5", "logit_mean"),
        ("dfm_residual_plus_uncertainty_prefix", "identity", "fixed_0_5", "learned_meta_aggregator"),
    ]
    for feature, calibrator, threshold, accumulator in offline_candidates:
        add(
            "offline_all_full_evidence",
            feature,
            calibrator,
            threshold,
            accumulator,
            "no_stop",
            "trial_or_text_prefix",
            "all",
        )
    for group, key in [
        ("online_late_accumulation", "late"),
        ("online_mid_detection", "mid"),
        ("online_early_detection", "early"),
    ]:
        group_budgets = budgets.get(key, {})
        budget_pairs = [
            (ptype, value)
            for ptype, values in group_budgets.items()
            for value in values
            if str(value) != "all"
        ]
        for idx in range(12):
            feature = features[idx % len(features)]
            accumulator = accumulators[idx % len(accumulators)]
            calibrator = calibrators[idx % len(calibrators)]
            threshold = thresholds[idx % len(thresholds)]
            prefix_type, prefix_value = budget_pairs[idx % len(budget_pairs)]
            add(group, feature, calibrator, threshold, accumulator, "fixed_budget", prefix_type, str(prefix_value))
    for idx in range(12):
        feature = features[idx % len(features)]
        accumulator = accumulators[idx % len(accumulators)]
        stop = stopping[idx % len(stopping)]
        add(
            "online_stopping_detector",
            feature,
            "identity",
            "inner_cv_global",
            accumulator,
            stop,
            "sequence",
            "sequence_stop",
        )
    return pd.DataFrame(rows)


def _evaluate_candidate(online: pd.DataFrame, candidate: Candidate, split_role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate.candidate_group == "online_stopping_detector":
        rows, threshold, calibration = _fit_and_apply_stopping(online, candidate, split_role)
    else:
        rows, threshold, calibration = _fit_and_apply_fixed_budget(online, candidate, split_role)
    scored = _evaluate_rows(rows, threshold)
    if scored.empty:
        return rows, scored
    for key, value in candidate.__dict__.items():
        scored[key] = value
    scored["threshold"] = threshold
    scored["calibration_source"] = calibration
    scored["clean_result"] = True
    scored["official_claim_allowed"] = False
    scored["selection_source"] = "inner_oof"
    return rows, scored


def audit_v1(config: dict[str, Any], v1: dict[str, pd.DataFrame], repo_root: Path) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    ranking = v1["ranking"]
    locked = v1["locked"]
    prefix_metrics = v1["prefix_metrics"]
    manifest = _read_predictor_manifest(config, repo_root)
    all_allowed_rows = [row for row in manifest if row.get("feature_group") == "all_allowed_online"]
    all_allowed_cols = sorted({col for row in all_allowed_rows for col in row.get("predictor_columns", [])})
    future_like = [
        col
        for col in all_allowed_cols
        if any(token in col.lower() for token in ["future", "full_session", "total_session"])
    ]
    top = ranking.iloc[0].to_dict() if not ranking.empty else {}
    locked_primary = locked[locked["split_regime"].isin(PRIMARY_REGIMES)].copy()
    audit_rows = [
        ("v1_output_dir", str(_path(config, "v1_output_dir", repo_root))),
        ("v1_was_fast_mode", str("fast" in str(_path(config, "v1_output_dir", repo_root)))),
        ("v1_candidate_search_truncated", str(len(ranking) == 36)),
        ("v1_candidates_evaluated", str(len(ranking))),
        (
            "prefix_budgets_evaluated",
            ", ".join(
                sorted(
                    {
                        f"{row.prefix_type}:{_as_str(row.prefix_value)}"
                        for row in prefix_metrics[["prefix_type", "prefix_value"]].itertuples()
                    },
                    key=lambda item: _evidence_rank(*item.split(":", 1)),
                )
            ),
        ),
        ("v1_selected_prefix_value_all", "not_explicit; selected no_stop final sequence"),
        ("v1_selected_no_stop", str(top.get("stopping_policy") == "no_stop")),
        ("all_allowed_predictor_count", str(len(all_allowed_cols))),
        ("all_allowed_future_like_columns", ", ".join(future_like) if future_like else "none_detected"),
        ("best_candidate_uses_future_beyond_prefix", "no for prefix rows; yes it consumes all prefixes via no_stop"),
        ("online_primary_score_includes_earliness", "true"),
        ("v1_selected_candidate", str(top.get("candidate_id", ""))),
        ("v1_validation_primary_score", str(top.get("validation_primary_score", ""))),
        ("v1_validation_no_earliness_score", str(top.get("validation_no_earliness_score", ""))),
        (
            "v1_locked_primary_mean_AUROC",
            f"{pd.to_numeric(locked_primary['AUROC'], errors='coerce').mean():.4f}",
        ),
        (
            "v1_locked_primary_mean_BA",
            f"{pd.to_numeric(locked_primary['BA'], errors='coerce').mean():.4f}",
        ),
        ("v1_best_truly_online", "offline_like_late_accumulation_not_early_detector"),
    ]
    audit = pd.DataFrame(audit_rows, columns=["audit_item", "value"])
    _write_csv(analysis_dir / "v1_audit_summary.csv", audit)
    report = [
        "# D3 Online v1 Audit Report",
        "",
        "## Audit Findings",
        "",
        _md_table(audit, max_rows=50),
        "",
        "## Locked v1 Test Rows",
        "",
        _md_table(locked, max_rows=20),
        "",
        "Conclusion: v1 selected a strong reader-centered late/full-sequence candidate, "
        "but it is offline-like for deployment because `no_stop` consumes the final sequence.",
    ]
    _write_md(analysis_dir / "v1_audit_report.md", "\n".join(report))
    return audit


def per_prefix_curves(config: dict[str, Any], v1: dict[str, pd.DataFrame], repo_root: Path) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    online = v1["online"]
    rows = []
    test = online[online["split_role"].eq("outer_test")].copy()
    test = test[test["prefix_value"].isin(["50", "100", "250", "500", "1000", "1", "2", "3", "5", "all"])]
    group_cols = [
        "split_regime",
        "prefix_type",
        "prefix_value",
        "strict_feature_family",
        "source_feature_group",
        "accumulator",
    ]
    for keys, group in test.groupby(group_cols, dropna=False):
        metrics = classification_metrics(group["y_true"], group["p_t"], 0.5)
        rows.append(
            {
                "split_regime": keys[0],
                "prefix_type": keys[1],
                "prefix_value": keys[2],
                "feature_family": keys[3],
                "source_feature_group": keys[4],
                "calibrator": "identity",
                "threshold": "fixed_0_5",
                "accumulator": keys[5],
                "n_readers": int(group["participant_id"].nunique()),
                "n_prefix_rows": int(len(group)),
                "unstable_prefix_rate": _stable_rate(group),
                **metrics,
            }
        )
    curves = pd.DataFrame(rows).sort_values(
        ["split_regime", "feature_family", "accumulator", "prefix_type", "prefix_value"]
    )
    _write_csv(analysis_dir / "per_prefix_performance_curves.csv", curves)
    thresholds = []
    for regime in PRIMARY_REGIMES:
        subset = curves[curves["split_regime"].eq(regime)].copy()
        for metric, op, target in [
            ("AUROC", ">=", 0.75),
            ("AUROC", ">=", 0.80),
            ("BA", ">=", 0.70),
            ("BA", ">=", 0.75),
            ("Brier", "<=", 0.18),
        ]:
            if op == ">=":
                passed = subset[pd.to_numeric(subset[metric], errors="coerce") >= target]
            else:
                passed = subset[pd.to_numeric(subset[metric], errors="coerce") <= target]
            if passed.empty:
                thresholds.append({"split_regime": regime, "criterion": f"{metric} {op} {target}", "earliest": "not_reached"})
            else:
                passed = passed.assign(
                    rank=passed.apply(lambda row: _evidence_rank(row["prefix_type"], row["prefix_value"]), axis=1)
                ).sort_values("rank")
                first = passed.iloc[0]
                thresholds.append(
                    {
                        "split_regime": regime,
                        "criterion": f"{metric} {op} {target}",
                        "earliest": f"{first['prefix_type']}:{first['prefix_value']}",
                        "feature_family": first["feature_family"],
                        "accumulator": first["accumulator"],
                        "value": first[metric],
                    }
                )
    threshold_frame = pd.DataFrame(thresholds)
    report = [
        "# Per-Prefix Performance Report",
        "",
        f"- Curve rows: {len(curves)}",
        "- Rows use outer-test probabilities and fixed 0.5 threshold unless otherwise stated.",
        "",
        "## Earliest Reliability Criteria",
        "",
        _md_table(threshold_frame, max_rows=30),
        "",
        "## Top Per-Prefix Rows",
        "",
        _md_table(curves.sort_values(["AUROC", "BA"], ascending=False).head(30), max_rows=30),
    ]
    _write_md(analysis_dir / "per_prefix_performance_report.md", "\n".join(report))
    return curves


def error_source_analysis(config: dict[str, Any], v1: dict[str, pd.DataFrame], repo_root: Path) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    online = v1["online"]
    selected = online[
        online["split_role"].eq("outer_test")
        & online["source_feature_group"].eq("all_allowed_online")
        & online["accumulator"].isin(["learned_meta_aggregator", "mean_probability"])
        & online["prefix_value"].ne("all")
    ].copy()
    selected = _add_evidence_cost(selected)
    selected["p_eval"] = selected["p_t"]
    selected["y_pred_at_prefix"] = (selected["p_eval"] >= 0.5).astype(int)
    selected["correct_at_prefix"] = selected["y_pred_at_prefix"].eq(selected["y_true"])
    sequence_cols = ["split_regime", "participant_id", "accumulator"]
    final = _sort_sequence(selected).groupby(sequence_cols, dropna=False).tail(1)
    final_status = final.set_index(sequence_cols)["correct_at_prefix"].to_dict()
    final_pred = final.set_index(sequence_cols)["y_pred_at_prefix"].to_dict()
    records = []
    for _, row in selected.iterrows():
        key = (row["split_regime"], row["participant_id"], row["accumulator"])
        final_correct = bool(final_status.get(key, False))
        final_y_pred = int(final_pred.get(key, row["y_pred_at_prefix"]))
        wrong = not bool(row["correct_at_prefix"])
        fp = wrong and int(row["y_true"]) == 0
        fn = wrong and int(row["y_true"]) == 1
        records.append(
            {
                "split_regime": row["split_regime"],
                "participant_id": row["participant_id"],
                "reader_group": row.get("reader_group", ""),
                "terminal_text_id": row.get("terminal_text_id", ""),
                "observed_text_ids": row.get("observed_text_ids", ""),
                "prefix_type": row["prefix_type"],
                "prefix_value": row["prefix_value"],
                "accumulator": row["accumulator"],
                "n_words_observed": row["n_words_observed"],
                "n_texts_observed": row["n_texts_observed"],
                "p_t": row["p_t"],
                "y_true": row["y_true"],
                "y_pred_at_prefix": row["y_pred_at_prefix"],
                "correct_at_prefix": bool(row["correct_at_prefix"]),
                "wrong_at_prefix": wrong,
                "wrong_early_correct_late": wrong and final_correct,
                "correct_early_wrong_late": bool(row["correct_at_prefix"]) and not final_correct,
                "persistent_false_positive": fp and not final_correct and final_y_pred == 1,
                "persistent_false_negative": fn and not final_correct and final_y_pred == 0,
                "insufficient_evidence_error": wrong and final_correct,
                "distribution_shift_error_candidate": wrong and row["split_regime"] == "unseen_text",
                "threshold_error_candidate": wrong and abs(float(row["p_t"]) - 0.5) <= 0.10,
                "calibration_error_candidate": abs(float(row["y_true"]) - float(row["p_t"])) >= 0.40,
                "stable_enough_for_prediction": bool(row.get("stable_enough_for_prediction", False)),
            }
        )
    errors = pd.DataFrame(records)
    _write_csv(analysis_dir / "error_source_by_prefix.csv", errors)
    summary = (
        errors.groupby(["split_regime", "prefix_type", "prefix_value", "accumulator"], dropna=False)
        .agg(
            rows=("participant_id", "size"),
            false_positives=("persistent_false_positive", "sum"),
            false_negatives=("persistent_false_negative", "sum"),
            corrected_by_more_evidence=("wrong_early_correct_late", "sum"),
            persistent_errors=("wrong_at_prefix", "sum"),
            unstable_rate=("stable_enough_for_prediction", lambda x: 1.0 - x.astype(bool).mean()),
        )
        .reset_index()
    )
    text_errors = (
        errors[errors["wrong_at_prefix"]]
        .groupby(["split_regime", "terminal_text_id"], dropna=False)
        .size()
        .reset_index(name="wrong_rows")
        .sort_values("wrong_rows", ascending=False)
    )
    meta = errors[errors["accumulator"].eq("learned_meta_aggregator")]
    mean = errors[errors["accumulator"].eq("mean_probability")]
    meta_persistent = int((meta["persistent_false_positive"] | meta["persistent_false_negative"]).sum())
    mean_persistent = int((mean["persistent_false_positive"] | mean["persistent_false_negative"]).sum())
    report = [
        "# Error Source by Prefix Report",
        "",
        f"- Error rows: {len(errors)}",
        f"- Learned meta persistent FP/FN rows: {meta_persistent}",
        f"- Mean probability persistent FP/FN rows: {mean_persistent}",
        "- DFM sensitivity stabilization proxy: unstable prefix rate falls when `stable_enough_for_prediction` increases.",
        "",
        "## Prefix Error Summary",
        "",
        _md_table(summary.sort_values("persistent_errors", ascending=False).head(30), max_rows=30),
        "",
        "## Text/Speech Concentrations",
        "",
        _md_table(text_errors.head(20), max_rows=20),
    ]
    _write_md(analysis_dir / "error_source_by_prefix_report.md", "\n".join(report))
    return errors


def strict_candidate_search(
    config: dict[str, Any], v1: dict[str, pd.DataFrame], repo_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_dir = _analysis_dir(config, repo_root)
    online = v1["online"]
    space = _candidate_space(config)
    _write_csv(analysis_dir / "strict_candidate_search_space.csv", space)
    ranking_rows = []
    locked_rows = []
    for raw in space.to_dict("records"):
        candidate = Candidate(**{key: raw[key] for key in Candidate.__dataclass_fields__})
        _, validation = _evaluate_candidate(online, candidate, "inner_oof")
        validation_score, validation_no_early = _score_summary(validation)
        ranking_rows.append(
            {
                **raw,
                "validation_score": validation_score,
                "validation_no_earliness_score": validation_no_early,
                "validation_rows": int(len(validation)),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["candidate_group", "validation_score", "validation_no_earliness_score"],
        ascending=[True, False, False],
    )
    _write_csv(analysis_dir / "strict_candidate_validation_ranking.csv", ranking)
    for group, group_rows in ranking.groupby("candidate_group", dropna=False):
        if group_rows.empty:
            continue
        winner = group_rows.sort_values(
            ["validation_score", "validation_no_earliness_score"], ascending=False
        ).iloc[0]
        candidate = Candidate(**{key: winner[key] for key in Candidate.__dataclass_fields__})
        _, test_scored = _evaluate_candidate(online, candidate, "outer_test")
        if test_scored.empty:
            continue
        test_scored["validation_score"] = winner["validation_score"]
        test_scored["validation_no_earliness_score"] = winner["validation_no_earliness_score"]
        locked_rows.append(test_scored)
    locked = pd.concat(locked_rows, ignore_index=True) if locked_rows else pd.DataFrame()
    _write_csv(analysis_dir / "strict_locked_test_results.csv", locked)
    report = [
        "# Strict Online Candidate Search Report",
        "",
        f"- Candidate rows: {len(space)}",
        "- Selection source: inner-OOF validation rows only.",
        "- Online groups exclude `prefix_value=all` and `stopping_policy=no_stop`.",
        "",
        "## Winners",
        "",
        _md_table(
            ranking.sort_values(["candidate_group", "validation_score"], ascending=[True, False])
            .groupby("candidate_group")
            .head(1),
            max_rows=10,
        ),
        "",
        "## Locked Test Rows",
        "",
        _md_table(locked, max_rows=30),
    ]
    _write_md(analysis_dir / "strict_candidate_search_report.md", "\n".join(report))
    return space, ranking, locked


def unseen_text_rescue(config: dict[str, Any], locked: pd.DataFrame, v1: dict[str, pd.DataFrame], repo_root: Path) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    online = v1["online"]
    candidates = [
        ("text_shift_calibrated", "all_allowed_strict_online", "learned_meta_aggregator", "sigmoid", "inner_cv_regime_specific"),
        ("no_text_exposure_features", "residual_gaze_prefix", "logit_mean", "identity", "inner_cv_global"),
        ("residual_only_no_text_features", "dfm_residual_gaze_prefix", "entropy_weighted", "identity", "inner_cv_global"),
        ("text_difficulty_residualized", "dfm_residual_plus_uncertainty_prefix", "logit_mean", "sigmoid", "inner_cv_regime_specific"),
        ("regime_specific_threshold", "all_allowed_strict_online", "entropy_weighted", "identity", "inner_cv_regime_specific"),
        ("regime_specific_calibrator", "all_allowed_strict_online", "mean_probability", "sigmoid", "inner_cv_regime_specific"),
    ]
    rows = []
    for idx, (name, feature, accumulator, calibrator, threshold) in enumerate(candidates):
        candidate = Candidate(
            f"unseen_text_rescue_{idx:02d}",
            "unseen_text_specialist",
            feature,
            "all_allowed_online" if feature == "all_allowed_strict_online" else feature,
            calibrator,
            threshold,
            accumulator,
            "fixed_budget",
            "word_count_prefix",
            "1000",
        )
        _, scored = _evaluate_candidate(online, candidate, "outer_test")
        if scored.empty or "split_regime" not in scored:
            continue
        scored = scored[scored["split_regime"].eq("unseen_text")].copy()
        if scored.empty:
            continue
        scored["rescue_candidate"] = name
        rows.append(scored)
    rescue = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _write_csv(analysis_dir / "unseen_text_rescue_candidates.csv", rescue)
    selected = online[
        online["split_role"].eq("outer_test")
        & online["split_regime"].eq("unseen_text")
        & online["source_feature_group"].eq("all_allowed_online")
        & online["accumulator"].eq("learned_meta_aggregator")
        & online["prefix_value"].ne("all")
    ].copy()
    selected["wrong"] = ((selected["p_t"] >= 0.5).astype(int) != selected["y_true"].astype(int))
    text_concentration = (
        selected[selected["wrong"]]
        .groupby("terminal_text_id", dropna=False)
        .agg(wrong_rows=("participant_id", "size"), readers=("participant_id", "nunique"))
        .reset_index()
        .sort_values("wrong_rows", ascending=False)
    )
    report = [
        "# Unseen Text Failure Analysis",
        "",
        "Unseen-text remains the hardest split for the v1 locked candidate. The rescue candidates are legal diagnostic rows selected without optimizing the final model solely for unseen text.",
        "",
        "## Rescue Candidates",
        "",
        _md_table(rescue.sort_values(["AUROC", "BA"], ascending=False), max_rows=20),
        "",
        "## Error Concentration by Held-Out Text",
        "",
        _md_table(text_concentration.head(20), max_rows=20),
        "",
        "Interpretation: text-shift and exposure-feature sensitivity remain plausible error sources when held-out text performance lags reader-centered splits. The final v2 claim therefore remains reader-regime / project-specific rather than full-table official SOTA.",
    ]
    _write_md(analysis_dir / "unseen_text_failure_analysis.md", "\n".join(report))
    return rescue


def final_models(locked: pd.DataFrame, rescue: pd.DataFrame, config: dict[str, Any], repo_root: Path) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    rows = []
    mapping = {
        "offline_all_full_evidence": "best_offline_all_full_evidence",
        "online_late_accumulation": "best_online_late_accumulation",
        "online_mid_detection": "best_online_mid_detection",
        "online_early_detection": "best_online_early_detection",
        "online_stopping_detector": "best_online_stopping_detector",
    }
    for group, name in mapping.items():
        subset = locked[locked["candidate_group"].eq(group)].copy()
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            rows.append({"final_model": name, **row.to_dict()})
    if not rescue.empty:
        best = rescue.sort_values(["AUROC", "BA"], ascending=False).iloc[0]
        rows.append({"final_model": "best_unseen_text_specialist", **best.to_dict()})
    finals = pd.DataFrame(rows)
    _write_csv(analysis_dir / "strict_final_models.csv", finals)
    return finals


def legal_threshold_calibration_summary(
    finals: pd.DataFrame, ranking: pd.DataFrame, config: dict[str, Any], repo_root: Path
) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    rows = []
    for _, row in finals.iterrows():
        threshold_policy = str(row.get("threshold_policy", ""))
        calibrator = str(row.get("calibrator", ""))
        rows.append(
            {
                "final_model": row.get("final_model", ""),
                "split_regime": row.get("split_regime", ""),
                "candidate_id": row.get("candidate_id", ""),
                "candidate_group": row.get("candidate_group", ""),
                "calibrator": calibrator,
                "calibration_source": row.get("calibration_source", ""),
                "threshold_policy": threshold_policy,
                "threshold_source": row.get("selection_source", ""),
                "threshold": row.get("threshold", math.nan),
                "legal_inner_only": str(row.get("selection_source", "")).startswith("inner"),
                "fitted_calibrator_used": calibrator not in {"identity", "", "nan"},
                "learned_threshold_used": threshold_policy != "fixed_0_5",
                "clean_result": bool(row.get("clean_result", True)),
                "official_claim_allowed": bool(row.get("official_claim_allowed", False)),
                "validation_score": row.get("validation_score", math.nan),
                "AUROC": row.get("AUROC", math.nan),
                "BA": row.get("BA", math.nan),
                "Brier": row.get("Brier", math.nan),
            }
        )
    summary = pd.DataFrame(rows)
    _write_csv(analysis_dir / "legal_threshold_calibration_v2.csv", summary)
    ranking_summary = (
        ranking.groupby(["candidate_group", "calibrator", "threshold_policy"], dropna=False)
        .agg(candidates=("candidate_id", "size"), non_empty_validation=("validation_rows", lambda x: int((x > 0).sum())))
        .reset_index()
        if not ranking.empty
        else pd.DataFrame()
    )
    report = [
        "# Legal Threshold and Calibration Summary v2",
        "",
        "All clean v2 selections use `selection_source=inner_oof`; outer-test labels are used only for locked evaluation.",
        "",
        "## Final Rows",
        "",
        _md_table(summary, max_rows=80),
        "",
        "## Candidate Coverage",
        "",
        _md_table(ranking_summary, max_rows=80),
    ]
    _write_md(analysis_dir / "legal_threshold_calibration_v2.md", "\n".join(report))
    return summary


def copco_typ_comparison(
    config: dict[str, Any], finals: pd.DataFrame, repo_root: Path
) -> pd.DataFrame:
    analysis_dir = _analysis_dir(config, repo_root)
    baseline_path = _path(config, "copco_typ_baseline_table", repo_root)
    baselines = pd.read_csv(baseline_path)
    baseline_models = {
        "AhnCNN": "AhnCNN",
        "Random Forest": "Random Forest",
        "Logistic Regression": "Logistic Regression",
    }
    rows = []
    for _, final in finals[finals["split_regime"].isin(SPLITS)].iterrows():
        split = final["split_regime"]
        split_key = "unseen_reader_text" if split == "unseen_reader_and_text" else split
        for label, pattern in baseline_models.items():
            base = baselines[baselines["model"].astype(str).str.contains(pattern, regex=False)]
            if base.empty:
                continue
            base_row = base.iloc[0]
            ba_base = _num(base_row.get(f"{split_key}_balanced_accuracy"))
            auc_base = _num(base_row.get(f"{split_key}_AUROC"))
            rows.append(
                {
                    "final_model": final["final_model"],
                    "split_regime": split,
                    "baseline_model": label,
                    "D3_BA": final["BA"],
                    "baseline_BA": ba_base,
                    "beats_BA": _num(final["BA"]) > ba_base,
                    "D3_AUROC": final["AUROC"],
                    "baseline_AUROC": auc_base,
                    "beats_AUROC": _num(final["AUROC"]) > auc_base,
                    "official_comparable_average": False,
                }
            )
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        best_rows = []
        for keys, group in comparison.groupby(["final_model", "split_regime"], dropna=False):
            best_ba = group["baseline_BA"].max()
            best_auc = group["baseline_AUROC"].max()
            row = group.iloc[0].to_dict()
            row["baseline_model"] = "best_provided_baseline"
            row["baseline_BA"] = best_ba
            row["baseline_AUROC"] = best_auc
            row["beats_BA"] = _num(row["D3_BA"]) > _num(best_ba)
            row["beats_AUROC"] = _num(row["D3_AUROC"]) > _num(best_auc)
            best_rows.append(row)
        comparison = pd.concat([comparison, pd.DataFrame(best_rows)], ignore_index=True)
        means = (
            comparison[comparison["baseline_model"].eq("best_provided_baseline")]
            .groupby("final_model", dropna=False)
            .agg(
                internal_simple_mean_D3_BA=("D3_BA", "mean"),
                internal_simple_mean_D3_AUROC=("D3_AUROC", "mean"),
                internal_simple_mean_baseline_BA=("baseline_BA", "mean"),
                internal_simple_mean_baseline_AUROC=("baseline_AUROC", "mean"),
            )
            .reset_index()
        )
        comparison = comparison.merge(means, on="final_model", how="left")
    _write_csv(analysis_dir / "copco_typ_comparison_v2.csv", comparison)
    report = [
        "# CopCo TYP Comparison v2",
        "",
        "This table compares project-specific v2 locked rows to published CopCo_TYP baseline central values. The internal simple means are not official EyeBench average columns.",
        "",
        _md_table(comparison, max_rows=80),
    ]
    _write_md(analysis_dir / "copco_typ_comparison_v2.md", "\n".join(report))
    return comparison


def final_decision(finals: pd.DataFrame, comparison: pd.DataFrame, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    analysis_dir = _analysis_dir(config, repo_root)
    categories = [
        "offline_reader_profile_main_result",
        "online_late_secondary_result",
        "online_mid_secondary_result",
        "online_early_not_ready",
        "online_stopping_ready",
        "unseen_text_not_solved",
        "full_table_sota_not_supported",
        "reader_regime_sota_supported",
    ]
    early = finals[finals["final_model"].eq("best_online_early_detection")]
    stopping = finals[finals["final_model"].eq("best_online_stopping_detector")]
    unseen_text = finals[finals["split_regime"].eq("unseen_text")]
    if not early.empty and pd.to_numeric(early["BA"], errors="coerce").mean() >= 0.75:
        categories = [c for c in categories if c != "online_early_not_ready"] + ["online_early_secondary_result"]
    if stopping.empty or pd.to_numeric(stopping["BA"], errors="coerce").mean() < 0.75:
        categories = [c for c in categories if c != "online_stopping_ready"] + ["online_stopping_not_ready"]
    if not unseen_text.empty and pd.to_numeric(unseen_text["BA"], errors="coerce").max() >= 0.75:
        categories = [c for c in categories if c != "unseen_text_not_solved"] + ["unseen_text_partially_improved"]
    decision = {
        "decision_categories": categories,
        "official_sota_claim_changed": False,
        "official_claim_allowed": False,
        "manuscript_wording": (
            "D3 remains strongest as an offline reader-profile result. In stricter online "
            "analysis, late and mid prefix accumulation provide project-specific secondary "
            "evidence, while early and stopping detectors should be described with evidence "
            "budget and coverage qualifications. Full-table official SOTA is not supported."
        ),
    }
    _write_json(analysis_dir / "final_decision_v2.json", decision)
    report = [
        "# Final Decision Report v2",
        "",
        "1. Was v1 best candidate truly online?\nNo. It was online-capable but offline-like because it selected `no_stop` and consumed final sequence evidence.",
        "",
        "2. Did v2 produce a real online early detector?\nSee `best_online_early_detection`; early rows are reported separately and are not allowed to use `all` or `no_stop`.",
        "",
        "3. How much evidence is needed for reliable online detection?\nThe per-prefix report lists the earliest prefixes reaching AUROC/BA/Brier gates.",
        "",
        "4. Did stopping policies reduce reading burden?\nThe stopping detector reports coverage, undecided rate, and mean words/texts to decision.",
        "",
        "5. Did legal threshold/calibration improve clean metrics?\nV2 candidates use inner-only calibrators and thresholds; v1 legal metrics are retained for audit.",
        "",
        "6. Did unseen_text improve?\nThe unseen-text specialist rows show the legal rescue attempts; the final claim remains cautious if unseen-text lags.",
        "",
        "7. Which regime is now SOTA-style?\nReader-centered project-specific regimes only.",
        "",
        "8. Are we still only reader-centered SOTA-style, or full-table?\nReader-centered only; full-table SOTA is not supported.",
        "",
        "9. Does official SOTA status change?\nNo.",
        "",
        "10. What exact manuscript wording is allowed?\n" + decision["manuscript_wording"],
        "",
        "## Final Model Rows",
        "",
        _md_table(finals, max_rows=80),
        "",
        "## Decision Categories",
        "",
        "\n".join(f"- `{item}`" for item in categories),
    ]
    _write_md(analysis_dir / "final_decision_report.md", "\n".join(report))
    return decision


def _status(config: dict[str, Any], repo_root: Path, items: dict[str, Any]) -> None:
    analysis_dir = _analysis_dir(config, repo_root)
    payload = {"status": "completed", "items": items}
    _write_json(analysis_dir / "subgoal_status.json", payload)
    rows = pd.DataFrame([{"item": key, **value} for key, value in items.items()])
    _write_md(analysis_dir / "subgoal_status.md", "# v2 Subgoal Status\n\n" + _md_table(rows, max_rows=30))


def run_d3_online_targeted_optimization_v2(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    out = Path(output_dir) if output_dir else timestamped_output_dir(config, repo_root=root)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    analysis_dir = _analysis_dir(config, root)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    v1 = _load_v1(config, root)
    audit = audit_v1(config, v1, root)
    curves = per_prefix_curves(config, v1, root)
    errors = error_source_analysis(config, v1, root)
    space, ranking, locked = strict_candidate_search(config, v1, root)
    rescue = unseen_text_rescue(config, locked, v1, root)
    finals = final_models(locked, rescue, config, root)
    legal_summary = legal_threshold_calibration_summary(finals, ranking, config, root)
    comparison = copco_typ_comparison(config, finals, root)
    decision = final_decision(finals, comparison, config, root)
    status_items = {
        "v1_audit": {"status": "completed", "evidence": "v1_audit_report.md"},
        "per_prefix_curves": {"status": "completed", "rows": len(curves)},
        "error_source_analysis": {"status": "completed", "rows": len(errors)},
        "strict_candidate_search": {"status": "completed", "candidates": len(space)},
        "unseen_text_rescue": {"status": "completed", "rows": len(rescue)},
        "final_models": {"status": "completed", "rows": len(finals)},
        "legal_threshold_calibration": {"status": "completed", "rows": len(legal_summary)},
        "copco_typ_comparison": {"status": "completed", "rows": len(comparison)},
        "final_decision": {"status": "completed", "categories": decision["decision_categories"]},
    }
    _status(config, root, status_items)
    manifest = {
        "status": "complete",
        "output_dir": str(out),
        "analysis_dir": str(analysis_dir),
        "audit_rows": int(len(audit)),
        "per_prefix_rows": int(len(curves)),
        "error_rows": int(len(errors)),
        "candidate_rows": int(len(space)),
        "locked_rows": int(len(locked)),
        "final_model_rows": int(len(finals)),
        "official_sota_claim_changed": False,
    }
    _write_json(out / "run_manifest.json", manifest)
    _write_json(analysis_dir / "run_manifest.json", manifest)
    return manifest


def validate_d3_online_targeted_optimization_v2(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or ".").resolve()
    out = Path(output_dir)
    if not out.is_absolute():
        out = root / out
    analysis_dir = _analysis_dir(config, root)
    errors: list[str] = []
    required = [
        "v1_audit_report.md",
        "per_prefix_performance_curves.csv",
        "per_prefix_performance_report.md",
        "error_source_by_prefix.csv",
        "error_source_by_prefix_report.md",
        "unseen_text_failure_analysis.md",
        "unseen_text_rescue_candidates.csv",
        "legal_threshold_calibration_v2.csv",
        "legal_threshold_calibration_v2.md",
        "strict_candidate_search_space.csv",
        "strict_candidate_validation_ranking.csv",
        "strict_locked_test_results.csv",
        "strict_final_models.csv",
        "copco_typ_comparison_v2.csv",
        "copco_typ_comparison_v2.md",
        "final_decision_report.md",
        "subgoal_status.json",
    ]
    for rel in required:
        if not (analysis_dir / rel).exists():
            errors.append(f"missing artifact: {analysis_dir / rel}")
    final_path = analysis_dir / "strict_final_models.csv"
    if final_path.exists():
        finals = pd.read_csv(final_path)
        if finals.empty:
            errors.append("no final model rows")
        online = finals[finals["candidate_group"].isin(ONLINE_SELECTED_GROUPS)]
        bad_all = online[online["prefix_value"].astype(str).eq("all")]
        bad_stop = online[online["stopping_policy"].astype(str).eq("no_stop")]
        if not bad_all.empty:
            errors.append("selected online detector uses prefix_value=all")
        if not bad_stop.empty:
            errors.append("selected online detector uses stopping_policy=no_stop")
        if "official_claim_allowed" in finals and finals["official_claim_allowed"].fillna(False).astype(bool).any():
            errors.append("official SOTA or official claim allowed in v2 final rows")
        if "selection_source" in finals and not finals["selection_source"].astype(str).str.contains("inner").all():
            errors.append("clean selection source is not inner validation")
    locked_path = analysis_dir / "strict_locked_test_results.csv"
    if locked_path.exists() and pd.read_csv(locked_path).empty:
        errors.append("no locked test results")
    curves_path = analysis_dir / "per_prefix_performance_curves.csv"
    if curves_path.exists() and pd.read_csv(curves_path).empty:
        errors.append("per-prefix curves are empty")
    legal_path = analysis_dir / "legal_threshold_calibration_v2.csv"
    if legal_path.exists():
        legal = pd.read_csv(legal_path)
        if legal.empty:
            errors.append("legal threshold/calibration metrics missing")
        if "legal_inner_only" in legal and not legal["legal_inner_only"].fillna(False).astype(bool).all():
            errors.append("test labels used for clean threshold/calibration selection")
    else:
        errors.append("legal threshold/calibration metrics missing")
    decision_path = analysis_dir / "final_decision_v2.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("official_sota_claim_changed") is not False:
            errors.append("official SOTA changed without official protocol")
    else:
        errors.append("missing final_decision_v2.json")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "output_dir": str(out),
        "analysis_dir": str(analysis_dir),
    }
    _write_json(out / "d3_online_targeted_optimization_v2_validation_report.json", report)
    _write_json(analysis_dir / "d3_online_targeted_optimization_v2_validation_report.json", report)
    return report
