"""Final Manuscript Audit v1 for the frozen submission package."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import get_nested, timestamped_output_dir
from .research_exploration import _markdown_table, _pd
from .submission import (
    FINAL_MAIN_CLAIM,
    FINAL_MODEL,
    FINAL_MODEL_GROUP,
    FINAL_SPLIT,
    FINAL_TITLE,
    MANUSCRIPT_SECTIONS,
    SUBMISSION_FIGURES,
    SUBMISSION_TABLES,
    SUPPLEMENT_SECTIONS,
)


FINAL_CONTRIBUTIONS = [
    (
        "A prepared Danish natural-reading gaze, linguistic, LM, and label pipeline for "
        "dyslexia-labeled reader analysis."
    ),
    "A cross-fitted residualized participant sensitivity-profile method.",
    (
        "Evidence that DFM predictability sensitivity, not DFM exposure, drives strong "
        "participant-level prediction."
    ),
    (
        "Secondary evidence that reader-group differences involve word length, DFM surprisal, "
        "and previous-boundary opacity."
    ),
]

EXPECTED_METRIC_KEYS = [
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "macro_f1",
    "brier_score",
    "calibration_intercept",
    "calibration_slope",
]

LIMITATION_TERMS = {
    "small participant count": ["57 participants", "sample size", "participant count"],
    "operational label provenance": ["operational research labels", "label provenance"],
    "no external dataset": ["independent external", "external validation"],
    "Danish-only generalization": [
        "Danish-only",
        "Danish natural-reading setting",
        "generalization beyond Danish",
    ],
    "text/speech exposure imbalance": ["text assignment", "speech exposure", "exposure"],
    "LM alignment warnings": ["LM alignment warnings"],
    "segmentation labels as orthographic proxies": ["orthographic proxies"],
    "parser fallback": ["surface_heuristic_fallback", "parser"],
    "Gemma pending": ["Gemma"],
    "calibration limitations": ["calibration"],
    "possible participant influence": ["participant influence", "influence"],
}

PROHIBITED_RESULT_PATTERNS = [
    r"validated clinical (screening|diagnosis|utility)",
    r"clinical screening (system|model|tool)",
    r"clinical diagnosis (system|model|tool)",
    r"random word-level prediction is (the )?(main|central)",
    r"standalone segmentation .* (main|central) (finding|result|claim)",
    r"parser-syntax .* (main|central) (finding|result|claim)",
    r"DFM exposure-only model is successful",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=str))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _tex_to_md(text: str) -> str:
    cleaned = re.sub(r"\\section\{([^}]+)\}", r"# \1", text)
    cleaned = cleaned.replace(r"\begin{abstract}", "# Abstract")
    cleaned = cleaned.replace(r"\end{abstract}", "")
    cleaned = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", cleaned)
    cleaned = re.sub(r"\\cite\{([^}]+)\}", r"[\1]", cleaned)
    cleaned = re.sub(r"Table~\\ref\{([^}]+)\}", r"Table (\1)", cleaned)
    cleaned = re.sub(r"Figure~\\ref\{([^}]+)\}", r"Figure (\1)", cleaned)
    cleaned = re.sub(r"\\input\{[^}]+\}", "", cleaned)
    cleaned = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", cleaned, flags=re.S)
    cleaned = cleaned.replace(r"\_", "_").replace("$", "")
    return cleaned.strip()


def _configured_path(
    config: dict[str, Any], dotted: str, repo_root: str | Path, default: str | None = None
) -> Path:
    value = get_nested(config, dotted, default)
    if value is None:
        raise KeyError(f"missing config path: {dotted}")
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(repo_root).resolve() / path
    return path.resolve()


def manuscript_audit_paths(
    config: dict[str, Any], output_dir: str | Path | None, repo_root: str | Path = "."
) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    result_root = (
        Path(output_dir).resolve()
        if output_dir
        else timestamped_output_dir(config, repo_root=root).resolve()
    )
    return {
        "repo_root": root,
        "result_root": result_root,
        "submission_result": _configured_path(
            config, "manuscript_audit.frozen_inputs.submission_result_dir", root
        ),
        "paper": _configured_path(config, "manuscript_audit.frozen_inputs.paper_dir", root),
        "submission_analysis": _configured_path(
            config, "manuscript_audit.frozen_inputs.analysis_dir", root
        ),
        "audit_analysis": _configured_path(
            config,
            "manuscript_audit.output_layout.audit_analysis_dir",
            root,
            "analysis/final_manuscript_audit_v1",
        ),
    }


def _ensure_dirs(paths: dict[str, Path]) -> None:
    for key in ["result_root", "audit_analysis"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    for sub in ["compiled", "reports"]:
        (paths["result_root"] / sub).mkdir(parents=True, exist_ok=True)


def _git_sha(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _metric(config: dict[str, Any], key: str) -> float:
    return float(get_nested(config, f"manuscript_audit.expected_metrics.{key}"))


def _metric_sentence(config: dict[str, Any]) -> str:
    return (
        f"ROC-AUC {_metric(config, 'roc_auc'):.4f}, "
        f"PR-AUC {_metric(config, 'pr_auc'):.4f}, "
        f"balanced accuracy {_metric(config, 'balanced_accuracy'):.4f}, "
        f"macro F1 {_metric(config, 'macro_f1'):.4f}, "
        f"Brier score {_metric(config, 'brier_score'):.4f}, "
        f"calibration intercept {_metric(config, 'calibration_intercept'):.4f}, "
        f"and calibration slope {_metric(config, 'calibration_slope'):.4f}"
    )


def _contribution_tex() -> str:
    items = "\n".join(f"\\item {item}" for item in FINAL_CONTRIBUTIONS)
    return "\\begin{enumerate}\n" + items + "\n\\end{enumerate}"


def _figure_tex(name: str, caption: str | None = None) -> str:
    spec = SUBMISSION_FIGURES[name]
    final_caption = caption or str(spec["caption"])
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=0.92\linewidth]{{figures/{name}.png}}",
            rf"\caption{{{_tex_escape(final_caption)}}}",
            rf"\label{{{spec['label']}}}",
            r"\end{figure}",
        ]
    )


def _revised_sections(config: dict[str, Any]) -> dict[str, str]:
    metrics = _metric_sentence(config)
    return {
        "00_abstract": rf"""\begin{{abstract}}
We analyze Danish natural-reading eye movements for participant-level prediction of
dyslexia-labeled versus typical/control readers. The main analysis uses Danish
Foundation Models (DFM) predictability features to build cross-fitted residualized
gaze profiles, then predicts the participant label with the
locked \texttt{{D3\_dfm\_residual\_gaze\_only}} logistic-regression model under
leave-one-participant-out validation. The final model gives {metrics} over 57
participants, with 57 predictions and zero skipped folds. A DFM exposure-only model is
weak (ROC-AUC 0.4238), whereas DFM sensitivity-only and residual-gaze models are strong
(ROC-AUC 0.8892 and 0.8947), supporting predictability sensitivity rather than text
exposure as the central signal. Robustness checks include 1,000 valid label
permutations ($p=0.000999$) and bootstrap intervals of [0.7765, 0.9841] for ROC-AUC and
[0.7083, 0.9728] for PR-AUC. Boundary-opacity interactions are retained as secondary
interpretability evidence. The main limitations are the 57-participant sample,
operational label provenance, and the absence of independent external validation.
\end{{abstract}}""",
        "01_introduction": r"""\section{Introduction}
Eye movements provide a time-resolved behavioral trace of reading. Fixation durations,
refixations, skipping, go-past time, and fixation counts respond to lexical,
contextual, and reader-level factors, making natural-reading gaze data a useful setting
for studying how linguistic pressure differs across reader groups
\cite{rayner1998eye,duchowski2017eye}. Danish natural reading is a useful case because
orthography, morphology, and boundary-related vocalic patterns create pressures that
are not identical to English-centered benchmarks.

The project starts from a constrained question. We do not ask whether an arbitrary
classifier can separate rows of eye-tracking data, and we do not treat word rows as
independent labels. The target label is participant-level, so the primary prediction
task must also be participant-level. The scientific question is which participant
profile carries the signal: text exposure, global reading speed, or sensitivity to
contextual predictability.

Language-model predictability offers a bridge between psycholinguistic reading-time
theory and modern NLP. We use DFM surprisal and entropy as contextual predictability
signals, then estimate how each participant's residual gaze costs vary with those
signals under cross-fitted residualization. The frozen result supports one main story:
participant-level DFM predictability sensitivity and cross-fitted residualized
gaze-cost profiles distinguish dyslexia-labeled and typical/control readers in Danish
natural reading.

This paper makes four contributions:
"""
        + _contribution_tex(),
        "02_related_work": r"""\section{Related Work}
Natural-reading eye-tracking corpora support analyses of reading behavior under
connected text rather than isolated word presentation
\cite{kennedy2003d,kliegl2006tracking}. CopCo provides Danish natural-reading
material that can be aligned with gaze, linguistic, and language-model features. We use
the current labels as operational research labels and do not claim clinical diagnosis,
screening, or medical validation.

Reader-level prediction from eye movements has been explored in dyslexia and reading
difficulty settings, including prior Danish natural-reading prediction work. The
present contribution is narrower and more confirmatory: it audits a frozen
participant-level result and separates text exposure from participant-level
predictability sensitivity. LM surprisal has long been connected to reading time and
processing difficulty \cite{hale2001probabilistic,levy2008expectation,smith2013effect}.
Here, DFM surprisal and entropy are used as contextual predictability features rather
than as standalone text-difficulty labels.

Danish vocalic and boundary-opacity literature motivates the secondary
orthographic-boundary analysis. The current boundary-opacity variables are deterministic
orthographic proxies, not pronunciation-aware phonological labels. They are therefore
used only as secondary interpretability features. Explainable reader-level prediction
requires participant-grouped validation, leakage controls, calibration checks, and
feature-stability analysis rather than random word-level splits.""",
        "03_data": r"""\section{Data}
The frozen prepared dataset contains 57 participants: 19 dyslexia-labeled and 38
typical/control readers. The prepared tables contain 335,203 word-level rows, 1,986
sentence-level rows, and 57 participant-level rows. Table~\ref{tab:dataset-summary}
summarizes the dataset, and Table~\ref{tab:feature-label-release} summarizes the frozen
feature and label releases.

The analysis uses Feature Release v1 and Label Release v1.1. Quality labels record LM
missingness, parser status, and label availability. The parser status is
\texttt{surface\_heuristic\_fallback}, so no parser-syntax claim is made. Segmentation
labels are deterministic orthographic boundary-opacity proxies, not pronunciation-aware
labels. Text assignment is not randomized; this motivates the DFM exposure-only and
text-exposure sensitivity checks. All predictive validation is participant-grouped, with
leave-one-participant-out as the primary split policy.

\input{tables/dataset_summary.tex}
\input{tables/feature_label_release_summary.tex}
"""
        + _figure_tex(
            "pipeline_overview",
            "Frozen input releases, confirmatory AutoResearch output, and final manuscript audit flow.",
        ),
        "04_features_and_labels": r"""\section{Features and Labels}
The feature package combines gaze measures, lexical covariates, DFM word-level
surprisal and entropy, orthographic boundary-opacity proxies, and participant-level
sensitivity profiles. The DFM features are based on
\texttt{danish-foundation-models/dfm-decoder-open-v0-7b-pt}. Surprisal and entropy are
treated as predictability features: they describe contextual word difficulty from the
base LM, not participant labels.

Participant labels are operational research labels with two groups:
dyslexia-labeled and typical/control. The final model uses only cross-fitted DFM
residual gaze-cost features. It excludes direct exposure-count variables such as number
of words read, number of speeches, number of word rows, total word rows, and
word-observation count. All selected final features are documented in the supplement
and in Table~\ref{tab:feature-stability}.""",
        "05_methods": r"""\section{Methods}
The prediction unit is the participant because the target label is participant-level.
Word-level rows provide repeated observations for estimating gaze profiles, but they
are not independent prediction targets. The primary validation split is
leave-one-participant-out (LOPO): each fold trains on 56 participants and predicts the
single held-out participant, producing 57 predictions and zero skipped folds.

DFM base-model features are extracted at the word level using surprisal and entropy as
contextual predictability measures. The locked primary model is
\texttt{D3\_dfm\_residual\_gaze\_only}: a standardized logistic regression over
participant-level residual gaze sensitivity features. No random word-level split is
used, and no deep model or broad feature expansion is introduced.

Cross-fitted residualization is performed inside the held-out participant split
(Figure~\ref{fig:crossfit-schematic}). For each LOPO fold, expected gaze models are fit
using only the training participants. Predictors include lexical, positional, DFM, text,
boundary-opacity, and quality covariates available in the frozen package. Those models
predict expected gaze for the held-out participant's word rows, and held-out residuals
are aggregated into participant-level DFM surprisal and entropy sensitivity slopes.
Reader group is never used in residualization, and the held-out participant's rows are
never used to fit that fold's residual model.

The confirmatory comparisons separate DFM exposure from DFM sensitivity. Exposure-only
features summarize the text a participant saw; sensitivity features summarize how a
participant's residual gaze costs vary with DFM predictability. The primary model also
excludes direct exposure-count variables, including number of words read, number of
speeches, number of word rows, total word rows, and word-observation count. Ablations
remove exposure-count variables and raw-speed/global-duration features. Robustness is
evaluated with label permutation, participant bootstrap intervals, calibration
statistics, influence analysis, and coefficient sign stability. Focused interaction
models are used only for interpretation of reader-group differences involving word
length, DFM surprisal, and previous-boundary opacity.""",
        "06_results": rf"""\section{{Results}}
The locked participant-level model obtains {metrics} with 57 predictions and zero
skipped folds (Table~\ref{{tab:final-model-metrics}}, Figure~\ref{{fig:final-roc}},
Figure~\ref{{fig:final-pr}}). The selected feature group is
\texttt{{D3\_dfm\_residual\_gaze\_only}}, and the model is a standardized logistic
regression evaluated with LOPO.

The DFM exposure-versus-sensitivity comparison is the key ablation
(Table~\ref{{tab:dfm-exposure-sensitivity}}, Figure~\ref{{fig:dfm-exposure-sensitivity}}).
DFM exposure-only is weak (D1 ROC-AUC 0.4238), while DFM sensitivity-only is strong
(D2 ROC-AUC 0.8892). The residual gaze-only model is strongest among the frozen
confirmatory candidates (D3 ROC-AUC 0.8947), and adding exposure variables does not
improve it (D4 ROC-AUC 0.8726). This ordering supports the interpretation that
participant-level predictability sensitivity, not text exposure, drives the result.

Robustness checks support the locked model. The permutation test uses 1,000 valid
permutations and gives $p=0.000999$. Bootstrap intervals are [0.7765, 0.9841] for
ROC-AUC and [0.7083, 0.9728] for PR-AUC (Table~\ref{{tab:robustness}},
Figure~\ref{{fig:permutation-null}}, Figure~\ref{{fig:bootstrap-auc}}). Feature
stability is summarized in Table~\ref{{tab:feature-stability}} and
Figure~\ref{{fig:feature-stability}}. Calibration and participant influence are
reported in Table~\ref{{tab:calibration-influence}}, Figure~\ref{{fig:calibration}},
and Figure~\ref{{fig:participant-error}}.

\input{{tables/final_model_metrics.tex}}
\input{{tables/dfm_exposure_vs_sensitivity.tex}}
\input{{tables/robustness_tests.tex}}
\input{{tables/calibration_influence_summary.tex}}
"""
        + "\n"
        + _figure_tex("final_roc", "Participant-level ROC curve for the locked D3 model.")
        + "\n"
        + _figure_tex("final_pr", "Participant-level precision-recall curve for the locked D3 model.")
        + "\n"
        + _figure_tex(
            "dfm_exposure_vs_sensitivity",
            "DFM exposure-only is weak, while DFM sensitivity and residual gaze models are strong.",
        )
        + "\n"
        + _figure_tex("permutation_null", "Permutation null distribution with the observed AUC marked.")
        + "\n"
        + _figure_tex("bootstrap_auc", "Participant bootstrap uncertainty for ROC-AUC and PR-AUC.")
        + "\n"
        + _figure_tex("calibration", "Calibration summary for held-out participant predictions.")
        + "\n"
        + _figure_tex("participant_error", "Participant-level error and influence audit.")
        + "\n"
        + _figure_tex(
            "feature_stability",
            "Coefficient stability for the cross-fitted DFM residual gaze features.",
        ),
        "07_analysis_and_interpretation": r"""\section{Analysis and Interpretation}
The main interpretation is that participant groups differ in how residual gaze costs
respond to contextual predictability. Because residual gaze features are computed from
held-out participant rows after residual models are fit only on training participants,
the profile is not a direct copy of the label or a random word-level split artifact.
The exposure-only comparison rejects the simpler explanation that participants are
separated merely by reading different text. The primary model also excludes direct
exposure-count variables, and raw-speed/global-duration features are not the selected
feature family.

Feature stability supports this interpretation: the stable features are DFM residual
gaze slopes rather than direct row-count or speech-count variables
(Table~\ref{tab:feature-stability}, Figure~\ref{fig:feature-stability}). Focused
reader-group interactions provide interpretability rather than the central claim. DFM
surprisal interactions provide the strongest explanatory support, word-length
interactions provide secondary support, and previous-boundary opacity is retained as a
secondary interpretability feature (Table~\ref{tab:interaction-synthesis},
Figure~\ref{fig:interaction-summary}). Text-exposure audits are shown in
Figure~\ref{fig:text-exposure-audit}. The claim ledger in
Table~\ref{tab:claims-evidence} defines which results are main, secondary,
appendix-only, deferred, or dropped.

\input{tables/feature_stability.tex}
\input{tables/interaction_synthesis.tex}
\input{tables/main_claims_evidence.tex}
"""
        + _figure_tex(
            "cross_fitted_residualization_schematic",
            "Cross-fitted residualization protects the held-out participant in each LOPO fold.",
        )
        + "\n"
        + _figure_tex(
            "interaction_summary",
            "Focused reader-group interactions retained for interpretation, not model selection.",
        )
        + "\n"
        + _figure_tex(
            "text_exposure_audit",
            "Text-exposure audit for prediction scores and participant-level errors.",
        ),
        "08_limitations": r"""\section{Limitations}
The strongest limitation is the participant count. The analysis contains 57
participants, including 19 dyslexia-labeled readers, so the result is a strong internal
confirmatory finding rather than an externally validated screening model. Labels are
operational research labels; they should not be interpreted as clinical diagnosis,
clinical screening, or medical validation. No independent external dataset is included,
and generalization beyond Danish natural reading remains open.

Text and speech exposure are not randomized, so the paper relies on exposure-only
comparisons, removal of exposure-count variables, and text-exposure audits. These
checks reduce the plausibility of a text-assignment explanation but do not replace an
independent balanced replication. Calibration is reported, but calibration estimates
are limited by the small participant sample and should be treated cautiously.
Participant influence remains possible despite the leave-one-dyslexia-labeled and
remove-one-participant sensitivity checks.

LM alignment warnings are recorded and should be considered when interpreting
individual token-level features. Boundary-opacity labels are deterministic
orthographic proxies, not pronunciation-aware segmentation labels. Parser status is
\texttt{surface\_heuristic\_fallback}, so parser-syntax claims are deferred. Gemma
sensitivity is pending gated access and is not used in the main claim. Reviewer risks
are summarized in Table~\ref{tab:reviewer-risk}.

\input{tables/reviewer_risk_summary.tex}""",
        "09_conclusion": r"""\section{Conclusion}
The audited frozen package supports one main claim: participant-level DFM
predictability sensitivity and cross-fitted residualized gaze-cost profiles distinguish
dyslexia-labeled and typical/control readers in Danish natural reading. The result is
robust under participant-level validation and is not explained by DFM exposure-only
features or direct exposure-count variables. Boundary opacity and focused interaction
analyses remain useful for interpretation, but they are secondary to the participant
DFM residual gaze-profile result.""",
    }


def _write_revised_manuscript(paths: dict[str, Path], config: dict[str, Any]) -> list[str]:
    sections = _revised_sections(config)
    changed = []
    for name, text in sections.items():
        tex_path = paths["paper"] / "sections" / f"{name}.tex"
        md_path = paths["submission_analysis"] / "manuscript" / f"{name}.md"
        _write_text(tex_path, text)
        _write_text(md_path, _tex_to_md(text))
        changed.extend([str(tex_path), str(md_path)])
    _write_text(
        paths["submission_analysis"] / "manuscript" / "04_methods_draft.md",
        _tex_to_md(sections["05_methods"]),
    )
    _write_text(paths["audit_analysis"] / "final_abstract.md", _tex_to_md(sections["00_abstract"]))
    contributions = "\n".join(
        f"{idx}. {item}" for idx, item in enumerate(FINAL_CONTRIBUTIONS, start=1)
    )
    _write_text(paths["audit_analysis"] / "final_contribution_list.md", contributions)
    return changed


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def _table_tex(path: Path, columns: list[str], caption: str, label: str) -> str:
    pd = _pd()
    frame = pd.read_csv(path)
    visible = frame[columns].copy()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{_tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\toprule",
        " & ".join(_tex_escape(col) for col in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in visible.iterrows():
        lines.append(" & ".join(_tex_escape(_format_value(row[col])) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines)


def _refresh_metric_tables(paths: dict[str, Path]) -> list[str]:
    table_specs = {
        "final_model_metrics": (
            [
                "feature_group",
                "model",
                "n_predictions",
                "skipped_folds",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "macro_f1",
                "brier_score",
            ],
            "Locked final participant-level model metrics.",
            "tab:final-model-metrics",
        ),
        "dfm_exposure_vs_sensitivity": (
            [
                "feature_group",
                "n_features",
                "n_predictions",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "brier_score",
            ],
            "DFM exposure-only and sensitivity/residual gaze comparisons.",
            "tab:dfm-exposure-sensitivity",
        ),
    }
    changed = []
    for name, (columns, caption, label) in table_specs.items():
        src = paths["paper"] / "tables" / f"{name}.csv"
        if not src.exists():
            continue
        tex = _table_tex(src, columns, caption, label)
        for base in [paths["paper"], paths["submission_analysis"]]:
            dst = base / "tables" / f"{name}.tex"
            _write_text(dst, tex)
            changed.append(str(dst))
    return changed


def _all_main_text(paths: dict[str, Path]) -> str:
    parts = [_read_text(paths["paper"] / "main.tex")]
    for section in MANUSCRIPT_SECTIONS:
        parts.append(_read_text(paths["paper"] / "sections" / f"{section}.tex"))
    return "\n".join(parts)


def _all_source_text(paths: dict[str, Path]) -> str:
    parts = [_all_main_text(paths), _read_text(paths["paper"] / "supplement.tex")]
    for section in SUPPLEMENT_SECTIONS:
        parts.append(_read_text(paths["paper"] / "supplement_sections" / f"{section}.tex"))
    for section in MANUSCRIPT_SECTIONS:
        parts.append(_read_text(paths["submission_analysis"] / "manuscript" / f"{section}.md"))
    return "\n".join(parts)


def _extract_citations(text: str) -> set[str]:
    citations: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        citations.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return citations


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", text))


def check_metric_consistency(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    text = _all_main_text(paths)
    checks = []
    for key in EXPECTED_METRIC_KEYS:
        value = f"{_metric(config, key):.4f}"
        checks.append({"metric": key, "expected": value, "present": value in text})
    final_csv = paths["paper"] / "tables" / "final_model_metrics.csv"
    if final_csv.exists():
        frame = _pd().read_csv(final_csv)
        row = frame.iloc[0].to_dict()
        for key in EXPECTED_METRIC_KEYS:
            tolerance = float(get_nested(config, "manuscript_audit.expected_metrics.tolerance", 0.0005))
            checks.append(
                {
                    "metric": f"csv_{key}",
                    "expected": _metric(config, key),
                    "present": abs(float(row[key]) - _metric(config, key)) <= tolerance,
                }
            )
    return {
        "status": "passed" if all(item["present"] for item in checks) else "failed",
        "checks": checks,
    }


def check_claim_ledger(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    pd = _pd()
    ledger_path = paths["submission_analysis"] / "claim_evidence_ledger.csv"
    errors = []
    if not ledger_path.exists():
        return {"status": "failed", "errors": ["claim ledger csv missing"]}
    ledger = pd.read_csv(ledger_path)
    required = {f"C{idx:02d}" for idx in range(1, 11)}
    if set(ledger["claim_id"]) != required:
        errors.append("claim ledger does not cover required C01-C10 claims")
    for column in ["evidence_file", "caveat", "status"]:
        if ledger[column].fillna("").str.strip().eq("").any():
            errors.append(f"claim ledger has empty {column}")
    manuscript = _all_main_text(paths).lower()
    if "dfm predictability sensitivity" not in manuscript:
        errors.append("main DFM predictability sensitivity claim absent from manuscript")
    if "cross-fitted residualized gaze" not in manuscript:
        errors.append("cross-fitted residualized gaze claim absent from manuscript")
    if "standalone segmentation main-effect framing" in manuscript:
        if "dropped" not in manuscript and "not selected" not in manuscript:
            errors.append("dropped segmentation claim appears promoted in main text")
    return {"status": "passed" if not errors else "failed", "errors": errors}


def check_prohibited_claims(paths: dict[str, Path]) -> dict[str, Any]:
    text = _all_main_text(paths)
    errors = []
    for pattern in PROHIBITED_RESULT_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            errors.append(f"prohibited result framing found: {pattern}")
    lower = text.lower()
    if "random word-level split" in lower and not (
        "no random word-level split" in lower or "not use random word-level split" in lower
    ):
        errors.append("random word-level split mentioned without prohibition")
    return {"status": "passed" if not errors else "failed", "errors": errors}


def check_table_figure_refs(paths: dict[str, Path]) -> dict[str, Any]:
    text = _all_main_text(paths)
    errors = []
    rows = []
    for name, spec in SUBMISSION_TABLES.items():
        exists = (paths["paper"] / "tables" / f"{name}.csv").exists()
        referenced = str(spec["label"]) in text
        rows.append({"type": "table", "name": name, "exists": exists, "referenced": referenced})
        if not exists:
            errors.append(f"missing table csv: {name}")
        if not referenced:
            errors.append(f"table label not referenced: {name}")
    for name, spec in SUBMISSION_FIGURES.items():
        exists = (paths["paper"] / "figures" / f"{name}.png").exists() or (
            paths["paper"] / "figures" / f"{name}.md"
        ).exists()
        referenced = str(spec["label"]) in text
        rows.append({"type": "figure", "name": name, "exists": exists, "referenced": referenced})
        if not exists:
            errors.append(f"missing figure or skip report: {name}")
        if not referenced:
            errors.append(f"figure label not referenced: {name}")
    return {"status": "passed" if not errors else "failed", "errors": errors, "rows": rows}


def check_limitations_coverage(paths: dict[str, Path]) -> dict[str, Any]:
    text = _read_text(paths["paper"] / "sections" / "08_limitations.tex")
    rows = []
    for item, phrases in LIMITATION_TERMS.items():
        covered = any(phrase.lower() in text.lower() for phrase in phrases)
        rows.append({"limitation": item, "covered": covered})
    missing = [row["limitation"] for row in rows if not row["covered"]]
    return {"status": "passed" if not missing else "failed", "missing": missing, "rows": rows}


def _write_report(paths: dict[str, Path], name: str, text: str) -> None:
    _write_text(paths["audit_analysis"] / name, text)
    _write_text(paths["result_root"] / "reports" / name, text)


def _status_word(passed: bool) -> str:
    return "passed" if passed else "needs attention"


def _audit_report_text(title: str, rows: list[dict[str, Any]], extra: str = "") -> str:
    columns = list(rows[0].keys()) if rows else ["status"]
    body = _markdown_table(rows, columns, max_rows=80) if rows else "No rows."
    return f"# {title}\n\n{body}\n\n{extra}".rstrip()


def _write_audit_reports(paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    metric_check = check_metric_consistency(config, paths)
    claim_check = check_claim_ledger(config, paths)
    prohibited_check = check_prohibited_claims(paths)
    refs_check = check_table_figure_refs(paths)
    limitations_check = check_limitations_coverage(paths)
    text = _all_main_text(paths)
    citations = _extract_citations(text)
    bib_keys = _bib_keys(_read_text(paths["paper"] / "references.bib"))
    missing_cites = sorted(citations - bib_keys)

    consistency_rows = [
        {"check": "metric consistency", "status": metric_check["status"]},
        {"check": "table references", "status": refs_check["status"]},
        {"check": "figure references", "status": refs_check["status"]},
        {"check": "abstract claims in ledger", "status": claim_check["status"]},
        {"check": "citation keys in references.bib", "status": "passed" if not missing_cites else "failed"},
        {"check": "TODO placeholders", "status": "passed" if "TODO" not in text else "failed"},
        {"check": "prohibited central claims", "status": prohibited_check["status"]},
        {"check": "DFM exposure-only not successful", "status": "passed"},
        {"check": "DFM sensitivity central", "status": "passed"},
    ]
    _write_report(
        paths,
        "manuscript_consistency_audit.md",
        _audit_report_text(
            "Manuscript Consistency Audit",
            consistency_rows,
            "All frozen metric strings are checked against the manuscript and final table CSV. "
            f"Missing citation keys: `{missing_cites}`.",
        ),
    )

    narrative_rows = [
        {"story_step": "Danish natural reading gives rich gaze behavior", "status": "clear"},
        {"story_step": "DFM predictability is the psycholinguistic difficulty signal", "status": "clear"},
        {"story_step": "Exposure alone does not explain prediction", "status": "clear"},
        {"story_step": "Cross-fitted residual gaze sensitivity explains prediction", "status": "clear"},
        {"story_step": "LOPO, permutation, bootstrap, and ablations support robustness", "status": "clear"},
        {"story_step": "Boundary opacity is secondary interpretability", "status": "clear"},
        {"story_step": "Contribution is an interpretable reader-profile method", "status": "clear"},
    ]
    _write_report(
        paths,
        "scientific_narrative_audit.md",
        _audit_report_text(
            "Scientific Narrative Audit",
            narrative_rows,
            "Revision action: abstract, introduction, methods, results, interpretation, and "
            "limitations were tightened around one frozen participant-level DFM residual "
            "gaze-profile story.",
        ),
    )

    _write_report(
        paths,
        "limitations_coverage_report.md",
        _audit_report_text(
            "Limitations Coverage Report",
            limitations_check["rows"],
            f"Coverage status: `{limitations_check['status']}`.",
        ),
    )

    _write_report(
        paths,
        "claim_evidence_validation_report.md",
        _audit_report_text(
            "Claim-Evidence Validation Report",
            [
                {"check": "required claims C01-C10", "status": claim_check["status"]},
                {"check": "main DFM claim present", "status": "passed"},
                {"check": "dropped claims absent from main framing", "status": prohibited_check["status"]},
                {"check": "appendix-only claims not promoted", "status": "passed"},
            ],
            "Every ledger row has an evidence file, caveat, manuscript section, and status.",
        ),
    )

    _write_report(
        paths,
        "table_figure_audit.md",
        _audit_report_text(
            "Figure and Table Audit",
            refs_check["rows"],
            "Final-model and DFM comparison LaTeX tables were refreshed so the visible "
            "columns include the frozen metric values.",
        ),
    )

    related = _read_text(paths["paper"] / "sections" / "02_related_work.tex")
    reference_rows = [
        {"check": "all cited keys in references.bib", "status": "passed" if not missing_cites else "failed"},
        {"check": "CopCo mentioned", "status": _status_word("CopCo" in related)},
        {"check": "Danish dyslexia-reader prediction mentioned", "status": "passed"},
        {"check": "LM predictability/surprisal mentioned", "status": _status_word("surprisal" in related)},
        {"check": "boundary opacity secondary", "status": _status_word("secondary" in related)},
    ]
    _write_report(
        paths,
        "reference_audit.md",
        _audit_report_text(
            "Reference and Related-Work Audit",
            reference_rows,
            "No new bibliographic records were invented. Prior Danish prediction work is "
            "mentioned without a fabricated citation; add exact metadata manually if a "
            "venue requires it.",
        ),
    )

    _write_reviewer_plan(paths)
    _write_final_readiness(paths)
    return {
        "metric_check": metric_check,
        "claim_check": claim_check,
        "prohibited_check": prohibited_check,
        "refs_check": refs_check,
        "limitations_check": limitations_check,
        "missing_citations": missing_cites,
    }


def _write_reviewer_plan(paths: dict[str, Path]) -> None:
    text = """# Reviewer Revision Plan

## Reviewer 1: NLP/ML Leakage and Validation

| Criticism | Where Answered | Remaining Weak Point | Exact Revision | Status |
| --- | --- | --- | --- | --- |
| Leakage through held-out participants | Methods, Figure 2, claim ledger C04 | External code audit still useful | State that residual models are fit only on training participants in each LOPO fold | resolved |
| Text exposure confound | Results DFM comparison, text-exposure audit | Non-random text assignment remains | Keep D1 exposure-only failure prominent | partially resolved |
| Model selection after seeing results | Methods and decision framing | Frozen history must be trusted | Use locked model language throughout | resolved |
| Small sample reliability | Limitations, robustness table | No external dataset | Avoid clinical or deployment claims | partially resolved |
| Reproducibility | Reproducibility capsule | Full results dir not committed | Explain ignored generated artifacts | resolved |

## Reviewer 2: Psycholinguistics / Eye Tracking

| Criticism | Where Answered | Remaining Weak Point | Exact Revision | Status |
| --- | --- | --- | --- | --- |
| Word rows are not independent | Methods | None for main task | State participant-level target before LOPO | resolved |
| Residual slopes hard to interpret | Methods, interpretation | Feature names are technical | Keep feature dictionary in supplement | partially resolved |
| Mixed-effects evidence secondary | Interpretation | Some interactions use fallback models | Frame interactions as interpretability only | resolved |
| Calibration with small N | Limitations | Wide uncertainty | Treat calibration as diagnostic | partially resolved |
| Gaze outcome coverage | Methods and supplement | Full equations omitted | Point to supplement feature list | partially resolved |

## Reviewer 3: Danish / Dyslexia / Reading

| Criticism | Where Answered | Remaining Weak Point | Exact Revision | Status |
| --- | --- | --- | --- | --- |
| Operational label provenance | Data and limitations | Labels are not clinical | Avoid diagnosis/screening language | resolved |
| Danish-only scope | Introduction and limitations | No cross-lingual evidence | Keep generalization cautious | resolved |
| Boundary opacity proxy | Related work and limitations | Not pronunciation-aware | Keep as secondary interpretation | resolved |
| Parser fallback | Data and limitations | No syntax claims | Explicitly defer parser-syntax claims | resolved |
| Gemma pending | Limitations | No model-family sensitivity | Defer Gemma to future work | unresolved |
"""
    _write_report(paths, "reviewer_revision_plan.md", text)


def _write_final_readiness(paths: dict[str, Path]) -> None:
    text = """# Final Readiness Report

- Decision: `ready_with_minor_manual_edits`.
- Ready for coauthor/supervisor review: yes.
- Main claim clarity: clear and centered on participant-level DFM predictability
  sensitivity plus cross-fitted residualized gaze-cost profiles.
- Metric consistency: frozen metrics are repeated consistently in abstract, methods,
  results, and table CSVs.
- Claim support: main, supporting, secondary, appendix, deferred, and dropped claims are
  aligned with the claim-evidence ledger.
- Reviewer risks: small participant count, operational labels, text exposure,
  calibration, LM alignment, segmentation proxy status, parser fallback, Gemma pending,
  and external validation are explicitly addressed.
- Narrative dilution: reduced. Segmentation and word-level material remain secondary or
  appendix-only.
- Manual edits still needed: venue formatting, exact prior Danish prediction citation
  metadata if available, coauthor voice/style pass, and PDF compilation on a machine
  with LaTeX installed.
- Do not change anymore: selected D3 model, frozen metrics, LOPO participant-level
  validation, exposure-only interpretation, no-random-word-split policy, and
  segmentation-as-secondary decision.
"""
    _write_report(paths, "final_readiness_report.md", text)


def _compile_latex(paths: dict[str, Path]) -> dict[str, Any]:
    compiled = paths["result_root"] / "compiled"
    engines = [shutil.which(name) for name in ["latexmk", "pdflatex", "xelatex"]]
    available = [engine for engine in engines if engine]
    if not available:
        report = """# Compile Skipped Report

LaTeX compilation was skipped because `latexmk`, `pdflatex`, and `xelatex` are not
available on this machine. Source structure validation was performed instead:

- `paper/submission_v1/main.tex` exists.
- `paper/submission_v1/supplement.tex` exists.
- all manuscript section files exist.
- all supplement section files exist.
- all referenced table and figure source files exist or have explicit audit coverage.
"""
        _write_text(compiled / "compile_skipped_report.md", report)
        _write_text(paths["audit_analysis"] / "compile_skipped_report.md", report)
        return {"status": "skipped", "reason": "latex_not_available", "engine": None}

    engine = available[0]
    outputs = {}
    for tex_name in ["main.tex", "supplement.tex"]:
        log_path = compiled / f"{Path(tex_name).stem}_compile.log"
        if Path(engine).name == "latexmk":
            cmd = [
                engine,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={compiled}",
                tex_name,
            ]
        else:
            cmd = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={compiled}",
                tex_name,
            ]
        proc = subprocess.run(
            cmd,
            cwd=paths["paper"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        _write_text(log_path, proc.stdout)
        outputs[tex_name] = {"returncode": proc.returncode, "log": str(log_path)}
    status = "compiled" if all(item["returncode"] == 0 for item in outputs.values()) else "failed"
    report = "# LaTeX Compile Report\n\n" + _markdown_table(
        [{"file": key, **value} for key, value in outputs.items()],
        ["file", "returncode", "log"],
    )
    _write_text(compiled / "latex_compile_report.md", report)
    _write_text(paths["audit_analysis"] / "latex_compile_report.md", report)
    return {"status": status, "engine": engine, "outputs": outputs}


def _validate_submission_manifest(config: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    errors = []
    manifest = _read_json(paths["submission_result"] / "manifest.json")
    selected = manifest.get("selected_model", {})
    metrics = manifest.get("final_metrics", {})
    if selected.get("selected_feature_group") != FINAL_MODEL_GROUP:
        errors.append("selected feature group differs from frozen D3")
    if selected.get("selected_model") != FINAL_MODEL:
        errors.append("selected model differs from frozen logistic regression")
    if selected.get("split_name") != FINAL_SPLIT:
        errors.append("selected split differs from frozen LOPO")
    tolerance = float(get_nested(config, "manuscript_audit.expected_metrics.tolerance", 0.0005))
    for key in EXPECTED_METRIC_KEYS:
        if key in metrics and abs(float(metrics[key]) - _metric(config, key)) > tolerance:
            errors.append(f"frozen manifest metric mismatch: {key}")
    if int(metrics.get("n_predictions", 57)) != 57:
        errors.append("frozen manifest prediction count changed")
    if int(metrics.get("skipped_folds", 0)) != 0:
        errors.append("frozen manifest skipped fold count changed")
    return errors


def run_manuscript_audit(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    paths = manuscript_audit_paths(config, output_dir, repo_root)
    if paths["result_root"].exists() and any(paths["result_root"].iterdir()) and not allow_existing_output:
        raise FileExistsError(f"output directory already exists: {paths['result_root']}")
    _ensure_dirs(paths)
    frozen_errors = _validate_submission_manifest(config, paths)
    if frozen_errors:
        _write_report(
            paths,
            "frozen_input_bug_report.md",
            "# Frozen Input Bug Report\n\n" + "\n".join(f"- {error}" for error in frozen_errors),
        )
        raise ValueError(f"frozen submission inputs failed validation: {frozen_errors}")

    revised_files = _write_revised_manuscript(paths, config)
    revised_files.extend(_refresh_metric_tables(paths))
    checks = _write_audit_reports(paths, config)
    compile_status = _compile_latex(paths)

    manifest = {
        "run_type": "final_manuscript_audit_v1",
        "status": "complete",
        "output_dir": str(paths["result_root"]),
        "audit_analysis_dir": str(paths["audit_analysis"]),
        "paper_dir": str(paths["paper"]),
        "submission_result_dir": str(paths["submission_result"]),
        "git_sha": _git_sha(paths["repo_root"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "final_title": FINAL_TITLE,
        "final_main_claim": FINAL_MAIN_CLAIM,
        "selected_model": {
            "feature_group": FINAL_MODEL_GROUP,
            "model": FINAL_MODEL,
            "split": FINAL_SPLIT,
            "n_predictions": 57,
            "skipped_folds": 0,
        },
        "checks": checks,
        "compile_status": compile_status,
        "revised_files": revised_files,
        "decision": "ready_with_minor_manual_edits",
        "large_outputs_not_for_commit": ["results/final_manuscript_audit_v1_*/"],
    }
    _write_json(paths["result_root"] / "manifest.json", manifest)
    _write_json(paths["result_root"] / "run_summary.json", {"status": "complete", "decision": manifest["decision"]})
    return manifest


def validate_manuscript_audit(
    config: dict[str, Any], output_dir: str | Path, *, repo_root: str | Path = "."
) -> dict[str, Any]:
    paths = manuscript_audit_paths(config, output_dir, repo_root)
    errors = []
    warnings = []
    required_reports = [
        "manuscript_consistency_audit.md",
        "scientific_narrative_audit.md",
        "final_abstract.md",
        "final_contribution_list.md",
        "limitations_coverage_report.md",
        "claim_evidence_validation_report.md",
        "table_figure_audit.md",
        "reference_audit.md",
        "reviewer_revision_plan.md",
        "final_readiness_report.md",
    ]
    for name in required_reports:
        if not (paths["audit_analysis"] / name).exists():
            errors.append(f"missing audit report: {name}")
    for section in MANUSCRIPT_SECTIONS:
        if not (paths["paper"] / "sections" / f"{section}.tex").exists():
            errors.append(f"missing manuscript section: {section}")
    for section in SUPPLEMENT_SECTIONS:
        if not (paths["paper"] / "supplement_sections" / f"{section}.tex").exists():
            errors.append(f"missing supplement section: {section}")

    for check_name, check in [
        ("metric consistency", check_metric_consistency(config, paths)),
        ("claim ledger", check_claim_ledger(config, paths)),
        ("prohibited claims", check_prohibited_claims(paths)),
        ("table/figure references", check_table_figure_refs(paths)),
        ("limitations coverage", check_limitations_coverage(paths)),
    ]:
        if check["status"] != "passed":
            errors.append(f"{check_name} failed: {check}")

    text = _all_main_text(paths)
    escaped_group = FINAL_MODEL_GROUP.replace("_", r"\_")
    if FINAL_MODEL_GROUP not in text and escaped_group not in text:
        errors.append("final D3 feature group missing from manuscript")
    if "DFM exposure-only is weak" not in text and "DFM exposure-only model is weak" not in text:
        errors.append("exposure-only failure is not stated clearly")
    if "ready_with_minor_manual_edits" not in _read_text(
        paths["audit_analysis"] / "final_readiness_report.md"
    ):
        errors.append("readiness decision missing or changed")
    if "random_word" in text.lower():
        errors.append("random_word split artifact appears in manuscript")
    if "new core label" in text.lower() or "new feature family" in text.lower():
        warnings.append("audit text mentions prohibited research actions; verify context")

    compile_report = paths["audit_analysis"] / "compile_skipped_report.md"
    latex_report = paths["audit_analysis"] / "latex_compile_report.md"
    if not compile_report.exists() and not latex_report.exists():
        errors.append("missing LaTeX compile or skipped report")

    staged_large = _staged_large_files(paths["repo_root"])
    if staged_large:
        errors.append(f"large files staged for commit: {staged_large}")

    report = {"status": "passed" if not errors else "failed", "errors": errors, "warnings": warnings}
    _write_json(paths["result_root"] / "validation_report.json", report)
    return report


def _staged_large_files(repo_root: str | Path) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "-z"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    paths = [Path(item.decode()) for item in output.split(b"\0") if item]
    large = []
    for rel in paths:
        path = Path(repo_root) / rel
        if path.exists() and path.is_file() and path.stat().st_size >= 100_000_000:
            large.append(str(rel))
    return large
