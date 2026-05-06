"""Console entrypoints for the CopCo research pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import get_nested, load_config, timestamped_output_dir
from .autoresearch import build_paper_package, run_autoresearch, validate_autoresearch
from .features import build_feature_tables
from .label_release import build_label_release, freeze_prepared_dataset, validate_label_release
from .lm_features import run_lm_features
from .manuscript_audit import run_manuscript_audit, validate_manuscript_audit
from .mixed_effects import fit_mixed_effects
from .modeling import run_models
from .phase4_confirmatory import run_phase4_confirmatory, validate_phase4_confirmatory
from .release import (
    build_modeling_tables,
    finalize_feature_release,
    run_analysis_package,
    run_embedding_features,
    run_parser_features,
    validate_feature_release,
    write_release_features,
)
from .research_exploration import run_research_exploration, validate_research_exploration
from .slurm import launcher_command
from .submission import build_submission_package, validate_submission_package
from .validation import validate_run


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, default=str))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/copco_dyslexia_full.yaml")
    parser.add_argument("--repo-root", default=".")


def build_features_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CopCo feature tables.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir")
    parser.add_argument("--sample-participants", type=int)
    parser.add_argument("--sample-speeches", type=int)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config, repo_root=args.repo_root)
    sample_participants = args.sample_participants
    if sample_participants is None:
        sample_participants = get_nested(config, "sample.participants")
    sample_speeches = args.sample_speeches
    if sample_speeches is None:
        sample_speeches = get_nested(config, "sample.speeches")
    if bool(get_nested(config, "feature_release.require_full_corpus", False)) and (
        sample_participants is not None or sample_speeches is not None
    ):
        parser.error("feature release config requires full corpus mode; sample limits are forbidden")

    if args.print_slurm_command:
        command = f"copco-build-features --config {args.config}"
        if args.output_dir:
            command += f" --output-dir {args.output_dir}"
        if sample_participants:
            command += f" --sample-participants {int(sample_participants)}"
        if sample_speeches:
            command += f" --sample-speeches {int(sample_speeches)}"
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_output_dir(
        config, repo_root=args.repo_root
    )
    manifest = build_feature_tables(
        config,
        output_dir,
        repo_root=args.repo_root,
        sample_participants=None if sample_participants is None else int(sample_participants),
        sample_speeches=None if sample_speeches is None else int(sample_speeches),
    )
    _print(manifest)
    return 0


def run_lm_features_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paragraph-sharded LM feature generation.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-label")
    parser.add_argument("--feature-kind", default="surprisal", choices=["surprisal", "embeddings"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--max-word-tokens", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--real-run", action="store_true", help="Override config language_models.dry_run.")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.real_run:
        parser.error("--dry-run and --real-run are mutually exclusive")

    if args.print_slurm_command:
        command = (
            "copco-run-lm-features "
            f"--config {args.config} --output-dir {args.output_dir} "
            f"--feature-kind {args.feature_kind} --shard-index {args.shard_index} "
            f"--shard-count {args.shard_count} --require-gpu"
        )
        if args.model_id:
            command += f" --model-id {args.model_id}"
        if args.model_label:
            command += f" --model-label {args.model_label}"
        if args.limit_items:
            command += f" --limit-items {args.limit_items}"
        if args.max_word_tokens:
            command += f" --max-word-tokens {args.max_word_tokens}"
        if args.real_run:
            command += " --real-run"
        print(launcher_command(command, repo_root=args.repo_root, mode="gpu"))
        return 0

    config = load_config(args.config, repo_root=args.repo_root)
    manifest = run_lm_features(
        config,
        args.output_dir,
        model_id=args.model_id,
        model_label=args.model_label,
        feature_kind=args.feature_kind,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        limit_items=args.limit_items,
        max_word_tokens=args.max_word_tokens,
        dry_run=args.dry_run,
        force_real_run=args.real_run,
        require_gpu=args.require_gpu,
    )
    _print(manifest)
    return 0


def run_models_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CopCo Models A-F.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)

    if args.print_slurm_command:
        command = f"copco-run-models --config {args.config} --output-dir {args.output_dir}"
        if args.seed is not None:
            command += f" --seed {args.seed}"
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0

    config = load_config(args.config, repo_root=args.repo_root)
    manifest = run_models(config, args.output_dir, seed=args.seed)
    _print(manifest)
    return 0


def fit_mixed_effects_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit CopCo mixed-effects models.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)

    if args.print_slurm_command:
        command = f"copco-fit-mixed-effects --config {args.config} --output-dir {args.output_dir}"
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0

    config = load_config(args.config, repo_root=args.repo_root)
    manifest = fit_mixed_effects(config, args.output_dir)
    _print(manifest)
    return 0


def validate_run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CopCo run artifacts.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = validate_run(args.output_dir)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def write_release_features_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write feature-release tables from base CopCo tables.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-write-release-features --config {args.config} --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(write_release_features(config, args.output_dir, repo_root=args.repo_root))
    return 0


def run_parser_features_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build parser and morphosyntactic release features.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-run-parser-features --config {args.config} --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(run_parser_features(config, args.output_dir))
    return 0


def run_embeddings_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build sentence and paragraph embedding feature files.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-label")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-run-embeddings --config {args.config} --output-dir {args.output_dir}"
    if args.model_id:
        command += f" --model-id {args.model_id}"
    if args.model_label:
        command += f" --model-label {args.model_label}"
    if args.baseline:
        command += " --baseline"
    if args.skip_baseline:
        command += " --skip-baseline"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="gpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(
        run_embedding_features(
            config,
            args.output_dir,
            model_id=args.model_id,
            model_label=args.model_label,
            run_baseline=args.baseline,
            skip_baseline=args.skip_baseline,
        )
    )
    return 0


def build_modeling_tables_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join release feature families into modeling tables.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-build-modeling-tables --config {args.config} --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(build_modeling_tables(config, args.output_dir))
    return 0


def run_analysis_package_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run feature-release analysis reports.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-run-analysis-package --config {args.config} --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(run_analysis_package(config, args.output_dir, repo_root=args.repo_root))
    return 0


def finalize_feature_release_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write final feature-release report.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-finalize-feature-release --config {args.config} --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(finalize_feature_release(config, args.output_dir, repo_root=args.repo_root))
    return 0


def validate_feature_release_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate feature-release outputs.")
    _add_config_arg(parser)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_feature_release(config, args.output_dir)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def build_label_release_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Label Release v1.1 and freeze prepared tables.")
    parser.add_argument("--config", default="configs/label_release_v1_1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-build-label-release --config {args.config}"
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(build_label_release(config, args.output_dir, repo_root=args.repo_root))
    return 0


def validate_label_release_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Label Release v1.1 outputs.")
    parser.add_argument("--config", default="configs/label_release_v1_1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_label_release(args.output_dir, config=config, repo_root=args.repo_root)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def freeze_prepared_dataset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild prepared-dataset tables from label release files.")
    parser.add_argument("--config", default="configs/label_release_v1_1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-freeze-prepared-dataset --config {args.config} --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(freeze_prepared_dataset(config, args.output_dir, repo_root=args.repo_root))
    return 0


def run_research_exploration_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 controlled research exploration.")
    parser.add_argument("--config", default="configs/research_exploration_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-run-research-exploration --config {args.config}"
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(run_research_exploration(config, args.output_dir, repo_root=args.repo_root))
    return 0


def validate_research_exploration_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 3 research exploration outputs.")
    parser.add_argument("--config", default="configs/research_exploration_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_research_exploration(config, args.output_dir, repo_root=args.repo_root)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def run_phase4_confirmatory_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 confirmatory sensitivity analysis.")
    parser.add_argument("--config", default="configs/phase4_confirmatory_sensitivity_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-run-phase4-confirmatory --config {args.config}"
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(run_phase4_confirmatory(config, args.output_dir, repo_root=args.repo_root))
    return 0


def validate_phase4_confirmatory_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 4 confirmatory analysis outputs.")
    parser.add_argument("--config", default="configs/phase4_confirmatory_sensitivity_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_phase4_confirmatory(config, args.output_dir, repo_root=args.repo_root)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def run_autoresearch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AutoResearch v1 publication decision loop.")
    parser.add_argument("--config", default="configs/autoresearch_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-slurm-command", action="store_true")
    parser.add_argument("--skip-heavy-bootstrap", action="store_true")
    parser.add_argument("--skip-heavy-permutation", action="store_true")
    parser.add_argument("--fail-on-decision-gate-failure", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-run-autoresearch --config {args.config}"
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    if args.dry_run:
        command += " --dry-run"
    if args.skip_heavy_bootstrap:
        command += " --skip-heavy-bootstrap"
    if args.skip_heavy_permutation:
        command += " --skip-heavy-permutation"
    if args.fail_on_decision_gate_failure:
        command += " --fail-on-decision-gate-failure"
    if args.allow_existing_output:
        command += " --allow-existing-output"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(
        run_autoresearch(
            config,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            dry_run=args.dry_run,
            skip_heavy_bootstrap=args.skip_heavy_bootstrap,
            skip_heavy_permutation=args.skip_heavy_permutation,
            fail_on_decision_gate_failure=args.fail_on_decision_gate_failure,
            allow_existing_output=args.allow_existing_output,
        )
    )
    return 0


def validate_autoresearch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AutoResearch v1 outputs.")
    parser.add_argument("--config", default="configs/autoresearch_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_autoresearch(config, args.output_dir, repo_root=args.repo_root)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def build_paper_package_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build AutoResearch paper-ready package files.")
    parser.add_argument("--config", default="configs/autoresearch_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    _print(build_paper_package(config, args.output_dir, repo_root=args.repo_root))
    return 0


def validate_paper_package_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AutoResearch paper-ready package files.")
    parser.add_argument("--config", default="configs/autoresearch_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_autoresearch(config, args.output_dir, repo_root=args.repo_root)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def build_submission_package_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SubmissionSprint v1 manuscript package.")
    parser.add_argument("--config", default="configs/submission_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-build-submission-package --config {args.config}"
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    if args.allow_existing_output:
        command += " --allow-existing-output"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(
        build_submission_package(
            config,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            allow_existing_output=args.allow_existing_output,
        )
    )
    return 0


def validate_submission_package_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SubmissionSprint v1 manuscript package.")
    parser.add_argument("--config", default="configs/submission_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_submission_package(config, args.output_dir, repo_root=args.repo_root)
    _print(report)
    return 0 if report["status"] == "passed" else 1


def run_manuscript_audit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Final Manuscript Audit v1.")
    parser.add_argument("--config", default="configs/final_manuscript_audit_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)
    command = f"copco-run-manuscript-audit --config {args.config}"
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    if args.allow_existing_output:
        command += " --allow-existing-output"
    if args.print_slurm_command:
        print(launcher_command(command, repo_root=args.repo_root, mode="cpu"))
        return 0
    config = load_config(args.config, repo_root=args.repo_root)
    _print(
        run_manuscript_audit(
            config,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            allow_existing_output=args.allow_existing_output,
        )
    )
    return 0


def validate_manuscript_audit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Final Manuscript Audit v1 outputs.")
    parser.add_argument("--config", default="configs/final_manuscript_audit_v1.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    report = validate_manuscript_audit(config, args.output_dir, repo_root=args.repo_root)
    _print(report)
    return 0 if report["status"] == "passed" else 1
