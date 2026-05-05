"""Console entrypoints for the CopCo research pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import get_nested, load_config, timestamped_output_dir
from .features import build_feature_tables
from .lm_features import run_lm_features
from .mixed_effects import fit_mixed_effects
from .modeling import run_models
from .release import (
    build_modeling_tables,
    finalize_feature_release,
    run_analysis_package,
    run_embedding_features,
    run_parser_features,
    validate_feature_release,
    write_release_features,
)
from .slurm import launcher_command
from .validation import validate_run


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


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
