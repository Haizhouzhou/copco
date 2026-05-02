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
    parser.add_argument("--feature-kind", default="surprisal", choices=["surprisal", "embeddings"])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--print-slurm-command", action="store_true")
    args = parser.parse_args(argv)

    if args.print_slurm_command:
        command = (
            "copco-run-lm-features "
            f"--config {args.config} --output-dir {args.output_dir} "
            f"--feature-kind {args.feature_kind} --shard-index {args.shard_index} "
            f"--shard-count {args.shard_count} --require-gpu"
        )
        if args.model_id:
            command += f" --model-id {args.model_id}"
        if args.limit_items:
            command += f" --limit-items {args.limit_items}"
        print(launcher_command(command, repo_root=args.repo_root, mode="gpu"))
        return 0

    config = load_config(args.config, repo_root=args.repo_root)
    manifest = run_lm_features(
        config,
        args.output_dir,
        model_id=args.model_id,
        feature_kind=args.feature_kind,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        limit_items=args.limit_items,
        dry_run=args.dry_run,
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
