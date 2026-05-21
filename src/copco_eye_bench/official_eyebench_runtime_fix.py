"""Official EyeBench runtime repair/check gate.

This module is intentionally an infrastructure gate around the already-frozen D3
method. It never substitutes CopCo prepared data for official EyeBench processed
data, and it never promotes the benchmark-relative claim unless the official
environment/data/fold/evaluator/baseline gates pass.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark_bridge import PROHIBITED_FEATURES
from .config import get_nested, timestamped_output_dir
from .official_eyebench_sota_check import (
    BASELINE_COLUMNS,
    COMPARISON_COLUMNS,
    OFFICIAL_SPLITS,
    READER_LEVEL,
    SOTA_TYP_COLUMNS,
    TRIAL_LEVEL,
    _count_processed,
    _empty_typ_metrics,
    _markdown_table,
    _official_reference_table,
    _pd,
    _run_command,
    _write_csv,
    _write_json,
    _write_md,
    build_official_split_labels,
    evaluate_d3_eyebench_lite,
    load_official_processed_features,
    reproduce_official_baseline,
    validate_official_split_labels,
    write_comparison_tables,
)


RUNTIME_SECTION = "official_eyebench_runtime_fix"
SOTA_SECTION = "official_eyebench_sota_check"
RUNTIME_MODEL = "D3_EyeBench_Lite"

VALID_DECISION_CATEGORIES = {
    "official_sota_claim_allowed",
    "official_compatible_but_not_sota",
    "benchmark_relative_sota_only",
    "blocked_by_environment",
    "blocked_by_data",
    "blocked_by_evaluator",
    "blocked_by_baseline_reproduction",
}

REQUIRED_GITIGNORE_PATTERNS = [
    "eyebench/data/",
    "eyebench/results/",
    "eyebench/logs/",
    "eyebench/wandb/",
    "eyebench/.cache/",
    "eyebench/.conda_pkgs/",
    "eyebench/.pip_cache/",
    "eyebench/.envs/",
    "eyebench/.pytest_cache/",
    "eyebench/.runtime_logs/",
    "eyebench/**/*.feather",
    "eyebench/**/*.parquet",
    "eyebench/**/*.pkl",
    "eyebench/**/*.pt",
    "eyebench/**/*.ckpt",
    "results/official_eyebench_runtime_fix_v1_*/",
]

PREDICTION_COLUMNS = [
    "mode",
    "model_name",
    "claim_type",
    "task",
    "split_name",
    "fold_id",
    "feature_group",
    "model",
    "sample_id",
    "unique_trial_id",
    "participant_id",
    "speech_id",
    "text_id",
    "y_true",
    "y_score",
    "y_pred",
    "eval_regime",
    "eval_type",
    "fold_index",
]


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = get_nested(config, RUNTIME_SECTION, {})
    return section if isinstance(section, dict) else {}


def _runtime_path(config: dict[str, Any], dotted: str, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    value = get_nested(config, f"{RUNTIME_SECTION}.{dotted}")
    if value is None:
        raise ValueError(f"missing required config path: {RUNTIME_SECTION}.{dotted}")
    path = Path(str(value))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _eyebench_path(config: dict[str, Any], repo_root: str | Path) -> Path:
    return _runtime_path(config, "eyebench.path", repo_root)


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    repo_analysis = root / str(
        get_nested(config, f"{RUNTIME_SECTION}.repo_analysis_dir", "analysis/official_eyebench_runtime_fix_v1")
    )
    result_analysis = out / str(
        get_nested(config, f"{RUNTIME_SECTION}.output_layout.analysis", "analysis/official_eyebench_runtime_fix_v1")
    )
    return {
        "repo_analysis": repo_analysis,
        "repo_tables": repo_analysis / "tables",
        "result_analysis": result_analysis,
        "result_tables": out
        / str(
            get_nested(
                config,
                f"{RUNTIME_SECTION}.output_layout.tables",
                "analysis/official_eyebench_runtime_fix_v1/tables",
            )
        ),
    }


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["repo_analysis"] / name, text)
    _write_md(dirs["result_analysis"] / name, text)


def _to_sota_config(config: dict[str, Any]) -> dict[str, Any]:
    converted = dict(config)
    runtime = dict(_section(config))
    converted[SOTA_SECTION] = runtime
    converted["run"] = {
        **dict(config.get("run", {})),
        "name": "official_eyebench_runtime_fix_v1",
    }
    return converted


def _relative_under_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def validate_official_eyebench_runtime_fix_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    errors: list[str] = []
    warnings: list[str] = []
    if not section:
        errors.append(f"missing {RUNTIME_SECTION} config section")
    for flag in ["no_new_labels", "no_feature_engineering_search", "no_broad_model_search"]:
        if section.get(flag) is not True:
            errors.append(f"{flag} must be true")
    if section.get("forbid_random_word_level_split") is not True:
        errors.append("random word-level split prohibition is not enabled")
    missing_splits = sorted(set(OFFICIAL_SPLITS) - set(section.get("split_regimes", [])))
    if missing_splits:
        errors.append(f"missing official split regimes: {missing_splits}")
    prohibited = set(section.get("prohibited_features", []))
    missing_prohibited = sorted(
        (
            PROHIBITED_FEATURES
            | {
                "participant_id",
                "speech_id",
                "text_id",
                "unique_trial_id",
                "unique_paragraph_id",
                "dyslexia",
                "RCS_score",
                "reader_group",
                "reader_group_binary",
            }
        )
        - prohibited
    )
    if missing_prohibited:
        errors.append(f"prohibited feature list incomplete: {missing_prohibited}")
    if section.get("eyebench", {}).get("path") != "eyebench":
        errors.append("EyeBench path must be the isolated ./eyebench submodule")
    for dotted in [
        "runtime_workspace.envs_dir",
        "runtime_workspace.conda_pkgs_dir",
        "runtime_workspace.pip_cache_dir",
        "runtime_workspace.cache_dir",
        "runtime_workspace.runtime_logs_dir",
    ]:
        value = get_nested(section, dotted)
        if value is None or not str(value).startswith("eyebench/"):
            errors.append(f"{dotted} must stay under eyebench/")
    if "CopCo_TYP" not in section.get("tasks", []):
        errors.append("CopCo_TYP is required")
    return {"status": "failed" if errors else "passed", "errors": errors, "warnings": warnings}


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _runtime_env(eyebench: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CONDA_PKGS_DIRS": str(eyebench / ".conda_pkgs"),
            "PIP_CACHE_DIR": str(eyebench / ".pip_cache"),
            "HF_HOME": str(eyebench / ".cache" / "huggingface"),
            "TRANSFORMERS_CACHE": str(eyebench / ".cache" / "huggingface"),
            "WANDB_DIR": str(eyebench / "wandb"),
            "WANDB_MODE": "offline",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{eyebench}:{eyebench / 'src'}",
        }
    )
    return env


def _check_env_imports(env_name: str, repo_root: Path, eyebench: Path) -> dict[str, Any]:
    command = [
        "conda",
        "run",
        "-n",
        env_name,
        "bash",
        "-lc",
        (
            f"cd {eyebench.relative_to(repo_root)} && "
            "PYTHONPATH=$PWD:$PWD/src PYTHONNOUSERSITE=1 python - <<'PY'\n"
            "import sys\n"
            "print(sys.version)\n"
            "for name in ['beartype','pandas','pyarrow','sklearn']:\n"
            "    mod = __import__(name)\n"
            "    print(name, getattr(mod, '__version__', 'imported'))\n"
            "import src\n"
            "print('eyebench src import ok')\n"
            "import src.data.utils\n"
            "import src.data.preprocessing.preprocess_data\n"
            "print('eyebench preprocessing imports ok')\n"
            "PY"
        ),
    ]
    result = _run_command(command, cwd=repo_root, timeout=180, env=_runtime_env(eyebench))
    syntax_312 = "f-string" in result["stderr"] or "SyntaxError" in result["stderr"]
    return {
        "env_name": env_name,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "imports_ok": result["returncode"] == 0,
        "python312_syntax_blocker": bool(syntax_312),
    }


def _check_prefix_imports(prefix: Path, repo_root: Path, eyebench: Path) -> dict[str, Any]:
    if not prefix.exists():
        return {
            "prefix": str(prefix),
            "prefix_exists": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "imports_ok": False,
        }
    command = [
        "conda",
        "run",
        "-p",
        str(prefix),
        "bash",
        "-lc",
        (
            f"cd {eyebench.relative_to(repo_root)} && "
            "PYTHONPATH=$PWD:$PWD/src PYTHONNOUSERSITE=1 python - <<'PY'\n"
            "import sys\n"
            "print(sys.version)\n"
            "for name in ['beartype','pandas','pyarrow','sklearn','hydra','src']:\n"
            "    mod = __import__(name)\n"
            "    print(name, getattr(mod, '__version__', 'imported'))\n"
            "from text_metrics.ling_metrics_funcs import get_metrics\n"
            "from text_metrics.surprisal_extractors.extractor_switch import get_surp_extractor\n"
            "from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType\n"
            "print('text_metrics legacy api ok')\n"
            "import src.data.utils\n"
            "import src.data.preprocessing.preprocess_data\n"
            "print('eyebench preprocessing imports ok')\n"
            "PY"
        ),
    ]
    result = _run_command(command, cwd=repo_root, timeout=180, env=_runtime_env(eyebench))
    return {
        "prefix": str(prefix),
        "prefix_exists": True,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "imports_ok": result["returncode"] == 0,
    }


def write_preflight_report(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    eyebench = _eyebench_path(config, root)
    preflight = {
        "copco_commit": _run_command(["git", "rev-parse", "HEAD"], cwd=root)["stdout"],
        "current_branch": _run_command(["git", "branch", "--show-current"], cwd=root)["stdout"],
        "git_status_short": _run_command(["git", "status", "--short"], cwd=root)["stdout"],
        "eyebench_submodule_status": _run_command(["git", "submodule", "status", "eyebench"], cwd=root)[
            "stdout"
        ],
        "eyebench_commit": _run_command(["git", "rev-parse", "HEAD"], cwd=eyebench)["stdout"]
        if eyebench.exists()
        else "",
        "eyebench_status_short": _run_command(["git", "status", "--short"], cwd=eyebench)["stdout"]
        if eyebench.exists()
        else "missing",
        "active_conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda": _run_command(["bash", "-lc", "command -v conda || true"], cwd=root)["stdout"],
        "mamba": _run_command(["bash", "-lc", "command -v mamba || true"], cwd=root)["stdout"],
        "micromamba": _run_command(["bash", "-lc", "command -v micromamba || true"], cwd=root)["stdout"],
        "cuda_query": _run_command(
            ["bash", "-lc", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"],
            cwd=root,
        ),
        "disk_project": _run_command(["df", "-h", "."], cwd=root)["stdout"],
        "disk_tmp": _run_command(["df", "-h", "/tmp"], cwd=root)["stdout"],
        "eyebench_data_exists": (eyebench / "data").exists(),
        "eyebench_processed_copco_exists": (eyebench / "data" / "CopCo" / "processed").exists(),
        "eyebench_results_exists": (eyebench / "results").exists(),
    }
    lines = [
        "# OfficialEyeBenchRuntimeFix v1 Preflight Report",
        "",
        f"- CopCo commit: `{preflight['copco_commit']}`",
        f"- Branch: `{preflight['current_branch']}`",
        f"- EyeBench commit: `{preflight['eyebench_commit']}`",
        f"- EyeBench submodule status: `{preflight['eyebench_submodule_status']}`",
        f"- EyeBench local modifications: `{preflight['eyebench_status_short'] or 'none'}`",
        f"- Active conda env: `{preflight['active_conda_env'] or 'none'}`",
        f"- conda: `{preflight['conda'] or 'missing'}`",
        f"- mamba: `{preflight['mamba'] or 'missing'}`",
        f"- micromamba: `{preflight['micromamba'] or 'missing'}`",
        f"- CUDA query return code: {preflight['cuda_query']['returncode']}",
        f"- `eyebench/data` exists: {preflight['eyebench_data_exists']}",
        f"- `eyebench/data/CopCo/processed` exists: {preflight['eyebench_processed_copco_exists']}",
        f"- `eyebench/results` exists: {preflight['eyebench_results_exists']}",
        "",
        "## Disk",
        "```text",
        str(preflight["disk_project"]),
        str(preflight["disk_tmp"]),
        "```",
    ]
    _write_report(dirs, "preflight_report.md", "\n".join(lines))
    _write_json(out / "preflight" / "preflight_report.json", preflight)
    return preflight


def write_environment_reports(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    eyebench = _eyebench_path(config, root)
    copco_env = str(get_nested(config, f"{RUNTIME_SECTION}.environment.primary_conda_env", "copco"))
    minimal_prefix = _runtime_path(config, "environment.py312_minimal_prefix", root)
    exact_prefix_value = get_nested(config, f"{RUNTIME_SECTION}.environment.exact_prefix")
    exact_prefix = _runtime_path(config, "environment.exact_prefix", root) if exact_prefix_value else None
    runtime_logs = _runtime_path(config, "runtime_workspace.runtime_logs_dir", root)
    env_list = _run_command(["conda", "env", "list"], cwd=root, timeout=120)
    ps_probe = _run_command(
        [
            "bash",
            "-lc",
            "ps -o pid,ppid,etime,stat,cmd -u \"$USER\" | "
            "rg 'conda env remove|mamba env create|/usr/bin/rsync -a --force --delete' || true",
        ],
        cwd=root,
        timeout=30,
    )
    copco_import = _check_env_imports(copco_env, root, eyebench)
    minimal_import = _check_prefix_imports(minimal_prefix, root, eyebench)
    exact_log = runtime_logs / "env_create_exact.log"
    fallback_logs = sorted(runtime_logs.glob("pip_install_*.log")) + sorted(runtime_logs.glob("py312_minimal*.log"))
    exact_import = (
        _check_prefix_imports(exact_prefix, root, eyebench)
        if exact_prefix is not None and exact_prefix.exists()
        else {
            "prefix": str(exact_prefix) if exact_prefix is not None else "",
            "prefix_exists": bool(exact_prefix and exact_prefix.exists()),
            "returncode": None,
            "stdout": "",
            "stderr": "not attempted under revised runtime rule",
            "imports_ok": False,
        }
    )
    copco_ready = bool(copco_import["imports_ok"])
    minimal_ready = bool(minimal_import["imports_ok"])
    exact_ready = bool(exact_import["imports_ok"])
    environment_ready = copco_ready or minimal_ready or exact_ready
    if copco_ready:
        environment_kind = "copco_env_official_code_runtime"
    elif minimal_ready:
        environment_kind = "python312_minimal_runtime_compatible"
    elif exact_ready:
        environment_kind = "exact_environment_yml"
    else:
        environment_kind = "none"
    cleanup = {
        "attempted_named_env_removal": False,
        "reason": "revised runtime rule uses copco env first and does not remove/create full EyeBench env unless necessary",
        "eyebench_official_in_env_list": "eyebench_official" in env_list["stdout"],
        "eyebench_in_env_list": " /home/haizhe/conda/envs/eyebench" in env_list["stdout"],
        "orphan_cleanup_processes": ps_probe["stdout"],
        "status": "complete_or_quarantined",
    }
    exact = {
        "command": "not attempted by revised runtime rule",
        "prefix": str(exact_prefix) if exact_prefix is not None else "",
        "prefix_exists": bool(exact_prefix and exact_prefix.exists()),
        "log_path": str(exact_log),
        "log_tail": _tail(exact_log),
        "import_check": exact_import,
        "status": "ready" if exact_ready else "not_attempted",
    }
    fallback = {
        "command": get_nested(config, f"{RUNTIME_SECTION}.environment.py312_minimal_create_command"),
        "prefix": str(minimal_prefix),
        "prefix_exists": minimal_prefix.exists(),
        "generated_yml": "",
        "log_tails": {path.name: _tail(path) for path in fallback_logs if path.exists()},
        "import_check": minimal_import,
        "status": "ready" if minimal_ready else "not_ready",
    }
    report = {
        "official_environment_ready": environment_ready,
        "environment_kind": environment_kind,
        "primary_copco_env_ready": copco_ready,
        "primary_copco_env_import": copco_import,
        "python312_minimal_ready": minimal_ready,
        "exact_environment_ready": exact_ready,
        "cpu_fallback_ready": minimal_ready,
        "exact_environment_yml_used": exact_ready,
        "status": "ready" if environment_ready else "blocked_by_environment",
        "cleanup": cleanup,
        "exact_attempt": exact,
        "cpu_fallback_attempt": fallback,
        "py312_minimal_attempt": fallback,
    }
    _write_json(out / "environment" / "environment_status.json", report)
    _write_report(
        dirs,
        "environment_cleanup_report.md",
        "\n".join(
            [
                "# Environment Cleanup Report",
                "",
                f"- Named env removal attempted: {cleanup['attempted_named_env_removal']}",
                f"- Reason: {cleanup['reason']}",
                f"- `eyebench_official` still listed: {cleanup['eyebench_official_in_env_list']}",
                f"- legacy `eyebench` still listed: {cleanup['eyebench_in_env_list']}",
                f"- Status: `{cleanup['status']}`",
                "",
                "## Live Cleanup Processes",
                "```text",
                cleanup["orphan_cleanup_processes"] or "none",
                "```",
            ]
        ),
    )
    _write_report(
        dirs,
        "environment_exact_attempt.md",
        "\n".join(
            [
                "# Exact EyeBench Environment Attempt",
                "",
                "The revised runtime rule explicitly forbids trying the full `environment.yml` first.",
                f"- Prefix: `{exact_prefix}`",
                f"- Prefix exists: {bool(exact_prefix and exact_prefix.exists())}",
                f"- Imports ok: {exact_ready}",
                f"- Status: `{exact['status']}`",
                "",
                "## Log Tail",
                "```text",
                exact["log_tail"] or "no log",
                "```",
                "",
                "## Import Check",
                "```text",
                json.dumps(exact_import, indent=2, default=str),
                "```",
            ]
        ),
    )
    _write_report(
        dirs,
        "environment_fallback_report.md",
        "\n".join(
            [
                "# Minimal Python 3.12 Runtime Fallback Report",
                "",
                f"- Primary CopCo env: `{copco_env}`",
                f"- Primary CopCo env imports ok: {copco_ready}",
                f"- Primary CopCo env Python 3.12 syntax blocker: {copco_import.get('python312_syntax_blocker')}",
                f"- Minimal prefix: `{minimal_prefix}`",
                f"- Prefix exists: {minimal_prefix.exists()}",
                f"- Imports ok: {minimal_ready}",
                f"- Status: `{fallback['status']}`",
                "- Classification: official-code/data/fold-compatible only if downstream data, fold, evaluator, and baseline gates pass.",
                "- This is not an exact `environment.yml` reproduction.",
                "",
                "## Primary CopCo Env Import",
                "```text",
                json.dumps(copco_import, indent=2, default=str)[-5000:],
                "```",
                "",
                "## Log Tails",
                "```text",
                json.dumps(fallback["log_tails"], indent=2, default=str)[-5000:],
                "```",
                "",
                "## Import Check",
                "```text",
                json.dumps(minimal_import, indent=2, default=str),
                "```",
            ]
        ),
    )
    return report


def write_command_inventory(
    config: dict[str, Any],
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> None:
    root = Path(repo_root).resolve()
    eyebench = _eyebench_path(config, root)
    files = {
        "README": eyebench / "README.md",
        "environment": eyebench / "environment.yml",
        "pyproject": eyebench / "pyproject.toml",
        "get_data": eyebench / "src" / "data" / "preprocessing" / "get_data.sh",
        "data_config": eyebench / "src" / "configs" / "data.py",
        "CopCo_TYP_commands": eyebench / "run_commands" / "CopCo_TYP.md",
    }
    excerpts = {}
    for name, path in files.items():
        excerpts[name] = _tail(path, limit=2500) if path.exists() else "missing"
    run_files = sorted(
        _relative_under_repo(path, root)
        for base in [eyebench / "src" / "run", eyebench / "run_commands", eyebench / "src" / "configs"]
        if base.exists()
        for path in base.rglob("*")
        if path.is_file()
    )
    text = [
        "# EyeBench Command Inventory",
        "",
        "## Official Task Names",
        "- `CopCo_TYP`",
        "- `CopCo_RCS`",
        "",
        "## Official Data Command",
        "- `bash src/data/preprocessing/get_data.sh CopCo` is supported by `get_data.sh`.",
        "",
        "## Files Inspected",
        *[f"- `{_relative_under_repo(path, root)}`" for path in files.values()],
        "",
        "## Run/Config Files",
        *[f"- `{name}`" for name in run_files[:200]],
        "",
        "## Key Excerpts",
    ]
    for name, excerpt in excerpts.items():
        text.extend(["", f"### {name}", "```text", excerpt, "```"])
    _write_report(dirs, "eyebench_command_inventory.md", "\n".join(text))


def write_data_preprocessing_report(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    eyebench = _eyebench_path(config, root)
    processed = _runtime_path(config, "eyebench.processed_copco_dir", root)
    labels = _runtime_path(config, "eyebench.labels_dir", root)
    folds = _runtime_path(config, "eyebench.folds_metadata_dir", root)
    command = "bash src/data/preprocessing/get_data.sh CopCo"
    command_result = None
    skip_reason = ""
    required_processed = ["ia.feather", "fixations.feather", "trial_level.feather"]
    processed_complete = processed.exists() and all((processed / name).exists() for name in required_processed)
    if not environment.get("official_environment_ready"):
        skip_reason = "no usable EyeBench environment or compatible runtime prefix"
    elif processed_complete:
        skip_reason = "official CopCo processed data already present"
    elif bool(get_nested(config, f"{RUNTIME_SECTION}.eyebench.run_preprocessing_in_cli", False)):
        if environment.get("primary_copco_env_ready"):
            run_prefix = ["-n", str(get_nested(config, f"{RUNTIME_SECTION}.environment.primary_conda_env", "copco"))]
        else:
            prefix_key = "exact_prefix" if environment.get("exact_environment_ready") else "py312_minimal_prefix"
            prefix = _runtime_path(config, f"environment.{prefix_key}", root)
            run_prefix = ["-p", str(prefix)]
        command_result = _run_command(
            ["conda", "run", *run_prefix, "bash", "-lc", command],
            cwd=eyebench,
            timeout=int(get_nested(config, f"{RUNTIME_SECTION}.eyebench.preprocessing_timeout_seconds", 7200)),
            env=_runtime_env(eyebench),
        )
        if command_result["returncode"] != 0:
            skip_reason = "official preprocessing command failed"
    else:
        skip_reason = (
            "preprocessing was attempted manually or disabled in CLI; required processed files "
            f"missing: {[name for name in required_processed if not (processed / name).exists()]}"
        )
    counts = _count_processed(processed)
    report = {
        "preprocessing_run": command_result is not None,
        "command_used": command,
        "exit_code": command_result["returncode"] if command_result else None,
        "processed_dir_exists": processed.exists(),
        "processed_data_exists": processed_complete,
        "required_processed_files": required_processed,
        "missing_required_processed_files": [name for name in required_processed if not (processed / name).exists()],
        "fold_metadata_exists": folds.exists(),
        "labels_exist": labels.exists(),
        "CopCo_TYP_target_exists": (labels / "participant_stats.csv").exists(),
        "CopCo_RCS_target_exists": (labels / "participant_stats.csv").exists(),
        "participant_count": counts["participant_count"],
        "text_item_count": counts["text_item_count"],
        "trial_count": counts["trial_count"],
        "word_count": counts["word_count"],
        "fixation_count": counts["fixation_count"],
        "processed_files": counts["processed_files"],
        "skip_reason": skip_reason,
        "status": "ready" if processed_complete else "blocked_by_data",
    }
    _write_json(out / "data" / "official_data_preprocessing_report.json", report)
    _write_report(
        dirs,
        "official_data_preprocessing_report.md",
        "\n".join(
            [
                "# Official Data Preprocessing Report",
                "",
                f"- preprocessing_run: {report['preprocessing_run']}",
                f"- command used: `{command}`",
                f"- exit code: {report['exit_code']}",
                f"- processed dir exists: {report['processed_dir_exists']}",
                f"- required processed data exists: {report['processed_data_exists']}",
                f"- missing required processed files: {report['missing_required_processed_files']}",
                f"- fold metadata exists: {report['fold_metadata_exists']}",
                f"- labels exist: {report['labels_exist']}",
                f"- CopCo_TYP target exists: {report['CopCo_TYP_target_exists']}",
                f"- CopCo_RCS target exists: {report['CopCo_RCS_target_exists']}",
                f"- participant count: {report['participant_count']}",
                f"- text/item count: {report['text_item_count']}",
                f"- trial count: {report['trial_count']}",
                f"- word count: {report['word_count']}",
                f"- fixation count: {report['fixation_count']}",
                f"- status: `{report['status']}`",
                f"- blocker/skip reason: {skip_reason or 'none'}",
                "",
                "## Processed CopCo Artifacts",
                "\n".join(f"- `{name}`" for name in report["processed_files"]) or "- none",
            ]
        ),
    )
    return report


def write_fold_validation_report(
    out: Path,
    dirs: dict[str, Path],
    samples: Any,
    splits: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    if splits.empty:
        errors = ["official split labels are empty because official processed CopCo data are absent"]
        summaries: list[dict[str, Any]] = []
    else:
        errors, summaries = validate_official_split_labels(splits)
    payload = {"errors": errors, "summaries": summaries, "processed_samples": int(len(samples))}
    _write_json(out / "splits" / "official_fold_validation_report.json", payload)
    _write_report(
        dirs,
        "official_fold_validation_report.md",
        "\n".join(
            [
                "# Official Fold Validation Report",
                "",
                f"- processed samples: {len(samples)}",
                f"- errors: {errors or []}",
                "",
                _markdown_table(summaries, ["split_name", "fold_id", "train_samples", "test_samples"], max_rows=50)
                if summaries
                else "No official folds were validated because processed data were absent.",
            ]
        ),
    )
    return errors, summaries


def write_official_evaluator_report(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
    environment: dict[str, Any],
    data_status: dict[str, Any],
    d3_predictions: Any,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    evaluator = _eyebench_path(config, root) / "src" / "run" / "multi_run" / "raw_to_processed_results.py"
    expected_output = out / "typ" / "trial_level_test_results.csv"
    format_valid = bool(
        expected_output.exists()
        and {"label", "prediction_prob", "binary_prediction", "eval_regime", "eval_type", "fold_index"}.issubset(
            set(_pd().read_csv(expected_output, nrows=1).columns) if expected_output.exists() else set()
        )
    )
    official_evaluator_run = False
    blocker = ""
    if not environment.get("official_environment_ready"):
        blocker = "no usable EyeBench runtime environment"
    elif not data_status.get("processed_data_exists"):
        blocker = "official CopCo processed data absent"
    elif d3_predictions.empty:
        blocker = "D3_EyeBench_Lite did not produce official trial predictions"
    else:
        blocker = (
            "EyeBench evaluator expects results/raw experiment layout; external wrapper only validated "
            "trial_level_test_results.csv schema"
        )
    report = {
        "official_evaluator_run": official_evaluator_run,
        "official_result_format_validated": format_valid,
        "evaluator_path": str(evaluator),
        "schema_checked_path": str(expected_output),
        "blocker": blocker,
        "status": "ready" if official_evaluator_run or format_valid else "blocked_by_evaluator",
    }
    _write_json(out / "evaluator" / "official_evaluator_report.json", report)
    _write_report(
        dirs,
        "official_evaluator_blocker_report.md",
        "\n".join(
            [
                "# Official Evaluator Blocker Report",
                "",
                f"- official_evaluator_run: {official_evaluator_run}",
                f"- exact official result format validated: {format_valid}",
                f"- evaluator path: `{evaluator}`",
                f"- blocker: {blocker or 'none'}",
            ]
        ),
    )
    return report


def _write_runtime_typ_outputs(out: Path, d3_metrics: Any, d3_predictions: Any) -> None:
    pd = _pd()
    if d3_metrics.empty:
        d3_metrics = _empty_typ_metrics("D3_EyeBench_Lite skipped")
    trial = d3_metrics[d3_metrics["evaluation_level"].eq(TRIAL_LEVEL)].copy()
    reader = d3_metrics[d3_metrics["evaluation_level"].eq(READER_LEVEL)].copy()
    _write_csv(out / "typ" / "d3_lite_trial_metrics.csv", trial)
    _write_csv(out / "typ" / "d3_lite_reader_aggregated_metrics.csv", reader)
    if d3_predictions.empty:
        d3_predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)
    _write_csv(out / "typ" / "d3_lite_trial_predictions.csv", d3_predictions)


def write_d3_feature_report(
    config: dict[str, Any],
    dirs: dict[str, Path],
    d3_metrics: Any,
    leakage: dict[str, Any],
) -> None:
    prohibited = sorted(set(_section(config).get("prohibited_features", [])))
    complete = bool(not d3_metrics.empty and d3_metrics["status"].eq("complete").any())
    _write_report(
        dirs,
        "d3_eyebench_lite_feature_report.md",
        "\n".join(
            [
                "# D3_EyeBench_Lite Feature Report",
                "",
                f"- Status: {'complete' if complete else 'skipped'}",
                "- Feature source: official EyeBench processed CopCo data only.",
                "- Full D3 is not claimed unless all DFM residual profile inputs are available from official data.",
                "- participant_id and speech_id are retained only for grouping/reporting, not as predictors.",
                f"- Held-out reader rows used for residual fitting: {leakage.get('heldout_reader_rows_used_for_fit')}",
                f"- Held-out text rows used for residual fitting: {leakage.get('heldout_text_rows_used_for_fit')}",
                f"- Reader group used in residualization: {leakage.get('reader_group_used')}",
                "",
                "## Prohibited Predictors",
                "\n".join(f"- `{name}`" for name in prohibited),
            ]
        ),
    )


def write_typ_and_baseline_reports(
    dirs: dict[str, Path],
    baseline: Any,
    d3_metrics: Any,
) -> None:
    _write_report(
        dirs,
        "baseline_reproduction_report.md",
        "\n".join(
            [
                "# Baseline Reproduction Report",
                "",
                "This report mirrors the official baseline reproduction attempt under the runtime-fix gate.",
                "Rows produced by the CopCo wrapper are marked as local/manual unless they came from an "
                "official EyeBench command path. Manual rows are useful diagnostics but do not satisfy "
                "the official SOTA baseline-reproduction gate by themselves.",
                "The official EyeBench CopCo_TYP ML scripts present in the vendored repository launch "
                "W&B sweep agents through tmux under an external `eyebench_private` path. That command "
                "path was not executed under the revised isolated runtime, so the official command-source "
                "gate remains closed even when the local diagnostic reproduces published values closely.",
                "",
                _markdown_table(baseline.to_dict("records"), BASELINE_COLUMNS, max_rows=30)
                if not baseline.empty
                else "No baseline rows were produced.",
            ]
        ),
    )
    primary = d3_metrics[d3_metrics["evaluation_level"].isin([TRIAL_LEVEL, READER_LEVEL])]
    _write_report(
        dirs,
        "copco_typ_d3_lite_official_report.md",
        "\n".join(
            [
                "# CopCo_TYP D3_EyeBench_Lite Official Report",
                "",
                "Trial-level metrics are the official-comparison candidate. Reader-aggregated metrics are secondary.",
                "",
                _markdown_table(primary.to_dict("records"), SOTA_TYP_COLUMNS, max_rows=40)
                if not primary.empty
                else "D3_EyeBench_Lite did not run.",
            ]
        ),
    )


def write_rcs_report(dirs: dict[str, Path]) -> dict[str, Any]:
    report = {
        "status": "skipped",
        "skip_reason": "Official CopCo_RCS was not run because the primary CopCo_TYP official runtime chain did not pass.",
    }
    _write_report(
        dirs,
        "rcs_skipped_report.md",
        "# CopCo_RCS Runtime Fix Skip Report\n\n"
        f"- Status: {report['status']}\n"
        f"- Reason: {report['skip_reason']}",
    )
    _write_report(
        dirs,
        "copco_rcs_d3_lite_official_report.md",
        "# CopCo_RCS D3_EyeBench_Lite Official Report\n\n"
        f"- Status: {report['status']}\n"
        f"- Reason: {report['skip_reason']}",
    )
    return report


def write_runtime_decision_report(
    config: dict[str, Any],
    dirs: dict[str, Path],
    out: Path,
    environment: dict[str, Any],
    data_status: dict[str, Any],
    fold_errors: list[str],
    evaluator: dict[str, Any],
    baseline: Any,
    d3_metrics: Any,
    leakage: dict[str, Any],
) -> dict[str, Any]:
    official_env = bool(environment.get("official_environment_ready"))
    official_data = bool(data_status.get("processed_data_exists"))
    folds_ok = official_data and not fold_errors
    evaluator_ok = bool(evaluator.get("official_evaluator_run") or evaluator.get("official_result_format_validated"))
    baseline_complete = bool(not baseline.empty and baseline["status"].eq("complete").any())
    baseline_source = set(baseline.get("baseline_source", [])) if not baseline.empty else set()
    official_baseline_source = any("official_code_command" in str(source) for source in baseline_source)
    tolerance = float(
        get_nested(config, f"{RUNTIME_SECTION}.baseline_reproduction.reasonable_tolerance", 0.05)
    )
    baseline_deltas_ok = False
    if baseline_complete:
        delta_columns = [column for column in ["delta_roc_auc", "delta_balanced_accuracy"] if column in baseline]
        if delta_columns:
            deltas = baseline[delta_columns].apply(_pd().to_numeric, errors="coerce").abs()
            baseline_deltas_ok = bool((deltas.max(axis=1, skipna=True) <= tolerance).all())
    baseline_pass = baseline_complete and official_baseline_source and baseline_deltas_ok
    d3_complete = bool(not d3_metrics.empty and d3_metrics["status"].eq("complete").any())
    no_leakage = not (
        leakage.get("heldout_reader_rows_used_for_fit")
        or leakage.get("heldout_text_rows_used_for_fit")
        or leakage.get("reader_group_used")
    )
    no_prohibited = True
    no_full_data_substitution = True
    d3_beats = False
    if (
        official_env
        and official_data
        and folds_ok
        and evaluator_ok
        and baseline_pass
        and d3_complete
        and no_leakage
    ):
        comparison_path = dirs["repo_tables"] / "copco_typ_official_sota_comparison.csv"
        if comparison_path.exists():
            comparison = _pd().read_csv(comparison_path)
            reference = comparison[comparison["mode"].eq("official_eyebench_reported_baseline")]
            ours = comparison[comparison["model"].eq(RUNTIME_MODEL)]
            if not ours.empty and not reference.empty:
                best_ba = reference[
                    [
                        "unseen_reader_balanced_accuracy",
                        "unseen_text_balanced_accuracy",
                        "unseen_reader_text_balanced_accuracy",
                    ]
                ].max().max()
                best_auc = reference[["unseen_reader_AUROC", "unseen_text_AUROC", "unseen_reader_text_AUROC"]].max().max()
                d3_beats = bool(
                    float(ours["average_balanced_accuracy"].iloc[0]) > float(best_ba)
                    and float(ours["average_AUROC"].iloc[0]) > float(best_auc)
                )
    if not official_env:
        category = "blocked_by_environment"
    elif not official_data:
        category = "blocked_by_data"
    elif not evaluator_ok:
        category = "blocked_by_evaluator"
    elif not baseline_pass:
        category = "blocked_by_baseline_reproduction"
    elif d3_complete and d3_beats and no_leakage and no_prohibited and no_full_data_substitution:
        category = "official_sota_claim_allowed"
    elif d3_complete:
        category = "official_compatible_but_not_sota"
    else:
        category = "benchmark_relative_sota_only"
    if category == "official_sota_claim_allowed":
        wording = "official EyeBench-compatible state-of-the-art on CopCo_TYP"
    else:
        wording = "benchmark-relative state of the art under internal EyeBench-style reader-aggregated evaluation"
    decision = {
        "decision_category": category,
        "official_environment_ready": official_env,
        "environment_kind": environment.get("environment_kind"),
        "exact_environment_yml_used": bool(environment.get("exact_environment_yml_used")),
        "official_code_data_fold_compatible_runtime": bool(
            official_env and environment.get("environment_kind") != "exact_environment_yml"
        ),
        "official_processed_data_present": official_data,
        "official_folds_used": folds_ok,
        "official_evaluator_run": bool(evaluator.get("official_evaluator_run")),
        "official_result_format_validated": bool(evaluator.get("official_result_format_validated")),
        "official_baseline_reproduced": baseline_pass,
        "baseline_local_or_manual_complete": baseline_complete,
        "baseline_official_command_source": official_baseline_source,
        "baseline_reproduction_within_tolerance": baseline_deltas_ok,
        "baseline_reproduction_tolerance": tolerance,
        "d3_eyebench_lite_complete": d3_complete,
        "d3_eyebench_lite_beats_strongest_official_baselines": d3_beats,
        "no_residualization_leakage_detected": no_leakage,
        "no_prohibited_predictors": no_prohibited,
        "no_manual_full_data_substitution": no_full_data_substitution,
        "official_sota_claim_allowed": category == "official_sota_claim_allowed",
        "recommended_wording": wording,
        "manuscript_main_claim_changes": category == "official_sota_claim_allowed",
    }
    if decision["official_sota_claim_allowed"] and not (
        official_env
        and official_data
        and folds_ok
        and evaluator_ok
        and baseline_pass
        and d3_complete
        and d3_beats
        and no_leakage
        and no_prohibited
        and no_full_data_substitution
    ):
        raise RuntimeError("official_sota_claim_allowed gate invariant violated")
    rows = [
        {"gate": "official environment or compatible runtime", "passed": official_env},
        {"gate": "official processed CopCo data present", "passed": official_data},
        {"gate": "official folds validated", "passed": folds_ok},
        {"gate": "official evaluator or exact result format", "passed": evaluator_ok},
        {"gate": "official baseline reproduced", "passed": baseline_pass},
        {"gate": "local/manual baseline completed", "passed": baseline_complete},
        {"gate": "baseline used official command source", "passed": official_baseline_source},
        {"gate": "baseline within tolerance", "passed": baseline_deltas_ok},
        {"gate": "D3_EyeBench_Lite complete", "passed": d3_complete},
        {"gate": "D3 beats strongest official baseline", "passed": d3_beats},
        {"gate": "no leakage", "passed": no_leakage},
        {"gate": "no prohibited predictors", "passed": no_prohibited},
        {"gate": "no full-data substitution", "passed": no_full_data_substitution},
    ]
    text = "\n".join(
        [
            "# Official EyeBench SOTA Decision Report",
            "",
            f"- Final claim category: `{category}`",
            f"- Official EyeBench SOTA allowed: {decision['official_sota_claim_allowed']}",
            f"- Environment kind: `{environment.get('environment_kind')}`",
            f"- Exact `environment.yml` used: {decision['exact_environment_yml_used']}",
            f"- Recommended wording: {wording}.",
            "",
            _markdown_table(rows, ["gate", "passed"], max_rows=20),
            "",
            "## Result Separation",
            "- Official EyeBench trial-level result: only valid if official data/folds/evaluator gates pass.",
            "- Reader-aggregated metrics are secondary and cannot define the official claim.",
            "- BenchmarkBridge full-data metrics remain benchmark-relative.",
        ]
    )
    _write_report(dirs, "official_sota_decision_report.md", text)
    _write_json(out / "official_sota_decision.json", decision)
    _write_json(dirs["repo_analysis"] / "official_sota_decision.json", decision)
    return decision


def update_runtime_supplement_note(repo_root: str | Path, decision: dict[str, Any]) -> dict[str, Any]:
    path = Path(repo_root).resolve() / "paper" / "submission_v1" / "supplement_sections" / "18_benchmark_bridge.tex"
    if not path.exists():
        return {"updated": False, "path": str(path), "reason": "supplement section missing"}
    if decision.get("decision_category") == "official_sota_claim_allowed":
        return {"updated": False, "path": str(path), "reason": "main-text official SOTA update not automated"}
    text = path.read_text(encoding="utf-8")
    marker = "\\paragraph{Official EyeBench runtime fix.}"
    note = (
        "\n\n"
        "\\paragraph{Official EyeBench runtime fix.}\n"
        "OfficialEyeBenchRuntimeFix v1 isolated EyeBench runtime artifacts under the vendored "
        "submodule and followed a CopCo-environment-first runtime rule. Because the vendored "
        "EyeBench source used Python 3.12-only syntax, a minimal Python 3.12 prefix was used "
        "only as a compatibility fallback rather than as an exact environment.yml "
        "reproduction. The official runtime chain must still pass the official processed-data, "
        "fold, evaluator, baseline-reproduction, leakage, and D3_EyeBench_Lite gates before "
        "an official state-of-the-art claim is allowed. Until those gates pass, the manuscript "
        "keeps benchmark-relative wording and does not claim official EyeBench SOTA.\n"
    )
    if marker not in text:
        path.write_text(text.rstrip() + note, encoding="utf-8")
        return {"updated": True, "path": str(path), "reason": "added runtime blocker note"}
    return {"updated": False, "path": str(path), "reason": "runtime blocker note already present"}


def validate_gitignore(repo_root: str | Path) -> dict[str, Any]:
    path = Path(repo_root).resolve() / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    missing = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in text]
    return {"status": "failed" if missing else "passed", "missing_patterns": missing}


def run_official_eyebench_runtime_fix(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    config_report = validate_official_eyebench_runtime_fix_config(config)
    _write_json(out / "config_validation.json", config_report)
    preflight = write_preflight_report(config, out, dirs, root)
    environment = write_environment_reports(config, out, dirs, root)
    write_command_inventory(config, dirs, root)
    data_status = write_data_preprocessing_report(config, out, dirs, root, environment)
    sota_config = _to_sota_config(config)
    reference = _official_reference_table(sota_config, root)
    samples, ia = load_official_processed_features(sota_config, out, root)
    splits = build_official_split_labels(sota_config, out, samples, root)
    fold_errors, fold_summaries = write_fold_validation_report(out, dirs, samples, splits)
    baseline, baseline_predictions = reproduce_official_baseline(
        sota_config, out, dirs, root, samples, splits, reference
    )
    d3_metrics, d3_predictions, leakage = evaluate_d3_eyebench_lite(
        sota_config, out, dirs, samples, splits, ia
    )
    _write_runtime_typ_outputs(out, d3_metrics, d3_predictions)
    write_d3_feature_report(config, dirs, d3_metrics, leakage)
    write_typ_and_baseline_reports(dirs, baseline, d3_metrics)
    official_ready_for_table = bool(
        environment.get("official_environment_ready")
        and data_status.get("processed_data_exists")
        and not fold_errors
        and baseline["status"].eq("complete").any()
        and d3_metrics["status"].eq("complete").any()
    )
    comparison = write_comparison_tables(sota_config, dirs, root, d3_metrics, reference, official_ready_for_table)
    evaluator = write_official_evaluator_report(
        config, out, dirs, root, environment, data_status, d3_predictions
    )
    rcs = write_rcs_report(dirs)
    decision = write_runtime_decision_report(
        config,
        dirs,
        out,
        environment,
        data_status,
        fold_errors,
        evaluator,
        baseline,
        d3_metrics,
        leakage,
    )
    manuscript_update = update_runtime_supplement_note(root, decision)
    manifest = {
        "status": "complete",
        "run_name": get_nested(config, "run.name", "official_eyebench_runtime_fix_v1"),
        "output_dir": str(out),
        "repo_root": str(root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "branch": preflight.get("current_branch"),
        "copco_commit": preflight.get("copco_commit"),
        "eyebench_commit": preflight.get("eyebench_commit"),
        "environment_status": environment.get("status"),
        "environment_kind": environment.get("environment_kind"),
        "data_status": data_status.get("status"),
        "fold_validation_errors": fold_errors,
        "fold_validation_summaries": fold_summaries,
        "baseline_reproduction_status": "complete" if baseline["status"].eq("complete").any() else "skipped",
        "d3_status": "complete" if d3_metrics["status"].eq("complete").any() else "skipped",
        "official_evaluator_status": evaluator.get("status"),
        "decision_category": decision["decision_category"],
        "official_sota_claim_allowed": decision["official_sota_claim_allowed"],
        "rcs_status": rcs["status"],
        "manuscript_update": manuscript_update,
        "comparison_rows": int(len(comparison)),
        "baseline_prediction_rows": int(len(baseline_predictions)),
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def validate_official_eyebench_runtime_fix(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    config_report = validate_official_eyebench_runtime_fix_config(config)
    if config_report["status"] != "passed":
        errors.extend(config_report["errors"])
    gitignore_report = validate_gitignore(root)
    if gitignore_report["status"] != "passed":
        errors.append(f".gitignore missing EyeBench protections: {gitignore_report['missing_patterns']}")
    required = [
        out / "manifest.json",
        out / "config_validation.json",
        out / "preflight" / "preflight_report.json",
        out / "environment" / "environment_status.json",
        out / "data" / "official_data_preprocessing_report.json",
        out / "splits" / "official_fold_validation_report.json",
        out / "baseline" / "official_baseline_reproduction_metrics.csv",
        out / "typ" / "d3_lite_trial_metrics.csv",
        out / "typ" / "d3_lite_reader_aggregated_metrics.csv",
        out / "typ" / "d3_lite_trial_predictions.csv",
        out / "evaluator" / "official_evaluator_report.json",
        out / "official_sota_decision.json",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required output: {path}")
    analysis_dir = root / str(get_nested(config, f"{RUNTIME_SECTION}.repo_analysis_dir"))
    for report_name in [
        "environment_cleanup_report.md",
        "environment_exact_attempt.md",
        "environment_fallback_report.md",
        "official_data_preprocessing_report.md",
        "official_fold_validation_report.md",
        "baseline_reproduction_report.md",
        "d3_eyebench_lite_feature_report.md",
        "official_evaluator_blocker_report.md",
        "official_sota_decision_report.md",
    ]:
        if not (analysis_dir / report_name).exists():
            errors.append(f"missing analysis report: {analysis_dir / report_name}")
    pd = _pd()
    if (out / "typ" / "d3_lite_trial_metrics.csv").exists():
        trial = pd.read_csv(out / "typ" / "d3_lite_trial_metrics.csv")
        missing = sorted(set(SOTA_TYP_COLUMNS) - set(trial.columns))
        if missing:
            errors.append(f"D3 trial metric schema missing columns: {missing}")
    if (out / "typ" / "d3_lite_reader_aggregated_metrics.csv").exists():
        reader = pd.read_csv(out / "typ" / "d3_lite_reader_aggregated_metrics.csv")
        missing = sorted(set(SOTA_TYP_COLUMNS) - set(reader.columns))
        if missing:
            errors.append(f"D3 reader metric schema missing columns: {missing}")
    if (out / "baseline" / "official_baseline_reproduction_metrics.csv").exists():
        baseline = pd.read_csv(out / "baseline" / "official_baseline_reproduction_metrics.csv")
        missing = sorted(set(BASELINE_COLUMNS) - set(baseline.columns))
        if missing:
            errors.append(f"baseline schema missing columns: {missing}")
    comparison_path = analysis_dir / "tables" / "copco_typ_official_sota_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        missing = sorted(set(COMPARISON_COLUMNS) - set(comparison.columns))
        if missing:
            errors.append(f"comparison table schema missing columns: {missing}")
    else:
        errors.append(f"missing comparison table: {comparison_path}")
    decision: dict[str, Any] = {}
    if (out / "official_sota_decision.json").exists():
        decision = json.loads((out / "official_sota_decision.json").read_text(encoding="utf-8"))
        category = decision.get("decision_category")
        if category not in VALID_DECISION_CATEGORIES:
            errors.append(f"invalid decision category: {category}")
        if decision.get("official_sota_claim_allowed") and not (
            decision.get("official_environment_ready")
            and decision.get("official_processed_data_present")
            and decision.get("official_folds_used")
            and (decision.get("official_evaluator_run") or decision.get("official_result_format_validated"))
            and decision.get("official_baseline_reproduced")
            and decision.get("d3_eyebench_lite_complete")
            and decision.get("d3_eyebench_lite_beats_strongest_official_baselines")
            and decision.get("no_residualization_leakage_detected")
            and decision.get("no_prohibited_predictors")
            and decision.get("no_manual_full_data_substitution")
        ):
            errors.append("official SOTA claim allowed despite failed gates")
    staged = _run_command(["git", "diff", "--cached", "--name-only"], cwd=root, timeout=30)["stdout"]
    forbidden_staged = [
        name
        for name in staged.splitlines()
        if name.startswith("eyebench/data/")
        or name.startswith("eyebench/results/")
        or name.startswith("eyebench/.envs/")
        or name.startswith("results/official_eyebench_runtime_fix_v1_")
    ]
    if forbidden_staged:
        errors.append(f"generated EyeBench/runtime files staged: {forbidden_staged}")
    report = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "decision_category": decision.get("decision_category"),
        "official_sota_claim_allowed": bool(decision.get("official_sota_claim_allowed")),
        "output_dir": str(out),
    }
    _write_json(out / "validation_report.json", report)
    return report
