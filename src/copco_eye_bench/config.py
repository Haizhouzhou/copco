"""Configuration helpers for reproducible CopCo runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/copco_dyslexia_full.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by CLI in lean envs
        raise RuntimeError(
            "PyYAML is required to read CopCo pipeline configs. "
            "Install project runtime dependencies inside the copco environment."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating either input."""

    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        elif key != "extends":
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving a single-level or nested ``extends`` chain."""

    root = Path(repo_root or ".").resolve()
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()

    config = _read_yaml(config_path)
    parent = config.get("extends")
    if parent:
        parent_config = load_config(parent, repo_root=root)
        return deep_merge(parent_config, config)
    return config


def get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Return a dotted config value with a default."""

    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def timestamped_output_dir(config: dict[str, Any], *, repo_root: str | Path | None = None) -> Path:
    """Return a timestamped output directory for a configured run."""

    from datetime import datetime

    root = Path(repo_root or ".").resolve()
    output_root = Path(get_nested(config, "run.output_root", "results"))
    if not output_root.is_absolute():
        output_root = root / output_root
    name = str(get_nested(config, "run.name", "copco_dyslexia_full"))
    fmt = str(get_nested(config, "run.timestamp_format", "%Y%m%d_%H%M%S"))
    return output_root / f"{name}_{datetime.now().strftime(fmt)}"
