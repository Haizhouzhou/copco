"""Lightweight repository validation for CopCo Eye Bench."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_PATHS = (
    "AGENTS.md",
    "README.md",
    ".gitignore",
    "pyproject.toml",
    "configs/copco_dyslexia_full.yaml",
    "configs/copco_dyslexia_smoke.yaml",
    "src/copco_eye_bench/__init__.py",
    "src/copco_eye_bench/cli.py",
    "src/copco_eye_bench/features.py",
    "src/copco_eye_bench/ids.py",
    "src/copco_eye_bench/lm_features.py",
    "src/copco_eye_bench/modeling.py",
    "src/copco_eye_bench/mixed_effects.py",
    "src/copco_eye_bench/resources.py",
    "src/copco_eye_bench/slurm.py",
    "src/copco_eye_bench/splits.py",
    "src/copco_eye_bench/validation.py",
    "scripts/validate_env.py",
    "tests/test_import.py",
    "docs/decisions.md",
    "logs/ai_runs/INDEX.md",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))

    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"repo_root: {repo_root}")

    package = importlib.import_module("copco_eye_bench")
    print(f"package_version: {package.__version__}")

    missing = [path for path in REQUIRED_PATHS if not (repo_root / path).exists()]
    if missing:
        print("missing_required_paths:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("required_structure: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
