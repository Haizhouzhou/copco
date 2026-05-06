#!/usr/bin/env bash
set -euo pipefail
conda run -n copco python scripts/validate_env.py
conda run -n copco copco-validate-label-release --config configs/label_release_v1_1.yaml --output-dir results/label_release_v1_1_20260506_0041
conda run -n copco copco-validate-phase4-confirmatory --config configs/phase4_confirmatory_sensitivity_v1.yaml --output-dir results/phase4_confirmatory_sensitivity_v1_20260506_0715
conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir "${1:-results/autoresearch_v1_reproduced}"
conda run -n copco copco-validate-autoresearch --config configs/autoresearch_v1.yaml --output-dir "${1:-results/autoresearch_v1_reproduced}"
