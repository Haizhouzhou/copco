#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/autoresearch_v1_reproduced}"
conda run -n copco copco-run-autoresearch --config configs/autoresearch_v1.yaml --output-dir "$OUT"
conda run -n copco copco-validate-autoresearch --config configs/autoresearch_v1.yaml --output-dir "$OUT"
