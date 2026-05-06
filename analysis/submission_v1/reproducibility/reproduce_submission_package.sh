#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/submission_v1_reproduced}"
conda run -n copco copco-build-submission-package --config configs/submission_v1.yaml --output-dir "$OUT" --allow-existing-output
conda run -n copco copco-validate-submission-package --config configs/submission_v1.yaml --output-dir "$OUT"
