#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/submission_v1_$(date +%Y%m%d_%H%M)}"
CLAIM_RESOURCE_LOG_DIR=logs/submission_v1_resource_logs \
  ~/bin/claim_best_immediate_resource.sh --mode cpu \
  --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" \
  "cd $(pwd) && conda run -n copco copco-build-submission-package --config configs/submission_v1.yaml --output-dir $OUT"
