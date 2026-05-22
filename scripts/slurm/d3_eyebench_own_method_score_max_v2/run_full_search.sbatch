#!/usr/bin/bash -l
#SBATCH --job-name=d3_own_search_v2
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=results/d3_eyebench_own_method_score_max_v2_sbatch/slurm/%x_%j.out
#SBATCH --error=results/d3_eyebench_own_method_score_max_v2_sbatch/slurm/%x_%j.err

set -euxo pipefail
cd /home/haizhe/copco

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="${D3_OWN_SCORE_MAX_CONFIG:-configs/d3_eyebench_own_method_score_max_v2.yaml}"
OUTDIR="${D3_OWN_SCORE_MAX_OUTDIR:-results/d3_eyebench_own_method_score_max_v2_${TIMESTAMP}}"
LOGROOT="results/d3_eyebench_own_method_score_max_v2_sbatch/slurm"
mkdir -p "$OUTDIR/slurm" "$LOGROOT"

hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_STEP_ID=${SLURM_STEP_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-}"
nproc
free -h
ulimit -a
nvidia-smi || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export WANDB_DIR="$PWD/eyebench/wandb"
export HF_HOME="$PWD/eyebench/.cache/huggingface"
export TRANSFORMERS_CACHE="$PWD/eyebench/.cache/huggingface"
export PIP_CACHE_DIR="$PWD/eyebench/.pip_cache"
export PYTHONPATH="$PWD/src:$PWD/eyebench:$PWD/eyebench/src:${PYTHONPATH:-}"

conda run -n copco python - <<'PY'
import os
print("python preflight")
print("cpu_count_os", os.cpu_count())
try:
    import psutil
    print("cpu_count_psutil", psutil.cpu_count())
    print("virtual_memory", psutil.virtual_memory())
except Exception as e:
    print("psutil_unavailable", e)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, torch.cuda.get_device_name(i), p.total_memory)
except Exception as e:
    print("torch_preflight_failed", e)
PY

START_TS="$(date --iso-8601=seconds)"
COMMAND="copco-run-d3-eyebench-own-method-score-max --config $CONFIG_PATH --output-dir $OUTDIR --stage full"
set +e
{
  conda run -n copco python -m pip install -e .
  conda run -n copco python scripts/validate_env.py
  conda run -n copco copco-run-d3-eyebench-own-method-score-max \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTDIR" \
    --stage full
  conda run -n copco copco-validate-d3-eyebench-own-method-score-max \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTDIR"
}
RC=$?
set -e
END_TS="$(date --iso-8601=seconds)"

cat > "$OUTDIR/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_manifest.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "command": "${COMMAND}",
  "git_commit": "$(git rev-parse HEAD)",
  "config": "${CONFIG_PATH}",
  "runtime_environment": "conda:copco",
  "start_time": "${START_TS}",
  "end_time": "${END_TS}",
  "exit_code": ${RC},
  "stdout_path": "${LOGROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out",
  "stderr_path": "${LOGROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err",
  "metrics_path": "${OUTDIR}/typ/d3_own_method_trial_metrics.csv",
  "output_dir": "${OUTDIR}"
}
EOF

set +e
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW
seff "$SLURM_JOB_ID"
set -e

exit "$RC"
