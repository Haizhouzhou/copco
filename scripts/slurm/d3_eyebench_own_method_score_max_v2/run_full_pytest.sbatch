#!/usr/bin/bash -l
#SBATCH --job-name=d3_own_pytest_v2
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=results/d3_eyebench_own_method_score_max_v2_sbatch/slurm/%x_%j.out
#SBATCH --error=results/d3_eyebench_own_method_score_max_v2_sbatch/slurm/%x_%j.err

set -euxo pipefail
cd /home/haizhe/copco

LOGROOT="results/d3_eyebench_own_method_score_max_v2_sbatch/slurm"
mkdir -p "$LOGROOT"

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

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/eyebench:$PWD/eyebench/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

conda run -n copco python - <<'PY'
import os
print("python preflight")
print("cpu_count_os", os.cpu_count())
try:
    import psutil
    print("cpu_count_psutil", psutil.cpu_count())
    print("virtual_memory", psutil.virtual_memory())
except Exception as exc:
    print("psutil_unavailable", exc)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
except Exception as exc:
    print("torch_preflight_failed", exc)
PY

START_TS="$(date --iso-8601=seconds)"
set +e
conda run -n copco python -m pytest tests/ -q
RC=$?
set -e
END_TS="$(date --iso-8601=seconds)"

cat > "$LOGROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_manifest.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "command": "conda run -n copco python -m pytest tests/ -q",
  "git_commit": "$(git rev-parse HEAD)",
  "runtime_environment": "conda:copco",
  "start_time": "${START_TS}",
  "end_time": "${END_TS}",
  "exit_code": ${RC},
  "stdout_path": "${LOGROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out",
  "stderr_path": "${LOGROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
}
EOF

set +e
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW
seff "$SLURM_JOB_ID"
set -e

exit "$RC"
