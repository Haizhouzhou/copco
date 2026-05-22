#!/usr/bin/bash -l
#SBATCH --job-name=d3_eyebench_opt_v1_accel
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/%x_%j.out
#SBATCH --error=results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/%x_%j.err

set -euxo pipefail
cd /home/haizhe/copco

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${D3_EYEBENCH_OPT_OUTDIR:-results/d3_eyebench_protocol_aligned_optimization_v1_accelerated_${TIMESTAMP}}"
mkdir -p "$OUTDIR/slurm" results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
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
set +e
{
  conda run -n copco python -m pip install -e .
  conda run -n copco python scripts/validate_env.py
  conda run -n copco copco-run-d3-eyebench-protocol-optimization \
    --config configs/d3_eyebench_protocol_aligned_optimization_v1_accelerated.yaml \
    --output-dir "$OUTDIR"
  conda run -n copco copco-validate-d3-eyebench-protocol-optimization \
    --config configs/d3_eyebench_protocol_aligned_optimization_v1_accelerated.yaml \
    --output-dir "$OUTDIR"
}
RC=$?
set -e
END_TS="$(date --iso-8601=seconds)"

cat > "$OUTDIR/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_manifest.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "command": "accelerated D3 EyeBench protocol-aligned optimization v1",
  "start_time": "${START_TS}",
  "end_time": "${END_TS}",
  "exit_code": ${RC},
  "stdout_path": "results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out",
  "stderr_path": "results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err",
  "output_dir": "${OUTDIR}"
}
EOF

set +e
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW
seff "$SLURM_JOB_ID"
set -e

exit "$RC"
