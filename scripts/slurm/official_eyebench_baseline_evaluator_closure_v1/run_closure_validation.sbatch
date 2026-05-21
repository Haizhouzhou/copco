#!/usr/bin/bash -l
#SBATCH --job-name=oe_closure_validate
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=results/official_eyebench_baseline_evaluator_closure_v1_sbatch/slurm/%x_%j.out
#SBATCH --error=results/official_eyebench_baseline_evaluator_closure_v1_sbatch/slurm/%x_%j.err

set -euxo pipefail
cd /home/haizhe/copco

OUTDIR="${CLOSURE_OUTDIR:-results/official_eyebench_baseline_evaluator_closure_v1_${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR/slurm"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "HOSTNAME=$(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
date

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
export PYTHONPATH="$PWD/eyebench:$PWD/eyebench/src:${PYTHONPATH:-}"

START_TS="$(date --iso-8601=seconds)"
set +e
conda run -n copco copco-validate-official-eyebench-baseline-evaluator-closure \
  --config configs/official_eyebench_baseline_evaluator_closure_v1.yaml \
  --output-dir "$OUTDIR"
RC=$?
set -e
END_TS="$(date --iso-8601=seconds)"

cat > "$OUTDIR/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_manifest.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "job_name": "${SLURM_JOB_NAME}",
  "command": "copco-validate-official-eyebench-baseline-evaluator-closure --config configs/official_eyebench_baseline_evaluator_closure_v1.yaml --output-dir ${OUTDIR}",
  "start_time": "${START_TS}",
  "end_time": "${END_TS}",
  "exit_code": ${RC},
  "stdout_path": "results/official_eyebench_baseline_evaluator_closure_v1_sbatch/slurm/%x_%j.out",
  "stderr_path": "results/official_eyebench_baseline_evaluator_closure_v1_sbatch/slurm/%x_%j.err",
  "output_dir": "${OUTDIR}"
}
EOF
exit "$RC"
