"""Slurm command construction for CopCo heavy jobs."""

from __future__ import annotations

import shlex
from pathlib import Path


TESTED_TEACHING_8GPU_CANDIDATE = (
    "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal "
    "--gres=gpu:8 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00"
)
TESTED_TEACHING_CPU_CANDIDATE = (
    "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal "
    "--nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00"
)


PREFLIGHT = r"""
hostname
echo SLURM_JOB_ID=$SLURM_JOB_ID
echo SLURM_STEP_ID=${SLURM_STEP_ID:-}
echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
echo SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}
echo SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}
echo SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-}
nproc
free -h
ulimit -a
nvidia-smi || true
python - <<'PY'
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
"""


def launcher_command(
    command: str,
    *,
    repo_root: str | Path,
    mode: str,
    prepend_tested_gpu_candidate: bool = True,
    conda_env: str = "copco",
) -> str:
    """Return a no-wait launcher command with required preflight included."""

    repo = Path(repo_root).resolve()
    activate = (
        f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate {shlex.quote(conda_env)}"
    )
    body = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo))}",
            activate,
            'export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"',
            'export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"',
            PREFLIGHT.strip(),
            command,
        ]
    )
    quoted = shlex.quote(body)
    if mode == "gpu":
        base = "~/bin/claim_best_immediate_resource.sh --mode gpu"
        if prepend_tested_gpu_candidate:
            candidate = shlex.quote(TESTED_TEACHING_8GPU_CANDIDATE)
            return f"{base} --candidate {candidate} {quoted}"
        return f"{base} {quoted}"
    if mode == "cpu":
        candidate = shlex.quote(TESTED_TEACHING_CPU_CANDIDATE)
        return f"~/bin/claim_best_immediate_resource.sh --mode cpu --candidate {candidate} {quoted}"
    raise ValueError(f"unknown slurm mode: {mode}")
