#!/bin/bash
#SBATCH --account=nn12104k
#SBATCH --partition=accel
#SBATCH --reservation=uiollm
#SBATCH --job-name=kimi-k25
#SBATCH --nodes=4
#SBATCH --nodelist=gpu-1-[14,31,83-84]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=96G
#SBATCH --time=04:00:00

SIF=kimi-k25.sif
CACHE=/cluster/software/gpujobscratch/uiollm/cache-uiollm
HEAD_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
DIST_INIT_ADDR=${HEAD_NODE}:29500

export APPTAINER_TMPDIR=${CACHE}/apptainer/tmp
export APPTAINER_CACHEDIR=${CACHE}/apptainer/cache
mkdir -p ${APPTAINER_TMPDIR} ${APPTAINER_CACHEDIR}

srun --mpi=pmi2 apptainer exec --nv \
    --env XDG_CACHE_HOME=${CACHE} \
    --env TRITON_CACHE_DIR=${CACHE}/triton \
    --env TORCHINDUCTOR_CACHE_DIR=${CACHE}/torchinductor \
    --env FLASHINFER_CACHE_DIR=${CACHE}/flashinfer \
    --bind ${CACHE}:${CACHE} \
    ${SIF} \
    bash -c "python -m sglang.launch_server \
        --model-path moonshotai/Kimi-K2.5 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 8000 \
        --tool-call-parser kimi_k2 \
        --reasoning-parser kimi_k2 \
        --mem-fraction-static 0.85 \
        --disable-radix-cache \
        --max-total-tokens 32768 \
        --dist-init-addr ${DIST_INIT_ADDR} \
        --tp 16 \
        --nnodes 4 \
        --node-rank \$SLURM_NODEID"
