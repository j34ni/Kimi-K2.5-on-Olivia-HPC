#!/bin/bash
#SBATCH --account=nn9999k
#SBATCH --partition=accel
#SBATCH --job-name=gpt-oss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=96G
#SBATCH --time=02:00:00

SIF=kimi-k25.sif
CACHE=/cluster/software/gpujobscratch/uiollm/cache-uiollm

export APPTAINER_TMPDIR=${CACHE}/apptainer/tmp
export APPTAINER_CACHEDIR=${CACHE}/apptainer/cache
mkdir -p ${APPTAINER_TMPDIR} ${APPTAINER_CACHEDIR}

export FLASHINFER_DISABLE_VERSION_CHECK=1
export no_proxy=localhost,127.0.0.1

rm -rf ${CACHE}/flashinfer/*

srun --mpi=none apptainer exec --nv \
    --env XDG_CACHE_HOME=${CACHE} \
    --env TRITON_CACHE_DIR=${CACHE}/triton \
    --env TORCHINDUCTOR_CACHE_DIR=${CACHE}/torchinductor \
    --env FLASHINFER_CACHE_DIR=${CACHE}/flashinfer \
    --env no_proxy=localhost,127.0.0.1 \
    --bind ${CACHE}:${CACHE} \
    ${SIF} \
    python -m sglang.launch_server \
        --model-path openai/gpt-oss-120b \
        --trust-remote-code \
        --host 127.0.0.1 \
        --port 8000 \
        --tp 4 \
        --mem-fraction-static 0.7 \
        --cuda-graph-max-bs 512
