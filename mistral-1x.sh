#!/bin/bash
#SBATCH --account=nn9999k
#SBATCH --partition=accel
#SBATCH --job-name=mistral-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=96G
#SBATCH --time=02:00:00

SIF=sglang-059.sif
CACHE=/cluster/software/gpujobscratch/uiollm/cache-uiollm
MISTRAL_PATH=${CACHE}/huggingface/models--mistralai--Mistral-Large-Instruct-2411/snapshots/ba78820945ae22361b0274cf0ae6d696c967c1a4

export APPTAINER_TMPDIR=${CACHE}/apptainer/tmp
export APPTAINER_CACHEDIR=${CACHE}/apptainer/cache
mkdir -p ${APPTAINER_TMPDIR} ${APPTAINER_CACHEDIR}

export no_proxy=localhost,127.0.0.1

srun --mpi=none apptainer exec --nv \
     --env XDG_CACHE_HOME=${CACHE} \
     --env TRITON_CACHE_DIR=${CACHE}/triton \
     --env TORCHINDUCTOR_CACHE_DIR=${CACHE}/torchinductor \
     --env FLASHINFER_CACHE_DIR=${CACHE}/flashinfer \
     --env no_proxy=localhost,127.0.0.1 \
     --bind ${CACHE}:${CACHE} \
     ${SIF} \
     python -m sglang.launch_server \
            --model-path ${MISTRAL_PATH} \
            --trust-remote-code \
            --host 127.0.0.1 \
            --port 8000 \
            --tp 4 \
            --mem-fraction-static 0.7 
