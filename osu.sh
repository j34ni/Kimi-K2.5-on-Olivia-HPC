#!/bin/bash
#SBATCH --account=nn9997k
#SBATCH --job-name=osu-check
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=16G

IMAGE=kimi-k25.sif

export MPICH_CH4_NETMOD=ofi
export FI_PROVIDER=cxi

echo "Nodes: $SLURM_NODELIST"

apptainer exec $IMAGE fi_info -p cxi

echo "=== Host to Host latency ==="
srun --mpi=pmi2 -n $SLURM_NTASKS \
    apptainer --quiet exec --nv $IMAGE \
    /opt/conda/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency -m 1:1048576 H H

echo "=== Host to Host bandwidth ==="
srun --mpi=pmi2 -n $SLURM_NTASKS \
    apptainer --quiet exec --nv $IMAGE \
    /opt/conda/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 1:1048576 H H

echo "=== Device to Device latency ==="
srun --mpi=pmi2 -n $SLURM_NTASKS \
    apptainer --quiet exec --nv $IMAGE \
    /opt/conda/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency -m 1:1048576 D D

echo "=== Device to Device bandwidth ==="
srun --mpi=pmi2 -n $SLURM_NTASKS \
    apptainer --quiet exec --nv $IMAGE \
    /opt/conda/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 1:1048576 D D
