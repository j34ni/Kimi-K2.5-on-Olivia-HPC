[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18968392.svg)](https://doi.org/10.5281/zenodo.18968392)

# Kimi-K2.5-on-Olivia-HPC
This repository contains the files needed to run Kimi K2.5 on the Olivia HPC cluster at the University of Oslo, using SGLang as the inference server.

## Container
The Apptainer container is built from kimi-k25.def and includes:

- Base image: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` (CUDA 12.6, required, since Olivia's driver does not support 12.9)
- Python 3.12 via Miniforge
- `mvapich=4.1` (`shs_cuda` variant from conda-forge), which brings in `shs-libfabric`, `shs-libcxi`, `shs-cassini-headers` and `gdrcopy` for Slingshot-11/CXI support
- `aws-ofi-nccl 1.17.0` built against the conda `shs-libfabric` enabling NCCL to use the CXI fabric directly for GPU-to-GPU transfers
- `SGLang 0.5.9`

Build the container *interactively* on a compute node (with something like `salloc --partition=accel --gres=gpu:1 --nodes=1 --ntasks-per-node=16 --cpus-per-task=1 --mem-per-cpu=2G --time=01:00:00 --account=nnXXXXk`) and use:

```
apptainer build kimi-k25.sif kimi-k25.def
```

## Fabric validation

Before running the full job, validate the Slingshot fabric and MPI setup across 2 nodes using the OSU micro-benchmarks (bundled with `mvapich` inside the container):
```
sbatch osu.sh
```

Expected results:

- `fi_info -p cxi` should show 4 CXI domains per node
- `H H` latency ~3-4µs
- `H H` bandwidth ~24 GB/s
- `D D` numbers will be lower due to MPI staging overhead, NCCL bypasses this and will achieve closer to `H H` bandwidth for actual inference traffic

## Note

The model weights (~630GB) must be pre-downloaded before submitting the job.
