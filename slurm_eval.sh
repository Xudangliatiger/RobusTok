#!/usr/bin/env bash
set -euxo pipefail

# Quick evaluation on gputest (15 min max). Switch to gpusmall/gpumedium for longer runs.
srun -A project_2002147 -p gputest \
     --gres=gpu:a100:1 \
     --cpus-per-task=4 --mem=32G \
     -t 00:10:00 \
     --export=ALL,PYTHONUNBUFFERED=1 \
     --label \
     bash -lc '
       # Activate conda environment
       source /scratch/project_2002147/dongli/miniconda3/etc/profile.d/conda.sh
       conda activate alitok

       # Ensure LD_LIBRARY_PATH contains NVIDIA libs shipped with the env
       NVDIR="$($CONDA_PREFIX/bin/python - <<'"'"'PY'"'"'
import importlib.util, pathlib
spec = importlib.util.find_spec("nvidia")
print(pathlib.Path(spec.submodule_search_locations[0]) if spec and spec.submodule_search_locations else "")
PY
)"
       if [ -n "$NVDIR" ]; then
         for d in cublas cudnn cufft curand cusolver cusparse nccl cuda_runtime cuda_nvrtc cuda_cupti; do
           if [ -d "$NVDIR/$d/lib" ]; then
             export LD_LIBRARY_PATH="$NVDIR/$d/lib:${LD_LIBRARY_PATH:-}"
           fi
         done
         if [ -d "$NVDIR/lib" ]; then
             export LD_LIBRARY_PATH="$NVDIR/lib:${LD_LIBRARY_PATH:-}"
         fi
       fi

       export NCCL_P2P_DISABLE=0
       export NCCL_IB_DISABLE=1
       export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
       export TF_CPP_MIN_LOG_LEVEL=2

       # Optional sanity check
       python - <<'"'"'PY'"'"'
import torch
print("Torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
PY

       cd /scratch/project_2002147/dongli/softcfg/RobusTok

       python -u evaluator.py \
              /scratch/project_2002147/dongli/UltraDet/alitok/checkpoints/VIRTUAL_imagenet256_labeled.npz \
              output/robus_l_soft_cfg_scale_5.5_pow_1.01.npz
     '
