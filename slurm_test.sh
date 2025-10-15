#!/usr/bin/env bash
set -eux

srun -A project_2002147 \
     -p gpusmall \
     --gres=gpu:a100:2 \
     --cpus-per-task=16 \
     --mem=100G \
     -t 02:15:00 \
     bash -c "\
       source /scratch/project_2002147/dongli/miniconda3/etc/profile.d/conda.sh && \
       conda activate alitok && \
       cd /scratch/project_2002147/dongli/softcfg/RobusTok && \
       torchrun --nnodes=1 --nproc_per_node=2 \
         --rdzv-endpoint=localhost:19904 \
         sample_imagenet_rar.py \
         config=configs/generator/rar.yaml \
         experiment.output_dir=output/robus_l_soft_cfg_scale_6.5_pow_1.01 \
         experiment.generator_checkpoint=checkpoints/rar_l.bin \
         model.vq_ckpt=checkpoints/post-train.pt \
         model.generator.guidance_scale=6.5 \
         model.generator.guidance_scale_pow=1.01 \
         model.generator.softcfg_strength=1 \
         model.generator.step_norm=True"
