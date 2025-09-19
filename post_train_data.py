"""Sampling scripts for TiTok on ImageNet.


Copyright (2024) Bytedance Ltd. and/or its affiliates


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at


   http://www.apache.org/licenses/LICENSE-2.0


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Reference:
   https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""
"""
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:9999 sample_imagenet_rar.py config=configs/training/generator/rar.yaml \
   experiment.output_dir="rar_b" \
   experiment.generator_checkpoint="rar_b.bin" \
   model.generator.hidden_size=768 \
   model.generator.num_hidden_layers=24 \
   model.generator.num_attention_heads=16 \
   model.generator.intermediate_size=3072 \
   model.generator.randomize_temperature=1.0 \
   model.generator.guidance_scale=16.0 \
   model.generator.guidance_scale_pow=2.75
  
  


"""
import demo_util
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import os
import math
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from torchvision import transforms
from tqdm import tqdm
from dataset.build import build_dataset
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from dataset.augmentation import random_crop_arr, center_crop_arr
from pathlib import Path
from torch.utils.data import Subset


class FilenameSubset(Dataset):
   def __init__(self, base_ds, num_samples, seed=42):
       self.base_ds = base_ds
       if num_samples < 0:
           self.indices = list(range(len(base_ds)))
       else:
           g = torch.Generator().manual_seed(seed)
           self.indices = torch.randperm(len(base_ds), generator=g)[:num_samples].tolist()


       # 提前缓存路径
       self.paths = [base_ds.samples[i][0] for i in self.indices]


   def __getitem__(self, idx):
       real_idx = self.indices[idx]
       img, label = self.base_ds[real_idx]
       p = Path(self.paths[idx])            # 或 Path(self.base_ds.samples[real_idx][0])
       last_two = os.path.join(p.parent.name, p.name)
       return img, label, last_two


   def __len__(self):
       return len(self.indices)


def main():
   parser = argparse.ArgumentParser(description="Sampling script for TiTok on ImageNet")
   parser.add_argument("--dataset", type=str, default="imagenet", help="Name of the dataset to use (default: imagenet)")
   parser.add_argument("--sigma", type=float, default=0.7)
   parser.add_argument("--data-path", type=str)
   parser.add_argument("--global-seed", type=int, default=12345)
   parser.add_argument("--num-workers", type=int, default=12)
   parser.add_argument("--image_size", type=int, default=256)
   parser.add_argument("--num_samples", type=int, default=200000)
   parser.add_argument("--seed", type=int, default=42)
   args, remaining = parser.parse_known_args()


   config = demo_util.get_config_cli()
   num_fid_samples = 50000
   per_proc_batch_size = 50
   sample_folder_dir = config.experiment.output_dir
   seed = 42


   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.deterministic = False
   torch.set_grad_enabled(False)


   # setup DDP.
   dist.init_process_group("nccl")
   rank = dist.get_rank()
   world_size = dist.get_world_size()
   device = rank % torch.cuda.device_count()
   global_batch_size = world_size * per_proc_batch_size
   seed = seed + rank
   torch.manual_seed(seed)
   torch.cuda.set_device(device)
   print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


   # if rank == 0:
   #     # download the maskgit-vq tokenizer
   #     hf_hub_download(repo_id="fun-research/TiTok", filename=f"{config.model.vq_model.pretrained_tokenizer_weight}", local_dir="./")
   #     # download the rar generator weight
   #     hf_hub_download(repo_id="yucornetto/RAR", filename=f"{config.experiment.generator_checkpoint}", local_dir="./")
   dist.barrier()


   # maskgit-vq as tokenizer
   tokenizer = create_pretrained_tokenizer(config)
   generator = demo_util.get_rar_generator(config)
   tokenizer.to(device)
   generator.to(device)


   if rank == 0:
       os.makedirs(sample_folder_dir, exist_ok=True)
       print(f"Saving .png samples at {sample_folder_dir}")
   dist.barrier()


   # Setup data:
   transform = transforms.Compose([
       transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
   ])
   base_dataset = build_dataset(args, transform=transform)
   dataset = FilenameSubset(base_dataset, num_samples=args.num_samples, seed=args.seed)
   sampler = DistributedSampler(
       dataset,
       num_replicas=dist.get_world_size(),
       rank=rank,
       shuffle=True,
       seed=args.global_seed
   )
   loader = DataLoader(
       dataset,
       batch_size=per_proc_batch_size,
       shuffle=False,
       sampler=sampler,
       num_workers=args.num_workers,
       pin_memory=True,
       drop_last=False
   )
   print(f"Dataset contains {len(dataset):,} images ({args.data_path})")


   total = 0


   for x, y, fnames in tqdm(loader, desc="Sampling"):
       imgs = x.to(device, non_blocking=True)
       labels = y.to(device, non_blocking=True)


       samples = demo_util.sample_fn_with_tf(
           generator=generator,
           tokenizer=tokenizer,
           labels=labels,
           images=imgs,
           prob=args.sigma,
           randomize_temperature=config.model.generator.randomize_temperature,
           guidance_scale=config.model.generator.guidance_scale,
           guidance_scale_pow=config.model.generator.guidance_scale_pow,
           device=device
       )
       # Save samples to disk as individual .png files
       for sample, fname in zip(samples, fnames):
           save_path = os.path.join(sample_folder_dir, fname)
           os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保子目录存在
           Image.fromarray(sample).save(save_path.replace(".JPEG", ".png"))


       total += global_batch_size


   # Make sure all processes have finished saving their samples before attempting to convert to .npz
   # if rank == 0:
   #     create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
   #     print("Done.")
   dist.destroy_process_group()


if __name__ == "__main__":
   main()
