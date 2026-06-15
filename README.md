## Just image Transformer (JiT) for Pixel-space Diffusion

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2511.13720-b31b1b.svg)](https://arxiv.org/abs/2511.13720)&nbsp;

<p align="center">
  <img src="demo/visual.jpg" width="100%">
</p>


This is a PyTorch/GPU re-implementation of the paper [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720):

```
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

JiT adopts a minimalist and self-contained design for pixel-level high-resolution image diffusion. 
The original implementation was in JAX+TPU. This re-implementation is in PyTorch+GPU.

<p align="center">
  <img src="demo/jit.jpg" width="40%">
</p>

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/LTH14/JiT.git
cd JiT
```

A suitable [conda](https://conda.io/) environment named `jit` can be created and activated with:

```
conda env create -f environment.yaml
conda activate jit
```

If you get ```undefined symbol: iJIT_NotifyEvent``` when importing ```torch```, simply
```
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Check this [issue](https://github.com/conda/conda/issues/13812#issuecomment-2071445372) for more details.

### Preparing ModelScope ImageNet parquet

JiT reads ImageNet through `torchvision.datasets.ImageFolder` at `<data_path>/train`, so a parquet copy of ImageNet
must first be expanded to the standard synset-folder layout. If your ModelScope download contains parquet shards with
`image` and `label` columns plus the ImageNet `classes.py` file, convert it with:
```
python tools/convert_imagenet_parquet_to_imagefolder.py \
--parquet_dir ${MODELSCOPE_IMAGENET_PARQUET_DIR} \
--classes_py ${MODELSCOPE_IMAGENET_CLASSES_PY} \
--output_dir ${IMAGENET_PATH} \
--output_split train \
--parquet_glob "*.parquet"
```

The output will be written as `${IMAGENET_PATH}/train/n01440764/...`, `${IMAGENET_PATH}/train/n01443537/...`, and so
on, which can be passed directly to the training commands below with `--data_path ${IMAGENET_PATH}`. If one directory
contains multiple splits, restrict the input shards, for example with `--parquet_glob "train*.parquet"`. For a quick
smoke test, add `--limit 100 --overwrite`; for an interrupted full conversion, rerun with `--resume`.

### Training
The below training scripts have been tested on 8 H200 GPUs.

By default, JiT logs training loss, learning rate, FID, and IS to TensorBoard under `--output_dir`. The examples below
also enable Weights & Biases on the main process with `--wandb --wandb_project JiT`; optionally add
`--wandb_run_name ${RUN_NAME}`. Gradient clipping uses global norm clipping with `--grad_clip 1.0` by default; set
`--grad_clip 0.0` to disable clipping. Pre-clip gradient norm, post-clip gradient norm, and clip ratio are logged to
TensorBoard and W&B.

For token-matched Flux2 VAE latent-space ablations, keep the same `--img_size` as the pixel-space run and reduce the
JiT patch size by the Flux2/Klein latent downsampling factor of 16. For example, pixel-space `JiT-B/16` at 256x256
has a 16x16 image-token grid, and the matching latent-space model is `JiT-B/1`; pixel-space `JiT-B/32` at 512x512
also has a 16x16 image-token grid, and the matching latent-space model is `JiT-B/2`. The default latent mode uses
Flux2/Klein 2x2 patchified VAE latents with 128 channels. The latent-space commands below use `--noise_scale 1.0`
for the BN-normalized Flux2 VAE latent space.

Example script for training JiT-B/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --online_eval
```

Token-matched Flux2 VAE latent-space ablation for the same 256x256 image resolution and 16x16 image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-B/1 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --online_eval
```

Example script for training JiT-B/32 on ImageNet 512x512 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 512 --noise_scale 2.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --online_eval
```

Token-matched Flux2 VAE latent-space ablation for the same 512x512 image resolution and 16x16 image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-B/2 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 512 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --online_eval
```

Example script for training JiT-H/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --online_eval
```

Token-matched Flux2 VAE latent-space ablation for the same 256x256 image resolution and 16x16 image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-H/1 \
--proj_dropout 0.2 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --online_eval
```

### Flux2 VAE latent-space details

JiT can also train in the same Flux2 VAE latent space used by the Diffusers `Flux2KleinPipeline` instead of directly
denoising RGB pixels. Diffusers encodes images as:

1. normalize image pixels to `[-1, 1]`;
2. run `AutoencoderKLFlux2.encode(image).latent_dist.mode()` by default;
3. reshape the 32-channel VAE latent into Flux2/Klein `2x2` latent patches, producing 128 channels;
4. normalize those patchified latents with `vae.bn.running_mean` and `sqrt(vae.bn.running_var + eps)`.

Decoding performs the exact inverse BN denormalization and unpatchify before `vae.decode(...)`.

The default `--flux2_vae_sample_mode argmax` matches the Flux2/Klein image-conditioning path. Generated samples are
decoded back to pixel images before saving and FID/IS evaluation. Pixel-space checkpoints and Flux2 VAE latent-space
checkpoints are not interchangeable; use the matching evaluation command for the data space used during training.

### Evaluation

PyTorch pre-trained models are available [here](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm&st=ffkl77al&dl=0).

Evaluate pre-trained JiT-B:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 (or JiT-B/32) \
--img_size 256 (or 512) --noise_scale 1.0 (or 2.0) \
--gen_bsz 256 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate a token-matched Flux2 VAE latent-space JiT-B/1 checkpoint at 256x256, matching the `JiT-B/16` 16x16
image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-B/1 \
--img_size 256 --noise_scale 1.0 \
--gen_bsz 256 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate a token-matched Flux2 VAE latent-space JiT-B/2 checkpoint at 512x512, matching the `JiT-B/32` 16x16
image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-B/2 \
--img_size 512 --noise_scale 1.0 \
--gen_bsz 256 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate pre-trained JiT-L:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-L/16 (or JiT-L/32) \
--img_size 256 (or 512) --noise_scale 1.0 (or 2.0) \
--gen_bsz 256 --num_images 50000 --cfg 2.4 (or 2.5) --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate a token-matched Flux2 VAE latent-space JiT-L/1 checkpoint at 256x256, matching the `JiT-L/16` 16x16
image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-L/1 \
--img_size 256 --noise_scale 1.0 \
--gen_bsz 256 --num_images 50000 --cfg 2.4 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate a token-matched Flux2 VAE latent-space JiT-L/2 checkpoint at 512x512, matching the `JiT-L/32` 16x16
image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-L/2 \
--img_size 512 --noise_scale 1.0 \
--gen_bsz 256 --num_images 50000 --cfg 2.5 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate pre-trained JiT-H:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-H/16 (or JiT-H/32) \
--img_size 256 (or 512) --noise_scale 1.0 (or 2.0) \
--gen_bsz 256 --num_images 50000 --cfg 2.2 (or 2.3) --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate a token-matched Flux2 VAE latent-space JiT-H/1 checkpoint at 256x256, matching the `JiT-H/16` 16x16
image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-H/1 \
--img_size 256 --noise_scale 1.0 \
--gen_bsz 256 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate a token-matched Flux2 VAE latent-space JiT-H/2 checkpoint at 512x512, matching the `JiT-H/32` 16x16
image-token grid:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--data_space flux2vae \
--flux2_vae_path black-forest-labs/FLUX.2-klein-base-9B --flux2_vae_subfolder vae \
--model JiT-H/2 \
--img_size 512 --noise_scale 1.0 \
--gen_bsz 256 --num_images 50000 --cfg 2.3 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--wandb --wandb_project JiT \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

We use a customized [```torch-fidelity```](https://github.com/LTH14/torch-fidelity)
to evaluate FID and IS against a reference image folder or statistics. You can use ```prepare_ref.py```
to prepare the reference image folder, or directly use our pre-computed reference stats
under ```fid_stats```.

### Acknowledgements

We thank Google TPU Research Cloud (TRC) for granting us access to TPUs, and the MIT
ORCD Seed Fund Grants for supporting GPU resources.

### Contact

If you have any questions, feel free to contact me through email (tianhong@mit.edu). Enjoy!
