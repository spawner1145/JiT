import sys
from pathlib import Path

import torch


def _ensure_local_diffusers_on_path():
    local_diffusers_src = Path(__file__).resolve().parent / "diffusers" / "src"
    if local_diffusers_src.exists():
        local_diffusers_src = str(local_diffusers_src)
        if local_diffusers_src not in sys.path:
            sys.path.insert(0, local_diffusers_src)


def _resolve_dtype(dtype_name, device):
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _retrieve_latents(encoder_output, generator=None, sample_mode="argmax"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class PixelSpace:
    name = "pixel"
    channels = 3
    vae_scale_factor = 1

    def __init__(self, img_size):
        self.size = img_size

    def encode(self, images):
        return images

    def decode(self, samples):
        return samples

    def __repr__(self):
        return f"PixelSpace(channels={self.channels}, size={self.size})"


class Flux2VAELatentSpace:
    name = "flux2vae"

    def __init__(self, args, device):
        _ensure_local_diffusers_on_path()
        from diffusers.models import AutoencoderKLFlux2

        self.device = device
        self.sample_mode = args.flux2_vae_sample_mode
        self.patchify = args.flux2_vae_patchify
        self.dtype = _resolve_dtype(args.flux2_vae_dtype, device)

        load_kwargs = {"torch_dtype": self.dtype}
        if args.flux2_vae_subfolder:
            load_kwargs["subfolder"] = args.flux2_vae_subfolder

        self.vae = AutoencoderKLFlux2.from_pretrained(args.flux2_vae_path, **load_kwargs)
        self.vae.requires_grad_(False)
        self.vae.eval().to(device)

        if args.flux2_vae_tiling:
            self.vae.enable_tiling()
        if args.flux2_vae_slicing:
            self.vae.enable_slicing()

        self.latent_channels = int(self.vae.config.latent_channels)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_patch_size = 2 if self.patchify else 1
        self.downsample_factor = self.vae_scale_factor * self.latent_patch_size

        if args.img_size % self.downsample_factor != 0:
            raise ValueError(
                f"Flux2 VAE latent space requires --img_size to be divisible by {self.downsample_factor}, "
                f"but got {args.img_size}."
            )

        self.size = args.img_size // self.downsample_factor
        self.channels = self.latent_channels * (self.latent_patch_size ** 2)

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels, height, width = latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Flux2 latent height/width must be even before patchify, got {height}x{width}.")
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels, height, width = latents.shape
        if num_channels % 4 != 0:
            raise ValueError(f"Flux2 patchified latent channels must be divisible by 4, got {num_channels}.")
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    def _bn_stats(self, latents):
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return mean, std

    @torch.no_grad()
    def encode(self, images):
        if images.ndim != 4:
            raise ValueError(f"Expected image tensor with shape (B, C, H, W), got {tuple(images.shape)}.")

        images = images.to(device=self.device, dtype=self.dtype)
        latents = _retrieve_latents(self.vae.encode(images), sample_mode=self.sample_mode)

        if self.patchify:
            latents = self._patchify_latents(latents)
            mean, std = self._bn_stats(latents)
            latents = (latents - mean) / std

        return latents.to(dtype=torch.float32)

    @torch.no_grad()
    def decode(self, samples):
        samples = samples.to(device=self.device, dtype=self.dtype)

        if self.patchify:
            mean, std = self._bn_stats(samples)
            samples = samples * std + mean
            samples = self._unpatchify_latents(samples)

        images = self.vae.decode(samples, return_dict=False)[0]
        return images.to(dtype=torch.float32)

    def __repr__(self):
        return (
            "Flux2VAELatentSpace("
            f"channels={self.channels}, size={self.size}, vae_scale_factor={self.vae_scale_factor}, "
            f"patchify={self.patchify}, sample_mode={self.sample_mode}, dtype={self.dtype})"
        )


def create_data_space(args, device):
    if args.data_space == "pixel":
        return PixelSpace(args.img_size)
    if args.data_space == "flux2vae":
        return Flux2VAELatentSpace(args, device)
    raise ValueError(f"Unsupported data space: {args.data_space}")
