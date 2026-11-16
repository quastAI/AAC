"""The Sieve's pixel decoder and the patch geometry its losses and metrics share.

A frame is 63 x 63 pixels: 9 x 9 patches of 7 x 7 (the 7 map rows, then the 2 rows of the inventory strip), the ViT's
patches. The decoder reads only the latent `z` (after the projector), never the patch tokens: the content has to sit on
the latent the predictor predicts and the planner compares. It writes one 7x7x3 patch per position, so that the
error-weighting, the gate and the metrics can work per patch. Pixels are in [0, 1] (the encoder's ImageNet
normalisation undone); a patch error is the mean squared error over its 147 values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

PATCH = 7
GRID = 9
PATCHES = GRID * GRID
PATCH_DIM = PATCH * PATCH * 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_unit(pixels: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised frames (..., 3, H, W) back to [0, 1], in float32."""
    shape = [1] * pixels.ndim
    shape[-3] = 3
    mean = torch.tensor(IMAGENET_MEAN, device=pixels.device).view(shape)
    std = torch.tensor(IMAGENET_STD, device=pixels.device).view(shape)
    return pixels.float() * std + mean


def uint8_to_unit(frames) -> torch.Tensor:
    """uint8 frames (N, 63, 63, 3), numpy or torch, to float (N, 3, 63, 63) in [0, 1]."""
    x = torch.as_tensor(np.ascontiguousarray(frames))
    return x.permute(0, 3, 1, 2).float().div(255.0)


def patchify(frames: torch.Tensor) -> torch.Tensor:
    """(..., 3, 63, 63) -> (..., 81, 147); patch i is row i // 9, column i % 9 (the ViT's order)."""
    lead = frames.shape[:-3]
    x = frames.reshape(-1, 3, GRID * PATCH, GRID * PATCH)
    x = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)  # (N, 3, 9, 9, 7, 7)
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, PATCHES, PATCH_DIM)
    return x.reshape(*lead, PATCHES, PATCH_DIM)


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    """Inverse of `patchify`: (..., 81, 147) -> (..., 3, 63, 63)."""
    lead = patches.shape[:-2]
    x = patches.reshape(-1, GRID, GRID, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(*lead, 3, GRID * PATCH, GRID * PATCH)


def patch_errors(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error per patch, (..., 81), in float32."""
    return (decoded.float() - target.float()).pow(2).mean(-1)


class PatchDecoder(nn.Module):
    """z (D) -> 81 patches of 7x7x3: an MLP to 81 x `patch_dim`, a learned embedding per patch position, then one shared
    per-patch head (LayerNorm, GELU, linear to 147 values).

    The buffers `pixel_variance` (the mean per-pixel variance of the training frames, in [0, 1] units) and `tau` (the
    detail level of stage 2, `tau_scale` x that variance) are set once by the training (`set_pixel_variance`) and saved
    with the model, so the metrics read the same `tau` as the loss.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, patch_dim: int = 128, tau_scale: float = 0.03):
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.tau_scale = float(tau_scale)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, PATCHES * self.patch_dim),
        )
        self.position = nn.Parameter(torch.zeros(PATCHES, self.patch_dim))
        nn.init.normal_(self.position, std=0.02)
        self.head = nn.Sequential(nn.LayerNorm(self.patch_dim), nn.GELU(), nn.Linear(self.patch_dim, PATCH_DIM))
        self.register_buffer("pixel_variance", torch.tensor(float("nan")))
        self.register_buffer("tau", torch.tensor(float("nan")))

    def set_pixel_variance(self, variance: float) -> None:
        self.pixel_variance.fill_(float(variance))
        self.tau.fill_(self.tau_scale * float(variance))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., D) -> (..., 81, 147) patches in [0, 1] units (not clamped)."""
        lead = z.shape[:-1]
        h = self.trunk(z.reshape(-1, z.shape[-1])).reshape(-1, PATCHES, self.patch_dim) + self.position
        return self.head(h).reshape(*lead, PATCHES, PATCH_DIM)


def pixel_variance(h5_path, frames: int = 4096, run: int = 64, seed: int = 0) -> float:
    """Mean per-pixel variance (over frames, in [0, 1] units) of `frames` frames of a file, read as runs of `run`
    consecutive frames from random episodes (the pixels are gzip in chunks: one read per run, not per frame)."""
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(Path(h5_path), "r") as f:
        lengths, offsets = f["ep_len"][:].astype(np.int64), f["ep_offset"][:].astype(np.int64)
        pixels = f["pixels"]
        total, sq, n = np.zeros(pixels.shape[1:]), np.zeros(pixels.shape[1:]), 0
        eligible = np.flatnonzero(lengths >= 1)
        while n < frames:
            ep = int(rng.choice(eligible))
            take = int(min(run, lengths[ep], frames - n))
            start = int(offsets[ep] + rng.integers(0, lengths[ep] - take + 1))
            x = pixels[start : start + take].astype(np.float64) / 255.0
            total += x.sum(0)
            sq += (x**2).sum(0)
            n += take
    mean = total / n
    return float((sq / n - mean**2).clip(min=0).mean())
