"""Lightweight 3D augmentations for multi-phase training (torch-only, cheap on CPU).

Geometry (flips) is applied IDENTICALLY to all phases and to the mask; intensity
transforms apply to images only. Rotation/elastic are deliberately omitted:
grid_sample is expensive on CPU, and data loading is already the bottleneck at
num_workers>0. Flips + intensity jitter give most of the anti-overfit benefit almost
for free.

Intensity ops are normalization-aware (`clip01`):
  - clip01=True  ([0,1] / HU data): contrast centered at 0.5, then clamp to [0,1].
  - clip01=False (z-score / ForegroundNorm): contrast centered at 0 (the mean),
    no clamp -- clamping z-scores to [0,1] would destroy the normalization.
"""
from __future__ import annotations

import torch


def augment_multiphase(
    phases: torch.Tensor,
    mask: torch.Tensor,
    *,
    p_flip: float = 0.5,
    contrast: float = 0.1,
    brightness: float = 0.1,
    noise_std: float = 0.02,
    clip01: bool = False,
):
    """phases: (P,1,D,H,W), mask: (1,D,H,W). -> (phases, mask).

    clip01: True if the input is in [0,1] (normalize='hu'); False for z-score
    (normalize='foreground').
    """
    # present phases (absent ones arrive as all-zeros) -- do not "revive" them.
    # abs-based check is robust for both [0,1] and z-score data.
    present = (phases.abs().reshape(phases.shape[0], -1).amax(1) > 0).float().view(-1, 1, 1, 1, 1)

    # --- geometry: flips over H (dim=-2) and W (dim=-1), synchronized phases+mask ---
    for axis in (-2, -1):
        if torch.rand(()) < p_flip:
            phases = torch.flip(phases, dims=[axis])
            mask = torch.flip(mask, dims=[axis])

    # --- intensity: images only (one gain/shift for all phases) ---
    if contrast > 0:
        g = 1.0 + (torch.rand(()) * 2 - 1) * contrast      # [1-contrast, 1+contrast]
        center = 0.5 if clip01 else 0.0                    # contrast around the data mean
        phases = (phases - center) * g + center
    if brightness > 0:
        phases = phases + (torch.rand(()) * 2 - 1) * brightness
    if noise_std > 0:
        phases = phases + torch.randn_like(phases) * noise_std

    if clip01:
        phases = phases.clamp(0.0, 1.0)                    # only valid for [0,1] data
    phases = phases * present                              # restore zeros for absent phases
    return phases, mask
