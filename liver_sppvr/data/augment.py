"""Lightweight 3D augmentations for multi-phase training (torch-only, cheap on CPU).

Geometry (flips) is applied IDENTICALLY to all phases and to the mask; intensity
transforms apply to images only. Rotation/elastic are deliberately omitted:
grid_sample is expensive on CPU, and data loading is already the bottleneck at
num_workers>0. Flips + intensity jitter give most of the anti-overfit benefit almost
for free.

Note: intensity ops and the [0,1] clamp assume phases are in [0,1] (normalize='hu').
With normalize='foreground' (z-score) the intensity block should be revisited.
"""
from __future__ import annotations

import torch


def augment_multiphase(
    phases: torch.Tensor,
    mask: torch.Tensor,
    *,
    p_flip: float = 0.5,
    brightness: float = 0.1,
    contrast: float = 0.1,
    noise_std: float = 0.02,
):
    """phases: (P,1,D,H,W) in [0,1], mask: (1,D,H,W). -> (phases, mask)."""
    # present phases (absent ones arrive as zeros) -- do not "revive" them
    present = (phases.reshape(phases.shape[0], -1).amax(1) > 0).float().view(-1, 1, 1, 1, 1)

    # --- geometry: flips over H (dim=-2) and W (dim=-1), synchronized phases+mask ---
    for axis in (-2, -1):
        if torch.rand(()) < p_flip:
            phases = torch.flip(phases, dims=[axis])
            mask = torch.flip(mask, dims=[axis])

    # --- intensity: images only (one shift/gain for all phases) ---
    if contrast > 0:
        c = 1.0 + (torch.rand(()) * 2 - 1) * contrast      # [1-contrast, 1+contrast]
        phases = (phases - 0.5) * c + 0.5
    if brightness > 0:
        phases = phases + (torch.rand(()) * 2 - 1) * brightness
    if noise_std > 0:
        phases = phases + torch.randn_like(phases) * noise_std

    phases = phases.clamp(0.0, 1.0) * present               # restore zeros for absent phases
    return phases, mask
