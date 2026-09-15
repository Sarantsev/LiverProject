"""Lightweight 3D augmentations (torch-only, cheap on CPU).

augment_multiphase: flips applied identically to all phases + mask; intensity jitter on
images only (normalization-aware via `clip01`). Rotation/elastic are omitted on purpose --
data loading is already the bottleneck.

perturb_mask: simulates segmentation errors on the tumour mask (dilate / erode / shift /
drop) so a classifier trained with masked pooling does not overfit to perfect masks and
degrades gracefully on the segmentation model's predictions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def augment_multiphase(phases, mask, *, p_flip: float = 0.5, contrast: float = 0.1,
                       brightness: float = 0.1, noise_std: float = 0.02, clip01: bool = False):
    """phases (P,1,D,H,W), mask (1,D,H,W) -> (phases, mask)."""
    present = (phases.abs().reshape(phases.shape[0], -1).amax(1) > 0).float().view(-1, 1, 1, 1, 1)
    for axis in (-2, -1):
        if torch.rand(()) < p_flip:
            phases = torch.flip(phases, dims=[axis]); mask = torch.flip(mask, dims=[axis])
    if contrast > 0:
        g = 1.0 + (torch.rand(()) * 2 - 1) * contrast
        center = 0.5 if clip01 else 0.0
        phases = (phases - center) * g + center
    if brightness > 0:
        phases = phases + (torch.rand(()) * 2 - 1) * brightness
    if noise_std > 0:
        phases = phases + torch.randn_like(phases) * noise_std
    if clip01:
        phases = phases.clamp(0.0, 1.0)
    return phases * present, mask


def _shift(m: torch.Tensor, shifts) -> torch.Tensor:
    """Shift (1,D,H,W) with zero fill (roll, then blank the wrapped slab)."""
    for ax, s in zip((1, 2, 3), shifts):
        if s == 0:
            continue
        m = torch.roll(m, shifts=int(s), dims=ax)
        idx = [slice(None)] * 4
        idx[ax] = slice(0, s) if s > 0 else slice(m.shape[ax] + s, None)
        m[tuple(idx)] = 0.0
    return m


def perturb_mask(mask: torch.Tensor, p: float = 0.5, max_shift: int = 4, max_kernel: int = 5,
                 p_drop: float = 0.1) -> torch.Tensor:
    """mask (1,D,H,W) binary. With probability p apply one of: dilate, erode, shift; with
    probability p_drop replace by an EMPTY mask (the head then falls back to GAP)."""
    if torch.rand(()) >= p:
        return mask
    if torch.rand(()) < p_drop:
        return torch.zeros_like(mask)
    op = int(torch.randint(0, 3, ()))
    m = mask[None]                                              # (1,1,D,H,W)
    if op in (0, 1):
        k = int(torch.randint(1, max_kernel // 2 + 1, ())) * 2 + 1
        if op == 0:                                             # dilate
            m = F.max_pool3d(m, k, stride=1, padding=k // 2)
        else:                                                   # erode
            m = 1.0 - F.max_pool3d(1.0 - m, k, stride=1, padding=k // 2)
            if m.sum() < 1:                                     # never erode away entirely
                m = mask[None]
        return m[0]
    shifts = [int(torch.randint(-max_shift, max_shift + 1, ())) for _ in range(3)]
    return _shift(mask.clone(), shifts)
