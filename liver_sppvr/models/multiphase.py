"""Multi-phase CT fusion (project-specific addition).

SegVol natively takes a single-channel volume (one phase). We add fusion of the
4 phases (non-contrast / arterial / portal / delayed), since contrast dynamics are
the radiological basis for distinguishing HCC / ICC / metastases.

Two modes are supported:

1. "concat_stem" -- early voxel-level fusion: the phases (as channels) pass through
   a small 3D-conv stem and are collapsed into 1 channel, which is fed to the
   unchanged SegVol image_encoder. Cheapest option, SegVol left untouched.

2. "attention" -- feature-level fusion: each phase is encoded separately (the encoder
   is called outside this module), and here the phase embeddings are aggregated by a
   learned attention pooling over the phase axis. More expensive, but keeps the
   per-phase information in the features -> stronger for classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseFusion(nn.Module):
    def __init__(
        self,
        mode: str = "attention",
        n_phases: int = 4,
        embed_dim: int = 768,
    ):
        super().__init__()
        if mode not in ("concat_stem", "attention"):
            raise ValueError(f"mode must be 'concat_stem' or 'attention', got {mode!r}")
        self.mode = mode
        self.n_phases = n_phases
        self.embed_dim = embed_dim

        if mode == "concat_stem":
            # (B, n_phases, D, H, W) -> (B, 1, D, H, W)
            self.stem = nn.Sequential(
                nn.Conv3d(n_phases, 8, kernel_size=3, padding=1),
                nn.InstanceNorm3d(8),
                nn.GELU(),
                nn.Conv3d(8, 1, kernel_size=1),
            )
        else:  # attention: score phases from their embeddings
            self.score = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )

    # --- concat_stem mode ---
    def fuse_input(self, phases: torch.Tensor) -> torch.Tensor:
        """phases: (B, n_phases, D, H, W) -> (B, 1, D, H, W) to feed into SegVol."""
        if self.mode != "concat_stem":
            raise RuntimeError("fuse_input is only available in 'concat_stem' mode.")
        return self.stem(phases)

    # --- attention mode ---
    def fuse_embeddings(self, phase_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """phase_embeddings: (B, P, C, d, h, w) -> fused (B, C, d, h, w).

        Attention is computed from each phase's globally-averaged embedding;
        the phase weights (B, P) are also returned for interpretability.
        """
        if self.mode != "attention":
            raise RuntimeError("fuse_embeddings is only available in 'attention' mode.")
        b, p, c, d, h, w = phase_embeddings.shape
        gap = phase_embeddings.flatten(3).mean(-1)        # (B, P, C)
        scores = self.score(gap).squeeze(-1)              # (B, P)
        weights = F.softmax(scores, dim=1)                # (B, P)
        fused = (phase_embeddings * weights[:, :, None, None, None, None]).sum(1)  # (B,C,d,h,w)
        return fused, weights
