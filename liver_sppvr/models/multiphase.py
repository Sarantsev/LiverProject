"""Multi-phase CT fusion (project-specific addition).

SegVol natively takes a single-channel volume (one phase). We add fusion of the
4 phases (non-contrast / arterial / portal / delayed), since contrast dynamics are
the radiological basis for distinguishing HCC / ICC / metastases.

Modes:

1. "concat_stem" -- early voxel-level fusion: the phases (as channels) pass through a
   small 3D-conv stem and collapse into 1 channel fed to the unchanged SegVol encoder.

2. "attention" -- feature-level attention *pooling*: each phase is scored from its
   globally-averaged embedding (one scalar weight per phase) and the phase embeddings
   are combined by a weighted sum. Cheap, but phases do not interact.

3. "cross_attention" -- multi-head self-attention across the phase axis at each spatial
   token: the 4 phases attend to each other (contrast dynamics become explicit), then
   are mean-pooled. Richer than "attention"; a bit more compute.
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
        n_heads: int = 8,
    ):
        super().__init__()
        if mode not in ("concat_stem", "attention", "cross_attention"):
            raise ValueError(f"mode must be 'concat_stem'|'attention'|'cross_attention', got {mode!r}")
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
        elif mode == "attention":
            self.score = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )
        else:  # cross_attention: MHSA across phases at each spatial token
            assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
            self.n_heads = n_heads
            self.head_dim = embed_dim // n_heads
            self.phase_emb = nn.Parameter(torch.zeros(n_phases, embed_dim))  # which phase is which
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.norm = nn.LayerNorm(embed_dim)

    # --- concat_stem mode ---
    def fuse_input(self, phases: torch.Tensor) -> torch.Tensor:
        """phases: (B, n_phases, D, H, W) -> (B, 1, D, H, W) to feed into SegVol."""
        if self.mode != "concat_stem":
            raise RuntimeError("fuse_input is only available in 'concat_stem' mode.")
        return self.stem(phases)

    # --- attention / cross_attention modes ---
    def fuse_embeddings(self, phase_embeddings: torch.Tensor):
        """phase_embeddings: (B, P, C, d, h, w) -> (fused (B, C, d, h, w), phase_weights (B, P))."""
        if self.mode == "attention":
            return self._fuse_attention(phase_embeddings)
        if self.mode == "cross_attention":
            return self._fuse_cross_attention(phase_embeddings)
        raise RuntimeError("fuse_embeddings requires 'attention' or 'cross_attention' mode.")

    def _fuse_attention(self, phase_embeddings: torch.Tensor):
        b, p, c, d, h, w = phase_embeddings.shape
        gap = phase_embeddings.flatten(3).mean(-1)        # (B, P, C)
        scores = self.score(gap).squeeze(-1)              # (B, P)
        weights = F.softmax(scores, dim=1)                # (B, P)
        fused = (phase_embeddings * weights[:, :, None, None, None, None]).sum(1)
        return fused, weights

    def _fuse_cross_attention(self, phase_embeddings: torch.Tensor):
        b, p, c, d, h, w = phase_embeddings.shape
        n = d * h * w
        # (B,P,C,d,h,w) -> (B*N, P, C): each spatial token is a length-P sequence of phases
        x = phase_embeddings.permute(0, 3, 4, 5, 1, 2).reshape(b * n, p, c)
        x = self.norm(x + self.phase_emb[None])           # add phase positional embedding
        qkv = self.qkv(x).reshape(b * n, p, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)              # each (B*N, heads, P, head_dim)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B*N, heads, P, P)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b * n, p, c)       # (B*N, P, C)
        out = self.out_proj(out)
        fused = out.mean(1).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # (B,C,d,h,w)
        # interpretability: average attention received by each phase (over heads, queries, tokens)
        weights = attn.mean(dim=(1, 2)).reshape(b, n, p).mean(1)    # (B, P)
        return fused, weights
