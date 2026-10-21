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
   token: the 4 phases attend to each other, then are mean-pooled.

4. "diff" -- per-phase embeddings AND their consecutive differences, concatenated and
   projected back. The default; see below for why the other three are inadequate.

Why "diff" exists. Modes 2 and 3 can only produce a CONVEX COMBINATION of the phase
embeddings: mode 2 is literally sum(w_p * E_p) with softmax weights, and mode 3 ends in
out.mean(1) over a softmax-weighted sum, which is again a non-negative mixture. No such
operation can represent a DIFFERENCE between phases -- yet the radiological basis for
every subtype in this dataset is exactly a difference:

    HCC          non-rim APHE then washout   -> E_arterial - E_portal is large positive
    ICC          progressive enhancement     -> E_delayed - E_arterial is positive
    haemangioma  peripheral nodular filling  -> a spatially structured difference
    metastasis   rim enhancement             -> differs from both of the above

A weighted average of the four phases destroys all of it and leaves only mean brightness.
Measured consequence: our imaging-only model reached accuracy 0.620 (4-class) against
0.706 for the dataset's own radiomics SVM, which concatenates per-phase features and can
therefore form any difference linearly. Mode "diff" gives the network the same ability:
the consecutive differences are supplied explicitly as extra streams, so washout is a
linear read-out rather than something the encoder must somehow infer.

Each stream (phase or difference) is normalised in its own GroupNorm group, because
differences have a systematically smaller scale than the raw embeddings and would
otherwise be swamped. The projection is a bottleneck (n_streams*C -> width -> C) to keep
the added parameter count near 1.6M: with 412 training patients per fold the image branch
already overfits (§6.3), so a full n_streams*C -> C conv at 4.1M was rejected.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ("concat_stem", "attention", "cross_attention", "diff")


class PhaseFusion(nn.Module):
    def __init__(
        self,
        mode: str = "diff",
        n_phases: int = 4,
        embed_dim: int = 768,
        n_heads: int = 8,
        width: int = 256,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.n_phases = n_phases
        self.embed_dim = embed_dim
        self.out_dim = embed_dim          # every mode projects back to the backbone width
        self.last_stream_weights = None   # (B, n_streams) relative magnitudes, "diff" mode

        if mode == "diff":
            # P raw phases + (P-1) consecutive differences
            self.n_streams = 2 * n_phases - 1
            ch = self.n_streams * embed_dim
            self.stream_norm = nn.GroupNorm(self.n_streams, ch)   # one group per stream
            self.proj = nn.Sequential(
                nn.Conv3d(ch, width, kernel_size=1), nn.GELU(),
                nn.Conv3d(width, embed_dim, kernel_size=1))
        elif mode == "concat_stem":
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
    def fuse_embeddings(self, phase_embeddings: torch.Tensor, phase_present=None):
        """phase_embeddings: (B, P, C, d, h, w) -> (fused (B, C, d, h, w), phase_weights (B, P)).

        phase_present: optional (B, P) 1/0 -- padded (absent) phases get zero weight and
        are excluded as attention keys, so a missing phase never contaminates the fusion.
        """
        if self.mode == "attention":
            return self._fuse_attention(phase_embeddings, phase_present)
        if self.mode == "cross_attention":
            return self._fuse_cross_attention(phase_embeddings, phase_present)
        if self.mode == "diff":
            return self._fuse_diff(phase_embeddings, phase_present)
        raise RuntimeError("fuse_embeddings is not available in 'concat_stem' mode.")

    def _fuse_diff(self, phase_embeddings: torch.Tensor, phase_present=None):
        """Phases and their consecutive differences as separate streams.

        Returns (fused (B,C,d,h,w), phase_weights (B,P)). The returned weights are the
        relative magnitude of each PHASE stream, not attention weights -- there is no
        attention here; the full per-stream magnitudes (phases and differences) are kept
        in self.last_stream_weights for interpretation.
        """
        b, p, c, d, h, w = phase_embeddings.shape
        streams = [phase_embeddings[:, i] for i in range(p)]
        streams += [phase_embeddings[:, i] - phase_embeddings[:, i - 1] for i in range(1, p)]
        x = torch.cat(streams, dim=1)                              # (B, n_streams*C, d,h,w)

        if phase_present is not None:
            # a difference is valid only when BOTH of its phases are real; an absent phase
            # is zero-filled upstream, so E_i - E_j would otherwise be a spurious signal
            ok = [phase_present[:, i] for i in range(p)]
            ok += [phase_present[:, i] * phase_present[:, i - 1] for i in range(1, p)]
            keep = torch.stack(ok, dim=1).to(x.dtype)              # (B, n_streams)
            x = x * keep.repeat_interleave(c, dim=1)[:, :, None, None, None]

        mag = x.detach().abs().flatten(2).mean(-1)                 # (B, n_streams*C)
        mag = mag.reshape(b, self.n_streams, c).mean(-1)           # (B, n_streams)
        self.last_stream_weights = mag / mag.sum(1, keepdim=True).clamp_min(1e-6)
        ph = mag[:, :p]
        weights = ph / ph.sum(1, keepdim=True).clamp_min(1e-6)     # (B, P), contract
        return self.proj(self.stream_norm(x)), weights

    def _fuse_attention(self, phase_embeddings: torch.Tensor, phase_present=None):
        b, p, c, d, h, w = phase_embeddings.shape
        gap = phase_embeddings.flatten(3).mean(-1)        # (B, P, C)
        scores = self.score(gap).squeeze(-1)              # (B, P)
        if phase_present is not None:
            scores = scores.masked_fill(phase_present < 0.5, float("-inf"))
        weights = F.softmax(scores, dim=1)                # (B, P)
        fused = (phase_embeddings * weights[:, :, None, None, None, None]).sum(1)
        return fused, weights

    def _fuse_cross_attention(self, phase_embeddings: torch.Tensor, phase_present=None):
        b, p, c, d, h, w = phase_embeddings.shape
        n = d * h * w
        # (B,P,C,d,h,w) -> (B*N, P, C): each spatial token is a length-P sequence of phases
        x = phase_embeddings.permute(0, 3, 4, 5, 1, 2).reshape(b * n, p, c)
        x = self.norm(x + self.phase_emb[None])           # add phase positional embedding
        qkv = self.qkv(x).reshape(b * n, p, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)              # each (B*N, heads, P, head_dim)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B*N, heads, P, P)
        if phase_present is not None:                     # absent phases are not valid keys
            key_ok = phase_present.repeat_interleave(n, dim=0)[:, None, None, :] > 0.5
            attn = attn.masked_fill(~key_ok, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b * n, p, c)       # (B*N, P, C)
        out = self.out_proj(out)
        fused = out.mean(1).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # (B,C,d,h,w)
        # interpretability: average attention received by each phase (over heads, queries, tokens)
        weights = attn.mean(dim=(1, 2)).reshape(b, n, p).mean(1)    # (B, P)
        return fused, weights
