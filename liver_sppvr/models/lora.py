"""Parameter-efficient fine-tuning for nn.Linear -- LoRA and DoRA.

No external dependencies (torch 1.13 compatible). Freezes the base weights of the
selected Linear layers and adds a small trainable adaptation. Lets the SegVol ViT
encoder adapt to the CT domain while training a fraction of a percent of the
parameters: less overfitting than full fine-tuning, more capacity than a full freeze.

- LoRA: y = base(x) + scaling * (B @ A) x
- DoRA (Liu et al. 2024, "Weight-Decomposed Low-Rank Adaptation"): decomposes the
  weight into magnitude m (per output row) and direction, applies LoRA to the
  direction and trains m separately. Consistently closer to full fine-tuning than
  LoRA at the same rank; slightly more compute (forms the combined weight per step).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrapper around nn.Linear: y = base(x) + (dropout(x) @ A^T @ B^T) * scaling."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # base frozen
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B = 0 -> at init the output exactly equals the base (safe initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


class DoRALinear(nn.Module):
    """Weight-Decomposed LoRA. W' = m * (W0 + s*B@A) / ||W0 + s*B@A||_row.

    m (magnitude, one per output row) is initialized to the base weight's row norms,
    so at init W' == W0 (lora_B=0). Trainable: lora_A, lora_B, magnitude.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # base frozen
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        out_f, in_f = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # magnitude = per-output-row L2 norm of the base weight (-> W'=W0 at init)
        self.magnitude = nn.Parameter(base.weight.detach().norm(p=2, dim=1))  # (out,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * (self.lora_B @ self.lora_A)          # (out, in)
        weight = self.base.weight + delta
        norm = weight.norm(p=2, dim=1) + 1e-8                       # (out,)
        scale = (self.magnitude / norm).unsqueeze(0)               # (1, out)
        # y = scale ⊙ (x@W0^T + s*lora(x)) + bias ; base(x)-bias gives x@W0^T
        base_out = self.base(x)
        b = self.base.bias
        linear_part = base_out if b is None else base_out - b
        lora_part = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        y = scale * (linear_part + lora_part)
        return y if b is None else y + b


_VARIANTS = {"lora": LoRALinear, "dora": DoRALinear}


def apply_lora(root: nn.Module, targets: Sequence[str], *, rank: int = 8,
               alpha: float = 16.0, dropout: float = 0.0, variant: str = "lora") -> int:
    """Wrap Linear layers of `root` whose name contains any of `targets` with LoRA/DoRA.

    variant: "lora" | "dora". Returns the number of wrapped layers. The base of `root`
    should be frozen beforehand. A target of "all" (or "*") wraps every Linear layer --
    the most adaptation capacity ("unfreezing") while staying parameter-efficient.
    If no layer matches, raises ValueError listing the available Linear names.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")
    cls = _VARIANTS[variant]
    targets = tuple(targets)
    wrap_all = any(t in ("all", "*") for t in targets)
    to_wrap = [name for name, mod in root.named_modules()
               if isinstance(mod, nn.Linear) and (wrap_all or any(t in name for t in targets))]
    if not to_wrap:
        linears = [n for n, m in root.named_modules() if isinstance(m, nn.Linear)]
        raise ValueError(
            f"{variant}: no Linear matched targets={targets}. "
            f"Available Linear names (sample): {linears[:24]}")
    for name in to_wrap:
        parent_path, attr = (name.rsplit(".", 1) if "." in name else ("", name))
        parent = root.get_submodule(parent_path) if parent_path else root
        base = getattr(parent, attr)
        setattr(parent, attr, cls(base, rank=rank, alpha=alpha, dropout=dropout))
    return len(to_wrap)
