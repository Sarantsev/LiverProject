"""Minimal LoRA for nn.Linear -- no external dependencies (torch 1.13 compatible).

Freezes the base weights of the selected Linear layers and adds trainable low-rank
A/B matrices. Lets the SegVol ViT encoder adapt to the CT domain while training a
fraction of a percent of the parameters: less overfitting than full fine-tuning,
more capacity than a full freeze.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


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


def apply_lora(root: nn.Module, targets: Sequence[str], *, rank: int = 8,
               alpha: float = 16.0, dropout: float = 0.0) -> int:
    """Replace Linear layers of `root` whose name contains any of `targets` with LoRALinear.

    Returns the number of wrapped layers. The base of `root` should be frozen beforehand.
    If no layer matches, raises ValueError listing the available Linear names (to help
    pick `targets` for a given encoder architecture).
    """
    targets = tuple(targets)
    to_wrap = [name for name, mod in root.named_modules()
               if isinstance(mod, nn.Linear) and any(t in name for t in targets)]
    if not to_wrap:
        linears = [n for n, m in root.named_modules() if isinstance(m, nn.Linear)]
        raise ValueError(
            f"LoRA: no Linear matched targets={targets}. "
            f"Available Linear names (sample): {linears[:24]}")
    for name in to_wrap:
        parent_path, attr = (name.rsplit(".", 1) if "." in name else ("", name))
        parent = root.get_submodule(parent_path) if parent_path else root
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    return len(to_wrap)
