"""Losses for multi-task training (device-agnostic, no hard-coded .cuda()).

Segmentation: Dice + BCEWithLogits (as in SegVol).
Classification: Focal/CE with class-weight support (important due to the imbalance
of the rare classes BCLM/HH).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Binary segmentation. logits and target: (B,1,D,H,W) or broadcast-compatible."""
    logits = logits.float()
    target = target.float()
    prob = torch.sigmoid(logits)
    p = prob.contiguous().view(prob.shape[0], -1)
    t = target.contiguous().view(target.shape[0], -1)
    num = 2 * (p * t).sum(1) + smooth
    den = p.sum(1) + t.sum(1) + smooth
    dice = 1 - num / den
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice.mean() + bce


def focal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-class focal loss. logits: (B,C); target: (B,) long.

    gamma=0 -> ordinary weighted cross-entropy.
    """
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")  # (B,)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        focal_gamma: float = 2.0,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.focal_gamma = focal_gamma
        self.register_buffer("class_weight", class_weight if class_weight is not None
                             else torch.empty(0))

    def forward(self, out: dict, target: dict) -> dict:
        """out: {'seg_logits'?, 'cls_logits'?}; target: {'mask', 'label'}."""
        cw = self.class_weight if self.class_weight.numel() > 0 else None
        seg = torch.zeros((), device=_pick_device(out))
        cls = torch.zeros((), device=_pick_device(out))
        if "seg_logits" in out and out["seg_logits"] is not None:
            seg = dice_bce_loss(out["seg_logits"], target["mask"])
        if "cls_logits" in out and out["cls_logits"] is not None:
            cls = focal_ce_loss(out["cls_logits"], target["label"],
                                gamma=self.focal_gamma, weight=cw)
        total = self.seg_weight * seg + self.cls_weight * cls
        return {"loss": total, "seg_loss": seg.detach(), "cls_loss": cls.detach()}


def _pick_device(out: dict) -> torch.device:
    for v in out.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
