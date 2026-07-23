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


def supcon_loss(z, labels, queue_z=None, queue_labels=None, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al.). z: (B, D) L2-normalized projections;
    labels: (B,). Optional detached memory queue (queue_z, queue_labels) provides extra
    positives/negatives -- important because our batch is small (often 0 same-class pairs).

    Pulls same-class projections together, pushes different-class apart.
    """
    B = z.shape[0]
    if queue_z is not None and queue_z.shape[0] > 0:
        cand_z = torch.cat([z, queue_z], dim=0)              # (B+Q, D)
        cand_labels = torch.cat([labels, queue_labels], dim=0)
    else:
        cand_z, cand_labels = z, labels
    sim = (z @ cand_z.t()) / temperature                     # (B, B+Q)
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()     # numerical stability
    self_mask = torch.zeros_like(sim)
    self_mask[:, :B] = torch.eye(B, device=z.device)         # exclude anchor==candidate
    pos_mask = (labels.view(-1, 1) == cand_labels.view(1, -1)).float() - self_mask
    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(1)
    valid = pos_count > 0                                     # anchors with >=1 positive
    if valid.sum() == 0:
        return z.sum() * 0.0                                 # no positives available yet
    loss = -(pos_mask * log_prob).sum(1) / pos_count.clamp(min=1)
    return loss[valid].mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        focal_gamma: float = 2.0,
        class_weight: Optional[torch.Tensor] = None,
        con_weight: float = 0.0,          # supervised contrastive weight (0 = off)
        con_temp: float = 0.1,
        con_dim: int = 128,
        queue_size: int = 512,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.focal_gamma = focal_gamma
        self.con_weight = con_weight
        self.con_temp = con_temp
        self.register_buffer("class_weight", class_weight if class_weight is not None
                             else torch.empty(0))
        if con_weight > 0:               # feature memory queue (labels -1 = empty slot)
            self.register_buffer("q_feat", torch.zeros(queue_size, con_dim))
            self.register_buffer("q_labels", torch.full((queue_size,), -1, dtype=torch.long))
            self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, feat, labels):
        Q = self.q_feat.shape[0]
        feat, labels = feat[-Q:].detach(), labels[-Q:]
        b = feat.shape[0]
        ptr = int(self.q_ptr)
        idx = (torch.arange(b, device=feat.device) + ptr) % Q
        self.q_feat[idx] = feat
        self.q_labels[idx] = labels
        self.q_ptr[0] = (ptr + b) % Q

    def forward(self, out: dict, target: dict) -> dict:
        """out: {'seg_logits'?, 'cls_logits'?, 'cls_proj'?}; target: {'mask', 'label'}."""
        cw = self.class_weight if self.class_weight.numel() > 0 else None
        dev = _pick_device(out)
        seg = torch.zeros((), device=dev)
        cls = torch.zeros((), device=dev)
        con = torch.zeros((), device=dev)
        if "seg_logits" in out and out["seg_logits"] is not None:
            seg = dice_bce_loss(out["seg_logits"], target["mask"])
        if "cls_logits" in out and out["cls_logits"] is not None:
            cls = focal_ce_loss(out["cls_logits"], target["label"],
                                gamma=self.focal_gamma, weight=cw)
        if self.con_weight > 0 and out.get("cls_proj") is not None:
            z, labels = out["cls_proj"], target["label"]
            valid_q = self.q_labels >= 0
            con = supcon_loss(z, labels, self.q_feat[valid_q], self.q_labels[valid_q],
                              temperature=self.con_temp)
            self._enqueue(z, labels)
        total = self.seg_weight * seg + self.cls_weight * cls + self.con_weight * con
        return {"loss": total, "seg_loss": seg.detach(), "cls_loss": cls.detach(),
                "con_loss": con.detach()}


def _pick_device(out: dict) -> torch.device:
    for v in out.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
