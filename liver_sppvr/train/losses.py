"""Classification losses (device-agnostic).

Focal/CE with optional class weights (MCT-LTDiag classes are balanced, ~100 each, so the
inverse-frequency weights are near-unit; focal gamma mainly sharpens confusable types)
plus an optional supervised-contrastive term (SupCon) with a memory queue -- the batch
is tiny, so the queue supplies the same-class positives.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_ce_loss(logits, target, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
    """Multi-class focal loss. logits (B,C); target (B,). gamma=0 -> weighted CE."""
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def supcon_loss(z, labels, queue_z=None, queue_labels=None, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al.). z: (B,D) L2-normalised; labels (B,).
    A detached queue (queue_z, queue_labels) adds candidates; anchors with no positive
    are skipped."""
    B = z.shape[0]
    if queue_z is not None and queue_z.shape[0] > 0:
        cand_z = torch.cat([z, queue_z], dim=0)
        cand_labels = torch.cat([labels, queue_labels], dim=0)
    else:
        cand_z, cand_labels = z, labels
    sim = (z @ cand_z.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    self_mask = torch.zeros_like(sim)
    self_mask[:, :B] = torch.eye(B, device=z.device)
    pos_mask = (labels.view(-1, 1) == cand_labels.view(1, -1)).float() - self_mask
    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    loss = -(pos_mask * log_prob).sum(1) / pos_count.clamp(min=1)
    return loss[valid].mean()


class ClsLoss(nn.Module):
    """focal-CE (+ supcon_weight * SupCon with a feature queue). forward -> dict of terms."""
    def __init__(self, class_weight: Optional[torch.Tensor] = None, focal_gamma: float = 2.0,
                 supcon_weight: float = 0.0, supcon_temp: float = 0.1, proj_dim: int = 128,
                 queue_size: int = 512):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.supcon_weight = supcon_weight
        self.supcon_temp = supcon_temp
        self.register_buffer("class_weight", class_weight if class_weight is not None
                             else torch.empty(0))
        if supcon_weight > 0:                    # label -1 = empty slot
            self.register_buffer("q_feat", torch.zeros(queue_size, proj_dim))
            self.register_buffer("q_labels", torch.full((queue_size,), -1, dtype=torch.long))
            self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, feat, labels):
        Q = self.q_feat.shape[0]
        feat, labels = feat[-Q:].detach(), labels[-Q:]
        idx = (torch.arange(feat.shape[0], device=feat.device) + int(self.q_ptr)) % Q
        self.q_feat[idx] = feat.float(); self.q_labels[idx] = labels
        self.q_ptr[0] = (int(self.q_ptr) + feat.shape[0]) % Q

    def forward(self, logits, labels, proj=None) -> dict:
        cw = self.class_weight if self.class_weight.numel() > 0 else None
        cls = focal_ce_loss(logits.float(), labels, gamma=self.focal_gamma, weight=cw)
        con = torch.zeros((), device=logits.device)
        if self.supcon_weight > 0 and proj is not None:
            ok = self.q_labels >= 0
            con = supcon_loss(proj.float(), labels, self.q_feat[ok], self.q_labels[ok],
                              temperature=self.supcon_temp)
            self._enqueue(proj, labels)
        return {"loss": cls + self.supcon_weight * con, "cls_loss": cls.detach(),
                "con_loss": con.detach()}
