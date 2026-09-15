"""LiverTumorClassifier = pretrained backbone + multi-phase fusion + classification head.

Input contract (identical to the SOTA baselines so one trainer serves all models):
    phases        (B, P, 1, D, H, W)   multi-phase CT (absent phases zero-filled)
    phase_present (B, P)               1 = real phase, 0 = padded
    extra_feat    (B, F) | None        clinical vector (early fusion)
    mask          (B, 1, D, H, W)|None tumour mask for masked pooling -- GT (semi-auto
                                       upper bound) or the segmentation model's prediction
                                       (fully automatic). Ignored when pool='gap'.
    -> logits (B, num_classes)  [+ L2-normalised projection when return_proj=True]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import Backbone
from .cls_head import TumorClassificationHead
from .multiphase import PhaseFusion


class LiverTumorClassifier(nn.Module):
    def __init__(self, backbone: Backbone, num_classes: int = 5, n_phases: int = 4,
                 fusion_mode: str = "attention", hidden_dim: int = 256, dropout: float = 0.3,
                 pool: str = "masked", extra_feat_dim: int = 0):
        super().__init__()
        self.backbone = backbone
        self.n_phases = n_phases
        self.phase_fusion = PhaseFusion(mode=fusion_mode, n_phases=n_phases,
                                        embed_dim=backbone.embed_dim)
        self.head = TumorClassificationHead(embed_dim=backbone.embed_dim, num_classes=num_classes,
                                            hidden_dim=hidden_dim, dropout=dropout, pool=pool,
                                            extra_feat_dim=extra_feat_dim)
        # autocast inside forward: with DataParallel the replica threads do not inherit the
        # caller's (thread-local) autocast context, so the trainer sets this flag instead.
        self.amp_in_forward = False
        self.last_phase_weights = None

    @property
    def uses_mask(self) -> bool:
        return self.head.pool == "masked"

    def encode_multiphase(self, phases: torch.Tensor, phase_present=None):
        """(B,P,1,D,H,W) -> fused (B,C,d,h,w), phase_weights (B,P)|None."""
        if phases.dim() == 5:
            phases = phases.unsqueeze(2)
        if self.phase_fusion.mode == "concat_stem":
            return self.backbone(self.phase_fusion.fuse_input(phases.squeeze(2))), None
        embs = torch.stack([self.backbone(phases[:, i]) for i in range(phases.shape[1])], dim=1)
        return self.phase_fusion.fuse_embeddings(embs, phase_present=phase_present)

    def forward(self, phases, phase_present=None, extra_feat=None, mask=None, return_proj=False):
        with torch.cuda.amp.autocast(enabled=self.amp_in_forward):
            fused, self.last_phase_weights = self.encode_multiphase(phases, phase_present)
            return self.head(fused, mask=mask, extra_feat=extra_feat, return_proj=return_proj)
