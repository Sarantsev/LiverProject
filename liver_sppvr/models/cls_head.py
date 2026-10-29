"""Classification head: image + clinical, fused so that BOTH modalities actually learn.

Why this is not a plain concatenation. Measured on MCT-LTDiag: a logistic regression on
the clinical features alone separates ICC from HCC at 0.77 accuracy, while the jointly
trained network reached ICC sensitivity 0.39 -- it does not learn to read features that are
trivially readable. The cause is a gradient imbalance: the image branch carries ~2.4M
trainable parameters and drives the training loss to 0.0000 by epoch ~27, after which the
few thousand clinical weights receive almost no gradient. The stronger modality starves the
weaker one, a standard failure of joint multimodal training.

Three mechanisms address it, all inside ONE end-to-end model (no ensembling, no blending,
no post-hoc thresholds):

1. Auxiliary heads. Each modality gets its own classifier and its own loss term, so the
   clinical branch must be predictive on its own and receives gradient regardless of what
   the image branch has memorised. Only the fused head is used at inference.
2. Modality dropout. During training each branch is independently zeroed per sample, so
   the model cannot rely on one modality alone. Never both at once.
3. Gated fusion (Gated Multimodal Unit, Arevalo et al. 2017) instead of concatenation: a
   learned gate decides per patient how much weight each modality gets. Besides accuracy
   this is interpretable -- the gate value says which modality drove a given prediction.

``head='concat'`` restores the previous raw-concatenation behaviour, which checkpoints
trained before this change require.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("gated", "concat")
POOLS = ("masked", "masked_std", "gap")


class TumorClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pool: str = "masked",           # "masked" | "masked_std" | "gap"
        extra_feat_dim: int = 0,        # clinical/tabular width; 0 = image only
        clinical_hidden: int = 64,      # tabular encoder width (concat head: 0 = raw)
        head: str = "gated",            # "gated" (GMU) | "concat" (legacy)
        fuse_dim: int = 256,            # width of the fused representation (gated head)
        img_drop: float = 0.3,          # p(zero the image branch) on train
        clin_drop: float = 0.15,        # p(zero the clinical branch) on train
        proj_dim: int = 128,            # projection for the supervised contrastive loss
    ):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        self.pool = pool
        self.head_kind = head
        self.extra_feat_dim = extra_feat_dim
        self.img_drop, self.clin_drop = img_drop, clin_drop
        # pool='masked_std' concatenates the second moment, so the IMAGE vector is 2C wide
        # while the backbone still produces C channels. Everything downstream of the pooling
        # must size itself on img_dim, not on embed_dim.
        img_dim = embed_dim * (2 if pool == "masked_std" else 1)
        self.img_dim = img_dim

        # --- tabular encoder (no input LayerNorm: the columns are already z-scored on the
        # training fold, and normalising ACROSS them would mix unrelated units) ---
        self.clin = None
        if extra_feat_dim > 0 and clinical_hidden > 0:
            self.clin = nn.Sequential(
                nn.Linear(extra_feat_dim, clinical_hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(clinical_hidden, clinical_hidden), nn.GELU(),
            )
        clin_out = clinical_hidden if self.clin is not None else extra_feat_dim

        if head == "concat":                         # legacy: one MLP over [image, clinical]
            in_dim = img_dim + clin_out
            self.fuse_dim = in_dim
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        else:                                        # GMU
            self.fuse_dim = fuse_dim
            self.img_proj = nn.Sequential(nn.LayerNorm(img_dim),
                                          nn.Linear(img_dim, fuse_dim), nn.Tanh())
            if clin_out:
                self.clin_proj = nn.Sequential(nn.Linear(clin_out, fuse_dim), nn.Tanh())
                self.gate = nn.Linear(img_dim + clin_out, fuse_dim)
            self.mlp = nn.Sequential(
                nn.LayerNorm(fuse_dim), nn.Linear(fuse_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
            # auxiliary classifiers: each modality must stand on its own
            self.aux_img = nn.Linear(img_dim, num_classes)
            self.aux_clin = nn.Linear(clin_out, num_classes) if clin_out else None

        self.proj = nn.Sequential(nn.Linear(self.fuse_dim, self.fuse_dim), nn.GELU(),
                                  nn.Linear(self.fuse_dim, proj_dim))
        self.last_gate = None            # mean gate value per sample, for interpretability

    # ------------------------------------------------------------------ pooling
    def _masked_pool(self, embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool the embedding over the mask voxels. embedding (B,C,d,h,w), mask (B,1,...).

        pool='masked'     -> mean over the tumour cells            (B,C)
        pool='masked_std' -> [mean, std] over the tumour cells      (B,2C)

        The second moment is not decoration. Necrosis, the peripheral nodular filling of a
        haemangioma and the mosaic pattern of HCC are VARIANCE phenomena INSIDE the lesion: the
        lesion is diagnostic because it is non-uniform. A mean over the mask is blind to them in
        exactly the way a convex combination of phases is blind to a phase difference (see
        models/multiphase.py) -- one level down, over voxels instead of over phases.
        """
        b, c, d, h, w = embedding.shape
        if mask.shape[2:] != (d, h, w):
            # max-pool to the embedding grid: a cell is tumour if ANY fine voxel in it is
            # (trilinear + threshold would drop small tumours into a silent GAP fallback)
            mask = F.adaptive_max_pool3d(mask.float(), (d, h, w))
        mask = (mask > 0.5).float()
        n = mask.flatten(2).sum(-1)                               # (B,1) tumour cells
        denom = n.clamp_min(1.0)
        # float32 accumulation: summing ~2k cells in fp16 loses precision, and the mask being
        # float32 promotes the product anyway. torch.where (rather than an indexed write) keeps
        # the fallback in the same dtype -- an fp16 source into an fp32 destination made the
        # former index_put_ raise under AMP, since the fusion projection ends in a Conv3d.
        x = embedding.float().flatten(2)
        m = mask.flatten(2)
        mean = (x * m).sum(-1) / denom
        empty = (n.squeeze(1) < 1.0)[:, None]                     # empty mask -> GAP fallback
        mean = torch.where(empty, x.mean(-1), mean)
        if self.pool == "masked":
            return mean
        # E[x^2] - E[x]^2 over the same cells; clamp because catastrophic cancellation can
        # push a near-uniform lesion a hair below zero.
        var = (x.pow(2) * m).sum(-1) / denom - mean.pow(2)
        std = var.clamp_min(0.0).sqrt()
        std = torch.where(empty, x.std(-1, unbiased=False), std)
        return torch.cat([mean, std], dim=1)

    def _modality_dropout(self, img, clin):
        """Zero each branch independently per sample (train only), never both at once."""
        if not self.training or clin is None:
            return img, clin
        B = img.shape[0]
        dev = img.device
        drop_i = torch.rand(B, device=dev) < self.img_drop
        drop_c = torch.rand(B, device=dev) < self.clin_drop
        both = drop_i & drop_c
        drop_c = drop_c & ~both                                   # keep at least one modality
        return img * (~drop_i).float()[:, None], clin * (~drop_c).float()[:, None]

    # ------------------------------------------------------------------ forward
    def forward(self, embedding, mask=None, extra_feat=None, return_extras: bool = False,
                return_proj: bool = False):
        """embedding (B,C,d,h,w) -> logits (B,num_classes).

        return_extras: also return {'proj', 'aux_img', 'aux_clin', 'gate'} (training).
        return_proj:   legacy shorthand returning (logits, proj).
        """
        if self.pool == "gap":
            img = embedding.flatten(2).mean(-1)
        else:
            if mask is None:
                raise ValueError(f"pool={self.pool!r} requires a mask (B,1,D,H,W).")
            img = self._masked_pool(embedding, mask)

        clin = None
        if self.extra_feat_dim > 0:
            if extra_feat is None:
                raise ValueError(f"extra_feat_dim={self.extra_feat_dim} but extra_feat=None.")
            e = extra_feat.float()
            clin = self.clin(e) if self.clin is not None else e

        img, clin = self._modality_dropout(img, clin)
        extras: dict = {}

        if self.head_kind == "concat":
            feat = img if clin is None else torch.cat([img, clin], dim=1)
        else:
            hi = self.img_proj(img)
            if clin is None:
                feat = hi
            else:
                hc = self.clin_proj(clin)
                z = torch.sigmoid(self.gate(torch.cat([img, clin], dim=1)))
                feat = z * hi + (1 - z) * hc
                self.last_gate = z.mean(1).detach()               # 1 = image, 0 = clinical
                extras["gate"] = self.last_gate
            if return_extras:
                extras["aux_img"] = self.aux_img(img)
                if self.aux_clin is not None:
                    extras["aux_clin"] = self.aux_clin(clin)

        logits = self.mlp(feat)
        if return_proj and not return_extras:
            return logits, F.normalize(self.proj(feat), dim=1)
        if return_extras:
            extras["proj"] = F.normalize(self.proj(feat), dim=1)
            return logits, extras
        return logits
