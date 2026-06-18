from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TumorClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pool: str = "masked",          # "masked" | "gap"
        extra_feat_dim: int = 0,       # размер внешних признаков (радиомика); 0 = нет
    ):
        super().__init__()
        if pool not in ("masked", "gap"):
            raise ValueError(f"pool must be 'masked' or 'gap', got {pool!r}")
        self.pool = pool
        self.extra_feat_dim = extra_feat_dim

        in_dim = embed_dim + extra_feat_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _masked_pool(embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Среднее эмбеддинга по вокселям маски.

        embedding: (B, C, d, h, w)
        mask:      (B, 1, D, H, W) или (B, 1, d, h, w) — приводится к разрешению эмбеддинга.
        return:    (B, C)
        """
        b, c, d, h, w = embedding.shape
        if mask.shape[2:] != (d, h, w):
            mask = F.interpolate(mask.float(), size=(d, h, w), mode="trilinear",
                                 align_corners=False)
        mask = (mask > 0.5).float()                      # (B,1,d,h,w)
        denom = mask.flatten(2).sum(-1).clamp_min(1.0)   # (B,1) — защита от пустой маски
        pooled = (embedding * mask).flatten(2).sum(-1) / denom  # (B,C)
        # fallback: если маска пустая, берём глобальный average pooling
        empty = (mask.flatten(2).sum(-1).squeeze(1) < 1.0)
        if empty.any():
            gap = embedding.flatten(2).mean(-1)          # (B,C)
            pooled[empty] = gap[empty]
        return pooled

    def forward(
        self,
        embedding: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """embedding: (B,C,d,h,w); mask: (B,1,...) для pool='masked'.

        return: logits (B, num_classes).
        """
        if self.pool == "masked":
            if mask is None:
                raise ValueError("pool='masked' требует mask (B,1,D,H,W).")
            feat = self._masked_pool(embedding, mask)
        else:
            feat = embedding.flatten(2).mean(-1)         # global average pooling

        if self.extra_feat_dim > 0:
            if extra_feat is None:
                raise ValueError(f"extra_feat_dim={self.extra_feat_dim}, но extra_feat=None.")
            feat = torch.cat([feat, extra_feat], dim=1)

        return self.mlp(feat)
