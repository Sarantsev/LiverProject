from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cls_head import TumorClassificationHead
from .multiphase import PhaseFusion


class SegVolMultiTask(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: Optional[nn.Module],
        mask_decoder: Optional[nn.Module],
        text_encoder: Optional[nn.Module],
        roi_size: Sequence[int] = (32, 256, 256),
        patch_size: Sequence[int] = (4, 16, 16),
        embed_dim: int = 768,
        num_classes: int = 5,
        cls_hidden_dim: int = 256,
        cls_dropout: float = 0.3,
        cls_pool: str = "masked",
        cls_extra_feat_dim: int = 0,
        fusion_mode: str = "attention",
        n_phases: int = 4,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.text_encoder = text_encoder

        self.feat_shape = (np.array(roi_size) / np.array(patch_size)).astype(int)  # (d,h,w)
        self.embed_dim = embed_dim
        self.n_phases = n_phases

        self.phase_fusion = PhaseFusion(mode=fusion_mode, n_phases=n_phases, embed_dim=embed_dim)
        self.cls_head = TumorClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=cls_hidden_dim,
            dropout=cls_dropout,
            pool=cls_pool,
            extra_feat_dim=cls_extra_feat_dim,
        )

    # ---------- фабрика из загруженной модели SegVol ----------
    @classmethod
    def from_segvol(cls, segvol_model: nn.Module, **kwargs) -> "SegVolMultiTask":
        """Собрать обёртку из компонентов уже инициализированной модели SegVol."""
        return cls(
            image_encoder=segvol_model.image_encoder,
            prompt_encoder=segvol_model.prompt_encoder,
            mask_decoder=segvol_model.mask_decoder,
            text_encoder=getattr(segvol_model, "text_encoder", None),
            **kwargs,
        )

    # ---------- энкодинг ----------
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B,1,D,H,W) -> embedding (B,C,d,h,w)."""
        bs = image.shape[0]
        emb, _ = self.image_encoder(image)               # (B, N_tokens, C)
        d, h, w = (int(x) for x in self.feat_shape)
        emb = emb.transpose(1, 2).view(bs, -1, d, h, w)  # (B, C, d, h, w)
        return emb

    def encode_multiphase(self, phases: torch.Tensor):
        """phases: (B, P, 1, D, H, W) (или (B,P,D,H,W)) -> (embedding, phase_weights|None).

        concat_stem: fuse в 1 канал, затем один прогон энкодера.
        attention:   энкодер по каждой фазе, затем attention-fusion эмбеддингов.
        """
        if phases.dim() == 6:
            phases = phases  # (B,P,1,D,H,W)
        elif phases.dim() == 5:
            phases = phases.unsqueeze(2)  # (B,P,1,D,H,W)
        else:
            raise ValueError(f"phases dim must be 5 or 6, got {phases.dim()}")
        b, p = phases.shape[0], phases.shape[1]

        if self.phase_fusion.mode == "concat_stem":
            fused_img = self.phase_fusion.fuse_input(phases.squeeze(2))  # (B,1,D,H,W)
            emb = self.encode(fused_img)
            return emb, None

        # attention: кодируем каждую фазу отдельно
        embs = []
        for i in range(p):
            embs.append(self.encode(phases[:, i]))       # (B,C,d,h,w)
        phase_embeddings = torch.stack(embs, dim=1)      # (B,P,C,d,h,w)
        fused, weights = self.phase_fusion.fuse_embeddings(phase_embeddings)
        return fused, weights

    # ---------- сегментация (реплика SegVol.forward_decoder) ----------
    def segment(self, embedding, img_shape, text=None, boxes=None, points=None):
        """embedding: (B,C,d,h,w) -> logits (B,1,D,H,W)."""
        assert self.prompt_encoder is not None and self.mask_decoder is not None, \
            "segment() требует prompt_encoder и mask_decoder."
        if boxes is not None and boxes.dim() == 2:
            boxes = boxes[:, None, :]
        text_embedding = self.text_encoder(text) if (text is not None and self.text_encoder) else None
        sparse_emb, dense_emb = self.prompt_encoder(
            points=points, boxes=boxes, masks=None, text_embedding=text_embedding,
        )
        dense_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=embedding,
            text_embedding=text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        logits = F.interpolate(low_res_masks, size=img_shape, mode="trilinear", align_corners=False)
        return logits

    # ---------- классификация ----------
    def classify(self, embedding, mask=None, extra_feat=None) -> torch.Tensor:
        return self.cls_head(embedding, mask=mask, extra_feat=extra_feat)

    # ---------- общий проход (multi-task) ----------
    def forward(
        self,
        image=None,
        phases=None,
        seg_prompt: Optional[dict] = None,
        cls_mask: Optional[torch.Tensor] = None,
        cls_extra_feat: Optional[torch.Tensor] = None,
        return_seg: bool = True,
        return_cls: bool = True,
    ) -> dict:
        """Принимает либо одиночный image (B,1,D,H,W), либо phases (B,P,...).

        seg_prompt: dict с ключами text/boxes/points для сегментации.
        cls_mask:   маска опухоли для masked-pooling классификатора (если pool='masked').
                    Если None, при сегментации используется предсказанная маска.
        return: {'seg_logits', 'cls_logits', 'phase_weights'} (наличие зависит от флагов).
        """
        out: dict = {}
        if phases is not None:
            embedding, phase_weights = self.encode_multiphase(phases)
            out["phase_weights"] = phase_weights
            img_shape = phases.shape[-3:]
        else:
            embedding = self.encode(image)
            img_shape = image.shape[-3:]

        seg_logits = None
        if return_seg and self.mask_decoder is not None:
            prompt = seg_prompt or {}
            seg_logits = self.segment(embedding, img_shape, **prompt)
            out["seg_logits"] = seg_logits

        if return_cls:
            mask = cls_mask
            if mask is None and seg_logits is not None and self.cls_head.pool == "masked":
                mask = (torch.sigmoid(seg_logits) > 0.5).float()
            out["cls_logits"] = self.classify(embedding, mask=mask, extra_feat=cls_extra_feat)

        return out
