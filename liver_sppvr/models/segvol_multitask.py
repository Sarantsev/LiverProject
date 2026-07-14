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
        seg_phase_idx: Optional[int] = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.text_encoder = text_encoder

        self.feat_shape = (np.array(roi_size) / np.array(patch_size)).astype(int)  # (d,h,w)
        self.embed_dim = embed_dim
        self.n_phases = n_phases
        # hybrid: if set, segmentation uses only this phase's embedding (on-distribution
        # for the pretrained SegVol decoder); classification always uses the fused embedding.
        self.seg_phase_idx = seg_phase_idx

        self.phase_fusion = PhaseFusion(mode=fusion_mode, n_phases=n_phases, embed_dim=embed_dim)
        self.cls_head = TumorClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=cls_hidden_dim,
            dropout=cls_dropout,
            pool=cls_pool,
            extra_feat_dim=cls_extra_feat_dim,
        )

    # ---------- factory from a loaded SegVol model ----------
    @classmethod
    def from_segvol(cls, segvol_model: nn.Module, **kwargs) -> "SegVolMultiTask":
        """Build the wrapper from the components of an already-initialized SegVol model."""
        return cls(
            image_encoder=segvol_model.image_encoder,
            prompt_encoder=segvol_model.prompt_encoder,
            mask_decoder=segvol_model.mask_decoder,
            text_encoder=getattr(segvol_model, "text_encoder", None),
            **kwargs,
        )

    # ---------- encoding ----------
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B,1,D,H,W) -> embedding (B,C,d,h,w)."""
        bs = image.shape[0]
        emb, _ = self.image_encoder(image)               # (B, N_tokens, C)
        d, h, w = (int(x) for x in self.feat_shape)
        emb = emb.transpose(1, 2).view(bs, -1, d, h, w)  # (B, C, d, h, w)
        return emb

    def encode_multiphase(self, phases: torch.Tensor):
        """phases: (B, P, 1, D, H, W) (or (B,P,D,H,W)) -> (fused, phase_weights|None, per_phase|None).

        concat_stem: fuse into 1 channel, then a single encoder pass (no per-phase embeddings).
        attention:   encode each phase, then attention-fuse; per-phase embeddings are also
                     returned (B,P,C,d,h,w) so the hybrid can pick a single phase for seg.
        """
        if phases.dim() == 6:
            phases = phases  # (B,P,1,D,H,W)
        elif phases.dim() == 5:
            phases = phases.unsqueeze(2)  # (B,P,1,D,H,W)
        else:
            raise ValueError(f"phases dim must be 5 or 6, got {phases.dim()}")
        p = phases.shape[1]

        if self.phase_fusion.mode == "concat_stem":
            fused_img = self.phase_fusion.fuse_input(phases.squeeze(2))  # (B,1,D,H,W)
            emb = self.encode(fused_img)
            return emb, None, None

        # attention: encode each phase separately
        embs = []
        for i in range(p):
            embs.append(self.encode(phases[:, i]))       # (B,C,d,h,w)
        phase_embeddings = torch.stack(embs, dim=1)      # (B,P,C,d,h,w)
        fused, weights = self.phase_fusion.fuse_embeddings(phase_embeddings)
        return fused, weights, phase_embeddings

    # ---------- segmentation (replica of SegVol.forward_decoder) ----------
    def segment(self, embedding, img_shape, text=None, boxes=None, points=None):
        """embedding: (B,C,d,h,w) -> logits (B,1,D,H,W)."""
        assert self.prompt_encoder is not None and self.mask_decoder is not None, \
            "segment() requires prompt_encoder and mask_decoder."
        if boxes is not None and boxes.dim() == 2:
            boxes = boxes[:, None, :]
        text_embedding = (self.text_encoder(text, embedding.device)
                          if (text is not None and self.text_encoder) else None)
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

    # ---------- classification ----------
    def classify(self, embedding, mask=None, extra_feat=None) -> torch.Tensor:
        return self.cls_head(embedding, mask=mask, extra_feat=extra_feat)

    # ---------- full multi-task pass ----------
    def forward(
        self,
        image=None,
        phases=None,
        seg_prompt: Optional[dict] = None,
        seg_text: Optional[str] = None,
        cls_mask: Optional[torch.Tensor] = None,
        cls_extra_feat: Optional[torch.Tensor] = None,
        return_seg: bool = True,
        return_cls: bool = True,
    ) -> dict:
        """Accepts either a single image (B,1,D,H,W) or phases (B,P,...).

        seg_prompt: dict with keys text/boxes/points for segmentation.
        cls_mask:   tumor mask for the masked-pooling classifier (if pool='masked').
                    If None, the predicted mask is used instead.
        return: {'seg_logits', 'cls_logits', 'phase_weights'} (presence depends on flags).
        """
        # autocast lives inside forward -- otherwise AMP is not applied in the
        # DataParallel replica threads (autocast is thread-local). The flag is set by
        # the training script.
        with torch.cuda.amp.autocast(enabled=getattr(self, "_amp_enabled", False)):
            out: dict = {}
            if phases is not None:
                cls_embedding, phase_weights, phase_embs = self.encode_multiphase(phases)
                out["phase_weights"] = phase_weights
                img_shape = phases.shape[-3:]
                # hybrid: segmentation on a single phase (PVP) if seg_phase_idx is set and
                # per-phase embeddings are available; classification on the fused embedding.
                if self.seg_phase_idx is not None and phase_embs is not None:
                    seg_embedding = phase_embs[:, self.seg_phase_idx]  # (B,C,d,h,w)
                else:
                    seg_embedding = cls_embedding
            else:
                seg_embedding = cls_embedding = self.encode(image)
                img_shape = image.shape[-3:]

            seg_logits = None
            if return_seg and self.mask_decoder is not None:
                prompt = dict(seg_prompt or {})
                # Build the text here, per local batch -- otherwise DataParallel splits a
                # pre-built list across devices incorrectly (see engine.py).
                if seg_text is not None and prompt.get("text") is None:
                    prompt["text"] = [seg_text] * seg_embedding.shape[0]
                seg_logits = self.segment(seg_embedding, img_shape, **prompt)
                out["seg_logits"] = seg_logits

            if return_cls:
                mask = cls_mask
                if mask is None and seg_logits is not None and self.cls_head.pool == "masked":
                    mask = (torch.sigmoid(seg_logits) > 0.5).float()
                out["cls_logits"] = self.classify(cls_embedding, mask=mask, extra_feat=cls_extra_feat)

        return out
