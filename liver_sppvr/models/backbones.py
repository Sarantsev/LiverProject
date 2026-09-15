"""Pretrained CT image backbones behind ONE interface: (B,1,D,H,W) -> (B,C,d,h,w).

Two encoders are compared head-to-head for tumour-type classification:
  * ``segvol`` -- BAAI/SegVol ViT-B (3D segmentation foundation model, 25k CT volumes);
                  adapted with DoRA low-rank adapters on its Linear layers.
  * ``merlin`` -- Stanford Merlin I3D-ResNet-152 (abdominal-CT vision-language model,
                  pretrained with radiology reports + EHR); convolutional -> partial finetune.

The classifier (phase fusion + head) is backbone-agnostic. Each backbone also declares
``preprocess_spec`` (spatial size / normalization / HU window) so the dataset feeds it
exactly the way it was pretrained. From-scratch encoders are deliberately NOT offered:
on MCT-LTDiag they cap at AUC ~0.76 vs ~0.85 with a pretrained CT encoder.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

FINETUNE_MODES = ("frozen", "full", "dora", "lora", "partial")


class Backbone(nn.Module):
    embed_dim: int
    preprocess_spec: dict          # {"spatial_size": (D,H,W), "normalize": str, "hu_window": (lo,hi)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,1,D,H,W) -> (B,C,d,h,w)
        raise NotImplementedError

    @property
    def encoder(self) -> nn.Module:
        """The pretrained module that finetune modes act on."""
        raise NotImplementedError

    def set_finetune(self, mode: str, cfg: dict | None = None) -> None:
        """frozen | full | dora/lora (Linear adapters, ViT) | partial (unfreeze named parts)."""
        cfg = cfg or {}
        if mode not in FINETUNE_MODES:
            raise ValueError(f"finetune must be one of {FINETUNE_MODES}, got {mode!r}")
        for p in self.encoder.parameters():
            p.requires_grad_(mode == "full")
        if mode in ("dora", "lora"):
            from .lora import apply_lora
            n = apply_lora(self.encoder, targets=cfg.get("targets", ["qkv"]),
                           rank=cfg.get("rank", 8), alpha=cfg.get("alpha", 16.0),
                           dropout=cfg.get("dropout", 0.0), variant=mode)
            print(f"{mode.upper()}: wrapped {n} encoder Linear layers")
        elif mode == "partial":
            targets = tuple(cfg.get("partial_targets") or ())
            if not targets:
                raise ValueError("finetune=partial needs a non-empty 'partial_targets' list")
            n = 0
            for name, p in self.encoder.named_parameters():
                if any(t in name for t in targets):
                    p.requires_grad_(True); n += p.numel()
            print(f"PARTIAL: unfroze {n/1e6:.2f}M encoder params matching {list(targets)}")
        tot = sum(p.numel() for p in self.encoder.parameters())
        tr = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"backbone {type(self).__name__}: {tr/1e6:.2f}M / {tot/1e6:.1f}M encoder params trainable ({mode})")


# ======================================================================================
class SegVolViTBackbone(Backbone):
    """SegVol image encoder (MONAI ViT) -> token sequence reshaped to a (d,h,w) grid."""
    def __init__(self, image_encoder: nn.Module, roi_size: Sequence[int] = (32, 256, 256),
                 patch_size: Sequence[int] = (4, 16, 16), embed_dim: int = 768,
                 normalize: str = "foreground", hu_window: Sequence[float] = (-175, 250)):
        super().__init__()
        self.image_encoder = image_encoder
        self.feat_shape = tuple(int(x) for x in np.array(roi_size) // np.array(patch_size))
        self.embed_dim = embed_dim
        self.preprocess_spec = {"spatial_size": tuple(roi_size), "normalize": normalize,
                                "hu_window": tuple(hu_window)}

    @property
    def encoder(self) -> nn.Module:
        return self.image_encoder

    def forward(self, x):
        emb, _ = self.image_encoder(x)                       # (B, N_tokens, C)
        d, h, w = self.feat_shape
        return emb.transpose(1, 2).reshape(x.shape[0], -1, d, h, w)

    @classmethod
    def from_pretrained(cls, sv_cfg: dict) -> "SegVolViTBackbone":
        """Load BAAI/SegVol from HF and keep ONLY the image encoder (no prompt/mask decoders)."""
        from transformers import AutoModel
        hf = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=False)
        inner = getattr(hf, "model", hf)
        if not hasattr(inner, "image_encoder"):
            raise AttributeError("loaded SegVol has no image_encoder; attrs: "
                                 f"{list(vars(inner).keys())[:20]}")
        return cls(inner.image_encoder, roi_size=sv_cfg["spatial_size"],
                   patch_size=sv_cfg["patch_size"], embed_dim=sv_cfg["embed_dim"],
                   normalize=sv_cfg.get("normalize", "foreground"),
                   hu_window=tuple(sv_cfg.get("hu_window", (-175, 250))))


# ======================================================================================
def build_backbone(name: str, cfg: dict) -> Backbone:
    """name: segvol | merlin -- reads the matching cfg block (geometry / normalisation / adapters)."""
    name = name.lower()
    if name == "segvol":
        return SegVolViTBackbone.from_pretrained(cfg["segvol"])
    if name == "merlin":
        from .merlin import MerlinBackbone
        return MerlinBackbone.from_pretrained(cfg["merlin"])
    raise ValueError(f"unknown backbone {name!r}; available: segvol, merlin")
