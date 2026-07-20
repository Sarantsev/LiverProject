from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from ..models import SegVolMultiTask


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 2023) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_segvol_multitask(cfg: dict, device) -> SegVolMultiTask:
    """Real model: pretrained SegVol + our heads. Run on the GPU box."""
    from transformers import AutoModel, AutoTokenizer  # lazy import

    sv = cfg["segvol"]
    tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
    hf = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=False)
    # inner SegVol module (see run_segvol_liver.py: hf.model.text_encoder...)
    inner = getattr(hf, "model", hf)
    if hasattr(inner, "text_encoder"):
        inner.text_encoder.tokenizer = tokenizer
    for attr in ("image_encoder", "prompt_encoder", "mask_decoder"):
        if not hasattr(inner, attr):
            raise AttributeError(
                f"The loaded SegVol model has no '{attr}'. "
                f"Available attributes: {list(vars(inner).keys())[:20]}")

    cls = cfg["classifier"]
    mp = cfg["multiphase"]
    # hybrid: resolve the segmentation phase name -> index (None -> seg uses fused embedding)
    seg_phase = mp.get("seg_phase")
    seg_phase_idx = None
    if seg_phase:
        if seg_phase not in mp["phases"]:
            raise ValueError(f"multiphase.seg_phase={seg_phase!r} not in phases {mp['phases']}")
        seg_phase_idx = mp["phases"].index(seg_phase)
    model = SegVolMultiTask.from_segvol(
        inner,
        roi_size=tuple(sv["spatial_size"]), patch_size=tuple(sv["patch_size"]),
        embed_dim=sv["embed_dim"], num_classes=cls["num_classes"],
        cls_hidden_dim=cls["hidden_dim"], cls_dropout=cls["dropout"],
        cls_pool=cls["pool"], cls_extra_feat_dim=cls.get("extra_feat_dim", 0),
        fusion_mode=mp["fusion"], n_phases=len(mp["phases"]),
        seg_phase_idx=seg_phase_idx,
    )
    if seg_phase_idx is not None:
        print(f"hybrid: segmentation on phase '{seg_phase}' (idx {seg_phase_idx}), "
              f"classification on all {len(mp['phases'])} phases")
    lora_cfg = sv.get("lora") or {}
    if lora_cfg.get("enabled", False):
        from ..models.lora import apply_lora
        set_encoder_trainable(model, False)               # encoder base frozen
        variant = lora_cfg.get("variant", "lora")
        n = apply_lora(model.image_encoder, targets=lora_cfg.get("targets", ["qkv"]),
                       rank=lora_cfg.get("rank", 8), alpha=lora_cfg.get("alpha", 16.0),
                       dropout=lora_cfg.get("dropout", 0.0), variant=variant)
        n_train = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
        print(f"{variant.upper()}: wrapped {n} encoder Linear layers, "
              f"trainable in encoder {n_train/1e6:.3f}M")
    elif sv.get("freeze_encoder", False):
        set_encoder_trainable(model, False)
        n_frozen = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"image_encoder frozen ({n_frozen/1e6:.1f}M params not trained)")
    return model.to(device)


def set_encoder_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze the SegVol ViT encoder (works with DataParallel too)."""
    m = model.module if hasattr(model, "module") else model
    for p in m.image_encoder.parameters():
        p.requires_grad_(trainable)


# ----------------- stub for CPU dry-run -----------------
class StubMultiTask(nn.Module):
    """Lightweight model with the same forward interface as SegVolMultiTask."""
    def __init__(self, num_classes: int = 5, hidden: int = 8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, hidden, 3, padding=1), nn.GELU())
        self.seg_out = nn.Conv3d(hidden, 1, 1)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, phases=None, image=None, seg_prompt=None, seg_text=None,
                cls_mask=None, cls_extra_feat=None, return_seg=True, return_cls=True) -> dict:
        x = phases.mean(1) if phases is not None else image   # (B,1,D,H,W)
        h = self.stem(x)
        out = {}
        if return_seg:
            out["seg_logits"] = self.seg_out(h)
        if return_cls:
            out["cls_logits"] = self.cls(h.flatten(2).mean(-1))
        out["phase_weights"] = None
        return out


def build_dryrun_model(cfg: dict) -> StubMultiTask:
    return StubMultiTask(num_classes=cfg["classifier"]["num_classes"])
