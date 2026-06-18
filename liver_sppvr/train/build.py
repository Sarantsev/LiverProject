from __future__ import annotations

import random
from typing import Optional

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
    """Реальная модель: предобученный SegVol + наши головы. Запускать на GPU-девайсе."""
    from transformers import AutoModel, AutoTokenizer  # ленивый импорт

    sv = cfg["segvol"]
    tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
    hf = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=False)
    # внутренний SegVol-модуль (см. run_segvol_liver.py: hf.model.text_encoder...)
    inner = getattr(hf, "model", hf)
    if hasattr(inner, "text_encoder"):
        inner.text_encoder.tokenizer = tokenizer
    for attr in ("image_encoder", "prompt_encoder", "mask_decoder"):
        if not hasattr(inner, attr):
            raise AttributeError(
                f"У загруженной модели SegVol нет '{attr}'. "
                f"Доступные атрибуты: {list(vars(inner).keys())[:20]}")

    cls = cfg["classifier"]
    mp = cfg["multiphase"]
    model = SegVolMultiTask.from_segvol(
        inner,
        roi_size=tuple(sv["spatial_size"]), patch_size=tuple(sv["patch_size"]),
        embed_dim=sv["embed_dim"], num_classes=cls["num_classes"],
        cls_hidden_dim=cls["hidden_dim"], cls_dropout=cls["dropout"],
        cls_pool=cls["pool"], fusion_mode=mp["fusion"], n_phases=len(mp["phases"]),
    )
    return model.to(device)


# ----------------- заглушка для dry-run на CPU -----------------
class StubMultiTask(nn.Module):
    """Лёгкая модель с тем же интерфейсом forward, что и SegVolMultiTask."""
    def __init__(self, num_classes: int = 5, hidden: int = 8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, hidden, 3, padding=1), nn.GELU())
        self.seg_out = nn.Conv3d(hidden, 1, 1)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, phases=None, image=None, seg_prompt=None, cls_mask=None,
                cls_extra_feat=None, return_seg=True, return_cls=True) -> dict:
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
