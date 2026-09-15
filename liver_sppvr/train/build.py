from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import torch

from ..models.backbones import build_backbone
from ..models.classifier import LiverTumorClassifier


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 2023) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1 (near-unit on the balanced MCT-LTDiag)."""
    cnt = Counter(labels)
    w = torch.tensor([1.0 / max(cnt.get(c, 0), 1) for c in range(num_classes)])
    return w / w.sum() * num_classes


def make_scheduler(optimizer, warmup: int, total: int):
    """Linear warm-up then cosine decay, stepped once per epoch."""
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        prog = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def build_classifier(cfg: dict, device, *, backbone: str | None = None,
                     finetune: str | None = None, mask_source: str | None = None,
                     clinical_dim: int = 0) -> LiverTumorClassifier:
    """Our model: pretrained backbone (segvol|merlin) + phase fusion + head.

    mask_source none -> global-average pooling; gt|pred -> masked pooling (the trainer
    supplies the mask). clinical_dim > 0 enables early fusion of the clinical vector.
    """
    ccfg = cfg["classifier"]
    mp = cfg["multiphase"]
    backbone = backbone or ccfg.get("backbone", "segvol")
    finetune = finetune or ccfg.get("finetune", "dora")
    mask_source = mask_source or ccfg.get("mask_source", "pred")
    bb = build_backbone(backbone, cfg)
    ft_cfg = dict(cfg.get(backbone, {}).get("lora", {}))
    ft_cfg["partial_targets"] = cfg.get(backbone, {}).get("partial_targets")
    bb.set_finetune(finetune, ft_cfg)
    model = LiverTumorClassifier(
        bb, num_classes=ccfg["num_classes"], n_phases=len(mp["phases"]),
        fusion_mode=mp.get("fusion", "attention"), hidden_dim=ccfg["hidden_dim"],
        dropout=ccfg["dropout"], pool=("gap" if mask_source == "none" else "masked"),
        extra_feat_dim=clinical_dim)
    print(f"classifier: backbone={backbone} finetune={finetune} mask_source={mask_source} "
          f"clinical_dim={clinical_dim} fusion={mp.get('fusion', 'attention')}")
    return model.to(device)
