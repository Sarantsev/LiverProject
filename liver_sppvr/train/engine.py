"""Shared forward/evaluate loop for every classifier (baselines, ours, the cascade).

All models take phases / phase_present / extra_feat (+ mask for masked pooling) and
return logits, so one loop serves training (scripts/train_cls.py) and the fully
automatic cascade (scripts/infer_cascade.py); metrics come from train.metrics.
"""
from __future__ import annotations

import torch

from .metrics import cls_metrics


def forward_batch(model, batch, device, *, use_mask: bool, want_proj: bool = False):
    kw = dict(phases=batch["phases"].to(device), phase_present=batch["phase_present"].to(device))
    if batch.get("clinical") is not None:
        kw["extra_feat"] = batch["clinical"].to(device)
    if use_mask:
        kw["mask"] = batch["mask"].to(device)
    if want_proj:
        kw["return_proj"] = True
    return model(**kw)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, *, use_mask: bool,
             return_preds: bool = False) -> dict:
    """-> cls_metrics dict; with return_preds also patient_id / y_true / y_pred / y_prob."""
    model.eval()
    pids, y_true, y_pred, y_prob = [], [], [], []
    for batch in loader:
        logits = forward_batch(model, batch, device, use_mask=use_mask)
        probs = torch.softmax(logits.float(), dim=1)
        y_prob.extend(probs.cpu().tolist())
        y_pred.extend(probs.argmax(1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
        pids.extend(batch["patient_id"])
    m = cls_metrics(y_true, y_pred, y_prob, num_classes)
    if return_preds:
        m.update(patient_id=pids, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    return m
