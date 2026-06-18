"""Движок обучения/валидации multi-task (device-agnostic, single-device).

Работает с любой моделью, чей forward принимает
    (phases=..., seg_prompt=..., cls_mask=..., return_seg=..., return_cls=...)
и возвращает dict с ключами 'seg_logits'/'cls_logits'. Это и реальная
SegVolMultiTask, и заглушка для dry-run.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _dice_score(seg_logits, mask, thr: float = 0.5) -> float:
    prob = torch.sigmoid(seg_logits.float())
    pred = (prob > thr).float()
    t = (mask > 0.5).float()
    inter = (pred * t).flatten(1).sum(1)
    denom = pred.flatten(1).sum(1) + t.flatten(1).sum(1)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return dice.mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device, *,
                    seg_text: str = "liver tumor", grad_clip: Optional[float] = 1.0) -> dict:
    model.train()
    agg = {"loss": 0.0, "seg_loss": 0.0, "cls_loss": 0.0, "n": 0}
    for batch in loader:
        batch = _batch_to_device(batch, device)
        bs = batch["label"].shape[0]
        seg_prompt = {"text": [seg_text] * bs}

        optimizer.zero_grad()
        out = model(phases=batch["phases"], seg_prompt=seg_prompt,
                    cls_mask=batch["mask"], return_seg=True, return_cls=True)
        losses = loss_fn(out, {"mask": batch["mask"], "label": batch["label"]})
        losses["loss"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        agg["loss"] += losses["loss"].item() * bs
        agg["seg_loss"] += float(losses["seg_loss"]) * bs
        agg["cls_loss"] += float(losses["cls_loss"]) * bs
        agg["n"] += bs
    n = max(agg.pop("n"), 1)
    return {k: v / n for k, v in agg.items()}


@torch.no_grad()
def evaluate(model, loader, device, *, num_classes: int, seg_text: str = "liver tumor") -> dict:
    model.eval()
    dices, y_true, y_pred = [], [], []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        bs = batch["label"].shape[0]
        out = model(phases=batch["phases"], seg_prompt={"text": [seg_text] * bs},
                    cls_mask=batch["mask"], return_seg=True, return_cls=True)
        if out.get("seg_logits") is not None:
            dices.append(_dice_score(out["seg_logits"], batch["mask"]))
        if out.get("cls_logits") is not None:
            y_pred.extend(out["cls_logits"].argmax(1).cpu().tolist())
            y_true.extend(batch["label"].cpu().tolist())

    metrics = {"dice": float(np.mean(dices)) if dices else float("nan")}
    if y_true:
        metrics["accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
        try:
            from sklearn.metrics import f1_score
            metrics["macro_f1"] = float(
                f1_score(y_true, y_pred, average="macro",
                         labels=list(range(num_classes)), zero_division=0))
        except Exception:
            pass
    return metrics
