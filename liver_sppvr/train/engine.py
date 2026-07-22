"""Multi-task train/eval engine (device-agnostic, single-device).

Works with any model whose forward accepts
    (phases=..., seg_prompt=..., cls_mask=..., return_seg=..., return_cls=...)
and returns a dict with keys 'seg_logits'/'cls_logits'. This covers both the real
SegVolMultiTask and the dry-run stub.
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


def _masks_to_boxes(masks: torch.Tensor, bbox_shift: int = 0) -> torch.Tensor:
    """masks (B,1,D,H,W) -> boxes (B,6) = [d0,h0,w0,d1,h1,w1] in voxels.

    Replica of SegVol.generate_box: min/max nonzero per array axis (D,H,W order matches
    the image axes), with optional bbox_shift jitter (box augmentation on train). An empty
    mask -> a box over the whole volume.
    """
    import random
    m = (masks.detach() > 0.5).cpu()                      # nonzero on CPU (safer than GPU on torch 1.13)
    out = []
    for i in range(m.shape[0]):
        idx = m[i, 0].nonzero(as_tuple=True)              # (D,H,W)
        shape = m[i, 0].shape
        if idx[0].numel() == 0:
            out.append([0, 0, 0, *shape])
            continue
        mn = [int(d.min()) for d in idx]
        mx = [int(d.max()) for d in idx]
        if bbox_shift:
            mn = [max(0, c + random.randint(-bbox_shift, bbox_shift)) for c in mn]
            mx = [min(shape[j], c + random.randint(-bbox_shift, bbox_shift)) for j, c in enumerate(mx)]
        out.append(mn + mx)
    return torch.tensor(out, dtype=torch.float32, device=masks.device)   # (B,6) on the mask's device


def _seg_prompt_kwargs(batch, mode: str, seg_text: str, bbox_shift: int) -> dict:
    """Assemble the segmentation-prompt args: text mode or box mode (from the GT mask)."""
    if mode == "box":
        boxes = _masks_to_boxes(batch["mask"], bbox_shift=bbox_shift)
        return dict(seg_prompt={"boxes": boxes}, seg_text=None)
    return dict(seg_text=seg_text)


def _per_class_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    """Per-class sensitivity/specificity/F1/AUC (one-vs-rest) from a confusion matrix.

    Reported alongside Cohen's kappa to match the MCT-LTDiag benchmark protocol.
    """
    from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
    labels = list(range(num_classes))
    yt, yp, pr = np.array(y_true), np.array(y_pred), np.array(y_prob)
    cm = confusion_matrix(yt, yp, labels=labels)          # rows=true, cols=pred
    total = cm.sum()
    f1 = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
    sens = np.full(num_classes, np.nan)
    spec = np.full(num_classes, np.nan)
    aucs = np.full(num_classes, np.nan)
    for c in labels:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        if tp + fn > 0:
            sens[c] = tp / (tp + fn)                       # recall for class c
        if tn + fp > 0:
            spec[c] = tn / (tn + fp)
        yc = (yt == c).astype(int)
        if 0 < yc.sum() < len(yc):                         # class present and not the only one
            try:
                aucs[c] = roc_auc_score(yc, pr[:, c])
            except Exception:
                pass
    return {"per_sens": sens.tolist(), "per_spec": spec.tolist(),
            "per_f1": [float(x) for x in f1], "per_auc": aucs.tolist()}


def _dice_score(seg_logits, mask, thr: float = 0.5) -> float:
    prob = torch.sigmoid(seg_logits.float())
    pred = (prob > thr).float()
    t = (mask > 0.5).float()
    inter = (pred * t).flatten(1).sum(1)
    denom = pred.flatten(1).sum(1) + t.flatten(1).sum(1)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return dice.mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device, *,
                    seg_text: str = "liver tumor", grad_clip: Optional[float] = 1.0,
                    scaler=None, seg_prompt_mode: str = "text", bbox_shift: int = 0) -> dict:
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    agg = {"loss": 0.0, "seg_loss": 0.0, "cls_loss": 0.0, "n": 0}
    for batch in loader:
        batch = _batch_to_device(batch, device)
        bs = batch["label"].shape[0]
        seg_kw = _seg_prompt_kwargs(batch, seg_prompt_mode, seg_text, bbox_shift)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(phases=batch["phases"], cls_mask=batch["mask"],
                        cls_extra_feat=batch.get("radiomics"),
                        return_seg=True, return_cls=True, **seg_kw)
            losses = loss_fn(out, {"mask": batch["mask"], "label": batch["label"]})
        if use_amp:
            scaler.scale(losses["loss"]).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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
def evaluate(model, loader, device, *, num_classes: int, seg_text: str = "liver tumor",
             seg_prompt_mode: str = "text", rf=None, rf_alpha: float = 0.5) -> dict:
    """rf: optional fitted radiomics classifier (RF/SVM) for late fusion. When given, the
    reported cls probabilities are rf_alpha*deep + (1-rf_alpha)*radiomics."""
    model.eval()
    dices, y_true, y_pred, y_prob = [], [], [], []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        seg_kw = _seg_prompt_kwargs(batch, seg_prompt_mode, seg_text, bbox_shift=0)  # no jitter on eval
        out = model(phases=batch["phases"], cls_mask=batch["mask"],
                    cls_extra_feat=batch.get("radiomics"),
                    return_seg=True, return_cls=True, **seg_kw)
        if out.get("seg_logits") is not None:
            dices.append(_dice_score(out["seg_logits"], batch["mask"]))
        if out.get("cls_logits") is not None:
            probs = torch.softmax(out["cls_logits"].float(), dim=1)
            if rf is not None and batch.get("radiomics") is not None:
                rp = rf.predict_proba(batch["radiomics"].cpu().numpy())     # (B, n_present)
                full = np.zeros((rp.shape[0], num_classes), dtype=np.float32)
                full[:, rf.classes_.astype(int)] = rp                        # map to 0..C-1
                probs = rf_alpha * probs + (1 - rf_alpha) * torch.tensor(full, device=probs.device)
            y_prob.extend(probs.cpu().tolist())
            y_pred.extend(probs.argmax(1).cpu().tolist())
            y_true.extend(batch["label"].cpu().tolist())

    metrics = {"dice": float(np.mean(dices)) if dices else float("nan")}
    if y_true:
        # metrics reported like the liver-tumor classification literature: accuracy + AUC
        # (macro one-vs-rest), balanced accuracy, macro-F1, Cohen's kappa, and per-class stats.
        metrics["accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
        try:
            from sklearn.metrics import (f1_score, balanced_accuracy_score,
                                         roc_auc_score, cohen_kappa_score)
            labels = list(range(num_classes))
            metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro",
                                                 labels=labels, zero_division=0))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
            metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred, labels=labels))
            try:
                metrics["auc"] = float(roc_auc_score(
                    y_true, np.array(y_prob), multi_class="ovr", average="macro", labels=labels))
            except Exception:
                metrics["auc"] = float("nan")   # a class may be absent in a small val fold
            metrics.update(_per_class_metrics(y_true, y_pred, y_prob, num_classes))
        except Exception:
            pass
    return metrics
