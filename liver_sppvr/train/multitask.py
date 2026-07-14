"""Multi-task (seg + cls) training CLI. Single-device, config-driven.

Run on the GPU box:
    python -m liver_sppvr.train.multitask --config configs/default.yaml

Smoke-test the loop on CPU without weights/data:
    python -m liver_sppvr.train.multitask --config configs/default.yaml --dry-run
"""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.device import resolve_device
from .build import load_config, set_seed, build_dryrun_model, build_segvol_multitask
from .engine import train_one_epoch, evaluate
from .losses import MultiTaskLoss


# --------- utilities ---------
def stratified_patient_split(labels_by_patient: dict, val_frac: float = 0.2, seed: int = 2023):
    """labels_by_patient: {patient_id: label} -> (train_ids, val_ids), stratified by class."""
    import random as _r
    rng = _r.Random(seed)
    by_cls = {}
    for pid, y in labels_by_patient.items():
        by_cls.setdefault(y, []).append(pid)
    train, val = [], []
    for y, pids in by_cls.items():
        pids = pids[:]; rng.shuffle(pids)
        k = max(1, int(round(len(pids) * val_frac))) if len(pids) > 1 else 0
        val += pids[:k]; train += pids[k:]
    return train, val


def class_weights(labels, num_classes: int) -> torch.Tensor:
    cnt = Counter(labels)
    w = torch.tensor([1.0 / max(cnt.get(c, 0), 1) for c in range(num_classes)])
    return w / w.sum() * num_classes


def make_scheduler(optimizer, warmup: int, total: int):
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        prog = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------- synthetic dataset for dry-run ---------
class _SyntheticDataset(Dataset):
    def __init__(self, n=8, n_phases=4, num_classes=5, spatial=(4, 16, 16)):
        self.n, self.p, self.k, self.s = n, n_phases, num_classes, spatial
    def __len__(self): return self.n
    def __getitem__(self, i):
        d, h, w = self.s
        return dict(
            phases=torch.rand(self.p, 1, d, h, w),
            mask=(torch.rand(1, d, h, w) > 0.7).float(),
            label=torch.tensor(i % self.k, dtype=torch.long),
            phase_present=torch.ones(self.p),
            patient_id=f"syn{i}",
        )


def _collate(batch):
    from ..data import collate_multiphase
    return collate_multiphase(batch)


# --------- main entry point ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="synthetic data + stub on CPU")
    ap.add_argument("--device", default=None, help="override device from config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    ap.add_argument("--num-workers", type=int, default=None, help="override train.num_workers")
    ap.add_argument("--work-dir", default=None, help="override train.work_dir (important for a 2nd run)")
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (overrides train.amp)")
    ap.add_argument("--lora", action="store_true", help="enable LoRA on the encoder (overrides segvol.lora.enabled)")
    ap.add_argument("--seg-prompt", choices=["text", "box"], default=None,
                    help="segmentation prompt mode (overrides train.seg_prompt)")
    ap.add_argument("--zoom", type=float, default=None,
                    help="probability of a zoom-in crop on train [0..1] (overrides train.zoom)")
    ap.add_argument("--zoom-eval", action="store_true",
                    help="zoom-in on validation (overrides train.zoom_eval)")
    ap.add_argument("--normalize", choices=["hu", "foreground"], default=None,
                    help="input normalization (overrides preprocess.normalize)")
    ap.add_argument("--seg-phase", default=None,
                    help="hybrid: phase name for segmentation (e.g. portal); 'all' = fused embedding "
                         "(overrides multiphase.seg_phase)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.lora:
        cfg.setdefault("segvol", {}).setdefault("lora", {})["enabled"] = True
    if args.seg_prompt:
        cfg["train"]["seg_prompt"] = args.seg_prompt
    if args.zoom is not None:
        cfg["train"]["zoom"] = args.zoom
    if args.zoom_eval:
        cfg["train"]["zoom_eval"] = True
    if args.normalize:
        cfg.setdefault("preprocess", {})["normalize"] = args.normalize
    if args.seg_phase:
        cfg["multiphase"]["seg_phase"] = None if args.seg_phase == "all" else args.seg_phase
    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg.get("device", "auto"))
    num_classes = cfg["classifier"]["num_classes"]
    tcfg = cfg["train"]
    epochs = args.epochs or (2 if args.dry_run else tcfg["num_epochs"])
    batch_size = args.batch_size or tcfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else tcfg.get("num_workers", 0)
    work_dir = args.work_dir or tcfg["work_dir"]
    use_amp = (args.amp or tcfg.get("amp", False)) and getattr(device, "type", str(device)) == "cuda"
    print(f"device={device} | epochs={epochs} | batch_size={batch_size} | "
          f"num_workers={num_workers} | amp={use_amp} | work_dir={work_dir}")

    # data
    if args.dry_run:
        train_ds = _SyntheticDataset(n=8, n_phases=len(cfg["multiphase"]["phases"]),
                                     num_classes=num_classes)
        val_ds = _SyntheticDataset(n=4, n_phases=len(cfg["multiphase"]["phases"]),
                                   num_classes=num_classes)
        model = build_dryrun_model(cfg).to(device)
        train_labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
    else:
        from ..data import load_manifest, MultiPhaseLiverDataset
        man = load_manifest(cfg["data"]["manifest"])
        full = MultiPhaseLiverDataset(
            man, class_names=cfg["classifier"]["class_names"],
            phases=cfg["multiphase"]["phases"],
            spatial_size=cfg["segvol"]["spatial_size"],
            hu_window=cfg["multiphase"]["hu_window"])
        labels_by_patient = {p["patient_id"]: p["label"] for p in full._patients}
        tr_ids, va_ids = stratified_patient_split(labels_by_patient, seed=cfg["project"]["seed"])
        zoom_margin = tcfg.get("zoom_margin", 0.5)
        pcfg = cfg.get("preprocess", {})
        normalize = pcfg.get("normalize", "hu")
        crop_fg = pcfg.get("crop_foreground", False)
        common = dict(class_names=cfg["classifier"]["class_names"],
                      phases=cfg["multiphase"]["phases"],
                      spatial_size=cfg["segvol"]["spatial_size"],
                      hu_window=cfg["multiphase"]["hu_window"],
                      normalize=normalize, crop_foreground=crop_fg)
        train_ds = MultiPhaseLiverDataset(man, patient_ids=tr_ids,
                                          augment=tcfg.get("augment", False),
                                          zoom=tcfg.get("zoom", 0.0), zoom_margin=zoom_margin,
                                          **common)
        val_ds = MultiPhaseLiverDataset(man, patient_ids=va_ids,
                                        zoom=(1.0 if tcfg.get("zoom_eval", False) else 0.0),
                                        zoom_margin=zoom_margin, **common)
        model = build_segvol_multitask(cfg, device)
        model._amp_enabled = use_amp    # autocast inside forward (needed for DataParallel)
        if torch.cuda.device_count() > 1:
            print(f"DataParallel: {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        train_labels = [labels_by_patient[p] for p in tr_ids]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=num_workers,
                              pin_memory=(getattr(device, "type", str(device)) == "cuda"),
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate, num_workers=num_workers,
                            pin_memory=(getattr(device, "type", str(device)) == "cuda"),
                            persistent_workers=num_workers > 0)

    # loss / optimizer
    cw = class_weights(train_labels, num_classes).to(device)
    loss_fn = MultiTaskLoss(seg_weight=tcfg["loss_weights"]["seg"],
                            cls_weight=tcfg["loss_weights"]["cls"],
                            class_weight=cw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = make_scheduler(optimizer, tcfg["warmup_epoch"], epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(work_dir, exist_ok=True)
    unfreeze_epoch = cfg["segvol"].get("unfreeze_epoch", -1)
    patience = tcfg.get("early_stop_patience", 0)
    seg_prompt_mode = tcfg.get("seg_prompt", "text")
    bbox_shift = tcfg.get("bbox_shift", 0)
    if not args.dry_run:
        pcfg = cfg.get("preprocess", {})
        print(f"seg_prompt={seg_prompt_mode}"
              + (f" (bbox_shift={bbox_shift})" if seg_prompt_mode == "box" else "")
              + f" | normalize={pcfg.get('normalize', 'hu')} crop_fg={pcfg.get('crop_foreground', False)}"
              + f" | zoom train_p={tcfg.get('zoom', 0.0)} eval={'on' if tcfg.get('zoom_eval', False) else 'off'}")

    best, best_epoch, no_improve = -1.0, -1, 0
    for epoch in range(epochs):
        if not args.dry_run and unfreeze_epoch >= 0 and epoch == unfreeze_epoch:
            from .build import set_encoder_trainable
            set_encoder_trainable(model, True)
            print(f"[epoch {epoch+1}] image_encoder unfrozen -- fine-tuning from here")
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler=scaler,
                             seg_prompt_mode=seg_prompt_mode, bbox_shift=bbox_shift)
        scheduler.step()
        ev = evaluate(model, val_loader, device, num_classes=num_classes,
                      seg_prompt_mode=seg_prompt_mode)
        print(f"[epoch {epoch+1}/{epochs}] train_loss={tr['loss']:.4f} "
              f"seg={tr['seg_loss']:.4f} cls={tr['cls_loss']:.4f} | "
              f"val_dice={ev.get('dice', float('nan')):.4f} "
              f"acc={ev.get('accuracy', float('nan')):.4f} "
              f"macroF1={ev.get('macro_f1', float('nan')):.4f}")
        score = ev.get("macro_f1", ev.get("accuracy", 0.0))
        if score is not None and score > best:
            best, best_epoch, no_improve = score, epoch + 1, 0
            _m = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({"epoch": epoch, "model": _m.state_dict(), "config": cfg},
                       os.path.join(work_dir, "best.pth"))
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"Early stop: val score did not improve for {patience} epochs "
                      f"(best at epoch {best_epoch}).")
                break
    print(f"Done. Best score={best:.4f} (epoch {best_epoch}). "
          f"Checkpoint: {os.path.join(work_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
