"""Segmentation-only fine-tuning on LiTS (train/val split by volume).

Unlike liver_sppvr.train.multitask (which stratifies a k-fold by tumor TYPE and
selects the checkpoint by classification AUC), LiTS has a single dummy class, so:
  * the split is a plain by-volume train/val split (no stratification);
  * the loss is segmentation-only (cls_weight = 0, con_weight = 0);
  * the best checkpoint is selected by val Dice.

Everything else reuses the normal pipeline (dataset, SegVol multitask model, engine
train_one_epoch / evaluate). Segmentation still runs on the single available phase
(portal). The classification head exists but is not trained and is ignored.

NB on resolution: the pipeline resamples each volume to spatial_size (default
32x256x256). That coarse depth caps volumetric tumor Dice vs sliding-window methods
(DPC-Net etc. use 128^3 patches at native spacing). This script gives an honest
train/val Dice for OUR setup; it is not a native-resolution challenge submission.

Example (on the GPU box):
    CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
      python scripts/train_lits.py --config configs/default.yaml \
        --manifest data/lits_manifest.csv --prompt box --epochs 60 \
        --batch-size 2 --num-workers 6 --amp --work-dir cv_lits_box --lora
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from liver_sppvr.data import MultiPhaseLiverDataset, collate_multiphase, load_manifest
from liver_sppvr.train.build import build_segvol_multitask, load_config, set_seed
from liver_sppvr.train.engine import evaluate, train_one_epoch
from liver_sppvr.train.losses import MultiTaskLoss
from liver_sppvr.utils.device import resolve_device


def split_by_volume(patient_ids, val_frac: float, seed: int):
    ids = sorted(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val = set(ids[:n_val])
    train = [p for p in ids if p not in val]
    return train, sorted(val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True, help="LiTS manifest CSV")
    ap.add_argument("--prompt", choices=["box", "text"], default="box",
                    help="segmentation prompt during train+eval (box = from GT mask, semi-auto)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--work-dir", default="cv_lits")
    ap.add_argument("--device", default=None)
    ap.add_argument("--lora", action="store_true", help="enable PEFT (LoRA/DoRA) on the encoder")
    ap.add_argument("--zoom", type=float, default=0.0, help="p of zoom-in crop on train [0..1]")
    ap.add_argument("--zoom-eval", action="store_true", help="zoom-in crop around GT on eval")
    ap.add_argument("--patience", type=int, default=15, help="early stop on val Dice (0 = off)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.lora:
        cfg.setdefault("segvol", {}).setdefault("lora", {})["enabled"] = True
    cfg.setdefault("classifier", {})["extra_feat_dim"] = 0     # no radiomics for LiTS
    set_seed(args.seed)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    use_amp = args.amp and getattr(device, "type", str(device)) == "cuda"

    pcfg = cfg.get("preprocess", {})
    tcfg = cfg.get("train", {})
    num_classes = cfg["classifier"]["num_classes"]

    man = load_manifest(args.manifest)
    # discover patient ids present for the configured classes
    probe = MultiPhaseLiverDataset(
        man, class_names=cfg["classifier"]["class_names"], phases=cfg["multiphase"]["phases"],
        spatial_size=cfg["segvol"]["spatial_size"], hu_window=cfg["multiphase"]["hu_window"])
    all_ids = [p["patient_id"] for p in probe._patients]
    tr_ids, va_ids = split_by_volume(all_ids, args.val_frac, args.seed)
    print(f"LiTS volumes: {len(all_ids)} -> train {len(tr_ids)} / val {len(va_ids)} | "
          f"prompt={args.prompt} amp={use_amp} lora={args.lora} "
          f"normalize={pcfg.get('normalize', 'hu')} crop_fg={pcfg.get('crop_foreground', False)}")

    common = dict(class_names=cfg["classifier"]["class_names"], phases=cfg["multiphase"]["phases"],
                  spatial_size=cfg["segvol"]["spatial_size"], hu_window=cfg["multiphase"]["hu_window"],
                  normalize=pcfg.get("normalize", "hu"),
                  crop_foreground=pcfg.get("crop_foreground", False))
    train_ds = MultiPhaseLiverDataset(man, patient_ids=tr_ids, augment=True,
                                      zoom=args.zoom, zoom_margin=tcfg.get("zoom_margin", 0.5), **common)
    val_ds = MultiPhaseLiverDataset(man, patient_ids=va_ids,
                                    zoom=(1.0 if args.zoom_eval else 0.0),
                                    zoom_margin=tcfg.get("zoom_margin", 0.5), **common)

    pin = getattr(device, "type", str(device)) == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_multiphase, num_workers=args.num_workers,
                              pin_memory=pin, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_multiphase, num_workers=args.num_workers,
                            pin_memory=pin, persistent_workers=args.num_workers > 0)

    model = build_segvol_multitask(cfg, device)
    model._amp_enabled = use_amp
    if torch.cuda.device_count() > 1:
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # segmentation-only objective
    loss_fn = MultiTaskLoss(seg_weight=1.0, cls_weight=0.0, con_weight=0.0).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    os.makedirs(args.work_dir, exist_ok=True)

    best, best_epoch, no_improve = -1.0, -1, 0
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler=scaler,
                             seg_prompt_mode=args.prompt, bbox_shift=tcfg.get("bbox_shift", 0))
        ev = evaluate(model, val_loader, device, num_classes=num_classes,
                      seg_prompt_mode=args.prompt)
        dice = ev.get("dice", float("nan"))
        print(f"[epoch {epoch+1}/{args.epochs}] train_seg_loss={tr['seg_loss']:.4f} | "
              f"val_dice={dice:.4f}")
        if dice == dice and dice > best:       # not NaN and improved
            best, best_epoch, no_improve = dice, epoch + 1, 0
            _m = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({"epoch": epoch, "model": _m.state_dict(), "config": cfg},
                       os.path.join(args.work_dir, "best.pth"))
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"Early stop: val Dice did not improve for {args.patience} epochs "
                      f"(best {best:.4f} @ epoch {best_epoch}).")
                break

    print(f"\nDone. Best val Dice = {best:.4f} (epoch {best_epoch}). "
          f"Checkpoint: {os.path.join(args.work_dir, 'best.pth')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
