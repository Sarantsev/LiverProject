"""ONE trainer for every tumour-type classifier: the 5 published baselines AND ours.

Same data, same stratified patient-level folds (seed from config), same preprocessing
rules, same metrics (accuracy / macro-AUC / balanced-acc / macro-F1 / Cohen's kappa +
per-class + confusion) -> the rows of the paper's tables are apples-to-apples; only the
network differs.

--model ours: pretrained backbone (--backbone segvol|merlin) + phase fusion + head.
  --mask-source none  image-only (global average pooling)
  --mask-source gt    masked pooling on the radiologist's mask  = semi-automatic upper bound
  --mask-source pred  masked pooling on the segmentation model's OUT-OF-FOLD masks
                      (--pred-masks) = fully automatic, the headline setting. Train-time
                      --mask-noise makes the head robust to segmentation errors.
  --clinical-csv      early fusion of the 5 objective clinical features (any model that
                      accepts extra_feat: ours, stic, racmformer).

Examples (GPU box):
  PYTHONPATH=. python scripts/train_cls.py --config configs/default.yaml --model sdrformer \
      --kfold 5 --amp --resize 32,128,128 --work-dir work/sdrformer
  PYTHONPATH=. python scripts/train_cls.py --config configs/default.yaml --model ours \
      --backbone segvol --mask-source pred --pred-masks work/segA/oof_masks/pred_masks.csv \
      --clinical-csv data/clinical.csv --kfold 5 --amp --work-dir work/clsB_segvol_pred
  # prove the folds are identical across pipelines (no model is built):
  PYTHONPATH=. python scripts/train_cls.py --config configs/default.yaml --model ours \
      --kfold 5 --dump-splits work/splits_cls.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from liver_sppvr.baselines import BASELINES, build_baseline
from liver_sppvr.data import MultiPhaseLiverDataset, collate_multiphase, load_manifest
from liver_sppvr.data.clinical import feat_dim, load_clinical
from liver_sppvr.data.dataset import load_pred_masks
from liver_sppvr.train.build import (build_classifier, class_weights, load_config,
                                     make_scheduler, set_seed)
from liver_sppvr.train.losses import ClsLoss
from liver_sppvr.train.metrics import cls_metrics, print_per_class, save_confusion, write_summary
from liver_sppvr.train.splits import (dump_splits, labels_from_manifest, make_kfold_splits,
                                      stratified_patient_split)
from liver_sppvr.utils.device import resolve_device

MODELS = (*BASELINES, "ours")
MASK_SOURCES = ("none", "gt", "pred")


def _forward(model, batch, device, *, use_mask: bool, want_proj: bool):
    kw = dict(phases=batch["phases"].to(device), phase_present=batch["phase_present"].to(device))
    if batch.get("clinical") is not None:
        kw["extra_feat"] = batch["clinical"].to(device)
    if use_mask:
        kw["mask"] = batch["mask"].to(device)
    if want_proj:
        kw["return_proj"] = True
    return model(**kw)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, *, use_mask: bool) -> dict:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        logits = _forward(model, batch, device, use_mask=use_mask, want_proj=False)
        probs = torch.softmax(logits.float(), dim=1)
        y_prob.extend(probs.cpu().tolist())
        y_pred.extend(probs.argmax(1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
    return cls_metrics(y_true, y_pred, y_prob, num_classes)


def train_fold(cfg, args, device, man, tr_ids, va_ids, labels_by_patient, work_dir, *,
               clinical, pred_masks) -> dict:
    tcfg, pcfg = cfg["train"], cfg.get("preprocess", {})
    phases = cfg["multiphase"]["phases"]
    num_classes = cfg["classifier"]["num_classes"]
    use_amp = args.amp and getattr(device, "type", str(device)) == "cuda"
    ours = args.model == "ours"

    # ---- model (built first: it declares the preprocessing it was pretrained with) ----
    if ours:
        core = build_classifier(cfg, device, backbone=args.backbone, finetune=args.finetune,
                                mask_source=args.mask_source, clinical_dim=feat_dim(clinical))
        spec = core.backbone.preprocess_spec
        use_mask = core.uses_mask
        core.amp_in_forward = use_amp
    else:
        core = build_baseline(args.model, num_classes, n_phases=len(phases),
                              clinical_dim=feat_dim(clinical), resize=args.resize).to(device)
        sv = cfg["segvol"]
        spec = {"spatial_size": tuple(sv["spatial_size"]), "normalize": sv.get("normalize", "foreground"),
                "hu_window": tuple(sv.get("hu_window", (-175, 250)))}
        use_mask = False
    model = core
    if args.dp and torch.cuda.device_count() > 1:
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(core)

    # ---- data ----
    roi = args.roi or ("pred" if (use_mask and args.mask_source == "pred") else pcfg.get("roi", "body"))
    override = pred_masks if (use_mask and args.mask_source == "pred") else None
    common = dict(class_names=cfg["classifier"]["class_names"], phases=phases,
                  spatial_size=spec["spatial_size"], hu_window=spec["hu_window"],
                  normalize=spec["normalize"], roi=roi, roi_margin=pcfg.get("roi_margin", 0.5),
                  mask_override=override, clinical=clinical)
    train_ds = MultiPhaseLiverDataset(man, patient_ids=tr_ids, augment=tcfg.get("augment", False),
                                      mask_noise=(args.mask_noise if use_mask else 0.0), **common)
    val_ds = MultiPhaseLiverDataset(man, patient_ids=va_ids, **common)
    pin = getattr(device, "type", str(device)) == "cuda"
    dl = dict(collate_fn=collate_multiphase, num_workers=args.num_workers, pin_memory=pin,
              persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl)
    print(f"data: roi={roi} masks={'OOF-pred' if override else ('GT' if use_mask else '-')} "
          f"clinical={feat_dim(clinical)} spatial={spec['spatial_size']} norm={spec['normalize']}")

    # ---- optimisation ----
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, tcfg.get("warmup_epoch", 0), args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    supcon = args.supcon if ours else 0.0
    loss_fn = ClsLoss(class_weights([labels_by_patient[p] for p in tr_ids], num_classes),
                      focal_gamma=tcfg.get("focal_gamma", 2.0), supcon_weight=supcon,
                      supcon_temp=tcfg.get("supcon_temp", 0.1),
                      queue_size=tcfg.get("supcon_queue", 512)).to(device)
    os.makedirs(work_dir, exist_ok=True)

    n_batches = len(train_loader)
    best, best_epoch, no_improve, best_ev = -1.0, -1, 0, {}
    for epoch in range(args.epochs):
        model.train()
        run, t0 = 0.0, time.time()
        for bi, batch in enumerate(train_loader):
            y = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = _forward(model, batch, device, use_mask=use_mask, want_proj=supcon > 0)
                logits, proj = out if isinstance(out, tuple) else (out, None)
                losses = loss_fn(logits, y, proj)
                loss = losses["loss"]
                aux = getattr(core, "last_aux", None)            # e.g. LCA-DB attention-similarity
                if aux is not None:
                    loss = loss + aux
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
            run += loss.item() * y.shape[0]
            if bi % args.log_interval == 0:
                sps = (bi + 1) * args.batch_size / max(time.time() - t0, 1e-6)
                print(f"  ep{epoch+1} [{bi+1}/{n_batches}] loss={loss.item():.4f} "
                      f"cls={float(losses['cls_loss']):.4f} con={float(losses['con_loss']):.4f} "
                      f"({sps:.1f} samp/s)", flush=True)
        scheduler.step()
        ev = evaluate(model, val_loader, device, num_classes, use_mask=use_mask)
        print(f"[epoch {epoch+1}/{args.epochs}] loss={run/max(len(train_ds),1):.4f} | "
              f"acc={ev['accuracy']:.4f} bal_acc={ev['balanced_accuracy']:.4f} "
              f"macroF1={ev['macro_f1']:.4f} AUC={ev['auc']:.4f} kappa={ev['kappa']:.4f}")
        score = ev["auc"] if ev["auc"] == ev["auc"] else ev["macro_f1"]
        if score > best:
            best, best_epoch, no_improve, best_ev = score, epoch + 1, 0, dict(ev)
            torch.save({"epoch": epoch, "model": core.state_dict(), "config": cfg,
                        "args": vars(args)}, os.path.join(work_dir, "best.pth"))
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"Early stop @ epoch {epoch+1} (best {best:.4f} @ {best_epoch}).")
                break

    print(f"Done fold. best AUC/F1={best:.4f} @ epoch {best_epoch}")
    print_per_class(best_ev, cfg["classifier"].get("class_names"))
    if "confusion" in best_ev:
        save_confusion(best_ev["confusion"], cfg["classifier"].get("class_names"),
                       os.path.join(work_dir, "confusion"), title=f"{args.run_name} (best epoch)")
    with open(os.path.join(work_dir, "metrics.json"), "w") as f:
        json.dump({"model": args.run_name, "best_epoch": best_epoch,
                   "class_names": cfg["classifier"].get("class_names"), **best_ev}, f, indent=2)
    del model, core, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"score": best, "epoch": best_epoch, **best_ev}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--manifest", default=None, help="override data.manifest")
    # ours
    ap.add_argument("--backbone", default=None, help="segvol | merlin (ours; default from config)")
    ap.add_argument("--finetune", default=None, help="dora|lora|frozen|full|partial (ours; default from config)")
    ap.add_argument("--mask-source", default=None, choices=list(MASK_SOURCES),
                    help="ours: none | gt | pred (default from config)")
    ap.add_argument("--pred-masks", default=None, help="pred_masks.csv (out-of-fold masks) for --mask-source pred")
    ap.add_argument("--mask-noise", type=float, default=None, help="train-time mask perturbation prob (default cfg)")
    ap.add_argument("--roi", default=None, choices=("none", "body", "pred"),
                    help="ROI crop; default: pred when --mask-source pred, else config preprocess.roi")
    ap.add_argument("--supcon", type=float, default=None, help="SupCon weight (ours; default cfg)")
    ap.add_argument("--clinical-csv", default=None, help="per-patient clinical CSV (default cfg data.clinical_csv)")
    # protocol
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--fold", type=int, default=None, help="run only this fold index (0..k-1)")
    ap.add_argument("--dump-splits", default=None, help="write the folds to this JSON and exit")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--dp", action="store_true", help="DataParallel over all visible GPUs")
    ap.add_argument("--resize", default=None, help="baselines only: resize phases to D,H,W (e.g. 32,128,128)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--work-dir", default="work/run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg, ccfg = cfg["train"], cfg["classifier"]
    for k, key in (("epochs", "num_epochs"), ("batch_size", "batch_size"), ("num_workers", "num_workers"),
                   ("lr", "lr"), ("weight_decay", "weight_decay"), ("patience", "early_stop_patience"),
                   ("mask_noise", "mask_noise"), ("supcon", "supcon")):
        if getattr(args, k) is None:
            setattr(args, k, tcfg[key])
    args.backbone = args.backbone or ccfg.get("backbone", "segvol")
    args.finetune = args.finetune or ccfg.get("finetune", "dora")
    args.mask_source = args.mask_source or ccfg.get("mask_source", "pred")
    args.clinical_csv = args.clinical_csv or cfg["data"].get("clinical_csv") or None
    args.pred_masks = args.pred_masks or cfg["data"].get("pred_masks_csv") or None
    args.resize = tuple(int(x) for x in args.resize.split(",")) if args.resize else None
    args.run_name = (f"ours-{args.backbone}-{args.mask_source}" if args.model == "ours" else args.model)
    if args.model == "ours" and args.mask_source == "pred" and not args.pred_masks and not args.dump_splits:
        ap.error("--mask-source pred needs --pred-masks (out-of-fold predicted masks)")

    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg.get("device", "auto"))
    man = load_manifest(args.manifest or cfg["data"]["manifest"])
    labels_by_patient = labels_from_manifest(man, ccfg["class_names"])
    seed = cfg["project"]["seed"]
    splits = (make_kfold_splits(labels_by_patient, k=args.kfold, seed=seed) if args.kfold > 1
              else [stratified_patient_split(labels_by_patient, seed=seed)])
    if args.dump_splits:
        dump_splits(splits, args.dump_splits)
        print(f"splits -> {args.dump_splits} ({len(labels_by_patient)} patients, k={args.kfold}, seed={seed})")
        return 0
    print(f"model={args.run_name} | patients={len(labels_by_patient)} | classes={ccfg['num_classes']} | "
          f"kfold={args.kfold} | clinical={'yes' if args.clinical_csv else 'no'} | device={device}")
    if args.model == "stic" and not args.clinical_csv:
        print("WARNING: STIC without --clinical-csv runs imaging-only (report as an ablation).")
    pred_masks = load_pred_masks(args.pred_masks) if (args.model == "ours" and args.mask_source == "pred") else None

    results = []
    for i, (tr_ids, va_ids) in enumerate(splits):
        if args.fold is not None and i != args.fold:
            continue
        clinical = load_clinical(args.clinical_csv, tr_ids)     # z-scored on THIS fold's train ids
        wd = os.path.join(args.work_dir, f"fold{i}") if args.kfold > 1 else args.work_dir
        print(f"\n===== {args.run_name} fold {i+1}/{args.kfold} (train {len(tr_ids)}, val {len(va_ids)}) =====")
        m = train_fold(cfg, args, device, man, tr_ids, va_ids, labels_by_patient, wd,
                       clinical=clinical, pred_masks=pred_masks)
        m["fold"] = i
        results.append(m)

    write_summary(results, args.work_dir, args.run_name, args.kfold, ccfg.get("class_names"))
    return 0


if __name__ == "__main__":
    # 'spawn': CUDA-safe and free of the fork+OpenMP deadlock that hangs DataLoader workers.
    import torch.multiprocessing as _mp
    try:
        _mp.set_start_method("spawn")
    except RuntimeError:
        pass
    sys.exit(main())
