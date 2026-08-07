"""Train a published SOTA baseline on MCT-LTDiag under OUR exact protocol.

Fair apples-to-apples comparison for the Q1 table: every baseline is trained on the
SAME data, the SAME stratified 5-fold split (make_kfold_splits, seed from config), the
SAME preprocessing (MultiPhaseLiverDataset) and scored with the SAME metrics
(accuracy / macro-AUC / balanced-acc / macro-F1 / Cohen's kappa + per-class + confusion)
as our model. Only the network differs.

Baselines are pure classifiers (no seg prompt, no GT mask): they take the multi-phase CT
(+ optional clinical vector for STIC) and output 5-class logits -- fully automatic.

Examples (on the GPU box):
    # H-LSTM (ResNet-BiLSTM), 5-fold
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_baseline.py \
        --config configs/default.yaml --model hlstm --kfold 5 --amp \
        --resize 32,128,128 --work-dir baselines/hlstm

    # STIC (imaging + clinical); build the clinical CSV first with scripts/build_clinical.py
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_baseline.py \
        --config configs/default.yaml --model stic --kfold 5 --amp \
        --resize 32,128,128 --clinical-csv data/clinical.csv --work-dir baselines/stic
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from liver_sppvr.baselines import BASELINES, build_baseline
from liver_sppvr.data import (MultiPhaseLiverDataset, collate_multiphase, load_manifest)
from liver_sppvr.train.build import load_config, set_seed
from liver_sppvr.train.engine import _per_class_metrics
from liver_sppvr.train.losses import focal_ce_loss
from liver_sppvr.train.multitask import (class_weights, load_radiomics, make_kfold_splits,
                                         save_confusion, stratified_patient_split)
from liver_sppvr.utils.device import resolve_device


# --------------------------------------------------------------------------------------
def cls_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    """Same metric set as engine.evaluate, computed for a pure classifier."""
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                                 confusion_matrix, f1_score, roc_auc_score)
    labels = list(range(num_classes))
    yt, yp, pr = np.array(y_true), np.array(y_pred), np.array(y_prob)
    m = {"accuracy": float(np.mean(yt == yp)),
         "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
         "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
         "kappa": float(cohen_kappa_score(yt, yp, labels=labels)),
         "confusion": confusion_matrix(yt, yp, labels=labels).tolist()}
    try:
        m["auc"] = float(roc_auc_score(yt, pr, multi_class="ovr", average="macro", labels=labels))
    except Exception:
        m["auc"] = float("nan")
    m.update(_per_class_metrics(yt, yp, pr, num_classes))
    return m


@torch.no_grad()
def evaluate_baseline(model, loader, device, num_classes: int) -> dict:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        phases = batch["phases"].to(device)
        pp = batch["phase_present"].to(device)
        extra = batch.get("radiomics")
        extra = extra.to(device) if extra is not None else None
        logits = model(phases=phases, phase_present=pp, extra_feat=extra)
        probs = torch.softmax(logits.float(), dim=1)
        y_prob.extend(probs.cpu().tolist())
        y_pred.extend(probs.argmax(1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
    return cls_metrics(y_true, y_pred, y_prob, num_classes)


def train_fold(cfg, args, device, man, tr_ids, va_ids, labels_by_patient, work_dir, *,
               epochs, num_classes, use_amp, clinical) -> dict:
    tcfg = cfg["train"]
    pcfg = cfg.get("preprocess", {})
    phases = cfg["multiphase"]["phases"]
    clinical_dim = len(next(iter(clinical.values()))) if clinical else 0

    common = dict(class_names=cfg["classifier"]["class_names"], phases=phases,
                  spatial_size=cfg["segvol"]["spatial_size"], hu_window=cfg["multiphase"]["hu_window"],
                  normalize=pcfg.get("normalize", "hu"),
                  crop_foreground=pcfg.get("crop_foreground", False), radiomics=clinical)
    train_ds = MultiPhaseLiverDataset(man, patient_ids=tr_ids, augment=tcfg.get("augment", False),
                                      **common)
    val_ds = MultiPhaseLiverDataset(man, patient_ids=va_ids, **common)
    pin = getattr(device, "type", str(device)) == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_multiphase, num_workers=args.num_workers,
                              pin_memory=pin, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_multiphase, num_workers=args.num_workers,
                            pin_memory=pin, persistent_workers=args.num_workers > 0)

    model = build_baseline(args.model, num_classes, n_phases=len(phases),
                           clinical_dim=clinical_dim, resize=args.resize).to(device)
    cw = class_weights([labels_by_patient[p] for p in tr_ids], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    os.makedirs(work_dir, exist_ok=True)

    import time
    n_batches = len(train_loader)
    best, best_epoch, no_improve, best_ev = -1.0, -1, 0, {}
    for epoch in range(epochs):
        model.train()
        run = 0.0
        t0 = time.time()
        for bi, batch in enumerate(train_loader):
            phases_t = batch["phases"].to(device)
            pp = batch["phase_present"].to(device)
            extra = batch.get("radiomics")
            extra = extra.to(device) if extra is not None else None
            y = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(phases=phases_t, phase_present=pp, extra_feat=extra)
                loss = focal_ce_loss(logits, y, gamma=tcfg.get("focal_gamma", 2.0), weight=cw)
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            run += loss.item() * y.shape[0]
            if bi % args.log_interval == 0:      # visible per-batch progress (data loading is slow)
                sps = (bi + 1) * args.batch_size / max(time.time() - t0, 1e-6)
                print(f"  ep{epoch+1} [{bi+1}/{n_batches}] loss={loss.item():.4f} "
                      f"({sps:.1f} samp/s)", flush=True)
        print(f"  ep{epoch+1} evaluating on {len(va_ids)} val patients...", flush=True)
        ev = evaluate_baseline(model, val_loader, device, num_classes)
        print(f"[epoch {epoch+1}/{epochs}] loss={run/max(len(train_ds),1):.4f} | "
              f"acc={ev['accuracy']:.4f} bal_acc={ev['balanced_accuracy']:.4f} "
              f"macroF1={ev['macro_f1']:.4f} AUC={ev['auc']:.4f} kappa={ev['kappa']:.4f}")
        score = ev["auc"] if ev["auc"] == ev["auc"] else ev["macro_f1"]
        if score > best:
            best, best_epoch, no_improve, best_ev = score, epoch + 1, 0, dict(ev)
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg,
                        "baseline": args.model}, os.path.join(work_dir, "best.pth"))
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"Early stop @ epoch {epoch+1} (best {best:.4f} @ {best_epoch}).")
                break

    print(f"Done fold. best AUC/F1={best:.4f} @ epoch {best_epoch}")
    if "confusion" in best_ev:
        save_confusion(best_ev["confusion"], cfg["classifier"].get("class_names"),
                       os.path.join(work_dir, "confusion"), title=f"{args.model} (best epoch)")
    with open(os.path.join(work_dir, "metrics.json"), "w") as f:      # numeric metrics -> disk
        json.dump({"model": args.model, "best_epoch": best_epoch,
                   "class_names": cfg["classifier"].get("class_names"), **best_ev}, f, indent=2)
    del model, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"score": best, "epoch": best_epoch, **best_ev}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, choices=list(BASELINES))
    ap.add_argument("--manifest", default=None, help="override data.manifest")
    ap.add_argument("--clinical-csv", default=None,
                    help="per-patient clinical/tabular CSV (STIC); same format as radiomics CSV")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--fold", type=int, default=None, help="run only this fold index (0..k-1)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=10, help="print train progress every N batches")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resize", default=None, help="resize each phase to D,H,W (e.g. 32,128,128) to cut VRAM")
    ap.add_argument("--device", default=None)
    ap.add_argument("--work-dir", default="baselines/run")
    args = ap.parse_args()
    if args.resize:
        args.resize = tuple(int(x) for x in args.resize.split(","))

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg.get("device", "auto"))
    use_amp = args.amp and getattr(device, "type", str(device)) == "cuda"
    num_classes = cfg["classifier"]["num_classes"]
    manifest = args.manifest or cfg["data"]["manifest"]
    man = load_manifest(manifest)

    full = MultiPhaseLiverDataset(
        man, class_names=cfg["classifier"]["class_names"], phases=cfg["multiphase"]["phases"],
        spatial_size=cfg["segvol"]["spatial_size"], hu_window=cfg["multiphase"]["hu_window"])
    labels_by_patient = {p["patient_id"]: p["label"] for p in full._patients}
    seed = cfg["project"]["seed"]
    print(f"model={args.model} | patients={len(labels_by_patient)} | classes={num_classes} | "
          f"kfold={args.kfold} | resize={args.resize} | amp={use_amp} | device={device}")

    if args.model == "stic" and not args.clinical_csv:
        print("WARNING: STIC without --clinical-csv runs imaging-only (report as an ablation).")

    splits = (make_kfold_splits(labels_by_patient, k=args.kfold, seed=seed) if args.kfold > 1
              else [stratified_patient_split(labels_by_patient, seed=seed)])

    results = []
    for i, (tr_ids, va_ids) in enumerate(splits):
        if args.fold is not None and i != args.fold:
            continue
        # clinical/tabular features standardized on TRAIN patients only (no leakage), per fold
        clinical = load_radiomics(args.clinical_csv, tr_ids) if args.clinical_csv else None
        wd = os.path.join(args.work_dir, f"fold{i}") if args.kfold > 1 else args.work_dir
        print(f"\n===== {args.model} fold {i+1}/{args.kfold} "
              f"(train {len(tr_ids)}, val {len(va_ids)}) =====")
        m = train_fold(cfg, args, device, man, tr_ids, va_ids, labels_by_patient, wd,
                       epochs=args.epochs, num_classes=num_classes, use_amp=use_amp,
                       clinical=clinical)
        m["fold"] = i
        results.append(m)

    # --- persist the numeric summary (the comparison table), always ---
    os.makedirs(args.work_dir, exist_ok=True)
    keys = ("auc", "accuracy", "balanced_accuracy", "macro_f1", "kappa")

    def ms(key):
        v = np.array([r.get(key, float("nan")) for r in results], dtype=float)
        v = v[~np.isnan(v)]
        return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))

    summary = {"model": args.model, "kfold": args.kfold,
               "per_fold": [{"fold": r["fold"], **{k: r.get(k) for k in keys},
                             "best_epoch": r.get("epoch")} for r in results],
               "mean_std": {k: dict(zip(("mean", "std"), ms(k))) for k in keys}}
    with open(os.path.join(args.work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # a one-line-per-metric CSV that pastes straight into the paper table
    with open(os.path.join(args.work_dir, "summary.csv"), "w") as f:
        f.write("metric," + ",".join(f"fold{r['fold']}" for r in results) + ",mean,std\n")
        for k in keys:
            row = [f"{r.get(k, float('nan')):.4f}" for r in results]
            mu, sd = ms(k)
            f.write(f"{k}," + ",".join(row) + f",{mu:.4f},{sd:.4f}\n")
    print(f"\nsummary -> {args.work_dir}/summary.json + summary.csv")

    if len(results) > 1:
        print(f"===== {args.model}: cross-validation summary (best epoch per fold) =====")
        for k in keys:
            mu, sd = ms(k)
            print(f"  {k:<18} {mu:.4f} ± {sd:.4f}")
        conf = [np.array(r["confusion"]) for r in results if r.get("confusion")]
        if conf:
            save_confusion(np.sum(conf, axis=0), cfg["classifier"].get("class_names"),
                           os.path.join(args.work_dir, "confusion_total"),
                           title=f"{args.model} (out-of-fold, all patients)")
    return 0


if __name__ == "__main__":
    # 'spawn' start method: CUDA-safe and free of the fork+OpenMP deadlock that hangs
    # DataLoader workers on the first batch (fork copies locked BLAS/OpenMP mutexes).
    import torch.multiprocessing as _mp
    try:
        _mp.set_start_method("spawn")
    except RuntimeError:
        pass
    sys.exit(main())
