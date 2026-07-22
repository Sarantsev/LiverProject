"""Multi-task (seg + cls) training CLI. Single-device, config-driven.

Run on the GPU box:
    python -m liver_sppvr.train.multitask --config configs/default.yaml

Smoke-test the loop on CPU without weights/data:
    python -m liver_sppvr.train.multitask --config configs/default.yaml --dry-run
"""
from __future__ import annotations

import argparse
import gc
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


def load_radiomics(csv_path: str, train_ids):
    """Load per-patient radiomics CSV -> {patient_id: np.float32 vector}, or None if disabled.

    Features are z-scored using TRAIN patients only (no leakage); NaNs -> 0.
    CSV columns: patient_id, <feature columns...>, [tumor_type] (built by
    scripts/extract_radiomics.py).
    """
    if not csv_path or not os.path.exists(csv_path):
        return None
    import numpy as np
    import pandas as pd
    df = pd.read_csv(csv_path).set_index("patient_id")
    feat_cols = [c for c in df.columns if c != "tumor_type"]
    train_rows = df.loc[df.index.intersection(list(train_ids)), feat_cols]
    mu = train_rows.mean()
    sd = train_rows.std().replace(0, 1.0)
    norm = ((df[feat_cols] - mu) / sd).fillna(0.0)
    return {pid: norm.loc[pid].to_numpy(dtype="float32") for pid in norm.index}


def fit_radiomics_classifier(radiomics: dict, train_ids, labels_by_patient, kind: str = "rf"):
    """Fit a RandomForest/SVM on train patients' (standardized) radiomics -> for late fusion."""
    import numpy as np
    ids = [p for p in train_ids if p in radiomics]
    X = np.stack([radiomics[p] for p in ids])
    y = np.array([labels_by_patient[p] for p in ids])
    if kind == "svm":
        from sklearn.svm import SVC
        clf = SVC(probability=True, class_weight="balanced", random_state=0)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     n_jobs=-1, random_state=0)
    clf.fit(X, y)
    return clf


def class_weights(labels, num_classes: int) -> torch.Tensor:
    cnt = Counter(labels)
    w = torch.tensor([1.0 / max(cnt.get(c, 0), 1) for c in range(num_classes)])
    return w / w.sum() * num_classes


def make_kfold_splits(labels_by_patient: dict, k: int = 5, seed: int = 2023):
    """Stratified k-fold on patients -> list of (train_ids, val_ids). Each patient is
    used for validation exactly once; class proportions are preserved per fold."""
    import random as _r
    rng = _r.Random(seed)
    by_cls: dict = {}
    for pid, y in labels_by_patient.items():
        by_cls.setdefault(y, []).append(pid)
    folds_val = [[] for _ in range(k)]
    for y, pids in by_cls.items():
        pids = pids[:]; rng.shuffle(pids)
        for i, pid in enumerate(pids):
            folds_val[i % k].append(pid)            # round-robin -> balanced fold sizes
    all_ids = list(labels_by_patient.keys())
    return [([p for p in all_ids if p not in set(va)], va) for va in folds_val]


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


def _print_per_class(ev: dict, class_names) -> None:
    """Print per-class sensitivity/specificity/F1/AUC + Cohen's kappa (benchmark protocol)."""
    if "per_f1" not in ev:
        return
    n = len(ev["per_f1"])
    names = list(class_names) if class_names else [f"c{i}" for i in range(n)]
    print("  per-class:  " + f"{'class':<8} {'sens':>6} {'spec':>6} {'F1':>6} {'AUC':>6}")
    for i in range(n):
        print(f"    {names[i]:<8} "
              f"{ev['per_sens'][i]:>6.3f} {ev['per_spec'][i]:>6.3f} "
              f"{ev['per_f1'][i]:>6.3f} {ev['per_auc'][i]:>6.3f}")
    print(f"  Cohen's kappa = {ev.get('kappa', float('nan')):.4f}")


# --------- shared training loop (one split) ---------
def _train_loop(cfg, args, device, model, train_ds, val_ds, train_labels, work_dir, *,
                epochs, batch_size, num_workers, use_amp, num_classes, rf=None, rf_alpha=0.5) -> dict:
    """Train on one split; save best.pth to work_dir; return the best-epoch metrics."""
    tcfg = cfg["train"]
    pin = getattr(device, "type", str(device)) == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=num_workers,
                              pin_memory=pin, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate, num_workers=num_workers,
                            pin_memory=pin, persistent_workers=num_workers > 0)

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

    best, best_epoch, no_improve, best_ev = -1.0, -1, 0, {}
    for epoch in range(epochs):
        if not args.dry_run and unfreeze_epoch >= 0 and epoch == unfreeze_epoch:
            from .build import set_encoder_trainable
            set_encoder_trainable(model, True)
            print(f"[epoch {epoch+1}] image_encoder unfrozen -- fine-tuning from here")
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler=scaler,
                             seg_prompt_mode=seg_prompt_mode, bbox_shift=bbox_shift)
        scheduler.step()
        ev = evaluate(model, val_loader, device, num_classes=num_classes,
                      seg_prompt_mode=seg_prompt_mode, rf=rf, rf_alpha=rf_alpha)
        print(f"[epoch {epoch+1}/{epochs}] train_loss={tr['loss']:.4f} "
              f"seg={tr['seg_loss']:.4f} cls={tr['cls_loss']:.4f} | "
              f"val_dice={ev.get('dice', float('nan')):.4f} | "
              f"acc={ev.get('accuracy', float('nan')):.4f} "
              f"bal_acc={ev.get('balanced_accuracy', float('nan')):.4f} "
              f"macroF1={ev.get('macro_f1', float('nan')):.4f} "
              f"AUC={ev.get('auc', float('nan')):.4f} "
              f"kappa={ev.get('kappa', float('nan')):.4f}")
        # select checkpoint by AUC (metric used in the literature); fall back to macro-F1
        score = ev.get("auc")
        if score is None or score != score:   # None or NaN
            score = ev.get("macro_f1", ev.get("accuracy", 0.0))
        if score is not None and score > best:
            best, best_epoch, no_improve, best_ev = score, epoch + 1, 0, dict(ev)
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
    print(f"  best epoch metrics: acc={best_ev.get('accuracy', float('nan')):.4f} "
          f"bal_acc={best_ev.get('balanced_accuracy', float('nan')):.4f} "
          f"macroF1={best_ev.get('macro_f1', float('nan')):.4f} "
          f"AUC={best_ev.get('auc', float('nan')):.4f}")
    _print_per_class(best_ev, cfg["classifier"].get("class_names"))
    return {"score": best, "epoch": best_epoch, **best_ev}


def _build_and_train_fold(cfg, args, device, man, tr_ids, va_ids, labels_by_patient, work_dir, *,
                          epochs, batch_size, num_workers, use_amp, num_classes) -> dict:
    """Build datasets + model for one split, then train. Returns best-epoch metrics."""
    from ..data import MultiPhaseLiverDataset
    tcfg = cfg["train"]
    pcfg = cfg.get("preprocess", {})
    zoom_margin = tcfg.get("zoom_margin", 0.5)
    # optional radiomics fusion: standardize per-patient features on TRAIN only (no leakage)
    dcfg = cfg.get("data", {})
    radiomics = load_radiomics(dcfg.get("radiomics_csv", ""), tr_ids)
    rf, rf_alpha = None, float(dcfg.get("radiomics_alpha", 0.5))
    if radiomics is not None:
        fusion = dcfg.get("radiomics_fusion", "early")
        if fusion == "late":
            cfg["classifier"]["extra_feat_dim"] = 0       # deep uses images only
            rf = fit_radiomics_classifier(radiomics, tr_ids, labels_by_patient,
                                          kind=dcfg.get("radiomics_clf", "rf"))
            print(f"radiomics LATE fusion: {dcfg.get('radiomics_clf', 'rf').upper()} "
                  f"blended, alpha={rf_alpha}")
        else:  # early
            cfg["classifier"]["extra_feat_dim"] = len(next(iter(radiomics.values())))
            print(f"radiomics EARLY fusion: {len(next(iter(radiomics.values())))} features")
    common = dict(class_names=cfg["classifier"]["class_names"],
                  phases=cfg["multiphase"]["phases"],
                  spatial_size=cfg["segvol"]["spatial_size"],
                  hu_window=cfg["multiphase"]["hu_window"],
                  normalize=pcfg.get("normalize", "hu"),
                  crop_foreground=pcfg.get("crop_foreground", False), radiomics=radiomics)
    train_ds = MultiPhaseLiverDataset(man, patient_ids=tr_ids, augment=tcfg.get("augment", False),
                                      zoom=tcfg.get("zoom", 0.0), zoom_margin=zoom_margin, **common)
    val_ds = MultiPhaseLiverDataset(man, patient_ids=va_ids,
                                    zoom=(1.0 if tcfg.get("zoom_eval", False) else 0.0),
                                    zoom_margin=zoom_margin, **common)
    model = build_segvol_multitask(cfg, device)
    model._amp_enabled = use_amp    # autocast inside forward (needed for DataParallel)
    if torch.cuda.device_count() > 1:
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    train_labels = [labels_by_patient[p] for p in tr_ids]
    result = _train_loop(cfg, args, device, model, train_ds, val_ds, train_labels, work_dir,
                         epochs=epochs, batch_size=batch_size, num_workers=num_workers,
                         use_amp=use_amp, num_classes=num_classes, rf=rf, rf_alpha=rf_alpha)
    # free this fold's model + cached GPU memory before the next fold (avoid OOM accumulation).
    # PyTorch models hold reference cycles, so an explicit gc.collect() is needed to release them.
    del model, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


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
    ap.add_argument("--lora", action="store_true", help="enable PEFT on the encoder (overrides segvol.lora.enabled)")
    ap.add_argument("--lora-variant", choices=["lora", "dora"], default=None,
                    help="PEFT variant (overrides segvol.lora.variant)")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="PEFT rank -- lower = less capacity, less overfit (alpha set to 2*rank)")
    ap.add_argument("--lora-targets", default=None,
                    help="comma-separated Linear name substrings to adapt, e.g. 'qkv' or "
                         "'qkv,out_proj' (fewer = less capacity). Overrides segvol.lora.targets")
    ap.add_argument("--lora-dropout", type=float, default=None,
                    help="PEFT dropout -- higher = more regularization (overrides segvol.lora.dropout)")
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
    ap.add_argument("--radiomics-csv", default=None,
                    help="path to per-patient radiomics CSV to fuse into the classifier "
                         "(overrides data.radiomics_csv)")
    ap.add_argument("--radiomics-fusion", choices=["early", "late"], default=None,
                    help="how radiomics enters the classifier (overrides data.radiomics_fusion)")
    ap.add_argument("--fusion", choices=["attention", "cross_attention", "concat_stem"], default=None,
                    help="multi-phase fusion mode (overrides multiphase.fusion)")
    ap.add_argument("--kfold", type=int, default=1,
                    help="stratified k-fold cross-validation (1 = single split)")
    ap.add_argument("--fold", type=int, default=None,
                    help="with --kfold, run only this fold index (0..k-1); default = all folds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.lora:
        cfg.setdefault("segvol", {}).setdefault("lora", {})["enabled"] = True
    if args.lora_variant:
        cfg.setdefault("segvol", {}).setdefault("lora", {})["variant"] = args.lora_variant
    _lcfg = cfg.setdefault("segvol", {}).setdefault("lora", {})
    if args.lora_rank is not None:
        _lcfg["rank"] = args.lora_rank
        _lcfg["alpha"] = 2 * args.lora_rank          # keep alpha = 2*rank
    if args.lora_targets:
        _lcfg["targets"] = [t.strip() for t in args.lora_targets.split(",")]
    if args.lora_dropout is not None:
        _lcfg["dropout"] = args.lora_dropout
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
    if args.radiomics_csv is not None:
        cfg.setdefault("data", {})["radiomics_csv"] = args.radiomics_csv
    if args.fusion:
        cfg["multiphase"]["fusion"] = args.fusion
    if args.radiomics_fusion:
        cfg.setdefault("data", {})["radiomics_fusion"] = args.radiomics_fusion
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

    # config summary (once)
    pcfg = cfg.get("preprocess", {})
    seg_prompt_mode = tcfg.get("seg_prompt", "text")
    if not args.dry_run:
        print(f"seg_prompt={seg_prompt_mode}"
              + (f" (bbox_shift={tcfg.get('bbox_shift', 0)})" if seg_prompt_mode == "box" else "")
              + f" | normalize={pcfg.get('normalize', 'hu')} crop_fg={pcfg.get('crop_foreground', False)}"
              + f" | zoom train_p={tcfg.get('zoom', 0.0)} eval={'on' if tcfg.get('zoom_eval', False) else 'off'}")

    common_kw = dict(epochs=epochs, batch_size=batch_size, num_workers=num_workers,
                     use_amp=use_amp, num_classes=num_classes)

    # --- dry-run: synthetic data + stub, single split ---
    if args.dry_run:
        train_ds = _SyntheticDataset(n=8, n_phases=len(cfg["multiphase"]["phases"]), num_classes=num_classes)
        val_ds = _SyntheticDataset(n=4, n_phases=len(cfg["multiphase"]["phases"]), num_classes=num_classes)
        model = build_dryrun_model(cfg).to(device)
        model._amp_enabled = use_amp
        train_labels = [train_ds[i]["label"].item() for i in range(len(train_ds))]
        _train_loop(cfg, args, device, model, train_ds, val_ds, train_labels, work_dir, **common_kw)
        return

    # --- real data ---
    from ..data import load_manifest, MultiPhaseLiverDataset
    man = load_manifest(cfg["data"]["manifest"])
    full = MultiPhaseLiverDataset(
        man, class_names=cfg["classifier"]["class_names"], phases=cfg["multiphase"]["phases"],
        spatial_size=cfg["segvol"]["spatial_size"], hu_window=cfg["multiphase"]["hu_window"])
    labels_by_patient = {p["patient_id"]: p["label"] for p in full._patients}
    seed = cfg["project"]["seed"]

    if args.kfold <= 1:
        tr_ids, va_ids = stratified_patient_split(labels_by_patient, seed=seed)
        _build_and_train_fold(cfg, args, device, man, tr_ids, va_ids, labels_by_patient,
                              work_dir, **common_kw)
        return

    # --- k-fold cross-validation ---
    splits = make_kfold_splits(labels_by_patient, k=args.kfold, seed=seed)
    results = []
    for i, (tr_ids, va_ids) in enumerate(splits):
        if args.fold is not None and i != args.fold:
            continue
        print(f"\n===== fold {i+1}/{args.kfold} (train {len(tr_ids)}, val {len(va_ids)}) =====")
        m = _build_and_train_fold(cfg, args, device, man, tr_ids, va_ids, labels_by_patient,
                                  os.path.join(work_dir, f"fold{i}"), **common_kw)
        m["fold"] = i
        results.append(m)

    if len(results) > 1:
        import numpy as np
        print("\n===== cross-validation summary (best epoch per fold) =====")
        for m in results:
            print(f"  fold {m['fold']}: AUC={m.get('auc', float('nan')):.4f} "
                  f"acc={m.get('accuracy', float('nan')):.4f} "
                  f"bal_acc={m.get('balanced_accuracy', float('nan')):.4f} "
                  f"macroF1={m.get('macro_f1', float('nan')):.4f} "
                  f"kappa={m.get('kappa', float('nan')):.4f} "
                  f"dice={m.get('dice', float('nan')):.4f}")

        def _ms(vals):
            v = np.array(vals, dtype=float); v = v[~np.isnan(v)]
            return (v.mean(), v.std()) if len(v) else (float('nan'), float('nan'))

        print("  --- mean ± std ---")
        for key in ("auc", "accuracy", "balanced_accuracy", "macro_f1", "kappa", "dice"):
            mu, sd = _ms([m.get(key, float('nan')) for m in results])
            print(f"  {key:<18} {mu:.4f} ± {sd:.4f}")

        # per-class mean ± std across folds (benchmark protocol)
        names = cfg["classifier"].get("class_names") or []
        if results and "per_f1" in results[0]:
            nC = len(results[0]["per_f1"])
            names = names or [f"c{i}" for i in range(nC)]
            print("  --- per-class (mean ± std across folds) ---")
            print(f"    {'class':<8} {'sens':>13} {'spec':>13} {'F1':>13} {'AUC':>13}")
            for c in range(nC):
                cells = []
                for key in ("per_sens", "per_spec", "per_f1", "per_auc"):
                    mu, sd = _ms([m.get(key, [float('nan')]*nC)[c] for m in results])
                    cells.append(f"{mu:.2f}±{sd:.2f}")
                print(f"    {names[c]:<8} " + " ".join(f"{x:>13}" for x in cells))


if __name__ == "__main__":
    main()
