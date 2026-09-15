"""Stage C -- fully automatic end-to-end evaluation: CT -> nnU-Net mask -> tumour type.

For every fold i the classifier trained on fold i (work/clsB_*/fold{i}/best.pth, trained
with --mask-source pred) is applied to fold i's validation patients using the nnU-Net
OUT-OF-FOLD masks (Stage A): ROI crop around the predicted mask, masked pooling on it,
clinical features z-scored with fold i's train statistics. Ground truth is touched only
when computing metrics. Writes per-patient predictions with the mask's native Dice, the
usual classification summary, and the classifier's sensitivity to segmentation quality
(metrics per Dice bin) -- the honest number for the paper.

    PYTHONPATH=. python scripts/infer_cascade.py --config configs/default.yaml \
        --cls-dir work/clsB_segvol_pred --pred-masks work/segA/oof_masks/pred_masks.csv \
        --clinical-csv data/clinical.csv --kfold 5 --amp --out work/cascadeC_segvol
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from liver_sppvr.data import MultiPhaseLiverDataset, collate_multiphase, load_manifest
from liver_sppvr.data.clinical import feat_dim, load_clinical
from liver_sppvr.data.dataset import load_pred_masks
from liver_sppvr.train.build import build_classifier, load_config, set_seed
from liver_sppvr.train.engine import evaluate
from liver_sppvr.train.metrics import cls_metrics, print_per_class, save_confusion, write_summary
from liver_sppvr.train.splits import labels_from_manifest, make_kfold_splits
from liver_sppvr.utils.device import resolve_device

DICE_BINS = ((0.0, 0.3), (0.3, 0.6), (0.6, 1.01))


def _native_dice(pred_path: str, gt_path: str) -> float:
    import nibabel as nib
    a = np.asarray(nib.load(pred_path).dataobj) > 0
    b = np.asarray(nib.load(gt_path).dataobj) > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s > 0 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cls-dir", required=True, help="train_cls.py work dir with fold{i}/best.pth (mask-source pred)")
    ap.add_argument("--pred-masks", required=True, help="Stage A out-of-fold pred_masks.csv")
    ap.add_argument("--clinical-csv", default=None)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg.get("device", "auto"))
    use_amp = args.amp and getattr(device, "type", str(device)) == "cuda"
    ccfg, pcfg = cfg["classifier"], cfg.get("preprocess", {})
    num_classes, class_names = ccfg["num_classes"], ccfg["class_names"]
    man = load_manifest(args.manifest or cfg["data"]["manifest"])
    labels = labels_from_manifest(man, class_names)
    splits = make_kfold_splits(labels, k=args.kfold, seed=cfg["project"]["seed"])
    pred_masks = load_pred_masks(args.pred_masks)
    with open(args.pred_masks) as f:
        dice_csv = {r["patient_id"]: r.get("dice") for r in csv.DictReader(f)}
    gt_by_pid = {pid: grp["mask_path"].iloc[0] for pid, grp in man.groupby("patient_id")}
    os.makedirs(args.out, exist_ok=True)

    results, per_patient = [], []
    for i, (tr_ids, va_ids) in enumerate(splits):
        ckpt_path = os.path.join(args.cls_dir, f"fold{i}", "best.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        targs = ckpt.get("args", {})
        if targs.get("mask_source") != "pred":
            print(f"WARNING fold {i}: classifier was trained with mask_source={targs.get('mask_source')}, "
                  f"not 'pred' -- the cascade feeds it predicted masks anyway (train/test mismatch).")
        clinical = load_clinical(args.clinical_csv or targs.get("clinical_csv"), tr_ids)
        model = build_classifier(ckpt["config"], device, backbone=targs.get("backbone"),
                                 finetune=targs.get("finetune"), mask_source="pred",
                                 clinical_dim=feat_dim(clinical))
        model.load_state_dict(ckpt["model"]); model.amp_in_forward = use_amp
        spec = model.backbone.preprocess_spec
        ds = MultiPhaseLiverDataset(man, class_names=class_names, phases=cfg["multiphase"]["phases"],
                                    spatial_size=spec["spatial_size"], hu_window=spec["hu_window"],
                                    normalize=spec["normalize"], spacing=spec.get("spacing"),
                                    patient_ids=va_ids, roi=targs.get("roi") or "pred",
                                    roi_margin=pcfg.get("roi_margin", 0.5),
                                    mask_override=pred_masks, clinical=clinical)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiphase,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        print(f"\n===== cascade fold {i+1}/{args.kfold}: {len(ds)} OOF patients, "
              f"backbone={targs.get('backbone')} ckpt epoch {ckpt.get('epoch')} =====")
        ev = evaluate(model, loader, device, num_classes, use_mask=True, return_preds=True)
        for pid, yt, yp, pr in zip(ev["patient_id"], ev["y_true"], ev["y_pred"], ev["y_prob"]):
            d = dice_csv.get(pid)
            d = float(d) if d not in (None, "") else _native_dice(pred_masks[pid], gt_by_pid[pid])
            per_patient.append({"patient_id": pid, "fold": i, "true": class_names[yt], "pred": class_names[yp],
                                "dice": d, **{f"p_{c}": p for c, p in zip(class_names, pr)}})
        print(f"  acc={ev['accuracy']:.4f} bal_acc={ev['balanced_accuracy']:.4f} "
              f"macroF1={ev['macro_f1']:.4f} AUC={ev['auc']:.4f} kappa={ev['kappa']:.4f}")
        print_per_class(ev, class_names)
        fd = os.path.join(args.out, f"fold{i}"); os.makedirs(fd, exist_ok=True)
        with open(os.path.join(fd, "metrics.json"), "w") as f:
            json.dump({k: v for k, v in ev.items() if k not in ("patient_id", "y_true", "y_pred", "y_prob")},
                      f, indent=2)
        save_confusion(ev["confusion"], class_names, os.path.join(fd, "confusion"), title=f"cascade fold {i}")
        results.append({"fold": i, "epoch": ckpt.get("epoch"),
                        "dice": float(np.mean([r["dice"] for r in per_patient if r["fold"] == i])), **ev})
        del model, ds, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(args.out, "per_patient.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_patient[0].keys())); w.writeheader(); w.writerows(per_patient)
    name = f"cascade-{results and torch.load(os.path.join(args.cls_dir, 'fold0', 'best.pth'), map_location='cpu')['args'].get('backbone')}"
    write_summary(results, args.out, name, args.kfold, class_names,
                  keys=("auc", "accuracy", "balanced_accuracy", "macro_f1", "kappa", "dice"))

    # classifier sensitivity to segmentation quality: metrics per native-Dice bin (all OOF patients)
    idx = {c: k for k, c in enumerate(class_names)}
    print("\n===== classification vs. segmentation quality (out-of-fold) =====")
    sens = {}
    for lo, hi in DICE_BINS:
        rows = [r for r in per_patient if lo <= r["dice"] < hi]
        if not rows:
            continue
        yt = [idx[r["true"]] for r in rows]; yp = [idx[r["pred"]] for r in rows]
        pr = [[r[f"p_{c}"] for c in class_names] for r in rows]
        m = cls_metrics(yt, yp, pr, num_classes)
        sens[f"dice_{lo:.1f}-{min(hi, 1.0):.1f}"] = {"n": len(rows), "accuracy": m["accuracy"],
                                                     "balanced_accuracy": m["balanced_accuracy"], "auc": m["auc"]}
        print(f"  dice in [{lo:.1f}, {min(hi, 1.0):.1f}): n={len(rows):3d}  acc={m['accuracy']:.3f}  "
              f"bal_acc={m['balanced_accuracy']:.3f}  AUC={m['auc']:.3f}")
    with open(os.path.join(args.out, "sensitivity_to_dice.json"), "w") as f:
        json.dump(sens, f, indent=2)
    return 0


if __name__ == "__main__":
    import torch.multiprocessing as _mp
    try:
        _mp.set_start_method("spawn")
    except RuntimeError:
        pass
    sys.exit(main())
