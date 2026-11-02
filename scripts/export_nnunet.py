"""Stage A glue: MCT-LTDiag <-> nnU-Net v2, with OUR patient folds.

export (default): manifest -> nnUNet_raw/<Dataset>/{imagesTr/<case>_0000.nii.gz (symlink),
    labelsTr/<case>.nii.gz (binary uint8), dataset.json}, pid_map.csv (patient_id <-> case_id),
    nnUNet_preprocessed/<Dataset>/splits_final.json built from make_kfold_splits(seed) so every
    nnU-Net fold is exactly the classifier's fold (nnU-Net keeps an existing splits_final.json),
    and <Dataset>/oof/fold{i}/ with that fold's validation images for out-of-fold prediction.
    --dump-splits writes the folds in the shared JSON format for `diff` against train_cls.py.

--collect: after `run_nnunet.sh predict-oof`, gather <pred-dir>/fold{i}/<case>.nii.gz ->
    <work-dir>/oof_masks/pred_masks.csv (patient_id, pred_mask_path, fold, dice) + per-fold
    native-resolution Dice -> summary.csv/json. Every patient's mask comes from the fold model
    that never saw it, so these masks are safe to train/evaluate the classifier on.

Examples (GPU box):
    PYTHONPATH=. python scripts/export_nnunet.py --config configs/default.yaml \
        --out nnUNet_raw/Dataset501_MCTLiver --preprocessed-dir nnUNet_preprocessed/Dataset501_MCTLiver \
        --dump-splits work/splits_seg.json
    PYTHONPATH=. python scripts/export_nnunet.py --config configs/default.yaml --collect \
        --out nnUNet_raw/Dataset501_MCTLiver --pred-dir work/segA/oof --work-dir work/segA
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

from liver_sppvr.data import load_manifest
from liver_sppvr.train.build import load_config
from liver_sppvr.train.splits import dump_splits, labels_from_manifest, make_kfold_splits


def case_id(patient_id: str) -> str:
    """nnU-Net case identifier: no ':' or path characters."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", patient_id)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s > 0 else 1.0


def export(args, cfg) -> int:
    import nibabel as nib
    man = load_manifest(args.manifest or cfg["data"]["manifest"])
    class_names = cfg["classifier"]["class_names"]
    labels = labels_from_manifest(man, class_names)
    if args.kfold < 2 or args.kfold > min(list(labels.values()).count(y) for y in set(labels.values())):
        raise ValueError("kfold must be at least 2 and no larger than the smallest class")
    phases = getattr(args, "phases", None) or [args.phase]
    if len(set(phases)) != len(phases):
        raise ValueError("Duplicate input phases")
    splits = make_kfold_splits(labels, k=args.kfold, seed=cfg["project"]["seed"])
    if args.dump_splits:
        dump_splits(splits, args.dump_splits)
        print(f"splits -> {args.dump_splits}")

    out = os.path.abspath(args.out)
    img_dir, lab_dir = os.path.join(out, "imagesTr"), os.path.join(out, "labelsTr")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    rows, skipped, cases = [], [], []
    seen = set()
    for pid, grp in man.groupby("patient_id"):
        if pid not in labels:
            continue
        mask_path = os.path.abspath(grp["mask_path"].iloc[0])
        msk = nib.load(mask_path)
        paths = []
        for phase in phases:
            sel = grp[grp["phase"] == phase]
            if len(sel) != 1:
                raise ValueError(f"{pid}: expected exactly one {phase} image, found {len(sel)}")
            path = os.path.abspath(sel["image_path"].iloc[0])
            img = nib.load(path)
            if len(img.shape) != 3 or img.shape != msk.shape or not np.allclose(img.affine, msk.affine, atol=1e-3):
                raise ValueError(f"{pid}/{phase}: geometry differs from mask; use registered images")
            paths.append(path)
        cid = case_id(pid)
        if cid in seen:
            raise ValueError(f"Case identifier collision: {cid}")
        seen.add(cid)
        cases.append((cid, paths, mask_path, pid))
        rows.append((pid, cid, class_names[labels[pid]], mask_path))
    # Validate every case before writing images. Re-exporting a different cohort or
    # channel layout into an existing dataset would leave stale nnU-Net inputs.
    meta_path = os.path.join(out, "experiment.json")
    metadata = {"phases": phases, "patients": [r[0] for r in rows],
                "kfold": args.kfold, "seed": cfg["project"]["seed"]}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            if json.load(f) != metadata:
                raise ValueError("Dataset already contains a different cohort/channel layout; choose a new output")
    expected = {f"{cid}_{c:04d}.nii.gz" for cid, paths, _, _ in cases for c in range(len(paths))}
    if set(os.listdir(img_dir)) - expected:
        raise ValueError("Stale image channels in output; choose a new dataset directory")
    if set(os.listdir(lab_dir)) - {f"{cid}.nii.gz" for cid, _, _, _ in cases}:
        raise ValueError("Stale labels in output; choose a new dataset directory")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    for cid, paths, mask_path, pid in cases:
        msk = nib.load(mask_path)
        for c, path in enumerate(paths):
            dst_img = os.path.join(img_dir, f"{cid}_{c:04d}.nii.gz")
            if os.path.lexists(dst_img):
                os.remove(dst_img)
            if args.copy:
                img = nib.load(path)
                nib.save(nib.Nifti1Image(np.asarray(img.dataobj).astype(np.float32), img.affine), dst_img)
            else:
                os.symlink(path, dst_img)
        lab = (np.asarray(msk.dataobj) > 0).astype(np.uint8)          # exactly {0,1}
        nib.save(nib.Nifti1Image(lab, msk.affine), os.path.join(lab_dir, f"{cid}.nii.gz"))
    with open(os.path.join(out, "pid_map.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient_id", "case_id", "tumor_type", "gt_mask_path"]); w.writerows(rows)
    with open(os.path.join(out, "dataset.json"), "w") as f:
        json.dump({"name": os.path.basename(out), "description": "MCT-LTDiag liver tumour: " + ", ".join(phases),
                   "channel_names": {str(i): "CT" for i in range(len(phases))}, "labels": {"background": 0, "tumor": 1},
                   "numTraining": len(rows), "file_ending": ".nii.gz"}, f, indent=2)

    # our folds -> nnU-Net's splits_final.json (+ per-fold val dirs for OOF prediction)
    pid2cid = {pid: cid for pid, cid, _, _ in rows}
    nn_splits = [{"train": [pid2cid[p] for p in tr if p in pid2cid],
                  "val": [pid2cid[p] for p in va if p in pid2cid]} for tr, va in splits]
    pre = os.path.abspath(args.preprocessed_dir or out.replace("nnUNet_raw", "nnUNet_preprocessed"))
    os.makedirs(pre, exist_ok=True)
    split_path = os.path.join(pre, "splits_final.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            if json.load(f) != nn_splits:
                raise ValueError("Existing nnU-Net folds differ; use a new dataset directory")
    for i, s in enumerate(nn_splits):
        d = os.path.join(out, "oof", f"fold{i}")
        expected = {f"{cid}_{c:04d}.nii.gz" for cid in s["val"] for c in range(len(phases))}
        if os.path.isdir(d) and set(os.listdir(d)) - expected:
            raise ValueError(f"Stale OOF validation inputs in {d}; choose a new dataset directory")
    with open(os.path.join(pre, "splits_final.json"), "w") as f:
        json.dump(nn_splits, f, indent=1)
    for i, s in enumerate(nn_splits):
        d = os.path.join(out, "oof", f"fold{i}"); os.makedirs(d, exist_ok=True)
        for cid in s["val"]:
            for c in range(len(phases)):
                name = f"{cid}_{c:04d}.nii.gz"
                dst = os.path.join(d, name)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.join(img_dir, name), dst)
    print(f"exported {len(rows)} cases -> {out} | splits_final.json -> {pre} "
          f"(folds: {[len(s['val']) for s in nn_splits]} val cases) | skipped {len(skipped)}")
    for pid, why in skipped:
        print(f"  skipped {pid}: {why}")
    return 0


def collect(args, cfg) -> int:
    import nibabel as nib
    out = os.path.abspath(args.out)
    with open(os.path.join(out, "pid_map.csv")) as f:
        pid_rows = list(csv.DictReader(f))
    cid2row = {r["case_id"]: r for r in pid_rows}
    pre = os.path.abspath(args.preprocessed_dir or out.replace("nnUNet_raw", "nnUNet_preprocessed"))
    with open(os.path.join(pre, "splits_final.json")) as f:
        nn_splits = json.load(f)
    mask_dir = os.path.join(os.path.abspath(args.work_dir), "oof_masks")
    os.makedirs(mask_dir, exist_ok=True)
    rows, per_fold = [], []
    for i, s in enumerate(nn_splits):
        dices = []
        for cid in s["val"]:
            pred = os.path.join(os.path.abspath(args.pred_dir), f"fold{i}", f"{cid}.nii.gz")
            if not os.path.exists(pred):
                raise FileNotFoundError(f"Missing OOF prediction: {pred}")
            gt_img = nib.load(cid2row[cid]["gt_mask_path"])
            pr_img = nib.load(pred)
            gt, pr = np.asarray(gt_img.dataobj), np.asarray(pr_img.dataobj)
            if gt.shape != pr.shape or not np.allclose(gt_img.affine, pr_img.affine, atol=1e-3):
                raise RuntimeError(f"{cid}: prediction shape {pr.shape} != GT {gt.shape}")
            d = _dice(gt, pr); dices.append(d)
            rows.append({"patient_id": cid2row[cid]["patient_id"], "pred_mask_path": pred,
                         "fold": i, "dice": f"{d:.4f}", "empty_pred": int((pr > 0).sum() == 0)})
        per_fold.append(dices)
        print(f"fold {i}: {len(dices)} cases, dice {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    with open(os.path.join(mask_dir, "pred_masks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "pred_mask_path", "fold", "dice", "empty_pred"])
        w.writeheader(); w.writerows(rows)
    means = [float(np.mean(d)) for d in per_fold if d]
    all_d = [float(r["dice"]) for r in rows]
    summary = {"model": getattr(args, "model_name", "nnunet-3d_fullres"), "kfold": len(nn_splits),
               "per_fold": [{"fold": i, "dice": m, "n": len(per_fold[i])} for i, m in enumerate(means)],
               "mean_std": {"dice": {"mean": float(np.mean(means)), "std": float(np.std(means))}},
               "per_patient": {"mean": float(np.mean(all_d)), "std": float(np.std(all_d)), "n": len(all_d),
                               "empty_predictions": int(sum(int(r["empty_pred"]) for r in rows))}}
    with open(os.path.join(args.work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.work_dir, "summary.csv"), "w") as f:
        f.write("metric," + ",".join(f"fold{i}" for i in range(len(means))) + ",mean,std\n")
        f.write("dice," + ",".join(f"{m:.4f}" for m in means)
                + f",{summary['mean_std']['dice']['mean']:.4f},{summary['mean_std']['dice']['std']:.4f}\n")
    print(f"\nOOF masks: {len(rows)} -> {mask_dir}/pred_masks.csv | native Dice "
          f"{summary['mean_std']['dice']['mean']:.4f} ± {summary['mean_std']['dice']['std']:.4f} "
          f"(per-patient {summary['per_patient']['mean']:.4f} ± {summary['per_patient']['std']:.4f}, "
          f"{summary['per_patient']['empty_predictions']} empty)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True, help="nnUNet_raw/<DatasetXXX_Name>")
    ap.add_argument("--preprocessed-dir", default=None, help="nnUNet_preprocessed/<Dataset> (splits_final.json)")
    ap.add_argument("--phase", default="portal", help="phase to segment (the mask lives on it)")
    ap.add_argument("--phases", nargs="+", choices=["non_contrast", "arterial", "portal", "delayed"])
    ap.add_argument("--model-name", default="nnunet-3d_fullres")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--dump-splits", default=None)
    ap.add_argument("--collect", action="store_true", help="gather OOF predictions -> pred_masks.csv + Dice")
    ap.add_argument("--pred-dir", default="work/segA/oof", help="--collect: nnUNetv2_predict outputs, fold{i}/")
    ap.add_argument("--work-dir", default="work/segA")
    args = ap.parse_args()
    cfg = load_config(args.config)
    return collect(args, cfg) if args.collect else export(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
