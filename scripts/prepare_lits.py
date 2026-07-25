"""Prepare LiTS -> layout for build_manifest.py (segmentation-only, TUMOR).

LiTS training volumes come as pairs:
    volume-<N>.nii(.gz)        portal-venous abdominal CT
    segmentation-<N>.nii(.gz)  labels: 0 background, 1 liver, 2 tumor

LiTS has NO tumor-type label, so we can only evaluate SEGMENTATION here. We lay
each volume out as a single-phase (portal) "patient" with a binary TUMOR mask
(seg == 2). A dummy tumor_type is assigned only so the multiphase dataset/manifest
accepts the record -- the classifier output is ignored for LiTS.

Result:
    <out>/<dummy_type>/lits_<N>/portal.nii.gz   (symlink or copy of volume-<N>)
                               mask.nii.gz       (= segmentation == 2, binary)

Next:
    python scripts/build_manifest.py --root <out> --dataset LiTS --out data/lits_manifest.csv
    # then eval:  python scripts/eval_lits.py --manifest data/lits_manifest.csv ...

Example:
    PYTHONPATH=. python scripts/prepare_lits.py --src data/LiTS/raw --out data/LiTS
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter


def _find_pairs(src: str):
    """Yield (n, volume_path, seg_path) for each image with a matching label.

    Handles two on-disk conventions:
      * LiTS / Kaggle:  volume-<n>.nii(.gz)  +  segmentation-<n>.nii(.gz)   (same dir)
      * MSD Task03:     imagesTr/liver_<n>.nii.gz  +  labelsTr/liver_<n>.nii.gz
    (MSD imagesTs/ has no labels -> skipped. AppleDouble '._*' files ignored.)
    """
    vols, segs = {}, {}
    for p in glob.glob(os.path.join(src, "**", "*.nii*"), recursive=True):
        base = os.path.basename(p)
        if base.startswith("._"):                      # macOS AppleDouble junk in tars
            continue
        parent = os.path.basename(os.path.dirname(p))
        m = re.match(r"segmentation[-_](\d+)\.nii(\.gz)?$", base)
        if m:
            segs[int(m.group(1))] = p; continue
        m = re.match(r"volume[-_](\d+)\.nii(\.gz)?$", base)
        if m:
            vols[int(m.group(1))] = p; continue
        m = re.match(r"liver[-_](\d+)\.nii(\.gz)?$", base)   # MSD Task03 (image & label share name)
        if m:
            n = int(m.group(1))
            if parent == "labelsTr":
                segs[n] = p
            elif parent == "imagesTs":
                continue                               # test split: no public label
            else:                                      # imagesTr (or flat) -> image
                vols[n] = p
    for n in sorted(vols):
        if n in segs:
            yield n, vols[n], segs[n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir with volume-*.nii(.gz) + segmentation-*.nii(.gz)")
    ap.add_argument("--out", required=True, help="output root (<out>/<type>/lits_<n>/...)")
    ap.add_argument("--dummy-type", default="HCC",
                    help="placeholder tumor_type (must be one of the trained class_names). Ignored for seg.")
    ap.add_argument("--require-tumor", action="store_true", default=True,
                    help="skip volumes without any tumor voxel (tumor Dice is undefined there)")
    ap.add_argument("--keep-empty", dest="require_tumor", action="store_false",
                    help="also keep volumes with liver but no tumor")
    ap.add_argument("--symlink", action="store_true", default=True,
                    help="symlink the volume instead of copying (default; saves disk)")
    ap.add_argument("--copy", dest="symlink", action="store_false", help="copy volumes instead of symlinking")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        raise SystemExit(f"need nibabel + numpy: {e}")

    pairs = list(_find_pairs(args.src))
    if not pairs:
        raise SystemExit(f"no volume-*/segmentation-* pairs found under {args.src}")

    stats = Counter()
    for n, vol_path, seg_path in pairs:
        pdir = os.path.join(args.out, args.dummy_type, f"lits_{n}")
        img_dst = os.path.join(pdir, "portal.nii.gz")
        mask_dst = os.path.join(pdir, "mask.nii.gz")
        if os.path.exists(mask_dst) and not args.overwrite:
            stats["exists"] += 1
            continue

        seg_img = nib.load(seg_path)
        seg = np.asanyarray(seg_img.dataobj)
        tumor = (seg == 2)
        n_tumor = int(tumor.sum())
        if args.require_tumor and n_tumor == 0:
            stats["no_tumor_skip"] += 1
            continue

        os.makedirs(pdir, exist_ok=True)
        # tumor mask: keep the volume's geometry (affine/header) from the seg file
        nib.save(nib.Nifti1Image(tumor.astype(np.uint8), seg_img.affine, seg_img.header), mask_dst)
        # image: symlink (or copy) the raw volume as the 'portal' phase
        if os.path.lexists(img_dst):
            os.remove(img_dst)
        if args.symlink:
            os.symlink(os.path.abspath(vol_path), img_dst)
        else:
            import shutil
            shutil.copy2(vol_path, img_dst)
        stats["ok"] += 1
        print(f"  lits_{n}: tumor_voxels={n_tumor}")

    print(f"\nDone. ok={stats['ok']} skipped_no_tumor={stats['no_tumor_skip']} "
          f"already={stats['exists']} (total pairs={len(pairs)})")
    print(f"Next: PYTHONPATH=. python scripts/build_manifest.py --root {args.out} "
          f"--dataset LiTS --out data/lits_manifest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
