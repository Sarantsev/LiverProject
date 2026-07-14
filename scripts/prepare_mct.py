"""Prepare MCT-LTDiag: .tar (DICOM+NIFTI) -> layout for build_manifest.py.

Each archive <patient_id>.tar contains:
    NIFTI/nc.nii.gz  art.nii.gz  pvp.nii.gz  delay.nii.gz
    mask_pvp.nii.gz         (tumor mask, on the pvp phase)
    liver_mask_pvp.nii.gz   (liver mask)
    DICOM/...               (raw, not needed)

The tumor type comes from meta_info_patient.csv (columns ID, type).

Result:
    <out>/<tumor_type>/<patient_id>/non_contrast.nii.gz
                                    arterial.nii.gz
                                    portal.nii.gz
                                    delayed.nii.gz
                                    mask.nii.gz          (= mask_pvp)
                                    liver_mask.nii.gz    (= liver_mask_pvp, optional)

Next:
    python scripts/build_manifest.py --root <out> --dataset MCT-LTDiag \
        --out data/manifest.csv

Example:
    python scripts/prepare_mct.py --src data/big --out data/MCT-LTDiag \
        --meta data/big/meta_info_patient.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile
from collections import Counter

# basename in the archive -> name in the target layout (exact basename match)
PHASE_MAP = {
    "nc.nii.gz": "non_contrast.nii.gz",
    "art.nii.gz": "arterial.nii.gz",
    "pvp.nii.gz": "portal.nii.gz",
    "delay.nii.gz": "delayed.nii.gz",
    "mask_pvp.nii.gz": "mask.nii.gz",
    "liver_mask_pvp.nii.gz": "liver_mask.nii.gz",
}
REQUIRED = {"non_contrast.nii.gz", "arterial.nii.gz", "portal.nii.gz",
            "delayed.nii.gz", "mask.nii.gz"}


def load_labels(meta_csv: str) -> dict:
    import pandas as pd
    df = pd.read_csv(meta_csv)
    if "ID" not in df.columns or "type" not in df.columns:
        raise SystemExit(f"{meta_csv} has no ID/type columns. Found: {list(df.columns)}")
    return {str(r.ID).strip(): str(r.type).strip() for r in df.itertuples(index=False)}


def extract_one(tar_path: str, out_root: str, labels: dict,
                keep_liver: bool = True, overwrite: bool = False) -> str:
    """Extract the needed files of one archive. Returns a status string."""
    pid = os.path.splitext(os.path.basename(tar_path))[0]
    tumor_type = labels.get(pid)
    if tumor_type is None:
        return f"SKIP {pid}: no tumor type in meta"

    dest = os.path.join(out_root, tumor_type, pid)
    if not overwrite and os.path.isdir(dest) and \
            REQUIRED.issubset(set(os.listdir(dest))):
        return f"OK   {pid} ({tumor_type}): already done"

    os.makedirs(dest, exist_ok=True)
    written = set()
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            target = PHASE_MAP.get(base)
            if target is None:
                continue
            if target == "liver_mask.nii.gz" and not keep_liver:
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            with open(os.path.join(dest, target), "wb") as f:
                f.write(src.read())
            written.add(target)

    missing = REQUIRED - written
    if missing:
        return f"WARN {pid} ({tumor_type}): missing files {sorted(missing)}"
    return f"OK   {pid} ({tumor_type})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/big", help="folder with *.tar")
    ap.add_argument("--out", default="data/MCT-LTDiag", help="target layout")
    ap.add_argument("--meta", default="data/big/meta_info_patient.csv")
    ap.add_argument("--no-liver", action="store_true", help="do not extract the liver mask")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process only N archives (test)")
    args = ap.parse_args()

    labels = load_labels(args.meta)
    tars = sorted(glob.glob(os.path.join(args.src, "*.tar")))
    if args.limit:
        tars = tars[:args.limit]
    if not tars:
        raise SystemExit(f"No .tar files in {args.src}")

    print(f"Archives: {len(tars)} | classes in meta: {Counter(labels.values())}")
    stats = Counter()
    failed = []
    for i, t in enumerate(tars, 1):
        try:
            msg = extract_one(t, args.out, labels,
                              keep_liver=not args.no_liver, overwrite=args.overwrite)
        except Exception as e:
            msg = f"FAIL {os.path.basename(t)}: {type(e).__name__}: {e}"
            failed.append(t)
        stats[msg.split()[0]] += 1
        print(f"[{i}/{len(tars)}] {msg}")
        sys.stdout.flush()

    print(f"\nTotal: {dict(stats)}")
    if failed:
        print(f"Corrupt archives ({len(failed)}) -- re-download:")
        for t in failed:
            print("  " + t)
    print(f"Done -> {args.out}")
    print("Next: python scripts/build_manifest.py --root "
          f"{args.out} --dataset MCT-LTDiag --out data/manifest.csv")


if __name__ == "__main__":
    main()
