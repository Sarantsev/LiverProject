"""CLI to build a single manifest from the standard dataset layout.

Expected layout:
    <root>/<tumor_type>/<patient_id>/<phase>.nii.gz
    <root>/<tumor_type>/<patient_id>/mask.nii.gz

Example:
    python scripts/build_manifest.py --root /data/MCT-LTDiag --dataset MCT-LTDiag \
        --out data/manifest.csv \
        --phases non_contrast arterial portal delayed

For non-standard layouts (HCC-TACE-Seg, Colorectal-Liver-Metastases) write your own
record_fn adapter and use liver_sppvr.data.build_manifest directly.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liver_sppvr.data import build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--phases", nargs="+",
                    default=["non_contrast", "arterial", "portal", "delayed"])
    ap.add_argument("--mask-name", default="mask.nii.gz")
    args = ap.parse_args()

    df = build_manifest(args.out, root=args.root, dataset=args.dataset,
                        phases=args.phases, mask_name=args.mask_name)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df.groupby('tumor_type')['patient_id'].nunique().rename('patients'))


if __name__ == "__main__":
    main()
