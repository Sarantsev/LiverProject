"""Extract per-patient radiomics from one CT phase + tumor mask -> cache CSV.

The CSV is consumed by training via `data.radiomics_csv` (fused into the classifier).
Run on the GPU box / any machine with pyradiomics installed and the data present.

Example:
    python scripts/extract_radiomics.py --manifest data/manifest.csv \
        --phase portal --out data/radiomics_portal.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liver_sppvr.data import load_manifest
from liver_sppvr.radiomics import RADIOMICS_AVAILABLE, batch_extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--phase", default="portal", help="CT phase to extract from (mask lives on portal)")
    ap.add_argument("--params", default="configs/radiomics_params.yaml")
    ap.add_argument("--out", default="data/radiomics_portal.csv")
    args = ap.parse_args()

    if not RADIOMICS_AVAILABLE:
        raise SystemExit("PyRadiomics is not installed. Install it: pip install pyradiomics")

    man = load_manifest(args.manifest)
    params = args.params if os.path.exists(args.params) else None
    print(f"Extracting radiomics from phase '{args.phase}' for "
          f"{man['patient_id'].nunique()} patients...")
    df = batch_extract(man, params=params, phase=args.phase)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out)   # patient_id is the index -> becomes the first column
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {args.out}")


if __name__ == "__main__":
    main()
