"""Build a per-patient CLINICAL feature CSV for fusion (imaging + clinical).

Reads meta_info_patient.csv and emits a CSV in the exact format the training
fusion path expects (same as scripts/extract_radiomics.py):

    patient_id, clin_sex, clin_age, clin_cirrhosis, clin_hepatitis, clin_chemo, tumor_type

Only OBJECTIVE clinical variables are used. The radiological descriptors in
meta_info_tumor.csv (Nonrim APHE / Washout / Capsule / Rim APHE / ...) are
DELIBERATELY EXCLUDED -- they are the radiologist's diagnostic read (e.g. LI-RADS
HCC = APHE + washout + capsule) and would leak the label (circular). Tumor
size/volume are imaging-derived and already covered by the radiomics branch.

patient_id is taken from the manifest so it matches exactly (the manifest uses
"<dataset>:<id>", while the meta file uses the bare "<id>"). Feature values are
left RAW (numeric): the loader z-scores them on TRAIN patients only (no leakage).

Usage:
    PYTHONPATH=. python scripts/build_clinical.py \
        --meta data/big/meta_info_patient.csv --manifest data/manifest.csv \
        --out data/clinical.csv
    # optional -- also emit a combined radiomics+clinical CSV:
    #   --merge-radiomics data/radiomics.csv --out-merged data/radiomics_clinical.csv

Then train, e.g.:
    ... --radiomics-csv data/clinical.csv --radiomics-fusion early   # imaging + clinical
"""
from __future__ import annotations

import argparse
import sys

FEATURES = ["clin_sex", "clin_age", "clin_cirrhosis", "clin_hepatitis", "clin_chemo"]


def _yn(v):
    s = str(v).strip().lower()
    if s in ("y", "yes", "1", "true", "t"):
        return 1.0
    if s in ("n", "no", "0", "false", "f"):
        return 0.0
    return float("nan")                     # unknown/blank -> NaN (loader maps to 0)


def _sex(v):
    s = str(v).strip().lower()
    if s in ("m", "male"):
        return 1.0
    if s in ("f", "female"):
        return 0.0
    return float("nan")


def _age(v):
    try:
        return float(str(v).strip())
    except ValueError:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True, help="meta_info_patient.csv")
    ap.add_argument("--manifest", required=True, help="manifest CSV (for the exact patient_id format)")
    ap.add_argument("--out", default="data/clinical.csv")
    ap.add_argument("--merge-radiomics", default=None,
                    help="optional radiomics CSV to concatenate features with (per patient_id)")
    ap.add_argument("--out-merged", default="data/radiomics_clinical.csv")
    args = ap.parse_args()

    import pandas as pd

    man = pd.read_csv(args.manifest)
    # map bare id (after the last ':') -> full manifest patient_id
    full_ids = sorted(set(man["patient_id"].astype(str)))
    bare_to_full = {}
    for fid in full_ids:
        bare = fid.split(":")[-1]
        bare_to_full.setdefault(bare, fid)          # first wins (ids are unique anyway)

    meta = pd.read_csv(args.meta)
    # tolerant column lookup (exact names from meta_info_patient.csv)
    col = {c.lower(): c for c in meta.columns}
    c_id = col.get("id", "ID")
    c_type = col.get("type", "type")
    c_sex = col["patient_sex"]
    c_age = col["patient_age"]
    c_cirr = next(c for c in meta.columns if c.lower().startswith("cirrhosis"))
    c_hep = next(c for c in meta.columns if "hepatitis" in c.lower())
    c_chemo = next(c for c in meta.columns if "chemotherapy" in c.lower())

    rows, matched, unmatched = [], 0, 0
    for r in meta.itertuples(index=False):
        d = r._asdict()
        bare = str(d[c_id]).strip()
        fid = bare_to_full.get(bare)
        if fid is None:
            unmatched += 1
            continue
        matched += 1
        rows.append({
            "patient_id": fid,
            "clin_sex": _sex(d[c_sex]),
            "clin_age": _age(d[c_age]),
            "clin_cirrhosis": _yn(d[c_cirr]),
            "clin_hepatitis": _yn(d[c_hep]),
            "clin_chemo": _yn(d[c_chemo]),
            "tumor_type": str(d.get(c_type, "")).strip(),
        })

    out = pd.DataFrame(rows, columns=["patient_id", *FEATURES, "tumor_type"])
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out} "
          f"(matched {matched}, unmatched-in-meta {unmatched}, manifest patients {len(full_ids)})")
    # quick sanity: how many NaNs per feature (loader will zero them)
    print("  NaNs per feature:", {f: int(out[f].isna().sum()) for f in FEATURES})

    if args.merge_radiomics:
        rad = pd.read_csv(args.merge_radiomics)
        rad_feats = [c for c in rad.columns if c not in ("patient_id", "tumor_type")]
        merged = rad[["patient_id", *rad_feats]].merge(
            out[["patient_id", *FEATURES, "tumor_type"]], on="patient_id", how="inner")
        merged.to_csv(args.out_merged, index=False)
        print(f"wrote {len(merged)} rows to {args.out_merged} "
              f"({len(rad_feats)} radiomics + {len(FEATURES)} clinical features)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
