"""Build a per-patient CLINICAL feature CSV for fusion (imaging + clinical).

Reads meta_info_patient.csv and emits a CSV in the exact format the training
fusion path expects:

    patient_id, clin_sex, clin_age, clin_cirrhosis, clin_hepatitis, clin_chemo, tumor_type

Only OBJECTIVE clinical variables are used. The radiological descriptors in
meta_info_tumor.csv (Nonrim APHE / Washout / Capsule / Rim APHE / ...) are
DELIBERATELY EXCLUDED -- they are the radiologist's diagnostic read (e.g. LI-RADS
HCC = APHE + washout + capsule) and would leak the label (circular). Tumor
size/volume are imaging-derived (the classifier sees them in the image).

patient_id is taken from the manifest so it matches exactly (the manifest uses
"<dataset>:<id>", while the meta file uses the bare "<id>"). Feature values are
left RAW (numeric): the loader z-scores them on TRAIN patients only (no leakage).

Usage:
    PYTHONPATH=. python scripts/build_clinical.py \
        --meta data/big/meta_info_patient.csv --manifest data/manifest.csv \
        --out data/clinical.csv

Then train, e.g.:
    PYTHONPATH=. python scripts/train_cls.py ... --clinical-csv data/clinical.csv   # imaging + clinical
"""
from __future__ import annotations

import argparse
import re
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


# DICOM Age String (VR "AS") units -> years
_AGE_UNITS = {"y": 1.0, "m": 1.0 / 12.0, "w": 1.0 / 52.1775, "d": 1.0 / 365.25}


def _age(v):
    """Age in YEARS from either a plain number ("42") or a DICOM Age String ("062Y").

    meta_info_patient.csv mixes both formats: ~15% of rows are plain integers and the
    rest are zero-padded DICOM AS values with a unit suffix, which float() rejects --
    silently NaN-ing 85% of the ages.
    """
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "na", "-"):
        return float("nan")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([ymwd])?", s, flags=re.IGNORECASE)
    if not m:
        return float("nan")
    return float(m.group(1)) * _AGE_UNITS[(m.group(2) or "y").lower()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True, help="meta_info_patient.csv")
    ap.add_argument("--manifest", required=True, help="manifest CSV (for the exact patient_id format)")
    ap.add_argument("--out", default="data/clinical.csv")
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

    # Tolerant column lookup: meta_info_patient.csv uses names like "Cirrhosis status (Y/N)".
    def _find(*needles) -> str:
        for c in meta.columns:
            if any(n in c.lower() for n in needles):
                return c
        raise SystemExit(f"{args.meta}: no column matching {needles}. "
                         f"Columns present: {list(meta.columns)}")

    col = {c.lower(): c for c in meta.columns}
    c_id = col.get("id", "ID")
    c_type = col.get("type", "type")
    c_sex = _find("patient_sex", "sex")
    c_age = _find("patient_age", "age")
    c_cirr = _find("cirrhosis")
    c_hep = _find("hepatitis")
    c_chemo = _find("chemotherapy", "chemo")

    rows, matched, unmatched = [], 0, 0
    # to_dict("records") keeps the ORIGINAL column names. itertuples() would rename any
    # column that is not a valid Python identifier ("Cirrhosis status (Y/N)" -> "_3"),
    # so looking it up by its real name raised KeyError.
    for d in meta.to_dict("records"):
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
    age = out["clin_age"].dropna()
    if len(age):
        print(f"  clin_age: n={len(age)} min={age.min():.0f} median={age.median():.0f} "
              f"max={age.max():.0f}  (sanity-check these look like adult years)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
