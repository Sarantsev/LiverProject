"""Register the non-portal CT phases to the portal (PVP) phase with ITKElastix.

The tumor mask lives on the portal phase, so we keep portal + mask fixed and warp the
other phases (non_contrast / arterial / delayed) onto it (rigid + B-spline). This makes
the multi-phase fusion spatially coherent -- the same voxel is the same anatomy across
phases (the MCT-LTDiag benchmark does this).

Writes registered NIfTI to <out_dir>/<patient>/<phase>.nii.gz and a new manifest pointing
to them (portal image + mask are copied unchanged). Originals are never modified.

    pip install itk-elastix
    python scripts/register_phases.py --manifest data/manifest.csv \
        --out-dir data/MCT-LTDiag-reg --out-manifest data/manifest_reg.csv

NOTE: registration is slow (~seconds-minutes/phase) and heavy. Verify a few cases
visually before trusting a full run. Not tested in this repo's dev environment.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liver_sppvr.data import load_manifest


def _safe(pid: str) -> str:
    return pid.replace(":", "_").replace("/", "_")


def _register(fixed_path, moving_path, out_path):
    import itk
    fixed = itk.imread(str(fixed_path), itk.F)
    moving = itk.imread(str(moving_path), itk.F)
    pobj = itk.ParameterObject.New()
    pobj.AddParameterMap(pobj.GetDefaultParameterMap("rigid"))    # coarse alignment
    pobj.AddParameterMap(pobj.GetDefaultParameterMap("bspline"))  # deformable refinement
    result, _ = itk.elastix_registration_method(
        fixed, moving, parameter_object=pobj, log_to_console=False)
    itk.imwrite(result, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="data/MCT-LTDiag-reg")
    ap.add_argument("--out-manifest", default="data/manifest_reg.csv")
    ap.add_argument("--fixed-phase", default="portal", help="reference phase (the mask lives on it)")
    ap.add_argument("--limit", type=int, default=None, help="process only N patients (test)")
    args = ap.parse_args()

    try:
        import itk  # noqa: F401
    except Exception:
        raise SystemExit("itk-elastix is not installed. Install it: pip install itk-elastix")
    import pandas as pd

    man = load_manifest(args.manifest)
    rows = []
    pids = list(man["patient_id"].unique())
    if args.limit:
        pids = pids[:args.limit]

    for n, pid in enumerate(pids, 1):
        grp = man[man["patient_id"] == pid]
        phase_to_path = dict(zip(grp["phase"], grp["image_path"]))
        fixed_path = phase_to_path.get(args.fixed_phase)
        if fixed_path is None:
            print(f"[{n}/{len(pids)}] SKIP {pid}: no '{args.fixed_phase}' phase")
            continue
        pdir = os.path.join(args.out_dir, _safe(pid))
        os.makedirs(pdir, exist_ok=True)
        base = grp.iloc[0]
        for _, r in grp.iterrows():
            phase = r["phase"]
            if phase == args.fixed_phase:
                out_img = fixed_path                       # keep the reference unchanged
            else:
                out_img = os.path.join(pdir, f"{phase}.nii.gz")
                try:
                    _register(fixed_path, r["image_path"], out_img)
                except Exception as e:
                    print(f"    {pid}/{phase}: registration failed ({e}); keeping original")
                    out_img = r["image_path"]
            rows.append(dict(patient_id=pid, dataset=base["dataset"], tumor_type=base["tumor_type"],
                             phase=phase, image_path=out_img, mask_path=base["mask_path"]))
        print(f"[{n}/{len(pids)}] {pid} registered -> {pdir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_manifest, index=False)
    print(f"\nDone. New manifest: {args.out_manifest} ({len(rows)} rows). "
          f"Train with: --config ... (set data.manifest to {args.out_manifest})")


if __name__ == "__main__":
    main()
