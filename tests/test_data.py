"""CPU test of the data pipeline on synthetic NIfTI (no real datasets).

    python -m pytest tests/test_data.py -q
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from liver_sppvr.data import (MultiPhaseLiverDataset, build_manifest, collate_multiphase,
                              load_manifest, load_pred_masks)

PHASES = ["non_contrast", "arterial", "portal", "delayed"]
CLASSES = ["HCC", "ICC"]
SPATIAL = (8, 32, 32)
SHAPE = (16, 40, 40)


def _nifti(path, arr):
    import nibabel as nib
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), path)


def _make_fake_dataset(root):
    for cls in CLASSES:
        for pid in (f"{cls}_p001", f"{cls}_p002"):
            pdir = os.path.join(root, cls, pid); os.makedirs(pdir, exist_ok=True)
            for phase in PHASES:
                _nifti(os.path.join(pdir, f"{phase}.nii.gz"), np.random.rand(*SHAPE) * 400 - 175)
            m = np.zeros(SHAPE); m[4:10, 10:25, 10:25] = 1
            _nifti(os.path.join(pdir, "mask.nii.gz"), m)


def _fake(tmp):
    root = os.path.join(tmp, "MCT"); _make_fake_dataset(root)
    csv = os.path.join(tmp, "manifest.csv")
    build_manifest(csv, root=root, dataset="MCT", phases=PHASES)
    return load_manifest(csv)


def test_manifest_dataset_collate():
    with tempfile.TemporaryDirectory() as tmp:
        man = _fake(tmp)
        assert len(man) == 2 * 2 * len(PHASES)
        ds = MultiPhaseLiverDataset(man, class_names=CLASSES, phases=PHASES, spatial_size=SPATIAL)
        assert len(ds) == 4
        item = ds[0]
        assert item["phases"].shape == (len(PHASES), 1, *SPATIAL) and item["mask"].shape == (1, *SPATIAL)
        assert "clinical" not in item
        batch = collate_multiphase([ds[0], ds[1]])
        assert batch["phases"].shape == (2, len(PHASES), 1, *SPATIAL) and batch["label"].shape == (2,)


def test_missing_phase_padded():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "MCT"); _make_fake_dataset(root)
        os.remove(os.path.join(root, "HCC", "HCC_p001", "delayed.nii.gz"))
        csv = os.path.join(tmp, "manifest.csv"); build_manifest(csv, root=root, dataset="MCT", phases=PHASES)
        ds = MultiPhaseLiverDataset(load_manifest(csv), class_names=CLASSES, phases=PHASES, spatial_size=SPATIAL)
        it = next(ds[i] for i in range(len(ds)) if ds[i]["patient_id"].endswith("HCC_p001"))
        assert it["phase_present"][PHASES.index("delayed")].item() == 0.0


def test_pred_mask_override_roi_noise_and_clinical():
    with tempfile.TemporaryDirectory() as tmp:
        man = _fake(tmp)
        pids = sorted(man["patient_id"].unique())
        # "predicted" masks: a shifted box, one file per patient + the csv the cascade writes
        rows = []
        for pid in pids:
            p = os.path.join(tmp, f"{pid.replace(':', '_')}_pred.nii.gz")
            m = np.zeros(SHAPE); m[6:12, 12:28, 12:28] = 1; _nifti(p, m); rows.append((pid, p))
        csv = os.path.join(tmp, "pred_masks.csv")
        with open(csv, "w") as f:
            f.write("patient_id,pred_mask_path\n" + "".join(f"{a},{b}\n" for a, b in rows))
        pred = load_pred_masks(csv)
        clinical = {pid: np.arange(5, dtype=np.float32) for pid in pids}

        ds = MultiPhaseLiverDataset(man, class_names=CLASSES, phases=PHASES, spatial_size=SPATIAL,
                                    roi="pred", mask_override=pred, mask_noise=1.0, augment=True,
                                    clinical=clinical)
        it = ds[0]
        assert it["mask"].shape == (1, *SPATIAL) and it["clinical"].shape == (5,)
        assert set(it["mask"].unique().tolist()) <= {0.0, 1.0}
        assert collate_multiphase([ds[0], ds[1]])["clinical"].shape == (2, 5)

        with pytest.raises(ValueError):                        # roi=pred without predicted masks = leak
            MultiPhaseLiverDataset(man, class_names=CLASSES, phases=PHASES, spatial_size=SPATIAL, roi="pred")
        with pytest.raises(KeyError):                          # strict: no silent GT fallback
            MultiPhaseLiverDataset(man, class_names=CLASSES, phases=PHASES, spatial_size=SPATIAL,
                                   mask_override={pids[0]: rows[0][1]})


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn(); print(f"OK  {k}")
    print("\nData tests passed.")
