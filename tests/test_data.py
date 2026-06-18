"""CPU-тест пайплайна данных на синтетических NIfTI (без реальных датасетов).

    ../segvol_env/bin/python tests/test_data.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from liver_sppvr.data import (
    build_manifest, load_manifest, MultiPhaseLiverDataset, collate_multiphase,
)

PHASES = ["non_contrast", "arterial", "portal", "delayed"]
CLASSES = ["HCC", "ICC"]
SPATIAL = (8, 32, 32)   # маленький размер для скорости теста


def _write_nifti(path, shape=(16, 40, 40)):
    import nibabel as nib
    arr = (np.random.rand(*shape) * 400 - 175).astype(np.float32)  # ~HU
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), path)


def _make_fake_dataset(root):
    for cls in CLASSES:
        for pid in [f"{cls}_p001", f"{cls}_p002"]:   # уникальные id пациентов
            pdir = os.path.join(root, cls, pid)
            os.makedirs(pdir, exist_ok=True)
            for phase in PHASES:
                _write_nifti(os.path.join(pdir, f"{phase}.nii.gz"))
            # маска
            import nibabel as nib
            m = np.zeros((16, 40, 40), np.float32); m[4:10, 10:25, 10:25] = 1
            nib.save(nib.Nifti1Image(m, np.eye(4)), os.path.join(pdir, "mask.nii.gz"))


def test_manifest_and_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "MCT")
        _make_fake_dataset(root)
        csv = os.path.join(tmp, "manifest.csv")
        df = build_manifest(csv, root=root, dataset="MCT", phases=PHASES)
        assert len(df) == 2 * 2 * len(PHASES)          # классы * пациенты * фазы

        man = load_manifest(csv)
        ds = MultiPhaseLiverDataset(man, class_names=CLASSES, phases=PHASES,
                                    spatial_size=SPATIAL)
        assert len(ds) == 4                            # 4 пациента

        item = ds[0]
        assert item["phases"].shape == (len(PHASES), 1, *SPATIAL)
        assert item["mask"].shape == (1, *SPATIAL)
        assert item["label"].item() in (0, 1)

        batch = collate_multiphase([ds[0], ds[1]])
        assert batch["phases"].shape == (2, len(PHASES), 1, *SPATIAL)
        assert batch["mask"].shape == (2, 1, *SPATIAL)
        assert batch["label"].shape == (2,)


def test_missing_phase_padded():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "MCT")
        _make_fake_dataset(root)
        # удаляем одну фазу у одного пациента
        os.remove(os.path.join(root, "HCC", "HCC_p001", "delayed.nii.gz"))
        csv = os.path.join(tmp, "manifest.csv")
        build_manifest(csv, root=root, dataset="MCT", phases=PHASES)
        ds = MultiPhaseLiverDataset(load_manifest(csv), class_names=CLASSES,
                                    phases=PHASES, spatial_size=SPATIAL)
        # находим пациента с отсутствующей фазой
        for i in range(len(ds)):
            it = ds[i]
            if it["patient_id"].endswith("HCC_p001") and it["label"].item() == 0:
                assert it["phase_present"][PHASES.index("delayed")].item() == 0.0
                assert it["phases"].shape == (len(PHASES), 1, *SPATIAL)
                break


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print("\nТесты данных прошли.")
