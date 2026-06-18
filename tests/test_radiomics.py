"""Тест радиомического модуля. Если PyRadiomics не установлен — корректный skip.

    ../segvol_env/bin/python tests/test_radiomics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from liver_sppvr.radiomics import RADIOMICS_AVAILABLE, extract_from_arrays


def test_extract_from_arrays():
    if not RADIOMICS_AVAILABLE:
        print("SKIP test_extract_from_arrays (PyRadiomics не установлен)")
        return
    rng = np.random.default_rng(0)
    image = (rng.random((20, 40, 40)) * 200).astype(np.float32)
    mask = np.zeros((20, 40, 40), np.uint8)
    mask[5:15, 12:28, 12:28] = 1
    feats = extract_from_arrays(image, mask, spacing=(1.0, 1.0, 1.0))
    assert len(feats) > 0
    assert all(isinstance(v, float) for v in feats.values())
    print(f"OK  извлечено {len(feats)} радиомических признаков")


def test_import_guard():
    # модуль импортируется даже без PyRadiomics; флаг доступности — булев
    assert isinstance(RADIOMICS_AVAILABLE, bool)
    print(f"OK  RADIOMICS_AVAILABLE={RADIOMICS_AVAILABLE}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print("\nТест радиомики завершён.")
