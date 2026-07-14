"""Test of the radiomics module. If PyRadiomics is not installed -> clean skip.

    ../segvol_env/bin/python tests/test_radiomics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from liver_sppvr.radiomics import RADIOMICS_AVAILABLE, extract_from_arrays


def test_extract_from_arrays():
    if not RADIOMICS_AVAILABLE:
        print("SKIP test_extract_from_arrays (PyRadiomics not installed)")
        return
    rng = np.random.default_rng(0)
    image = (rng.random((20, 40, 40)) * 200).astype(np.float32)
    mask = np.zeros((20, 40, 40), np.uint8)
    mask[5:15, 12:28, 12:28] = 1
    feats = extract_from_arrays(image, mask, spacing=(1.0, 1.0, 1.0))
    assert len(feats) > 0
    assert all(isinstance(v, float) for v in feats.values())
    print(f"OK  extracted {len(feats)} radiomics features")


def test_import_guard():
    # the module imports even without PyRadiomics; the availability flag is a bool
    assert isinstance(RADIOMICS_AVAILABLE, bool)
    print(f"OK  RADIOMICS_AVAILABLE={RADIOMICS_AVAILABLE}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print("\nRadiomics test finished.")
