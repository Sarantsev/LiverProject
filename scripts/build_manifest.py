"""CLI для сборки единого манифеста из типовой раскладки датасета.

Ожидаемая раскладка:
    <root>/<tumor_type>/<patient_id>/<phase>.nii.gz
    <root>/<tumor_type>/<patient_id>/mask.nii.gz

Пример:
    python scripts/build_manifest.py --root /data/MCT-LTDiag --dataset MCT-LTDiag \
        --out data/manifest.csv \
        --phases non_contrast arterial portal delayed

Для нестандартных раскладок (HCC-TACE-Seg, Colorectal-Liver-Metastases) напишите
свой адаптер record_fn и используйте liver_sppvr.data.build_manifest напрямую.
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
    print(f"Записано {len(df)} строк в {args.out}")
    print(df.groupby('tumor_type')['patient_id'].nunique().rename('patients'))


if __name__ == "__main__":
    main()
