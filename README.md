# liver-sppvr — liver tumour diagnosis on multi-phase CT (MCT-LTDiag)

Fully automatic pipeline: **CT → tumour mask (nnU-Net) → tumour type (pretrained CT encoder + clinical data)**,
evaluated with patient-level 5-fold cross-validation on MCT-LTDiag (516 patients, 4 contrast phases,
5 classes: HCC / ICC / CRLM / BCLM / HH).

Two independent models, combined in a cascade:

| Stage | What | Where |
|-------|------|-------|
| **A** segmentation | nnU-Net v2 `3d_fullres`, portal phase, trained on **our** folds → out-of-fold (OOF) masks | `scripts/export_nnunet.py`, `scripts/run_nnunet.sh` |
| **B** classification | `LiverTumorClassifier` = pretrained backbone (**SegVol ViT + DoRA** or **Merlin I3D-ResNet-152**) → 4-phase attention fusion → head (+ 5 clinical features) | `liver_sppvr/models/`, `scripts/train_cls.py` |
| **C** cascade | OOF mask → ROI + masked pooling → type; metrics + sensitivity to Dice | `scripts/infer_cascade.py` |

The 5 published baselines (H-LSTM, STIC, SDR-Former, LCA-Net, RA-CMFormer) are re-implemented in
`liver_sppvr/baselines/` and trained by the **same** `train_cls.py` on the **same** folds and metrics.

## Why this design
* The classifier's mask source is explicit: `--mask-source none | gt | pred`. `gt` (radiologist's mask)
  is a semi-automatic upper bound; **`pred` (nnU-Net OOF masks) is the headline, fully automatic
  setting**. Masks are never silently taken from GT, and the ROI crop is only ever derived from a
  predicted mask.
* Every patient's predicted mask comes from the nnU-Net fold model that never saw that patient
  (`splits_final.json` is generated from our seed), so training the classifier on those masks is
  leakage-free and matches deployment.
* Train-time mask perturbation (`--mask-noise`) makes the head robust to segmentation errors.
* Only 5 objective clinical variables are fused (sex, age, cirrhosis, hepatitis, chemotherapy);
  radiological descriptors of the tumour are excluded as label leakage. Clinical features are
  z-scored on the training fold only.

## Environments
* `segvol_env` (torch 1.13, monai 0.9, transformers 4.30) — everything in this repo. `requirements-train.txt`.
* `nnunet_env` (torch ≥ 2, `nnunetv2`) — **only** nnU-Net; created by `scripts/run_nnunet.sh setup`.
* Merlin weights are downloaded from HF `stanfordmimi/Merlin`; the image tower is vendored in
  `liver_sppvr/models/merlin.py` (no `merlin-vlm` install needed).

## Quick start (GPU box, `PYTHONPATH=.`)
```bash
# data
python scripts/prepare_mct.py ... && python scripts/build_manifest.py ... -> data/manifest.csv
python scripts/build_clinical.py --meta data/big/meta_info_patient.csv --manifest data/manifest.csv -> data/clinical.csv

# Stage A: segmentation (own venv)
bash scripts/run_nnunet.sh setup && bash scripts/run_nnunet.sh export && bash scripts/run_nnunet.sh plan
bash scripts/run_nnunet.sh train-all          # 5 folds over 2 GPUs
bash scripts/run_nnunet.sh predict-oof && bash scripts/run_nnunet.sh collect   # -> work/segA/oof_masks/pred_masks.csv

# Stage B: classification -- ours, both backbones x three mask sources (same folds)
for BB in segvol merlin; do for MS in gt none pred; do
  python scripts/train_cls.py --config configs/default.yaml --model ours --backbone $BB --mask-source $MS \
    --pred-masks work/segA/oof_masks/pred_masks.csv --clinical-csv data/clinical.csv \
    --kfold 5 --amp --work-dir work/clsB_${BB}_${MS}
done; done
# baselines
python scripts/train_cls.py --config configs/default.yaml --model sdrformer --kfold 5 --amp --resize 32,128,128 --work-dir work/sdrformer

# Stage C: fully automatic end-to-end
python scripts/infer_cascade.py --config configs/default.yaml --cls-dir work/clsB_segvol_pred \
  --pred-masks work/segA/oof_masks/pred_masks.csv --clinical-csv data/clinical.csv --kfold 5 --amp --out work/cascadeC_segvol
```
Every run writes `summary.json` / `summary.csv` (one row per metric: folds, mean, std) and confusion
matrices into its work dir — the rows of the paper's tables.

Prove the folds are identical across pipelines (no model is built):
```bash
python scripts/train_cls.py --config configs/default.yaml --model ours --kfold 5 --dump-splits work/splits_cls.json
bash scripts/run_nnunet.sh export     # writes work/splits_seg.json
diff work/splits_cls.json work/splits_seg.json   # empty = identical
```

## Layout
```
liver_sppvr/
  data/      manifest, dataset (mask_override / roi / mask_noise / clinical), preprocess, augment, clinical
  models/    backbones (interface + SegVol ViT), merlin (vendored I3D-ResNet-152), classifier, multiphase, cls_head, lora
  train/     build, engine, losses (focal-CE + SupCon), splits, metrics
  baselines/ the 5 published architectures under the same input contract
scripts/     prepare_mct, build_manifest, build_clinical | export_nnunet, run_nnunet.sh | train_cls | infer_cascade
configs/default.yaml   tests/   docs/ (references, methods, diary)
```
Tests (CPU, synthetic data): `../segvol_env/bin/python -m pytest tests -q`.
