# Liver-SPPVR — liver-tumor CDSS built on SegVol

Clinical decision support pipeline for liver-tumor diagnosis from CT. One multi-task
model built on **SegVol** (3D foundation model, NeurIPS 2024 Spotlight):

- **Tumor segmentation** — SegVol's native prompt encoder + mask decoder (Dice + BCE).
- **Tumor-type classification** — a masked-pooling head over 5 classes
  (HCC / ICC / CRLM / BCLM / HH), Focal/weighted-CE.
- **Multi-phase input** — 4 CT phases (non-contrast / arterial / portal / delayed)
  fused by attention, since contrast dynamics are the radiological basis for the
  differential diagnosis.

A radiomics module (PyRadiomics) and a report generator are included as an isolated,
optional extension. See [segvol.md](segvol.md) for how the SegVol model works and
exactly how we apply it.

## What we train

We do **not** train SegVol from scratch. We load the pretrained `BAAI/SegVol` from
HuggingFace and fine-tune it multi-task on our data:

- The ViT **image encoder is frozen** (or adapted with **LoRA**) — the main
  anti-overfit lever, since the dataset is small (~400 patients).
- The **prompt encoder, mask decoder, fusion, and classification head are trained**.
- Input preprocessing matches SegVol: **ForegroundNorm** (z-score) + **CropForeground**.
- Segmentation can be prompted by **text** or by a **box** derived from the GT mask
  (SegVol's strong native mode).

## Structure

```
configs/
  default.yaml            # all paths/hyperparameters; device: auto
  radiomics_params.yaml   # PyRadiomics parameters
liver_sppvr/
  models/
    cls_head.py           # classification head (masked pooling + optional radiomics hybrid)
    multiphase.py         # 4-phase fusion (concat_stem | attention)
    lora.py               # minimal LoRA for nn.Linear (no external deps)
    segvol_multitask.py   # wrapper: SegVol encoder + segmentation + fusion + cls
  data/
    preprocess.py         # ForegroundNorm / HU windowing, CropForeground, zoom-in crop, resample
    augment.py            # lightweight 3D augmentations (flips + intensity jitter)
    manifest.py           # single CSV manifest (build/load/validate)
    dataset.py            # MultiPhaseLiverDataset + collate
  train/
    losses.py             # Dice+BCE (seg) and Focal/CE (cls), MultiTaskLoss
    engine.py             # train_one_epoch / evaluate (dice, accuracy, macro-F1); box prompt
    build.py              # build the real model (HF SegVol) + dry-run stub; LoRA/freeze
    multitask.py          # training CLI
  radiomics/extract.py    # PyRadiomics: features from the tumor mask (optional)
  inference/report.py     # end-to-end CDSS report (mask + type + volume + radiomics) (optional)
scripts/
  prepare_mct.py          # MCT-LTDiag .tar -> per-patient layout
  build_manifest.py       # build the CSV manifest from the layout
  download_tcia.py        # download TCIA collections (NBIA REST API)
  check_model.py          # load real SegVol + heads, forward on synthetic data
tests/                    # CPU tests (no GPU/data): smoke, data, radiomics
```

## Environment

- **Development** — on this machine (no GPU). Env: `../segvol_env`
  (torch 1.13.1, monai 0.9.0). Extra deps in `requirements.txt` / `requirements-train.txt`.
- **Training** — on a separate GPU box. The code is device-agnostic (`device: auto`)
  and transfers as-is. `pyradiomics` is only needed if you use the radiomics module.

## Quick check (CPU, no data)

```bash
../segvol_env/bin/python tests/test_smoke.py      # models and wrapper
../segvol_env/bin/python tests/test_data.py       # manifest + dataset
../segvol_env/bin/python tests/test_radiomics.py  # radiomics (skips without pyradiomics)

# run the training loop on synthetic data (stub instead of SegVol):
../segvol_env/bin/python -m liver_sppvr.train.multitask \
    --config configs/default.yaml --dry-run --device cpu
```

## Training on the GPU box

```bash
# 1) data -> per-patient layout -> manifest
python scripts/prepare_mct.py --src data/big --out data/MCT-LTDiag \
    --meta data/big/meta_info_patient.csv
python scripts/build_manifest.py --root data/MCT-LTDiag --dataset MCT-LTDiag \
    --out data/manifest.csv

# 2) train (downloads pretrained SegVol from HuggingFace BAAI/SegVol on first run)
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -m liver_sppvr.train.multitask --config configs/default.yaml \
    --num-workers 6 --amp --lora --seg-prompt box
```

Use both GPUs (single run via DataParallel): `CUDA_VISIBLE_DEVICES=0,1 ... --batch-size 4 --num-workers 8`.
Run two independent experiments instead: give each its own `--work-dir` so their
checkpoints do not collide.

## Key config / CLI knobs

Everything lives in `configs/default.yaml`; the most useful overrides are CLI flags:

| Flag | Config key | Purpose |
|---|---|---|
| `--lora` | `segvol.lora.enabled` | LoRA on the encoder instead of a full freeze |
| `--seg-prompt text\|box` | `train.seg_prompt` | segmentation prompt (box = from GT mask) |
| `--normalize hu\|foreground` | `preprocess.normalize` | input normalization (foreground = SegVol) |
| `--zoom P`, `--zoom-eval` | `train.zoom`, `train.zoom_eval` | zoom-in crop around the tumor |
| `--amp` | `train.amp` | mixed precision |
| `--batch-size`, `--num-workers`, `--work-dir`, `--epochs` | `train.*` | run overrides |

Anti-overfit is on by default: `freeze_encoder`, `augment`, `weight_decay: 1e-2`,
`early_stop_patience: 20`, `num_epochs: 80`. Checkpoints go to `work_dir/best.pth`
(best macro-F1). Metrics per epoch: `val_dice`, `accuracy`, `macro_f1`.

## Datasets (with tumor-type labels — needed for classification)

| Dataset | Role | Access |
|---|---|---|
| **MCT-LTDiag** (4 phases, ~517 patients, 5 classes + masks) | primary | Harvard Dataverse |
| **HCC-TACE-Seg** (105 HCC + masks) | HCC class / external validation | TCIA |
| **Colorectal-Liver-Metastases** (~197, CRLM + masks) | metastasis class | TCIA |

> LiTS / MSD-Liver / 3D-IRCADb / CHAOS — masks only, no tumor type;
> not used for the classifier.
