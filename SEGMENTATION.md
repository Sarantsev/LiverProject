# Segmentation experiments

Run from the repository root on the GPU machine, with Python 3.10/3.11.
All four experiments export the complete manifest cohort and reuse the
classifier's original class-stratified patient folds and project seed.
An absent phase or mismatched NIfTI shape/affine stops export: use the
registered images, not the raw unregistered acquisitions. No GT-based crop
is used. The mask is the training target and evaluation reference only.

| Experiment | Model | Channels | Dataset ID | Work directory |
|---|---|---|---|---|
| portal | nnU-Net v2 3d_fullres | portal | 501 | work/seg_portal |
| triphasic | nnU-Net v2 3d_fullres | NC, arterial, portal | 502 | work/seg_triphasic |
| fourphase | nnU-Net v2 3d_fullres | NC, arterial, portal, delayed | 503 | work/seg_fourphase |
| dual | ConvNeXt-Small + CSWin, 2D | NC, arterial, portal | 504 | work/seg_dual |

ID 501 is the legacy portal dataset. Existing compatible portal exports can
be reused. Choose empty experiment work directories to avoid overwriting runs.
Do not change a dataset's channels, seed or cohort after planning/training.
After any data content change, create a new dataset ID/directory and replan;
nnU-Net preprocessing is not automatically invalidated by this exporter.

## Environment

Keep segmentation separate from the torch-1.13 SegVol environment.

```bash
bash scripts/run_nnunet.sh setup
nnunet_env/bin/python -m pip install -r requirements-seg.txt
export SEGVOL_PY="$PWD/nnunet_env/bin/python"
export GPUS="0 1"  # use GPUS="0" for a single GPU
```

The existing setup installs CUDA-12.1 wheels; adapt that installation to the
GPU driver's supported PyTorch/CUDA combination if necessary.
The runner disables torch.compile by default to avoid Inductor autotuning
cache failures. Set nnUNet_compile=true explicitly to opt into compilation.
The ConvNeXt branch downloads timm ImageNet weights on its first training run.
Use --no-pretrained only for smoke checks or a separately labelled ablation.

## Three nnU-Net experiments

```bash
for experiment in portal triphasic fourphase; do
  bash scripts/run_segmentation.sh "$experiment" export
  bash scripts/run_segmentation.sh "$experiment" plan
  bash scripts/run_segmentation.sh "$experiment" train-all
  bash scripts/run_segmentation.sh "$experiment" predict-oof
  bash scripts/run_segmentation.sh "$experiment" collect
done
```

train-all waits for every GPU worker and returns failure if any worker fails.
Its logs are work/seg_<experiment>/train_gpu*.log. The default trainer uses
250 epochs for all three experiments. To use the standard nnU-Net schedule,
set TRAINER=nnUNetTrainer consistently for training and prediction.
For one fold: bash scripts/run_segmentation.sh triphasic train 0 0.
The final argument is the physical CUDA device index.

## Dual encoder

```bash
bash scripts/run_segmentation.sh dual export
bash scripts/run_segmentation.sh dual cache
CUDA_VISIBLE_DEVICES=0 bash scripts/run_segmentation.sh dual train-all
CUDA_VISIBLE_DEVICES=0 bash scripts/run_segmentation.sh dual predict-oof
bash scripts/run_segmentation.sh dual collect
```

The five folds train sequentially on one GPU. To distribute them manually,
launch train --fold 0, etc., with different CUDA_VISIBLE_DEVICES. Generate
the shared cache once before launching parallel folds. Each process must own
a distinct fold. Checkpoints are not overwritten and resume is not implemented.
Predict a selected fold with predict-oof --fold 0; collect requires all folds.

Examples of explicit overrides:

```bash
bash scripts/run_segmentation.sh dual train --fold 0 --batch-size 3 --workers 4 --epochs 100
SEG_PY=/path/to/python bash scripts/run_segmentation.sh dual predict-oof --batch-size 8
```

The cache contains full resized axial slices in memory-mapped .npy files,
including negative slices. Budget approximately 0.85 MB per slice, multiplied
by the total slice count. A source path/mtime/size signature invalidates the
dual cache when input files change. It is not a substitute for a dataset checksum.

### Implementation versus the preprint

This is a paper-inspired implementation, not an exact reproduction or a
downloaded author checkpoint. The CNN is timm ConvNeXt-Small. The scratch
CSWin branch has widths 32/64/128/256, depths 1/2/21/1, heads 2/4/8/8,
stripe widths 1/2/8/global and window-local depthwise positional encoding.
Transformer/CNN maps are aligned by 1x1 convolutions and added at four scales.
The U-Net-style decoder uses bilinear upsampling, GroupNorm and GELU.
Attention uses PyTorch SDPA. Inputs are 256x256 axial slices, three phase
channels, HU [-15,135]. Loss is weighted BCE (positive weight 500) plus
0.5 Dice loss; AdamW uses lr/weight decay 1e-4. No AMP is enabled.

Deliberate protocol choices: five shared folds rather than the paper's three;
checkpoint selection, LR scheduling and early stopping use native-volume
patient-mean Dice (scheduler monitors 1-Dice), not slice scores or validation
loss. Augmentation uses synchronized horizontal/vertical flips. Training
does not implement the paper's alternate VGG19/Swin model. Parameter count
and speed are properties of this implementation, not copied paper claims.
The default model has 60,463,457 parameters in the tested timm build; it does
not match the paper's claimed approximately 29M. Budget GPU memory accordingly.

## Outputs and classifier integration

Every experiment writes summary.json, summary.csv and
work/seg_<experiment>/oof_masks/pred_masks.csv. Native-space predictions retain
the reference NIfTI shape and affine. Dual inference thresholds probabilities
after resizing to the original in-plane size. Collection refuses missing masks
or mismatched geometry. OOF CSV columns match the existing classifier contract.

```bash
diff work/seg_portal/splits.json work/seg_triphasic/splits.json
diff work/seg_portal/splits.json work/seg_fourphase/splits.json
diff work/seg_portal/splits.json work/seg_dual/splits.json

python scripts/compare_segmentation.py work/seg_portal work/seg_triphasic \
  work/seg_fourphase work/seg_dual --out work/seg_comparison.csv

CLASSIFIER_PY=/path/to/segvol_env/bin/python
PYTHONPATH=. "$CLASSIFIER_PY" scripts/train_cls.py --config configs/default.yaml \
  --model ours --mask-source pred \
  --pred-masks work/seg_triphasic/oof_masks/pred_masks.csv \
  --kfold 5 --amp --work-dir work/cls_triphasic
```

For the classifier command use the SegVol-compatible interpreter, not the
segmentation interpreter used for export above. Set SEGVOL_PY accordingly.

The reported Dice is a patient mean at native resolution, empty/empty = 1.
Training-fold validation also selects the best epoch: results are development
CV estimates, not an untouched independent test. A final superiority claim
requires a held-out evaluation or nested selection. Compare per-patient
distributions, lesion failures and downstream classifier results as well as
the aggregate Dice. No published 0.9091 score is promised by this code.
