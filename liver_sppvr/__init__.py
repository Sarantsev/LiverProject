"""Liver-tumour CDSS on multi-phase CT (MCT-LTDiag).

Two independent models, combined in a fully automatic cascade:
- segmentation: nnU-Net (external; scripts/export_nnunet.py + run_nnunet.sh)
- classification: `LiverTumorClassifier` = pretrained CT backbone (SegVol ViT | Merlin)
  + multi-phase fusion + head, optionally fused with clinical features
- cascade: predicted mask -> ROI + masked pooling -> tumour type (scripts/infer_cascade.py)
"""

__version__ = "0.2.0"
