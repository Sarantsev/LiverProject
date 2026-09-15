# Methods to test on MCT-LTDiag

Multi-phase CECT liver tumour subtype classification (HCC / ICC / metastasis / hemangioma).
Protocol: 5-fold cross-validation, patient-level metrics, VOI 10–150 mm, folds fixed by patient (no VOI leakage).

**Benchmark dataset:** Wu X. et al., *A Multi-phase CT Dataset for Automated Differential Diagnosis of Liver Tumors*, **Scientific Data** 13:31 (2026) — <https://doi.org/10.1038/s41597-025-06343-4>
**Split / code:** <https://github.com/Hoyant-Su/Multi-phase_LT_Benchmark>

---

## Tier A — already benchmarked on MCT-LTDiag

Numbers already exist in the dataset paper (Table 4); code is in the benchmark repo. Re-run on your own fixed split to confirm reproducibility.

| Method | Paper (venue, year) | Origin modality | Classes / task | Code | Why include |
|---|---|---|---|---|---|
| CNN-2D | [Yasaka et al., *Radiology*, 2018](https://doi.org/10.1148/radiol.2017170706) | CT (CECT) | Liver-mass differentiation | in `Multi-phase_LT_Benchmark` | Classic CT-CNN lower bound |
| CNN-3D / LiAIDS | [Ying et al., *Nature Communications*, 2024](https://doi.org/10.1038/s41467-024-45325-9) | CT | Focal liver lesion diagnosis | in `Multi-phase_LT_Benchmark` | Large multicenter AI system; strong CT baseline |
| UniFormer | [Li et al., arXiv, 2022](https://arxiv.org/abs/2201.04676) | General (spatiotemporal) | Backbone for classification | official repo + benchmark | Best pure-image DL on MCT-LTDiag (Acc 69.25) — direct rival |
| Random Forest + ReliefF | [Bae et al., *European Radiology*, 2021](https://doi.org/10.1007/s00330-021-07877-y) | CT radiomics | Focal lesion classification | in `Multi-phase_LT_Benchmark` | Top macro-AUC baseline (89.83); must beat |
| SVM radiomics | [Zhao et al., *Frontiers in Oncology*, 2022](https://doi.org/10.3389/fonc.2022.650797) | CT radiomics | Focal lesion differentiation | in `Multi-phase_LT_Benchmark` | Best overall Acc / Kappa baseline (70.63 / 0.561) |

---

## Tier B — recent Q1 methods with open code (MUST retrain)

The rows that most strengthen the paper. Retrain each on the **same** 5-fold split as Tier A.

| Method | Paper (venue, year) | Origin modality | Classes / task | Code | Status on CT |
|---|---|---|---|---|---|
| **SDR-Former** | [Lou et al., *Neural Networks*, 2025](https://doi.org/10.1016/j.neunet.2025.107228) · [arXiv](https://arxiv.org/abs/2402.17246) | CT (3-phase) + MR (8-phase) | Multi-phase liver-lesion classification | [github.com/LMMMEng/LLD-MMRI-Dataset](https://github.com/LMMMEng/LLD-MMRI-Dataset) | CT-native (already tested on 3-phase CT) |
| **LCA-Net** (cross-attention dual-branch) | [Wang R. et al., *Information Fusion*, 2025](https://doi.org/10.1016/j.inffus.2024.102713) | MRI (LLD-MMRI-7) | 7-class liver tumour classification | [github.com/Wangrui-berry/Cross-attention](https://github.com/Wangrui-berry/Cross-attention) | Needs adaptation (MRI → CT input) |
| **H-LSTM** (ResNet-BiLSTM) | [Huang et al., *J. Cancer Research and Clinical Oncology*, 2024](https://doi.org/10.1007/s00432-024-05977-y) | CT (CECT) | HCC / ICC / normal | [github.com/ljwa2323/ResNet_BiLSTM](https://github.com/ljwa2323/ResNet_BiLSTM) | CT-native; handles variable phases |
| **STIC** | [Shi et al., *Journal of Hematology & Oncology*, 2021](https://doi.org/10.1186/s13045-021-01167-2) | CT (CECT) + clinical | HCC / ICC / metastasis | [github.com/ruitian-olivia/STIC-model](https://github.com/ruitian-olivia/STIC-model) | CT-native; exact class match |

---

## Reference — LiTS segmentation SOTA (different task)

> **Note:** the table below is **liver-tumour SEGMENTATION on the LiTS test set**, not subtype classification on MCT-LTDiag. Keep it separate from Tier A/B — the task, dataset and metrics differ (Dice/ASSD vs AUC/Acc/κ). It is useful only for the segmentation / VOI-generation side of the pipeline (e.g. when positioning a SegVol-based segmenter against specialised networks).

Source: **Table 5** of [Zhang et al., *Decoupled Pyramid Correlation Network for Liver Tumor Segmentation from CT images*, **Medical Physics** 49:7207–7221, 2022](https://doi.org/10.1002/mp.15723) · [arXiv:2205.13199](https://arxiv.org/abs/2205.13199).
Evaluated on the held-out LiTS test set; metrics: DSC [%], RVD, ASSD [mm], HD [mm], and precision/recall at 50 % overlap. Higher DSC and precision/recall are better; RVD, ASSD, HD better near 0.

| Method | Paper | Tumor DSC [%] | Tumor RVD | Tumor ASSD [mm] | Tumor HD [mm] | Liver DSC [%] | Liver RVD | Liver ASSD [mm] | Liver HD [mm] | Prec @50% | Recall @50% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SSF-Net | [Liu et al., *Medical Physics*, 2021](https://doi.org/10.1002/mp.14585) | 59.2 | — | 1.585 | — | 93.7 | — | 3.678 | — | — | — |
| AH-Net | [Liu et al., *MICCAI*, 2018](https://arxiv.org/abs/1711.08580) | 63.4 | 0.365 | 1.185 | 6.482 | 96.3 | −0.004 | 1.099 | 2.398 | 0.468 | 0.301 |
| H-DenseUNet | [Li et al., *IEEE TMI*, 2018](https://doi.org/10.1109/TMI.2018.2845918) | 72.2 | −0.072 | 1.102 | 6.228 | 96.1 | −0.018 | 1.450 | 3.150 | 0.384 | 0.393 |
| LW-HCN | [Zhang et al., *IJCAI*, 2019](https://doi.org/10.24963/ijcai.2019/593) | 73.0 | — | — | — | 96.5 | — | — | — | — | — |
| VA-MaskRCNN | [Wang et al., *MICCAI*, 2019](https://doi.org/10.1007/978-3-030-32226-7_20) | 74.1 | −0.177 | 1.224 | 6.497 | 96.1 | −0.009 | 1.140 | 2.298 | 0.419 | 0.438 |
| nnU-Net | [Isensee et al., *Nature Methods*, 2021](https://doi.org/10.1038/s41592-020-01008-z) | 74.8 | −0.076 | 1.044 | 6.132 | 96.3 | 0.014 | 1.342 | 2.134 | 0.437 | 0.439 |
| **DPC-Net** | [Zhang et al., *Medical Physics*, 2022](https://doi.org/10.1002/mp.15723) | **76.4** | −0.063 | **0.838** | 5.339 | 96.0 | 0.012 | 1.636 | 4.692 | 0.434 | 0.424 |