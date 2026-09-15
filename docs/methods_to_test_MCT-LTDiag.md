# Methods to test on MCT-LTDiag

Multi-phase CECT liver tumour subtype classification (HCC / ICC / metastasis / hemangioma).
Protocol: 5-fold cross-validation, patient-level metrics, VOI 10–150 mm, folds fixed by patient (no VOI leakage).
Benchmark dataset: Wu X. et al., *A Multi-phase CT Dataset for Automated Differential Diagnosis of Liver Tumors*, **Scientific Data** 13:31 (2026), DOI 10.1038/s41597-025-06343-4. Split / code: <https://github.com/Hoyant-Su/Multi-phase_LT_Benchmark>.

---

## Tier A — already benchmarked on MCT-LTDiag

Numbers already exist in the dataset paper (Table 4); code is in the benchmark repo. Re-run on your own fixed split to confirm reproducibility.

| Method | Paper (venue, year) | Origin modality | Classes / task | Code | Why include |
|---|---|---|---|---|---|
| CNN-2D | Yasaka et al., *Radiology*, 2018 | CT (CECT) | Liver-mass differentiation | in `Multi-phase_LT_Benchmark` | Classic CT-CNN lower bound |
| CNN-3D / LiAIDS | Ying et al., *Nature Communications*, 2024 | CT | Focal liver lesion diagnosis | in `Multi-phase_LT_Benchmark` | Large multicenter AI system; strong CT baseline |
| UniFormer | Li et al., arXiv, 2022 | General (spatiotemporal) | Backbone for classification | official repo + benchmark | Best pure-image DL on MCT-LTDiag (Acc 69.25) — direct rival |
| Random Forest + ReliefF | Bae et al., *European Radiology*, 2021 | CT radiomics | Focal lesion classification | in `Multi-phase_LT_Benchmark` | Top macro-AUC baseline (89.83); must beat |
| SVM radiomics | Zhao et al., *Frontiers in Oncology*, 2022 | CT radiomics | Focal lesion differentiation | in `Multi-phase_LT_Benchmark` | Best overall Acc / Kappa baseline (70.63 / 0.561) |

---

## Tier B — recent Q1 methods with open code (MUST retrain)

The rows that most strengthen the paper. Retrain each on the **same** 5-fold split as Tier A.

| Method | Paper (venue, year) | Origin modality | Classes / task | Code | Status on CT |
|---|---|---|---|---|---|
| **SDR-Former** | Lou et al., *Neural Networks*, 2025 | CT (3-phase) + MR (8-phase) | Multi-phase liver-lesion classification | github.com/LMMMEng (SDR-Former / LLD-MMRI) | CT-native (already tested on 3-phase CT) |
| **LCA-Net** (cross-attention dual-branch) | Wang R. et al., *Information Fusion*, 2025 | MRI (LLD-MMRI-7) | 7-class liver tumour classification | github.com/Wangrui-berry/Cross-attention | Needs adaptation (MRI → CT input) |
| **H-LSTM** (ResNet-BiLSTM) | Huang et al., *J. Cancer Research and Clinical Oncology*, 2024 | CT (CECT) | HCC / ICC / normal | github.com/ljwa2323/ResNet_BiLSTM | CT-native; handles variable phases |
| **STIC** | *Deep learning for differential diagnosis of malignant hepatic tumors based on multi-phase CECT and clinical data*, *J. Hematology & Oncology*, 2021 | CT (CECT) + clinical | HCC / ICC / metastasis | github.com/ruitian-olivia/STIC-model | CT-native; exact class match |

---

## Tier C — generic 3D backbones (reviewer-expected lower bounds)

Code available everywhere (e.g. MONAI). Group into one block in the final table.

| Method | Reference | Origin modality | Task | Code | Status on MCT-LTDiag |
|---|---|---|---|---|---|
| ResNet-3D / MedicalNet | Hara et al., 2018 / Chen et al., 2019 | General 3D | Volumetric classification | MONAI / official repos | Needs training on MCT-LTDiag |
| DenseNet-3D | Huang et al., CVPR, 2017 (3D variant) | General 3D | Volumetric classification | MONAI | Needs training on MCT-LTDiag |
| Swin Transformer 3D / ViT-3D | Liu et al., ICCV, 2021 / Dosovitskiy et al., 2021 | General 3D | Volumetric classification | MONAI / official repos | Needs training on MCT-LTDiag |

---

## Notes on a fair comparison

- Fix folds by patient (no VOI leakage), VOI 10–150 mm, patient-level aggregation. Reuse the split from `do_test_liver_K.sh` in `github.com/Hoyant-Su/Multi-phase_LT_Benchmark` so every method shares one protocol.
- "Needs adaptation" = MRI-native; feed multi-phase CT volumes instead of MR sequences and state the change in Methods. SDR-Former is the safest MRI-lineage port because it already reports 3-phase CT results.
- Verify each journal's current quartile against your target list — quartiles shift by year/category (Information Fusion and Neural Networks are clearly Q1; JCRCO is Q1/Q2 borderline in oncology).
- Confirm each competitor repo trains end-to-end before committing; challenge-participant code quality varies.
