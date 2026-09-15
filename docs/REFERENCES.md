# References — liver-sppvr

Все источники проекта: датасет, базовая модель, применённые методы, работы для
сравнения и метрики. Сгруппировано по назначению.

---

## 1. Датасет

| Что | Описание | Ссылка |
|---|---|---|
| **MCT-LTDiag** (данные) | Мультифазная КТ печени, 517 пациентов, 5 подтипов (HCC/ICC/CRLM/BCLM/HH), 4 фазы. Только изображения + маски + метка типа | [Harvard Dataverse, doi:10.7910/DVN/S3RW15](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/S3RW15) |
| **MCT-LTDiag** (статья) | Описание датасета + baseline (radiomics ML + DL), 5-fold CV. Их бенчмарк | [Nature Scientific Data 2025, s41597-025-06343-4](https://www.nature.com/articles/s41597-025-06343-4) |
| **Multi-phase LT Benchmark** (код) | Репозиторий бенчмарка, модель RU-Net, **клинические метаданные** (`meta/meta_info_patient.csv`: пол/возраст/цирроз/гепатит/химия; `meta_info_tumor.csv`: радиологические дескрипторы) | [GitHub: Hoyant-Su/Multi-phase_LT_Benchmark](https://github.com/Hoyant-Su/Multi-phase_LT_Benchmark) |

**5 классов:** HCC — гепатоцеллюлярная карцинома (первичный рак); ICC — внутрипечёночная
холангиокарцинома (первичный рак); CRLM — метастаз колоректального рака; BCLM — метастаз
рака груди; HH — гемангиома (доброкачественная).

---

## 2. Базовая модель

| Что | Описание | Ссылка |
|---|---|---|
| **SegVol** | 3D foundation-модель для сегментации КТ (ViT-энкодер, prompt/mask-декодеры, CLIP-текст). Предобучена на ~90K КТ. Основа нашей модели | [arXiv:2311.13385 (NeurIPS 2024 Spotlight)](https://arxiv.org/abs/2311.13385) · [BAAI/SegVol (HuggingFace)](https://huggingface.co/BAAI/SegVol) |

---

## 3. Методы, которые мы применили

| Метод | Где у нас | Референс |
|---|---|---|
| **LoRA** — low-rank адаптация энкодера | `models/lora.py` | [Hu et al. 2021, arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **DoRA** — weight-decomposed LoRA (сильнее LoRA) | `models/lora.py` (дефолт) | [Liu et al. 2024, arXiv:2402.09353](https://arxiv.org/abs/2402.09353) |
| **Supervised Contrastive Loss (SupCon)** — разделение путаемых классов | `train/losses.py` | [Khosla et al. 2020, arXiv:2004.11362](https://arxiv.org/abs/2004.11362) |
| **Focal Loss** — классификация | `train/losses.py` | [Lin et al. 2017, arXiv:1708.02002](https://arxiv.org/abs/1708.02002) |
| **Mixup** (обсуждался как анти-оверфит) | — | [Zhang et al. 2017, arXiv:1710.09412](https://arxiv.org/abs/1710.09412) |
| **PyRadiomics** — извлечение радиомических признаков | `radiomics/extract.py` | [van Griethuysen et al. 2017, Cancer Research, doi:10.1158/0008-5472.CAN-17-0339](https://doi.org/10.1158/0008-5472.CAN-17-0339) |
| **ITKElastix** — регистрация фаз (написан скрипт) | `scripts/register_phases.py` | [itk-elastix](https://github.com/InsightSoftwareConsortium/ITKElastix) |

---

## 4. Работы для сравнения (SOTA: классификация опухолей печени по мультифазной КТ)

| Работа | Идея (что берём/сравниваем) | Ссылка |
|---|---|---|
| **LIDIA** (MICCAI 2024) | Итеративное слияние фаз + **asymmetric contrastive learning** для спорных классов. Основа нашего contrastive-подхода | [arXiv:2407.13217](https://arxiv.org/html/2407.13217) |
| **Cross-Phase Fusion Transformer** | Двунаправленное взаимодействие признаков между фазами (наша cross-attention фузия) | [Springer, 10.1007/978-3-031-51455-5_15](https://link.springer.com/chapter/10.1007/978-3-031-51455-5_15) |
| **TransLiver** | Гибридный трансформер + cross-phase tokens для мультифазной классификации | по названию (MICCAI/CVPR-adjacent) |
| **MULLET** | Трансформерный модуль исследования межфазного контекста | по названию |
| **Adversarial Multi-Task Learning** | Сегментация + **регрессия динамики контрастирования** + классификация (мультизадачность, как у нас) | [arXiv:2511.20793](https://arxiv.org/pdf/2511.20793) |
| **LiLNet / flexible framework** | Вариативная мультифазность + **клинические данные**, высокая точность | [PMC11450020](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450020/) |
| **GIIM** | Граф зависимостей между «видами» (фазами) | [arXiv:2603.09446](https://arxiv.org/pdf/2603.09446) |
| **DL for differential diagnosis of malignant hepatic tumors** | Мультифаз КТ + клинические данные | [PMC8474892](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8474892/) |
| **Focal liver lesion diagnosis (multistage CT)** | Nature Communications 2024 | [s41467-024-51260-6](https://www.nature.com/articles/s41467-024-51260-6) |
| **Yasaka et al.** — CNN дифференцировка образований печени | Radiology 2017, AUC ~0.92 (органные/грубые классы) | [Radiology, radiol.2017170706](https://pubs.rsna.org/doi/abs/10.1148/radiol.2017170706) |
| **Beyond radiologist-level lesion detection** | iScience 2023 | [S2589-0042(23)02260-5](https://www.cell.com/iscience/fulltext/S2589-0042(23)02260-5) |

**НЕ сравнимо напрямую (для контекста, разная постановка):**

| Работа | Почему не сравнима |
|---|---|
| [Frontiers in Oncology 2026, fonc.2026.1836325](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2026.1836325/full) | 3 класса (norm/доброкач./злокач.), **2D-срезы**, 327 изображений, **сплит по срезам** (риск утечки), тест 66 картинок. Их 96.97% ≠ наша 5-классовая 3D-задача по пациентам |

---

## 5. Метрики (для сравнения с бенчмарком)

Считаются по протоколу авторов датасета: на уровне **пациента**, **5-fold CV**.

| Метрика | Что | Референс |
|---|---|---|
| **Balanced accuracy** | средняя по классам доля верных (устойчива к дисбалансу) | scikit-learn |
| **AUC** (macro one-vs-rest) | качество ранжирования, порог-независимая | scikit-learn `roc_auc_score` |
| **macro-F1** | среднее F1 по классам | scikit-learn |
| **Sensitivity / Specificity** (per-class) | чувствительность/специфичность по каждому подтипу | из confusion matrix |
| **Cohen's κ** | согласие с поправкой на случай; интерпретация 0.41–0.60 = «умеренное» | [Landis & Koch 1977, Biometrics 33(1)](https://doi.org/10.2307/2529310) |
| **Dice** (сегментация) | перекрытие предсказанной и эталонной маски | стандарт |

---

## 6. Наши результаты (сводка, 5-fold CV по пациентам, imaging-only)

Лучший сетап: reduced DoRA (0.32M) + радиомика (early) + zoom-in + box-промпт +
**supervised contrastive** + гибрид (seg=PVP / cls=4 фазы).

| Метрика | Значение |
|---|---|
| balanced accuracy | **0.61 ± 0.04** |
| accuracy | 0.61 ± 0.04 |
| macro-F1 | 0.60 ± 0.04 |
| AUC (macro OVR) | **0.85 ± 0.02** |
| Cohen's κ | **0.51 ± 0.05** (умеренное согласие) |
| Dice (сегментация) | 0.63–0.65 |

**Абляция contrastive:** bal_acc 0.58 → 0.61, κ 0.47 → 0.51, разброс между фолдами
почти вдвое меньше.

**Per-class:** HH (гемангиома) — легко (sens 0.87, AUC 0.98); HCC/ICC/CRLM/BCLM
путаются внутри групп. Основные ошибки: **CRLM↔BCLM** (метастаз-vs-метастаз, ~28% всех
ошибок) и **HCC↔ICC** (первичный-vs-первичный). Обе пары различаются **клиническим
контекстом** (пол, цирроз/гепатит), которого нет в изображении.

**Бенчмарк датасета:** ~0.96 accuracy / AUC ~0.98 — почти наверняка использует
клинические данные (недоступные в imaging-only постановке). Наш результат сравним с их
**imaging-only** бейзлайном, не с топ-числом.

---

## 7. Сравнительные таблицы

### Таблица A — 5-классовая дифдиагностика (та же задача, что у нас: HCC/ICC/CRLM/BCLM/HH)

Прямых 5-классовых работ на этом датасете мало (в основном сам бенчмарк). Родственные
работы делают меньше классов — указано в столбце «классы».

| Работа | Классы | Accuracy | AUC | κ | Данные / примечание |
|---|---|---|---|---|---|
| [**MCT-LTDiag benchmark**](https://www.nature.com/articles/s41597-025-06343-4) | 5 (те же) | ~0.96 | ~0.98 | — | вероятно **+ клиника**; 5-fold, по пациентам |
| **Наш метод (imaging-only)** | **5 (те же)** | **0.61 ± 0.04** | **0.85 ± 0.02** | **0.51 ± 0.05** | **только КТ**; 5-fold, по пациентам |
| [LiLNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450020/) | 3 (HCC/ICC/метастаз) | 0.887 | 0.956 | — | меньше классов, внешняя валидация |
| [Yasaka et al. (Radiology 2017)](https://pubs.rsna.org/doi/abs/10.1148/radiol.2017170706) | ~4 (HCC/ICC/гемангиома/метастаз) | — | 0.92 | — | меньше классов |
| H-LSTM | мульти-класс | — | 0.93 (AUROC) | — | HCC 0.97, ICC 0.90 |


### Таблица B — бинарная задача (доброкачественная vs злокачественная)

Клинически ключевой вопрос — «рак ли это». Наш бинарный результат выведен **из той же
confusion matrix** (доброкач. = HH; злокач. = HCC/ICC/CRLM/BCLM).

| Работа | Accuracy | AUC | Balanced acc | Sens (злокач.) | Данные / примечание |
|---|---|---|---|---|---|
| **Наш метод (derived, imaging-only)** | **0.94** | **~0.98** | **0.91** | **0.95** | из 5-классовой модели, 5-fold, по пациентам |
| [LiLNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450020/) | 0.947 | 0.972 | — | — | внешняя валидация в 4 центрах |
| [Frontiers Oncology 2026](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2026.1836325/full) | 0.97* | 0.995* | — | — | *3 класса (+norm), **2D-срезы, сплит по срезам** — завышено, не сравнивать напрямую |

*Наш бинарный расчёт (по out-of-fold confusion, 516 пациентов): злокач. распознаны
401/421 (sens 0.95), доброкач. 83/95 (spec 0.87), общая acc 484/516 = 0.94, AUC ≈ 0.98
(one-vs-rest для HH).*

### Таблица C — сегментация опухоли печени (Dice)

Сегментация — вспомогательная задача нашей мультизадачной модели (основной фокус —
классификация). Важно: сегментация опухолей **принципиально труднее** сегментации органа
(dice печени как органа ~0.96, а опухоли — 0.6–0.75 у сильных методов).

| Работа | Dice (опухоль) | Задача / примечание |
|---|---|---|
| [LiTS challenge (топ-методы)](https://arxiv.org/abs/1901.04056) | **0.67–0.74** | fully automatic, native-space; lesion sensitivity 46–63% |
| [nnU-Net (спец. система детекции+seg)](https://www.nature.com/articles/s41592-020-01008-z) | ~0.82 (lesion-level) | specialized, сильный baseline |
| [Swin UNETR](https://arxiv.org/abs/2201.01266) / трансформерные | ~0.7 | современные backbone'ы |
| **Наш метод** | **0.63–0.65** (пик ~0.71) | **semi-automatic** (box из GT-маски) + zoom crop-space; 5-fold |
| [SegVol (наша база)](https://arxiv.org/abs/2311.13385) | — | 3D foundation, prompt-based (box/point/text) |

**Дополнительные референсы по сегментации:**

| Работа | Ссылка |
|---|---|
| **LiTS** — Liver Tumor Segmentation Benchmark | [arXiv:1901.04056](https://arxiv.org/abs/1901.04056) |
| **nnU-Net** — self-configuring segmentation | [Isensee et al., Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z) |
| **Swin UNETR** — трансформер для 3D медицинской сегментации | [Hatamizadeh et al. 2022, arXiv:2201.01266](https://arxiv.org/abs/2201.01266) |
| **TriALS** — triphasic-aided liver lesion segmentation benchmark | [arXiv:2605.16572](https://arxiv.org/pdf/2605.16572) |
| **MULLET / Cross-Phase Fusion Transformer** — мультифазная seg | см. раздел 4 |

---

## 8. Полный список источников

Все упомянутые работы и ссылки одним списком.

**Датасет и бенчмарк**
1. MCT-LTDiag (данные) — Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/S3RW15
2. MCT-LTDiag (статья) — *A Multi-phase CT Dataset for Automated Differential Diagnosis of Liver Tumors*, Nature Scientific Data 2025: https://www.nature.com/articles/s41597-025-06343-4
3. Multi-phase LT Benchmark / RU-Net (код + клинические метаданные) — https://github.com/Hoyant-Su/Multi-phase_LT_Benchmark

**Базовая модель**
4. SegVol — *SegVol: Universal and Interactive Volumetric Medical Image Segmentation*, arXiv:2311.13385 (NeurIPS 2024): https://arxiv.org/abs/2311.13385
5. BAAI/SegVol (веса) — https://huggingface.co/BAAI/SegVol

**Применённые методы**
6. LoRA — Hu et al. 2021, arXiv:2106.09685: https://arxiv.org/abs/2106.09685
7. DoRA — *Weight-Decomposed Low-Rank Adaptation*, Liu et al. 2024, arXiv:2402.09353: https://arxiv.org/abs/2402.09353
8. Supervised Contrastive Learning (SupCon) — Khosla et al. 2020, arXiv:2004.11362: https://arxiv.org/abs/2004.11362
9. Focal Loss — Lin et al. 2017, arXiv:1708.02002: https://arxiv.org/abs/1708.02002
10. Mixup — Zhang et al. 2017, arXiv:1710.09412: https://arxiv.org/abs/1710.09412
11. PyRadiomics — van Griethuysen et al. 2017, Cancer Research, doi:10.1158/0008-5472.CAN-17-0339: https://doi.org/10.1158/0008-5472.CAN-17-0339
12. ITKElastix — https://github.com/InsightSoftwareConsortium/ITKElastix

**SOTA — классификация опухолей печени**
13. LIDIA — *Precise Liver Tumor Diagnosis... Iterative Fusion and Asymmetric Contrastive Learning*, MICCAI 2024, arXiv:2407.13217: https://arxiv.org/html/2407.13217
14. Cross-Phase Fusion Transformer — Springer, 10.1007/978-3-031-51455-5_15: https://link.springer.com/chapter/10.1007/978-3-031-51455-5_15
15. TransLiver — гибридный трансформер + cross-phase tokens (по названию)
16. MULLET — трансформерное исследование межфазного контекста (по названию)
17. Adversarial Multi-Task Learning (seg + enhancement regression + classification) — arXiv:2511.20793: https://arxiv.org/pdf/2511.20793
18. LiLNet / flexible DL framework — PMC11450020: https://pmc.ncbi.nlm.nih.gov/articles/PMC11450020/
19. GIIM — граф зависимостей между видами, arXiv:2603.09446: https://arxiv.org/pdf/2603.09446
20. DL for differential diagnosis of malignant hepatic tumors (multi-phase + clinical) — PMC8474892: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8474892/
21. Focal liver lesion diagnosis with deep learning and multistage CT — Nature Communications 2024: https://www.nature.com/articles/s41467-024-51260-6
22. Yasaka et al. — *Deep Learning with CNN for Differentiation of Liver Masses at Dynamic CE-CT*, Radiology 2017: https://pubs.rsna.org/doi/abs/10.1148/radiol.2017170706
23. Beyond radiologist-level liver lesion detection — iScience 2023: https://www.cell.com/iscience/fulltext/S2589-0042(23)02260-5
24. Frontiers in Oncology 2026 (3-класс, 2D — НЕ сравнимо) — https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2026.1836325/full

**Метрики**
25. Cohen's κ — Landis & Koch 1977, Biometrics 33(1), doi:10.2307/2529310: https://doi.org/10.2307/2529310
26. scikit-learn (реализация метрик) — https://scikit-learn.org/stable/modules/model_evaluation.html

**SOTA — сегментация опухоли печени**
27. LiTS — *The Liver Tumor Segmentation Benchmark*, arXiv:1901.04056: https://arxiv.org/abs/1901.04056
28. nnU-Net — Isensee et al., Nature Methods 2021: https://www.nature.com/articles/s41592-020-01008-z
29. Swin UNETR — Hatamizadeh et al. 2022, arXiv:2201.01266: https://arxiv.org/abs/2201.01266
30. TriALS — Triphasic-Aided Liver Lesion Segmentation Benchmark, arXiv:2605.16572: https://arxiv.org/pdf/2605.16572

---

## 9. Стандартные форматы сравнительных таблиц

В работах по этой теме таблицы сравнения почти стандартизированы. Ниже — канонические
форматы, заполненные нашими результатами. Пустые строки-бейзлайны — что обычно
добавляют для сравнения (наши прочерки `—` = не запускали / не считали).

### Таблица D — Классификация: сравнение методов (overall)

Стандартные столбцы: метод, Accuracy, macro-AUC, macro-F1, Cohen's κ (иногда + balanced acc).

| Method | Accuracy | AUC | macro-F1 | κ | Bal. acc |
|---|---|---|---|---|---|
| Radiomics + RF/SVM (baseline) | — | — | — | — | — |
| 3D ResNet / DenseNet (baseline) | — | — | — | — | — |
| ViT / Swin (baseline) | — | — | — | — | — |
| [MCT-LTDiag benchmark (prior SOTA)](https://www.nature.com/articles/s41597-025-06343-4) | ~0.96 | ~0.98 | — | — | — |
| **Ours (SegVol + DoRA + radiomics + SupCon)** | **0.61** | **0.85** | **0.60** | **0.51** | **0.61** |

> Стандартные бейзлайны для этой строки-задачи: **radiomics+ML** (RF/SVM/XGBoost),
> **3D CNN** (ResNet3D, DenseNet3D), **трансформеры** (ViT, Swin), + prior SOTA на датасете.

### Таблица E — Классификация: per-class (стандарт)

Стандартные столбцы: класс × Sensitivity / Specificity / Precision / F1 / AUC.
Наши числа — out-of-fold по всем 516 пациентам (5-fold, imaging-only, + SupCon).

| Class | Sensitivity | Specificity | Precision | F1 | AUC |
|---|---|---|---|---|---|
| HCC | 0.58 | 0.92 | 0.64 | 0.61 | 0.85 |
| ICC | 0.51 | 0.89 | 0.53 | 0.52 | 0.78 |
| CRLM | 0.52 | 0.86 | 0.48 | 0.50 | 0.82 |
| BCLM | 0.57 | 0.89 | 0.60 | 0.58 | 0.86 |
| HH | 0.87 | 0.95 | 0.81 | 0.84 | 0.98 |
| **macro avg** | **0.61** | **0.90** | **0.61** | **0.61** | **0.86** |

### Таблица F — Сегментация: сравнение методов (стандарт)

Стандартные столбцы: Dice (DSC), Jaccard/IoU, HD95, ASD, Sensitivity, Precision.
У нас посчитан Dice; Jaccard выведен из Dice (IoU = D/(2−D)); дистанционные метрики
(HD95, ASD) требуют сохранённых воксельных предсказаний — не считали.

| Method | Dice | Jaccard/IoU | HD95 | ASD | Задача |
|---|---|---|---|---|---|
| U-Net / V-Net (baseline) | — | — | — | — | fully auto |
| [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z) | ~0.82* | — | — | — | fully auto, specialized |
| [Swin UNETR](https://arxiv.org/abs/2201.01266) | ~0.70 | — | — | — | fully auto |
| [LiTS top methods](https://arxiv.org/abs/1901.04056) | 0.67–0.74 | — | — | — | fully auto, native |
| **Ours (SegVol hybrid)** | **0.63–0.65** | **~0.47** | — | — | **semi-auto** (box), zoom crop |

> \*lesion-level. Стандартные бейзлайны сегментации: **U-Net, V-Net, nnU-Net, Attention
> U-Net, TransUNet, Swin UNETR, SegVol**. Дистанционные метрики (HD95, ASD) добавляют,
> когда важна точность границы; для лезий часто ограничиваются Dice + Jaccard +
> lesion-wise sensitivity.

> **Оговорка (та же):** наш Dice — semi-automatic (рамка из GT) + crop-space, поэтому
> сравнение с fully-automatic native-space методами не «в лоб».
