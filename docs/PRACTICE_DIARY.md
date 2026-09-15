# Дневник прохождения практики

Проект: **liver-sppvr** — мультизадачная модель на базе SegVol для анализа мультифазной КТ печени
(сегментация опухоли + дифференциальная диагностика 5 подтипов образований на датасете MCT-LTDiag).

---

## Индивидуальное задание

**На русском:**
Разработать и исследовать метод глубокого обучения для детекции и дифференциальной диагностики
опухолей печени по мультифазной компьютерной томографии. На основе предобученной фундаментальной
модели SegVol реализовать мультизадачную архитектуру, одновременно решающую задачу сегментации
опухоли и классификации её типа (5 классов: HCC, ICC, CRLM, BCLM, гемангиома). Провести обучение,
оценку по протоколу 5-fold кросс-валидации на уровне пациента и сравнение с существующими работами.

**На английском:**
Develop and study a deep-learning method for detection and differential diagnosis of liver tumors
from multiphase CT. Based on the pretrained SegVol foundation model, implement a multi-task
architecture that simultaneously performs tumor segmentation and tumor-type classification
(5 classes: HCC, ICC, CRLM, BCLM, hemangioma). Train the model, evaluate it under a patient-level
5-fold cross-validation protocol, and compare it against prior work.

---

## Планируемые результаты

**На русском:**
Обученная мультизадачная модель; воспроизводимый пайплайн подготовки данных, обучения и оценки;
измеренные метрики (balanced accuracy, AUC, macro-F1, Cohen's κ, Dice) по 5-fold CV на уровне
пациента; анализ ошибок (confusion matrix); сравнительные таблицы с SOTA и бенчмарком датасета;
обоснованные направления улучшения качества.

**На английском:**
A trained multi-task model; a reproducible pipeline for data preparation, training and evaluation;
measured metrics (balanced accuracy, AUC, macro-F1, Cohen's κ, Dice) under patient-level 5-fold CV;
error analysis (confusion matrix); comparison tables against SOTA and the dataset benchmark; and
justified directions for further quality improvement.

---

## Краткое описание достигнутого результата

**На русском:**
Реализован и обучен мультизадачный SegVol: PEFT-адаптация энкодера (LoRA/DoRA), голова классификации,
мультифазное слияние (cross-attention), ранняя фузия радиомических признаков и supervised contrastive
loss для разделения путаемых классов. Итоговые метрики (imaging-only, 5-fold CV по пациентам):
balanced accuracy 0.61 ± 0.04, AUC 0.85 ± 0.02, Cohen's κ 0.51 ± 0.05, Dice сегментации 0.63–0.65.
Contrastive-компонент поднял bal_acc с 0.58 до 0.61 и снизил разброс между фолдами. Анализ confusion
matrix показал систематическую путаницу внутри клинически родственных пар (CRLM↔BCLM, HCC↔ICC),
различимых по клиническому контексту, а не по изображению. Составлены сравнительные таблицы с SOTA;
начата кросс-датасетная валидация сегментации на LiTS (Task03_Liver).

**На английском:**
A multi-task SegVol was implemented and trained: PEFT encoder adaptation (LoRA/DoRA), a classification
head, multiphase cross-attention fusion, early fusion of radiomics features, and a supervised
contrastive loss to separate confusable classes. Final metrics (imaging-only, patient-level 5-fold CV):
balanced accuracy 0.61 ± 0.04, AUC 0.85 ± 0.02, Cohen's κ 0.51 ± 0.05, segmentation Dice 0.63–0.65.
The contrastive component raised balanced accuracy from 0.58 to 0.61 and reduced inter-fold variance.
Confusion-matrix analysis revealed systematic errors within clinically related pairs (CRLM↔BCLM,
HCC↔ICC) that are distinguished by clinical context rather than by imaging. Comparison tables against
SOTA were compiled, and cross-dataset segmentation validation on LiTS (Task03_Liver) was started.

---

## Саморефлексия

**На русском:**
За время практики я освоил работу с фундаментальными моделями в медицинской визуализации и методами
их эффективной дообучения (LoRA/DoRA), построение мультизадачных архитектур и обучение на удалённом
GPU-сервере. Наиболее ценным стало понимание принципов честной оценки: разбиение на уровне пациента
во избежание утечки, интерпретация метрик с поправкой на дисбаланс, критический анализ чужих
результатов (например, завышение Dice за счёт 2D-оценки по срезам). Я научился отделять реальный
прогресс от артефактов протокола и формулировать ограничения метода. Отдельным опытом стала работа с
инфраструктурными трудностями (ограничения сети на сервере, подготовка и конвертация датасетов). Вижу
чёткие направления развития: добавление клинического контекста для разделения оставшихся ошибок и
доведение кросс-датасетной валидации на LiTS до head-to-head сравнения.

**На английском:**
During the internship I learned to work with foundation models in medical imaging and with
parameter-efficient fine-tuning (LoRA/DoRA), to build multi-task architectures and train them on a
remote GPU server. The most valuable insight was the principles of honest evaluation: patient-level
splits to avoid leakage, interpreting metrics under class imbalance, and critically analysing others'
results (e.g., inflated Dice from 2D slice-wise evaluation). I learned to separate real progress from
protocol artefacts and to state a method's limitations. Handling infrastructure difficulties (network
restrictions on the server, dataset preparation and conversion) was a separate valuable experience.
I see clear next steps: adding clinical context to resolve the remaining errors, and completing the
cross-dataset LiTS validation into a head-to-head comparison.

---

# Записи дневника

## Этап 1
**Дата начала:** 06/15/2026 **Дата окончания:** 06/25/2026

**Содержание практики (на русском):**
Изучение предметной области (дифференциальная диагностика опухолей печени по мультифазной КТ) и
фундаментальной модели SegVol. Развёртывание окружения на удалённом GPU-сервере, подготовка датасета
MCT-LTDiag: распаковка архивов пациентов, конвертация фаз (nc/art/pvp/delay) и масок в единый формат,
построение манифеста и базового пайплайна загрузки данных.

**Содержание практики (на английском):**
Studying the domain (differential diagnosis of liver tumors on multiphase CT) and the SegVol
foundation model. Setting up the environment on a remote GPU server and preparing the MCT-LTDiag
dataset: unpacking per-patient archives, converting phases (nc/art/pvp/delay) and masks to a unified
format, building the manifest and the base data-loading pipeline.

**Результаты практики (на русском):**
Готово рабочее окружение и воспроизводимый пайплайн подготовки данных (516/517 пациентов). Собран
манифест с разбиением на уровне пациента. Запущен базовый прогон SegVol как отправная точка.

**Результаты практики (на английском):**
A working environment and a reproducible data-preparation pipeline are ready (516/517 patients). A
manifest with patient-level splitting was built. A baseline SegVol run was launched as a starting point.

---

## Этап 2
**Дата начала:** 06/26/2026 **Дата окончания:** 07/05/2026

**Содержание практики (на русском):**
Реализация мультизадачной архитектуры: голова классификации поверх энкодера SegVol и PEFT-адаптация
(LoRA, затем DoRA как более сильный вариант) при замороженном энкодере для борьбы с переобучением.
Реализация мультифазного слияния признаков (attention и cross-attention), настройка функций потерь
(Dice + Focal) и обучение на сервере с mixed precision.

**Содержание практики (на английском):**
Implementing the multi-task architecture: a classification head on top of the SegVol encoder and PEFT
adaptation (LoRA, then DoRA as a stronger variant) with a frozen encoder to fight overfitting.
Implementing multiphase feature fusion (attention and cross-attention), tuning the loss functions
(Dice + Focal), and training on the server with mixed precision.

**Результаты практики (на русском):**
Получена работающая мультизадачная модель, одновременно выдающая маску опухоли и её тип. Подобрана
рабочая точка обучения на GPU (batch/AMP/num_workers), устранено узкое место загрузки данных.
Зафиксированы первые сквозные метрики сегментации и классификации.

**Результаты практики (на английском):**
A working multi-task model that simultaneously outputs a tumor mask and its type was obtained. A stable
GPU training configuration was found (batch/AMP/num_workers), and the data-loading bottleneck was
eliminated. The first end-to-end segmentation and classification metrics were recorded.

---

## Этап 3
**Дата начала:** 07/06/2026 **Дата окончания:** 07/15/2026

**Содержание практики (на русском):**
Повышение качества и корректная оценка: ранняя фузия радиомических признаков (PyRadiomics), внедрение
supervised contrastive loss для разделения путаемых классов, zoom-in кроп вокруг опухоли и box-промпт
для сегментации. Переход на строгий протокол 5-fold кросс-валидации на уровне пациента, подсчёт
per-class метрик и confusion matrix, серия абляций.

**Содержание практики (на английском):**
Improving quality and evaluating rigorously: early fusion of radiomics features (PyRadiomics), adding a
supervised contrastive loss to separate confusable classes, a zoom-in crop around the tumor and a box
prompt for segmentation. Switching to a strict patient-level 5-fold cross-validation protocol, computing
per-class metrics and the confusion matrix, and running a series of ablations.

**Результаты практики (на русском):**
Итоговые метрики: balanced accuracy 0.61, AUC 0.85, Cohen's κ 0.51, Dice 0.63–0.65. Contrastive поднял
bal_acc с 0.58 до 0.61 и снизил разброс между фолдами. Анализ ошибок выявил систематическую путаницу
пар CRLM↔BCLM и HCC↔ICC, различимых по клиническому контексту, отсутствующему в изображении.

**Результаты практики (на английском):**
Final metrics: balanced accuracy 0.61, AUC 0.85, Cohen's κ 0.51, Dice 0.63–0.65. The contrastive loss
raised balanced accuracy from 0.58 to 0.61 and reduced inter-fold variance. Error analysis revealed
systematic confusion of the CRLM↔BCLM and HCC↔ICC pairs, which are distinguished by clinical context
absent from the image.

---

## Этап 4
**Дата начала:** 07/16/2026 **Дата окончания:** 07/26/2026

**Содержание практики (на русском):**
Позиционирование результата относительно литературы: сбор источников и составление сравнительных
таблиц по классификации и сегментации, критический анализ SOTA (выявление завышенных метрик из-за
2D-оценки по срезам и утечки при сплите по срезам). Запуск кросс-датасетной валидации сегментации на
LiTS (Task03_Liver): написаны скрипты подготовки данных и seg-only оценки, отлажена загрузка датасета
на сервер с ограничениями сети.

**Содержание практики (на английском):**
Positioning the result against the literature: collecting sources and building comparison tables for
classification and segmentation, and critically analysing SOTA (identifying inflated metrics from 2D
slice-wise evaluation and leakage from slice-level splits). Starting cross-dataset segmentation
validation on LiTS (Task03_Liver): writing data-preparation and seg-only evaluation scripts, and
troubleshooting dataset download onto the network-restricted server.

**Результаты практики (на русском):**
Подготовлен документ REFERENCES с источниками и сравнительными таблицами (классификация, сегментация),
где результат честно позиционирован относительно бенчмарка и SOTA. Написаны и проверены скрипты для
LiTS-валидации; сформулирован план head-to-head сравнения и дальнейшего добавления клинического
контекста.

**Результаты практики (на английском):**
A REFERENCES document with sources and comparison tables (classification, segmentation) was prepared,
honestly positioning the result relative to the benchmark and SOTA. Scripts for LiTS validation were
written and verified; a plan for a head-to-head comparison and for further adding clinical context was
formulated.

---

## Анкета удовлетворённости

**Удовлетворены ли Вы результатами полученных практических знаний, умений, навыков в период
прохождения практики?**
→ *Полностью удовлетворён* (освоены фундаментальные модели, PEFT, мультизадачное обучение, строгая
оценка и критический анализ результатов; получен измеримый результат и понятный план развития).

**Удовлетворены ли Вы качеством организационно-методического сопровождения проведения практики?**
→ *Полностью удовлетворён* (были обеспечены доступ к GPU-серверу, данные и понятная постановка задач;
возникавшие инфраструктурные трудности удавалось оперативно решать).
