# Liver-SPPVR — СППВР для диагностики опухолей печени на базе SegVol

End-to-end пайплайн поддержки принятия врачебных решений: дообученная **SegVol**
(3D foundation model, NeurIPS 2024) + **классификационная голова** (мультиклассовая
дифференциальная диагностика) + **радиомический модуль PyRadiomics**. Только КТ,
мультифазный fusion (4 фазы). Подробности и обоснование — в [PLAN.md](PLAN.md).

## Структура

```
configs/
  default.yaml            # все пути/гиперпараметры; device: auto
  radiomics_params.yaml   # параметры PyRadiomics
liver_sppvr/
  models/
    cls_head.py           # классификационная голова (masked pooling + гибрид с радиомикой)
    multiphase.py         # fusion 4 фаз (concat_stem | attention)
    segvol_multitask.py   # обёртка: encoder SegVol + сегментация + fusion + cls
  data/
    preprocess.py         # HU windowing, resample
    manifest.py           # единый CSV-манифест (build/load/validate)
    dataset.py            # MultiPhaseLiverDataset + collate
  train/
    losses.py             # Dice+BCE (seg) и Focal/CE (cls), MultiTaskLoss
    engine.py             # train_one_epoch / evaluate (dice, accuracy, macro-F1)
    build.py              # сборка реальной модели (HF SegVol) и заглушки для dry-run
    multitask.py          # CLI обучения
  radiomics/extract.py    # PyRadiomics: признаки из маски опухоли
  inference/report.py     # end-to-end отчёт СППВР (маска + тип + объём + радиомика)
scripts/build_manifest.py # CLI сборки манифеста
tests/                    # CPU-тесты (без GPU/данных): smoke, data, radiomics
```

## Среда

- **Разработка** — на этой машине (без GPU). Окружение: `../segvol_env`
  (torch 1.13.1, monai 0.9.0). Доп. зависимости — `requirements.txt`.
- **Обучение** — на отдельном GPU-девайсе. Код device-agnostic
  (`device: auto`), переносится как есть. Туда же нужно поставить `pyradiomics`.

## Быстрая проверка (CPU, без данных)

```bash
../segvol_env/bin/python tests/test_smoke.py      # модели и обёртка
../segvol_env/bin/python tests/test_data.py       # манифест + датасет
../segvol_env/bin/python tests/test_radiomics.py  # радиомика (skip без pyradiomics)

# прогон цикла обучения на синтетике (заглушка вместо SegVol):
../segvol_env/bin/python -m liver_sppvr.train.multitask \
    --config configs/default.yaml --dry-run --device cpu
```

## Запуск на GPU-девайсе

```bash
# 1) данные -> манифест
python scripts/build_manifest.py --root /data/MCT-LTDiag --dataset MCT-LTDiag \
    --out data/manifest.csv

# 2) обучение (грузит предобученный SegVol с HuggingFace BAAI/SegVol)
python -m liver_sppvr.train.multitask --config configs/default.yaml
```

## Датасеты (с метками типа опухоли — для классификации)

| Датасет | Роль | Доступ |
|---|---|---|
| **MCT-LTDiag** (4 фазы, 517 пациентов, 5 классов + маски) | основной | Harvard Dataverse |
| **HCC-TACE-Seg** (105 HCC + маски) | класс HCC / внешняя валидация | TCIA |
| **Colorectal-Liver-Metastases** (~197, CRLM + маски) | класс метастазов | TCIA |

> LiTS / MSD-Liver / 3D-IRCADb / CHAOS — только маски без типа опухоли;
> для обучения классификатора не используются (см. PLAN.md).
