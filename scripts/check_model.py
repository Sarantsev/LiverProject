"""Проверка реальной модели: загрузка SegVol с HuggingFace + сборка наших голов
+ forward на синтетике (без данных). Прогон на GPU де-рискует пайплайн до данных.

Запуск (из корня репозитория):
    python scripts/check_model.py                      # device=auto (cuda если есть)
    python scripts/check_model.py --device cpu         # принудительно CPU (медленно)
    python scripts/check_model.py --config configs/default.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from liver_sppvr.train.build import load_config, build_segvol_multitask
from liver_sppvr.utils.device import resolve_device


def main():
    ap = argparse.ArgumentParser(description="Smoke-проверка реальной модели SegVol + головы.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--batch", type=int, default=1, help="размер батча для теста")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = resolve_device(args.device)
    n_phases = len(cfg["multiphase"]["phases"])
    d, h, w = cfg["segvol"]["spatial_size"]

    print(f"device={dev} | фаз={n_phases} | объём={(d, h, w)}")
    print("Гружу SegVol с HuggingFace (первый раз скачает веса ~сотни МБ)...")
    model = build_segvol_multitask(cfg, dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Модель собрана на {dev} | параметров: {n_params/1e6:.1f}M "
          f"(обучаемых: {n_train/1e6:.1f}M)")

    phases = torch.rand(args.batch, n_phases, 1, d, h, w, device=dev)
    model.eval()
    with torch.no_grad():
        out = model(phases=phases, seg_prompt={"text": ["liver tumor"] * args.batch},
                    return_seg=True, return_cls=True)

    print("--- выходы forward ---")
    for k, v in out.items():
        print(f"  {k}: {getattr(v, 'shape', v)}")

    if dev.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"пик VRAM: {mem:.2f} ГБ")
    print("OK — реальный SegVol + головы работают.")


if __name__ == "__main__":
    main()
