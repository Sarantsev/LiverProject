from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


@torch.no_grad()
def generate_report(
    model,
    phases: torch.Tensor,                  # (1, P, 1, D, H, W)
    class_names: Sequence[str],
    device,
    spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    seg_text: str = "liver tumor",
    radiomics_params: Optional[str] = None,
) -> dict:
    """Вернуть dict-отчёт по одному пациенту.

    Поля: predicted_type, class_probs, tumor_voxels, tumor_volume_ml,
          phase_weights (если fusion=attention), radiomics (если доступен PyRadiomics).
    """
    model.eval()
    phases = phases.to(device)
    out = model(phases=phases, seg_prompt={"text": [seg_text]},
                return_seg=True, return_cls=True)

    probs = torch.softmax(out["cls_logits"], dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    mask = (torch.sigmoid(out["seg_logits"][0, 0]) > 0.5).cpu().numpy().astype(np.uint8)

    voxels = int(mask.sum())
    vol_ml = voxels * float(np.prod(spacing_mm)) / 1000.0

    report = {
        "predicted_type": class_names[pred_idx],
        "class_probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "tumor_voxels": voxels,
        "tumor_volume_ml": round(vol_ml, 1),
        "mask": mask,
    }
    pw = out.get("phase_weights")
    if pw is not None:
        report["phase_weights"] = pw[0].cpu().numpy().tolist()

    # радиомика по опорной фазе (по умолчанию портальная = индекс 2, если есть)
    try:
        from ..radiomics import RADIOMICS_AVAILABLE, extract_from_arrays
        if RADIOMICS_AVAILABLE and voxels > 0:
            ref_phase = phases[0, min(2, phases.shape[1] - 1), 0].cpu().numpy()
            report["radiomics"] = extract_from_arrays(
                ref_phase, mask, spacing=spacing_mm, params=radiomics_params)
    except Exception as e:  # радиомика не критична для отчёта
        report["radiomics_error"] = str(e)

    return report
