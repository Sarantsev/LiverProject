"""Classification metrics + result writers shared by every trainer and the cascade.

Metric set follows the liver-tumour classification literature and the MCT-LTDiag
benchmark protocol: accuracy, macro one-vs-rest AUC, balanced accuracy, macro-F1,
Cohen's kappa, per-class sensitivity/specificity/F1/AUC and the confusion matrix.
"""
from __future__ import annotations

import json
import os

import numpy as np

SUMMARY_KEYS = ("auc", "accuracy", "balanced_accuracy", "macro_f1", "kappa")


def per_class_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    """Per-class sensitivity / specificity / F1 / one-vs-rest AUC."""
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    labels = list(range(num_classes))
    yt, yp, pr = np.array(y_true), np.array(y_pred), np.array(y_prob)
    cm = confusion_matrix(yt, yp, labels=labels)                 # rows=true, cols=pred
    total = cm.sum()
    f1 = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
    sens, spec, aucs = (np.full(num_classes, np.nan) for _ in range(3))
    for c in labels:
        tp = cm[c, c]; fn = cm[c, :].sum() - tp; fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        if tp + fn > 0:
            sens[c] = tp / (tp + fn)
        if tn + fp > 0:
            spec[c] = tn / (tn + fp)
        yc = (yt == c).astype(int)
        if 0 < yc.sum() < len(yc):
            try:
                aucs[c] = roc_auc_score(yc, pr[:, c])
            except Exception:
                pass
    return {"per_sens": sens.tolist(), "per_spec": spec.tolist(),
            "per_f1": [float(x) for x in f1], "per_auc": aucs.tolist()}


def cls_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                                 confusion_matrix, f1_score, roc_auc_score)
    labels = list(range(num_classes))
    yt, yp, pr = np.array(y_true), np.array(y_pred), np.array(y_prob)
    m = {"accuracy": float(np.mean(yt == yp)),
         "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
         "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
         "kappa": float(cohen_kappa_score(yt, yp, labels=labels)),
         "confusion": confusion_matrix(yt, yp, labels=labels).tolist()}
    try:
        m["auc"] = float(roc_auc_score(yt, pr, multi_class="ovr", average="macro", labels=labels))
    except Exception:
        m["auc"] = float("nan")                      # a class may be absent in a small fold
    m.update(per_class_metrics(yt, yp, pr, num_classes))
    return m


def print_per_class(ev: dict, class_names) -> None:
    if "per_f1" not in ev:
        return
    n = len(ev["per_f1"])
    names = list(class_names) if class_names else [f"c{i}" for i in range(n)]
    print("  per-class:  " + f"{'class':<8} {'sens':>6} {'spec':>6} {'F1':>6} {'AUC':>6}")
    for i in range(n):
        print(f"    {names[i]:<8} {ev['per_sens'][i]:>6.3f} {ev['per_spec'][i]:>6.3f} "
              f"{ev['per_f1'][i]:>6.3f} {ev['per_auc'][i]:>6.3f}")
    print(f"  Cohen's kappa = {ev.get('kappa', float('nan')):.4f}")


def save_confusion(cm, class_names, out_prefix: str, title: str = "Confusion matrix") -> None:
    """rows=true, cols=pred -> out_prefix.csv (+ out_prefix.png when matplotlib is available)."""
    import csv
    cm = np.array(cm)
    names = list(class_names) if class_names else [f"c{i}" for i in range(len(cm))]
    with open(out_prefix + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + names)
        for i, name in enumerate(names):
            w.writerow([name] + cm[i].tolist())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(5.6, 5))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized (recall)")
        fig.tight_layout(); fig.savefig(out_prefix + ".png", dpi=150); plt.close(fig)
        print(f"  confusion matrix -> {out_prefix}.csv / .png")
    except Exception as e:
        print(f"  confusion matrix -> {out_prefix}.csv (PNG skipped: {e})")


def write_summary(results, work_dir: str, name: str, kfold: int, class_names=None,
                  keys=SUMMARY_KEYS) -> dict:
    """Per-fold results -> summary.json + summary.csv (one row per metric: folds, mean, std)
    + pooled out-of-fold confusion matrix. Returns the summary dict."""
    os.makedirs(work_dir, exist_ok=True)

    def ms(key):
        v = np.array([r.get(key, float("nan")) for r in results], dtype=float)
        v = v[~np.isnan(v)]
        return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))

    summary = {"model": name, "kfold": kfold,
               "per_fold": [{"fold": r["fold"], **{k: r.get(k) for k in keys},
                             "best_epoch": r.get("epoch")} for r in results],
               "mean_std": {k: dict(zip(("mean", "std"), ms(k))) for k in keys}}
    with open(os.path.join(work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(work_dir, "summary.csv"), "w") as f:
        f.write("metric," + ",".join(f"fold{r['fold']}" for r in results) + ",mean,std\n")
        for k in keys:
            mu, sd = ms(k)
            f.write(f"{k}," + ",".join(f"{r.get(k, float('nan')):.4f}" for r in results)
                    + f",{mu:.4f},{sd:.4f}\n")
    print(f"\nsummary -> {work_dir}/summary.json + summary.csv")
    if len(results) > 1:
        print(f"===== {name}: cross-validation summary (best epoch per fold) =====")
        for k in keys:
            mu, sd = ms(k)
            print(f"  {k:<18} {mu:.4f} ± {sd:.4f}")
        conf = [np.array(r["confusion"]) for r in results if r.get("confusion")]
        if conf:
            save_confusion(np.sum(conf, axis=0), class_names,
                           os.path.join(work_dir, "confusion_total"),
                           title=f"{name} (out-of-fold, all patients)")
    return summary
