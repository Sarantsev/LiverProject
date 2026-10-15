"""How much accuracy is lost in the DECISION, not in the features?

A model can rank well and still decide badly. On MCT-LTDiag the gap is large: macro-AUC
~0.91 against accuracy ~0.67, and ICC sits at sensitivity 0.26 with specificity 0.98 --
the signal is there (ICC AUC ~0.78-0.85) but plain argmax almost never selects it, because
ICC competes with HCC on imaging and loses the softmax contest.

This script re-decides from the per-patient probabilities the cascade already wrote, with
no retraining, and compares decision rules:

  argmax    what we report today
  prior     p_c / mean_i(p_ic): equalise the average predicted mass per class. Principled
            here because the true class priors are nearly uniform (max/min = 1.21)
  tuned     argmax of w_c * p_c, with w found by coordinate ascent on balanced accuracy

Every rule is fitted ONLY on the other folds and applied to the held-out fold, so the
numbers are out-of-fold throughout -- a threshold tuned on the same patients it is scored
on would be meaningless. Several --runs are averaged into an ensemble first.

With --clinical-csv a logistic regression on the 5 clinical features is added as another
probability source and blended in (weight searched out-of-fold). This is worth doing
because the deep model demonstrably under-uses those features: on this data the clinical
logreg separates ICC from HCC at 0.77 accuracy while the deep model reaches ICC
sensitivity 0.39, so late fusion recovers signal the network did not learn to read.

    PYTHONPATH=. python scripts/calibrate_decision.py \
        --runs work/cascade_segvol work/cascade_merlin \
        --clinical-csv data/clinical.csv --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np

from liver_sppvr.train.metrics import cls_metrics, print_per_class


def load_run(run_dir: str):
    """per_patient.csv -> (ids, fold, y_true, probs, class_names)."""
    path = os.path.join(run_dir, "per_patient.csv")
    rows = list(csv.DictReader(open(path)))
    names = [c[2:] for c in rows[0] if c.startswith("p_")]
    idx = {n: i for i, n in enumerate(names)}
    ids = [r["patient_id"] for r in rows]
    fold = np.array([int(r["fold"]) for r in rows])
    y = np.array([idx[r["true"]] for r in rows])
    p = np.array([[float(r[f"p_{n}"]) for n in names] for r in rows])
    return ids, fold, y, p, names


def clinical_probs(csv_path: str, config: str, ids, fold, C: int):
    """Out-of-fold class probabilities from a logistic regression on the clinical features.

    Fitted per fold on that fold's training patients only, with the same z-scoring the deep
    model uses (train statistics), so these probabilities are directly blendable.
    """
    from sklearn.linear_model import LogisticRegression
    from liver_sppvr.data.clinical import load_clinical
    from liver_sppvr.data import load_manifest
    from liver_sppvr.train.build import load_config
    from liver_sppvr.train.splits import labels_from_manifest, make_kfold_splits

    cfg = load_config(config)
    names = cfg["classifier"]["class_names"]
    lab = labels_from_manifest(load_manifest(cfg["data"]["manifest"]), names)
    pos = {pid: i for i, pid in enumerate(ids)}
    out = np.zeros((len(ids), C))
    for f, (tr, va) in enumerate(make_kfold_splits(lab, k=len(np.unique(fold)),
                                                   seed=cfg["project"]["seed"])):
        clin = load_clinical(csv_path, tr)
        tr = [p for p in tr if p in clin and p in pos]
        va = [p for p in va if p in clin and p in pos]
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            np.stack([clin[p] for p in tr]), np.array([lab[p] for p in tr]))
        pr = m.predict_proba(np.stack([clin[p] for p in va]))
        for k, pid in enumerate(va):
            out[pos[pid], m.classes_.astype(int)] = pr[k]
    return out


def blend_weight(pa, pb, y, fold, C, grid=np.linspace(0, 1, 21)):
    """Search the mixing weight a in a*pa + (1-a)*pb, out-of-fold, on balanced accuracy."""
    best_a, best = 0.5, -1.0
    for a in grid:
        pred = np.zeros(len(y), dtype=int)
        for f in np.unique(fold):
            te = fold == f
            pred[te] = (a * pa[te] + (1 - a) * pb[te]).argmax(1)
        s = _balanced_accuracy(y, pred, C)
        if s > best:
            best, best_a = s, float(a)
    return best_a


def _balanced_accuracy(y, pred, C):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(C) if (y == c).any()]))


def fit_weights(p, y, C, iters: int = 6, grid=None):
    """Coordinate ascent on per-class multiplicative weights, maximising balanced accuracy."""
    grid = grid if grid is not None else np.exp(np.linspace(-1.5, 1.5, 25))
    w = np.ones(C)
    best = _balanced_accuracy(y, (p * w).argmax(1), C)
    for _ in range(iters):
        improved = False
        for c in range(C):
            keep = w[c]
            for g in grid:
                w[c] = g
                s = _balanced_accuracy(y, (p * w).argmax(1), C)
                if s > best + 1e-9:
                    best, keep, improved = s, g, True
            w[c] = keep
        if not improved:
            break
    return w


def decide(fold, y, p, C, rule: str):
    """Out-of-fold predictions: the rule is fitted on the other folds only."""
    pred = np.zeros(len(y), dtype=int)
    for f in np.unique(fold):
        te, tr = fold == f, fold != f
        if rule == "argmax":
            pred[te] = p[te].argmax(1)
        elif rule == "prior":
            q = p[tr].mean(0).clip(min=1e-8)          # average predicted mass per class
            pred[te] = (p[te] / q).argmax(1)
        elif rule == "tuned":
            w = fit_weights(p[tr], y[tr], C)
            pred[te] = (p[te] * w).argmax(1)
        else:
            raise ValueError(rule)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="cascade dirs with per_patient.csv")
    ap.add_argument("--rules", nargs="+", default=["argmax", "prior", "tuned"])
    ap.add_argument("--clinical-csv", default=None, help="add a clinical logreg as a source")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="write a summary CSV here")
    args = ap.parse_args()

    loaded = [load_run(r) for r in args.runs]
    ids, fold, y, _, names = loaded[0]
    for other in loaded[1:]:                                   # same patients, same order
        assert other[0] == ids and (other[2] == y).all(), "runs disagree on patients/labels"
    C = len(names)

    # each run on its own, then the probability-averaged ensemble
    variants = [(os.path.basename(r.rstrip("/")), l[3]) for r, l in zip(args.runs, loaded)]
    deep = np.mean([l[3] for l in loaded], axis=0)
    if len(loaded) > 1:
        variants.append(("ENSEMBLE(" + "+".join(v[0] for v in variants) + ")", deep))
    if args.clinical_csv:
        pc = clinical_probs(args.clinical_csv, args.config, ids, fold, C)
        variants.append(("CLINICAL-only(logreg)", pc))
        a = blend_weight(deep, pc, y, fold, C)
        print(f"\nblend weight found out-of-fold: {a:.2f}*deep + {1-a:.2f}*clinical")
        variants.append((f"DEEP+CLINICAL(a={a:.2f})", a * deep + (1 - a) * pc))

    out_rows = []
    for vname, p in variants:
        print(f"\n{'='*72}\n{vname}   (n={len(y)}, {C} classes, out-of-fold)\n{'='*72}")
        for rule in args.rules:
            pred = decide(fold, y, p, C, rule)
            m = cls_metrics(y, pred, p, C)
            print(f"\n-- {rule} --")
            print(f"  accuracy {m['accuracy']:.4f} | balanced {m['balanced_accuracy']:.4f} | "
                  f"macroF1 {m['macro_f1']:.4f} | kappa {m['kappa']:.4f} | AUC {m['auc']:.4f}")
            print("  sens: " + "  ".join(f"{n}={s:.3f}" for n, s in zip(names, m["per_sens"])))
            out_rows.append({"variant": vname, "rule": rule,
                             **{k: round(m[k], 4) for k in
                                ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "auc")},
                             **{f"sens_{n}": round(s, 3) for n, s in zip(names, m["per_sens"])}})
        print()
        print_per_class(cls_metrics(y, decide(fold, y, p, C, args.rules[-1]), p, C), names)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
        print(f"\nsummary -> {args.out}")
    print("\nNOTE: AUC is identical across rules -- a decision rule reorders nothing, it only "
          "moves the operating point. Read accuracy / balanced accuracy / kappa instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
