"""Compare native-volume OOF Dice on exactly the same patients and folds."""
import argparse
import csv
import math
import statistics
from pathlib import Path


def compare(directories):
    results, reference = [], None
    for directory in directories:
        path = Path(directory) / "oof_masks" / "pred_masks.csv"
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assignment = {r["patient_id"]: int(r["fold"]) for r in rows}
        if not rows or len(rows) != len(assignment):
            raise ValueError(f"{path}: empty or duplicate patients")
        if reference is None:
            reference = assignment
        elif reference != assignment:
            raise ValueError(f"{path}: patients or OOF fold assignments differ")
        scores = [float(r["dice"]) for r in rows]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
            raise ValueError(f"{path}: invalid Dice")
        fold_scores = [statistics.mean(float(r["dice"]) for r in rows if int(r["fold"]) == fold)
                       for fold in sorted(set(assignment.values()))]
        results.append({"experiment": str(directory), "patients": len(rows),
                        "patient_mean_dice": statistics.mean(scores),
                        "patient_median_dice": statistics.median(scores),
                        "fold_mean_dice": statistics.mean(fold_scores),
                        "fold_std_dice": statistics.pstdev(fold_scores),
                        "empty_predictions": sum(int(r["empty_pred"]) for r in rows)})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directories", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = compare(args.directories)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['experiment']}: native patient Dice={row['patient_mean_dice']:.4f}")
