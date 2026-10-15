"""Audit saved baseline scores without training or changing the original checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import load_baseline
from .thresholds import select_threshold
from .train import metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_distribution(frame):
    rows = []
    for label, values in frame.groupby("label"):
        p = values.seizure_probability
        rows.append({"label": int(label), "windows": len(values), "mean": float(p.mean()),
                     "median": float(p.median()), "min": float(p.min()), "max": float(p.max()),
                     "p10": float(p.quantile(.1)), "p90": float(p.quantile(.9))})
    return rows


def audit(baseline, output, max_fpr=.05):
    baseline, output = Path(baseline), Path(output)
    if output.exists():
        raise ValueError("Audit directory already exists; choose a new path.")
    checkpoint = baseline / "baseline.pt"
    bundle = load_baseline(checkpoint)
    validation = pd.read_csv(baseline / "validation_predictions.csv")
    # This operating point is frozen BEFORE any test scores are read.
    selection, curve = select_threshold(validation, max_fpr)
    test = pd.read_csv(baseline / "test_predictions.csv")
    before = metrics(test, bundle.threshold)
    after = metrics(test, selection["threshold"])
    counts = np.asarray(bundle.payload["training"]["class_counts_train"])
    result = {"mode": "saved-score audit; no new training", "baseline_sha256": digest(checkpoint),
              "validation_predictions_sha256": digest(baseline / "validation_predictions.csv"),
              "test_predictions_sha256": digest(baseline / "test_predictions.csv"),
              "threshold_selection": selection, "original_held_out": before,
              "threshold_only_held_out": after,
              "score_distributions": {"validation": score_distribution(validation), "test": score_distribution(test)},
              "training_class_counts": counts.tolist(), "original_ce_weights": (counts.sum() / (2 * counts)).tolist(),
              "expected_positive_free_random_batch_fraction": float((counts[0] / counts.sum()) ** bundle.payload["training"]["batch_size"]),
              "limitations": ["Threshold-only audit reuses a previously examined test set; descriptive, not a fresh confirmatory test.",
                              "One validation patient/event and one test patient/event; windows are temporally correlated.",
                              "Weighted CE uses a batch-dependent weight denominator; all-negative batches still have full mean loss.",
                              "Per-window z-score is consistent in train/inference but removes absolute amplitude information."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    curve.to_csv(output / "baseline_validation_threshold_curve.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/presentation_baseline"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-val-fpr", type=float, default=.05)
    args = parser.parse_args()
    try:
        audit(args.baseline, args.output, args.max_val_fpr)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Audit unavailable: {exc}\n")


if __name__ == "__main__":
    main()
