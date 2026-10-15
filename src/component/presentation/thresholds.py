"""Exact validation-only operating-point selection, including its false positives."""
import numpy as np
import pandas as pd


def threshold_curve(validation):
    """O(n log n) sweep of all distinct >= decision sets. No test data argument.

    Include the no-alert operating point. Equal-score windows move together;
    a boundary between tied scores cannot be used to fabricate precision.
    """
    y = validation.label.to_numpy(dtype=np.int64)
    p = validation.seizure_probability.to_numpy(dtype=np.float64)
    if (len(y) == 0 or set(np.unique(y)) != {0, 1} or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any()):
        raise ValueError("Threshold selection requires finite scores and both validation classes.")
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    # Model checkpoints require thresholds strictly inside (0,1).
    valid = (p[ends] > 0) & (p[ends] < 1)
    table = pd.DataFrame({"threshold": p[ends][valid], "true_positives": tp[valid], "false_positives": fp[valid]})
    if p[0] < 1:
        upper = min(np.nextafter(p[0], 1.), np.nextafter(1., 0.))
        if upper > p[0]:
            table.loc[len(table)] = [upper, 0, 0]
    positives, negatives = int(y.sum()), int((y == 0).sum())
    table["false_negatives"] = positives - table.true_positives
    table["true_negatives"] = negatives - table.false_positives
    table["precision"] = table.true_positives / (table.true_positives + table.false_positives).clip(lower=1)
    table["recall"] = table.true_positives / positives
    table["f1"] = 2 * table.true_positives / (2 * table.true_positives + table.false_positives + table.false_negatives)
    table["false_positive_rate"] = table.false_positives / negatives
    table["specificity"] = 1 - table.false_positive_rate
    return table.sort_values("threshold", ascending=False).reset_index(drop=True)


def select_threshold(validation, max_fpr=.05):
    """Maximize validation F1 subject to a prespecified validation-window FPR cap.

    Ties favor fewer false positives, then a higher threshold. The cap is not a
    promise about unseen patients or event-level false alarms/hour.
    """
    if not np.isfinite(max_fpr) or not 0 <= max_fpr <= 1:
        raise ValueError("Validation FPR cap must be between zero and one.")
    curve = threshold_curve(validation)
    eligible = curve[curve.false_positive_rate <= max_fpr + 1e-12]
    if eligible.empty:
        raise ValueError("No feasible validation threshold within the requested false-positive cap.")
    chosen = eligible.sort_values(["f1", "false_positives", "threshold"], ascending=[False, True, False]).iloc[0]
    selection = {"method": "exact validation F1; ties: fewer FP, higher threshold", "validation_fpr_cap": float(max_fpr),
                 "threshold": float(chosen.threshold), "validation_f1": float(chosen.f1),
                 "validation_recall": float(chosen.recall), "validation_precision": float(chosen.precision),
                 "validation_false_positives": int(chosen.false_positives),
                 "validation_false_positive_rate": float(chosen.false_positive_rate)}
    return selection, curve
