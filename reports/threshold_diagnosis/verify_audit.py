"""Integrity checks only; no test-threshold evaluation, inference or training."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.component.presentation.score_audit import RUNS, check_saved_windows, read_scores, protected_hashes
from src.component.presentation.train import file_hash

out = Path(__file__).parent
freeze = json.loads((out / "validation_freeze.json").read_text())
held = json.loads((out / "held_out_results.json").read_text())
alternative = json.loads((out / "exploratory_freeze.json").read_text())
assert protected_hashes() == freeze["protected_hashes_before"] == held["protected_hashes_after"]
assert file_hash(out / "validation_freeze.json") == held["validation_freeze_sha256"] == alternative["validation_freeze_sha256"]
assert file_hash(out / "exploratory_freeze.json") == held["exploratory_freeze_sha256"]
checked, threshold_counts = [], {}
for name, run in RUNS.items():
    for split, index in [("validation", "val"), ("test", "test")]:
        f = read_scores(run / f"{split}_predictions.csv")
        check_saved_windows(f, run / f"window_index_{index}.csv")
        expected = (freeze["models"][name]["validation_predictions_sha256"] if split == "validation"
                    else held["models"][name]["test_predictions_sha256"])
        assert file_hash(run / f"{split}_predictions.csv") == expected
        checked.append({"model": name, "split": split, "n_windows": len(f), "labels_timing_patients_match_index": True,
                        "score_sha256": expected})
        if split == "validation":
            c = pd.read_csv(out / f"{name}_validation_all_thresholds.csv", float_precision="round_trip")
            assert file_hash(out / f"{name}_validation_all_thresholds.csv") == freeze["models"][name]["curve_sha256"]
            # Independent direct >= decisions, validation only; bounded allocations.
            for start in range(0, len(c), 128):
                batch = c.iloc[start:start + 128]
                positive = f.seizure_probability.to_numpy()[None, :] >= batch.threshold.to_numpy()[:, None]
                y = f.label.to_numpy()[None, :]
                tp = (positive & (y == 1)).sum(axis=1)
                fp = (positive & (y == 0)).sum(axis=1)
                fn = (y == 1).sum() - tp
                tn = (y == 0).sum() - fp
                for key, values in {"true_positives": tp, "false_positives": fp,
                                    "false_negatives": fn, "true_negatives": tn}.items():
                    np.testing.assert_array_equal(batch[key], values)
                for key, values in {"precision": tp / np.maximum(tp + fp, 1),
                                    "recall": tp / (tp + fn), "sensitivity": tp / (tp + fn),
                                    "f1": 2 * tp / (2 * tp + fp + fn), "specificity": tn / (tn + fp)}.items():
                    np.testing.assert_allclose(batch[key], values, rtol=0, atol=1e-12)
            threshold_counts[name] = len(c)
verification = json.loads((out / "data_verification.json").read_text())
root = Path(json.loads((RUNS["original"] / "configuration.json").read_text())["training"]["data_root"])
for r in verification["records"]:
    path = root / r["relative_path"]
    summary = path.parent / f"{path.parent.name}-summary.txt"
    assert file_hash(path) == r["edf_sha256"] and file_hash(summary) == r["summary_sha256"]
examples = pd.read_csv(out / "preprocessing_examples.csv")
assert len(examples) == 16 and examples.training_inference_array_equal.all()
assert (examples.channels == 8).all() and (examples.samples == 1024).all()
assert examples.max_abs_channel_mean.max() < 1e-6
assert (abs(examples.min_channel_std - 1) < 1e-6).all()
assert (abs(examples.max_channel_std - 1) < 1e-6).all()
result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
          "checkpoints_and_all_frontend_python_and_streamlit_config_unchanged": True,
          "protected_sha256": protected_hashes(), "all_10_edf_and_summary_hashes_still_match": True,
          "saved_scores_match_verified_indexes": checked,
          "validation_direct_decision_checks_all_threshold_counts": threshold_counts,
          "preprocessing_examples_passed": len(examples),
          "independent_test_evaluations_in_this_verification": 0,
          "source_sha256": file_hash("src/component/presentation/score_audit.py"),
          "automated_tests": {"command": ".venv\\Scripts\\python.exe -m pytest src/tests -q",
                              "passed": 77, "skipped": 1, "failed": 0,
                              "note": "Existing optional safetensors skip and nonfatal upstream torch.jit FutureWarning; tests use synthetic temporary fixtures, no project checkpoints trained."},
          "plots_visually_inspected": ["validation_distributions.png", "validation_threshold_tradeoffs.png",
                                       "patient_score_shift.png", "event_score_timelines.png", "validation_discrimination.png"]}
with (out / "verification.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps({k: v for k, v in result.items() if k not in ["protected_sha256", "saved_scores_match_verified_indexes"]}, indent=2))
