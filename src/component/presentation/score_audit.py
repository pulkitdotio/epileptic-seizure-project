"""Read-only saved-score diagnosis. Freeze validation decisions before test access.

No training, inference, downloads, checkpoint writes, or dashboard imports.
The existing test patients have already been examined: this is exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import load_baseline
from .pipeline import annotations_for, open_edf, read_window, PreprocessConfig
from .thresholds import threshold_curve
from .train import file_hash, metrics, patient_group, LazyWindows

RUNS = {"original": Path("results/presentation_baseline"),
        "balanced_bce": Path("results/presentation_balanced_bce")}
CAPS = [0, .001, .005, .01, .02, .05, .1, 1]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def protected_hashes():
    paths = [run / "baseline.pt" for run in RUNS.values()]
    paths += sorted(Path("demo").glob("*.py"))
    paths += sorted(Path(".streamlit").glob("*"))
    return {path.as_posix(): file_hash(path) for path in paths if path.is_file()}


def read_scores(path):
    # Preserve float64 values exactly at >= boundaries, including tied scores.
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {"path", "patient", "start_sec", "end_sec", "label", "seizure_probability"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing saved-score columns or empty predictions.")
    numeric = frame[["start_sec", "end_sec", "seizure_probability", "label"]].to_numpy()
    if (not np.isfinite(numeric).all() or not set(frame.label.unique()) <= {0, 1}
            or not frame.seizure_probability.between(0, 1).all()
            or (frame.start_sec < 0).any() or (frame.end_sec <= frame.start_sec).any()
            or frame.duplicated(["path", "start_sec", "end_sec"]).any()):
        raise ValueError("Invalid saved scores, labels, times, or duplicate windows.")
    for _, rows in frame.groupby("path"):
        rows = rows.sort_values("start_sec")
        if (rows.start_sec.to_numpy()[1:] < rows.end_sec.to_numpy()[:-1] - 1e-9).any():
            raise ValueError("Overlapping windows cannot define this audit's exposure time.")
    return frame


def distribution(frame, model, split):
    rows = []
    for (patient, path, label), group in frame.groupby(["patient", "path", "label"]):
        p = group.seizure_probability
        rows.append({"model": model, "split": split, "patient": patient,
                     "recording": Path(path).name, "label": int(label), "windows": len(p),
                     "mean": p.mean(), "std": p.std(ddof=0), "min": p.min(), "max": p.max(),
                     **{f"q{q:02d}": p.quantile(q / 100) for q in [5, 10, 25, 50, 75, 90, 95, 99]}})
    return pd.DataFrame(rows)


def check_saved_windows(frame, index_path):
    """Scores must align one-to-one with the timing, patient and label index."""
    keys = ["path", "patient", "start_sec", "end_sec", "label"]
    expected = pd.read_csv(index_path, float_precision="round_trip")
    actual = frame[keys].sort_values(keys).reset_index(drop=True)
    expected = expected[keys].sort_values(keys).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Saved scores differ from the verified window index.")


def alarm_statistics(frame, threshold, intervals_by_path):
    """Offline alarm episodes: adjacent positive windows merge within each EDF.

    A purely false episode has zero overlap with ANY annotated seizure. An
    episode touching a seizure is not called purely false; its background
    duration is still counted. No smoothing, refractory period, or cross-file
    merging. Denominator = covered seconds minus exact annotated overlap.
    This is retrospective event scoring, not causal or clinical validation.
    """
    episodes = false_episodes = detected_events = total_events = 0
    covered = seizure_seconds = background_alert_seconds = 0.
    labelled_background_seconds = 0.
    for path, rows in frame.groupby("path", sort=False):
        rows = rows.sort_values("start_sec")
        starts, ends = rows.start_sec.to_numpy(), rows.end_sec.to_numpy()
        intervals = []
        for s, e in sorted(intervals_by_path[path]):
            if intervals and s <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], e)
            else:
                intervals.append([s, e])
        overlap = np.zeros(len(rows))
        for s, e in intervals:
            overlap += np.maximum(0, np.minimum(ends, e) - np.maximum(starts, s))
        positive = rows.seizure_probability.to_numpy() >= threshold
        widths = ends - starts
        covered += widths.sum()
        seizure_seconds += overlap.sum()
        labelled_background_seconds += widths[rows.label.to_numpy() == 0].sum()
        background_alert_seconds += (widths - overlap)[positive].sum()
        positions = np.flatnonzero(positive)
        alarm_intervals = []
        for i in positions:
            if alarm_intervals and abs(starts[i] - alarm_intervals[-1][1]) < 1e-9:
                alarm_intervals[-1][1] = ends[i]
            else:
                alarm_intervals.append([starts[i], ends[i]])
        episodes += len(alarm_intervals)
        false_episodes += sum(not any(a < e and b > s for s, e in intervals)
                              for a, b in alarm_intervals)
        observed_events = [(s, e) for s, e in intervals
                           if np.any((starts < e) & (ends > s))]
        total_events += len(observed_events)
        detected_events += sum(any(a < e and b > s for a, b in alarm_intervals)
                               for s, e in observed_events)
    background_hours = (covered - seizure_seconds) / 3600
    if background_hours <= 0 or labelled_background_seconds <= 0:
        raise ValueError("No background exposure to estimate false alarms/hour.")
    fp = int(((frame.seizure_probability >= threshold) & (frame.label == 0)).sum())
    return {"covered_hours": float(covered / 3600), "nonseizure_hours": float(background_hours),
            "labelled_background_hours": float(labelled_background_seconds / 3600),
            "false_positive_windows_per_labelled_background_hour": fp / (labelled_background_seconds / 3600),
            "alarm_episodes": episodes, "pure_false_alarm_episodes": false_episodes,
            "false_alarm_episodes_per_nonseizure_hour": false_episodes / background_hours,
            "background_alert_seconds": float(background_alert_seconds),
            "background_alert_seconds_per_nonseizure_hour": float(background_alert_seconds / background_hours),
            "events_detected": detected_events, "events_total": total_events,
            "event_sensitivity": detected_events / total_events if total_events else None}


def choose(curve, objective="f1", cap=.05, max_episodes_per_hour=None):
    """Validation-only policy; caller supplies no test scores.

    The secondary diagnostic policy maximizes recall with <=1% window FPR and
    <=1 pure false episode/nonseizure hour. These are declared engineering
    budgets for this audit, not clinically established acceptable rates.
    """
    feasible = curve[curve.false_positive_rate <= cap + 1e-12]
    if max_episodes_per_hour is not None:
        feasible = feasible[feasible.false_alarm_episodes_per_nonseizure_hour <= max_episodes_per_hour + 1e-12]
    keys = [objective] + (["f1"] if objective != "f1" else []) + ["false_positives", "threshold"]
    ascending = [False] * (len(keys) - 2) + [True, False]
    if feasible.empty:
        raise ValueError("No feasible validation operating point.")
    return feasible.sort_values(keys, ascending=ascending).iloc[0].to_dict()


def verify_data(bundles):
    """Verify both checkpoint contracts against unchanged local EDFs and indexes."""
    first, second = bundles.values()
    for key in ("channels", "preprocessing", "window_sec"):
        if first.payload[key] != second.payload[key]:
            raise ValueError(f"Models have different {key}.")
    records = first.payload["training"]["records"]
    if records != second.payload["training"]["records"]:
        raise ValueError("Models were not trained on the same record splits/content.")
    groups = {name: set() for name in ("train", "val", "test")}
    seen = set()
    intervals, examples, checks = {}, [], []
    cfg = first.preprocessing
    root = Path(first.payload["training"]["data_root"])
    for record in records:
        path = root / record["relative_path"]
        summary = path.parent / f"{path.parent.name}-summary.txt"
        if (file_hash(path) != record["edf_sha256"]
                or file_hash(summary) != record["summary_sha256"]):
            raise ValueError("Dataset or summary changed since training.")
        if record["edf_sha256"] in seen:
            raise ValueError("Duplicate recording content.")
        seen.add(record["edf_sha256"])
        patient = patient_group(path.parent.name)
        if patient != record["patient"]:
            raise ValueError("Patient group differs from checkpoint.")
        groups[record["split"]].add(patient)
        raw = open_edf(path)
        try:
            duration = raw.n_times / raw.info["sfreq"]
            ann = annotations_for(summary.read_text(encoding="utf-8-sig"), path.name, duration)
            if ann is None:
                raise ValueError("Unknown annotations.")
            intervals[str(path)] = ann
            if raw.info["sfreq"] != cfg.sfreq or any(ch not in raw.ch_names for ch in first.channels):
                raise ValueError("Sampling frequency or channels incompatible.")
            for label, start in [(0, 0.)] + ([(1, float(np.ceil(ann[0][0] / first.window_sec) * first.window_sec))] if ann else []):
                frame = pd.DataFrame({"path": [str(path)], "start_sec": [start], "label": [label]})
                dataset = LazyWindows(frame, first.channels, cfg, first.window_sec)
                try:
                    tensor, _ = dataset[0]
                finally:
                    dataset.close()
                _, array = read_window(raw, first.channels, start, first.window_sec, cfg)
                if not np.array_equal(tensor.numpy(), array) or not np.isfinite(array).all():
                    raise ValueError("Training/inference preprocessing mismatch.")
                _, unscaled = read_window(raw, first.channels, start, first.window_sec,
                                          PreprocessConfig(**{**cfg.to_dict(), "normalization": "none"}))
                examples.append({"recording": path.name, "patient": patient, "split": record["split"],
                                 "example_label": label, "start_sec": start, "channels": array.shape[0],
                                 "samples": array.shape[1], "training_inference_array_equal": True,
                                 "max_abs_channel_mean": float(abs(array.mean(axis=1)).max()),
                                 "min_channel_std": float(array.std(axis=1).min()),
                                 "max_channel_std": float(array.std(axis=1).max()),
                                 "filtered_rms_microvolt": float(np.sqrt(np.mean(unscaled ** 2)) * 1e6)})
            # Saved indexes are metadata, not test probabilities; validate timing and labels.
            for name, run in RUNS.items():
                idx = pd.read_csv(run / f"window_index_{record['split']}.csv")
                idx = idx[idx.path == str(path)]
                from .pipeline import window_index
                expected = window_index(raw.n_times, cfg.sfreq, first.window_sec, intervals=ann)
                if not np.array_equal(idx[["start_sec", "end_sec", "label"]].to_numpy(),
                                      expected[["start_sec", "end_sec", "label"]].to_numpy()):
                    raise ValueError(f"{name}: window timing or labels changed.")
            checks.append({**record, "sampling_frequency": float(raw.info["sfreq"]),
                           "duration_sec": duration, "events": ann, "hashes_verified": True,
                           "window_indexes_verified": True})
        finally:
            raw.close()
    if any(groups[a] & groups[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise ValueError("Patient leakage across splits.")
    return {"patient_groups": {k: sorted(v) for k, v in groups.items()},
            "matching_model_input_contract": {"channels": first.channels, "window_sec": first.window_sec,
                                              "preprocessing": cfg.to_dict()},
            "records": checks}, intervals, pd.DataFrame(examples)


def validation_phase(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new audit output directory; existing reports are preserved.")
    before = protected_hashes()
    bundles = {name: load_baseline(run / "baseline.pt") for name, run in RUNS.items()}
    verification, intervals, examples = verify_data(bundles)
    output.mkdir(parents=True)
    examples.to_csv(output / "preprocessing_examples.csv", index=False)
    write_json(output / "data_verification.json", verification)
    frozen = {"mode": "validation-only decisions frozen before test probability access",
              "test_status": "Previously examined test patients: exploratory, not untouched confirmatory data",
              "alarm_policy": alarm_statistics.__doc__, "protected_hashes_before": before,
              "data_verification_sha256": file_hash(output / "data_verification.json"), "models": {}}
    distributions, candidates = [], []
    for name, run in RUNS.items():
        validation = read_scores(run / "validation_predictions.csv")
        check_saved_windows(validation, run / "window_index_val.csv")
        curves = threshold_curve(validation)
        curves["sensitivity"] = curves.recall
        stats = [alarm_statistics(validation, t, intervals) for t in curves.threshold]
        curves = pd.concat([curves.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
        curves.to_csv(output / f"{name}_validation_all_thresholds.csv", index=False)
        selected = {"saved": {"threshold": bundles[name].threshold},
                    "f1_under_5pct_fpr": choose(curves, "f1", .05),
                    "recall_guarded": choose(curves, "recall", .01, 1.)}
        for cap in CAPS:
            for objective in ("f1", "recall"):
                row = choose(curves, objective, cap)
                candidates.append({"model": name, "policy": f"maximum_{objective}_cap_{cap:g}", "cap": cap, **row})
        for policy, row in selected.items():
            t = row["threshold"]
            candidates.append({"model": name, "policy": policy, "cap": .01 if policy == "recall_guarded" else None,
                               **metrics(validation, t), **alarm_statistics(validation, t, intervals)})
        frozen["models"][name] = {
            "validation_predictions_sha256": file_hash(run / "validation_predictions.csv"),
            "curve_sha256": file_hash(output / f"{name}_validation_all_thresholds.csv"),
            "policies": {key: value["threshold"] for key, value in selected.items()},
            "held_out_evaluation_policy": "recall_guarded",
            "held_out_evaluation_policy_definition": "maximum validation recall with window FPR <=1% and pure false episodes/nonseizure hour <=1; ties highest F1, fewer FP, higher threshold",
            "validation_saved_metrics": metrics(validation, bundles[name].threshold)}
        distributions.append(distribution(validation, name, "validation"))
    pd.concat(distributions).to_csv(output / "validation_distributions.csv", index=False)
    pd.DataFrame(candidates).to_csv(output / "validation_candidate_thresholds.csv", index=False)
    if protected_hashes() != before:
        raise ValueError("Protected model/frontend files changed during audit.")
    write_json(output / "validation_freeze.json", frozen)
    print(pd.DataFrame(candidates).query("policy in ['saved','f1_under_5pct_fpr','recall_guarded']")
          [["model", "policy", "threshold", "precision", "recall", "f1", "false_positives",
            "false_alarm_episodes_per_nonseizure_hour"]].to_string(index=False))


def freeze_alternatives(output):
    """Freeze finite validation-derived cost diagnostics, without reading test CSVs."""
    output = Path(output)
    if (output / "held_out_results.json").exists():
        raise ValueError("Test evaluation already performed; cannot add thresholds.")
    frozen = json.loads((output / "validation_freeze.json").read_text())
    specs = {}
    for name, run in RUNS.items():
        spec = frozen["models"][name]
        curve_path = output / f"{name}_validation_all_thresholds.csv"
        if file_hash(curve_path) != spec["curve_sha256"]:
            raise ValueError("Frozen validation curve changed.")
        curve = pd.read_csv(curve_path, float_precision="round_trip")
        specs[name] = {"recall_cap_half_percent": choose(curve, "recall", .005)["threshold"],
                       "recall_cap_5percent": choose(curve, "recall", .05)["threshold"]}
    write_json(output / "exploratory_freeze.json", {
        "validation_freeze_sha256": file_hash(output / "validation_freeze.json"),
        "intent": "Finite validation-only sensitivity/cost diagnostics; not recommendations or deployment changes",
        "rules": "Maximum validation recall under 0.5% or 5% window FPR; ties highest F1, fewer FP, higher threshold. No test input.",
        "models": specs})
    print(json.dumps(specs, indent=2))


def test_phase(output):
    output = Path(output)
    if (output / "held_out_results.json").exists():
        raise ValueError("Frozen held-out evaluation already performed; do not rerun or retune.")
    frozen = json.loads((output / "validation_freeze.json").read_text())
    alternatives = json.loads((output / "exploratory_freeze.json").read_text())
    if alternatives["validation_freeze_sha256"] != file_hash(output / "validation_freeze.json"):
        raise ValueError("Frozen decisions changed.")
    if protected_hashes() != frozen["protected_hashes_before"]:
        raise ValueError("Protected files changed after freezing thresholds.")
    verification = json.loads((output / "data_verification.json").read_text())
    if file_hash(output / "data_verification.json") != frozen["data_verification_sha256"]:
        raise ValueError("Verification changed.")
    root = Path(json.loads((RUNS["original"] / "configuration.json").read_text())["training"]["data_root"])
    intervals = {str(root / r["relative_path"]): r["events"] for r in verification["records"]}
    results, distributions, previous = {}, [], None
    for name, run in RUNS.items():
        spec = frozen["models"][name]
        if (file_hash(run / "validation_predictions.csv") != spec["validation_predictions_sha256"]
                or file_hash(output / f"{name}_validation_all_thresholds.csv") != spec["curve_sha256"]):
            raise ValueError("Validation inputs changed after freezing.")
        test = read_scores(run / "test_predictions.csv")
        check_saved_windows(test, run / "window_index_test.csv")
        keys = ["path", "patient", "start_sec", "end_sec", "label"]
        if previous is not None and not test[keys].equals(previous[keys]):
            raise ValueError("Models have different held-out windows.")
        previous = test
        distributions.append(distribution(test, name, "test_descriptive_only"))
        # Finite frozen policies only. Repeated thresholds are evaluated once.
        thresholds = {key: spec["policies"][key] for key in ["saved", spec["held_out_evaluation_policy"]]}
        thresholds.update(alternatives["models"][name])
        measured = {}
        cache = {}
        for policy, t in thresholds.items():
            if t not in cache:
                cache[t] = {**metrics(test, t), **alarm_statistics(test, t, intervals)}
            measured[policy] = cache[t]
        results[name] = {"test_predictions_sha256": file_hash(run / "test_predictions.csv"),
                         "policies": measured,
                         "unique_thresholds_evaluated_once": sorted(cache),
                         "test_seizure_score_max": float(test[test.label == 1].seizure_probability.max()),
                         "test_background_score_max": float(test[test.label == 0].seizure_probability.max())}
    pd.concat(distributions).to_csv(output / "test_descriptive_distributions.csv", index=False)
    if protected_hashes() != frozen["protected_hashes_before"]:
        raise ValueError("Protected model/frontend files changed during audit.")
    write_json(output / "held_out_results.json", {
        "status": frozen["test_status"], "validation_freeze_sha256": file_hash(output / "validation_freeze.json"),
        "exploratory_freeze_sha256": file_hash(output / "exploratory_freeze.json"),
        "no_test_threshold_sweep": True, "models": results,
        "protected_hashes_after": protected_hashes()})
    print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["validation", "freeze-alternatives", "test"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        {"validation": validation_phase, "freeze-alternatives": freeze_alternatives,
         "test": test_phase}[args.phase](args.output)
    except (ValueError, OSError) as exc:
        p.exit(2, f"Audit unavailable: {exc}\n")


if __name__ == "__main__":
    main()
