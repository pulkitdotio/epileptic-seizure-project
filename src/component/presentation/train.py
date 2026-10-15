"""Reproducible, lazy-window CPU trainer. Never downloads data or overwrites runs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

from src.component.models.utils.metrics import compute_f1_score, compute_sensitivity, compute_specificity
from .pipeline import open_edf, annotations_for, window_index, read_window, PreprocessConfig
from .model import BaselineCNN1D, DEFAULT_CHANNELS, FORMAT, ModelBundle, predict_windows
from .download_subset import load_manifest
from .objectives import BalancedBatchSampler, make_loss
from .thresholds import select_threshold


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_group(case):
    # CHB-MIT chb21 is the same patient as chb01 (PhysioNet documentation).
    return "chb01" if case == "chb21" else case


def discover(root, cfg, channels, window_sec, manifest=None):
    root = Path(root).resolve()
    records, frames, fingerprints = [], [], set()
    if manifest:
        selection = load_manifest(manifest)
        selected = selection["recordings"]
        paths = [root / record["path"] for record in selected]
        missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"Selected subset incomplete: {len(missing)} EDFs missing: {', '.join(missing)}. "
                             "Approve and download the manifest subset before training.")
    else:
        paths = sorted(root.glob("chb*/*.edf"))
    for path in paths:
        case = path.parent.name
        if not re.fullmatch(r"chb\d{2}", case):
            continue
        summary = path.parent / f"{case}-summary.txt"
        if not summary.is_file():
            raise ValueError(f"Missing annotation summary: {summary}. Unknown labels cannot be used for training.")
        summary_hash = file_hash(summary)
        if manifest:
            expected_summary = selection.get("summaries", {}).get(f"{case}/{case}-summary.txt")
            if not expected_summary or summary_hash != expected_summary["sha256"]:
                raise ValueError(f"{summary.name}: selected annotation checksum mismatch.")
        raw = open_edf(path)
        try:
            if abs(raw.info["sfreq"] - cfg.sfreq) > 1e-6 or any(ch not in raw.ch_names for ch in channels):
                raise ValueError(f"{path.name}: channel order or sampling frequency incompatible with training config.")
            intervals = annotations_for(summary.read_text(encoding="utf-8-sig"), path.name, raw.n_times / cfg.sfreq)
            if intervals is None:
                raise ValueError(f"{path.name}: no matching summary entry; cannot assume non-seizure.")
            frame = window_index(raw.n_times, cfg.sfreq, window_sec, intervals=intervals)
        finally:
            raw.close()
        if frame.empty:
            raise ValueError(f"{path.name}: recording shorter than one training window.")
        fingerprint = file_hash(path)
        if manifest:
            expected = next(r for r in selected if r["path"] == path.relative_to(root).as_posix())
            if fingerprint != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
                raise ValueError(f"{path.name}: selected dataset size/checksum mismatch.")
        if fingerprint in fingerprints:
            raise ValueError("Duplicate EDF content found. Remove duplicates from the selected training subset.")
        fingerprints.add(fingerprint)
        frame["path"] = str(path)
        frame["patient"] = patient_group(case)
        frames.append(frame)
        records.append({"relative_path": path.relative_to(root).as_posix(), "patient": patient_group(case),
                        "edf_sha256": fingerprint, "summary_sha256": summary_hash})
    if not frames:
        raise ValueError("No recordings found. Expected --data-root/chbNN/*.edf with matching summaries.")
    return pd.concat(frames, ignore_index=True), records


def split_windows(frame, *, split="patient", seed=42, fit_only=False, split_file=None, root=None):
    """Split WHOLE patient groups or recordings before any negative subsampling."""
    if fit_only:
        if split_file:
            raise ValueError("--fit-only cannot be combined with a split file.")
        return {"train": frame.copy(), "val": frame.iloc[:0].copy(), "test": frame.iloc[:0].copy()}
    column = "patient" if split == "patient" else "path"
    groups = sorted(frame[column].unique())
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 independent {split} groups for train/validation/test; found {len(groups)}. "
                         "No model was trained. Use more selected recordings/patients, or explicitly use --fit-only for an unevaluated training walkthrough.")
    if split_file:
        mapping = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if set(mapping) != {"train", "val", "test"}:
            raise ValueError("Split JSON requires train, val and test lists.")
        if split == "recording":
            mapping = {key: [str((Path(root) / value).resolve()) for value in values] for key, values in mapping.items()}
        else:
            mapping = {key: [patient_group(value) for value in values] for key, values in mapping.items()}
        assigned = [group for values in mapping.values() for group in values]
        if len(set(assigned)) != len(assigned) or set(assigned) != set(groups):
            raise ValueError("Split groups must cover the selected dataset exactly once. chb01 and chb21 are one patient.")
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n_val = max(1, int(len(groups) * .15))
        n_test = max(1, int(len(groups) * .15))
        n_train = len(groups) - n_val - n_test
        mapping = {"train": groups[:n_train], "val": groups[n_train:n_train + n_val], "test": groups[n_train + n_val:]}
    result = {key: frame[frame[column].isin(values)].copy().reset_index(drop=True) for key, values in mapping.items()}
    for key, rows in result.items():
        if set(rows.label.unique()) != {0, 1}:
            raise ValueError(f"{key} split needs genuine seizure and non-seizure windows. "
                             "Select more data or supply a deliberate --split-file; no random window split is used.")
    return result


def cap_training_background(frame, cap, seed):
    if cap <= 0:
        return frame.reset_index(drop=True)
    chunks = []
    for _, rows in frame.groupby("path"):
        negative = rows[rows.label == 0]
        if len(negative) > cap:
            negative = negative.sample(cap, random_state=seed)
        chunks.extend([rows[rows.label == 1], negative])
    return pd.concat(chunks, ignore_index=True)


class LazyWindows(Dataset):
    """At most two lazy EDF headers retained, signals fetched one window at a time."""
    def __init__(self, frame, channels, cfg, width):
        self.frame = frame.reset_index(drop=True)
        self.channels, self.cfg, self.width = channels, cfg, width
        self.readers = {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        if row.path not in self.readers:
            if len(self.readers) >= 2:
                self.readers.pop(next(iter(self.readers))).close()
            self.readers[row.path] = open_edf(row.path)
        _, array = read_window(self.readers[row.path], self.channels, float(row.start_sec), self.width, self.cfg)
        return torch.from_numpy(array), int(row.label)

    def close(self):
        for raw in self.readers.values():
            raw.close()
        self.readers.clear()


def predict_frame(bundle, frame, batch_size, deadline=None):
    parts = []
    for path, rows in frame.groupby("path", sort=False):
        raw = open_edf(path)
        try:
            result = predict_windows(bundle, raw, rows, batch_size, deadline=deadline)
        finally:
            raw.close()
        result["label"] = rows.label.to_numpy(dtype=np.int64)
        result["path"] = path
        result["patient"] = rows.patient.to_numpy()
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def metrics(results, threshold):
    truth = results.label.to_numpy(dtype=np.int64)
    predicted = (results.seizure_probability.to_numpy() >= threshold).astype(int)
    both_classes = len(np.unique(truth)) == 2
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "f1": compute_f1_score(truth, predicted),
            "sensitivity": compute_sensitivity(truth, predicted) if (truth == 1).any() else None,
            "specificity": compute_specificity(truth, predicted) if (truth == 0).any() else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "true_positives": int(tp), "false_positives": int(fp), "true_negatives": int(tn), "false_negatives": int(fn),
            "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
            "roc_auc": float(roc_auc_score(truth, results.seizure_probability)) if both_classes else None,
            "pr_auc": float(average_precision_score(truth, results.seizure_probability)) if both_classes else None,
            "n_windows": len(truth), "positive_windows": int(truth.sum()), "threshold": threshold}


def probability_loss(results, class_weights):
    """Same weighted cross-entropy as training, from measured softmax outputs."""
    labels = results.label.to_numpy(dtype=np.int64)
    p = results.seizure_probability.to_numpy()
    selected = np.where(labels == 1, p, 1 - p)
    weights = np.asarray(class_weights)[labels]
    return float(np.sum(-np.log(np.clip(selected, 1e-7, 1.)) * weights) / weights.sum())


def resource_preflight(frame, splits, training_frame, channels, cfg, args):
    """Benchmark a single bounded batch on a disposable model; no weights/artifacts saved."""
    indices = next(iter(BalancedBatchSampler(training_frame.label, args.batch_size, 1, args.seed))) if args.sampling == "balanced" else list(range(min(args.batch_size, len(training_frame))))
    dataset = LazyWindows(training_frame.iloc[indices], channels, cfg, args.window_sec)
    try:
        started = time.perf_counter()
        x, y = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)))
        read_seconds = time.perf_counter() - started
        model = BaselineCNN1D(in_channels=len(channels), hidden_dim=32)
        started = time.perf_counter()
        counts = np.bincount(training_frame.label.to_numpy(dtype=np.int64), minlength=2)
        make_loss(args.loss, counts, args.sampling, args.focal_gamma)(model(x), y).backward()
        step_seconds = time.perf_counter() - started
        batches = args.steps_per_epoch if args.sampling == "balanced" else int(np.ceil(len(training_frame) / len(x)))
        # Evaluation is only a rough extrapolation; disk/cache and CPU load vary.
        estimated = args.epochs * (batches + int(np.ceil(len(splits['val']) / len(x)))) * (read_seconds + step_seconds)
        return {"mode": "preflight only; no checkpoint saved", "independent_patients": int(frame.patient.nunique()),
                "split_patients": {k: sorted(rows.patient.unique()) for k, rows in splits.items()},
                "split_windows": {k: len(rows) for k, rows in splits.items()}, "train_windows_used": len(training_frame),
                "batch_shape": list(x.shape), "input_batch_mib": x.numel() * x.element_size() / 1024**2,
                "model_parameters": sum(p.numel() for p in model.parameters()), "bounded_read_seconds": read_seconds,
                "one_training_step_seconds": step_seconds, "rough_run_minutes": estimated / 60,
                "sampling": args.sampling, "loss": args.loss, "batches_per_epoch": batches,
                "limitation": "Single-batch timing estimate, not a runtime guarantee; test inference and file hashing add time."}
    finally:
        dataset.close()


def train(args):
    if args.epochs < 1 or not 1 <= args.batch_size <= 128 or not 1 <= args.threads <= 4 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Use positive epochs/lr, batch size 1–128 and CPU threads 1–4.")
    if not 1 <= args.window_sec <= 30:
        raise ValueError("Window duration must be 1–30 seconds.")
    if not np.isfinite(args.max_minutes) or args.max_minutes <= 0:
        raise ValueError("--max-minutes must be positive.")
    if args.output.exists():
        raise ValueError(f"Run directory already exists: {args.output}. Choose a new --output; existing artifacts are never overwritten.")
    channels = args.channels.split(",")
    if len(set(channels)) != len(channels) or not 1 <= len(channels) <= 64:
        raise ValueError("Use 1–64 unique exact channel names.")
    cfg = PreprocessConfig(args.sfreq, args.low_hz, args.high_hz, "zscore")
    cfg.validate()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    frame, records = discover(args.data_root, cfg, channels, args.window_sec, args.dataset_manifest)
    splits = split_windows(frame, split=args.split, seed=args.seed, fit_only=args.fit_only,
                          split_file=args.split_file, root=args.data_root)
    training_frame = cap_training_background(splits["train"], args.max_background, args.seed)
    counts = np.bincount(training_frame.label.to_numpy(dtype=np.int64), minlength=2)
    if (counts == 0).any():
        raise ValueError("Training requires both seizure and non-seizure windows.")
    criterion = make_loss(args.loss, counts, args.sampling, args.focal_gamma)
    sampler = BalancedBatchSampler(training_frame.label, args.batch_size, args.steps_per_epoch, args.seed) if args.sampling == "balanced" else None
    if args.early_stopping_patience < 0:
        raise ValueError("Early-stopping patience must be nonnegative; zero disables it.")
    if not np.isfinite(args.max_val_fpr) or not 0 <= args.max_val_fpr <= 1:
        raise ValueError("Validation FPR cap must be between zero and one.")
    if args.preflight:
        print(json.dumps(resource_preflight(frame, splits, training_frame, channels, cfg, args), indent=2), flush=True)
        return
    for record in records:
        path = str((args.data_root.resolve() / record["relative_path"]).resolve())
        record["split"] = next(name for name, rows in splits.items() if path in set(rows.path))
    scope = "fit-only" if args.fit_only else ("patient-held-out" if args.split == "patient" else "recording-held-out")
    provenance = {"evaluation_scope": scope, "seed": args.seed, "epochs_requested": args.epochs, "epochs_completed": 0,
                  "created_utc": datetime.now(timezone.utc).isoformat(), "data_root": str(args.data_root.resolve()),
                  "records": records, "split_window_counts": {name: len(rows) for name, rows in splits.items()},
                  "train_windows_used": len(training_frame), "class_counts_train": counts.tolist(),
                  "batch_size": args.batch_size, "learning_rate": args.lr, "threads": args.threads,
                  "sampling": args.sampling, "loss": args.loss, "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
                  "steps_per_epoch": args.steps_per_epoch if sampler is not None else None,
                  "early_stopping_patience": args.early_stopping_patience,
                  "class_counts_by_split": {name: np.bincount(rows.label.to_numpy(dtype=np.int64), minlength=2).tolist() for name, rows in splits.items()},
                  "dataset_manifest": str(args.dataset_manifest) if args.dataset_manifest else None,
                  "label_policy": "complete non-overlapping windows; seizure union overlap >= 50%; partial tail discarded",
                  "threshold_selection": "fixed 0.5 (fit-only)" if args.fit_only else args.threshold_policy,
                  "max_validation_false_positive_rate": args.max_val_fpr if args.threshold_policy == "exact-f1" else None}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in splits.items():
        rows.to_csv(args.output / f"window_index_{name}.csv", index=False)
    training_frame.to_csv(args.output / "training_windows_used.csv", index=False)
    (args.output / "configuration.json").write_text(json.dumps({"channels": channels, "window_sec": args.window_sec,
                                                              "preprocessing": cfg.to_dict(), "training": provenance}, indent=2), encoding="utf-8")
    model = BaselineCNN1D(in_channels=len(channels), hidden_dim=32)
    bundle = ModelBundle(model, channels, args.window_sec, cfg, .5, {})
    dataset = LazyWindows(training_frame, channels, cfg, args.window_sec)
    loader = (DataLoader(dataset, batch_sampler=sampler, num_workers=0) if sampler is not None else
              DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                         generator=torch.Generator().manual_seed(args.seed)))
    weights = torch.tensor(len(training_frame) / (2 * counts), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best_state, best_f1 = [], None, -1.
    stale_epochs = 0
    deadline = time.monotonic() + args.max_minutes * 60
    try:
        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch - 1)
            model.train()
            total, examples, positive_free_batches = 0., 0, 0
            for x, y in loader:
                if time.monotonic() > deadline:
                    raise ValueError("Training exceeded --max-minutes; run stopped with no checkpoint. Choose a new output directory to retry.")
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; no checkpoint saved.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += loss.item() * len(y)
                examples += len(y)
                positive_free_batches += int(not (y == 1).any())
            model.eval()
            entry = {"epoch": epoch, "training_loss": total / examples,
                     "training_examples_drawn": examples, "positive_free_batches": positive_free_batches}
            if not args.fit_only:
                validation = predict_frame(bundle, splits["val"], args.batch_size, deadline)
                entry["validation_loss"] = probability_loss(validation, weights.numpy())
                entry["validation_f1_at_0_5"] = metrics(validation, .5)["f1"]
                if args.threshold_policy == "exact-f1":
                    selection, _ = select_threshold(validation, args.max_val_fpr)
                    entry.update(selection)
                    score = selection["validation_f1"]
                else:
                    score = entry["validation_f1_at_0_5"]
                if score > best_f1:
                    best_f1, best_state = score, deepcopy(model.state_dict())
                    provenance["selected_epoch"] = epoch
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            else:
                best_state = deepcopy(model.state_dict())
                provenance["selected_epoch"] = epoch
            history.append(entry)
            provenance["epochs_completed"] = epoch
            pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
            print(json.dumps(entry), flush=True)
            if not args.fit_only and args.early_stopping_patience and stale_epochs >= args.early_stopping_patience:
                provenance["stopping_reason"] = "validation early stopping"
                break
    finally:
        dataset.close()
    model.load_state_dict(best_state)
    model.eval()
    evaluation = None
    if not args.fit_only:
        validation = predict_frame(bundle, splits["val"], args.batch_size, deadline)
        if args.threshold_policy == "exact-f1":
            selection, curve = select_threshold(validation, args.max_val_fpr)
            bundle.threshold = selection["threshold"]
            provenance["threshold_operating_point"] = selection
            curve.to_csv(args.output / "validation_threshold_curve.csv", index=False)
            (args.output / "threshold_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        else:
            candidates = [.5] + [float(round(x, 2)) for x in np.arange(.1, .91, .05) if abs(x - .5) > .001]
            bundle.threshold = max(candidates, key=lambda value: metrics(validation, value)["f1"])
        validation["prediction"] = (validation.seizure_probability >= bundle.threshold).astype(int)
        validation.to_csv(args.output / "validation_predictions.csv", index=False)
        test_results = predict_frame(bundle, splits["test"], args.batch_size, deadline)
        evaluation = metrics(test_results, bundle.threshold)
        test_results.to_csv(args.output / "test_predictions.csv", index=False)
        (args.output / "evaluation.json").write_text(json.dumps({"scope": scope, **evaluation}, indent=2), encoding="utf-8")
        print("Held-out test:", json.dumps(evaluation), flush=True)
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    payload = {"format": FORMAT, "model_state_dict": model.state_dict(), "channels": channels,
               "window_sec": args.window_sec, "preprocessing": cfg.to_dict(), "threshold": bundle.threshold,
               "training": provenance, "evaluation": evaluation, "history": history}
    (args.output / "configuration.json").write_text(json.dumps({"channels": channels, "window_sec": args.window_sec,
                                                               "preprocessing": cfg.to_dict(), "training": provenance}, indent=2), encoding="utf-8")
    with (args.output / "baseline.pt").open("xb") as handle:
        torch.save(payload, handle)
    print(f"Saved {args.output / 'baseline.pt'} · {scope}. Academic baseline; not clinically validated.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/raw/chbmit"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling", choices=["natural", "balanced"], default="natural")
    p.add_argument("--loss", choices=["weighted-ce", "bce", "weighted-bce", "focal"], default="weighted-ce")
    p.add_argument("--focal-gamma", type=float, default=2.)
    p.add_argument("--steps-per-epoch", type=int, default=60, help="1–500 balanced batches per epoch; minority windows repeat.")
    p.add_argument("--threshold-policy", choices=["legacy-grid", "exact-f1"], default="legacy-grid")
    p.add_argument("--max-val-fpr", type=float, default=.05, help="Prespecified validation-window FPR cap for exact-f1 selection.")
    p.add_argument("--early-stopping-patience", type=int, default=0, help="Validation-only early stopping; 0 disables it.")
    p.add_argument("--window-sec", type=float, default=4.)
    p.add_argument("--sfreq", type=float, default=256.)
    p.add_argument("--low-hz", type=float, default=1.)
    p.add_argument("--high-hz", type=float, default=40.)
    p.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    p.add_argument("--max-background", type=int, default=400, help="Cap training background per EDF; 0 keeps all. Validation/test are never sampled.")
    p.add_argument("--split", choices=["patient", "recording"], default="patient")
    p.add_argument("--split-file", type=Path, help="JSON train/val/test lists of patient groups, or relative EDF paths for recording split.")
    p.add_argument("--dataset-manifest", type=Path, help="Train only on the explicitly selected, checksum-verified EDFs in this manifest.")
    p.add_argument("--preflight", action="store_true", help="Validate splits and benchmark one disposable CPU batch; no run/checkpoint saved.")
    p.add_argument("--max-minutes", type=float, default=15., help="Training loop time limit; stops without a checkpoint if exceeded.")
    p.add_argument("--fit-only", action="store_true", help="Explicit unevaluated exploratory fit, no validation or test scores.")
    return p


def main():
    p = parser()
    try:
        train(p.parse_args())
    except (ValueError, OSError) as exc:
        p.exit(2, f"Training unavailable: {exc}\n")


if __name__ == "__main__":
    main()
