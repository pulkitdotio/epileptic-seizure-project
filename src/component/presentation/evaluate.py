"""Re-evaluate a checkpoint on its ORIGINAL held-out test recordings only."""
import argparse
import json
from pathlib import Path

import torch

from .model import load_baseline
from .pipeline import open_edf, annotations_for, window_index
from .train import file_hash, predict_frame, metrics
import pandas as pd


def evaluate(checkpoint, data_root=None, batch_size=16, threads=2):
    bundle = load_baseline(checkpoint)
    provenance = bundle.payload["training"]
    if provenance["evaluation_scope"] == "fit-only":
        raise ValueError("Fit-only checkpoints have no held-out evaluation data.")
    torch.set_num_threads(threads)
    root = Path(data_root or provenance["data_root"])
    frames = []
    for record in provenance["records"]:
        if record["split"] != "test":
            continue
        relative = Path(record["relative_path"])
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Test path is outside data root.")
        summary = path.parent / f"{path.parent.name}-summary.txt"
        if file_hash(path) != record["edf_sha256"] or file_hash(summary) != record["summary_sha256"]:
            raise ValueError("Evaluation data or annotations changed since training; report not reproduced.")
        raw = open_edf(path)
        try:
            intervals = annotations_for(summary.read_text(encoding="utf-8-sig"), path.name, raw.n_times / raw.info["sfreq"])
            if intervals is None:
                raise ValueError("Missing matching annotations.")
            frame = window_index(raw.n_times, raw.info["sfreq"], bundle.window_sec, intervals=intervals)
        finally:
            raw.close()
        frame["path"], frame["patient"] = str(path), record["patient"]
        frames.append(frame)
    if not frames:
        raise ValueError("Checkpoint has no test recordings.")
    results = predict_frame(bundle, pd.concat(frames, ignore_index=True), batch_size)
    return {"scope": provenance["evaluation_scope"], **metrics(results, bundle.threshold)}, results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, help="Optional relocated data root; content hashes must still match.")
    p.add_argument("--output", type=Path, help="Optional new JSON report; adjacent CSV also saved. Never overwrites.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    try:
        if not 1 <= args.threads <= 4 or not 1 <= args.batch_size <= 128:
            raise ValueError("Use 1–4 CPU threads and batch size 1–128.")
        if args.output and (args.output.exists() or args.output.with_suffix(".csv").exists()):
            raise ValueError("Output already exists; choose a new report path.")
        report, results = evaluate(args.checkpoint, args.data_root, args.batch_size, args.threads)
        print(json.dumps(report, indent=2))
        if args.output:
            if args.output.suffix.lower() != ".json":
                raise ValueError("Evaluation --output must have a .json extension.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            results.to_csv(args.output.with_suffix(".csv"), index=False, mode="x")
    except (ValueError, OSError) as exc:
        p.exit(2, f"Evaluation unavailable: {exc}\n")


if __name__ == "__main__":
    main()
