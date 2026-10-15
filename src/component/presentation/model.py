"""Allowlisted baseline checkpoint contract and real CPU inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import pickle
import time

import numpy as np
import pandas as pd
import torch

from src.component.models.hugging_face_mamba_moe.architectures.hf_cnn_models import BaselineCNN1D
from .pipeline import PreprocessConfig, read_window

FORMAT = "eeg-presentation-baseline-v1"
DEFAULT_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]


@dataclass
class ModelBundle:
    model: BaselineCNN1D
    channels: list[str]
    window_sec: float
    preprocessing: PreprocessConfig
    threshold: float
    payload: dict


def load_baseline(path):
    """Never import checkpoint-supplied code or fall back to unsafe pickle."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Model not trained: compatible checkpoint does not exist.")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Baseline checkpoint exceeds the 20 MiB limit.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise ValueError("Incompatible checkpoint format; use the presentation baseline trainer.")
        channels = payload["channels"]
        if (not isinstance(channels, list) or not 1 <= len(channels) <= 64 or
                any(not isinstance(ch, str) or not ch for ch in channels) or len(set(channels)) != len(channels)):
            raise ValueError("Checkpoint has invalid channel order.")
        cfg = PreprocessConfig(**payload["preprocessing"])
        cfg.validate()
        width, threshold = float(payload["window_sec"]), float(payload["threshold"])
        if not math.isfinite(width) or not 1 <= width <= 30 or not 32 <= cfg.sfreq <= 1024:
            raise ValueError("Unsupported checkpoint input size.")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Invalid model threshold.")
        training = payload["training"]
        if not isinstance(training, dict) or training.get("epochs_completed", 0) < 1:
            raise ValueError("Checkpoint contains no completed training.")
        if training.get("evaluation_scope") not in {"patient-held-out", "recording-held-out", "fit-only"}:
            raise ValueError("Checkpoint has no recognized evaluation scope.")
        model = BaselineCNN1D(in_channels=len(channels), hidden_dim=32)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Checkpoint contains non-finite model weights.")
        model.eval()
        return ModelBundle(model, channels, width, cfg, threshold, payload)
    except (KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Invalid baseline checkpoint: {exc}") from exc


def check_recording(bundle, raw):
    missing = [ch for ch in bundle.channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Model requires these exact channels: {', '.join(missing)}")
    if abs(raw.info["sfreq"] - bundle.preprocessing.sfreq) > 1e-6:
        raise ValueError(f"Model requires {bundle.preprocessing.sfreq:g} Hz; resample explicitly before inference.")


def predict_windows(bundle, raw, windows, batch_size=16, progress=None, deadline=None):
    """Bounded batches; no labels are consulted to produce model scores."""
    check_recording(bundle, raw)
    if not 1 <= batch_size <= 128:
        raise ValueError("Inference batch size must be 1–128.")
    scores = []
    for offset in range(0, len(windows), batch_size):
        if deadline is not None and time.monotonic() > deadline:
            raise ValueError("Training/evaluation exceeded --max-minutes; no checkpoint saved.")
        batch = windows.iloc[offset:offset + batch_size]
        arrays = []
        for row in batch.itertuples():
            if abs((row.end_sec - row.start_sec) - bundle.window_sec) > 1e-6:
                raise ValueError("Window duration does not match checkpoint.")
            _, signal = read_window(raw, bundle.channels, row.start_sec, bundle.window_sec, bundle.preprocessing)
            arrays.append(signal)
        with torch.inference_mode():
            probabilities = bundle.model(torch.from_numpy(np.stack(arrays))).softmax(dim=1)[:, 1].numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Model produced non-finite probabilities.")
        scores.extend(probabilities.tolist())
        if progress:
            progress(min(1.0, (offset + len(batch)) / len(windows)))
    result = windows[["start_sec", "end_sec"]].copy()
    result["seizure_probability"] = scores
    result["prediction"] = (result.seizure_probability >= bundle.threshold).astype(int)
    return result
