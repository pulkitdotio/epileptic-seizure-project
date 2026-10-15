"""Lazy EDF access, bounded signal processing, and sample-exact window labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from demo.security import MAX_EDF_UPLOAD_BYTES, validate_edf_upload
from src.component.dataloaders.chbmit.download import parse_summary_text


def open_edf(path: str | Path):
    """Validate a local EDF and read its header, without preloading signal data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")
    with path.open("rb") as handle:
        validate_edf_upload(handle.read(256), path.name)
    try:
        # MNE gives duplicate labels deterministic suffixes; select by exact names
        # or integer indices afterwards, never by the ambiguous original labels.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Channel names are not unique.*")
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        if raw.n_times < 2 or not math.isfinite(raw.info["sfreq"]) or raw.info["sfreq"] <= 0:
            raw.close()
            raise ValueError("Recording has no usable signal samples.")
        return raw
    except (OSError, ValueError, RuntimeError, IndexError, AssertionError) as exc:
        raise ValueError(f"Cannot read EDF recording: {exc}") from exc


def annotations_for(text: str, filename: str, duration: float):
    entries = parse_summary_text(text)
    if filename not in entries:
        return None  # Unknown labels must never become negative training examples.
    intervals = entries[filename]
    if any(end > duration + 1e-6 for _, end in intervals):
        raise ValueError("Summary annotations extend beyond this recording.")
    return intervals


@dataclass(frozen=True)
class PreprocessConfig:
    sfreq: float = 256.0
    low_hz: float = 1.0
    high_hz: float = 40.0
    normalization: str = "zscore"
    order: int = 4
    context_sec: float = 2.0

    def validate(self):
        if not all(math.isfinite(x) for x in (self.sfreq, self.low_hz, self.high_hz, self.context_sec)):
            raise ValueError("Preprocessing values must be finite.")
        if not 0 < self.low_hz < self.high_hz < self.sfreq / 2:
            raise ValueError("Use 0 < low cutoff < high cutoff < sampling frequency / 2.")
        if self.normalization not in {"none", "zscore"}:
            raise ValueError("Normalization must be none or zscore.")
        if self.order != 4 or not 0 <= self.context_sec <= 10:
            raise ValueError("Prototype supports order 4 and 0–10 seconds of context.")

    def to_dict(self):
        self.validate()
        return asdict(self)


def preprocess_array(data: np.ndarray, cfg: PreprocessConfig, *, normalize=True):
    """Zero-phase SOS Butterworth filter. Offline processing, not causal forecasting."""
    cfg.validate()
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError("Expected finite EEG array (channels, samples).")
    sos = butter(cfg.order, [cfg.low_hz, cfg.high_hz], btype="bandpass", fs=cfg.sfreq, output="sos")
    try:
        filtered = sosfiltfilt(sos, data, axis=-1)
    except ValueError as exc:
        raise ValueError("Signal is too short for this bandpass filter.") from exc
    if normalize and cfg.normalization == "zscore":
        filtered = standardize(filtered)
    return filtered.astype(np.float32)


def standardize(data):
    """Per-channel, per-window z-score; constant channels stay finite at zero."""
    return ((data - data.mean(axis=-1, keepdims=True)) /
            np.maximum(data.std(axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def read_window(raw, channels, start_sec, duration_sec, cfg=None):
    """Read only selected channels and interval (+ context), preserving absolute time.

    Sampling rates must match; no channel padding or silent resampling. Filtering
    uses context on each side, then crops before per-window normalization. At file
    edges the available context is shorter and SOS reflection padding is used.
    """
    sfreq = float(raw.info["sfreq"])
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("Select at least one unique channel.")
    if any(ch not in raw.ch_names for ch in channels):
        raise ValueError("Recording is missing one or more requested channels.")
    if not math.isfinite(start_sec) or not math.isfinite(duration_sec) or start_sec < 0 or duration_sec <= 0:
        raise ValueError("Window start must be nonnegative and duration positive.")
    start, count = round(start_sec * sfreq), round(duration_sec * sfreq)
    stop = start + count
    if count < 2 or stop > raw.n_times:
        raise ValueError("Requested window exceeds recording bounds or is too short.")
    times = np.arange(start, stop) / sfreq
    picks = [raw.ch_names.index(ch) for ch in channels]
    if cfg is None:
        return times, raw.get_data(picks=picks, start=start, stop=stop).astype(np.float32)
    cfg.validate()
    if abs(cfg.sfreq - sfreq) > 1e-6:
        raise ValueError(f"Expected {cfg.sfreq:g} Hz; recording is {sfreq:g} Hz. Resampling is not implicit.")
    padding = round(cfg.context_sec * sfreq)
    left, right = max(0, start - padding), min(raw.n_times, stop + padding)
    context = raw.get_data(picks=picks, start=left, stop=right)
    result = preprocess_array(context, cfg, normalize=False)[:, start - left:stop - left]
    if cfg.normalization == "zscore":
        result = standardize(result)
    return times, result


def window_index(n_samples, sfreq, window_sec=4.0, stride_sec=None, intervals=None):
    """Metadata only. Complete half-open windows; discard the trailing partial one.

    Label 1 if the UNION of seizures covers >=50% of the window, otherwise 0.
    Unknown annotations produce nullable labels. Starts are integer sample indices.
    """
    stride_sec = window_sec if stride_sec is None else stride_sec
    if not all(math.isfinite(x) and x > 0 for x in (sfreq, window_sec, stride_sec)):
        raise ValueError("Sampling frequency, window and stride must be positive.")
    width, step = round(window_sec * sfreq), round(stride_sec * sfreq)
    if width < 2 or step < 1:
        raise ValueError("Window or stride is shorter than one sample.")
    count = max(0, (n_samples - width) // step + 1)
    if count > 250_000:
        raise ValueError("Too many windows; increase window duration or stride.")
    starts = np.arange(count, dtype=np.int64) * step
    ends = starts + width
    overlap = np.zeros(count)
    merged = []
    for s, e in sorted(intervals or []):
        if not (math.isfinite(s) and math.isfinite(e) and 0 <= s < e <= n_samples / sfreq + 1e-6):
            raise ValueError("Invalid seizure interval.")
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    for s, e in merged:
        overlap += np.maximum(0, np.minimum(ends / sfreq, e) - np.maximum(starts / sfreq, s))
    fraction = overlap / (width / sfreq)
    labels = pd.array((fraction >= 0.5).astype(int), dtype="Int64") if intervals is not None else pd.array([None] * count, dtype="Int64")
    return pd.DataFrame({"start_sample": starts, "end_sample": ends,
                         "start_sec": starts / sfreq, "end_sec": ends / sfreq,
                         "overlap_fraction": fraction if intervals is not None else np.nan,
                         "label": labels})
