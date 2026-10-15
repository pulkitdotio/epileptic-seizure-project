"""Offline monitoring semantics, independent of Streamlit and of model training."""
import math

import numpy as np


def cnn_status(probability, threshold):
    """Only a genuine CNN score and its configured threshold determine an AI alert.

    Neither annotations nor a caller-supplied prediction flag may trigger it.
    Equality is positive, consistent with training/evaluation's >= rule.
    """
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid CNN probability; AI status unavailable.")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid CNN threshold; AI status unavailable.")
    alert = probability >= threshold
    return {"alert": alert, "state": "AI Seizure Detected" if alert else "No AI seizure alert",
            "probability": float(probability), "threshold": float(threshold)}


def annotation_status(start, end, intervals):
    """Any positive half-open overlap triggers an ANNOTATION alert, never an AI alert."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Selected interval must have finite increasing times.")
    if intervals is None:
        return {"available": False, "alert": False, "events": [], "start": start, "end": end}
    events = [(float(s), float(e)) for s, e in intervals if s < end and e > start]
    return {"available": True, "alert": bool(events), "events": events, "start": start, "end": end}


def apply_threshold(results, threshold):
    """Change decisions only; probabilities remain genuine, unmodified CNN outputs."""
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between zero and one.")
    scores = results.seizure_probability.to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid model probabilities.")
    result = results.copy()
    result["prediction"] = (scores >= threshold).astype(int)
    return result


def compare_annotations(results, intervals):
    """Annotation labels use the training policy; alert overlap is a separate concept."""
    result = results.copy()
    if intervals is None:
        result["annotation_label"] = None
        result["agreement"] = None
        return result
    # Union coverage, so overlapping annotations cannot double-count time.
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    labels = []
    for row in result.itertuples():
        coverage = sum(max(0., min(row.end_sec, e) - max(row.start_sec, s)) for s, e in merged)
        labels.append(int(coverage / (row.end_sec - row.start_sec) >= .5))
    result["annotation_label"] = labels
    result["agreement"] = result.prediction.to_numpy() == np.asarray(labels)
    return result


def advance_playback(start, recording_duration, window_sec):
    """Return next bounded start and end-of-recording state; no wrapping or skipped tail."""
    if (not all(math.isfinite(x) for x in (start, recording_duration, window_sec)) or
            min(start, recording_duration, window_sec) < 0 or window_sec <= 0 or recording_duration <= 0):
        raise ValueError("Invalid playback times.")
    last = max(0., recording_duration - window_sec)
    next_start = min(last, start + window_sec)
    return next_start, next_start >= last
