"""Synthetic saved-score audit tests; never train or write project checkpoints."""
import json

import numpy as np
import pandas as pd
import pytest

from src.component.presentation.score_audit import alarm_statistics, choose, read_scores, check_saved_windows, test_phase as run_test_phase
from src.component.presentation.thresholds import threshold_curve


def frame(scores, labels, path="a"):
    return pd.DataFrame({"path": path, "patient": "p", "start_sec": np.arange(len(scores)) * 4.,
                         "end_sec": np.arange(1, len(scores) + 1) * 4.,
                         "seizure_probability": scores, "label": labels})


def test_alarm_episodes_boundaries_exposure_and_background_burden():
    f = frame([.9, .9, .1, .9, .9], [0, 0, 0, 1, 0])
    measured = alarm_statistics(f, .9, {"a": [(13., 17.)]})
    assert measured["alarm_episodes"] == 2
    assert measured["pure_false_alarm_episodes"] == 1
    assert measured["events_detected"] == 1
    assert measured["nonseizure_hours"] == pytest.approx(16 / 3600)
    assert measured["background_alert_seconds"] == 12
    assert measured["false_positive_windows_per_labelled_background_hour"] == pytest.approx(3 * 3600 / 16)


def test_episode_union_file_seams_and_missing_windows():
    a = frame([.9, .9], [0, 1])
    b = frame([.9], [0], "b")
    f = pd.concat([a, b], ignore_index=True)
    m = alarm_statistics(f, .5, {"a": [(4, 7), (6, 8)], "b": []})
    assert m["alarm_episodes"] == 2 and m["pure_false_alarm_episodes"] == 1
    assert m["nonseizure_hours"] == pytest.approx(8 / 3600)  # Union, not double counted.
    a.loc[1, ["start_sec", "end_sec"]] = [12, 16]
    assert alarm_statistics(a, .5, {"a": []})["alarm_episodes"] == 2
    # Touching the seizure onset at the episode end is not overlap.
    assert alarm_statistics(frame([.9, .1], [0, 1]), .5, {"a": [(4, 8)]})["pure_false_alarm_episodes"] == 1


def test_guarded_validation_selection_and_exact_ties():
    f = frame([.1, .2, .3, .8, .8, .9], [0, 0, 0, 0, 1, 1])
    curve = threshold_curve(f)
    curve["false_alarm_episodes_per_nonseizure_hour"] = curve.false_positives * 2.
    assert choose(curve, "recall", .25, 1)["threshold"] == .9
    assert choose(curve, "recall", .25, 2)["threshold"] == .8
    assert choose(curve, "f1", .25)["false_positives"] == 1


def test_saved_score_round_trip_and_reject_overlap(tmp_path):
    path = tmp_path / "scores.csv"
    f = frame([.5674257874488831, .1], [1, 0])
    f.to_csv(path, index=False)
    assert read_scores(path).seizure_probability.iloc[0] == f.seizure_probability.iloc[0]
    f.loc[1, "start_sec"] = 2
    f.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Overlapping"):
        read_scores(path)


def test_saved_scores_must_match_verified_index(tmp_path):
    f = frame([.9, .1], [1, 0])
    path = tmp_path / "index.csv"
    f.drop(columns="seizure_probability").to_csv(path, index=False)
    check_saved_windows(f.iloc[::-1], path)
    f.loc[0, "label"] = 0
    with pytest.raises(ValueError, match="verified window index"):
        check_saved_windows(f, path)


def test_test_phase_refuses_second_evaluation_before_reading_scores(tmp_path, monkeypatch):
    (tmp_path / "held_out_results.json").write_text(json.dumps({}))
    monkeypatch.setattr("src.component.presentation.score_audit.read_scores",
                        lambda *args: pytest.fail("Must not read test scores again"))
    with pytest.raises(ValueError, match="already performed"):
        run_test_phase(tmp_path)
