"""Monitoring semantics and UI checks; synthetic checkpoints remain test-only artifacts."""
import json
from pathlib import Path
import time
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from demo.plots import waveform, probability_plot, training_curves, confusion_plot, BACKGROUND
from src.component.presentation.monitoring import annotation_status, apply_threshold, compare_annotations, advance_playback
from src.component.presentation.pipeline import PreprocessConfig
from src.component.presentation.train import parser, train, probability_loss, metrics
from src.tests.test_presentation_pipeline import trained

ROOT = Path(__file__).resolve().parents[2]


def test_annotation_alert_half_open_boundaries_and_unknown():
    events = [(2996., 3036.)]
    assert not annotation_status(2990, 2996, events)["alert"]
    assert not annotation_status(3036, 3040, events)["alert"]
    status = annotation_status(2995, 2997, events)
    assert status["alert"] and status["events"] == events
    assert not annotation_status(0, 4, None)["available"]
    assert annotation_status(0, 4, [])["available"]
    with pytest.raises(ValueError):
        annotation_status(4, 4, events)


def test_threshold_changes_decisions_without_changing_scores():
    outputs = pd.DataFrame({"start_sec": [0., 4., 8.], "end_sec": [4., 8., 12.],
                            "seizure_probability": [.2, .5, .8], "prediction": [0, 1, 1]})
    result = apply_threshold(outputs, .7)
    assert result.prediction.tolist() == [0, 0, 1]
    np.testing.assert_array_equal(result.seizure_probability, outputs.seizure_probability)
    assert outputs.prediction.tolist() == [0, 1, 1]
    comparison = compare_annotations(result, [(3, 6)])
    assert comparison.annotation_label.tolist() == [0, 1, 0]
    assert comparison.agreement.tolist() == [True, False, False]
    assert compare_annotations(result, None).annotation_label.isna().all()
    # Union coverage avoids double-counting overlapping annotations.
    assert compare_annotations(result, [(0, 1.5), (1, 1.9)]).annotation_label.iloc[0] == 0
    with pytest.raises(ValueError):
        apply_threshold(outputs, 0)


def test_playback_bounds_and_last_window():
    assert advance_playback(0, 12, 4) == (4, False)
    assert advance_playback(4, 12, 4) == (8, True)
    assert advance_playback(8, 12, 4) == (8, True)
    assert advance_playback(8, 13, 4) == (9, True)


def test_dark_plot_theme_and_model_annotation_separation():
    from matplotlib.colors import to_hex
    theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert theme["theme"]["base"] == "dark"
    t = np.arange(50 * 256) / 256 + 2991
    fig = waveform(t, np.sin(t)[None] * 1e-4, ["FP1-F7"], [(2996, 3036)])
    assert to_hex(fig.get_facecolor()) == BACKGROUND
    assert to_hex(fig.axes[0].get_facecolor()) == BACKGROUND
    assert to_hex(fig.axes[0]._left_title.get_color()) == "#e8edf3"
    patch = fig.axes[0].patches[0]
    assert patch.get_x() == 2996 and patch.get_width() == 40
    out = pd.DataFrame({"start_sec": [2992., 2996.], "end_sec": [2996., 3000.], "seizure_probability": [.8, .2]})
    fig2 = probability_plot(out, .5, [(2996, 3036)])
    labels = fig2.axes[0].get_legend_handles_labels()[1]
    assert "CNN alert (top strip)" in labels and "Annotation (shading)" in labels
    assert to_hex(training_curves(pd.DataFrame({"epoch": [1], "training_loss": [.5], "validation_loss": [.7]})).get_facecolor()) == BACKGROUND
    assert to_hex(confusion_plot([[10, 2], [1, 3]]).get_facecolor()) == BACKGROUND


def assert_single_sidebar(app, upload=False):
    assert len(app.sidebar.radio) == 2
    assert sum(r.label == "Navigation" for r in app.sidebar.radio) == 1
    assert sum(r.label == "Choose source" for r in app.sidebar.radio) == 1
    assert len(app.get("file_uploader")) == (2 if upload else 1)
    assert len([w for w in app.sidebar.text_input if w.key == "checkpoint_path"]) == 1


@pytest.mark.skipif(not (ROOT / "data/raw/chbmit/chb01/chb01_03.edf").exists(), reason="Local EDF required")
def test_sidebar_and_offline_playback_across_reruns():
    app = AppTest.from_file(str(ROOT / "demo/presentation_app.py"), default_timeout=30).run()
    assert_single_sidebar(app)
    app.sidebar.radio[0].set_value("Raw EEG Analysis").run()
    next(b for b in app.button if b.key == "jump_event").click().run()
    assert any("Annotated seizure" in m.value and "ground-truth" in m.value for m in app.markdown)
    next(b for b in app.button if b.key == "play_start").click().run()
    assert not app.exception and app.session_state["playing"]
    assert_single_sidebar(app)
    assert app.session_state["duration_sec"] == 4
    current = app.session_state["start_sec"]
    app.session_state["playback_tick"] = time.monotonic() - 2
    app.run()
    assert not app.exception and app.session_state["start_sec"] == current + 4
    next(b for b in app.button if b.key == "play_pause").click().run()
    assert not app.exception and not app.session_state["playing"]
    paused = app.session_state["start_sec"]
    app.run()
    assert app.session_state["start_sec"] == paused
    next(b for b in app.button if b.key == "play_reset").click().run()
    assert app.session_state["start_sec"] == 0 and not app.session_state["playing"]
    assert_single_sidebar(app)
    app.sidebar.radio[0].set_value("AI Seizure Detection").run()
    assert_single_sidebar(app)
    assert not app.exception
    expected = "Source: trained CNN inference" if (ROOT / "results/presentation_baseline/baseline.pt").exists() else "prediction unavailable"
    assert any(expected in m.value for m in app.markdown)


def test_preflight_no_checkpoint_and_saved_history_auc(trained, capsys):
    root, output, original_args = trained
    payload = __import__("torch").load(output / "baseline.pt", weights_only=True)
    assert len(payload["history"]) == 1
    assert payload["history"][0]["validation_loss"] >= 0
    assert 0 <= payload["evaluation"]["roc_auc"] <= 1
    assert 0 <= payload["evaluation"]["pr_auc"] <= 1
    assert payload["training"]["class_counts_by_split"]["test"] == [2, 1]
    new_output = output.parent / "preflight"
    args = parser().parse_args(["--data-root", str(root), "--output", str(new_output), "--preflight",
                               "--sfreq", "128", "--channels", "C1,C2", "--batch-size", "2"])
    train(args)
    assert not new_output.exists()
    assert "one_training_step_seconds" in capsys.readouterr().out


def test_probability_loss_and_auc_undefined_single_class():
    result = pd.DataFrame({"label": [0, 1], "seizure_probability": [.1, .9]})
    assert probability_loss(result, [1, 1]) == pytest.approx(-np.log(.9))
    assert metrics(result, .5)["roc_auc"] == 1
    one_class = pd.DataFrame({"label": [0, 0], "seizure_probability": [.1, .2]})
    assert metrics(one_class, .5)["roc_auc"] is None and metrics(one_class, .5)["pr_auc"] is None


def test_training_deadline_never_installs_incomplete_checkpoint(trained):
    root, output, _ = trained
    limited = output.parent / "deadline_limited"
    args = parser().parse_args(["--data-root", str(root), "--output", str(limited), "--max-minutes", "0.000000001",
                               "--sfreq", "128", "--channels", "C1,C2", "--batch-size", "2"])
    with pytest.raises(ValueError, match="exceeded"):
        train(args)
    assert not (limited / "baseline.pt").exists()
    assert json.loads((limited / "configuration.json").read_text())["training"]["epochs_completed"] == 0


def test_dashboard_monitor_reads_trained_weights_and_preserves_saved_evaluation(trained, monkeypatch):
    from types import SimpleNamespace
    from src.component.presentation.model import load_baseline, predict_windows, check_recording
    from src.component.presentation.pipeline import open_edf
    from src.tests.test_presentation_pipeline import summary
    root, output, _ = trained
    path = root / "chb01/chb01_01.edf"
    content = path.read_bytes()
    text = summary(path.name).encode()
    def uploader(label, **kwargs):
        if "summary" in label:
            return SimpleNamespace(name="summary.txt", size=len(text), getvalue=lambda: text)
        return SimpleNamespace(name=path.name, file_id="monitor-weights", size=len(content), getvalue=lambda: content)
    monkeypatch.setattr("streamlit.file_uploader", uploader)
    bundle = load_baseline(output / "baseline.pt")
    raw = open_edf(path)
    try:
        expected = predict_windows(bundle, raw, pd.DataFrame({"start_sec": [0.], "end_sec": [4.]}))
        with pytest.raises(ValueError, match="requires these exact channels"):
            check_recording(replace(bundle, channels=["missing"]), raw)
        with pytest.raises(ValueError, match="requires 256 Hz"):
            check_recording(replace(bundle, preprocessing=PreprocessConfig()), raw)
    finally:
        raw.close()
    app = AppTest.from_file(str(ROOT / "demo/presentation_app.py"), default_timeout=30).run()
    app.sidebar.radio[1].set_value("Upload EDF").run()
    app.text_input[0].set_value(str(output / "baseline.pt")).run()
    app.sidebar.radio[0].set_value("AI Seizure Detection").run()
    assert not app.exception and not app.error
    card = next(m.value for m in app.markdown if 'class="label">CNN prediction' in m.value)
    assert f"Model probability: {expected.seizure_probability.iloc[0]:.6f}" in card
    assert "Source: trained CNN inference" in card
    next(w for w in app.checkbox if w.label == "Use exploratory display threshold").check().run()
    next(w for w in app.slider if w.label == "Display decision threshold").set_value(.95).run()
    card = next(m.value for m in app.markdown if 'class="label">CNN prediction' in m.value)
    assert "Decision threshold: 0.950" in card
    assert f"Model probability: {expected.seizure_probability.iloc[0]:.6f}" in card
    app.sidebar.radio[0].set_value("Model Training & Evaluation").run()
    assert not app.exception and not app.error
    assert any(m.label == "ROC-AUC" for m in app.metric)
    saved = load_baseline(output / "baseline.pt")
    assert saved.payload["evaluation"] == bundle.payload["evaluation"]


def test_manifest_patient_selection_and_present_file_checksum():
    from src.component.presentation.download_subset import load_manifest, sha256
    manifest = load_manifest(ROOT / "configs/presentation_subset.json")
    assert len(manifest["recordings"]) == 10
    assert sum(len(r["events"]) for r in manifest["recordings"]) == 6
    groups = [{r["path"].split('/')[0] for r in manifest["recordings"] if r["split"] == split} for split in ("train", "val", "test")]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    record = next(r for r in manifest["recordings"] if r["path"] == "chb01/chb01_03.edf")
    path = ROOT / "data/raw/chbmit" / record["path"]
    if path.exists():
        assert sha256(path) == record["sha256"]
