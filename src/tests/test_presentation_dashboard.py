from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from src.tests.test_presentation_pipeline import trained

ROOT = Path(__file__).resolve().parents[2]


def test_upload_session_lifetime_annotations_and_error_recovery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.tests.test_presentation_pipeline import write_test_edf, summary
    path = write_test_edf(tmp_path / "recording.edf")
    bytes_edf = path.read_bytes()
    bytes_summary = summary().encode()
    uploads = {
        "edf": SimpleNamespace(name="recording.edf", file_id="test-good", size=len(bytes_edf), getvalue=lambda: bytes_edf),
        "summary": SimpleNamespace(name="summary.txt", size=len(bytes_summary), getvalue=lambda: bytes_summary),
    }
    def uploader(label, **kwargs):
        return uploads["summary"] if "summary" in label else uploads["edf"]
    monkeypatch.setattr("streamlit.file_uploader", uploader)
    app = AppTest.from_file(str(ROOT / "demo/presentation_app.py"), default_timeout=30).run()
    app.sidebar.radio[1].set_value("Upload EDF").run()
    assert not app.exception and not app.error
    session = app.session_state["recording_session"]
    uploaded_path = Path(session.raw.filenames[0])
    assert uploaded_path.exists() and not session.raw.preload
    app.sidebar.radio[0].set_value("EEG Preprocessing").run()
    assert not app.exception and not app.error
    assert uploaded_path.exists()  # Lazy EDF file survives Streamlit reruns.
    app.sidebar.radio[0].set_value("Segmentation").run()
    assert not app.exception and not app.error
    assert {m.label: m.value for m in app.metric}["Seizure windows"] == "1"
    uploads["edf"] = SimpleNamespace(name="bad.edf", file_id="test-bad", size=3, getvalue=lambda: b"bad")
    app.run()
    assert not app.exception and any("Unable to load EEG" in error.value for error in app.error)
    assert not uploaded_path.exists()  # Only the old session's own temporary copy is cleaned up.
    uploads["edf"] = SimpleNamespace(name="recording.edf", file_id="test-good", size=len(bytes_edf), getvalue=lambda: bytes_edf)
    app.run()
    assert not app.exception and not app.error


def test_trained_model_dashboard_outputs_and_saved_evaluation(trained, monkeypatch):
    from types import SimpleNamespace
    from src.tests.test_presentation_pipeline import summary
    root, output, _ = trained
    path = root / "chb01/chb01_01.edf"
    content = path.read_bytes()
    text = summary(path.name).encode()
    def uploader(label, **kwargs):
        if "summary" in label:
            return SimpleNamespace(name="summary.txt", size=len(text), getvalue=lambda: text)
        return SimpleNamespace(name=path.name, file_id="trained-test", size=len(content), getvalue=lambda: content)
    monkeypatch.setattr("streamlit.file_uploader", uploader)
    app = AppTest.from_file(str(ROOT / "demo/presentation_app.py"), default_timeout=30).run()
    app.sidebar.radio[1].set_value("Upload EDF").run()
    app.sidebar.radio[0].set_value("AI Seizure Detection").run()
    app.text_input[0].set_value(str(output / "baseline.pt")).run()
    assert not app.exception and not app.error
    next(b for b in app.button if b.key == "run_inference").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.session_state["inference_result"][1]) == 3
    app.sidebar.radio[0].set_value("Model Training & Evaluation").run()
    assert not app.exception and not app.error
    assert any(m.label == "Accuracy" for m in app.metric)


@pytest.mark.skipif(not (ROOT / "data/raw/chbmit/chb01/chb01_03.edf").exists(), reason="Local CHB-MIT sample required")
def test_dashboard_navigation_controls_and_honest_status():
    app = AppTest.from_file(str(ROOT / "demo/presentation_app.py"), default_timeout=30).run()
    assert not app.exception
    for section in ["Raw EEG Analysis", "EEG Preprocessing", "Seizure Annotations", "Segmentation", "AI Seizure Detection", "Model Training & Evaluation", "Early Prediction (Research)"]:
        app.sidebar.radio[0].set_value(section).run()
        assert not app.exception, section
        assert not app.error, section
        if section == "Raw EEG Analysis":
            app.slider[0].set_value(2990.).run()
            app.multiselect[0].set_value(["FP1-F7", "FP2-F4"]).run()
            assert not app.exception
        if section == "Seizure Annotations":
            app.button[0].click().run()
            assert app.slider[0].value == 2991.
            assert app.number_input[0].value == 50.
        if section == "Segmentation":
            counts = {item.label: item.value for item in app.metric}
            assert counts["Complete windows"] == "900" and counts["Seizure windows"] == "10"
        if section == "AI Seizure Detection" and not (ROOT / "results/presentation_baseline/baseline.pt").exists():
            assert any("model not trained" in warning.value for warning in app.warning)
    app.sidebar.radio[0].set_value("EEG Preprocessing").run()
    next(widget for widget in app.number_input if widget.label == "Low cutoff (Hz)").set_value(45.).run()
    assert not app.exception
    assert any("cutoff" in error.value for error in app.error)
