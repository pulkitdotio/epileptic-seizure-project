"""Pipeline contract tests. Synthetic EEG is only a test fixture, never demo data."""
from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd
import pytest
import torch

from demo.security import validate_edf_upload
from src.component.dataloaders.chbmit.download import parse_summary_text, parse_summary
from src.component.presentation.pipeline import annotations_for, open_edf, window_index, read_window, preprocess_array, PreprocessConfig
from src.component.presentation.model import load_baseline, predict_windows, BaselineCNN1D
from src.component.presentation.train import parser, train, split_windows, patient_group, cap_training_background, metrics
from src.component.presentation.evaluate import evaluate
from src.component.presentation.download_subset import selected_files, download_selected


def write_test_edf(path, seed=1, names=("C1", "C2")):
    """Minimal EDF with 12 one-second records at 128 Hz; no exporter dependency."""
    def field(value, width):
        return str(value).encode("ascii").ljust(width, b" ")
    n = len(names)
    header = b"".join([field("0", 8), field(f"Test patient {seed}", 80), field("Synthetic test fixture", 80),
                       field("01.01.24", 8), field("00.00.00", 8), field(256 + 256*n, 8),
                       field("", 44), field(12, 8), field(1, 8), field(n, 4)])
    columns = [(names, 16), ([""]*n, 80), (["uV"]*n, 8), ([-1000]*n, 8), ([1000]*n, 8),
               ([-32768]*n, 8), ([32767]*n, 8), ([""]*n, 80), ([128]*n, 8), ([""]*n, 32)]
    for values, width in columns:
        header += b"".join(field(value, width) for value in values)
    rng = np.random.default_rng(seed)
    t = np.arange(12 * 128) / 128
    signal = np.array([300*np.sin(2*np.pi*(8+i)*t) + rng.normal(0, 10, len(t)) for i in range(n)])
    signal[:, 4*128:8*128] += 400*np.sin(2*np.pi*3*t[4*128:8*128])
    digital = np.clip(signal * 32.767, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        for second in range(12):
            handle.write(digital[:, second*128:(second+1)*128].tobytes())
    return path


def summary(name="recording.edf"):
    return f"File Name: {name}\nNumber of Seizures in File: 1\nSeizure Start Time: 4 seconds\nSeizure End Time: 8 seconds\n"


@pytest.fixture
def edf(tmp_path):
    return write_test_edf(tmp_path / "recording.edf")


def test_edf_validation_and_lazy_read(edf, tmp_path):
    raw = open_edf(edf)
    try:
        assert not raw.preload
        assert raw.n_times == 1536 and raw.info["sfreq"] == 128
        t, data = read_window(raw, ["C2"], 4, 2)
        assert data.shape == (1, 256)
        assert t[0] == 4 and t[-1] == 6 - 1/128
        with pytest.raises(ValueError, match="bounds"):
            read_window(raw, ["C1"], 11, 4)
        with pytest.raises(ValueError, match="missing"):
            read_window(raw, ["missing"], 0, 4)
    finally:
        raw.close()
    bad = tmp_path / "bad.edf"
    bad.write_bytes(b"garbage")
    with pytest.raises(ValueError):
        open_edf(bad)
    with pytest.raises(FileNotFoundError):
        open_edf(tmp_path / "missing.edf")
    with pytest.raises(ValueError, match="extension"):
        validate_edf_upload(edf.read_bytes(), "bad.txt")
    with pytest.raises(ValueError, match="limit"):
        validate_edf_upload(edf.read_bytes(), "large.edf", max_bytes=100)


def test_duplicate_channel_names(tmp_path):
    path = write_test_edf(tmp_path / "duplicate.edf", names=("C1", "C1"))
    raw = open_edf(path)
    try:
        assert len(set(raw.ch_names)) == 2
        _, data = read_window(raw, raw.ch_names, 0, 4)
        assert data.shape == (2, 512)
    finally:
        raw.close()


def test_summary_parser_multiple_missing_and_zero(tmp_path):
    text = """File Name: a.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 10 seconds
Seizure 1 End Time: 14 seconds
Seizure 2 Start Time: 20 seconds
Seizure 2 End Time: 25 seconds
File Name: b.edf
Number of Seizures in File: 0
"""
    assert parse_summary_text(text) == {"a.edf": [(10., 14.), (20., 25.)], "b.edf": []}
    assert annotations_for(text, "missing.edf", 30) is None
    assert annotations_for(text, "b.edf", 30) == []
    with pytest.raises(ValueError, match="beyond"):
        annotations_for(text, "a.edf", 21)
    assert parse_summary(tmp_path / "missing.txt") == {}


@pytest.mark.parametrize("text", [summary().replace("End Time: 8", "End Time: 2"),
                                     summary().replace("in File: 1", "in File: 2"),
                                     summary().replace("Seizure End", "Missing End"), "not a summary"])
def test_reject_invalid_annotations(text):
    with pytest.raises(ValueError):
        parse_summary_text(text)


def test_window_boundary_union_tail_and_unknown():
    frame = window_index(13*128, 128, 4, intervals=[(3, 6)])
    assert frame.label.tolist() == [0, 1, 0]  # 25%, 50%, 0%
    assert frame.end_sec.iloc[-1] == 12
    union = window_index(4*128, 128, 4, intervals=[(0, 1.5), (1, 1.9)])
    assert union.label.tolist() == [0]  # Do not double-count overlapping intervals.
    assert window_index(512, 128, 4, intervals=None).label.isna().all()
    assert window_index(512, 128, 4, intervals=[]).label.tolist() == [0]
    assert len(window_index(12*128, 128, 4, 2, [(4, 8)])) == 5
    with pytest.raises(ValueError):
        window_index(512, 128, 4, 0)


def test_filter_frequency_attenuation_and_normalization(edf):
    sfreq = 256
    time = np.arange(10*sfreq) / sfreq
    signal = np.sin(2*np.pi*10*time) + np.sin(2*np.pi*80*time)
    cfg = PreprocessConfig(normalization="none")
    filtered = preprocess_array(signal[None], cfg)[0, sfreq:-sfreq]
    frequencies = np.fft.rfftfreq(len(filtered), 1/sfreq)
    power = abs(np.fft.rfft(filtered))
    assert power[np.argmin(abs(frequencies-10))] > 20 * power[np.argmin(abs(frequencies-80))]
    raw = open_edf(edf)
    try:
        t, normalized = read_window(raw, raw.ch_names, 4, 4, PreprocessConfig(sfreq=128))
        assert t[0] == 4 and normalized.shape == (2, 512)
        np.testing.assert_allclose(normalized.mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=1), 1, atol=1e-6)
        with pytest.raises(ValueError, match="Expected"):
            read_window(raw, raw.ch_names, 0, 4, PreprocessConfig())
    finally:
        raw.close()
    assert np.isfinite(preprocess_array(np.zeros((2, 512)), PreprocessConfig())).all()
    with pytest.raises(ValueError):
        preprocess_array(signal[None], PreprocessConfig(high_hz=200))


def test_patient_split_and_recording_leakage_guards(tmp_path):
    rows = pd.DataFrame([{"path": f"{case}/file.edf", "patient": patient_group(case), "label": label}
                         for case in ["chb01", "chb21", "chb02", "chb03"] for label in [0, 1]])
    splits = split_windows(rows)
    groups = [set(part.patient) for part in splits.values()]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    assert patient_group("chb21") == "chb01"
    with pytest.raises(ValueError, match="at least 3"):
        split_windows(rows[rows.patient == "chb01"])
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": ["chb01"], "val": ["chb21", "chb02"], "test": ["chb03"]}))
    with pytest.raises(ValueError, match="exactly once"):
        split_windows(rows, split_file=split_file)
    negatives = pd.DataFrame({"path": ["a"]*11, "label": [1] + [0]*10})
    capped = cap_training_background(negatives, 2, 42)
    assert len(capped) == 3 and capped.label.sum() == 1


@pytest.fixture
def trained(tmp_path):
    root = tmp_path / "data"
    for i in range(1, 4):
        case = f"chb{i:02d}"
        name = f"{case}_01.edf"
        write_test_edf(root / case / name, seed=i)
        (root / case / f"{case}-summary.txt").write_text(summary(name))
    output = tmp_path / "run"
    args = parser().parse_args(["--data-root", str(root), "--output", str(output), "--epochs", "1",
                               "--batch-size", "2", "--sfreq", "128", "--channels", "C1,C2"])
    train(args)
    return root, output, args


def test_real_training_checkpoint_inference_and_evaluation(trained):
    root, output, args = trained
    bundle = load_baseline(output / "baseline.pt")
    assert bundle.model(torch.zeros(2, 2, 512)).shape == (2, 2)
    raw = open_edf(root / "chb01/chb01_01.edf")
    try:
        windows = window_index(raw.n_times, 128, 4, intervals=[(4, 8)])
        result = predict_windows(bundle, raw, windows, batch_size=2)
        assert result.seizure_probability.between(0, 1).all()
        unknown = windows.copy()
        unknown["label"] = 1 - unknown.label
        np.testing.assert_allclose(predict_windows(bundle, raw, unknown).seizure_probability, result.seizure_probability, atol=1e-7)
    finally:
        raw.close()
    report, results = evaluate(output / "baseline.pt")
    assert report["n_windows"] == 3 and sum(map(sum, report["confusion_matrix"])) == 3
    assert report == {"scope": "patient-held-out", **bundle.payload["evaluation"]}
    with pytest.raises(ValueError, match="already exists"):
        train(args)
    test_record = next(record for record in bundle.payload["training"]["records"] if record["split"] == "test")
    changed = root / test_record["relative_path"]
    with changed.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        evaluate(output / "baseline.pt")


def test_checkpoint_missing_incompatible_and_weights(trained, tmp_path):
    _, output, _ = trained
    with pytest.raises(FileNotFoundError, match="Model not trained"):
        load_baseline(tmp_path / "absent.pt")
    wrong = tmp_path / "wrong.pt"
    torch.save({"format": "other"}, wrong)
    with pytest.raises(ValueError, match="format"):
        load_baseline(wrong)
    payload = torch.load(output / "baseline.pt", weights_only=True)
    payload["channels"] = ["C1", "C1"]
    torch.save(payload, wrong)
    with pytest.raises(ValueError, match="channel"):
        load_baseline(wrong)
    payload["channels"] = ["C1", "C2"]
    payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
    torch.save(payload, wrong)
    with pytest.raises(ValueError, match="checkpoint"):
        load_baseline(wrong)


def test_metrics_known_confusion_and_downloader_dry_run(tmp_path, monkeypatch):
    report = metrics(pd.DataFrame({"label": [0, 0, 1, 1], "seizure_probability": [.1, .8, .7, .2]}), .5)
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]
    assert all(report[key] == .5 for key in ["accuracy", "precision", "recall", "f1", "sensitivity", "specificity"])
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("Dry run must not use the network"))
    download_selected(["chb01/chb01_03.edf"], tmp_path / "download")
    assert not (tmp_path / "download").exists()
    with pytest.raises(ValueError):
        selected_files(["../anything.edf"])


def test_downloaded_chb01_recording():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/raw/chbmit/chb01/chb01_03.edf"
    if not path.exists():
        pytest.skip("Local CHB-MIT sample not installed")
    before = path.stat().st_size
    raw = open_edf(path)
    try:
        events = annotations_for(path.with_name("chb01-summary.txt").read_text(), path.name, raw.n_times / 256)
        assert events == [(2996., 3036.)]
        assert len(raw.ch_names) == 23 and raw.n_times / raw.info["sfreq"] == 3600
        frame = window_index(raw.n_times, 256, 4, intervals=events)
        assert len(frame) == 900 and frame.label.sum() == 10
        times, signal = read_window(raw, raw.ch_names[:5], 2991, 50, PreprocessConfig())
        assert signal.shape == (5, 12800) and times[0] == 2991
    finally:
        raw.close()
    assert path.stat().st_size == before
