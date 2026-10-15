"""Detection regressions. Test fixtures/metrics are never installed as project results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.component.presentation.objectives import BalancedBatchSampler, make_loss
from src.component.presentation.thresholds import select_threshold, threshold_curve
from src.component.presentation.monitoring import cnn_status, annotation_status
from src.component.presentation.train import parser, train, metrics, LazyWindows
from src.component.presentation.model import load_baseline, predict_windows
from src.component.presentation.pipeline import open_edf, read_window
from src.component.presentation.evaluate import evaluate
from src.tests.test_presentation_pipeline import trained


def test_balanced_batches_reproducible_mixed_and_bounded():
    labels = np.array([0] * 100 + [1] * 2)
    sampler = BalancedBatchSampler(labels, 16, 4, seed=42)
    first = list(sampler)
    assert len(first) == 4
    assert all(len(batch) == 16 and labels[batch].sum() == 8 for batch in first)
    assert first == list(BalancedBatchSampler(labels, 16, 4, seed=42))
    sampler.set_epoch(1)
    assert first != list(sampler)
    with pytest.raises(ValueError):
        BalancedBatchSampler(labels, 15, 4)
    with pytest.raises(ValueError):
        make_loss("weighted-bce", [100, 2], "balanced")


def test_binary_losses_stability_and_global_positive_weight():
    logits = torch.tensor([[0., -1000.], [0., 1000.], [0., -2.], [0., 2.]], requires_grad=True)
    labels = torch.tensor([1, 0, 1, 0])
    for name in ("bce", "weighted-bce", "focal"):
        value = make_loss(name, [100, 2], "natural")(logits, labels)
        assert torch.isfinite(value)
        value.backward(retain_graph=True)
        assert torch.isfinite(logits.grad).all()
    negative = make_loss("weighted-bce", [100, 2], "natural")(torch.zeros(1, 2), torch.tensor([0]))
    positive = make_loss("weighted-bce", [100, 2], "natural")(torch.zeros(1, 2), torch.tensor([1]))
    assert positive / negative == pytest.approx(50)
    # Inverse-frequency CE on separate one-class batches cancels the weight.
    ce = make_loss("weighted-ce", [100, 2], "natural")
    assert ce(torch.zeros(1, 2), torch.tensor([0])) == ce(torch.zeros(1, 2), torch.tensor([1]))
    # Focal gamma=0 on balanced batches equals plain BCE, without a second weighting.
    assert make_loss("focal", [100, 2], "balanced", 0)(logits, labels) == make_loss("bce", [100, 2], "balanced")(logits, labels)


def test_exact_threshold_ties_fpr_and_validation_only():
    validation = pd.DataFrame({"label": [0, 0, 0, 0, 1, 1], "seizure_probability": [.1, .2, .3, .8, .8, .9]})
    constrained, curve = select_threshold(validation, max_fpr=0)
    assert constrained["threshold"] == .9 and constrained["validation_false_positives"] == 0
    relaxed, _ = select_threshold(validation, max_fpr=.25)
    assert relaxed["threshold"] == .8 and relaxed["validation_false_positives"] == 1
    assert relaxed["validation_recall"] == 1
    for row in curve.itertuples():
        measured = metrics(validation, row.threshold)
        assert measured["false_positives"] == row.false_positives
        assert measured["f1"] == pytest.approx(row.f1)
    assert relaxed == select_threshold(validation.sample(frac=1, random_state=42), .25)[0]
    with pytest.raises(ValueError, match="both validation classes"):
        threshold_curve(validation[validation.label == 0])


def test_ai_alert_independent_of_annotation_and_invalid_scores():
    assert annotation_status(4, 8, [(4, 8)])["alert"]
    assert not cnn_status(.49, .5)["alert"]
    assert cnn_status(.51, .5)["state"] == "AI Seizure Detected"
    assert cnn_status(.5, .5)["alert"]  # The documented >= contract.
    for p in (float("nan"), float("inf"), -.1, 1.1):
        with pytest.raises(ValueError):
            cnn_status(p, .5)


def test_balanced_training_save_load_and_frozen_evaluation(trained):
    root, previous, _ = trained
    before = (previous / "baseline.pt").read_bytes()
    output = previous.parent / "balanced"
    args = parser().parse_args(["--data-root", str(root), "--output", str(output), "--epochs", "2",
                               "--batch-size", "2", "--sfreq", "128", "--channels", "C1,C2",
                               "--sampling", "balanced", "--loss", "bce", "--steps-per-epoch", "3",
                               "--threshold-policy", "exact-f1", "--max-val-fpr", "1"])
    train(args)
    bundle = load_baseline(output / "baseline.pt")
    assert bundle.payload["training"]["sampling"] == "balanced"
    assert all(e["positive_free_batches"] == 0 and e["training_examples_drawn"] == 6 for e in bundle.payload["history"])
    selection = json.loads((output / "threshold_selection.json").read_text())
    validation = pd.read_csv(output / "validation_predictions.csv")
    assert selection == select_threshold(validation, 1)[0]
    report, _ = evaluate(output / "baseline.pt")
    assert report == {"scope": "patient-held-out", **bundle.payload["evaluation"]}
    assert (previous / "baseline.pt").read_bytes() == before


def test_lazy_training_and_inference_identical_preprocessed_tensor(trained, monkeypatch):
    root, output, _ = trained
    bundle = load_baseline(output / "baseline.pt")
    path = root / "chb01/chb01_01.edf"
    frame = pd.DataFrame({"path": [str(path)], "start_sec": [4.], "end_sec": [8.], "label": [1]})
    dataset = LazyWindows(frame, bundle.channels, bundle.preprocessing, bundle.window_sec)
    raw = open_edf(path)
    try:
        trained_tensor, _ = dataset[0]
        _, expected = read_window(raw, bundle.channels, 4, 4, bundle.preprocessing)
        np.testing.assert_array_equal(trained_tensor.numpy(), expected)
        captured = []
        hook = bundle.model.register_forward_pre_hook(lambda model, args: captured.append(args[0].clone()))
        result = predict_windows(bundle, raw, frame)
        hook.remove()
        torch.testing.assert_close(captured[0][0], trained_tensor, rtol=0, atol=0)
        assert result.prediction.iloc[0] == cnn_status(result.seizure_probability.iloc[0], bundle.threshold)["alert"]
    finally:
        raw.close()
        dataset.close()


def test_ai_card_uses_probability_not_stale_prediction_flag(monkeypatch):
    from demo.model_panel import ai_status
    cards = []
    monkeypatch.setattr("demo.model_panel.status_card", lambda *args: cards.append(args))
    monkeypatch.setattr("demo.model_panel.st.caption", lambda *args: None)
    result = pd.DataFrame({"start_sec": [4.], "end_sec": [8.], "seizure_probability": [.49], "prediction": [1]})
    ai_status(result, .5)
    assert cards[-1][1] == "No AI seizure alert"
    result["seizure_probability"], result["prediction"] = .51, 0
    ai_status(result, .5)
    assert cards[-1][1] == "AI Seizure Detected"
