import pytest

from src.tests.test_presentation_pipeline import write_test_edf, summary
from src.component.presentation.train import parser, train
from src.component.presentation.model import load_baseline
from src.component.presentation.evaluate import evaluate


def test_opt_in_fit_only_saves_weights_without_evaluation(tmp_path):
    root = tmp_path / "one_patient"
    name = "chb01_01.edf"
    write_test_edf(root / "chb01" / name)
    (root / "chb01/chb01-summary.txt").write_text(summary(name))
    output = tmp_path / "fit"
    args = parser().parse_args(["--data-root", str(root), "--output", str(output), "--epochs", "1",
                               "--batch-size", "2", "--sfreq", "128", "--channels", "C1,C2"])
    with pytest.raises(ValueError, match="at least 3"):
        train(args)
    assert not output.exists()
    args.fit_only = True
    train(args)
    bundle = load_baseline(output / "baseline.pt")
    assert bundle.payload["training"]["evaluation_scope"] == "fit-only"
    assert bundle.payload["evaluation"] is None
    assert not (output / "evaluation.json").exists()
    with pytest.raises(ValueError, match="Fit-only"):
        evaluate(output / "baseline.pt")
