"""Frontend regressions using preserved local artifacts; never train a model."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from demo.model_panel import saved_curve_scores
from demo.plots import discrimination_plot
from src.component.presentation.model import load_baseline

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "results/presentation_baseline/baseline.pt"
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="Preserved local checkpoint required")


def test_overview_shortcut_and_comparison_layout_preserve_navigation():
    app = AppTest.from_file(str(ROOT / "demo/presentation_app.py"), default_timeout=30).run()
    next(b for b in app.button if b.key == "overview_seizure").click().run()
    assert app.session_state["navigation"] == "Raw EEG Analysis"
    assert app.session_state["start_sec"] == 2991
    assert app.session_state["duration_sec"] == 50
    app.sidebar.radio[0].set_value("EEG Preprocessing").run()
    next(r for r in app.radio if r.key == "comparison_layout").set_value("Stacked").run()
    assert not app.exception and len(app.sidebar.radio) == 2
    app.sidebar.radio[0].set_value("Segmentation").run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric}["Seizure windows"] == "10"


def test_model_warning_and_saved_metrics_are_visible_and_unchanged():
    bundle = load_baseline(CHECKPOINT)
    app = AppTest.from_file(str(ROOT / "demo/presentation_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("AI Seizure Detection").run()
    assert not app.exception
    assert sum("0% seizure recall" in w.value for w in app.warning) == 1
    card = next(m.value for m in app.markdown if 'class="label">CNN prediction' in m.value)
    assert f"Decision threshold: {bundle.threshold:.6f}" in card
    assert "Source: trained CNN inference" in card
    assert "Dataset annotation" in " ".join(m.value for m in app.markdown)
    assert not next(w for w in app.checkbox if w.label == "Use exploratory display threshold").value
    app.sidebar.radio[0].set_value("Model Training & Evaluation").run()
    assert not app.exception
    assert sum("0% seizure recall" in w.value for w in app.warning) == 1
    report = bundle.payload["evaluation"]
    values = {m.label: m.value for m in app.metric}
    assert values["Recall"] == f"{report['recall']:.3f}"
    assert values["Specificity"] == f"{report['specificity']:.3f}"
    assert any("43 false negatives" in c.value for c in app.caption)
    assert load_baseline(CHECKPOINT).payload["evaluation"] == report


def test_saved_curves_use_existing_test_scores_and_preserve_values():
    scores = saved_curve_scores(ROOT, load_baseline(CHECKPOINT))
    assert scores is not None and len(scores) == 1800 and scores.label.sum() == 43
    before = scores.copy(deep=True)
    for kind in ("roc", "pr"):
        fig = discrimination_plot(scores, kind)
        assert len(fig.axes[0].lines) == 2
        assert fig.axes[0].get_xlim() == (0, 1)
    pd.testing.assert_frame_equal(scores, before)
    with pytest.raises(ValueError, match="Unknown curve"):
        discrimination_plot(scores, "unknown")
