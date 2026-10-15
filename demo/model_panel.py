"""Real model monitoring, saved evaluation, and a reviewable data/training plan."""
from pathlib import Path
import json

import pandas as pd
import numpy as np
import streamlit as st
import torch

from demo.ui import status_card
from demo.plots import probability_plot, confusion_plot, training_curves, discrimination_plot
from src.component.presentation.model import predict_windows, check_recording
from src.component.presentation.pipeline import window_index
from src.component.presentation.monitoring import apply_threshold, compare_annotations, cnn_status


def ai_status(result=None, threshold=None, unavailable=None, scope=None, evaluation=None):
    if result is None:
        status_card("CNN prediction", "Model not trained" if unavailable is None else "Inference unavailable",
                    [unavailable or "Load a compatible trained checkpoint."],
                    "Source: CNN checkpoint required · prediction unavailable")
        return
    row = result.iloc[-1]
    status = cnn_status(float(row.seizure_probability), threshold)
    alert = status["alert"]
    status_card("CNN prediction", status["state"],
                [f"Model probability: {row.seizure_probability:.6f} · Decision threshold: {threshold:.6f}",
                 f"Model window: {row.start_sec:g}–{row.end_sec:g} s",
                 "Score ≥ threshold." if alert else "Below threshold does not rule out a seizure."],
                "Source: trained CNN inference · BaselineCNN1D · Available", "alert" if alert else "neutral")
    if scope == "fit-only":
        st.caption("Unevaluated fit-only model: these predictions may reflect training-recording memorization.")


def scope_notice(bundle):
    scope = bundle.payload["training"]["evaluation_scope"]
    if scope == "fit-only":
        st.warning("Fit-only model · No independent evaluation; predictions may reflect memorization.")
    elif scope == "recording-held-out":
        st.warning("Recording-held-out baseline · Patients may be shared between splits.")
    report = bundle.payload.get("evaluation")
    if report and report.get("positive_windows", 0) > 0 and report.get("sensitivity") == 0:
        st.warning(f"Research baseline · 0% seizure recall on held-out test data · {report['positive_windows']} missed seizure windows")


def render_inference(raw, source_id, intervals, bundle, model_id, threshold, show_figure):
    st.subheader("Window predictions")
    if bundle is None:
        st.info("Inference unavailable until a compatible trained checkpoint is loaded.")
        return
    try:
        check_recording(bundle, raw)
        cfg = bundle.preprocessing
        with st.expander("Model input and score interpretation"):
            st.caption(f"BaselineCNN1D · {cfg.low_hz:g}–{cfg.high_hz:g} Hz · {cfg.normalization} · "
                       "Saved channel order and sampling rate. Display filtering does not change model inputs.")
            st.caption("Softmax output is an uncalibrated model score. Annotation labels do not determine CNN alerts.")
        windows = window_index(raw.n_times, raw.info["sfreq"], bundle.window_sec, intervals=None)
        if windows.empty:
            st.info("Recording is shorter than one model window.")
            return
        a, b = st.columns(2)
        first = a.number_input("First model window", 0, len(windows) - 1,
                               min(int(st.session_state.get("start_sec", 0) / bundle.window_sec), len(windows) - 1))
        count = b.number_input("Windows to infer (maximum 300)", 1, min(300, len(windows) - first),
                               min(30, len(windows) - first))
        request_id = (source_id, model_id, int(first), int(count))
        if st.button("Run CPU inference", type="primary", key="run_inference"):
            torch.set_num_threads(2)
            progress = st.progress(0., text="Reading and processing bounded EEG batches")
            result = predict_windows(bundle, raw, windows.iloc[first:first + count], progress=progress.progress)
            progress.empty()
            st.session_state["inference_result"] = (request_id, result)
        stored = st.session_state.get("inference_result")
        if stored and stored[0] == request_id:
            result = compare_annotations(apply_threshold(stored[1], threshold), intervals)
            positives = int(result.prediction.sum())
            st.caption(f"{positives} / {len(result)} CNN-alert windows · Threshold {threshold:.6f} · Card above shows its current window")
            show_figure(probability_plot(result, threshold, intervals))
            with st.expander("Prediction table", expanded=False):
                st.dataframe(result, width="stretch", hide_index=True)
                st.caption("Annotation label: ≥50% coverage. Viewed agreement may include training data; it is not an independent test score.")
            st.download_button("Download model outputs (CSV)", result.to_csv(index=False), "model_outputs.csv", "text/csv")
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Inference unavailable: {exc}")


def render_cohort(root, bundle):
    with st.expander("Dataset and training workflow", expanded=bundle is None):
        manifest = json.loads((root / "configs/presentation_subset.json").read_text(encoding="utf-8"))
        rows = [{"Recording": r["path"], "Patient": r["path"].split("/")[0], "Split": r["split"],
                 "Annotated events": len(r["events"]), "MiB": round(r["size_bytes"] / 1024**2, 2),
                 "Local status": "Present" if (root / "data/raw/chbmit" / r["path"]).exists() else "Missing"}
                for r in manifest["recordings"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pending = sum(r["size_bytes"] for r in manifest["recordings"] if not (root / "data/raw/chbmit" / r["path"]).exists())
        st.caption("Four patient groups · Ten recordings · Six annotated seizures")
        if pending:
            st.write(f"Missing EDF transfer: **{pending / 1024**2:.1f} MiB**, plus small summaries. Approval required before transfer.")
        else:
            st.caption("All selected recordings are present.")
        st.caption("Train: chb01 / chb03 · Validation: chb05 · Test: chb08. The small cohort cannot establish generalization.")
        run_path = "results/presentation_baseline_run2" if (root / "results/presentation_baseline").exists() else "results/presentation_baseline"
        st.code("python -m src.component.presentation.download_subset --manifest configs/presentation_subset.json --estimate\n"
                "python -m src.component.presentation.download_subset --manifest configs/presentation_subset.json --download --max-mib 400\n"
                f"python -m src.component.presentation.train --dataset-manifest configs/presentation_subset.json --split-file configs/presentation_splits.json --output {run_path} --epochs 5 --batch-size 16 --threads 2 --preflight\n"
                f"python -m src.component.presentation.train --dataset-manifest configs/presentation_subset.json --split-file configs/presentation_splits.json --output {run_path} --epochs 5 --batch-size 16 --threads 2 --max-minutes 15", language="bat")
        st.caption("Reference commands only. The dashboard does not launch training or downloads.")


def saved_curve_scores(root, bundle):
    """Read paired saved artifacts for presentation, never run evaluation/inference."""
    path = Path(st.session_state.get("checkpoint_path", "results/presentation_baseline/baseline.pt"))
    if not path.is_absolute():
        path = root / path
    config_path, scores_path = path.parent / "configuration.json", path.parent / "test_predictions.csv"
    if not config_path.is_file() or not scores_path.is_file():
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if (cfg.get("training", {}).get("created_utc") != bundle.payload["training"].get("created_utc")
                or any(cfg.get(k) != bundle.payload[k] for k in ["channels", "window_sec", "preprocessing"])):
            return None
        scores = pd.read_csv(scores_path, float_precision="round_trip", usecols=["label", "seizure_probability"])
        report = bundle.payload.get("evaluation", {})
        if (len(scores) != report.get("n_windows") or scores.label.sum() != report.get("positive_windows")
                or set(scores.label.unique()) != {0, 1} or not np.isfinite(scores.to_numpy()).all()
                or not scores.seizure_probability.between(0, 1).all()):
            return None
        return scores
    except (ValueError, OSError, KeyError):
        return None


def render_training(root, bundle, show_figure):
    if bundle is None:
        st.info("No trained checkpoint available. Load a model to view its saved evaluation.")
        render_cohort(root, bundle)
        return
    scope_notice(bundle)
    provenance = bundle.payload["training"]
    report = bundle.payload.get("evaluation")
    st.caption(f"BaselineCNN1D · {provenance['evaluation_scope']} · Saved threshold {bundle.threshold:.6f}")
    if report:
        st.subheader("Saved test results")
        for col, metric in zip(st.columns(4), ["recall", "precision", "f1", "specificity"]):
            value = report.get(metric)
            label = {"f1": "F1-score"}.get(metric, metric.title())
            col.metric(label, "Unavailable" if value is None else f"{value:.3f}")
        tn, fp = report["confusion_matrix"][0]
        fn, tp = report["confusion_matrix"][1]
        st.caption(f"{report['n_windows']:,} windows · {report['positive_windows']} seizure windows · "
                   f"{fn} false negatives · {fp} false positives")
        chart, counts = st.columns([1.5, 1])
        with chart:
            show_figure(confusion_plot(report["confusion_matrix"]))
        with counts:
            st.dataframe(pd.DataFrame({"Outcome": ["True negative", "False positive", "False negative", "True positive"],
                                       "Windows": [tn, fp, fn, tp]}), hide_index=True, width="stretch")
            st.caption("Previously examined test cohort · Preliminary results")
            st.caption("High accuracy can reflect class imbalance. These results are not for the currently viewed EDF.")
        with st.expander("ROC and precision–recall curves"):
            scores = saved_curve_scores(root, bundle)
            if scores is None:
                st.info("Compatible saved test scores unavailable. Curves cannot be displayed.")
            else:
                with st.container(key="preprocess_compare"):
                    left, right = st.columns(2)
                    with left:
                        show_figure(discrimination_plot(scores, "roc"))
                    with right:
                        show_figure(discrimination_plot(scores, "pr"))
                st.caption("Curves use saved test scores; no new inference or threshold selection.")
        with st.expander("Additional metrics and interpretation"):
            for col, metric in zip(st.columns(4), ["accuracy", "sensitivity", "roc_auc", "pr_auc"]):
                value = report.get(metric)
                label = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC (AP)"}.get(metric, metric.title())
                col.metric(label, "Unavailable" if value is None else f"{value:.3f}")
            st.caption(f"Frozen evaluation threshold {report['threshold']:.6f}. Display threshold changes never alter this report. "
                       "Sensitivity equals recall. PR-AUC is average precision; AUC is undefined for a single class.")
            st.caption("Window false positives are not event-level false alarms/hour. This small, previously examined cohort is exploratory; clinical validation is unavailable.")
    else:
        st.info("No independent test report is available for this checkpoint.")
    history = bundle.payload.get("history")
    with st.expander("Training history"):
        if history:
            show_figure(training_curves(pd.DataFrame(history)))
            st.caption(f"Training objective: {provenance.get('loss', 'weighted-ce')} · Validation loss: class-balanced cross-entropy")
        else:
            st.caption("Training history was not saved in this checkpoint.")
    with st.expander("Model input and training provenance"):
        st.json({"channels": bundle.channels, "window_sec": bundle.window_sec,
                 "preprocessing": bundle.preprocessing.to_dict(), "saved_threshold": bundle.threshold,
                 "training": provenance})
    records = provenance.get("records", [])
    if records:
        distribution = []
        for split in ("train", "val", "test"):
            selected = [r for r in records if r["split"] == split]
            counts = provenance.get("class_counts_by_split", {}).get(split)
            distribution.append({"Split": split, "Unique patient groups": len({r["patient"] for r in selected}),
                                 "Recordings": len(selected), "Windows": provenance["split_window_counts"][split],
                                 "Non-seizure windows": counts[0] if counts else "Not saved in older checkpoint",
                                 "Seizure windows": counts[1] if counts else "Not saved in older checkpoint"})
        with st.expander("Patient and window distribution"):
            st.dataframe(pd.DataFrame(distribution), hide_index=True, width="stretch")
            st.caption(f"Training used {provenance['train_windows_used']} windows; class counts {provenance['class_counts_train']}. "
                       "Validation/test keep their natural imbalance.")
    render_cohort(root, bundle)
