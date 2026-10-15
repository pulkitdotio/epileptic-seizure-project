"""Single app layout. Sidebar is rendered once and remains outside playback fragments."""
from pathlib import Path
import tempfile

import streamlit as st

from demo.security import validate_edf_upload, MAX_EDF_UPLOAD_BYTES
from demo.ui import initialize, status_card, step_card, page_header, recording_line
from demo.design import SECTION_LABELS
from demo.analysis_panel import render_analysis, annotation_panel
from demo.model_panel import render_training, scope_notice
from src.component.presentation.pipeline import annotations_for

SECTIONS = ["Overview / Dashboard", "Raw EEG Analysis", "EEG Preprocessing", "Seizure Annotations",
            "Segmentation", "AI Seizure Detection", "Model Training & Evaluation", "Early Prediction (Research)"]


def render_sidebar(root):
    """Exactly one navigation, source selector, EDF uploader and summary uploader per run."""
    with st.sidebar:
        st.markdown('<div class="eeg-brand">EEG Analysis<small>CHB-MIT research workspace</small></div>', unsafe_allow_html=True)
        section = st.radio("Navigation", SECTIONS, format_func=SECTION_LABELS.get, key="navigation", label_visibility="collapsed")
        st.divider()
        st.subheader("Recording")
        mode = st.radio("Choose source", ["Downloaded sample", "Upload EDF"], key="recording_source")
        uploaded, path, identity, filename = None, None, None, None
        if mode == "Downloaded sample":
            paths = sorted((root / "data/raw/chbmit").glob("**/*.edf"))
            if paths:
                default_index = next((i for i, p in enumerate(paths) if p.parent.name == "chb01" and p.name == "chb01_03.edf"), 0)
                path = st.selectbox("Local recording", paths, index=default_index,
                                    format_func=lambda p: f"{p.parent.name} / {p.name}", key="local_recording")
                identity = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
                filename = path.name
            else:
                st.info("No local EDF found. Select Upload EDF.")
        else:
            uploaded = st.file_uploader("Upload EEG recording (.edf)", type=["edf"], key="edf_upload")
            st.caption(f"EDF · up to {MAX_EDF_UPLOAD_BYTES // 1024**2} MiB")
            if uploaded is not None:
                identity = (uploaded.file_id, uploaded.name, uploaded.size)
                filename = uploaded.name
        with st.expander("Annotation summary"):
            summary_upload = st.file_uploader("Optional CHB-MIT summary (.txt)", type=["txt"], key="summary_upload")
        st.divider()
        with st.expander("Model checkpoint"):
            checkpoint = st.text_input("Baseline checkpoint path", value="results/presentation_baseline/baseline.pt", key="checkpoint_path")
        st.caption("Prerecorded data · No live acquisition")
    return section, mode, uploaded, path, identity, filename, summary_upload, checkpoint


def load_recording(source_id, source_path, uploaded, filename, session_class):
    if st.session_state.get("source_id") != source_id:
        previous = st.session_state.pop("recording_session", None)
        st.session_state.pop("source_id", None)
        if previous:
            previous.close()
        if uploaded is not None:
            content = uploaded.getvalue()
            validate_edf_upload(content, filename)
            temporary = tempfile.TemporaryDirectory(prefix="eeg_presentation_")
            source_path = Path(temporary.name) / "recording.edf"
            try:
                source_path.write_bytes(content)
            except Exception:
                temporary.cleanup()
                raise
            session = session_class(source_path, temporary)
        else:
            session = session_class(source_path)
        st.session_state["recording_session"] = session
        st.session_state["source_id"] = source_id
        for key in ["start_sec", "duration_sec", "selected_channels", "inference_result", "segment_id", "annotation_event"]:
            st.session_state.pop(key, None)
        st.session_state["playing"] = False
        st.session_state["playback_complete"] = False
    return st.session_state["recording_session"].raw


def research_view():
    st.info("Planned feature · No seizure forecasting model available")
    a, b, c = st.columns(3)
    with a:
        step_card("Interictal", "Background activity outside defined exclusion intervals.")
    with b:
        step_card("Preictal", "A defined interval before seizure onset.")
    with c:
        step_card("Ictal", "The ongoing seizure. Detection is separate from forecasting.")
    with st.expander("Research requirements"):
        st.write("A forecasting experiment needs repeated events across independent patients, defensible preictal labels, "
             "postictal and temporal exclusion gaps, a seizure prediction horizon and occurrence period, causal preprocessing, "
             "and event-based held-out evaluation with false alarms per hour.")
        st.caption("The current offline filter uses future context and cannot support a live forecasting claim.")


def open_signal(interval=None, duration=None):
    """Navigation shortcut only; preprocessing/model decisions are unchanged."""
    st.session_state["navigation"] = "Raw EEG Analysis"
    if interval is not None:
        onset, offset = interval
        width = min(120., duration, offset - onset + 10.)
        st.session_state["duration_sec"] = width
        st.session_state["start_sec"] = min(max(0., onset - 5.), duration - width)


def run(root, title, session_class, signal_slice, checkpoint_bundle, show_figure):
    initialize(title)
    section, mode, uploaded, path, source_id, filename, summary_upload, supplied = render_sidebar(root)
    page_header(section)
    if st.session_state.get("previous_section") != section:
        st.session_state["playing"] = False
        st.session_state["previous_section"] = section
    model_path = Path(supplied)
    if not model_path.is_absolute():
        model_path = root / model_path
    bundle, model_id, model_error = None, None, None
    if model_path.is_file():
        try:
            signature = (model_path.stat().st_mtime_ns, model_path.stat().st_size)
            model_id = (str(model_path.resolve()), *signature)
            bundle = checkpoint_bundle(str(model_path), signature)
        except (ValueError, OSError, RuntimeError) as exc:
            model_error = str(exc)
    if section == "Model Training & Evaluation":
        render_training(root, bundle, show_figure)
        if model_error:
            st.error(f"Checkpoint unavailable: {model_error}")
        return
    if section == "Early Prediction (Research)":
        research_view()
        return
    if source_id is None:
        st.session_state["playing"] = False
        st.info("Select a downloaded recording or upload an EDF to begin.")
        return
    try:
        raw = load_recording(source_id, path, uploaded, filename, session_class)
    except (ValueError, OSError, RuntimeError) as exc:
        st.session_state["playing"] = False
        st.error(f"Unable to load EEG: {exc}")
        return
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    intervals, annotation_source = None, "Unavailable"
    try:
        text = None
        if summary_upload is not None:
            if summary_upload.size > 2 * 1024**2:
                raise ValueError("Summary file exceeds 2 MiB.")
            text = summary_upload.getvalue().decode("utf-8-sig")
            annotation_source = summary_upload.name
        elif mode == "Downloaded sample":
            summary_path = path.parent / f"{path.parent.name}-summary.txt"
            if summary_path.is_file():
                text = summary_path.read_text(encoding="utf-8-sig")
                annotation_source = summary_path.name
        if text is not None:
            intervals = annotations_for(text, filename, duration)
    except (ValueError, UnicodeError, OSError) as exc:
        st.warning(f"Annotations unavailable: {exc}")
    recording_line(filename, len(raw.ch_names), sfreq, duration)
    if intervals is None:
        st.info("Annotations unavailable. Upload the matching summary to label windows.")
    threshold = bundle.threshold if bundle else .5
    if section == "Overview / Dashboard":
        for col, (label, value) in zip(st.columns(4), [("Channels", len(raw.ch_names)),
                ("Sampling frequency", f"{sfreq:g} Hz"), ("Duration", f"{duration / 60:g} min"),
                ("Annotated seizures", len(intervals) if intervals is not None else "Unknown")]):
            col.metric(label, value)
        if bundle:
            scope_notice(bundle)
        a, b = st.columns(2)
        with a:
            status_card("Dataset annotations", f"{len(intervals)} annotated seizure{'s' if len(intervals) != 1 else ''}" if intervals is not None else "Annotations unavailable",
                        [f"{s:g}–{e:g} s · {e-s:g} seconds" for s, e in intervals or []] or ["No seizure intervals available."],
                        f"Source: CHB-MIT · {annotation_source}", "neutral")
        with b:
            status_card("CNN model", "BaselineCNN1D · Available" if bundle else "Model not trained" if not model_error else "Checkpoint unavailable",
                        [f"{len(bundle.channels)} channels · {bundle.window_sec:g}-second windows · Threshold {bundle.threshold:.6f}" if bundle else model_error or "Load a compatible trained checkpoint."],
                        "Source: checkpoint status · Predictions shown in AI Detection", "neutral")
        a, b = st.columns(2)
        with a:
            st.button("View raw EEG", on_click=open_signal, width="stretch", type="primary", key="overview_signal")
        with b:
            st.button("View annotated seizure", on_click=open_signal,
                      args=(intervals[0], duration) if intervals else (None, None),
                      disabled=not bool(intervals), width="stretch", key="overview_seizure")
        with st.expander("Project details"):
            st.write(title)
            st.caption("CHB-MIT scalp EEG · Offline academic prototype")
            st.write("Raw EEG → Preprocessing → Segmentation → Annotations → CNN detection")
            st.caption("Early prediction is planned. No forecasting probabilities are available.")
        return
    if section == "AI Seizure Detection":
        if bundle:
            scope_notice(bundle)
            with st.expander("Decision threshold", expanded=False):
                override = st.checkbox("Use exploratory display threshold", key=f"override_{model_id}")
                displayed = st.slider("Display decision threshold", .001, .999, min(.999, max(.001, bundle.threshold)), .001,
                                  key=f"threshold_{model_id}", disabled=not override)
                st.caption(f"Saved threshold {bundle.threshold:.6f}. Display overrides leave saved evaluation unchanged.")
            threshold = displayed if override else bundle.threshold
            if override:
                st.warning("Exploratory threshold override: saved validation/test false-positive rates do not apply to these display decisions.")
        elif model_error:
            st.error(f"Checkpoint unavailable: {model_error}")
        else:
            st.warning("AI model not trained — prediction unavailable")
        # Timer reruns only this body. No sidebars, file selectors, or uploaders occur inside it.
    playback_enabled = section in {"Raw EEG Analysis", "AI Seizure Detection"} and st.session_state.get("playing", False)
    st.fragment(run_every=1. if playback_enabled else None)(render_analysis)(
        section, raw, source_id, filename, intervals, bundle, model_id, model_error, threshold, signal_slice, show_figure)
