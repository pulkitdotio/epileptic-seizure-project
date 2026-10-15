"""Bounded EEG views and timer-driven OFFLINE playback; never renders a sidebar."""
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from demo.ui import status_card
from demo.plots import waveform
from demo.model_panel import ai_status, render_inference, scope_notice
from src.component.presentation.pipeline import PreprocessConfig, window_index
from src.component.presentation.model import check_recording, predict_windows
from src.component.presentation.monitoring import annotation_status, advance_playback, apply_threshold, compare_annotations


@st.cache_data(max_entries=64, ttl=600, show_spinner=False)
def model_window(source_id, model_id, _raw, _bundle, start):
    """One real CNN window; cache key includes source and checkpoint file signature."""
    torch.set_num_threads(2)
    width = _bundle.window_sec
    frame = pd.DataFrame({"start_sec": [start], "end_sec": [start + width]})
    return predict_windows(_bundle, _raw, frame, batch_size=1)


def annotation_panel(filename, start, end, intervals):
    status = annotation_status(start, end, intervals)
    if not status["available"]:
        state, tone = "Annotations unavailable", "neutral"
        detail = ["Upload the matching CHB-MIT summary."]
    elif status["alert"]:
        state, tone = "Annotated seizure", "alert"
        detail = [f"{s:g}–{e:g} s · {e - s:g} seconds" for s, e in status["events"]]
    else:
        state, tone = "No annotated seizure", "neutral"
        detail = [f"Selected interval {start:g}–{end:g} s"]
    status_card("Dataset annotation", state, detail,
                "Source: CHB-MIT ground-truth annotations", tone)


def render_playback(duration, step):
    st.caption(f"Prerecorded EEG · {step:g} s per refresh · Accelerated playback")
    a, b, c, d, e = st.columns(5)
    playing = st.session_state.get("playing", False)
    if a.button("Start", key="play_start", disabled=playing, width="stretch"):
        st.session_state["playing"] = True
        st.session_state["playback_complete"] = False
        st.session_state["playback_tick"] = time.monotonic()
        st.session_state["duration_sec"] = min(step, duration)
        st.session_state["start_sec"] = min(st.session_state.get("start_sec", 0.), max(0., duration - step))
        st.rerun()  # Full rerun installs the timer; sidebar is rendered once.
    if b.button("Pause", key="play_pause", disabled=not playing, width="stretch"):
        st.session_state["playing"] = False
        st.rerun()
    if c.button("Reset", key="play_reset", width="stretch"):
        st.session_state["playing"] = False
        st.session_state["playback_complete"] = False
        st.session_state["start_sec"] = 0.
        st.rerun()
    if d.button("Previous", key="play_previous", disabled=playing, width="stretch", help="Step back one window"):
        st.session_state["start_sec"] = max(0., st.session_state.get("start_sec", 0.) - step)
    if e.button("Next", key="play_next", disabled=playing, width="stretch", help="Step forward one window"):
        st.session_state["start_sec"] = min(max(0., duration - st.session_state.get("duration_sec", min(10., duration))),
                                            st.session_state.get("start_sec", 0.) + step)
    if playing and time.monotonic() - st.session_state.get("playback_tick", time.monotonic()) >= 1.:
        start, finished = advance_playback(st.session_state.get("start_sec", 0.), duration, step)
        st.session_state["start_sec"] = start
        st.session_state["playback_tick"] = time.monotonic()
        if finished:
            st.session_state["playing"] = False
            st.session_state["playback_complete"] = True
            st.rerun()
    state = "Playing" if st.session_state.get("playing", False) else "Finished" if st.session_state.get("playback_complete", False) else "Paused"
    st.caption(f"{state} · Position {st.session_state.get('start_sec', 0.):g} / {duration:g} s")


def render_analysis(section, raw, source_id, filename, intervals, bundle, model_id, model_error, threshold, signal_slice, show_figure):
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    if bundle and section == "AI Seizure Detection":
        record = next((r for r in bundle.payload["training"].get("records", []) if Path(r["relative_path"]).name == filename), None)
        if record:
            st.caption(f"Recording split: {record['split']} · Viewed agreement is not an independent test score.")
    if section == "Seizure Annotations" and intervals is not None:
        st.dataframe(pd.DataFrame([{"Onset (s)": s, "Offset (s)": e, "Duration (s)": e - s} for s, e in intervals]),
                     hide_index=True, width="stretch")
        if not intervals:
            st.success("The summary explicitly reports no seizures in this recording.")
    if intervals:
        a, b = st.columns([3, 1], vertical_alignment="bottom")
        event = a.selectbox("Annotated event", range(len(intervals)), key="annotation_event",
                            format_func=lambda i: f"Event {i+1}: {intervals[i][0]:g}–{intervals[i][1]:g} s")
        if b.button("Jump to seizure", key="jump_event", width="stretch"):
            onset, offset = intervals[event]
            display_duration = min(120., duration, offset - onset + 10.)
            st.session_state["duration_sec"] = display_duration
            st.session_state["start_sec"] = min(max(0., onset - 5.), duration - display_duration)
            if st.session_state.get("playing", False):
                st.session_state["playing"] = False
                st.rerun()
    if section in {"Raw EEG Analysis", "AI Seizure Detection"}:
        with st.expander("Recording playback", expanded=False):
            with st.container(key="playback"):
                render_playback(duration, bundle.window_sec if bundle else min(4., duration))
    with st.expander("Signal view", expanded=section not in {"AI Seizure Detection", "Segmentation"}):
        a, b = st.columns([3, 1])
        channels = a.multiselect("EEG channels (up to 12)", raw.ch_names, default=raw.ch_names[:5], max_selections=12, key="selected_channels")
        if "duration_sec" not in st.session_state:
            st.session_state["duration_sec"] = min(10., duration)
        display_duration = b.number_input("Displayed duration (seconds)", min_value=min(.25, duration), max_value=min(120., duration),
                                       value=None, step=.25, key="duration_sec", disabled=st.session_state.get("playing", False))
        max_start = max(0., duration - display_duration)
        st.session_state["start_sec"] = min(st.session_state.get("start_sec", 0.), max_start)
        start = st.slider("Start time (seconds)", 0., max_start, step=.25, key="start_sec",
                       disabled=st.session_state.get("playing", False)) if max_start > 0 else 0.
        with st.expander("Waveform spacing"):
            spacing = st.select_slider("Waveform spacing (µV)", [100, 250, 500, 1000, 2000], value=1000, key="spacing")
    if not channels:
        st.info("Select at least one EEG channel.")
        return
    try:
        if section == "Segmentation":
            a, b = st.columns(2)
            width = a.selectbox("Window duration (seconds)", [2., 4., 8., 10.], index=1)
            overlap = b.selectbox("Window overlap", ["None", "50%"])
            windows = window_index(raw.n_times, sfreq, width, width if overlap == "None" else width / 2, intervals)
            st.caption("Seizure label: ≥50% annotated coverage · Partial windows excluded")
            a, b, c = st.columns(3)
            a.metric("Complete windows", len(windows))
            b.metric("Seizure windows", "Unknown" if intervals is None else int((windows.label == 1).sum()))
            c.metric("Non-seizure windows", "Unknown" if intervals is None else int((windows.label == 0).sum()))
            st.download_button("Download window labels (CSV)", windows.to_csv(index=False), "window_labels.csv", "text/csv")
            category = st.selectbox("Window preview category", ["All", "Seizure", "Non-seizure"])
            candidates = windows if category == "All" else windows[windows.label == (1 if category == "Seizure" else 0)]
            if candidates.empty:
                st.info("No complete windows in this category.")
                return
            chosen = st.selectbox("Segment to inspect", candidates.index.tolist(),
                                  format_func=lambda i: f"#{i}: {windows.loc[i, 'start_sec']:g}–{windows.loc[i, 'end_sec']:g} s · label {windows.loc[i, 'label']}")
            with st.expander("Window labels and labeling policy"):
                st.dataframe(candidates.head(100), hide_index=True, width="stretch")
                st.caption("Complete half-open windows; labels use seizure-union coverage. Missing annotations remain unknown. "
                           "Overlapping display windows are never an independent train/test split.")
            start, display_duration = float(windows.loc[chosen, "start_sec"]), width
        def statuses():
            left, right = st.columns(2)
            with left:
                annotation_panel(filename, start, start + display_duration, intervals)
            with right:
                render_cnn_status()

        def render_cnn_status():
            if bundle is None:
                ai_status(unavailable=model_error)
            else:
                try:
                    check_recording(bundle, raw)
                    if duration < bundle.window_sec:
                        raise ValueError("Recording is shorter than the model window.")
                    model_start = min(start, duration - bundle.window_sec)
                    result = compare_annotations(apply_threshold(model_window(source_id, model_id, raw, bundle, model_start), threshold), intervals)
                    ai_status(result, threshold, scope=bundle.payload["training"]["evaluation_scope"], evaluation=bundle.payload.get("evaluation"))
                    if intervals is not None:
                        row = result.iloc[0]
                        st.caption(f"Window annotation: {'seizure' if row.annotation_label else 'non-seizure'} · "
                                   f"{'Agrees with' if row.agreement else 'Differs from'} CNN")
                except (ValueError, OSError, RuntimeError) as exc:
                    ai_status(unavailable=str(exc))
        times, values = signal_slice(source_id, raw, tuple(channels), start, display_duration)
        if section == "AI Seizure Detection":
            statuses()
        if section == "EEG Preprocessing":
            a, b, c = st.columns(3)
            low = a.number_input("Low cutoff (Hz)", .1, max(.1, sfreq / 2 - .2), 1., .5)
            high = b.number_input("High cutoff (Hz)", .2, max(.2, sfreq / 2 - .1), min(40., sfreq / 2 - .1), 1.)
            normalize = c.selectbox("Normalization", ["none", "zscore"])
            cfg = PreprocessConfig(sfreq, low, high, normalize)
            _, filtered = signal_slice(source_id, raw, tuple(channels), start, display_duration, cfg)
            layout = st.radio("Comparison layout", ["Side by side", "Stacked"], horizontal=True, key="comparison_layout")
            with st.container(key="preprocess_compare"):
                if layout == "Side by side":
                    left, right = st.columns(2)
                    with left:
                        show_figure(waveform(times, values, channels, intervals, "Raw EEG", spacing=spacing, compact=True))
                    with right:
                        show_figure(waveform(times, filtered, channels, intervals, "Processed EEG", standardized=normalize == "zscore", spacing=spacing, compact=True))
                else:
                    show_figure(waveform(times, values, channels, intervals, "Raw EEG", spacing=spacing))
                    show_figure(waveform(times, filtered, channels, intervals, "Processed EEG", standardized=normalize == "zscore", spacing=spacing))
            st.caption(f"Applied: {low:g}–{high:g} Hz bandpass · {'Per-channel z-score' if normalize == 'zscore' else 'Amplitude preserved'}")
            with st.expander("Processing details"):
                st.write(f"Fourth-order Butterworth {low:g}–{high:g} Hz bandpass; zero-phase forward/backward filtering "
                     f"with up to two seconds of context on each side; {'per-channel z-score after cropping' if normalize == 'zscore' else 'amplitude retained in microvolts'}. "
                     "Absolute timestamps and sampling frequency are preserved.")
                st.caption("Offline filtering uses future context. These display controls do not change saved model preprocessing.")
        else:
            show_figure(waveform(times, values, channels, intervals, "Raw EEG", spacing=spacing))
        if section != "AI Seizure Detection":
            with st.expander("Annotation & CNN status", expanded=section == "Seizure Annotations"):
                if bundle:
                    scope_notice(bundle)
                statuses()
        if section == "AI Seizure Detection":
            render_inference(raw, source_id, intervals, bundle, model_id, threshold, show_figure)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Unable to process selected signal: {exc}")
