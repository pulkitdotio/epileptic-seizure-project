"""Small, escaped presentation components. Widget layout stays in the app."""
from html import escape

import streamlit as st
from demo.design import SECTION_LABELS


CSS = """
<style>
.block-container {padding-top: 4.5rem; padding-bottom: 2rem; max-width: 1440px;}
[data-testid="stSidebar"] {min-width: 250px; max-width: 290px;}
[data-testid="stSidebar"] hr {margin: .75rem 0;}
[data-testid="stMetric"] {padding: .25rem 0 .75rem; border-bottom: 1px solid #29323d;}
[data-testid="stMetricLabel"] {color: #97a3b3;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
[data-testid="stCaptionContainer"] {color: #97a3b3;}
[data-testid="stCaptionContainer"] p {color: #97a3b3; font-size: .9rem; line-height: 1.5;}
[data-testid="stBaseButton-primary"]:not(:disabled),
[data-testid="stBaseButton-primary"]:not(:disabled) p {color: #0b0f14;}
[data-testid="stMultiSelectTagsContainer"] [data-tag] {background: #1b222c; border: 1px solid #29323d; color: #e8edf3;}
[data-testid="stMultiSelectTagsContainer"] [data-tag] span {color: #e8edf3;}
.eeg-brand {font-weight: 550; font-size: 1.15rem; color: #e8edf3;}
.eeg-brand small {display: block; font-size: .86rem; color: #97a3b3; font-weight: 400; margin-top: 3px;}
.eeg-header {border-bottom: 1px solid #29323d; padding-bottom: 1rem; margin-bottom: .25rem;}
.eeg-header h1 {font-size: 1.85rem; font-weight: 550; line-height: 1.3; padding: 0; margin: 0;}
.eeg-header p {color: #97a3b3; margin: .35rem 0 0; font-size: .92rem;}
.eeg-status {background: #151b23; border: 1px solid #29323d; border-left: 2px solid #97a3b3; border-radius: 7px; padding: 13px 16px; margin: 0;}
.eeg-status.alert {border-left-color: #e47878;}
.eeg-status.normal {border-left-color: #65b99a;}
.eeg-status .label {color: #97a3b3; font-size: .88rem; font-weight: 500;}
.eeg-status .state {color: #e8edf3; font-size: 1.08rem; font-weight: 550; margin: 7px 0; line-height: 1.4;}
.eeg-status.alert .state {color: #e47878;}
.eeg-status .detail {color: #e8edf3; font-size: .92rem; line-height: 1.6;}
.eeg-status .source {color: #97a3b3; font-size: .86rem; margin-top: 7px; line-height: 1.4;}
.eeg-step {padding: 12px 0; border-bottom: 1px solid #29323d;}
.eeg-step strong {color: #e8edf3; font-size: 1rem; font-weight: 550;}
.eeg-step p {color: #97a3b3; font-size: .92rem; margin: 6px 0 0; line-height: 1.5;}
.eeg-recording {color: #97a3b3; font-size: .92rem; margin: .25rem 0; overflow-wrap: anywhere;}
.st-key-playback {padding: .85rem !important;}
.st-key-preprocess_compare {margin-top: .5rem;}
@media (max-width: 1150px) {
 .st-key-preprocess_compare [data-testid="stHorizontalBlock"] {flex-direction: column;}
 .st-key-preprocess_compare [data-testid="stColumn"] {width: 100%; flex: 1 1 auto;}
}
@media (max-width: 700px) {.block-container {padding: 4.5rem 1rem 1rem;} .eeg-header h1 {font-size: 1.65rem;}}
</style>
"""


def initialize(title):
    st.set_page_config(page_title="EEG Analysis", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(section):
    st.markdown(f'<header class="eeg-header"><h1>{escape(SECTION_LABELS[section])}</h1>'
                '<p>Prerecorded EEG · Research workspace</p></header>', unsafe_allow_html=True)


def recording_line(filename, channels, sfreq, duration):
    st.markdown(f'<div class="eeg-recording">{escape(filename)} · {channels} channels · '
                f'{sfreq:g} Hz · {duration / 60:g} min</div>', unsafe_allow_html=True)


def status_card(label, state, details, source, tone="neutral"):
    if tone not in {"neutral", "alert", "normal"}:
        raise ValueError("Unknown status tone")
    lines = "<br>".join(escape(str(line)) for line in details)
    st.markdown(f'<div class="eeg-status {tone}"><div class="label">{escape(label)}</div>'
                f'<div class="state">{escape(state)}</div><div class="detail">{lines}</div>'
                f'<div class="source">{escape(source)}</div></div>', unsafe_allow_html=True)


def step_card(title, text):
    st.markdown(f'<div class="eeg-step"><strong>{escape(title)}</strong><p>{escape(text)}</p></div>', unsafe_allow_html=True)
