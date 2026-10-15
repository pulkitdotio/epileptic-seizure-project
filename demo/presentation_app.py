"""Working academic dashboard. Run from the project root with Streamlit."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from matplotlib import pyplot as plt

from src.component.presentation.pipeline import open_edf, read_window
from src.component.presentation.model import load_baseline
from demo.dashboard import run, SECTIONS

TITLE = "Advanced EEG Signal Analysis for Epileptic Seizure Detection and Early Prediction"


class RecordingSession:
    """Own lazy reader and optional upload file for the lifetime of one session."""
    def __init__(self, path, temporary=None):
        self.temporary = temporary
        try:
            self.raw = open_edf(path)
        except Exception:
            if temporary:
                temporary.cleanup()
            raise

    def close(self):
        if getattr(self, "raw", None) is not None:
            self.raw.close()
            self.raw = None
        if self.temporary:
            self.temporary.cleanup()
            self.temporary = None

    def __del__(self):
        self.close()


@st.cache_data(max_entries=12, ttl=600, show_spinner=False)
def signal_slice(source_id, _raw, channels, start, duration, config=None):
    return read_window(_raw, channels, start, duration, config)


@st.cache_resource(max_entries=2, show_spinner=False)
def checkpoint_bundle(path, signature):
    return load_baseline(path)


def show_figure(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def main():
    run(ROOT, TITLE, RecordingSession, signal_slice, checkpoint_bundle, show_figure)


if __name__ == "__main__":
    main()
