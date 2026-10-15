# Frontend refinement and verification

The existing Streamlit dashboard now uses a compact dark analysis layout. This
phase changes presentation only. Product Design's context and audit guidance was
used to review the workflow and inspect the implemented screens.

## Changes

- Centralized neutral surfaces, typography, borders and restrained semantic colors.
- Short navigation labels for all eight existing sections; existing navigation state values retained.
- Compact page headers and overview metrics, active-recording summary, signal/seizure shortcuts.
- Grouped channel/time/spacing controls; optional playback, summary and checkpoint controls in expanders.
- Dark thin-waveform plots, transparent red annotation intervals with actual onset/offset labels.
- Raw/processed comparison side by side or stacked; responsive chart columns.
- Separate dataset annotation and genuine CNN prediction panels with actual probability, model window, threshold and source.
- One concise visible zero-recall warning per model/result section; missed seizure windows remain explicit.
- Compact saved-test metrics, confusion matrix/outcome table, ROC/precision–recall plots from existing saved scores.
- Secondary processing, evaluation and provenance explanations collapsed by default.
- Prerecorded playback retains Start, Pause, Reset, Previous, Next and position/state.

Visual refinements after inspection corrected header clipping beneath Streamlit's
toolbar, reduced the confusion matrix's enlarged typography, improved supporting
text contrast, reduced selected-channel color saturation and improved primary-button
text contrast. Playback starts collapsed to prioritize the waveform.

## Files changed in this phase

| File | Purpose |
| --- | --- |
| `.streamlit/config.toml` | Supported theme, type scale and semantic colors |
| `demo/design.py` | Shared colors and displayed section names (new) |
| `demo/ui.py` | Compact headers, status components and scoped CSS |
| `demo/dashboard.py` | Sidebar/overview layout and navigation shortcuts |
| `demo/analysis_panel.py` | Grouped controls, comparison layout and playback presentation |
| `demo/model_panel.py` | Concise CNN status and saved evaluation layout |
| `demo/plots.py` | Scientific dark chart styling and saved-score curve visualization |
| `src/tests/test_presentation_dashboard.py` | Navigation assertions scoped to sidebar |
| `src/tests/test_presentation_monitoring.py` | Updated visible-copy/color assertions; sidebar selectors |
| `src/tests/test_presentation_design.py` | Three new frontend regressions using existing trained artifacts |
| `README.md`, `demo/README.md`, `PRESENTATION_PROGRESS.md` | Current UI labels, workflow and verification |
| `reports/frontend_polish/` | QA screenshots, browser checks, test results, preserved source and integrity evidence |

`demo/presentation_app.py` is byte-for-byte unchanged from this phase's source
snapshot. No packages were installed or upgraded. SHA-256 verification confirms
all **182 protected files** are unchanged: backend Python sources, dataset/split
configuration, CHB-MIT data/annotations, checkpoints and saved results. See
[integrity evidence](protected_after.json).

## Browser verification

The dashboard was launched locally on port 8503. After explicit approval, an isolated
headless Chromium profile captured only the local dashboard; existing Chrome tabs
were not used. Native-browser capture initially encountered an approval rejection
because the selected Chrome window contained unrelated private content. The isolated
browser provided the subsequent visual evidence. There is no accepted before-change
browser screenshot; original frontend source is retained in `before_code/`.

All eight sections were captured and visually reviewed at **1366×768** and
**1920×1080**, including lower-page chart views. The [browser checks](browser_checks.json)
record correct headings, loaded chart images, no page-wide horizontal overflow,
no displayed app errors and headers below the fixed toolbar.

| Section | 1366×768 | 1920×1080 |
| --- | --- | --- |
| Overview | [Screenshot](screenshots/overview_1366x768.png) | [Screenshot](screenshots/overview_1920x1080.png) |
| Raw EEG | [Screenshot](screenshots/raw_1366x768.png) | [Screenshot](screenshots/raw_1920x1080.png) |
| Preprocessing | [Comparison](screenshots/preprocessing_lower_1366x768.png) | [Screenshot](screenshots/preprocessing_1920x1080.png) |
| Seizure Annotations | [Screenshot](screenshots/annotations_lower_1366x768.png) | [Screenshot](screenshots/annotations_1920x1080.png) |
| Segmentation | [Screenshot](screenshots/segmentation_1366x768.png) | [Screenshot](screenshots/segmentation_1920x1080.png) |
| AI Detection | [Screenshot](screenshots/ai_1366x768.png) | [Screenshot](screenshots/ai_1920x1080.png) |
| Training & Evaluation | [Screenshot](screenshots/evaluation_1366x768.png) | [Screenshot](screenshots/evaluation_1920x1080.png) |
| Early Prediction | [Screenshot](screenshots/early_1366x768.png) | [Screenshot](screenshots/early_1920x1080.png) |

Additional browser interactions verified:

- Actual upload of the existing 40.4 MiB EDF: 23 channels, 256 Hz, 60 minutes.
- Upload of its matching summary restores the annotated event; missing-summary state is explicit.
- Jump to seizure opens 2991–3041 seconds; plot shading identifies 2996–3036 seconds.
- Start advances prerecorded windows, Pause stops, Reset returns to zero, Next/Previous step 0→4→0 seconds.
- Browser CSV export was downloaded and checked: [900 windows, 10 seizure / 890 non-seizure](qa_exports/window_labels.csv).
- Existing CPU-inference button runs 30 windows starting at index 747; actual original-model scores remain below 0.50 and the UI reports **0 / 30 CNN-alert windows**.
- [Probability/annotation chart](screenshots/ai_outputs_1366x768.png) and [saved-score curves](screenshots/evaluation_curves_1366x768.png) were inspected.

The CPU-inference check exercises the existing dashboard control; it is not a new
training or threshold experiment. It only creates session results. Existing prediction
files and evaluation artifacts remain unchanged. No training, downloads or threshold
optimization was performed.

## Automated checks

**32 passed, 1 skipped, 0 failed; 6 deliberately deselected.**
The optional safetensors-sidecar test was skipped because that package is absent.
Six selected-file tests that invoke training fixtures were deselected to respect
the frontend-only instruction. This is a targeted verification, not a new claim
that the full historical 77-test suite was rerun. The three new frontend regressions
are included. [JUnit output](tests.xml) records the executed cases.

Coverage includes EDF validation/lazy reads, upload lifetime and recovery, annotations,
window labeling/counts, preprocessing consistency, eight-section navigation,
playback, dark plots, alert semantics, saved metrics/default threshold preservation,
overview shortcuts, comparison layout and paired saved-score curves.

```bat
.venv\Scripts\python.exe -m pytest src/tests/test_demo_security.py src/tests/test_presentation_pipeline.py src/tests/test_presentation_dashboard.py src/tests/test_presentation_monitoring.py src/tests/test_presentation_design.py src/tests/test_presentation_detection.py::test_ai_alert_independent_of_annotation_and_invalid_scores src/tests/test_presentation_detection.py::test_ai_card_uses_probability_not_stale_prediction_flag -q -k "not real_training and not checkpoint_missing and not trained and not preflight and not deadline and not monitor_reads"
```

## Remaining limits and launch

No blocking frontend issue was observed at the two tested sizes. Expanded details
and long signal/result pages still require normal vertical scrolling, particularly
at 768 pixels high. Screen-reader and full accessibility certification were not
performed. Native Streamlit controls, textual alert states and focus behavior are retained.

The CNN's **0% held-out recall / 43 false negatives** is unchanged. Early forecasting
remains planned. Visual polish does not improve scientific performance.

Restart an already running Streamlit process to reload theme and imported UI modules.
From the project root, run:

```bat
.venv\Scripts\python.exe -m streamlit run demo/presentation_app.py
```

For the presentation, use Overview → View annotated seizure, compare preprocessing,
inspect segmentation, then show the honest AI and saved-evaluation results.
