# Demo — working EEG presentation dashboard

Use the local academic dashboard for offline EEG exploration and genuine model inference when compatible trained weights exist.

The approved four-patient CPU baseline is available at
`results/presentation_baseline/baseline.pt` and loads by default. It missed all 43
held-out seizure windows at threshold 0.50; the dashboard displays this limitation.
Its high accuracy reflects class imbalance and does not establish useful detection.
See [the real results report](../reports/presentation_checks/REAL_BASELINE_RESULTS.md).

## Run

```bat
python -m streamlit run demo/presentation_app.py
# Opens at http://localhost:8501
```

## Features

- Upload `.edf` EEG files (validated: `.edf` extension, EDF header, 50 MB max)
- Channel/time selection and multichannel EEG visualization
- Configurable bandpass filtering and optional z-score normalization
- CHB-MIT annotation parsing, seizure highlighting, window labeling and CSV export
- CPU CNN inference only with a compatible trained presentation checkpoint
- Genuine held-out evaluation report only when available
- Compact dark theme with grouped controls and one sidebar
- Side-by-side or stacked preprocessing comparison
- Separate ground-truth and genuine trained-CNN monitoring/alert cards
- Offline recording playback with start, pause, reset and window stepping
- Reviewable four-patient subset, download size and CPU preflight/training commands
- Saved class/patient distribution, confusion matrix, ROC/precision–recall plots, AUC metrics and loss curves
- Early prediction remains planned; no forecast probabilities

## Security

- **Upload validation** (`demo/security.py`): local uploads must use a `.edf` extension, pass an EDF/BDF header check, and stay under 50 MB.
- **Presentation checkpoint loading** (`src/component/presentation/model.py`): fixed baseline architecture, validated preprocessing and provenance, strict weights, CPU-only `torch.load(..., weights_only=True)`; no dynamic builder imports or unsafe pickle fallback.

## Figures
The latest [frontend QA report](../reports/frontend_polish/REPORT.md) links screenshots
of all eight sections at 1366×768 and 1920×1080.

## Legacy frontend reference

Run `python -m streamlit run demo/presentation_app.py` from the project root.
See [the project README](../README.md) and [verified progress](../PRESENTATION_PROGRESS.md).
The original research dashboard `app.py` is preserved,
but depends on missing artifact/registry and `data_loader.core` modules and is not
the supported presentation entry point. Its inference claims require actual compatible weights.
