# Presentation milestone: implemented work and remaining research

Verified in this Windows workspace on **14 October 2026 (India time)**, Python 3.11.9, CPU-only
PyTorch 2.14.1+cpu, NumPy 1.26.4, SciPy 1.14.1, MNE 1.9.0, Streamlit 1.65.0.
No dependencies were upgraded. No dataset or existing checkpoints were overwritten.

This is a functioning first academic milestone. It can support a presentation of
the analysis pipeline and initial model software. **There is no objectively measured
40% completion claim. The actual small-cohort baseline failed to detect held-out
seizures at its saved threshold, despite functioning training/inference software.**

## Implemented and verified

| Capability | Evidence / scope |
|---|---|
| Streamlit presentation dashboard | Navy/charcoal theme, eight requested sections, cyan signals and coral/green alerts; single sidebar renderer outside playback; automated uniqueness/navigation tests |
| EDF file handling | Real CHB-MIT sample opened lazily; synthetic upload-session tests, malformed files, extension/size checks, temporary-file lifetime and recovery |
| Raw EEG selection | Channels, start time, duration, amplitude spacing; absolute recording time; bounded reads and cached small slices |
| Offline preprocessing | Fourth-order SOS Butterworth filter, initial 1–40 Hz; contextual reads then cropping; optional per-window/channel z-score; before/after plots |
| Duplicate channel handling | MNE unique channel suffixes retained; exact names and integer channel indices; test fixture with duplicate EDF labels |
| Ground-truth annotations | Existing CHB-MIT parser reused and strengthened; multiple/numbered events, zero events, missing entries, malformed summaries |
| Seizure visualization | Actual sample event 2996–3036 seconds; jump opens 2991–3041 seconds; shaded regions explicitly labeled as ground truth |
| Annotation monitoring panel | Any positive half-open interval overlap produces a ground-truth alert with event/recording/selected times; boundary/unknown-status tests; never called AI inference |
| CNN monitoring panel | Approved real CHB-MIT checkpoint loaded by default; genuine per-window probabilities; independent source and explicit failed-evaluation warning; cached bounded inference; mismatch/missing-checkpoint handling |
| Offline recording playback | Start, pause, reset, previous/next; sequential model-length windows with one-second accelerated refresh; timer scoped to signal fragment; no live-device claim |
| AI/annotation comparison | CNN-positive windows use a coral top strip, annotations use amber timeline shading; table uses ≥50% label policy; viewed agreement is not held-out evaluation |
| Segmentation | Sample-exact metadata index, configurable duration/overlap, documented ≥50% seizure-union overlap policy, partial-tail discard, preview and CSV export |
| CPU model path | Existing `BaselineCNN1D` reused with eight default montage channels and a 32-unit classifier hidden layer; actual train → save → load → infer tests |
| Checkpoint contract | Saved ordered channels, sample rate, window duration, filter/normalization, threshold, training scope and EDF/summary hashes; strict loading and finite-weight checks |
| Reproducible training | Actual five-epoch CPU run completed within the approved 15-minute limit; configurable paths/epochs/batch/lr/threads/seed; lazy readers, weighted loss, training-only negative cap, clipping |
| Split safeguards | Whole patients by default, explicit preliminary recording split option; chb01/chb21 grouped as one patient; duplicate EDF hashes rejected; no random window-level split |
| Real evaluation | Actual held-out chb08 report and independent frozen-threshold recheck agree; all 43 positive windows missed; validation-only selection; measured loss curves and patient/class counts saved and shown |
| CPU resource preflight | Data/split validation and one disposable bounded batch benchmark; no saved weights; approximate runtime projection; configurable 15-minute default training/inference deadline |
| Selected downloader | Approved nine-EDF/three-summary transfer completed (364.04 MiB new data); all 14 selected files checksum-verified; explicit manifest, HTTP-range resume, budgets and no-overwrite tests |
| Documentation | Windows CMD setup/run/train/evaluate commands, pinned compatible dependencies, presentation walkthrough, original README preserved as upstream reference |

Actual `chb01_03.edf` has **23 channels, 256 Hz, 3600 seconds** and **one 40-second
annotated seizure**. Four-second, non-overlapping segmentation produces **900
windows: 10 seizure-labeled and 890 non-seizure-labeled**. These counts are labels,
not neural-network outputs or evidence of model performance.

Static plot exports from the real recording were visually inspected:
[raw seizure](reports/presentation_checks/raw_seizure.png) and
[filtered seizure](reports/presentation_checks/filtered_seizure.png).
The plotting code uses independent Matplotlib Figure objects, avoiding a broken
local Tk installation and avoiding global pyplot figure creation in Streamlit threads.
The new dark counterparts are
[raw seizure](reports/presentation_checks/raw_seizure_dark.png) and
[filtered seizure](reports/presentation_checks/filtered_seizure_dark.png).

Frontend rendering is separated into `demo/dashboard.py` (one sidebar and app layout),
`demo/analysis_panel.py` (signal/monitoring/playback fragment), `demo/model_panel.py`
(genuine inference and saved evaluation), and `demo/ui.py` (escaped status cards and
scoped presentation CSS). `demo/presentation_app.py` retains the same entry point,
lazy session-owned reader, cached slices and safe cached checkpoint loading.
No duplicate widgets are hidden with CSS. A duplicate block was not reproduced in
the previous source; the new layout explicitly renders the sidebar only once and
tests count its widgets across navigation and playback reruns.

## Original frontend investigation

`demo/presentation_app.py` originally contained a working EDF uploader and a lazy
first-ten-seconds/five-channel waveform view. Those capabilities were extended,
with loading moved into a session-owned resource so the EDF remains available across
reruns. The application entry point is preserved at the same path.

`demo/app.py` is preserved unchanged as upstream reference. It has useful waveform,
bandpower, spectrogram, correlation, model-comparison UI and static assets, but its
integration assumes a different project layout:

- `models.utils.artifacts` and `models.utils.registry` do not exist in this checkout.
- `data_loader.core.channels`, `.io`, and `.signal` do not exist.
- `feature.feature_engineering` points at a stale package; the existing implementation
  is under `src/component/features/` and has research-pipeline dependencies.
- Models actually live under `src/component/models/`; the existing baseline CNN and
  its relative imports work and are reused.
- Its `SRC_DIR = APP_DIR.parent` is the repository root, then `PROJECT_ROOT = SRC_DIR.parent`
  incorrectly points one directory outside the workspace. Artifact/config/upload
  paths also assume older `models`, `feature`, and `data_loader` locations.
- It preloads and processes the entire EDF before selecting a window, which is
  unnecessary for this CPU/RAM-constrained milestone.
- Its general input adaptation can pad/reduce channels and resample window arrays;
  the milestone instead requires the exact saved channel order and sample rate.

Repairing the complete research artifact ecosystem would exceed this first milestone.
No empty compatibility modules were created. The presentation dashboard reuses the
existing EDF validation, CHB-MIT parser, CNN, metrics helpers and the
offset-waveform concept. Small relative-import fixes in the dataloader package allow
its parser to be imported from the project root without temporary path hacks.
The upstream UI assets remain preserved in `demo/assets`.

## Verification record

- Existing suite before changes: **30 passed, 1 skipped**.
- Suite before this second milestone: **48 passed, 1 skipped**.
- Suite before the detection investigation: **64 passed, 1 skipped**.
- Current suite: **71 passed, 1 skipped, 0 failed**; includes pipeline,
  trained-checkpoint dashboard, upload-session and real-EDF dashboard tests.
- Setup checker: **14/14 passed**.
- `python -m pip check`: **No broken requirements found**.
- Streamlit started successfully at `http://127.0.0.1:8502` during verification;
  its health endpoint returned **HTTP 200 / ok**, and configuration reported **dark**.
- Windows browser automation found Chrome, then stopped because the helper could
  not reliably determine its current URL to enforce policy. No further browser
  automation was attempted. The browser-use surface itself exposed no browsers.
  Dashboard behavior was exercised through Streamlit AppTest, and actual signal
  plots were rendered and visually inspected. Full-page visual inspection in Chrome
  remains a manual final presentation check.
- The single-patient default training command was executed and correctly refused
  to train before creating a run/checkpoint.
- Before download, subset preflight correctly refused to create output while selected
  EDFs were missing. After approval/download, all four patient groups passed validation
  and the CPU preflight completed; its report is saved in
  [subset_cpu_preflight.json](reports/presentation_checks/subset_cpu_preflight.json).
- The training deadline test stopped a synthetic run and confirmed that no incomplete
  `baseline.pt` checkpoint was installed.
- The S3 metadata estimate identified **364.04 MiB new transfer**. After explicit user
  approval, nine EDFs and three summaries were downloaded. All ten EDFs/four summaries
  were verified by official SHA-256; original sample and summary timestamps stayed
  unchanged, with no partial files or download locks left behind.
- A disposable batch benchmark against the real sample measured shape `[16,8,1024]`,
  **0.5 MiB input**, approximately **0.11 s** bounded reads/preprocessing and **0.06 s**
  for one training step. No weights/evaluation were saved. This is not a completed
  training run. A subsequent selected-cohort preflight measured approximately
  0.116 s reads and 0.065 s per training step, projecting roughly four minutes.
- The approved **five-epoch real CHB-MIT CPU run completed**, saved a safe-loadable
  178,529-byte checkpoint, history/configuration/window indices and real predictions.
  Epoch 1 and threshold 0.50 were selected using validation data. Test data remained
  isolated during selection. A separate evaluator reproduced the original test report.
- Streamlit AppTest exercised the actual default recording/checkpoint: overview,
  real CNN monitoring and all eight held-out metrics rendered without exceptions.
  The original recording remains the default even after the subset download.
- Actual one-epoch CPU training, frozen-threshold test evaluation, checkpoint loading,
  probability inference and dashboard model outputs were verified on tiny synthetic
  **test fixtures only**, inside pytest temporary directories. Their metrics are
  not project results and their checkpoints are not installed in the dashboard.
- The existing skipped checkpoint-sidecar test requires optional `safetensors`.
  Two additional stale architecture test files remain excluded by the original
  `src/tests/conftest.py`. They are not counted as newly verified functionality.
- An upstream `torch.jit.script` deprecation warning remains non-fatal.

## Actual trained baseline and its limitation

The initial single recording was insufficient for independent patient-wise splits.
After explicit approval, the selected subset in `configs/presentation_subset.json`
was downloaded: **ten EDFs, four independent patients, six annotated seizures**,
including four non-seizure-only recordings. All 404.49 MiB of selected files are
present and verified. Cases chb01 and chb21 are grouped as one patient, per the
[PhysioNet dataset documentation](https://physionet.org/content/chbmit/1.0.0/).

| Split | Patient groups | EDFs | Complete windows | Non-seizure | Seizure |
|---|---|---:|---:|---:|---:|
| Train | chb01, chb03 | 6 | 5400 | 5353 | 47 |
| Validation | chb05 | 2 | 1802 | 1773 | 29 |
| Test | chb08 | 2 | 1800 | 1757 | 43 |

Training-only subsampling retained 2447 windows (2400 non-seizure, 47 seizure).
Validation and test retain every complete window. No patient or overlapping window
is shared between splits. Default inputs are eight ordered channels at 256 Hz,
four-second windows, 1–40 Hz contextual filtering and per-window/channel z-score.
The 41,250-parameter CNN used weighted cross-entropy, Adam at 0.001, batch 16,
two CPU threads and seed 42. Five epochs completed within 15 minutes; epoch 1 was
selected using validation F1, with a final validation-selected threshold of 0.50.

The dashboard now uses **`results/presentation_baseline/baseline.pt`**. It never
substitutes annotations for model probabilities. Missing/incompatible checkpoints
still disable inference with an explanatory message.

| Actual held-out test metric | Value |
|---|---:|
| Accuracy | 97.61% |
| Precision | 0 |
| Recall / sensitivity | 0 |
| F1 | 0 |
| Specificity | 1.000 |
| ROC-AUC | 0.825482 |
| PR-AUC (average precision) | 0.158814 |

**Confusion matrix: `[[1757, 0], [43, 0]]`. All 43 seizure windows were missed.**
Accuracy equals the always-non-seizure baseline on this imbalanced test set and
cannot be presented as successful detection. The dashboard prominently warns about
zero sensitivity. Falling training loss with rising validation loss indicates
overfitting. One validation patient/event and one test patient/event support only
a preliminary experiment; more independent data and model development are required.
The test threshold was not retuned after seeing these results.

Actual inference around the original chb01_03 seizure also missed its ten positive
windows at threshold 0.50. That recording belongs to training and is explicitly
identified as such in the dashboard. The ground-truth card still correctly alerts;
the independently computed CNN card preserves the disagreement.
Display-threshold changes affect exploratory decisions only, leaving saved test
metrics unchanged. No reliable detection or forecasting claim is made.

Evidence: [real run report](reports/presentation_checks/REAL_BASELINE_RESULTS.md),
[saved evaluation](results/presentation_baseline/evaluation.json),
[independent recheck](results/baseline_recheck.json), and
[actual epoch history](results/presentation_baseline/training_history.csv).
Real output plots were rendered and visually inspected:
[sample probabilities](reports/presentation_checks/real_sample_probability.png),
[measured losses](reports/presentation_checks/real_training_losses.png), and
[held-out confusion matrix](reports/presentation_checks/real_test_confusion.png).

The optional `--fit-only` walkthrough remains unevaluated and has not been run on
the real data. Synthetic fixture checkpoints stay in pytest temporary directories.

## Detection investigation follow-up

The observed 30-window inference starting at index 747 was reproduced using the
original checkpoint: 10 annotated seizure windows, zero predicted seizures, genuine
scores 0.244254–0.489712 at threshold 0.50. An exact validation-only threshold audit
selects 0.5302148461 and still misses all 43 held-out positive windows. It is not
installed as a new model or advertised as an improvement.

Implemented/tested training additions: fixed balanced batches, unweighted/weighted
BCE and focal loss, no double weighting, exact validation F1/FPR operating points,
validation-only early stopping and bounded steps. After subsequent user approval,
the exact three-epoch balanced-BCE experiment completed in **95.00 seconds**,
using the existing CNN, LR 0.0003, 60 batches/epoch and two CPU threads. It saved
separately at `results/presentation_balanced_bce/baseline.pt`; original data and
the baseline checkpoint hash remain unchanged, with no new downloads.

Validation selected **epoch 1**, threshold **0.5674257874**, validation FPR 0%,
recall 20.69%, precision 100% and F1 0.342857. The frozen chb08 test has **0%
recall/sensitivity/precision**, F1 0, specificity 100%, FP 0 and confusion matrix
`[[1757,0],[43,0]]`. ROC-AUC/AP fell to 0.667893/0.064731. A separate evaluator
reproduced the saved report. All three epochs used mixed 8/8 batches; eliminating
positive-free batches did not solve held-out detection. No further model or test
threshold tuning was attempted. **The candidate was not promoted to dashboard default.**
Streamlit AppTest loaded the candidate, verified its exact threshold without an
override and rendered its actual inference/evaluation panels without errors. The
full test suite remains **71 passed, 1 skipped, 0 failed**.

Monitoring cards now derive AI alerts from actual score/threshold rather than trusting
a supplied prediction flag. Saved thresholds are used exactly by default; exploratory
overrides require an explicit checkbox and show a false-positive-rate warning.
Annotations remain separate. Below-threshold scores do not rule out seizures.

[Comparison and verification](reports/cnn_investigation/COMPARISON.md) records old
versus threshold-only and trained-candidate metrics, actual user-window
inference, supported concerns and remaining scientific limitations. The existing
test patient has already been examined and cannot provide a fresh blinded comparison.

## Planned / not yet demonstrated

- Reliable trained seizure detection across unseen patients: more independent data,
  broader evaluation and model development required.
- LSTM and hybrid CNN-LSTM comparisons: existing research architectures remain, but
  are not trained or integrated into this presentation milestone.
- Feature-based ML inference: existing feature code is retained; no compatible
  feature-model checkpoint is integrated into the presentation dashboard.
- Seizure-level event sensitivity, false alarms/hour, alert calibration, monitoring
  latency and clinical validation: not implemented in this milestone.
- Early prediction: no preictal forecasting model or forecasting probabilities.
  Requires a prediction horizon/occurrence period, defensible preictal/interictal
  labels, multiple independent events, exclusion gaps, causal preprocessing and
  event-based held-out evaluation. The zero-phase offline filter is not appropriate
  for a real-time forecasting claim.
- Live acquisition/clinical monitoring: current dashboard analyzes stored recordings.
- A fresh environment installation: commands document the verified current versions;
  the existing environment was preserved rather than reinstalled.

## Final frontend refinement

Implemented a compact dark workstation layout across all eight sections, with grouped
controls, raw/processed comparison, dark scientific plots, separate annotation/CNN
panels and concise visible zero-recall warnings. Saved metrics and model behavior remain
unchanged. The entry point `demo/presentation_app.py` was preserved byte-for-byte.

Browser verification used an explicitly approved isolated Chromium profile at
1366×768 and 1920×1080. All sections were inspected, including uploads, seizure
navigation, prerecorded playback and actual existing-model inference. Targeted
automated verification: **32 passed, 1 skipped, 0 failed**, with six training-fixture
tests intentionally deselected. SHA-256 verification confirms **182 protected files**
unchanged. No training, downloads or threshold analysis occurred in this phase.
See [the frontend report and screenshot gallery](reports/frontend_polish/REPORT.md).

## Faculty demonstration wording

“We have implemented and verified the offline EEG analysis pipeline: EDF loading,
interactive signals, bandpass preprocessing, seizure annotations and segmentation.
We trained a small CNN on an approved four-patient subset, kept patients separate
across train/validation/test, and obtained genuine probabilities and evaluation.
The baseline missed the held-out seizure at its selected threshold, so improving
the detector with more independent data and model development is our next step.
Early prediction is a later research phase.”

Run instructions and the optional data/training steps are in [README.md](README.md).
