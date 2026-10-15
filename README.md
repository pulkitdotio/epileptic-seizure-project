# Advanced EEG Signal Analysis for Epileptic Seizure Detection and Early Prediction

A local Streamlit + MNE + PyTorch academic prototype for Python 3.11 on Windows,
8 GB RAM, and CPU-only inference. The implemented milestone is EEG exploration,
offline preprocessing, annotation visualization, window labeling, and a reproducible
baseline training/inference workflow. After explicit approval, a selected four-patient
subset was downloaded and a five-epoch CPU baseline was trained. The dashboard loads
its genuine checkpoint. **It missed every held-out seizure window at its saved threshold**;
the working software does not yet provide reliable seizure detection. There is no
clinical forecasting capability or numerical project completion claim.

## What works now

- Load the downloaded CHB-MIT sample or upload an EDF (50 MiB limit).
- Select up to 12 channels, recording start time, display duration, and waveform spacing.
- Compare raw and configurable 1–40 Hz filtered EEG; optional per-window z-score.
- Parse CHB-MIT summaries, including multiple events and explicit zero-seizure entries.
- Jump to and highlight ground-truth seizures, separately from model outputs.
- Build a window metadata index, inspect segments and export CSV labels.
- Train the existing three-layer `BaselineCNN1D` on CPU with lazy window reads.
- Load a compatible checkpoint to display genuine per-window model outputs.
- Display real saved held-out window metrics when such an evaluation exists.
- Use a compact dark workspace with eight sections, restrained chart colors and grouped controls.
- See separate annotation and trained-CNN monitoring cards, with explicitly labeled sources.
- Start, pause, reset or step through **Offline EEG Recording Playback**.
- Review an explicit, checksum-verified four-patient download plan before transfer.

**Original demonstration recording:** `data/raw/chbmit/chb01/chb01_03.edf`, 23 channels,
256 Hz, 3600 seconds, with one annotated seizure at **2996–3036 seconds**.
Default four-second segmentation yields **900 windows: 10 seizure and 890 non-seizure**.
The approved local subset now contains **ten EDFs, four independent patients and six
seizures**. Training uses chb01/chb03, validation chb05, and testing chb08.
`results/presentation_baseline/baseline.pt` is loaded by default. Actual held-out
recall/sensitivity and F1 are **0**, with **43 missed seizure windows**. Its 97.61%
accuracy reflects class imbalance and cannot be presented as successful detection.
Tiny synthetic training tests are separate from these real project results.

## Windows CMD setup

Your existing `.venv` is already usable; it was tested without changing installed packages.
Activate it and start the application from the project root:

```bat
.venv\Scripts\activate.bat
python -m streamlit run demo/presentation_app.py
```

Open the localhost URL shown by Streamlit (usually `http://localhost:8501`).
For a **new** environment only:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-presentation.txt
python -m pip install -e . --no-deps
python src/component/check_setup.py
```

The pinned versions mirror the verified local environment. NumPy stays at 1.26.4,
SciPy at 1.14.1, and MNE at 1.9.0. Do not run an unrestricted dependency upgrade.
No CUDA, Docker, database, cloud service, or entire-dataset download is required.

## Presentation walkthrough

1. Keep **Recording source → Downloaded sample** selected. The local summary loads automatically.
2. **Overview** shows recording statistics, honest model status and shortcuts to signals or the seizure.
3. **Raw EEG** lets you choose channels, time, duration, and spacing. Reads are bounded;
   the entire recording is not preloaded. Long plots use min/max envelopes to retain spikes.
4. **Preprocessing** shows raw and processed dark-themed waveforms side by side or stacked. Start with 1–40 Hz, then compare
   retained microvolt amplitude against optional per-channel z-score.
5. **Seizure Annotations → Jump to seizure** opens 2991–3041 seconds,
   giving five seconds of context around the 2996–3036 second ground-truth event.
6. **Segmentation** defaults to four-second non-overlapping windows. Filter previews
   to **Seizure**, inspect a window, and download the index CSV.
7. **AI Detection** displays genuine checkpoint probabilities and a warning about
   its failed held-out detection. **Training & Evaluation** shows measured losses,
   patient/class counts and the saved test report. **Early Prediction**
   explains the remaining research requirements.

The **Dataset annotation** and **CNN prediction** panels use separate sources. Any positive overlap with
an annotated event shows **Annotated seizure**, including its onset,
offset and duration. An interval ending exactly at seizure onset
does not overlap. This alert is explicitly **CHB-MIT ground truth, not AI inference**.
With compatible trained weights, the second card performs actual CNN inference on
the model-length window beginning at the selected start (clamped at the recording end).
Its displayed time interval identifies exactly which signal was classified.

In **Raw EEG** or **AI Detection**, expand **Recording playback** to use **Start**, **Pause**,
**Reset**, **Previous**, and **Next**. Playback displays sequential
four-second windows (or the checkpoint's saved duration) on a one-second refresh.
This is **accelerated Offline EEG Recording Playback**, not wall-clock real-time
or a live EEG device. Starting playback sets the displayed duration to its step length.
Pause stops the timer; navigation and source changes stop playback. Playback
updates only a Streamlit fragment; sidebar widgets are rendered once outside it.
No sidebar or uploader is hidden to mask duplication. Restart the existing
Streamlit process with **Ctrl+C**, then relaunch to load the new theme configuration.

To upload the same file instead, select **Upload EDF**, choose `chb01_03.edf`, and also
upload `chb01-summary.txt` inside **Annotation summary**. The checkpoint path is under **Model checkpoint**.
The signal and result pages keep detailed processing, provenance and secondary metrics in expanders.
See the [frontend visual QA report](reports/frontend_polish/REPORT.md) for screenshots and checks.
Uploaded recordings are stored
in a session-owned temporary directory; original data is never modified. A matching
summary is required for labels. A missing entry means **unknown**, not non-seizure.
MNE gives duplicate channel labels unique suffixes; these exact names appear in selection controls.

## Processing and label contract

- Fourth-order Butterworth bandpass in second-order sections, forward/backward zero phase.
- Up to two seconds of actual recording context on each side, filter, then crop.
  Recording edges have reduced context and reflection padding; edge artifacts remain possible.
- Optional z-score is calculated per channel **after cropping each window**. Constant channels remain finite.
- Timing and sampling frequency are preserved. There is no implicit resampling,
  channel reordering, channel padding, or conversion to a different montage.
- Labels use half-open intervals `[start, end)`: union seizure coverage **≥50%** of
  a complete window is seizure; less coverage is non-seizure. A partial trailing window is discarded.
- Visualization can overlap windows by 50%; training uses non-overlapping windows.
  Training/validation/test are assigned by whole groups before window subsampling.
- Zero-phase filtering uses future samples and is for **offline** detection analysis,
  not a causal live-monitoring or early-forecasting pipeline.

## Baseline training

The reused `BaselineCNN1D` accepts `[batch, channels, samples]`. Three temporal
Conv1D blocks have 32/64/128 outputs and kernels 7/5/3, each with batch normalization,
ReLU and dropout. Two max-pools reduce temporal resolution, followed by adaptive
average pooling and a 128 → 32 → 2 classifier. The default eight-channel model has
**41,250 parameters**. Four-second 256 Hz windows have shape `[batch, 8, 1024]`.
Seizure probabilities come from the two-class softmax of these actual trained logits.

Default training requires at least three independent patient groups and seizure/non-seizure
examples in **every** split. This is a minimum software requirement, not evidence of adequate
sample size. More independent patients and seizure events are needed for credible conclusions.
`chb01` and `chb21` are grouped together because they are the same patient.
Duplicate EDF content is rejected using SHA-256; unknown annotations cannot enter training.

The approved run has already completed. To reproduce it in a **new** output directory:

```bat
python -m src.component.presentation.train --dataset-manifest configs/presentation_subset.json --split-file configs/presentation_splits.json --output results/presentation_baseline_run2 --epochs 5 --batch-size 16 --lr 0.001 --threads 2 --seed 42 --max-minutes 15
```

If only the original single recording is available, default patient-held-out training
refuses to start; a recording alone cannot supply independent patient groups.
The initial eight channels are `FP1-F7,F7-T7,T7-P7,P7-O1,FP2-F4,F4-C4,C4-P4,P4-O2`.
Use `--channels` with comma-separated exact names to select a different common montage.
Other options: `--window-sec`, `--sfreq`, `--low-hz`, `--high-hz`, `--max-background`.
Sampling frequencies must already match; channel compatibility is validated before training.

Random patient allocation uses seed 42; if a split lacks a class, the script refuses to train.
For deliberate allocation, use `--split-file patient_splits.json` with all selected groups
assigned exactly once, for example:

```json
{"train": ["chb01"], "val": ["chb02"], "test": ["chb03"]}
```

That example is only a file format illustration; those patients must actually exist locally.
`--split recording` enables whole-recording splits when at least three recordings exist;
results are clearly marked **recording-held-out / preliminary**, with no patient-independent claim.
With that mode, split-file entries are relative EDF paths rather than patient IDs.

An optional **unevaluated training walkthrough** is also available:

```bat
python -m src.component.presentation.train --data-root data/raw/chbmit --output results/exploratory_fit --fit-only --epochs 2 --batch-size 16 --threads 2
```

This opt-in command fits genuine weights on the discovered labeled data but has
**no validation/test scores** and saves an explicitly marked `fit-only` checkpoint.
It can memorize training recordings and does not demonstrate generalization.
This mode has not been run on your real data during implementation.
Each command requires a new output directory; existing runs/checkpoints are never overwritten.

Training uses weighted cross-entropy for class imbalance, reproducible seeds, gradient
clipping, two CPU threads, a 400-background-window-per-recording training cap, and lazy
EDF readers. Validation/test retain **all** complete windows and their natural imbalance.
The best epoch is selected using validation F1 at 0.5; the final threshold is selected on
validation F1 only. The test set is evaluated once after selection.

Outputs include `baseline.pt` (weights, exact channels, filter/normalization config,
window duration, threshold, split provenance and hashes), window-index CSVs,
`training_history.csv`, and, when held-out data exists, `evaluation.json`,
`validation_predictions.csv`, and `test_predictions.csv`.
New checkpoints also embed measured epoch history, validation loss and split class
counts. The run's configuration JSON records the final selected epoch and provenance.
`--max-minutes 15` bounds training and inference phases; if the deadline is exceeded,
the run stops without installing a checkpoint. Use a fresh output directory for a retry.
`--preflight` checks the data/splits and benchmarks one disposable CPU batch, without
creating a run or saving model weights. Its runtime estimate is approximate, not a guarantee.

## Evaluation and dashboard model loading

Recompute the original held-out test report with the frozen checkpoint threshold:

```bat
python -m src.component.presentation.evaluate --checkpoint results/presentation_baseline/baseline.pt --output results/baseline_recheck_run2.json --batch-size 16 --threads 2
```

The evaluator checks original EDF and annotation hashes and refuses changed data.
`--data-root` supports relocating the same dataset. It refuses fit-only checkpoints.
It computes accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC,
PR-AUC (average precision), and a confusion
matrix from real held-out labels and outputs; no event-level or clinical metrics are claimed.

The completed run and its independent frozen-threshold recheck agree:

| Held-out test metric | Actual value |
|---|---:|
| Accuracy | 97.61% |
| Precision | 0 |
| Recall / sensitivity | 0 |
| F1 | 0 |
| Specificity | 1.000 |
| ROC-AUC | 0.825482 |
| PR-AUC (average precision) | 0.158814 |

Confusion matrix: **`[[1757, 0], [43, 0]]`**. All 1800 test windows were classified
as non-seizure. This is a failed detector at threshold 0.50, despite the high accuracy.
Five epochs completed; epoch 1 was selected using validation data. Declining training
loss and rising validation loss indicate overfitting. One test patient/event provides
only a preliminary experiment. Do not retune thresholds on this test set.
See [the real run report](reports/presentation_checks/REAL_BASELINE_RESULTS.md),
[saved evaluation](results/presentation_baseline/evaluation.json), and
[recomputed evaluation](results/baseline_recheck.json).

In the sidebar's **CNN checkpoint** section, enter `results/presentation_baseline/baseline.pt`.
Open **AI Seizure Detection**, choose a
model window range, and click **Run CPU inference**. For the optional fit-only run use
`results/exploratory_fit/baseline.pt`; the dashboard prominently marks its limitation.
Inference uses saved preprocessing, exact channel names/order, and sampling rate.
It does not use seizure annotations to compute probabilities. Missing or incompatible
checkpoints produce a meaningful message. Original research checkpoints require an
explicit compatible preprocessing/provenance adapter; they are not silently loaded.

The saved validation-selected threshold is used exactly by default. Enable
**Use exploratory display threshold** to adjust it; the dashboard warns that saved
false-positive rates do not apply to that override. Cached model probabilities stay unchanged; only the displayed decisions
change. The saved held-out evaluation always retains its frozen threshold. A coral
strip marks AI-positive windows on the probability chart, while amber shading marks
ground truth. A comparison table uses the training label policy (≥50% coverage).
**AI Seizure Detected** depends only on the genuine CNN score and active threshold
(probability ≥ threshold), independently of annotations or a stale prediction flag.
A below-threshold result is labeled **No AI seizure alert** and does not rule out
a seizure. Scores and thresholds show six decimals. Ground-truth event alerts
remain separately labeled. Agreement on the current recording is **not** an unbiased evaluation score.
Model scores are not calibrated clinical confidence estimates.
**Model Training & Evaluation** displays patient/class distribution, embedded
training/validation loss curves, eight metrics and a confusion matrix when real
checkpoint data exists. Older checkpoints without newly embedded fields remain usable.

## Detection investigation and completed balanced-BCE experiment

The baseline has 47 positive versus 2400 background training windows. About 73% of
random batches contain no positive examples; per-batch weighted CE cancels its
class weight on those batches. Loss history also indicates overfitting and scores
vary by held-out patient. Train/inference preprocessing is identical and tested.
These are contributing hypotheses, not proof of one root cause.

The [investigation and experiment plan](reports/cnn_investigation/EXPERIMENT_PLAN.md)
documents the evidence, limitations, loss choices and exact commands. The saved-score
audit chose threshold **0.5302148461 using validation only**, with zero validation
false-positive windows. Test recall remained **0**, with zero test false positives
and confusion matrix `[[1757, 0], [43, 0]]`. Threshold refinement did not solve detection.

After subsequent approval, the candidate completed all three epochs in **95.00
seconds** and saved separately in `results/presentation_balanced_bce`. Its original
invocation is shown below; the existing output directory cannot be overwritten:

```bat
python -m src.component.presentation.train --dataset-manifest configs/presentation_subset.json --split-file configs/presentation_splits.json --output results/presentation_balanced_bce --epochs 3 --batch-size 16 --lr 0.0003 --threads 2 --seed 42 --sampling balanced --loss bce --steps-per-epoch 60 --threshold-policy exact-f1 --max-val-fpr 0.05 --early-stopping-patience 2 --max-minutes 5
```

The candidate uses the same small CNN and preprocessing, balanced 8/8 batches and
unweighted BCE on the two-logit margin. Do not double-correct sampling with inverse
frequency weights. Alternative tested numerical paths are `--sampling natural
--loss weighted-bce` and `--loss focal`; their real comparative performance is unknown.
The explicit 60-step budget bounds repeated minority exposure. Validation/test keep
their natural class distribution. `--threshold-policy exact-f1` selects epoch and
threshold using validation F1, with a prespecified 5% validation-window FPR cap;
ties prefer fewer FP, then a higher threshold. Early stopping uses validation only.
Training still refuses existing output directories.

Preflight passed and the actual run completed within its five-minute limit.
Validation selected epoch 1 and threshold **0.5674257874**, with validation recall
20.69%, precision 100%, F1 0.342857 and zero FP (0% validation-window FPR).
The frozen held-out test still missed all **43** seizure windows: recall/precision/F1
**0**, specificity 100%, FP 0 and confusion matrix `[[1757,0],[43,0]]`. Test ROC-AUC
declined from 0.825482 to 0.667893; average precision declined from 0.158814 to
0.064731. An independent recheck reproduced the candidate's report exactly.
**No improvement was demonstrated, so the dashboard default remains the original CNN.**
The [comparison report](reports/cnn_investigation/COMPARISON.md) includes actual
results, runtime, checksums, loss/score plots and limitations. For manual inspection
only, enter `results/presentation_balanced_bce/baseline.pt` in the sidebar checkpoint
field; this does not replace the original file or change the application's default.
To recheck its frozen report later, choose a new output filename:

```bat
python -m src.component.presentation.evaluate --checkpoint results/presentation_balanced_bce/baseline.pt --output results/balanced_bce_recheck_run2.json --batch-size 16 --threads 2
```

The previously examined test patient can support only exploratory comparisons;
more independent events/patients and a new untouched test cohort are needed for
credible generalization conclusions. False-positive **windows** are not seizure-level
false alarms/hour. Do not choose future settings based on this test patient's results.

## Optional selected downloads

The approved selection is [configs/presentation_subset.json](configs/presentation_subset.json),
with official SHA-256 hashes, mirror sizes, and summary intervals checked on 13 October 2026.
The split file is [configs/presentation_splits.json](configs/presentation_splits.json).

| Split | Independent patients | Explicit EDF selection | Annotated events |
|---|---|---|---|
| Train | chb01, chb03 | chb01_01, chb01_03, chb01_04; chb03_01, chb03_02, chb03_05 | 4 |
| Validation | chb05 | chb05_01, chb05_06 | 1 |
| Test | chb08 | chb08_02, chb08_03 | 1 |

All names end in `.edf`. Four non-seizure-only recordings supplement the background
in seizure-containing recordings. The selection totals **404.49 MiB**, including
summaries. The approved **364.04 MiB** transfer of nine new EDFs and three summaries
completed; all selected files were checksum-verified. Your original EDF/summary were
retained unchanged. The dataset is excluded from Git.
Two training patients and one validation/test patient each support only a small,
preliminary experiment. Event selection is deliberate; these results cannot represent
the full dataset, all patients, or clinical reliability. More patients/events and
multiple patient-held-out runs are needed later.

The downloader is **network-free by default**. Review names without fetching anything:

```bat
python -m src.component.presentation.download_subset --manifest configs/presentation_subset.json
```

Verify current mirror sizes, existing file hashes and available summary entries:

```bat
python -m src.component.presentation.download_subset --manifest configs/presentation_subset.json --estimate
```

`--estimate` reads only small metadata; no EDF or summary is written.
The approved download was run using this command. It now verifies and skips the
completed files; any future substantial additions should be reviewed first:

```bat
python -m src.component.presentation.download_subset --manifest configs/presentation_subset.json --download --max-mib 400
```

Completed matching files are checksum-verified and skipped. Interrupted transfers
retain `.edf.part` files and resume using verified HTTP ranges. Final size/SHA-256
must match before renaming; mismatched existing files are preserved and reported.
A corrupted complete partial cannot resume; inspect it before removing that specific
`.part` file. No complete existing EDF, annotation or checkpoint is overwritten.
The tool supports `--patients chb03 chb05` to filter an explicit manifest and
`--files chb03/chb03_01.edf` for individually selected files. It never expands a
patient selection into an entire patient folder or the full dataset.

Before any new run, benchmark the selected data and CPU workload:

```bat
python -m src.component.presentation.train --dataset-manifest configs/presentation_subset.json --split-file configs/presentation_splits.json --output results/presentation_baseline_run2 --epochs 5 --batch-size 16 --threads 2 --preflight
```

To reproduce the completed five-epoch experiment, use the fresh-directory command
in [Baseline training](#baseline-training).
The original approved invocation was:

```bat
python -m src.component.presentation.train --dataset-manifest configs/presentation_subset.json --split-file configs/presentation_splits.json --output results/presentation_baseline --epochs 5 --batch-size 16 --lr 0.001 --threads 2 --seed 42 --max-minutes 15
```

`--dataset-manifest` restricts the experiment to those exact, checksum-verified EDFs
and summaries; unrelated additional downloads do not silently join the run.
Do not rerun the original invocation into its existing output directory: it refuses
to overwrite the saved checkpoint. The completed subset preflight is recorded in
[subset_cpu_preflight.json](reports/presentation_checks/subset_cpu_preflight.json):
0.5 MiB input tensor, approximately 0.116 s reading/preprocessing and 0.065 s for
one training step, with an approximate four-minute run estimate. The actual five
epochs and held-out evaluation finished within the approved 15-minute limit.
Preflight itself saved no weights; the subsequent genuine training run did.
Dataset description and citation: [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/).

## Verification and project status

```bat
python -m pytest -q
python src/component/check_setup.py
```

Tests cover malformed EDFs, lazy access and duplicate channels, summary parsing,
window boundaries and unknown labels, filter attenuation and normalization, actual
CPU training on synthetic **test fixtures**, model shapes, checkpoint validation,
inference independence from annotations, split integrity, evaluation reproduction,
and dashboard interactions against the real downloaded EDF when available. New
tests cover annotation alerts, sidebar uniqueness, start/pause/reset/stepping,
model-score equivalence to loaded trained weights, threshold independence, dark
plots, saved loss/AUC data, downloader budgets/checksums/resume and manifest exclusion.
Synthetic test scores are never used as project results.

Latest verification: **71 passed, 1 skipped, 0 failed**; setup checker **14/14 passed**.
The skip requires optional `safetensors`. The Streamlit server health check passed
and real plot exports were visually inspected. Automated full-browser inspection
stopped because the browser helper could not establish the current URL for its policy
check; open Chrome manually for the final presentation layout check.

See [PRESENTATION_PROGRESS.md](PRESENTATION_PROGRESS.md) for verified work, the original
frontend diagnosis, presentation scope, and outstanding research tasks.
The original `demo/app.py` is preserved as an upstream research reference and still
depends on absent modules. Use `demo/presentation_app.py` for this milestone.
The original research README is preserved at [reports/UPSTREAM_README.md](reports/UPSTREAM_README.md);
its reported results/features are not verified local milestone results.

Remaining work: improve the failed baseline using more independent data and validation,
LSTM and hybrid comparisons, feature-model integration, event-level evaluation,
causal monitoring, calibrated alerts, and a properly defined and evaluated preictal
forecasting module. No clinically reliable seizure forecasting is implemented.
