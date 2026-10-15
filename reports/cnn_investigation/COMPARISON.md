# Actual detection comparison — 14 October 2026

The subsequently approved balanced-BCE experiment completed three CPU epochs in
**95.00 seconds**. The new checkpoint is saved separately in
`results/presentation_balanced_bce/baseline.pt`. The original checkpoint and dashboard
default threshold 0.50 are unchanged because the candidate did not improve detection.

| Held-out chb08 metric | Original CNN | Balanced-BCE CNN |
|---|---:|---:|
| Frozen threshold | 0.500000 | 0.567426 |
| Recall | 0% | 0% |
| Precision | 0% | 0% |
| F1 | 0 | 0 |
| Sensitivity | 0% | 0% |
| Specificity | 100% | 100% |
| False-positive windows | 0 | 0 |
| False-negative windows | 43 | 43 |
| True-positive windows | 0 | 0 |
| True-negative windows | 1757 | 1757 |
| ROC-AUC | 0.825482 | 0.667893 |
| PR-AUC (average precision) | 0.158814 | 0.064731 |
| Confusion matrix, rows actual / columns predicted | `[[1757,0],[43,0]]` | `[[1757,0],[43,0]]` |

Both measured columns classify every test window as background. Zero FP is achieved
by missing the entire seizure; it is not evidence of useful detection. Precision
is reported as zero under the project's explicit zero-division convention when
there are no positive predictions.

The earlier threshold-only audit used saved genuine original-model probabilities. Its
threshold maximizes **validation** F1 under a prespecified validation-window FPR cap
of 5%. Validation recall is 6/29 = 20.69%, precision 100%, F1 0.342857 and FP 0/1773.
The threshold is frozen before test scores are read by the audit function. It was
not selected using chb01_03 or test labels, and was not installed as a new checkpoint.
The coarser original threshold grid selected 0.50; exact selection does not rescue
the original model's held-out detection. Original-model test ROC-AUC 0.825482 and AP 0.158814 are unchanged,
because a threshold-only change cannot alter score ranking.

## Completed balanced-BCE experiment

The approved configuration was used unchanged: the same 41,250-parameter CNN,
eight ordered channels, four-second 256 Hz inputs, 1–40 Hz contextual filtering,
per-window/channel z-score, Adam learning rate 0.0003, batch size 16, seed 42 and
two CPU threads. Each epoch used **60 batches of 8 seizure/8 background windows**,
960 training draws and zero positive-free batches. Minority windows repeat;
2880 total training draws do not mean 2880 independent windows/events.

Whole-patient splits stayed fixed: train chb01/chb03, validation chb05, test chb08.
The original and candidate test windows, labels, EDF/summary hashes and preprocessing
were checked to be identical. Validation/test kept all 1802/1800 complete windows.

Validation F1 under the prespecified **5% validation-window FPR cap** selected
**epoch 1**, then threshold **0.5674257874488831** was frozen. At this operating
point validation has TP 6, FN 23, TN 1773, FP 0: recall 20.69%, precision 100%,
F1 0.342857 and FPR 0%. Epochs 2/3 gave lower optimized validation F1 (0.270270 /
0.260870); the final epoch was not blindly installed. The cap does not promise
test recall or event-level false alarms/hour.

Training BCE fell from 0.6360 to 0.1935 while diagnostic class-balanced validation
cross-entropy rose from 0.6654 to 1.0644. These are different loss reductions,
but each curve's trajectory and falling validation F1 indicate continued overfitting.
Balanced sampling eliminated seizure-free batches without solving patient generalization.
This combined sampling/loss/lr experiment does not isolate a single causal factor.

The separate frozen-threshold evaluator reproduced the candidate's saved test
report exactly. All 43 positive test windows were missed. Its lower ROC-AUC/AP
provide additional evidence against promotion. **No dashboard default was changed;
no further training, threshold tuning on test data or downloads were performed.**

Evidence:
[saved test report](../../results/presentation_balanced_bce/evaluation.json),
[independent recheck](../../results/presentation_balanced_bce/frozen_recheck.json),
[training history](../../results/presentation_balanced_bce/training_history.csv),
[validation selection](../../results/presentation_balanced_bce/threshold_selection.json),
[comparison JSON](balanced_bce_comparison.json), and [measured runtime](balanced_bce_timing.json).
Plots: [score comparison](original_vs_balanced_scores.png),
[actual losses](balanced_bce_training_losses.png), [confusion matrix](balanced_bce_test_confusion.png).

## Reproduced user observation

Actual inference was rerun on **30 windows starting at index 747** of chb01_03.edf:
10 annotated seizure windows, **0 CNN-positive windows** at threshold 0.50.
Scores range from 0.244254 to 0.489712. Results are in
[sample_747_30.csv](sample_747_30.csv), with provenance in
[sample_747_30.json](sample_747_30.json). This is a training recording and is not
an independent test. Ground-truth alerts remain valid even when the CNN disagrees.

## Verified changes

Training software now supports fixed mixed-class batches, stable unweighted/weighted
BCE and focal loss, rejection of double weighting, exact validation-only threshold
search with FP reporting, validation early stopping and a bounded step budget.
The executed candidate preserved the existing small CNN and preprocessing and
reduced the learning rate to 0.0003. It provided no held-out detection improvement.

AI alert cards are computed from genuine score/active threshold rather than trusting
an existing prediction flag. They show six-decimal values and distinguish **AI
Seizure Detected**, **No AI seizure alert**, and unavailable inference. Annotations
remain a separate source. The saved threshold is used exactly unless the user
explicitly enables an exploratory override, which carries an FP-rate warning.

**Tests: 71 passed, 1 skipped, 0 failed.** The existing skip requires optional
`safetensors`; the upstream Torch JIT deprecation warning is non-fatal. Added tests
cover batch balance/reproducibility, stable losses and weighting, threshold ties/FPR,
alert independence, exact train/inference preprocessing, and synthetic-fixture
train/save/load/evaluation. Fixture weights and metrics never become project outputs.
The candidate also loaded in Streamlit AppTest without inference/evaluation errors;
its exact saved threshold was used with the override disabled, and all eight real
metrics rendered. The default checkpoint remained the original model. Evidence:
[candidate dashboard check](balanced_bce_dashboard_check.json). Actual output plots
were visually inspected.

The original checkpoint SHA-256 remains
`4ad53ff2dac951a33f524f6d9969b921641edcea9936a5bc60f61da98648bdf1`.
No EDF, summary or original checkpoint was overwritten; no data was downloaded.
The candidate SHA-256 is
`fe2b8aecf2fd0b052ee5e925db8a3f1794b81e219e7d372d004983b88239af9a`.

## Remaining uncertainty

Optimization imbalance, overfitting and patient score shift are supported concerns;
the investigation does not isolate a single cause. More epochs alone are not
supported by either run's validation history. The completed bounded experiment
tested a more appropriate optimization scheme, but repeated oversampled windows
cannot replace more independent events/patients. Credible generalization requires
a larger cohort and a new untouched test set. The already examined chb08 comparison
is exploratory, and window FP counts are not event-level false alarms/hour.

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), [baseline_audit.json](baseline_audit.json),
and [baseline score distributions](baseline_score_distributions.png).
