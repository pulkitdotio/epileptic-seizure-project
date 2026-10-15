# Real CHB-MIT baseline run — 14 October 2026 (India time)

This is an actual CPU training run on the approved CHB-MIT subset, not a synthetic
fixture. It demonstrates working training/inference software. **The classifier
failed to detect the held-out seizure at its validation-selected threshold.**

Checkpoint: `results/presentation_baseline/baseline.pt` (178,529 bytes).
Authoritative results: `results/presentation_baseline/evaluation.json`;
independently recomputed report: `results/baseline_recheck.json`.

| Split | Patient groups | EDFs | Complete windows | Non-seizure | Seizure |
|---|---|---:|---:|---:|---:|
| Train | chb01, chb03 | 6 | 5400 | 5353 | 47 |
| Validation | chb05 | 2 | 1802 | 1773 | 29 |
| Test | chb08 | 2 | 1800 | 1757 | 43 |

Training-only negative subsampling retained **2447 windows: 2400 non-seizure and
47 seizure**. Validation/test retained all complete windows. The experiment used
four independent patients and six seizure events, with one event in each held-out
patient. chb01 and chb03 never occur in validation/test. The manifest includes
official file sizes and SHA-256 hashes; all EDFs/summaries were verified.

The reused CNN has 41,250 parameters: temporal Conv1D blocks 32/64/128 (kernels
7/5/3), batch normalization, ReLU/dropout, max-pooling, global average pooling and
a 128 → 32 → 2 classifier. Inputs are eight ordered channels, 256 Hz, four seconds,
1–40 Hz zero-phase filtering with two-second context and per-window/channel z-score.
CPU settings: batch 16, threads 2, Adam learning rate 0.001, seed 42, weighted
cross-entropy and gradient clipping. Five epochs completed within the 15-minute
limit; **epoch 1** was selected by validation F1 at 0.5, then the threshold was
selected on validation F1 only and remained **0.50**. Test data was not used to
choose the epoch or threshold.

| Held-out test metric | Actual value |
|---|---:|
| Accuracy | 0.976111 |
| Precision | 0.000000 |
| Recall / sensitivity | 0.000000 |
| F1 | 0.000000 |
| Specificity | 1.000000 |
| ROC-AUC | 0.825482 |
| PR-AUC (average precision) | 0.158814 |

Confusion matrix (rows: actual non-seizure/seizure; columns: predicted
non-seizure/seizure): `[[1757, 0], [43, 0]]`.

The model classified every test window as non-seizure: **all 43 seizure windows
were missed**. Accuracy matches an always-non-seizure classifier on this imbalanced
test set, so it cannot be presented as successful detection. ROC-AUC measures score
ranking across possible thresholds; it does not change the observed zero sensitivity
at the saved threshold. Further model/data development is necessary. One test
patient/event is far too little for robust generalization conclusions.

Training loss declined from 0.4764 to 0.1354 while validation loss rose from 0.7015
to 1.6491 and validation F1 at 0.5 declined after epoch 1. This is evidence of
overfitting; retaining the best validation state avoided replacing it with a worse
final-epoch state.

Actual inference on the original chb01_03 event (a **training recording**) is saved
in `reports/presentation_checks/real_sample_inference.csv`. All ten annotated
four-second seizure windows in that preview are below 0.50 and are missed too.
The dashboard preserves this disagreement: red ground-truth annotation status
and the separately computed CNN state. Changing the display threshold changes
only the exploratory UI decisions; it never changes the saved evaluation.

Figures from real outputs: `real_sample_probability.png`, `real_training_losses.png`,
and `real_test_confusion.png` in this directory. No synthetic test metrics or
checkpoints were installed as project results.
