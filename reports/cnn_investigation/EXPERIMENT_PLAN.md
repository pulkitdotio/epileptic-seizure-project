# CNN detection investigation — 14 October 2026

The original checkpoint is preserved at `results/presentation_baseline/baseline.pt`.
SHA-256: `4ad53ff2dac951a33f524f6d9969b921641edcea9936a5bc60f61da98648bdf1`.
The user subsequently approved the prepared experiment. It completed all three
epochs in **95.00 seconds**, saving `results/presentation_balanced_bce/baseline.pt`.
No additional data was downloaded. **Held-out detection did not improve**, so the
dashboard default remains the original checkpoint.

## Findings from the original run

- Training uses 2400 background and 47 seizure windows (1.92% positive). The original
  inverse-frequency CE weights are 0.5098 and 26.0319. A random batch of 16 has an
  estimated 73.32% probability of containing no seizure example. Weighted CE's mean
  divides by the sum of weights in that batch, so weights cancel in all-negative
  batches. This is a plausible optimization problem, not proof of the sole cause.
- Training loss fell from 0.4764 to 0.1354 while validation loss rose from 0.7015 to
  1.6491. Longer training with the original settings is not justified by that history.
- Validation seizure/background mean scores are 0.4210/0.4228; test means are
  0.3941/0.3254. Validation ROC-AUC is 0.4477 while test ROC-AUC is 0.8255.
  The score distribution varies by patient and is poorly separated on validation.
- The original test maximum seizure score is 0.4767. Threshold 0.50 misses every
  positive window. Exact **validation-only** F1 optimization under a prespecified
  5% validation-window FPR cap selects **0.5302148461**, with validation recall
  20.69%, precision 100%, F1 0.342857 and zero FP. This threshold still gives zero
  test recall. The threshold-only audit provides no detection improvement.
- Train and inference share `read_window`, saved filter configuration, channel
  order, sampling frequency and per-window z-score. A regression test checks exact
  tensor equality. Normalization is consistent; it does discard absolute amplitude
  information. No channel padding/resampling is performed.
- Whole patients remain separate: train chb01/chb03, validation chb05, test chb08.
  No overlapping windows or patient identities cross splits. The sample chb01_03
  belongs to training and cannot serve as an independent improvement target.
- The reused 41,250-parameter CNN has BatchNorm and global average pooling. Sparse
  positive batches may skew normalization statistics, and pooling can dilute local
  patterns. These are hypotheses. Keep the architecture unchanged in the first run
  rather than adding an untested larger model.

Audit data and exact operating points are in `baseline_audit.json` and
`baseline_validation_threshold_curve.csv`; synthetic fixtures are not project metrics.

## Prespecified bounded candidate — executed

- Existing CNN and exact saved preprocessing/channel contract; train **from scratch**.
- Exactly 8 seizure and 8 background samples per batch; 60 batches per epoch.
  Minority windows repeat. This does not add independent seizures. Background comes
  only from the training subset. Validation/test retain all natural windows.
- BCE on the difference between the two logits; its sigmoid equals the existing
  softmax seizure score. With balanced batches, use no inverse-frequency weight.
- Learning rate 0.0003, Adam, seed 42, CPU threads 2, at most 3 epochs.
- Choose epoch and threshold using exact validation F1, with validation-window FPR
  at most 5%; ties favor fewer FP and then a higher threshold. Patience 2.
- Freeze selection, then evaluate the unchanged chb08 test split once. Report
  precision/recall/F1/sensitivity/specificity, FP/FN/TP/TN and confusion matrix.
- Save separately to `results/presentation_balanced_bce`. Original files/checkpoint
  are never overwritten. Five-minute training/inference limit; no checkpoint on timeout.

```bat
python -m src.component.presentation.train --dataset-manifest configs/presentation_subset.json --split-file configs/presentation_splits.json --output results/presentation_balanced_bce --epochs 3 --batch-size 16 --lr 0.0003 --threads 2 --seed 42 --sampling balanced --loss bce --steps-per-epoch 60 --threshold-policy exact-f1 --max-val-fpr 0.05 --early-stopping-patience 2 --max-minutes 5
```

Preflight passed using a disposable real-data batch: `[16,8,1024]`, 0.5 MiB input,
approximately 0.125 s reading/preprocessing and 0.085 s training step; rough runtime
1.82 minutes. This estimate is not a guarantee; file hashing and test inference add time.
Preflight saved no weights. After subsequent approval, the exact invocation above
completed within its five-minute limit. It is retained as the historical command;
the trainer refuses to overwrite that now-existing output directory.

Validation selected **epoch 1**, threshold **0.5674257874**, F1 **0.342857**,
recall **20.69%**, precision **100%** and **0/1773** false-positive background windows.
Thus the prespecified validation FPR cap was satisfied. The frozen test report has
**zero recall/precision/F1**, specificity 100%, FP 0 and confusion matrix
`[[1757,0],[43,0]]`: every test seizure window was missed. Test ROC-AUC declined
from 0.825482 to 0.667893; average precision declined from 0.158814 to 0.064731.
An independent frozen-threshold evaluator reproduced the report exactly.
See [COMPARISON.md](COMPARISON.md) and [balanced_bce_comparison.json](balanced_bce_comparison.json).

## Other supported objectives

`--sampling natural --loss weighted-bce` uses positive weight N_background/N_seizure
and a fixed batch-size denominator. `--loss focal` downweights easy samples; natural
sampling applies train-derived class weights, balanced sampling uses unit weights.
Focal gamma defaults to 2. These numerical paths have tests, but no real comparative
training has been performed. They are future controlled ablations, not claimed improvements.
Balanced sampling with inverse-frequency weighted CE/BCE is explicitly rejected.

The first candidate changes sampling, objective and learning rate together. It is
an engineering comparison, not a causal ablation proving which change helped.

## Evaluation limits

The test patient has already been examined in the failed baseline investigation.
Future comparisons on it are descriptive exploratory results, even though candidate
selection uses validation only. They are not a fresh blinded generalization estimate.
One validation/test event and four training events are too few to establish robust
performance. Windows from an event are correlated; oversampling does not fix this.
A validation-window FPR cap is not a guarantee on test patients and is not event-level
false alarms/hour. Do not tune the next run using its test outcomes. More independent
patients/events and a new untouched evaluation cohort are needed for credible claims.

## Dashboard alert fixes

Annotation alerts use CHB-MIT event overlap. **AI Seizure Detected** is derived solely
from the genuine CNN probability and active threshold, even if an old prediction flag
disagrees. Scores/thresholds show six decimals; below-threshold scores do not rule out
seizures. The saved threshold is used exactly by default; an explicit exploratory
override warns that saved false-positive rates no longer apply. Missing/incompatible
checkpoints cannot generate AI alerts. Saved zero-sensitivity warnings remain visible.

## References

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html): weighted mean denominator.
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html): stable binary loss and positive weighting.
- [Lin et al., Focal Loss](https://arxiv.org/abs/1708.02002): hard-example emphasis; its original object-detection results do not establish EEG efficacy.
