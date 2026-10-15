"""Bounded class-balanced batches and numerically stable binary training objectives."""
import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Exactly half seizure/half background in every TRAINING batch.

    Minority windows repeat across batches; this is not additional independent data.
    Readers remain lazy and validation/test are never sampled by this class.
    """
    def __init__(self, labels, batch_size, steps, seed=42):
        labels = np.asarray(labels)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Balanced batches require both binary classes.")
        if batch_size < 2 or batch_size % 2 or not 1 <= steps <= 500:
            raise ValueError("Balanced sampling needs an even batch size and 1–500 steps per epoch.")
        self.groups = [np.flatnonzero(labels == c) for c in (0, 1)]
        self.half, self.steps, self.seed, self.epoch = batch_size // 2, steps, seed, 0

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            batch = np.concatenate([rng.choice(group, self.half, replace=len(group) < self.half)
                                    for group in self.groups])
            rng.shuffle(batch)
            yield batch.tolist()


def make_loss(kind, counts, sampling, focal_gamma=2.):
    """BCE on logit margin is equivalent to the existing two-logit softmax score.

    Weighted BCE averages over batch SIZE, unlike weighted CE's batch-dependent
    sum-of-weights denominator. Balanced sampling must not also use inverse-frequency
    CE/BCE weights, which would correct the imbalance twice.
    """
    counts = np.asarray(counts)
    if counts.shape != (2,) or (counts <= 0).any():
        raise ValueError("Loss requires positive counts for both classes.")
    if sampling == "balanced" and kind in {"weighted-ce", "weighted-bce"}:
        raise ValueError("Do not combine balanced sampling with inverse-frequency weighted loss.")
    if not np.isfinite(focal_gamma) or focal_gamma < 0:
        raise ValueError("Focal gamma must be finite and nonnegative.")
    if kind == "weighted-ce":
        return torch.nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32))
    if kind not in {"bce", "weighted-bce", "focal"}:
        raise ValueError("Unknown binary training loss.")
    pos_weight = torch.tensor(float(counts[0] / counts[1]) if kind == "weighted-bce" else 1.)
    class_weights = torch.tensor(counts.sum() / (2 * counts) if sampling == "natural" else [1., 1.], dtype=torch.float32)

    def loss(logits, labels):
        margin = logits[:, 1] - logits[:, 0]
        targets = labels.to(dtype=logits.dtype)
        values = torch.nn.functional.binary_cross_entropy_with_logits(margin, targets, pos_weight=pos_weight, reduction="none")
        if kind == "focal":
            p = margin.sigmoid()
            pt = torch.where(labels == 1, p, 1 - p)
            values = values * (1 - pt).pow(focal_gamma) * class_weights[labels]
        return values.mean()
    return loss
