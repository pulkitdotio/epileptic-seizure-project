"""Presentation plots derived from the original dashboard's offset waveform view."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from demo.design import BACKGROUND, TEXT, SECONDARY, ACCENT, DANGER, BORDER, SURFACE, WARNING

TEAL = ACCENT
CORAL = DANGER


def dark_axes(fig, ax):
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(color=SECONDARY, alpha=.12, linewidth=.5)


def dark_legend(ax):
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(loc="upper right", fontsize=8, facecolor=SURFACE, edgecolor=BORDER, framealpha=.95)
        for text in legend.get_texts():
            text.set_color(TEXT)


def waveform(times, data, channels, intervals=None, title="EEG signals", standardized=False, spacing=250, compact=False):
    fig = Figure(figsize=(5.5 if compact else 12, max(3.4, len(channels) * .65)))
    ax = fig.subplots()
    spacing = 6.0 if standardized else spacing
    values = data if standardized else data * 1e6
    # Min/max envelopes preserve narrow spikes when long views are downsampled.
    block = max(1, int(np.ceil(len(times) / 2500)))
    for i, channel in enumerate(channels):
        offset = -i * spacing
        if block == 1:
            ax.plot(times, values[i] + offset, lw=.65, color=TEAL)
        else:
            length = len(times) // block * block
            x = times[:length].reshape(-1, block).mean(axis=1)
            y = values[i, :length].reshape(-1, block)
            ax.fill_between(x, y.min(axis=1) + offset, y.max(axis=1) + offset, color=TEAL, alpha=.8, lw=.2)
    for i, (start, end) in enumerate(intervals or []):
        if end > times[0] and start <= times[-1]:
            ax.axvspan(max(start, times[0]), min(end, times[-1]), color=CORAL, alpha=.12,
                       label=f"Annotated seizure · {start:g}–{end:g} s" if not ax.get_legend_handles_labels()[0] else None)
            for edge in (start, end):
                if times[0] <= edge <= times[-1]:
                    ax.axvline(edge, color=CORAL, lw=.8, ls="--", alpha=.8)
    ax.set_yticks(-np.arange(len(channels)) * spacing, channels)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time from recording start (seconds)")
    ax.set_title(title, loc="left", fontsize=11, pad=14, color=TEXT)
    dark_axes(fig, ax)
    dark_legend(ax)
    ax.text(1, 1.01, "Z-score" if standardized else f"µV · {spacing:g} spacing",
            ha="right", transform=ax.transAxes, fontsize=8, color=SECONDARY)
    fig.tight_layout()
    return fig


def probability_plot(results, threshold, intervals):
    fig = Figure(figsize=(12, 3.5))
    ax = fig.subplots()
    x = (results.start_sec + results.end_sec) / 2
    ax.plot(x, results.seizure_probability, marker=".", color=TEAL, label="CNN output")
    ax.axhline(threshold, ls="--", color=SECONDARY, label=f"Threshold {threshold:.6f}")
    positives = results[results.seizure_probability >= threshold]
    for i, row in enumerate(positives.itertuples()):
        ax.axvspan(row.start_sec, row.end_sec, ymin=.92, ymax=1., color=CORAL, alpha=.8,
                   label="CNN alert (top strip)" if i == 0 else None)
    for i, (s, e) in enumerate(intervals or []):
        ax.axvspan(s, e, alpha=.12, color=CORAL, label="Annotation (shading)" if i == 0 else None)
    ax.set(xlabel="Window center (seconds)", ylabel="Seizure probability", ylim=(-.02, 1.02))
    ax.set_xlim(float(results.start_sec.min()), float(results.end_sec.max()))
    dark_axes(fig, ax)
    dark_legend(ax)
    fig.tight_layout()
    return fig


def training_curves(history):
    fig = Figure(figsize=(12, 3.5))
    ax = fig.subplots()
    ax.plot(history.epoch, history.training_loss, marker="o", markersize=4, color=TEAL, label="Training loss")
    if "validation_loss" in history:
        ax.plot(history.epoch, history.validation_loss, marker="o", markersize=4, color=WARNING, label="Validation loss (weighted CE)")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Measured training history")
    ax.set_xticks(history.epoch)
    dark_axes(fig, ax)
    dark_legend(ax)
    fig.tight_layout()
    return fig


def confusion_plot(matrix):
    fig = Figure(figsize=(7, 3.3))
    ax = fig.subplots()
    matrix = np.asarray(matrix)
    ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("eeg_counts", [SURFACE, ACCENT]), vmin=0, aspect="auto")
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, str(value), ha="center", va="center", fontsize=13,
                color=BACKGROUND if value > matrix.max() / 2 else TEXT)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-seizure", "Seizure"],
           yticklabels=["Non-seizure", "Seizure"], xlabel="CNN prediction", ylabel="Ground truth", title="Held-out confusion matrix")
    dark_axes(fig, ax)
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.grid(False)
    fig.tight_layout()
    return fig


def discrimination_plot(scores, kind):
    """Display curves from saved scores; no inference or threshold selection."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig = Figure(figsize=(6, 3.8))
    ax = fig.subplots()
    if kind == "roc":
        x, y, _ = roc_curve(scores.label, scores.seizure_probability)
        ax.plot(x, y, color=ACCENT, lw=1.3, label="Saved test scores")
        ax.plot([0, 1], [0, 1], color=SECONDARY, lw=.8, ls="--", label="Chance")
        ax.set(xlabel="False-positive rate", ylabel="Sensitivity", title="ROC curve", xlim=(0, 1), ylim=(0, 1.02))
    elif kind == "pr":
        y, x, _ = precision_recall_curve(scores.label, scores.seizure_probability)
        ax.step(x, y, where="post", color=ACCENT, lw=1.3, label="Saved test scores")
        ax.axhline(scores.label.mean(), color=SECONDARY, lw=.8, ls="--", label="Seizure prevalence")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve", xlim=(0, 1), ylim=(0, 1.02))
    else:
        raise ValueError("Unknown curve kind")
    dark_axes(fig, ax)
    dark_legend(ax)
    fig.tight_layout()
    return fig
