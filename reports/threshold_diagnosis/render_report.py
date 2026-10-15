"""Render the completed saved-score diagnosis; never select or evaluate test thresholds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.component.presentation.score_audit import RUNS, read_scores, distribution, alarm_statistics, protected_hashes
from src.component.presentation.train import metrics, file_hash

OUT = Path(__file__).parent
freeze = json.loads((OUT / "validation_freeze.json").read_text())
alternatives = json.loads((OUT / "exploratory_freeze.json").read_text())
held = json.loads((OUT / "held_out_results.json").read_text())
verify = json.loads((OUT / "data_verification.json").read_text())
root = Path(json.loads((RUNS["original"] / "configuration.json").read_text())["training"]["data_root"])
intervals = {str(root / row["relative_path"]): row["events"] for row in verify["records"]}
frames = {name: {split: read_scores(run / f"{split}_predictions.csv") for split in ["validation", "test"]}
          for name, run in RUNS.items()}
curves = {name: pd.read_csv(OUT / f"{name}_validation_all_thresholds.csv", float_precision="round_trip") for name in RUNS}
candidates = pd.read_csv(OUT / "validation_candidate_thresholds.csv", float_precision="round_trip")
names = {"original": "Original CNN", "balanced_bce": "Balanced BCE"}
colors = {0: "#3478bd", 1: "#d74e45"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2, "figure.dpi": 140})


def save(fig, name):
    path = OUT / name
    if path.exists():
        raise ValueError(f"Plot exists: {path}")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ecdf(ax, values, label, color, style="-"):
    values = np.sort(np.asarray(values))
    ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label, color=color, ls=style)


fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for col, (name, splits) in enumerate(frames.items()):
    val = splits["validation"]
    for label in [0, 1]:
        p = val[val.label == label].seizure_probability
        text = f"{'Seizure' if label else 'Background'} (n={len(p)})"
        axs[0, col].hist(p, bins=np.linspace(.2, .75, 45), weights=np.ones(len(p)) / len(p),
                         alpha=.55, color=colors[label], label=text)
        ecdf(axs[1, col], p, text, colors[label])
    threshold = freeze["models"][name]["policies"]["f1_under_5pct_fpr"]
    for ax in axs[:, col]:
        ax.axvline(threshold, color="black", ls="--", label=f"F1 threshold {threshold:.4f}")
        ax.set_xlabel("Saved model seizure score (uncalibrated)")
        ax.set_xlim(.2, .75)
        ax.legend(fontsize=8)
    axs[0, col].set_title(names[name] + " - validation score distribution")
    axs[0, col].set_ylabel("Fraction within each class / bin")
    axs[1, col].set_ylabel("Within-class cumulative fraction")
fig.suptitle("Validation only: 1 patient / 1 seizure event; 29 seizure and 1,773 background windows")
save(fig, "validation_distributions.png")

fig, axs = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
for col, (name, curve) in enumerate(curves.items()):
    c = curve.sort_values("threshold")
    for metric, color in [("recall", "#d74e45"), ("precision", "#3478bd"), ("f1", "#36a27c")]:
        axs[0, col].plot(c.threshold, c[metric], label=metric.capitalize(), color=color)
    axs[0, col].set_ylim(0, 1.02)
    axs[0, col].set_title(names[name])
    axs[0, col].set_ylabel("Validation window metric")
    axs[1, col].plot(c.threshold, c.false_positive_rate * 100, label="Window FPR (%)", color="#a66b24")
    axs[1, col].axhline(5, color="black", ls=":", label="5% cap")
    axs[1, col].set_ylim(0, 12)
    axs[1, col].set_ylabel("Window FPR (%)")
    axs[2, col].plot(c.threshold, c.false_alarm_episodes_per_nonseizure_hour, color="#7b52a6", label="Pure false episodes / hour")
    axs[2, col].axhline(1, color="black", ls=":", label="Diagnostic 1 episode/h budget")
    axs[2, col].set_ylabel("False episodes / nonseizure hour")
    for ax in axs[:, col]:
        ax.axvline(freeze["models"][name]["policies"]["f1_under_5pct_fpr"], color="black", ls="--", label="Maximum F1")
        ax.axvline(alternatives["models"][name]["recall_cap_5percent"], color="gray", ls="-.", label="Maximum recall, <=5% FPR")
        ax.set_xlim(.3, .65)
        ax.set_xlabel("Decision threshold (>=)")
        ax.legend(fontsize=8)
fig.suptitle("Validation operating points: no test-set threshold optimization")
save(fig, "validation_threshold_tradeoffs.png")

fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for col, (name, splits) in enumerate(frames.items()):
    for label in [0, 1]:
        ax = axs[label, col]
        for split, color in [("validation", "#3478bd"), ("test", "#d74e45")]:
            p = splits[split].query("label == @label").seizure_probability
            ecdf(ax, p, f"{'chb05 validation' if split == 'validation' else 'chb08 test (descriptive)'} n={len(p)}", color)
        ax.axvline(freeze["models"][name]["policies"]["f1_under_5pct_fpr"], color="black", ls="--", label="Frozen maximum-F1 threshold")
        ax.axvline(alternatives["models"][name]["recall_cap_5percent"], color="gray", ls="-.", label="Frozen recall <=5% threshold")
        ax.set_title(names[name] + (" - seizure windows" if label else " - background windows"))
        ax.set_xlabel("Uncalibrated seizure score")
        ax.set_ylabel("Within-class cumulative fraction")
        ax.set_xlim(.2, .75)
        ax.legend(fontsize=8)
fig.suptitle("Patient-associated score shift: descriptive test inspection AFTER decisions were frozen")
save(fig, "patient_score_shift.png")

fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for col, (name, splits) in enumerate(frames.items()):
    for row, split in enumerate(["validation", "test"]):
        f = splits[split]
        path = next(p for p in f.path.unique() if intervals[p])
        s, e = intervals[path][0]
        data = f[(f.path == path) & (f.start_sec >= s - 60) & (f.end_sec <= e + 60)]
        ax = axs[row, col]
        ax.step(data.start_sec, data.seizure_probability, where="post", color="#3478bd", label="Model score")
        ax.axvspan(s, e, color="#d74e45", alpha=.15, label="Annotated seizure")
        ax.axhline(freeze["models"][name]["policies"]["f1_under_5pct_fpr"], color="black", ls="--", label="Frozen maximum-F1 threshold")
        ax.axhline(alternatives["models"][name]["recall_cap_5percent"], color="gray", ls="-.", label="Frozen recall <=5% threshold")
        ax.set_title(f"{names[name]} - {split}: {Path(path).name}")
        ax.set_xlabel("Recording time (seconds)")
        ax.set_ylabel("Uncalibrated seizure score")
        ax.set_ylim(.2, .75)
        ax.legend(fontsize=8)
fig.suptitle("Event score timelines (test shown descriptively; no thresholds selected from this plot)")
save(fig, "event_score_timelines.png")

fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
for name, splits in frames.items():
    f = splits["validation"]
    precision, recall, _ = precision_recall_curve(f.label, f.seizure_probability)
    fpr, tpr, _ = roc_curve(f.label, f.seizure_probability)
    vm = metrics(f, freeze["models"][name]["policies"]["saved"])
    axs[0].plot(recall, precision, label=f"{names[name]} AP={vm['pr_auc']:.3f}")
    axs[1].plot(fpr, tpr, label=f"{names[name]} AUC={vm['roc_auc']:.3f}")
axs[0].axhline(29 / 1802, color="gray", ls=":", label="Validation prevalence")
axs[1].plot([0, 1], [0, 1], color="gray", ls=":", label="Chance ordering")
axs[0].set(xlabel="Recall / sensitivity", ylabel="Precision", title="Validation precision-recall")
axs[1].set(xlabel="False-positive rate", ylabel="Sensitivity", title="Validation ROC")
for ax in axs:
    ax.legend(fontsize=9)
fig.suptitle("Validation discrimination; 29 windows from ONE event, not 29 independent seizures")
save(fig, "validation_discrimination.png")


def table(columns, rows):
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


def pct(value):
    return f"{value * 100:.2f}%"


pooled, shift = [], []
for name, splits in frames.items():
    for split, f in splits.items():
        for label in [0, 1]:
            p = f[f.label == label].seizure_probability
            pooled.append({"model": name, "split": split, "patient": "chb05" if split == "validation" else "chb08",
                           "label": label, "n": len(p), "mean": p.mean(), "median": p.median(),
                           "p10": p.quantile(.1), "p90": p.quantile(.9), "min": p.min(), "max": p.max()})
    for label in [0, 1]:
        val = splits["validation"].query("label == @label").seizure_probability
        test = splits["test"].query("label == @label").seizure_probability
        shift.append({"model": name, "label": label, "validation_median": val.median(),
                      "test_median": test.median(), "median_test_minus_validation": test.median() - val.median(),
                      "validation_p90": val.quantile(.9), "test_p90": test.quantile(.9)})
pd.DataFrame(pooled).to_csv(OUT / "pooled_score_distributions.csv", index=False, mode="x")
pd.DataFrame(shift).to_csv(OUT / "patient_score_shifts.csv", index=False, mode="x")

val_policies = []
test_policies = []
for name in RUNS:
    policies = {key: freeze["models"][name]["policies"][key] for key in ["saved", "recall_guarded"]}
    policies.update(alternatives["models"][name])
    for key, t in policies.items():
        vm = {**metrics(frames[name]["validation"], t), **alarm_statistics(frames[name]["validation"], t, intervals)}
        val_policies.append({"model": name, "policy": key, **vm})
        test_policies.append({"model": name, "policy": key, **held["models"][name]["policies"][key]})
pd.DataFrame(val_policies).to_csv(OUT / "frozen_policies_validation.csv", index=False, mode="x")
pd.DataFrame(test_policies).to_csv(OUT / "frozen_policies_test.csv", index=False, mode="x")

texts = ["# Saved-score CNN diagnosis — 14 October 2026\n",
"The original and Balanced BCE CNNs both fail to detect the chb08 seizure at their saved operating points. Validation-only alternatives also detect **0/43 test seizure windows and 0/1 test events**. Lower diagnostic thresholds introduce false positives rather than recover detection. Keep the existing frontend, original checkpoint and default threshold unchanged.\n",
"## Scope and evaluation status\n",
"No model was trained, no new inference or downloads were performed, and neither checkpoint nor any `demo/*.py` / Streamlit configuration file was modified. This audit uses actual saved scores, with round-trip float parsing and the exact `score >= threshold` rule. Hashes protect inputs and frontend files.\n",
"**There is no untouched final test set available in this subset.** chb08 was evaluated during both previous experiments and the original threshold audit. It remains excluded from model training and validation threshold selection, but this report is an exploratory reanalysis of a previously examined test patient. No claim of fresh confirmatory evaluation is justified. Scores were read only after validation decisions were frozen; no threshold was selected or revised using test performance.\n",
"## Validation distribution and candidate thresholds\n",
"Validation: chb05_01 and chb05_06, 1,802 complete four-second windows (1,773 background; 29 seizure windows from one event). Distributions below are pooled by class; per-recording quantiles are in `validation_distributions.csv`. Scores are softmax outputs, not established calibrated clinical risks.\n",
table(["Model", "Class", "n", "Mean", "Median", "10–90%", "Range"],
      [[names[r['model']], 'Seizure' if r['label'] else 'Background', r['n'], f"{r['mean']:.4f}", f"{r['median']:.4f}", f"{r['p10']:.4f}–{r['p90']:.4f}", f"{r['min']:.4f}–{r['max']:.4f}"] for r in pooled if r['split'] == 'validation']),
"![Validation class score distributions](validation_distributions.png)\n",
"The seizure median is below the background median for both models. Only the high-score tail of this one seizure is separable at low false-positive rates; lowering a global threshold rapidly admits background windows.\n",
"Selection rules, fixed before accessing test scores:\n\n- Reference: each checkpoint's saved threshold. Original used legacy-grid validation F1, not the new 5% policy.\n- Primary comparison: maximum validation F1 with window FPR ≤5%; ties fewer FP, then higher threshold.\n- Guarded diagnostic: maximum validation recall with window FPR ≤1% AND ≤1 purely false episode/nonseizure hour; ties higher F1, fewer FP, higher threshold. This is a declared engineering budget, not a clinical standard.\n- Cost diagnostics: maximum validation recall under 0.5% or 5% window FPR, same ties. These are exploratory alternatives, not deployment recommendations.\n",
table(["Model / validation policy", "Threshold", "Precision", "Recall = sensitivity", "F1", "Specificity", "FP windows", "False episodes/h"],
      [[names[r['model']] + ' / ' + r['policy'], f"{r['threshold']:.9f}", pct(r['precision']), pct(r['recall']), f"{r['f1']:.4f}", pct(r['specificity']), r['false_positives'], f"{r['false_alarm_episodes_per_nonseizure_hour']:.2f}"] for r in val_policies]),
"All distinct reachable >= decisions, including the no-alert decision, were swept on validation only. The full CSVs contain precision, recall, sensitivity, F1, specificity, FP/FN/TP/TN, FPR and alarm burden for every threshold. `validation_candidate_thresholds.csv` additionally summarizes F1 and recall objectives at caps 0%, 0.1%, 0.5%, 1%, 2%, 5%, 10%, and unconstrained. No test threshold sweep was run.\n",
"![Validation threshold tradeoffs](validation_threshold_tradeoffs.png)\n",
"## Why the 5% cap yields missed held-out seizures\n",
"The cap itself is **not binding** at maximum F1: for BOTH models, unconstrained maximum validation F1 selects the same threshold as the 5% cap, with 6 TP, 23 FN and 0 FP (recall 20.69%, F1 0.342857). Removing the cap while retaining the F1 objective does not fix this. F1 favors a small high-scoring seizure tail with perfect precision over extra seizure windows buried in background. This audit's maximum-F1 thresholds are 0.530214846 for original and 0.567425787 for Balanced BCE; the saved checkpoints and dashboard threshold remain unchanged.\n",
"Using the cap to maximize recall instead gives 8/29 TP with 47 FP for original, or 9/29 TP with 68 FP for Balanced BCE. This only increases validation recall to 27.59% / 31.03%, while causing 19.79 / 31.47 purely false episodes/hour and precision 14.55% / 11.69%. These rates are poor for a monitoring demonstration that treats each episode as an alert.\n",
"Balanced BCE's milder 0.530476272 threshold gives 7/29 TP, 6 FP, precision 53.85%, F1 0.333333, and 3.05 false episodes/hour. It gains just one validation seizure window versus maximum F1. Whether three nuisance alerts/hour is acceptable would need an explicit use-case budget; it is not treated as justified for automatic deployment. The guarded ≤1 episode/hour rule recovers no additional validation seizure windows for either model.\n",
"## Patient-associated shifts and input checks\n",
table(["Model", "Class", "chb05 median", "chb08 median", "Median shift", "chb05 / chb08 p90"],
      [[names[r['model']], 'Seizure' if r['label'] else 'Background', f"{r['validation_median']:.4f}", f"{r['test_median']:.4f}", f"{r['median_test_minus_validation']:+.4f}", f"{r['validation_p90']:.4f} / {r['test_p90']:.4f}"] for r in shift]),
"![Patient score shifts](patient_score_shift.png)\n",
"chb08's seizure-score maxima are 0.476711154 (original) and 0.466002792 (Balanced BCE), below even the 5%-cap recall thresholds (0.489078879 / 0.486873358). This explains zero TP at every frozen candidate. A downward background shift also reduces FP on test compared with validation. These are **patient-associated descriptive shifts**, confounded by event morphology and recording conditions; one patient/event per split cannot isolate a patient effect statistically. No chb01/chb03 training-score predictions were saved, so a full training-patient score distribution cannot be reported without additional inference, which this audit does not perform.\n",
"Verified against actual local files:\n\n- Train patients chb01/chb03 (six EDFs), validation chb05 (two), test chb08 (two); no shared patient group, repeated EDF fingerprint or recording across splits. Existing chb01/chb21 identity alias is preserved; PhysioNet identifies them as the same subject. [Dataset documentation](https://physionet.org/content/chbmit/1.0.0/chb21/)\n- All ten EDF and matching annotation-summary hashes agree with BOTH saved checkpoint provenance records; both models used identical records/splits.\n- Eight ordered channels: FP1-F7, F7-T7, T7-P7, P7-O1, FP2-F4, F4-C4, C4-P4, P4-O2; 256 Hz, shape (8, 1024), four-second non-overlapping windows. No silent channel padding or resampling.\n- Shared 1–40 Hz, order-4 zero-phase SOS filtering with up to two seconds of surrounding context, crop, then per-channel per-window z-score. Live EDF examples from every recording gave identical lazy-training and shared-inference preprocessing arrays, finite values, near-zero mean and unit standard deviation. `preprocessing_examples.csv` records 16 examples.\n- Every saved training/validation/test window index matches regenerated local annotation/timing labels: seizure-union overlap ≥50%, complete windows only, trailing partial windows discarded (two seconds in chb05_01). Context remains within a single recording/patient, so it does not cross a data split. This offline filter uses future samples and is not a causal early-prediction evaluation.\n",
"The verified consistency rules out the checked timing/channel/normalization mismatches; it does not prove feature adequacy. Z-score removes absolute amplitude information, eight channels omit other available spatial information, and the existing architecture includes BatchNorm and global average pooling. Their contribution to poor cross-patient transfer is a hypothesis, not an established causal finding from these scores.\n",
"## False alarms and event interpretation\n",
"An alarm episode is a maximal run of contiguous predicted-positive windows within one EDF. There is no smoothing, gap merging, refractory period or merging between files. A **purely false episode** has zero overlap with any ground-truth seizure. A seizure-touching episode is counted as event detection; its background duration is still reported so a long alarm is not made to look harmless. This retrospective event rule uses any positive temporal overlap, whereas window metrics use the existing ≥50% labeling rule.\n",
"Validation exposure is 7,208 covered seconds minus 115 annotated seizure seconds = **1.970278 nonseizure hours**. Test exposure is 7,200 minus 171 = **1.952500 nonseizure hours**. Background-label-window exposure differs slightly at seizure boundaries; the CSVs keep that denominator separate for FP-window/hour. A 5% window FPR corresponds to about 45 FP four-second windows per background hour, NOT necessarily 45 alarm episodes. Event rates depend on adjacency and are computed directly here. Also report background-alert seconds/hour, rather than assume episode count captures alarm duration.\n",
"At maximum-F1 thresholds validation detects 1/1 event but only 6/29 positive windows. That 100% event fraction is a single observed event, not robust event sensitivity. On test every frozen policy detects 0/1 event. Zero observed false alarms over roughly two hours does not establish a low population false-alarm rate; correlated windows and one event prevent credible narrow confidence bounds.\n",
"## Frozen held-out evaluation — exploratory, not a fresh final test\n",
table(["Model / frozen policy", "Threshold", "Precision", "Recall = sensitivity", "F1", "Specificity", "FP", "False episodes/h", "Confusion [[TN,FP],[FN,TP]]"],
      [[names[r['model']] + ' / ' + r['policy'], f"{r['threshold']:.9f}", pct(r['precision']), pct(r['recall']), f"{r['f1']:.4f}", pct(r['specificity']), r['false_positives'], f"{r['false_alarm_episodes_per_nonseizure_hour']:.3f}", r['confusion_matrix']] for r in test_policies]),
"Zero-positive-prediction precision is reported as zero (the implementation's zero_division=0 convention), not evidence of zero-error detection. Distinct thresholds were each evaluated once in this audit (three per model); duplicate policy thresholds reuse those computed results. The test phase refuses to run a second time in the same report directory. The exact full precision, sensitivity, specificity, counts, rates and AUC/AP are in `held_out_results.json` and `frozen_policies_test.csv`. No threshold was revised after viewing these results.\n",
"![Event score timelines](event_score_timelines.png)\n",
"## Discrimination, imbalance and training evidence\n",
"![Validation discrimination](validation_discrimination.png)\n",
"The models were selected at epoch 1. Both saved histories show falling training loss but worsening class-balanced validation cross-entropy in later epochs, consistent with overfitting to the small subset. Loss values between original weighted CE and candidate balanced BCE are different objectives and should not be directly ranked. Training retained 47 seizure windows and 2,400 background windows; original natural sampling yields about 73.3% all-background batches at batch size 16. Balanced BCE eliminates those batches but repeats only 47 unique seizure windows across 1,440 positive draws over three epochs. Repeated draws are not new seizure diversity.\n",
"Validation was already used to choose epochs and thresholds, and this audit explores additional policies on that same one event. Validation performance is therefore selection-dependent, not an independent estimate of generalization. No window-wise significance tests or narrow confidence intervals are claimed.\n",
"Test AUC/AP, descriptive only: original 0.825482 / 0.158814; Balanced BCE 0.667893 / 0.064731. Rank separation on the one test event is better for original despite low absolute scores; neither gives a defensible validation-derived alert threshold with useful test recall. Balanced sampling alone did not resolve transfer. Score calibration was not fitted or established. Calibration requires independent calibration/evaluation data; a monotone remapping cannot rescue poor rank discrimination. [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)\n",
"## Recommendations\n",
"1. Keep the existing dashboard model and threshold unchanged; retain the prominent zero-held-out-recall warning and distinct annotated versus CNN alerts. Neither threshold relaxation nor Balanced BCE promotion is supported by these measurements.\n2. Define an alert budget in episodes/hour **and** background-alert duration, then preregister the selection objective and temporal postprocessing. A 5% window FPR cap alone is too permissive for monitoring and does not force better recall. Do not use chb08 to tune this budget, thresholds, model epochs, or smoothing.\n3. Before another model experiment, review per-channel filtered traces and artifacts around the annotated events, preserving absolute amplitude descriptors alongside z-scores; compare channel coverage and acquisition differences. The normalization and architecture hypotheses need controlled validation tests, not an asserted fix.\n4. Meaningful threshold transfer and calibration need multiple independent validation patients and events, and a NEW untouched test cohort. The present 47 unique training-positive windows from four events/two patients are insufficient to substantiate generalization. Additional patient/event diversity is genuinely needed for a credible research claim, though it does not guarantee improvement. No data was downloaded here.\n5. If future experiments are authorized, vary one factor at a time (amplitude-preserving features, normalization/BatchNorm alternative, channel coverage or temporal aggregation) with patient-grouped validation and a fixed resource budget. Score calibration should follow adequate discrimination and must use separate/grouped calibration data. Do not automatically schedule more training.\n",
"## Reproduction and files\n",
"Run from the project root using a NEW output directory (the audit never overwrites reports):\n\n```bat\n.venv\\Scripts\\python.exe -m src.component.presentation.score_audit --phase validation --output reports\\threshold_diagnosis_run2\n.venv\\Scripts\\python.exe -m src.component.presentation.score_audit --phase freeze-alternatives --output reports\\threshold_diagnosis_run2\n.venv\\Scripts\\python.exe -m src.component.presentation.score_audit --phase test --output reports\\threshold_diagnosis_run2\n```\n\nThese commands are for audit reproducibility, not a request to rerun test evaluation or reopen threshold tuning. They do not train or run neural-network inference. `render_report.py` renders this directory's frozen outputs and refuses existing plots/tables.\n",
"Evidence: `validation_freeze.json`, `exploratory_freeze.json`, `held_out_results.json`, `data_verification.json`, all-threshold CSVs, candidate table, distribution and shift CSVs, preprocessing examples and five plots. Source: `src/component/presentation/score_audit.py`; regression tests: `src/tests/test_presentation_score_audit.py`. Test and preservation verification are recorded separately in `verification.json`.\n"]

if protected_hashes() != freeze["protected_hashes_before"]:
    raise ValueError("Protected files changed.")
if (OUT / "verification.json").is_file():
    verified = json.loads((OUT / "verification.json").read_text())
    tests = verified["automated_tests"]
    texts.append(f"Verification: **{tests['passed']} passed, {tests['skipped']} skipped, {tests['failed']} failed**. "
                 "Protected frontend and checkpoint files are unchanged; all validation thresholds independently checked "
                 "and all five plots visually inspected. See `verification.json` for the full evidence.\n")
with (OUT / "REPORT.md").open("x", encoding="utf-8") as handle:
    handle.write("\n".join(texts))
print("Report and five plots written; no test threshold evaluation or training performed.")
