# Random Forest imbalance experiments

The optional research runner reuses the checksum-verified exports in
`logsentinel_artifacts/`. It does not change the notebook, HMM, custom Forward,
existing thresholds, or dashboard. The original dashboard dependencies and lock
remain unchanged. Install the optional dependency and run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments.txt
.\.venv\Scripts\python.exe scripts/run_imbalance_experiments.py
```

The default creates a new timestamped directory under
`outputs/imbalance_experiments/`. Use `--output outputs/imbalance_experiments/my_run`
for a named run; an existing directory is rejected, including a failed run.
`--artifacts` accepts another copy of the same frozen bundle, whose checksums must
match. Configuration is recorded from `configs/imbalance_experiments.json`.

Actual results from the completed run are in
[imbalance_results_20261113.md](imbalance_results_20261113.md).

Independently reload a trusted local run and check every holdout prediction,
threshold, metric, identity and confusion count without fitting:

```powershell
.\.venv\Scripts\python.exe scripts/verify_imbalance_run.py outputs/imbalance_experiments/run_20261113_01
```

## Features and evaluation

Each block becomes 53 numeric columns in frozen event-ID order: counts of each
event, including UNKNOWN at zero, stored as float32. Their sum is the original
sequence length. No block IDs, timestamps, severity, split names or labels are
features. No vocabulary, scaling or feature selection is fitted on holdouts.
Event order is discarded. Counts are unscaled; frequent events can dominate
SMOTE's Euclidean neighbor distances. Synthetic count vectors may be fractional
and need not represent possible HDFS event sequences.

The original strict, time-purged chronological membership is loaded and checked
against the entire split manifest. Boundary-crossing blocks remain excluded.
Both supervised models use all 357,136 training blocks, including 11,674 labeled
anomalies. The HMM and frequency reference models were fitted on normal training
blocks only, so this comparison changes supervision as well as model family. It
does not isolate the incremental benefit of imbalance treatment; an unweighted RF
control would be needed for that separate question.

The class-weight experiment uses `class_weight="balanced"`. The SMOTE experiment
uses an unweighted forest after training-only minority oversampling to 345,462
examples per class (333,788 synthetic rows). SMOTE uses five minority neighbors,
Euclidean distance, chunked brute-force neighbor search, and seed 42. Validation
and test rows are never passed to SMOTE. Their feature digests are checked again
after training. The library describes the leakage risk of resampling before a
split in its [official guidance](https://imbalanced-learn.org/stable/common_pitfalls.html).

Both forests use the same fixed settings: 200 trees, Gini impurity, bootstrap
sampling, `max_features="sqrt"`, depth at most 20, at most 4,096 leaves per tree,
and at least two samples per leaf. Seed 42 and two forest workers are fixed.
These settings are specified in advance, without a hyperparameter search.
Parallel probability accumulation can differ at floating-point roundoff level;
the saved run verifier allows 1e-12 score tolerance and requires exact decisions.

For each new model, the anomaly-class probability threshold maximizes validation
F1 over unique validation scores. Ties choose the lowest threshold. The selected
threshold is persisted before test prediction; decisions use `score >= threshold`.
Validation and test retain their original class imbalance. Average precision is
computed from continuous scores, independently of the threshold. False-alarm
rate is FP/(FP+TN). Predictions, metrics and confusion matrices use block labels
with Normal=0 and Anomaly=1.

The existing HMM and frequency thresholds are preserved. Reference metrics are
recalculated from the checksum-verified saved final scores; preliminary pilot HMM
scores are not used. All published reference metrics must agree within 1e-12.
The existing full frozen-inference test separately verifies the actual Forward
and frequency calculations against those scores.

## Memory and saved outputs

Before allocating SMOTE output, the runner reads current available physical RAM
and saves a conservative memory plan. It includes the full feature matrix,
partition copies, resampled and synthetic temporary arrays, bounded forest
storage, neighbor distance chunks and reserve space. It refuses SMOTE if the
estimate exceeds half of available RAM, if memory cannot be read, or if there
are too few minority rows. This is an estimate rather than a peak-memory
measurement; changes in other applications' memory use can still cause failure.
Windows and Linux memory inspection are supported; container memory quotas are
not detected. Forests run sequentially to limit concurrent allocations.

Every run saves configuration, input hashes, versions, class counts, feature
schema, full comparison CSV/JSON, and a completion checksum manifest. Each model
has its own directory, with validation/test metrics, exact prediction NPZs with
block identities and replay indices, and confusion matrices in CSV and SVG.
The new forests additionally save a trusted local joblib bundle containing the
estimator, threshold and feature contract. Do not load untrusted pickle/joblib
files. SMOTE's training preflight and actual class counts are recorded separately.

A failure produces `error.json` and preserves any completed model outputs.
It is never filled with synthetic metrics. Reruns require a fresh directory.

## Interpretation limits

This is a single-seed, single chronological split experiment, not an uncertainty
estimate. Validation F1 is optimistic because validation selected the threshold.
Test results are the held-out comparison. RF probabilities after balancing are
scores rather than calibrated estimates of deployment anomaly prevalence.
The raw HDFS files are absent: these experiments start from the exported event
sequences. Existing parser-design exposure and HMM convergence limitations remain
documented in `docs/notebook_provenance.md`. There is no live/early-warning or
cross-dataset validation. Do not tune another run using the reported test results.
