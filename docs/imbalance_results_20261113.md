# Completed imbalance experiments: 13 November 2026

Both experiments ran successfully on the existing exported HDFS block sequences.
The run is [outputs/imbalance_experiments/run_20261113_01](../outputs/imbalance_experiments/run_20261113_01/).
Full precision results are in its [comparison.csv](../outputs/imbalance_experiments/run_20261113_01/comparison.csv)
and [results.json](../outputs/imbalance_experiments/run_20261113_01/results.json).
No dashboard changes were made.

## Inspection and setup

Inspection covered the original 26-cell notebook, backend, dashboard and its six
views, tests, scripts, configuration, dependency records, artifact manifests,
existing verification outputs and handoff structure. The raw/processed/split
data directories and `artifacts/` contain placeholders. The project directory
has no Git repository metadata, so preservation was verified with file hashes.

The 22 original exports include a flat `uint16` event array of length 9,625,001,
493,861 `int64` sequence offsets, retained metadata, full split manifest, the
53-symbol vocabulary, final HMM, normal-only baseline probabilities, final
validation/test scores, and preliminary HMM candidate scores. These experiments
used the final scores for reference comparisons, not the pilot scores.

The original checksum manifest and `load_replay()` verified sequence identity,
event lengths, row ordering, complete-block chronological boundaries and frozen
membership. The split remains:

| Partition | Normal | Anomalous | Total |
| --- | ---: | ---: | ---: |
| Train | 345,462 | 11,674 | 357,136 |
| Validation | 49,485 | 971 | 50,456 |
| Test | 85,152 | 1,116 | 86,268 |

The 81,201 boundary-crossing blocks remain excluded. Every block became 53
float32 event-count features, including UNKNOWN, in frozen vocabulary order.
Block IDs, timestamps and labels were excluded from the features. The feature
counts sum exactly to original sequence lengths.

Both forests used seed 42, 200 trees, depth at most 20, at most 4,096 leaves per
tree, at least two samples per leaf, bootstrap sampling, square-root feature
sampling and two workers. No hyperparameter search was run. Full settings and
package versions are saved in the run's `config.json` and `provenance.json`.

## Actual held-out test results

Positive class is Anomaly. Precision/recall/F1/AP and false-alarm rate below are
fractions, rounded to six decimals. False-alarm rate is FP/(FP+TN).

| Model | Precision | Recall | F1 | Average precision | False-alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF, balanced class weights | 1.000000 | 0.993728 | 0.996854 | 0.999912 | 0.000000 |
| RF, training-only SMOTE | 1.000000 | 0.997312 | 0.998654 | 0.999826 | 0.000000 |
| Existing HMM | 0.695449 | 0.534050 | 0.604156 | 0.692690 | 0.003065 |
| Existing frequency baseline | 0.808696 | 1.000000 | 0.894231 | 0.887492 | 0.003100 |

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| RF, balanced class weights | 85,152 | 0 | 7 | 1,109 |
| RF, training-only SMOTE | 85,152 | 0 | 3 | 1,113 |
| Existing HMM | 84,891 | 261 | 520 | 596 |
| Existing frequency baseline | 84,888 | 264 | 0 | 1,116 |

On this held-out split, SMOTE has the stronger RF recall and F1; the weighted
forest has slightly higher AP. The frequency baseline detects all anomalies,
with 264 false alarms. These are observations from one test comparison, not a
test-based tuning or deployment-selection procedure.

## Validation thresholds and metrics

The new thresholds maximize validation F1, with the lowest threshold breaking
ties and inclusive `score >= threshold` decisions. They were saved before test
prediction. Original reference thresholds were loaded unchanged.

| Model | Threshold | Precision | Recall | F1 | AP | False-alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF, class weights | 0.5338195658225056 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| RF, SMOTE | 0.6095269337467838 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Existing HMM | 2.991257009736873 | 0.973896 | 0.499485 | 0.660313 | 0.746911 | 0.000263 |
| Existing frequency | 2.4509147402999014 | 0.984787 | 1.000000 | 0.992335 | 0.989830 | 0.000303 |

Both RF validation matrices are TN=49,485, FP=0, FN=0, TP=971. Validation metrics
are selection-set measurements; the held-out test table is the final comparison.

## SMOTE memory and partition safeguards

SMOTE used five minority neighbors and seed 42. It received only the 357,136
training rows and created 333,788 synthetic rows, producing 345,462 rows per class
(690,924 total). No validation or test resampling occurred. Holdout feature
digests were unchanged after training.

Immediately before SMOTE, available physical RAM was 11,750,408,192 bytes
(10.94 GiB). The conservative working estimate was 3,048,431,968 bytes (2.84 GiB),
below the configured 50% available-memory budget of 5,875,204,096 bytes (5.47 GiB).
The resampled feature/label arrays alone were estimated at 152,003,280 bytes
(144.96 MiB). These are preflight estimates, not measured peak memory.
The plan is saved in `rf_smote/memory_preflight.json`, and actual row counts are
saved in `rf_smote/resampling.json`.

Total runner time was 100.84 seconds. Fitting plus resampling took 39.24 seconds
for class weighting and 49.56 seconds for SMOTE. Timing is machine-dependent.

## Saved artifacts and verification

Each new model is saved separately as `rf_class_weight/model.joblib` and
`rf_smote/model.joblib`. Each bundle includes the estimator, frozen threshold,
feature schema and configuration. All four compared models have their own
validation/test metrics, prediction NPZs with block IDs and replay indices, and
confusion matrices in CSV and standalone SVG form.

All 103 tests passed, including six new experiment tests, the full frozen-score
regression for all existing methods, independent Forward checks and Streamlit
interaction checks. The test evidence is
[imbalance_experiments_regression_20261113.xml](../outputs/imbalance_experiments_regression_20261113.xml).

The independent saved-run verifier passed for all four models on all 50,456
validation and 86,268 test blocks, with zero prediction mismatches. Reloaded
weighted-RF scores differed by at most 2.22e-16 due to parallel floating-point
accumulation, below the 1e-12 tolerance; SMOTE-RF and reference scores reproduced
exactly. Validation-optimal thresholds, metrics and every confusion count were
verified. See [verification.json](../outputs/imbalance_experiments/run_20261113_01/verification.json).

All 49 pre-existing files in the preservation snapshot remained byte-for-byte
unchanged, including the original notebook, all 22 exports, original backend,
dashboard files and original configuration/manifests. The snapshot is
[imbalance_inspection_20261113.json](../outputs/imbalance_inspection_20261113.json).
Only new experiment files and documentation were added, plus a README pointer.
The existing handoff ZIP was not rebuilt.

## Errors and limitations

The first dependency-install attempt was blocked by sandbox network permissions;
an approved retry installed `imbalanced-learn==0.14.0`. Existing dependencies
remained at their locked versions, and `pip check` passed.

During SMOTE, joblib warned that Windows physical-core detection could not find
its system helper and fell back to logical-core detection. Execution completed;
forest worker count and neighbor thread limits were explicitly configured.
There were no experiment failures or fabricated fallback results.

The RFs are supervised with anomalous training labels, whereas the existing HMM
and frequency models were fitted only on normal blocks. Gains therefore cannot
be attributed solely to imbalance handling. There is no unweighted RF control,
multi-seed confidence interval, alternate time split or cross-dataset evaluation.
Zero false alarms here does not guarantee zero future false alarms.

Count features discard event order. SMOTE uses unscaled Euclidean count
distances; frequent events can dominate neighbors, and fractional synthetic
vectors may not correspond to real log sequences. Balancing also means RF
scores are not calibrated deployment probabilities. Raw logs are absent, and
the original parser-design exposure and HMM convergence limitations remain.
No live, early-warning or dashboard integration claims are made.
