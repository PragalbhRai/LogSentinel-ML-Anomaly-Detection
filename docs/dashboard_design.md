# Faculty review dashboard

The existing Streamlit entry point now exposes only Overview, Model Comparison
and Anomaly Detection. Use the sidebar in that order for a faculty review.

Overview explains the task in simple English. It displays the full labeled HDFS
dataset: 575,061 blocks (558,223 normal, 16,838 anomalous), with one bar chart.
A caption distinguishes the 493,860 blocks retained for the actual experiments.

Model Comparison displays only HMM, class-weighted Random Forest and SMOTE
Random Forest, with precision, recall, F1 and false-alarm rate. One chart compares
F1; SMOTE RF is highlighted at 0.998654. The page explains training-only SMOTE,
different training supervision, and the 99.90% repeated-pattern limitation.
Scores are rounded for display; no decision threshold changes.

Anomaly Detection searches all 86,268 saved test blocks. The selector displays
up to 200 matching IDs at a time to keep the browser responsive. Click Predict
to replay each model's verified saved decision, with actual labels beside it.
Changing the block hides the old result. An empty search result offers no
prediction button or fabricated fallback. This is historical replay, not live
monitoring or new-input inference.

The dark theme uses readable text, large headings, generous spacing and stacked
cards on narrow screens. There are no heatmaps, timelines, mathematical formulas,
manual event inputs, partition selectors or threshold controls in this interface.
Unused six-page view modules have been removed. Shared helpers still needed by
the dashboard or regression tests are retained.

## Data integrity

`app/review_data.py` reuses the original verified dashboard adapter for HMM
results. It checks every file hash in the completed RF run manifest, verifies
that the run used the original bundle, aligns RF predictions by block ID, replay
index and label, checks inclusive frozen-threshold decisions, and recalculates
metrics against saved JSON and CSV results. It never loads arbitrary uploaded
models or calls fit. File changes invalidate the resource cache. Missing or
corrupt inputs stop the app before dataset metrics or predictions are shown.

The default run is `outputs/imbalance_experiments/run_20261113_01/`. An optional
`LOGSENTINEL_EXPERIMENT_DIR` points at another complete copy. Original models,
notebook, split membership, RF outputs and the existing handoff ZIP are preserved.

Tests cover three-page navigation, chart counts, actual result tables, prediction
identity, anomaly searches, stale-result handling, old-link fallback, cache
invalidation and missing/corrupt artifacts. `scripts/browser_qa.py` checks rendered
desktop/mobile views and records screenshots separately from earlier QA outputs.
