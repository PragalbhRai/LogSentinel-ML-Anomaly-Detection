LogSentinel separates sequence inference from replay metadata and evaluation.

Phase B adds optional observational tracing to the original Forward recurrence.
`InferenceEngine.explain()` returns its scaling factors, cumulative likelihood and
filtered state weights without changing arithmetic or classification. Pages use
this backend API instead of implementing another Forward calculation.

```text
raw text -> original parser -> frozen vocabulary -> block event IDs
                                                    |
saved events + offsets -----------------------------+
                                                    v
                                          InferenceEngine
                                 custom Forward / frequency / Markov
                                                    |
                                      mean NLL and fixed >= decision

ground-truth labels -> evaluation and historical comparisons only
```

`load_models()` checks hashes before trusted joblib loading, validates the exact
three-state/53-symbol contract and copies probability arrays as read-only values.
Vocabulary and thresholds are read-only mappings. It does not load label vectors
or replay datasets. No public prediction method accepts labels or training data.

`load_replay()` verifies the paired metadata/NPZ hashes, monotonic offsets, IDs,
event counts, chronological row order, time intervals, whole-block split
membership and equality with the full manifest's retained subset. Sequence
identity is positional, which is why both hashes and ordering checks matter.
Replay arrays are read-only. Its pandas metadata is for exploration; callers
should copy before transforming it and never pass it into the scoring layer.

`verify_frozen_inference()` recalculates every validation and test score for all
three methods using scalar, notebook-equivalent arithmetic. It checks score
agreement at absolute tolerance 1e-12, exact prediction equality, label/block
alignment, test metrics and original checksums. Tolerance never changes a score,
prediction or threshold. No threshold search is performed.

Runtime paths are explicit arguments to the loaders. This version supports an
editable repository install; the default artifact/manifest paths are repository
relative. A standalone distributable model bundle is a future packaging task.
The checksum manifest and source code must be trusted together; user-supplied
pickle/joblib uploads are not supported.

The verified integration environment deliberately differs from Colab's recorded
NumPy and joblib versions. Its exact Python and resolved libraries are captured
in `manifests/integration_environment.json` and `requirements.lock.txt`. The
original dependency export is not edited. The integration environment supports
inference and testing, not replaying the notebook's dataset downloads/training.

Limitations: raw logs and dataset hashes are absent; the five real raw lines in
notebook output provide a prefix parity fixture, not a complete raw-dataset
round-trip. The full exported encoded sequences are checked end to end. Events
have positions but no individual timestamps. Block timestamps have no recorded
timezone. UNKNOWN exists but was not observed in retained experimental data.
