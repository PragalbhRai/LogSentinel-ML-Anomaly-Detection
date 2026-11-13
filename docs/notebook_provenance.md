The original `logsentinel.ipynb` remains at the repository root, unchanged. It
contains execution outputs and its SHA-256 is recorded alongside the original
exports in `manifests/artifact_checksums.json`. Cell numbers below are zero-based.
No training cell is executed by the package or tests.

| Cells | Original behavior | Reusable implementation |
| --- | --- | --- |
| 4, 6, 8–9 | Raw audit, first-block metadata, strict time-purged membership | Frozen membership is loaded and checked; not regenerated |
| 10 | Five ordered regex substitutions; six-field parser | `preprocessing.py` |
| 11–12 | Parser diagnostics; component-aware vocabulary counted on all strict training blocks | Existing vocabulary loaded unchanged |
| 13 | UNKNOWN=0; deterministic sorted template IDs; traces preserve raw line order and first block ID | `preprocessing.py`, `artifacts.py` |
| 18–19 | Scaled Forward recurrence and reference parity | `forward.py`, `test_numerics.py` |
| 21 | Negative log-likelihood divided by sequence length | `inference.py` |
| 21 | Candidates ranked by validation AP, then F1, descending | Documented; selection is never rerun |
| 22 | Final three-state model initialized from pilot, trained on all normal training blocks | Existing model loaded; no training implementation |
| 23 | Final validation F1 threshold; inclusive classification | Exact frozen threshold and `>=` |
| 24 | Frequency and first-order Markov likelihoods, add-one normal-training estimates | Existing probabilities loaded; scoring only |

The candidate model's threshold and AP differ from the final model because the
selected pilot was followed by full normal-training fitting and final validation
scoring. The notebook explicitly selects three states by AP, even though the
five-state candidate has better validation F1.

The parser originally assumes a six-field line, creates `20YYMMDDHHMMSS` strings,
and does not validate dates. The integration preserves these parsing semantics.
Malformed block-bearing lines now raise an explicit error at sequence assembly
rather than causing a None-subscript failure. Unassociated lines are skipped.
No timestamp sorting, deduplication, event filtering or truncation is introduced.
Multiple block IDs retain the original first-match behavior; the notebook's saved
HDFS_v1 audit reports zero lines with multiple distinct block IDs. New-input
results using this rule must be described accordingly.

Vocabulary fitting uses all strict training blocks, including anomalous training
blocks, without using labels to define the event ID. HMM and baseline fitting use
only normal training blocks. Cell 11 also inspects a systematic sample across the
whole timeline and anomalous training examples. Thus parser-design independence
from holdout content cannot be proved; report this exposure as a methodological
limitation, even though the saved rules precede that diagnostic cell and the
vocabulary is training-only. Test-label use for tuning is not shown in the code.

The numerical core of Forward is retained; the integration rejects fractional,
boolean and out-of-range event IDs before conversion. An impossible observation
has log likelihood `-inf` and anomaly score `+inf`. The saved model probabilities
are strictly positive, so valid exported sequences produce finite scores.

The notebook reports strict convergence as false after 35 full-training iterations.
It is preserved as an experimental limitation, not corrected by retraining.
