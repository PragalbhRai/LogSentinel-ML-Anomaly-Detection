"""Validated loading of the trusted, frozen Colab export.

joblib uses pickle: only the original checksum-verified model is loaded. Export
checksums are checked before deserialization, not treated as authentication for
arbitrary uploaded models. No checksum bypass is exposed by the loader.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM

from .inference import FINAL_THRESHOLD
from .validation import probability_array

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "logsentinel_artifacts"
CHECKSUM_MANIFEST = PROJECT_ROOT / "manifests/artifact_checksums.json"


class ArtifactError(ValueError):
    """The export is missing, corrupt, incompatible or internally inconsistent."""


def _require(condition, message):
    if not condition:
        raise ArtifactError(message)


def verify_checksums(directory=DEFAULT_ARTIFACT_DIR, names=None):
    """Verify selected exports, or all 22 exports; never rewrite originals."""
    directory = Path(directory)
    try:
        manifest = json.loads(CHECKSUM_MANIFEST.read_text(encoding="utf-8"))["files"]
        expected = {k.removeprefix("logsentinel_artifacts/"): v for k, v in manifest.items()
                    if k.startswith("logsentinel_artifacts/")}
        selected = list(expected) if names is None else list(names)
        for name in selected:
            _require(name in expected, f"No frozen checksum for {name}.")
            with (directory / name).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            _require(digest == expected[name], f"Checksum mismatch: {name}.")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot verify artifact bundle: {exc}") from exc


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def validate_vocabulary(vocabulary):
    _require(isinstance(vocabulary, dict) and len(vocabulary) == 53,
             "Expected the frozen 53-symbol vocabulary.")
    _require(all(isinstance(k, str) and type(v) is int for k, v in vocabulary.items()),
             "Vocabulary must map strings to integer IDs.")
    _require(vocabulary.get("<UNKNOWN>") == 0 and sorted(vocabulary.values()) == list(range(53)),
             "Vocabulary IDs must be unique, contiguous, with UNKNOWN=0.")


@dataclass(frozen=True)
class ModelBundle:
    vocabulary: object
    startprob: np.ndarray
    transmat: np.ndarray
    emissionprob: np.ndarray
    event_prob: np.ndarray
    start_prob: np.ndarray
    transition_prob: np.ndarray
    thresholds: object


def load_models(directory=DEFAULT_ARTIFACT_DIR) -> ModelBundle:
    """Load small inference artifacts without loading labels or replay datasets."""
    directory = Path(directory)
    names = ["hmm_final_3state.joblib", "event_vocabulary.json", "baseline_models.npz",
             "hmm_final_evaluation.json", "final_model_comparison.csv"]
    verify_checksums(directory, names)
    try:
        vocabulary = read_json(directory / names[1])
        validate_vocabulary(vocabulary)
        model = joblib.load(directory / names[0])
        _require(isinstance(model, CategoricalHMM), "Expected a CategoricalHMM.")
        _require(model.n_components == 3 and model.n_features == len(vocabulary),
                 "Model dimensions do not match the frozen 3-state/53-symbol contract.")
        report = read_json(directory / "hmm_final_evaluation.json")
        _require(report["threshold"] == FINAL_THRESHOLD and report["hidden_states"] == 3
                 and report["anomaly_score"] == "negative_log_likelihood_per_event",
                 "Final scoring contract or HMM threshold has changed.")
        comparison = read_csv(directory / "final_model_comparison.csv").set_index("model")
        _require(comparison.index.is_unique and set(comparison.index) == {"HMM_3state", "frequency", "markov"},
                 "Unexpected model comparison rows.")
        thresholds = {"hmm": FINAL_THRESHOLD,
                      "frequency": float(comparison.loc["frequency", "threshold"]),
                      "markov": float(comparison.loc["markov", "threshold"])}
        _require(comparison.loc["HMM_3state", "threshold"] == FINAL_THRESHOLD
                 and np.isfinite(list(thresholds.values())).all(), "Invalid thresholds.")
        with np.load(directory / "baseline_models.npz", allow_pickle=False) as baseline:
            return ModelBundle(
                MappingProxyType(vocabulary),
                probability_array(model.startprob_, (3,), "HMM start"),
                probability_array(model.transmat_, (3, 3), "HMM transitions"),
                probability_array(model.emissionprob_, (3, 53), "HMM emissions"),
                probability_array(baseline["event_prob"], (53,), "Frequency events"),
                probability_array(baseline["start_prob"], (53,), "Markov starts"),
                probability_array(baseline["transition_prob"], (53, 53), "Markov transitions"),
                MappingProxyType(thresholds),
            )
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        raise ArtifactError(f"Cannot load inference artifacts: {exc}") from exc


def validate_sequences(events, offsets, metadata, vocabulary_size=53):
    """Validate positional trace identity, dimensions, lengths and chronology."""
    required = {"BlockId", "first_timestamp", "last_timestamp", "first_line", "event_count", "strict_split"}
    _require(required.issubset(metadata.columns), "Missing metadata columns.")
    _require(not metadata[list(required)].isna().any().any(), "Missing replay metadata.")
    _require(metadata.BlockId.is_unique, "Duplicate block IDs.")
    _require(events.ndim == 1 and events.dtype.kind in "iu", "Events must be 1D integer IDs.")
    _require(offsets.ndim == 1 and offsets.dtype.kind in "iu" and len(offsets) == len(metadata) + 1,
             "Offsets must be a 1D integer array with one terminator per block.")
    _require(len(events) > 0 and offsets[0] == 0 and offsets[-1] == len(events), "Invalid offset endpoints.")
    _require(bool((offsets[1:] > offsets[:-1]).all()), "Offsets must be strictly increasing.")
    _require(bool((events < vocabulary_size).all() and (events >= 0).all()), "Event outside vocabulary.")
    _require(np.array_equal(np.diff(offsets), metadata.event_count.to_numpy()), "Sequence/metadata length mismatch.")
    _require(set(metadata.strict_split) == {"train", "validation", "test"}, "Unexpected replay partitions.")
    _require(metadata.sort_values(["first_timestamp", "first_line"]).index.equals(metadata.index),
             "Replay rows must preserve chronological order and raw-line tie breaks.")
    try:
        first = pd.to_datetime(metadata.first_timestamp.astype(str), format="%Y%m%d%H%M%S")
        last = pd.to_datetime(metadata.last_timestamp.astype(str), format="%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ArtifactError("Invalid metadata timestamp.") from exc
    _require(bool((last >= first).all()), "Block ends before it starts.")
    if "duration_seconds" in metadata:
        _require(np.array_equal((last-first).dt.total_seconds(), metadata.duration_seconds), "Duration mismatch.")


@dataclass(frozen=True)
class ReplayData:
    events: np.ndarray
    offsets: np.ndarray
    metadata: pd.DataFrame

    def sequence(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.metadata):
            raise IndexError("Replay block index out of range.")
        return self.events[self.offsets[index]:self.offsets[index + 1]]

    def indices(self, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split.")
        return np.flatnonzero(self.metadata.strict_split.to_numpy() == split)


def load_replay(directory=DEFAULT_ARTIFACT_DIR) -> ReplayData:
    """Load the paired sequence/metadata export and verify frozen membership."""
    directory = Path(directory)
    verify_checksums(directory, ["hdfs_event_sequences.npz", "retained_block_metadata.csv",
                                 "hdfs_split_manifest.csv", "split_config.json", "project_summary.json"])
    try:
        with np.load(directory / "hdfs_event_sequences.npz", allow_pickle=False) as data:
            events, offsets = data["events"], data["offsets"]
        metadata = read_csv(directory / "retained_block_metadata.csv")
        validate_sequences(events, offsets, metadata)
        manifest = read_csv(directory / "hdfs_split_manifest.csv")
        retained = manifest[manifest.strict_split.isin(["train", "validation", "test"])].reset_index(drop=True)
        _require(manifest.BlockId.is_unique and retained.equals(metadata), "Manifest/metadata alignment mismatch.")
        config = read_json(directory / "split_config.json")
        b1, b2 = int(config["train_boundary"]), int(config["validation_test_boundary"])
        expected = np.where(manifest.last_timestamp < b1, "train",
                   np.where((manifest.first_timestamp >= b1) & (manifest.last_timestamp < b2), "validation",
                   np.where(manifest.first_timestamp >= b2, "test", "excluded")))
        _require(np.array_equal(expected, manifest.strict_split), "Split boundary leakage or mismatch.")
        summary = read_json(directory / "project_summary.json")
        _require(len(metadata) == summary["retained_blocks"] == 493860
                 and len(manifest) == summary["labeled_blocks"] == 575061
                 and len(manifest) - len(metadata) == config["excluded_boundary_crossing_blocks"] == 81201,
                 "Frozen block counts do not reconcile.")
        for array in (events, offsets):
            array.setflags(write=False)
        return ReplayData(events, offsets, metadata)
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ArtifactError(f"Cannot load replay artifacts: {exc}") from exc
