"""Verified saved results for the faculty review; never trains models."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.data import load_dashboard
from logsentinel.artifacts import (ArtifactError, CHECKSUM_MANIFEST, DEFAULT_ARTIFACT_DIR,
                                   PROJECT_ROOT, read_csv, read_json)
from logsentinel.evaluation import calculate_metrics

DEFAULT_RUN = PROJECT_ROOT / "outputs/imbalance_experiments/run_20261113_01"
MODEL_NAMES = {"hmm": "HMM", "rf_class_weight": "Class-weighted Random Forest",
               "rf_smote": "SMOTE Random Forest"}


@dataclass(frozen=True)
class ReviewData:
    counts: dict
    retained_blocks: int
    comparison: pd.DataFrame
    test: pd.DataFrame
    revision: str


def load_review_data(artifacts=DEFAULT_ARTIFACT_DIR, run=DEFAULT_RUN):
    """Invalidate the cache when an original or experiment file changes."""
    artifacts, run = Path(artifacts), Path(run)
    try:
        original = read_json(CHECKSUM_MANIFEST)["files"]
        paths = [artifacts / name.removeprefix("logsentinel_artifacts/")
                 for name in original if name.startswith("logsentinel_artifacts/")]
        completion = read_json(run / "completion.json")
        paths += [run / "completion.json"] + [run / name for name in completion["files_sha256"]]
        revision = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
        return _load_review_data(str(artifacts), str(run), revision)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(f"Cannot verify saved dashboard results: {exc}") from exc


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_review_data(artifacts, run, revision):
    artifacts, run = Path(artifacts), Path(run)
    completion = read_json(run / "completion.json")
    if completion["status"] != "complete":
        raise ArtifactError("Random Forest experiment is incomplete.")
    for name, digest in completion["files_sha256"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ArtifactError("Unsafe experiment manifest path.")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ArtifactError(f"Experiment checksum mismatch: {name}")
    original = read_json(CHECKSUM_MANIFEST)["files"]
    inputs = read_json(run / "provenance.json")["input_artifact_sha256"]
    if inputs != {k: v for k, v in original.items() if k.startswith("logsentinel_artifacts/")}:
        raise ArtifactError("Experiment and original artifact bundles differ.")
    legacy = load_dashboard(str(artifacts))
    test = legacy.table("test")[["BlockId", "Label", "replay_index", "hmm_prediction"]].copy()
    results = [{"model": "hmm", **legacy.metrics["test"]["hmm"]}]
    comparison = read_csv(run / "comparison.csv")
    for method in ("rf_class_weight", "rf_smote"):
        directory = run / method
        threshold = read_json(directory / "threshold.json")
        if threshold["source_partition"] != "validation" or threshold["test_used"]:
            raise ArtifactError("RF threshold was not frozen using validation only.")
        with np.load(directory / "test_predictions.npz", allow_pickle=False) as saved:
            expected = (test.BlockId.to_numpy(dtype=str), test.replay_index.to_numpy(),
                        test.Label.eq("Anomaly").to_numpy(dtype=int))
            for name, values in zip(("block_ids", "replay_indices", "labels"), expected):
                if not np.array_equal(saved[name], values):
                    raise ArtifactError(f"{method}: saved test block identity mismatch.")
            scores, predictions = saved["scores"], saved["predictions"]
            if (scores.shape != expected[2].shape or not np.isfinite(scores).all()
                    or not np.array_equal(predictions, (scores >= threshold["threshold"]).astype(int))):
                raise ArtifactError(f"{method}: invalid saved decisions.")
            metrics = calculate_metrics(expected[2], predictions, scores)
            recorded = read_json(directory / "test_metrics.json")
            row = comparison[(comparison.model == method) & (comparison.split == "test")]
            if len(row) != 1 or recorded["threshold"] != threshold["threshold"]:
                raise ArtifactError("Experiment comparison or threshold mismatch.")
            for name, value in metrics.items():
                if (not np.isclose(value, recorded[name], rtol=0, atol=1e-12)
                        or not np.isclose(value, row.iloc[0][name], rtol=0, atol=1e-12)):
                    raise ArtifactError("Saved metrics disagree with actual predictions.")
            test[method + "_prediction"] = predictions
            results.append({"model": method, **metrics})
    manifest = read_csv(artifacts / "hdfs_split_manifest.csv")
    if set(manifest.Label.unique()) != {"Normal", "Anomaly"}:
        raise ArtifactError("Unexpected dataset labels.")
    counts = {"total": len(manifest), "normal": int(manifest.Label.eq("Normal").sum()),
              "anomaly": int(manifest.Label.eq("Anomaly").sum())}
    if counts["total"] != legacy.foundation.summary["labeled_blocks"]:
        raise ArtifactError("Dataset total does not match original summary.")
    return ReviewData(counts, len(legacy.replay.metadata), pd.DataFrame(results), test,
                      hashlib.sha256(repr(revision).encode()).hexdigest())


load_review_data.clear = _load_review_data.clear
