"""Optional supervised experiments, independent of frozen inference and the app.

Only event counts enter the forests. Training labels are intentionally used;
holdout labels are used for validation threshold selection and final evaluation.
"""

import ctypes
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .artifacts import (CHECKSUM_MANIFEST, DEFAULT_ARTIFACT_DIR, PROJECT_ROOT,
                        ArtifactError, load_models, load_replay, read_csv,
                        read_json, verify_checksums)
from .evaluation import calculate_metrics

DEFAULT_CONFIG = PROJECT_ROOT / "configs/imbalance_experiments.json"


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def event_count_features(events, offsets, vocabulary_size):
    """One float32 count per event ID, including UNKNOWN; no fitted transform.

    A block's feature sum is its length. Order within a block is discarded.
    Inputs are the validated replay arrays, never metadata or labels.
    """
    events, offsets = np.asarray(events), np.asarray(offsets)
    if (events.ndim != 1 or events.dtype.kind not in "iu" or not events.size
            or vocabulary_size < 1 or (events >= vocabulary_size).any()
            or (events < 0).any()):
        raise ValueError("Expected nonempty integer events in the vocabulary.")
    if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != len(events)
            or not (offsets[1:] > offsets[:-1]).all()):
        raise ValueError("Invalid block offsets.")
    features = np.empty((len(offsets) - 1, vocabulary_size), dtype=np.float32)
    for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        features[row] = np.bincount(events[start:end].astype(np.int64), minlength=vocabulary_size)
    if not np.array_equal(features.sum(axis=1, dtype=np.float64), np.diff(offsets)):
        raise ValueError("Feature counts do not reconcile with block lengths.")
    return features


def select_validation_threshold(labels, scores):
    """Maximize validation F1 with >= decisions; ties take lowest threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all()
            or set(np.unique(labels)) != {0, 1}):
        raise ValueError("Threshold selection requires finite scores and both binary labels.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator != 0)
    return float(thresholds[int(np.argmax(f1))])


def available_memory_bytes():
    """Read available physical RAM without adding a runtime dependency."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available",
                 "virtual_total", "virtual_available", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot read available physical memory; refusing unchecked SMOTE.")
        return int(status.available)
    if Path("/proc/meminfo").exists():
        entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(entries["MemAvailable"].split()[0]) * 1024
    raise OSError("Memory preflight supports Windows/Linux; refusing unchecked SMOTE.")


def smote_memory_plan(train_features, train_labels, config, *, other_feature_bytes=0):
    """Conservative working estimate, not a guarantee or measured peak RSS.

    Includes multiple dense/synthetic copies, forest storage and two neighbor
    distance chunks. Neighbors are explicitly chunked by sklearn working_memory.
    """
    labels, counts = np.unique(train_labels, return_counts=True)
    if set(labels) != {0, 1} or counts[1] >= counts[0]:
        raise ValueError("SMOTE experiment expects Anomaly=1 as the minority class.")
    smote = config["smote"]
    if smote["sampling_strategy"] != "auto":
        raise ValueError("Memory estimator supports full minority balancing only.")
    if counts[1] <= smote["k_neighbors"]:
        raise ValueError("Not enough minority training rows for configured SMOTE neighbors.")
    forest = config["forest"]
    if not isinstance(forest["max_leaf_nodes"], int) or forest["max_leaf_nodes"] < 2:
        raise ValueError("A finite forest leaf limit is required for the memory estimate.")
    fraction = smote["maximum_available_memory_fraction"]
    if not 0 < fraction <= 0.5:
        raise ValueError("SMOTE memory budget must be at most half available physical RAM.")
    synthetic = int(counts[0] - counts[1])
    resampled = int(2 * counts[0])
    row_bytes = train_features.shape[1] * train_features.dtype.itemsize + 8
    resampled_bytes = resampled * row_bytes
    synthetic_bytes = synthetic * row_bytes
    forest_bytes = forest["n_estimators"] * (2 * forest["max_leaf_nodes"] - 1) * 128
    estimated = (other_feature_bytes + 4 * (train_features.nbytes + train_labels.nbytes)
                 + 8 * resampled_bytes + 4 * synthetic_bytes + forest_bytes
                 + 2 * smote["working_memory_mib"] * 1024**2 + 512 * 1024**2)
    available = available_memory_bytes()
    return {
        "training_normal": int(counts[0]), "training_anomaly": int(counts[1]),
        "synthetic_training_rows": synthetic, "resampled_training_rows": resampled,
        "resampled_feature_and_label_bytes": resampled_bytes,
        "estimated_working_bytes": int(estimated), "available_physical_bytes": available,
        "budget_bytes": int(available * fraction),
        "maximum_available_memory_fraction": fraction,
        "passed": bool(estimated <= available * fraction),
        "scope": "Conservative estimate, not a measured peak; includes existing forest and temporary copies.",
    }


def resample_training_only(train_features, train_labels, config, output, *, other_feature_bytes=0):
    """Guard immediately before SMOTE. This API has no holdout inputs."""
    plan = smote_memory_plan(train_features, train_labels, config,
                             other_feature_bytes=other_feature_bytes)
    write_json(Path(output) / "memory_preflight.json", plan)
    if not plan["passed"]:
        raise MemoryError("SMOTE memory estimate exceeds half of available RAM; see memory_preflight.json.")
    from imblearn.over_sampling import SMOTE

    settings = config["smote"]
    if settings["neighbor_algorithm"] != "brute":
        raise ValueError("Memory estimate requires chunked brute-force neighbors.")
    neighbors = NearestNeighbors(n_neighbors=settings["k_neighbors"] + 1,
                                 algorithm="brute", n_jobs=1)
    sampler = SMOTE(sampling_strategy=settings["sampling_strategy"],
                    random_state=config["random_state"], k_neighbors=neighbors)
    with config_context(working_memory=settings["working_memory_mib"]), \
            threadpool_limits(limits=settings["thread_limit"]):
        features, labels = sampler.fit_resample(train_features, train_labels)
    if not np.array_equal(np.bincount(labels), [plan["training_normal"]] * 2):
        raise ValueError("SMOTE class counts differ from the memory plan.")
    write_json(Path(output) / "resampling.json", {
        "input_partition": "train", "validation_resampled": False, "test_resampled": False,
        "original_rows": len(train_labels), "resampled_rows": len(labels),
        "class_counts": np.bincount(labels).tolist(), "random_state": config["random_state"],
        "k_neighbors": settings["k_neighbors"], "synthetic_features_may_be_fractional": True,
    })
    return features, labels


def train_and_select(train_features, train_labels, validation_features, validation_labels,
                     config, class_weight):
    """Fit training data and freeze a validation threshold before test access."""
    estimator = RandomForestClassifier(**config["forest"],
                                        random_state=config["random_state"], class_weight=class_weight)
    estimator.fit(train_features, train_labels)
    scores = estimator.predict_proba(validation_features)[:, list(estimator.classes_).index(1)]
    threshold = select_validation_threshold(validation_labels, scores)
    return estimator, threshold, scores


def save_evaluation(directory, split, labels, scores, threshold, rows, indices):
    """Save reproducible scores, identities and separate numeric/visual matrices."""
    predictions = (scores >= threshold).astype(np.int64)
    metrics = calculate_metrics(labels, predictions, scores)
    matrix = [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]
    write_json(directory / f"{split}_metrics.json", {"threshold": threshold, **metrics})
    pd.DataFrame(matrix, index=["actual_Normal", "actual_Anomaly"],
                 columns=["predicted_Normal", "predicted_Anomaly"]).to_csv(
                     directory / f"{split}_confusion_matrix.csv", mode="x")
    # SVG is a standalone figure with exact counts; no extra plotting dependency.
    cells = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            x, y = 170 + j * 145, 95 + i * 85
            cells.append(f'<rect x="{x}" y="{y}" width="140" height="80" fill="#e1edf5"/>'
                         f'<text x="{x+70}" y="{y+46}" text-anchor="middle">{count:,}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="500" height="310" '
           'viewBox="0 0 500 310"><rect width="500" height="310" fill="white"/>'
           '<g font-family="sans-serif" font-size="16" fill="#14283a">'
           f'<text x="250" y="28" text-anchor="middle">{directory.name}: {split}</text>'
           '<text x="315" y="56" text-anchor="middle">Predicted label</text>'
           '<text x="240" y="83" text-anchor="middle">Normal</text>'
           '<text x="385" y="83" text-anchor="middle">Anomaly</text>'
           '<text x="25" y="139">Actual Normal</text>'
           '<text x="25" y="224">Actual Anomaly</text>' + ''.join(cells) + '</g></svg>')
    with (directory / f"{split}_confusion_matrix.svg").open("x", encoding="utf-8") as handle:
        handle.write(svg)
    with (directory / f"{split}_predictions.npz").open("xb") as handle:
        np.savez_compressed(handle, replay_indices=indices, block_ids=rows.BlockId.to_numpy(dtype=str),
                            labels=labels, scores=scores, predictions=predictions)
    return metrics


def run_experiments(output, artifact_dir=DEFAULT_ARTIFACT_DIR, config_path=DEFAULT_CONFIG):
    output, artifact_dir = Path(output).resolve(), Path(artifact_dir).resolve()
    if output == artifact_dir or artifact_dir in output.parents:
        raise ValueError("Experiment outputs must be outside the original artifact directory.")
    config = read_json(config_path)
    # Exclusive directory creation reserves an entirely new run, including failures.
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run(output, artifact_dir, config)
    except Exception as exc:
        write_json(output / "error.json", {"type": type(exc).__name__, "message": str(exc),
                                           "status": "failed_or_partial; existing run files retained"})
        raise


def _run(output, artifact_dir, config):
    started = time.perf_counter()
    print("Verifying frozen exports and chronological membership...", flush=True)
    verify_checksums(artifact_dir)
    original = read_json(CHECKSUM_MANIFEST)["files"]
    if file_hash(PROJECT_ROOT / "logsentinel.ipynb") != original["logsentinel.ipynb"]:
        raise ArtifactError("Original notebook checksum mismatch.")
    app_hashes = {str(p.relative_to(PROJECT_ROOT)): file_hash(p)
                  for p in (PROJECT_ROOT / "app").rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts}
    frozen, replay = load_models(artifact_dir), load_replay(artifact_dir)
    if set(replay.metadata.Label.unique()) != {"Normal", "Anomaly"}:
        raise ArtifactError("Unexpected ground-truth labels.")
    indices = {split: replay.indices(split) for split in ("train", "validation", "test")}
    labels = {split: replay.metadata.iloc[ix].Label.eq("Anomaly").to_numpy(dtype=np.int64)
              for split, ix in indices.items()}
    class_counts = {split: np.bincount(y, minlength=2).tolist() for split, y in labels.items()}
    versions = {name: metadata.version(name) for name in
                ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                 "joblib", "hmmlearn", "threadpoolctl")}
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", {
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
        "input_artifact_sha256": {k: v for k, v in original.items()
                                  if k.startswith("logsentinel_artifacts/")},
        "notebook_sha256": original["logsentinel.ipynb"], "dashboard_sha256_before": app_hashes,
        "class_counts_Normal_Anomaly": class_counts, "split": read_json(artifact_dir / "split_config.json"),
        "baseline_source": "checksum-verified saved scores, unchanged validation-selected thresholds",
        "comparison_caveat": "Forests use normal and anomalous training labels; HMM/frequency were fitted on normal training blocks only.",
        "hyperparameters": "Fixed in advance; no search or test-based selection.",
    })
    print("Building 53 float32 event-count features per block...", flush=True)
    features = event_count_features(replay.events, replay.offsets, len(frozen.vocabulary))
    by_split = {split: features[ix] for split, ix in indices.items()}
    validation_digest = hashlib.sha256(by_split["validation"].tobytes()).hexdigest()
    test_digest = hashlib.sha256(by_split["test"].tobytes()).hexdigest()
    feature_schema = {
        "type": "event_counts", "dtype": "float32", "normalization": "none",
        "columns": [{"column": identifier, "template": template}
                    for template, identifier in sorted(frozen.vocabulary.items(), key=lambda pair: pair[1])],
        "metadata_features": [], "event_order_retained": False,
        "validation_feature_sha256": validation_digest, "test_feature_sha256": test_digest,
    }
    write_json(output / "feature_schema.json", feature_schema)
    rows = []
    for name, class_weight in (("rf_class_weight", "balanced"), ("rf_smote", None)):
        model_dir = output / name
        model_dir.mkdir()
        fit_started = time.perf_counter()
        train_x, train_y = by_split["train"], labels["train"]
        if name == "rf_smote":
            print("Checking RAM and applying SMOTE to TRAINING ONLY...", flush=True)
            train_x, train_y = resample_training_only(
                train_x, train_y, config, model_dir,
                other_feature_bytes=features.nbytes + sum(x.nbytes for x in by_split.values()))
        print(f"Fitting {name}: {len(train_y):,} training rows...", flush=True)
        estimator, threshold, validation_scores = train_and_select(
            train_x, train_y, by_split["validation"], labels["validation"], config, class_weight)
        fit_seconds = time.perf_counter() - fit_started
        # Persist the frozen threshold before any test prediction/evaluation.
        write_json(model_dir / "threshold.json", {
            "threshold": threshold, "comparison": ">=", "source_partition": "validation",
            "objective": "maximum_f1", "tie_break": "lowest_threshold", "test_used": False,
        })
        with (model_dir / "model.joblib").open("xb") as handle:
            joblib.dump({"estimator": estimator, "threshold": threshold, "positive_class": 1,
                         "feature_schema": feature_schema, "config": config, "class_weight": class_weight,
                         "input_artifact_sha256": original}, handle, compress=3)
        scores = {"validation": validation_scores,
                  "test": estimator.predict_proba(by_split["test"])[:, list(estimator.classes_).index(1)]}
        for split in ("validation", "test"):
            metrics = save_evaluation(model_dir, split, labels[split], scores[split], threshold,
                                      replay.metadata.iloc[indices[split]], indices[split])
            rows.append({"model": name, "split": split, "threshold": threshold, **metrics})
        write_json(model_dir / "training.json", {
            "training_rows": len(train_y), "class_counts_Normal_Anomaly": np.bincount(train_y).tolist(),
            "class_weight": class_weight, "fit_and_resampling_seconds": fit_seconds,
            "tree_nodes": sum(tree.tree_.node_count for tree in estimator.estimators_),
            "forest_parameters": estimator.get_params(),
        })
        print(f"{name}: threshold={threshold:.12g}; test F1={rows[-1]['f1']:.6f}", flush=True)
        del train_x, train_y, estimator, scores

    comparison = read_csv(artifact_dir / "final_model_comparison.csv").set_index("model")
    hmm_report = read_json(artifact_dir / "hmm_final_evaluation.json")
    with np.load(artifact_dir / "hmm_final_evaluation_scores.npz", allow_pickle=False) as hmm, \
            np.load(artifact_dir / "baseline_scores.npz", allow_pickle=False) as baseline:
        for name, method in (("hmm", "hmm"), ("frequency", "frequency")):
            model_dir = output / name
            model_dir.mkdir()
            for split in ("validation", "test"):
                if not np.array_equal(labels[split], hmm[split + "_labels"]):
                    raise ArtifactError("Frozen scores and replay labels are misaligned.")
                scores = hmm[split + "_scores"] if method == "hmm" else baseline[method + "_" + split]
                if scores.shape != labels[split].shape or not np.isfinite(scores).all():
                    raise ArtifactError("Invalid frozen comparison scores.")
                threshold = frozen.thresholds[method]
                metrics = save_evaluation(model_dir, split, labels[split], scores, threshold,
                                          replay.metadata.iloc[indices[split]], indices[split])
                expected = hmm_report[split] if method == "hmm" else (
                    comparison.loc[method].to_dict() if split == "test" else None)
                if expected and any(not np.isclose(v, expected[k], rtol=0, atol=1e-12)
                                    for k, v in metrics.items()):
                    raise ArtifactError("Recomputed baseline metrics differ from frozen results.")
                rows.append({"model": name, "split": split, "threshold": threshold, **metrics})
    if (validation_digest != hashlib.sha256(by_split["validation"].tobytes()).hexdigest()
            or test_digest != hashlib.sha256(by_split["test"].tobytes()).hexdigest()):
        raise ValueError("Holdout features were modified during training.")
    verify_checksums(artifact_dir)
    if any(file_hash(PROJECT_ROOT / name) != digest for name, digest in app_hashes.items()):
        raise ValueError("Dashboard files changed during experiment.")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparison.csv", index=False, mode="x")
    write_json(output / "results.json", rows)
    write_json(output / "completion.json", {
        "status": "complete", "elapsed_seconds": time.perf_counter() - started,
        "original_artifacts_unchanged": True, "dashboard_unchanged": True,
        "validation_and_test_features_unchanged": True,
        "files_sha256": {str(p.relative_to(output)): file_hash(p)
                         for p in output.rglob("*") if p.is_file()},
    })
    print(frame[frame.split == "test"].to_string(index=False), flush=True)
    return frame
