"""Read-only verification of frozen scores; labels are used only here."""

import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from .artifacts import (DEFAULT_ARTIFACT_DIR, PROJECT_ROOT, ArtifactError, load_models,
                        load_replay, read_csv, read_json, verify_checksums)
from .inference import InferenceEngine, METHODS, classify


def calculate_metrics(labels, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "average_precision": float(average_precision_score(labels, scores)),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def _equal(actual, expected, description):
    if not np.array_equal(actual, expected):
        raise ArtifactError(f"Frozen verification failed: {description}.")


def _close(actual, expected, description, tolerance):
    if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=tolerance):
        raise ArtifactError(f"Frozen verification failed: {description}.")


def verify_frozen_inference(directory=DEFAULT_ARTIFACT_DIR, tolerance=1e-12):
    """Recompute all validation/test scores and exact decisions without fitting.

    A tolerance applies only to numerical comparisons, never classification.
    Does not select thresholds or optimize anything on the test set.
    """
    directory = Path(directory)
    verify_checksums(directory)
    models, replay = load_models(directory), load_replay(directory)
    engine = InferenceEngine(models)
    evaluation = read_json(directory / "hmm_final_evaluation.json")
    comparison = read_csv(directory / "final_model_comparison.csv").set_index("model")
    result = {"threshold": models.thresholds["hmm"], "comparison": ">=", "splits": {}}
    with np.load(directory / "hmm_final_evaluation_scores.npz", allow_pickle=False) as final, \
         np.load(directory / "baseline_scores.npz", allow_pickle=False) as baselines:
        for split in ("validation", "test"):
            indices = replay.indices(split)
            rows = replay.metadata.iloc[indices]
            labels = rows.Label.eq("Anomaly").astype(int).to_numpy()
            _equal(labels, final[split + "_labels"], split + " label alignment")
            split_result = {"blocks": len(indices), "models": {}}
            for method in METHODS:
                scores = engine.score_many((replay.sequence(int(i)) for i in indices), method)
                saved = final[split + "_scores"] if method == "hmm" else baselines[method + "_" + split]
                _close(scores, saved, f"{method} {split} scores", tolerance)
                predictions = classify(scores, models.thresholds[method])
                _equal(predictions, classify(saved, models.thresholds[method]), f"{method} {split} decisions")
                metrics = calculate_metrics(labels, predictions, scores)
                expected = evaluation[split] if method == "hmm" else (
                    comparison.loc[method].to_dict() if split == "test" else None)
                if expected is not None:
                    for name, value in metrics.items():
                        _close(np.asarray(value), expected[name], f"{method} {split} {name}", tolerance)
                if method == "hmm" and split == "test":
                    _equal(predictions, final["test_predictions"], "saved HMM predictions")
                    table = read_csv(directory / "hmm_test_predictions.csv")
                    columns = ["BlockId", "Label", "first_timestamp", "last_timestamp", "event_count"]
                    _equal(table[columns].to_numpy(), rows[columns].to_numpy(), "prediction table block alignment")
                    _close(scores, table.anomaly_score.to_numpy(), "prediction CSV scores", tolerance)
                    _equal(predictions, table.predicted_anomaly.to_numpy(), "prediction CSV decisions")
                split_result["models"][method] = {
                    "max_absolute_score_difference": float(np.max(np.abs(scores - saved))),
                    "prediction_mismatches": int(np.count_nonzero(predictions != classify(saved, models.thresholds[method]))),
                    "metrics": metrics,
                }
            result["splits"][split] = split_result
    # Ensure neither the notebook nor any experimental export was mutated.
    original = read_json(PROJECT_ROOT / "manifests/artifact_checksums.json")["files"]
    notebook = PROJECT_ROOT / "logsentinel.ipynb"
    with notebook.open("rb") as handle:
        _equal(hashlib.file_digest(handle, "sha256").hexdigest(), original["logsentinel.ipynb"], "original notebook checksum")
    verify_checksums(directory)
    return result
