"""Independently reload and verify a trusted local experiment run, without fitting."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from logsentinel.artifacts import DEFAULT_ARTIFACT_DIR, load_models, load_replay, read_json, verify_checksums
from logsentinel.evaluation import calculate_metrics
from logsentinel.imbalance_experiments import (event_count_features, file_hash,
                                              select_validation_threshold, write_json)


def verify(run, artifacts=DEFAULT_ARTIFACT_DIR):
    run, artifacts = Path(run), Path(artifacts)
    completion = read_json(run / "completion.json")
    if completion["status"] != "complete":
        raise ValueError("Run is incomplete.")
    for name, digest in completion["files_sha256"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Run checksum mismatch: {name}")
    verify_checksums(artifacts)
    replay, frozen = load_replay(artifacts), load_models(artifacts)
    features = event_count_features(replay.events, replay.offsets, len(frozen.vocabulary))
    comparison = pd.read_csv(run / "comparison.csv", float_precision="round_trip")
    provenance = read_json(run / "provenance.json")
    findings = {}
    for method in ("rf_class_weight", "rf_smote", "hmm", "frequency"):
        directory = run / method
        bundle = joblib.load(directory / "model.joblib") if method.startswith("rf_") else None
        findings[method] = {}
        for split in ("validation", "test"):
            indices = replay.indices(split)
            rows = replay.metadata.iloc[indices]
            labels = rows.Label.eq("Anomaly").to_numpy(dtype=np.int64)
            if np.bincount(labels).tolist() != provenance["class_counts_Normal_Anomaly"][split]:
                raise ValueError("Holdout class counts changed.")
            with np.load(directory / f"{split}_predictions.npz", allow_pickle=False) as saved:
                for actual, expected in ((saved["replay_indices"], indices), (saved["labels"], labels),
                                         (saved["block_ids"], rows.BlockId.to_numpy(dtype=str))):
                    np.testing.assert_array_equal(actual, expected)
                threshold = read_json(directory / f"{split}_metrics.json")["threshold"]
                if bundle:
                    scores = bundle["estimator"].predict_proba(features[indices])[:, 1]
                    assert threshold == bundle["threshold"]
                else:
                    filename = "hmm_final_evaluation_scores.npz" if method == "hmm" else "baseline_scores.npz"
                    with np.load(artifacts / filename, allow_pickle=False) as baseline:
                        scores = baseline[split + "_scores"] if method == "hmm" else baseline["frequency_" + split]
                    assert threshold == frozen.thresholds[method]
                np.testing.assert_allclose(scores, saved["scores"], rtol=0, atol=1e-12)
                np.testing.assert_array_equal((scores >= threshold).astype(int), saved["predictions"])
                metrics = calculate_metrics(labels, saved["predictions"], saved["scores"])
                if bundle and split == "validation":
                    assert select_validation_threshold(labels, saved["scores"]) == threshold
                recorded = read_json(directory / f"{split}_metrics.json")
                table_row = comparison[(comparison.model == method) & (comparison.split == split)].iloc[0]
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, recorded[key], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(value, table_row[key], rtol=0, atol=1e-12)
                matrix = pd.read_csv(directory / f"{split}_confusion_matrix.csv", index_col=0).to_numpy()
                np.testing.assert_array_equal(matrix, [[metrics["TN"], metrics["FP"]],
                                                       [metrics["FN"], metrics["TP"]]])
                findings[method][split] = {
                    "blocks": len(indices), "prediction_mismatches": 0,
                    "max_absolute_reloaded_score_difference": float(np.max(np.abs(scores - saved["scores"]))),
                    "metrics_and_confusion_counts_verified": True,
                }
        if bundle:
            training = read_json(directory / "training.json")
            expected_counts = provenance["class_counts_Normal_Anomaly"]["train"]
            if method == "rf_smote":
                expected_counts = [expected_counts[0]] * 2
                assert read_json(directory / "memory_preflight.json")["passed"]
                assert read_json(directory / "resampling.json")["input_partition"] == "train"
            assert training["class_counts_Normal_Anomaly"] == expected_counts
            assert training["training_rows"] == sum(expected_counts)
    verify_checksums(artifacts)
    return {"status": "passed", "all_payload_hashes_verified": True, "models": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Trusted local runner output directory")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.run, args.artifacts)
    if args.output:
        write_json(args.output, report)
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
