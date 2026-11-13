"""Safety checks for the optional training workflow; no frozen retraining."""

import numpy as np
import pytest

from logsentinel.artifacts import read_json
from logsentinel.imbalance_experiments import (
    DEFAULT_CONFIG, event_count_features, resample_training_only,
    run_experiments, select_validation_threshold, smote_memory_plan,
    train_and_select,
)


def test_event_histogram_preserves_counts_unknown_and_ignores_order():
    features = event_count_features(np.array([2, 0, 2, 1, 1, 2, 0, 2]),
                                    np.array([0, 4, 8]), 3)
    np.testing.assert_array_equal(features, [[1, 1, 2], [1, 1, 2]])
    assert features.dtype == np.float32
    for events, offsets in (([0, 3], [0, 2]), ([0, 1], [0, 0, 2]),
                            ([0, 1], [1, 2]), ([0.5, 1], [0, 2])):
        with pytest.raises(ValueError):
            event_count_features(np.array(events), np.array(offsets), 3)


def test_threshold_matches_exhaustive_validation_search_with_ties():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.4, 0.4, 0.8, 0.9])
    threshold = select_validation_threshold(labels, scores)
    candidates = np.unique(scores)
    def f1(t):
        prediction = scores >= t
        tp = np.sum(prediction & (labels == 1))
        return 2 * tp / (prediction.sum() + labels.sum())
    expected = max(candidates, key=f1)  # Sorted candidates choose lowest on a tie.
    assert threshold == expected
    assert select_validation_threshold([0, 1], [0.5, 0.5]) == 0.5
    with pytest.raises(ValueError):
        select_validation_threshold([0, 0], [0.1, 0.9])


def test_memory_rejection_happens_before_sampler_and_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("logsentinel.imbalance_experiments.available_memory_bytes", lambda: 1)
    x = np.ones((20, 3), dtype=np.float32)
    y = np.array([0] * 14 + [1] * 6)
    config = read_json(DEFAULT_CONFIG)
    plan = smote_memory_plan(x, y, config)
    assert plan["synthetic_training_rows"] == 8
    assert plan["resampled_training_rows"] == 28
    assert plan["resampled_feature_and_label_bytes"] == 28 * (3 * 4 + 8)
    assert not plan["passed"]
    with pytest.raises(MemoryError):
        resample_training_only(x, y, config, tmp_path)
    assert not read_json(tmp_path / "memory_preflight.json")["passed"]


def test_actual_smote_receives_training_only_and_preserves_inputs(tmp_path, monkeypatch):
    imblearn = pytest.importorskip("imblearn.over_sampling")
    monkeypatch.setattr("logsentinel.imbalance_experiments.available_memory_bytes", lambda: 32 * 1024**3)
    x = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [0, 3]], dtype=np.float32)
    y = np.array([0, 0, 0, 0, 1, 1])
    original_x, original_y = x.copy(), y.copy()
    config = read_json(DEFAULT_CONFIG)
    config["smote"]["k_neighbors"] = 1
    calls = []
    actual = imblearn.SMOTE.fit_resample
    def spy(self, values, labels):
        calls.append((values.copy(), labels.copy()))
        return actual(self, values, labels)
    monkeypatch.setattr(imblearn.SMOTE, "fit_resample", spy)
    resampled_x, resampled_y = resample_training_only(x, y, config, tmp_path)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], original_x)
    np.testing.assert_array_equal(calls[0][1], original_y)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(np.bincount(resampled_y), [4, 4])
    assert resampled_x.shape == (8, 2)
    assert read_json(tmp_path / "resampling.json")["input_partition"] == "train"


def test_fit_uses_training_and_threshold_uses_validation_only(monkeypatch):
    calls = []
    class Estimator:
        def __init__(self, **kwargs):
            self.classes_ = np.array([0, 1])
        def fit(self, x, y):
            calls.append(("fit", x, y))
        def predict_proba(self, x):
            calls.append(("score", x))
            return np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    monkeypatch.setattr("logsentinel.imbalance_experiments.RandomForestClassifier", Estimator)
    train_x, train_y = np.zeros((8, 2)), np.array([0] * 6 + [1] * 2)
    val_x, val_y = np.ones((3, 2)), np.array([0, 1, 1])
    _, threshold, _ = train_and_select(train_x, train_y, val_x, val_y,
                                     read_json(DEFAULT_CONFIG), "balanced")
    assert calls[0][0] == "fit" and calls[0][1] is train_x and calls[0][2] is train_y
    assert calls[1][0] == "score" and calls[1][1] is val_x
    assert len(calls) == 2 and threshold == 0.4


def test_output_cannot_overwrite_or_enter_originals(tmp_path):
    with pytest.raises(FileExistsError):
        run_experiments(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        run_experiments(tmp_path / "original" / "experiment", artifact_dir=tmp_path / "original")
