from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from logsentinel.artifacts import ArtifactError, load_models, validate_sequences, validate_vocabulary, verify_checksums
from logsentinel.validation import probability_array


def tiny_metadata():
    return pd.DataFrame({"BlockId": ["a", "b", "c"], "first_timestamp": [20081109000000, 20081110000000, 20081111000000],
                         "last_timestamp": [20081109000000, 20081110000000, 20081111000000],
                         "first_line": [0, 1, 2], "event_count": [1, 1, 1], "strict_split": ["train", "validation", "test"]})


@pytest.mark.parametrize("offsets", [[1, 2, 3, 3], [0, 1, 1, 3], [0, 1, 2, 4], [0, 1, 3], [0., 1., 2., 3.]])
def test_bad_offsets(offsets):
    with pytest.raises(ArtifactError):
        validate_sequences(np.array([1, 2, 3]), np.array(offsets), tiny_metadata())


def test_metadata_order_counts_duplicates_and_events():
    events, offsets, metadata = np.array([1, 2, 3]), np.arange(4), tiny_metadata()
    validate_sequences(events, offsets, metadata)
    for column, values in [("BlockId", ["a", "a", "c"]), ("event_count", [2, 1, 1]),
                            ("first_timestamp", [20081111000000, 20081110000000, 20081109000000])]:
        with pytest.raises(ArtifactError):
            validate_sequences(events, offsets, metadata.assign(**{column: values}))
    with pytest.raises(ArtifactError):
        validate_sequences(np.array([1, 53, 3]), offsets, metadata)


def test_vocabulary_and_probability_contracts(models):
    bad = dict(models.vocabulary); bad["<UNKNOWN>"] = 1
    with pytest.raises(ArtifactError):
        validate_vocabulary(bad)
    for value, shape in [(np.ones(2), (3,)), (np.array([0.5, 0.6]), (2,)),
                          (np.array([np.nan, 1]), (2,)), (np.array([-0.1, 1.1]), (2,))]:
        with pytest.raises(ValueError):
            probability_array(value, shape, "test")
    with pytest.raises(ValueError):
        models.startprob[0] = 0
    with pytest.raises(TypeError):
        models.vocabulary["new"] = 53


def test_corruption_rejected_before_deserialization(tmp_path, monkeypatch):
    (tmp_path / "hmm_final_3state.joblib").write_bytes(b"corrupted")
    def forbidden(*args, **kwargs):
        pytest.fail("Unverified pickle must never be loaded")
    monkeypatch.setattr("logsentinel.artifacts.joblib.load", forbidden)
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        load_models(tmp_path)
    with pytest.raises(ArtifactError, match="Cannot verify"):
        verify_checksums(tmp_path, ["event_vocabulary.json"])


def test_loaded_model_dimensions_checked(monkeypatch):
    from hmmlearn.hmm import CategoricalHMM
    bad = CategoricalHMM(n_components=5, n_features=53)
    monkeypatch.setattr("logsentinel.artifacts.joblib.load", lambda _: bad)
    with pytest.raises(ArtifactError, match="dimensions"):
        load_models()


def test_replay_bounds_and_label_independence(replay, engine):
    score = engine.score(replay.sequence(0))
    changed = replace(replay, metadata=replay.metadata.assign(Label="Anomaly"))
    assert engine.score(changed.sequence(0)) == score
    with pytest.raises(IndexError):
        replay.sequence(-1)
    with pytest.raises(ValueError):
        replay.indices("future")
