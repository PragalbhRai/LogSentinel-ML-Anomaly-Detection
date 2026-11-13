import ast
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from logsentinel.forward import scaled_forward
from logsentinel.inference import FINAL_THRESHOLD, METHODS, classify


def test_forward_against_exhaustive_hidden_paths():
    start = np.array([0.6, 0.4]); trans = np.array([[0.7, 0.3], [0.2, 0.8]])
    emission = np.array([[0.9, 0.1], [0.4, 0.6]]); obs = [0, 1, 0, 1]
    probability = 0.0
    for states in itertools.product(range(2), repeat=len(obs)):
        value = start[states[0]] * emission[states[0], obs[0]]
        for t in range(1, len(obs)):
            value *= trans[states[t-1], states[t]] * emission[states[t], obs[t]]
        probability += value
    assert scaled_forward(obs, start, trans, emission) == pytest.approx(np.log(probability), abs=1e-14)


def test_forward_parity_with_original_and_hmmlearn(models, replay):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "logsentinel.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][18]["source"]))
    namespace = {"np": np}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]),
                 "colab_forward_definition", "exec"), namespace)
    reference = joblib.load(root / "logsentinel_artifacts/hmm_final_3state.joblib")
    indices = np.linspace(0, len(replay.metadata)-1, 128, dtype=int)
    sequences = [replay.sequence(int(i)) for i in indices]
    sequences += [np.array([0]), np.array([52]), np.zeros(10000, dtype=int)]
    for seq in sequences:
        actual = scaled_forward(seq, models.startprob, models.transmat, models.emissionprob)
        original = namespace["scaled_forward"](seq, models.startprob, models.transmat, models.emissionprob)
        assert actual == original
        assert actual == pytest.approx(reference.score(seq.reshape(-1, 1)), abs=1e-8)


@pytest.mark.parametrize("bad", [[], [1.5], [True], [-1], [53], [[1]], [np.nan], [np.inf], [2**64-1]])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_sequence_rejected(engine, bad, method):
    with pytest.raises(ValueError):
        engine.predict(bad, method)


@pytest.mark.parametrize("method", METHODS)
def test_single_event_and_unknown_are_supported(engine, method):
    prediction = engine.predict([0], method)
    assert np.isfinite(prediction.anomaly_score)
    assert prediction.event_count == prediction.unknown_count == 1


def test_inclusive_threshold_and_invalid_scores():
    np.testing.assert_array_equal(classify([np.nextafter(FINAL_THRESHOLD, -np.inf), FINAL_THRESHOLD,
                                           np.nextafter(FINAL_THRESHOLD, np.inf), np.inf]), [0, 1, 1, 1])
    for bad in [np.nan, -np.inf]:
        with pytest.raises(ValueError):
            classify(bad)


def test_impossible_observation_and_unknown_method(models, engine):
    assert scaled_forward([1], np.array([1.]), np.array([[1.]]), np.array([[1., 0.]])) == -np.inf
    with pytest.raises(ValueError, match="Unknown"):
        engine.score([1], "other")
