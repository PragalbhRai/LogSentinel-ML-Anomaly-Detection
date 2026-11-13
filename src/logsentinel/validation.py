"""Input and probability validation shared by loaders and inference."""

import numpy as np


def observations(values, vocabulary_size):
    array = np.asarray(values)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "iu":
        raise ValueError("Events must be a non-empty 1D array of integer IDs.")
    if np.any(array < 0) or np.any(array >= vocabulary_size):
        raise ValueError("Observation ID is outside the frozen vocabulary.")
    return array.astype(np.int64, copy=False)


def probability_array(value, shape, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name}: invalid probability shape or values; expected {shape}.")
    if not np.allclose(array.sum(axis=-1), 1.0, rtol=0, atol=1e-10):
        raise ValueError(f"{name}: probabilities must sum to one.")
    array.setflags(write=False)
    return array
