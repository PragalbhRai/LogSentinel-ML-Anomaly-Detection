"""Sequence-only HMM and baseline inference; no fitting or threshold tuning."""

from dataclasses import dataclass

import numpy as np

from .forward import scaled_forward
from .preprocessing import build_block_sequences
from .validation import observations

FINAL_THRESHOLD = 2.991257009736873
METHODS = ("hmm", "frequency", "markov")


def classify(scores, threshold=FINAL_THRESHOLD):
    """Apply the inclusive frozen decision boundary, including exact ties."""
    array = np.asarray(scores, dtype=np.float64)
    if np.isnan(array).any() or np.isneginf(array).any():
        raise ValueError("Scores cannot contain NaN or negative infinity.")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    return (array >= threshold).astype(np.int64)


@dataclass(frozen=True)
class Prediction:
    method: str
    event_count: int
    unknown_count: int
    anomaly_score: float
    threshold: float
    predicted_anomaly: bool


class InferenceEngine:
    """An inference service that has no ground-truth label input."""

    def __init__(self, models):
        self.models = models
        # Colab cell 24 computes these once, outside its per-sequence loop.
        with np.errstate(divide="ignore"):
            self._log_event = np.log(models.event_prob)
            self._log_start = np.log(models.start_prob)
            self._log_transition = np.log(models.transition_prob)

    def score(self, events, method="hmm") -> float:
        seq = observations(events, len(self.models.vocabulary))
        if method == "hmm":
            ll = scaled_forward(seq, self.models.startprob, self.models.transmat,
                                self.models.emissionprob)
        elif method == "frequency":
            ll = self._log_event[seq].sum()
        elif method == "markov":
            ll = self._log_start[seq[0]]
            if len(seq) > 1:
                ll += self._log_transition[seq[:-1], seq[1:]].sum()
        else:
            raise ValueError(f"Unknown inference method: {method}")
        return float(-ll / len(seq))

    def predict(self, events, method="hmm") -> Prediction:
        seq = observations(events, len(self.models.vocabulary))
        score = self.score(seq, method)
        threshold = self.models.thresholds[method]
        return Prediction(method, len(seq), int((seq == 0).sum()), score, threshold,
                          bool(classify(score, threshold)))

    def score_many(self, sequences, method="hmm") -> np.ndarray:
        """Use the original scalar recurrence for every sequence."""
        return np.asarray([self.score(seq, method) for seq in sequences], dtype=np.float64)

    def explain(self, events):
        """Return the custom Forward result and observed recurrence steps."""
        seq = observations(events, len(self.models.vocabulary))
        steps = []
        ll = scaled_forward(seq, self.models.startprob, self.models.transmat,
                            self.models.emissionprob, trace=steps)
        return {"log_likelihood": ll, "anomaly_score": float(-ll / len(seq)), "steps": steps}

    def predict_logs(self, lines, method="hmm") -> dict[str, Prediction]:
        sequences = build_block_sequences(lines, self.models.vocabulary)
        return {block: self.predict(seq, method) for block, seq in sequences.items()}
