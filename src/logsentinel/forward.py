"""Unit 4 scaled Forward algorithm, extracted from Colab cell 18.

Arithmetic and zero-likelihood behavior are preserved. Strict integer input
validation is added at the public boundary to prevent silent float truncation.
"""

import numpy as np

from .validation import observations as validate_observations


def scaled_forward(observations, startprob, transmat, emissionprob, *, trace=None):
    """Return sequence log likelihood using normalized Forward recurrences."""
    obs = validate_observations(observations, emissionprob.shape[1])
    # Initialization: alpha_1(j) = pi_j * B_j(o_1)
    alpha = startprob * emissionprob[:, obs[0]]
    scale = alpha.sum()
    if scale <= 0:
        return float("-inf")
    alpha = alpha / scale
    log_likelihood = np.log(scale)
    if trace is not None:
        trace.append(_trace_step(1, obs[0], scale, log_likelihood, alpha))
    for t in range(1, len(obs)):
        # alpha_t(j) = B_j(o_t) * sum_i(alpha_(t-1)(i) * A_ij)
        alpha = (alpha @ transmat) * emissionprob[:, obs[t]]
        scale = alpha.sum()
        if scale <= 0:
            return float("-inf")
        alpha = alpha / scale
        log_likelihood += np.log(scale)
        if trace is not None:
            trace.append(_trace_step(t + 1, obs[t], scale, log_likelihood, alpha))
    return float(log_likelihood)


def _trace_step(position, event, scale, log_likelihood, alpha):
    """Observe the existing recurrence without changing its arithmetic."""
    return {"position": position, "event_id": int(event), "scale": float(scale),
            "log_likelihood": float(log_likelihood),
            **{f"state_{i}": float(value) for i, value in enumerate(alpha)}}
