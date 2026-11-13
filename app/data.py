"""Small, verified historical data for the foundation; no live telemetry."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

from logsentinel.artifacts import (DEFAULT_ARTIFACT_DIR, ArtifactError, load_models,
                                   read_csv, read_json, verify_checksums)
from logsentinel import InferenceEngine, load_replay
from logsentinel.evaluation import calculate_metrics
from logsentinel.inference import METHODS, classify


@dataclass(frozen=True)
class FoundationData:
    evaluation: dict
    comparison: pd.DataFrame
    summary: dict


@st.cache_data(show_spinner=False)
def load_foundation(directory=str(DEFAULT_ARTIFACT_DIR)):
    directory = Path(directory)
    verify_checksums(directory, ["project_summary.json"])
    load_models(directory)  # Verify model compatibility before showing results.
    evaluation = read_json(directory / "hmm_final_evaluation.json")
    summary = read_json(directory / "project_summary.json")
    comparison = read_csv(directory / "final_model_comparison.csv")
    if evaluation["test"] != summary["test_metrics"]:
        raise ArtifactError("Summary and evaluation metrics disagree.")
    return FoundationData(evaluation, comparison, summary)


SEVERITY_RULE = "Display bands: score/threshold <1: below; [1,1.25): above; [1.25,2): elevated; ≥2: high. Not a probability or failure category."


def severity(scores, threshold):
    ratio = np.asarray(scores) / threshold
    return np.select([ratio < 1, ratio < 1.25, ratio < 2],
                     ["Below threshold", "Above threshold", "Elevated score"], default="High score")


@dataclass(frozen=True)
class DashboardData:
    models: object
    replay: object
    engine: object
    foundation: FoundationData
    tables: dict
    metrics: dict
    candidates: pd.DataFrame
    training: dict

    def table(self, split):
        if split not in self.tables:
            raise ValueError("Only frozen validation and test partitions are available.")
        return self.tables[split].copy()

    def sequence(self, block):
        for table in self.tables.values():
            if block in table.index:
                return self.replay.sequence(int(table.loc[block, "replay_index"]))
        raise ValueError("Block is outside the saved holdout partitions.")


def load_dashboard(directory=str(DEFAULT_ARTIFACT_DIR)):
    """Invalidate cached resources if any frozen file changes or disappears."""
    directory = Path(directory)
    try:
        from logsentinel.artifacts import CHECKSUM_MANIFEST
        names = read_json(CHECKSUM_MANIFEST)["files"]
        revision = []
        for name in names:
            if name.startswith("logsentinel_artifacts/"):
                stat = (directory / name.removeprefix("logsentinel_artifacts/")).stat()
                revision.append((name, stat.st_size, stat.st_mtime_ns))
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactError(f"Cannot inspect the artifact bundle: {exc}") from exc
    return _load_dashboard(str(directory), tuple(revision))


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_dashboard(directory, revision):
    """Verify once per bundle/session server; never recompute thresholds."""
    directory = Path(directory)
    verify_checksums(directory)
    models, replay = load_models(directory), load_replay(directory)
    foundation = load_foundation(str(directory))
    tables, metrics = {}, {}
    try:
        with np.load(directory / "hmm_final_evaluation_scores.npz", allow_pickle=False) as final, \
             np.load(directory / "baseline_scores.npz", allow_pickle=False) as baseline:
            for split in ("validation", "test"):
                indices = replay.indices(split)
                rows = replay.metadata.iloc[indices].copy()
                rows["replay_index"] = indices
                labels = rows.Label.eq("Anomaly").astype(int).to_numpy()
                if not np.array_equal(labels, final[split + "_labels"]):
                    raise ArtifactError("Saved labels are not aligned with replay metadata.")
                metrics[split] = {}
                for method in METHODS:
                    scores = final[split + "_scores"] if method == "hmm" else baseline[method + "_" + split]
                    if scores.shape != labels.shape or not np.isfinite(scores).all():
                        raise ArtifactError("Invalid or misaligned saved scores.")
                    predictions = classify(scores, models.thresholds[method])
                    rows[method + "_score"] = scores
                    rows[method + "_prediction"] = predictions
                    metrics[split][method] = calculate_metrics(labels, predictions, scores)
                    expected = foundation.evaluation[split] if method == "hmm" else (
                        foundation.comparison.set_index("model").loc[method] if split == "test" else None)
                    if expected is not None and any(not np.isclose(value, expected[key], rtol=0, atol=1e-12)
                                                    for key, value in metrics[split][method].items()):
                        raise ArtifactError("Displayed metrics disagree with the frozen experiment.")
                rows["Prediction"] = np.where(rows.hmm_prediction == 1, "Anomalous", "Normal")
                rows["Score / threshold"] = rows.hmm_score / models.thresholds["hmm"]
                rows["Score severity"] = severity(rows.hmm_score, models.thresholds["hmm"])
                rows["Outcome"] = np.select(
                    [(labels == 1) & (rows.hmm_prediction == 1), (labels == 0) & (rows.hmm_prediction == 1),
                     (labels == 1) & (rows.hmm_prediction == 0)],
                    ["True positive", "False alarm", "Missed anomaly"], default="True negative")
                rows["Block start"] = pd.to_datetime(rows.first_timestamp.astype(str), format="%Y%m%d%H%M%S")
                rows["Block end"] = pd.to_datetime(rows.last_timestamp.astype(str), format="%Y%m%d%H%M%S")
                if split == "test":
                    saved = read_csv(directory / "hmm_test_predictions.csv")
                    if not (np.array_equal(saved.BlockId, rows.BlockId) and np.array_equal(saved.Label, rows.Label)
                            and np.array_equal(saved.predicted_anomaly, rows.hmm_prediction)
                            and np.array_equal(final["test_predictions"], rows.hmm_prediction)
                            and np.allclose(saved.anomaly_score, rows.hmm_score, rtol=0, atol=1e-12)):
                        raise ArtifactError("Test predictions are not aligned with saved results.")
                tables[split] = rows.set_index("BlockId", drop=False).rename_axis(None)
        return DashboardData(models, replay, InferenceEngine(models), foundation, tables, metrics,
                             read_csv(directory / "hmm_validation_comparison.csv"),
                             read_json(directory / "hmm_training_report.json"))
    except (KeyError, OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot prepare historical dashboard data: {exc}") from exc


load_dashboard.clear = _load_dashboard.clear


def filter_blocks(table, *, search="", actual=(), predicted=(), score_range=None,
                  event_range=None, sort_by="hmm_score", ascending=False):
    mask = table.BlockId.str.contains(search.strip(), regex=False, case=False)
    if actual:
        mask &= table.Label.isin(actual)
    if predicted:
        mask &= table.Prediction.isin(predicted)
    if score_range is not None:
        mask &= table.hmm_score.between(*score_range)
    if event_range is not None:
        mask &= table.event_count.between(*event_range)
    if sort_by not in {"hmm_score", "event_count", "Block start", "BlockId"}:
        raise ValueError("Unsupported sort column.")
    return table.loc[mask].sort_values(sort_by, ascending=ascending, kind="stable")


def historical_activity(table, minutes=15):
    """Counts assigned to observed block start, never event/incident times."""
    values = table.assign(**{"Ground-truth anomalous": table.Label.eq("Anomaly").astype(int),
                             "Predicted anomalous": table.hmm_prediction,
                             "False alarms": table.Outcome.eq("False alarm").astype(int)})
    columns = ["Ground-truth anomalous", "Predicted anomalous", "False alarms"]
    return values.set_index("Block start")[columns].resample(f"{minutes}min").sum()


def sequence_table(data, block, transition_cutoff=0.01):
    if not 0 <= transition_cutoff <= 1:
        raise ValueError("Transition probability cutoff must be in [0,1].")
    seq = data.sequence(block)
    vocabulary = {value: key for key, value in data.models.vocabulary.items()}
    table = pd.DataFrame({"Position": np.arange(1, len(seq) + 1), "Event ID": seq,
                          "Template": [vocabulary[int(i)] for i in seq]})
    table["Occurrence"] = table.groupby("Event ID").cumcount() + 1
    table["Repeated event"] = table.Occurrence > 1
    probabilities = np.full(len(seq), np.nan)
    if len(seq) > 1:
        probabilities[1:] = data.models.transition_prob[seq[:-1], seq[1:]]
    table["Markov transition P"] = probabilities
    table["Unusual transition"] = table["Markov transition P"] < transition_cutoff
    return table


def export_csv(table):
    columns = ["BlockId", "Label", "Prediction", "Outcome", "hmm_score", "Score / threshold",
               "Score severity", "event_count", "Block start", "Block end"]
    return table[columns].to_csv(index=False, float_format="%.17g").encode("utf-8")


def parse_manual_ids(text, vocabulary_size=53, limit=10000):
    tokens = re.split(r"[\s,]+", text.strip())
    if not text.strip() or len(tokens) > limit or any(not re.fullmatch(r"\d+", t) for t in tokens):
        raise ValueError(f"Enter 1–{limit:,} integer event IDs separated by spaces or commas.")
    # Bound before int64 conversion so exceptionally large integers fail cleanly.
    if any(len(t) > 10 or int(t) >= vocabulary_size for t in tokens):
        raise ValueError(f"Event IDs must be between 0 and {vocabulary_size - 1}.")
    return np.array([int(t) for t in tokens], dtype=np.int64)
