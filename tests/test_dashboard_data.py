import io

import numpy as np
import pandas as pd
import pytest

from app.charts import precision_recall_chart, score_distribution
from app.data import (export_csv, filter_blocks, historical_activity, load_dashboard,
                      parse_manual_ids, sequence_table, severity)
from logsentinel.artifacts import ArtifactError


@pytest.fixture(scope="module")
def dashboard():
    return load_dashboard()


def test_displayed_metrics_and_partition_alignment(dashboard):
    table = dashboard.table("test")
    assert len(table) == 86268
    assert dashboard.metrics["test"]["hmm"]["TP"] == 596
    assert dashboard.metrics["test"]["hmm"]["FP"] == 261
    assert dashboard.metrics["test"]["frequency"]["f1"] == pytest.approx(.8942307692307693)
    assert dashboard.metrics["validation"]["hmm"] == dashboard.foundation.evaluation["validation"]
    assert table.Outcome.value_counts().to_dict() == {"True negative": 84891, "True positive": 596, "Missed anomaly": 520, "False alarm": 261}
    table.loc[:, "Label"] = "changed"
    assert not dashboard.table("test").Label.eq("changed").any()


def test_filters_sort_and_exact_csv_export(dashboard):
    table = dashboard.table("test")
    filtered = filter_blocks(table, actual=["Normal"], predicted=["Anomalous"], sort_by="event_count", ascending=True)
    assert len(filtered) == 261 and filtered.event_count.is_monotonic_increasing
    by_id = filter_blocks(table, search=table.iloc[0].BlockId, sort_by="BlockId")
    assert len(by_id) == 1
    assert filter_blocks(table, search=".*").empty  # Literal, not a regex.
    with pytest.raises(ValueError):
        filter_blocks(table, sort_by="untrusted")
    result = filter_blocks(table, score_range=(3, 4), event_range=(10, 30))
    assert result.hmm_score.between(3, 4).all() and result.event_count.between(10, 30).all()
    decoded = pd.read_csv(io.BytesIO(export_csv(result)), float_precision="round_trip")
    np.testing.assert_array_equal(decoded.hmm_score, result.hmm_score)
    assert decoded.BlockId.tolist() == result.BlockId.tolist()


def test_explicit_score_bands_and_inclusive_boundary():
    assert severity([.99, 1, 1.249, 1.25, 1.999, 2], 1).tolist() == [
        "Below threshold", "Above threshold", "Above threshold", "Elevated score", "Elevated score", "High score"]


def test_activity_bins_preserve_counts(dashboard):
    table = dashboard.table("test")
    for minutes in [5, 15, 30, 60]:
        activity = historical_activity(table, minutes)
        assert activity["Ground-truth anomalous"].sum() == 1116
        assert activity["Predicted anomalous"].sum() == 857
        assert activity["False alarms"].sum() == 261


def test_events_repetitions_transitions_and_no_fake_timestamps(dashboard):
    block = dashboard.table("test").iloc[0].BlockId
    events = sequence_table(dashboard, block)
    seq = dashboard.sequence(block)
    assert list(events.Position) == list(range(1, len(seq)+1))
    np.testing.assert_array_equal(events["Event ID"], seq)
    assert pd.isna(events.iloc[0]["Markov transition P"])
    np.testing.assert_array_equal(events["Markov transition P"].iloc[1:], dashboard.models.transition_prob[seq[:-1], seq[1:]])
    assert events["Repeated event"].sum() == len(seq) - len(np.unique(seq))
    assert not any("time" in column.lower() for column in events)
    with pytest.raises(ValueError):
        sequence_table(dashboard, block, -1)


@pytest.mark.parametrize("text", ["", "1.5", "-1", "53", "1,abc", "9999999999999999999999", ",1", "True", "1 "*10001])
def test_invalid_manual_inputs(text):
    with pytest.raises(ValueError):
        parse_manual_ids(text)


def test_manual_unknown_and_missing_artifacts(tmp_path):
    np.testing.assert_array_equal(parse_manual_ids("0, 1 52\n18"), [0, 1, 52, 18])
    with pytest.raises(ArtifactError):
        load_dashboard(str(tmp_path))


def test_chart_data_preserves_all_observations(dashboard):
    table = dashboard.table("test")
    chart = score_distribution(table, dashboard.models.thresholds["hmm"], density=False)
    assert sum(sum(trace.y) for trace in chart.data) == len(table)
    curve = precision_recall_chart(table)
    assert len(curve.data) == 3
    assert all(max(trace.x) == 1 and max(trace.y) == 1 for trace in curve.data)


def test_backend_forward_trace_is_observational(engine, replay):
    seq = replay.sequence(0)
    result = engine.explain(seq)
    assert result["anomaly_score"] == engine.score(seq)
    assert result["steps"][-1]["log_likelihood"] == result["log_likelihood"]
    assert len(result["steps"]) == len(seq)
    for step in result["steps"]:
        assert sum(step[f"state_{i}"] for i in range(3)) == pytest.approx(1)
