"""Parity references are extracted definitions, not execution of the notebook."""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest

from logsentinel.preprocessing import (PATTERNS, build_block_sequences, event_id,
                                       extract_template, parse_log_line)


def notebook_parser():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "logsentinel.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][10]["source"]))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PATTERNS" for t in n.targets))]
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "colab_parser_definitions", "exec"), namespace)
    return namespace


def test_original_patterns_and_all_exported_templates(models):
    reference = notebook_parser()
    assert [(p.pattern, r) for p, r in PATTERNS] == [(p.pattern, r) for p, r in reference["PATTERNS"]]
    for template, identifier in models.vocabulary.items():
        if identifier == 0:
            continue
        component, message = template.split(" | ", 1)
        for token, value in {"<BLOCK>": "blk_-42", "<ADDRESS>": "/10.1.2.3:50010",
                             "<NUM>": "123", "<PATH>": "/mnt/volume/file", "<JOB>": "job_123_456"}.items():
            message = message.replace(token, value)
        line = f"081109 203518 143 INFO {component}: {message}\n"
        assert parse_log_line(line) == reference["parse_log_line"](line)
        assert event_id(parse_log_line(line), models.vocabulary) == identifier


def test_normalization_order_and_whitespace():
    message = "blk_-99 /10.2.3.4:12 job_123_45 task_12_3 attempt_1_2 /tmp/blk_12 +42 -2  a1  \t"
    assert extract_template(message) == notebook_parser()["extract_template"](message)


def test_actual_notebook_raw_prefix_matches_saved_sequence(models, replay, engine):
    notebook = json.loads((Path(__file__).resolve().parents[1] / "logsentinel.ipynb").read_text(encoding="utf-8"))
    output = "".join(notebook["cells"][3]["outputs"][0]["text"])
    lines = [line for line in output.splitlines() if line.startswith("081109 ")]
    assert len(lines) == 5
    sequences = build_block_sequences(lines, models.vocabulary)
    block = str(replay.metadata.iloc[0].BlockId)
    np.testing.assert_array_equal(sequences[block], replay.sequence(0)[:5])
    assert sequences[block].tolist() == [18, 47, 18, 18, 39]
    result = engine.predict_logs(lines)[block]
    assert result.event_count == 5 and result.unknown_count == 0
    assert result.anomaly_score == engine.score(replay.sequence(0)[:5])


def test_order_first_block_policy_unknown_and_filter(models):
    lines = ["081109 203520 1 INFO new.Component: new event blk_1 blk_2",
             "081109 203518 1 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1 terminating"]
    result = build_block_sequences(lines, models.vocabulary)
    assert set(result) == {"blk_1"}  # Original first-block behavior, no timestamp sort.
    assert result["blk_1"].tolist() == [0, 39]
    assert build_block_sequences(lines, models.vocabulary, ["blk_2"])["blk_2"].size == 0
    assert parse_log_line("malformed") is None
    assert build_block_sequences(["unassociated text"], models.vocabulary) == {}
    with pytest.raises(ValueError, match="Malformed"):
        build_block_sequences(["bad blk_1"], models.vocabulary)


def test_labels_and_other_fields_are_not_features(models):
    record = parse_log_line("081109 203518 1 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1 terminating")
    original = event_id(record, models.vocabulary)
    record.update(Label="Anomaly", pid="999", timestamp="unrelated", block_id="blk_99", level="ERROR")
    assert event_id(record, models.vocabulary) == original == 39
