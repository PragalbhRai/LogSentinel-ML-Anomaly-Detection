"""Colab cells 10 and 13: ordered substitutions, component-aware IDs and traces.

The source notebook is retained unchanged at the repository root. The original
first-block policy and input line order are intentional, including for new input.
"""

import re
from collections.abc import Iterable, Mapping

import numpy as np

PATTERNS = [
    (re.compile(r"blk_-?\d+"), "<BLOCK>"),
    (re.compile(r"/?\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?"), "<ADDRESS>"),
    (re.compile(r"\b(?:job|task|attempt)_\d+(?:_\d+)*\b"), "<JOB>"),
    (re.compile(r"/[^\s,]+"), "<PATH>"),
    (re.compile(r"(?<![\w])[-+]?\d+(?![\w])"), "<NUM>"),
]
BLOCK_PATTERN = re.compile(r"blk_-?\d+")


def extract_template(message):
    """Normalize variable fields in exactly the notebook substitution order."""
    template = message
    for pattern, replacement in PATTERNS:
        template = pattern.sub(replacement, template)
    return " ".join(template.split())


def parse_log_line(line):
    """Original six-field Colab parser; malformed lines return None."""
    parts = line.rstrip("\n").split(maxsplit=5)
    if len(parts) != 6:
        return None
    date, time, pid, level, component, message = parts
    block_match = re.search(r"blk_-?\d+", message)
    return {
        "timestamp": "20" + date + time,
        "pid": pid,
        "level": level,
        "component": component.rstrip(":"),
        "block_id": block_match.group() if block_match else None,
        "message": message,
        "template": extract_template(message),
    }


def event_id(record: Mapping, vocabulary: Mapping[str, int]) -> int:
    """Labels, timestamps, severity, PID and block IDs are not observations."""
    return vocabulary.get(record["component"] + " | " + record["template"], 0)


def build_block_sequences(
    lines: Iterable[str], vocabulary: Mapping[str, int],
    retained_block_ids: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Preserve input order and first-block matching from Colab cell 13.

    Optional membership filters out excluded blocks without accepting labels.
    Malformed block-bearing lines raise explicitly instead of silently dropping
    observations. Timestamp validation/sorting is deliberately not introduced.
    """
    ids = None if retained_block_ids is None else list(retained_block_ids)
    if ids is not None and len(set(ids)) != len(ids):
        raise ValueError("Retained block IDs must be unique.")
    sequences = {} if ids is None else {block: [] for block in ids}
    for number, line in enumerate(lines, start=1):
        match = BLOCK_PATTERN.search(line)
        if match is None:
            continue
        block = match.group()
        if ids is not None and block not in sequences:
            continue
        record = parse_log_line(line)
        if record is None:
            raise ValueError(f"Malformed block-bearing log line {number}.")
        sequences.setdefault(block, []).append(event_id(record, vocabulary))
    return {block: np.asarray(seq, dtype=np.int64) for block, seq in sequences.items()}
